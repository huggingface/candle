//! Audio tower (conv downsample + encoder transformer).
//!
//! This is a faithful port of `Qwen3ASRAudioEncoder` from the official Python
//! implementation:
//! `Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`.
//!
//! Note: the official code uses `cu_seqlens` to run windowed self-attention
//! inside the encoder (block-diagonal attention). This Rust port implements the
//! same segmentation for correctness; a FlashAttention2 varlen fast-path is
//! added separately (behind `--features flash-attn` and load options).

use candle::{DType, Device, Result, Tensor};
use candle_nn::{
    conv2d, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    VarBuilder,
};

use crate::config::AudioEncoderConfig;
use crate::processor::feat_lengths::feat_extract_output_length;

#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn_varlen;

const LAYER_NORM_EPS: f64 = 1e-5;

fn maybe_contiguous_for_metal(x: Tensor) -> Result<Tensor> {
    if x.device().is_metal() {
        x.contiguous()
    } else {
        Ok(x)
    }
}

fn cu_seqlens_from_aftercnn_len(
    aftercnn_len: usize,
    max_chunk_len_aftercnn: usize,
    window: usize,
    window_infer: usize,
) -> Result<Vec<usize>> {
    if aftercnn_len == 0 {
        return Ok(vec![0usize]);
    }
    if window == 0 {
        candle::bail!("audio_config window must be > 0");
    }
    let ratio = window_infer / window;
    if ratio == 0 {
        candle::bail!(
            "audio_config.n_window_infer must be >= audio_config.n_window*2: n_window_infer={window_infer} window={window}"
        );
    }

    let window_aftercnn =
        max_chunk_len_aftercnn
            .checked_mul(ratio)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "window_aftercnn overflow: max_chunk_len_aftercnn={max_chunk_len_aftercnn} ratio={ratio}"
                ))
            })?;
    if window_aftercnn == 0 {
        candle::bail!(
            "invalid window_aftercnn=0: max_chunk_len_aftercnn={max_chunk_len_aftercnn} ratio={ratio}"
        );
    }

    let mut cu: Vec<usize> = Vec::new();
    cu.push(0usize);

    let mut offset = 0usize;
    while offset < aftercnn_len {
        let remaining = aftercnn_len.saturating_sub(offset);
        let step = remaining.min(window_aftercnn);
        offset = offset
            .checked_add(step)
            .ok_or_else(|| candle::Error::Msg("cu_seqlens offset overflow".to_string()))?;
        cu.push(offset);
    }

    if cu.last().copied().is_none_or(|last| last != aftercnn_len) {
        candle::bail!(
            "internal error: cu_seqlens did not reach aftercnn_len: last={:?} aftercnn_len={aftercnn_len}",
            cu.last().copied()
        );
    }

    Ok(cu)
}

fn feature_lens_from_mask(mask: &Tensor, frames: usize) -> Result<Vec<usize>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != frames {
        candle::bail!("feature_attention_mask frames mismatch: expected={frames}, got={t2}");
    }

    // Avoid transferring the full mask back to the host. We only need per-sample
    // lengths, so compute them on-device and materialize a (batch,) vector.
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let nonzero = mask_f32.ne(0f32)?.to_dtype(DType::F32)?;
    let lens = nonzero.sum(1)?; // (batch,)
    let lens = lens.to_vec1::<f32>()?;
    if lens.len() != batch {
        candle::bail!(
            "internal error: feature_attention_mask lens mismatch: expected={batch}, got={}",
            lens.len()
        );
    }

    let mut out: Vec<usize> = Vec::with_capacity(batch);
    for (i, v) in lens.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            candle::bail!("invalid feature_attention_mask length at {i}: {v}");
        }
        if v > frames as f32 + 0.01 {
            candle::bail!(
                "feature_attention_mask length at {i} exceeds frames: len={v} frames={frames}"
            );
        }
        let len = v.round() as usize;
        if (len as f32 - v).abs() > 0.01 {
            candle::bail!("feature_attention_mask length at {i} is not integral: {v}");
        }
        out.push(len);
    }
    Ok(out)
}

fn sinusoidal_positional_embedding(
    length: usize,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    if !channels.is_multiple_of(2) {
        candle::bail!("sinusoidal positional embedding requires even channels, got {channels}");
    }
    if length == 0 {
        return Tensor::zeros((0usize, channels), DType::F32, device);
    }

    let half = channels / 2;
    let max_timescale = 10_000.0f64;
    let denom = (half as f64) - 1.0;
    if denom <= 0.0 {
        candle::bail!("invalid channels for sinusoidal embedding: {channels}");
    }
    let log_timescale_increment = max_timescale.ln() / denom;

    let mut inv_timescales = Vec::with_capacity(half);
    for i in 0..half {
        inv_timescales.push((-log_timescale_increment * (i as f64)).exp());
    }

    let mut data = Vec::with_capacity(length * channels);
    for t in 0..length {
        let t = t as f64;
        for &inv in &inv_timescales {
            data.push((t * inv).sin() as f32);
        }
        for &inv in &inv_timescales {
            data.push((t * inv).cos() as f32);
        }
    }

    Tensor::from_vec(data, (length, channels), device)
}

fn gelu(xs: &Tensor) -> Result<Tensor> {
    xs.gelu_erf()
}

#[derive(Debug, Clone)]
struct AudioAttention {
    use_flash_attn: bool,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl AudioAttention {
    fn load(cfg: &AudioEncoderConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_heads = cfg.encoder_attention_heads;
        if num_heads == 0 {
            candle::bail!("encoder_attention_heads must be > 0");
        }
        if !embed_dim.is_multiple_of(num_heads) {
            candle::bail!("d_model must be divisible by num_heads");
        }
        let head_dim = embed_dim / num_heads;
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            use_flash_attn,
            num_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let (seq_len, embed_dim) = hidden_states.dims2()?;
        if seq_len == 0 {
            return Tensor::zeros(
                (0usize, embed_dim),
                hidden_states.dtype(),
                hidden_states.device(),
            );
        }

        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((seq_len, self.num_heads, self.head_dim))?;

        let k = self.k_proj.forward(hidden_states)?;
        let k = k.reshape((seq_len, self.num_heads, self.head_dim))?;

        let v = self.v_proj.forward(hidden_states)?;
        let v = v.reshape((seq_len, self.num_heads, self.head_dim))?;

        if self.use_flash_attn {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = cu_seqlens;
                candle::bail!("flash-attn support is not enabled in this build");
            }

            #[cfg(feature = "flash-attn")]
            {
                if cu_seqlens.len() < 2 {
                    candle::bail!(
                        "cu_seqlens must have at least 2 elements, got {}",
                        cu_seqlens.len()
                    );
                }
                if cu_seqlens.first().copied().unwrap_or(1) != 0 {
                    candle::bail!("cu_seqlens must start at 0, got {:?}", cu_seqlens.first());
                }
                if cu_seqlens.last().copied().unwrap_or_default() != seq_len {
                    candle::bail!(
                        "cu_seqlens last must equal seq_len: last={:?} seq_len={seq_len}",
                        cu_seqlens.last().copied()
                    );
                }
                for w in cu_seqlens.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if b < a {
                        candle::bail!("cu_seqlens must be non-decreasing: {cu_seqlens:?}");
                    }
                }

                if !hidden_states.device().is_cuda() {
                    candle::bail!(
                        "flash-attn requires a CUDA device, got {:?}",
                        hidden_states.device().location()
                    );
                }
                match hidden_states.dtype() {
                    DType::F16 | DType::BF16 => {}
                    other => {
                        candle::bail!("flash-attn requires dtype f16/bf16, got {other:?}")
                    }
                }

                let mut max_len = 0usize;
                for w in cu_seqlens.windows(2) {
                    let len = w[1].saturating_sub(w[0]);
                    max_len = max_len.max(len);
                }
                if max_len == 0 {
                    candle::bail!("invalid cu_seqlens: max_len=0 for seq_len={seq_len}");
                }

                let mut cu_u32: Vec<u32> = Vec::with_capacity(cu_seqlens.len());
                for (i, &v) in cu_seqlens.iter().enumerate() {
                    cu_u32.push(u32::try_from(v).map_err(|_| {
                        candle::Error::Msg(format!(
                            "cu_seqlens overflows u32 at index {i}: value={v}"
                        ))
                    })?);
                }
                let cu = Tensor::from_vec(cu_u32, (cu_seqlens.len(),), hidden_states.device())?;

                let softmax_scale = self.scaling as f32;
                let out = flash_attn_varlen(
                    &q,
                    &k,
                    &v,
                    &cu,
                    &cu,
                    max_len,
                    max_len,
                    softmax_scale,
                    false,
                )?;

                let attn = out.reshape((seq_len, embed_dim))?;
                return self.out_proj.forward(&attn);
            }
        }

        let q = q.transpose(0, 1)?.unsqueeze(0)?; // (1, h, s, d)
        let k = k.transpose(0, 1)?.unsqueeze(0)?; // (1, h, s, d)
        let v = v.transpose(0, 1)?.unsqueeze(0)?; // (1, h, s, d)

        // Metal GEMM kernels only support contiguous (or simple transposed) layouts.
        // These tensors are produced via reshapes/transposes that yield strided views,
        // so materialize contiguous buffers when running on Metal.
        let q = maybe_contiguous_for_metal(q)?;
        let v = maybe_contiguous_for_metal(v)?;

        let k_t = maybe_contiguous_for_metal(k.transpose(2, 3)?)?; // (1, h, d, s)
        let attn_weights = q.matmul(&k_t)?.affine(self.scaling, 0.0)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let attn = attn_weights.matmul(&v)?; // (1, h, s, d)
        let attn = attn.squeeze(0)?.transpose(0, 1)?; // (s, h, d)
        let attn = attn.reshape((seq_len, embed_dim))?;
        self.out_proj.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct AudioEncoderLayer {
    self_attn: AudioAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl AudioEncoderLayer {
    fn load(cfg: &AudioEncoderConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let self_attn = AudioAttention::load(cfg, vb.pp("self_attn"), use_flash_attn)?;
        let self_attn_layer_norm = layer_norm(
            embed_dim,
            candle_nn::LayerNormConfig::from(LAYER_NORM_EPS),
            vb.pp("self_attn_layer_norm"),
        )?;
        let final_layer_norm = layer_norm(
            embed_dim,
            candle_nn::LayerNormConfig::from(LAYER_NORM_EPS),
            vb.pp("final_layer_norm"),
        )?;
        let fc1 = linear(embed_dim, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.encoder_ffn_dim, embed_dim, vb.pp("fc2"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.self_attn_layer_norm.forward(hidden_states)?;
        let x = self.self_attn.forward(&x, cu_seqlens)?;
        let x = (&residual + &x)?;

        let residual = x.clone();
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = gelu(&x)?;
        let x = self.fc2.forward(&x)?;
        let x = (&residual + &x)?;

        if x.dtype() == DType::F16 {
            // Match the official float16 clamp to prevent infs.
            let clamp = 65_504.0f32 - 1000.0;
            x.clamp(-clamp, clamp)
        } else {
            Ok(x)
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioTower {
    config: AudioEncoderConfig,
    positional_embedding: Tensor,
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    use_flash_attn: bool,
}

impl AudioTower {
    pub fn load(cfg: &AudioEncoderConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let device = vb.device().clone();

        let positional_embedding =
            sinusoidal_positional_embedding(cfg.max_source_positions, cfg.d_model, &device)?;

        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Conv2dConfig::default()
        };
        let conv2d1 = conv2d(1, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv2d3 = conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;

        let mel_after = cfg.num_mel_bins.div_ceil(2).div_ceil(2).div_ceil(2);
        let conv_out_in = cfg.downsample_hidden_size * mel_after;
        let conv_out = linear_no_bias(conv_out_in, cfg.d_model, vb.pp("conv_out"))?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::load(
                cfg,
                vb.pp("layers").pp(i),
                use_flash_attn,
            )?);
        }

        let ln_post = layer_norm(
            cfg.d_model,
            candle_nn::LayerNormConfig::from(LAYER_NORM_EPS),
            vb.pp("ln_post"),
        )?;
        let proj1 = linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        Ok(Self {
            config: cfg.clone(),
            positional_embedding,
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            use_flash_attn,
        })
    }

    pub fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }

    pub fn uses_flash_attn(&self) -> bool {
        self.use_flash_attn
    }

    pub fn forward(
        &self,
        input_features: &Tensor,
        feature_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, _mel, frames) = input_features.dims3()?;
        let feature_lens = match feature_attention_mask {
            None => vec![frames; batch],
            Some(mask) => {
                let (b2, t2) = mask.dims2()?;
                if (b2, t2) != (batch, frames) {
                    candle::bail!(
                        "feature_attention_mask dims mismatch: expected=({batch},{frames}), got=({b2},{t2})"
                    );
                }
                feature_lens_from_mask(mask, frames)?
            }
        };
        self.forward_with_lens(input_features, feature_lens.as_slice())
    }

    pub fn forward_with_lens(
        &self,
        input_features: &Tensor,
        feature_lens: &[usize],
    ) -> Result<Tensor> {
        let want_dtype = self.conv2d1.weight().dtype();
        let input_features = if input_features.dtype() == want_dtype {
            input_features.clone()
        } else {
            input_features.to_dtype(want_dtype)?
        };

        let (batch, mel, frames) = input_features.dims3()?;
        if mel != self.config.num_mel_bins {
            candle::bail!(
                "input_features mel bins mismatch: expected={}, got={mel}",
                self.config.num_mel_bins
            );
        }
        if feature_lens.len() != batch {
            candle::bail!(
                "feature_lens batch mismatch: expected={batch}, got={}",
                feature_lens.len()
            );
        }
        for (i, &len) in feature_lens.iter().enumerate() {
            if len > frames {
                candle::bail!("feature_lens[{i}] exceeds frames: len={len} frames={frames}");
            }
        }

        let mut outs: Vec<Tensor> = Vec::with_capacity(batch);
        for (idx, &len) in feature_lens.iter().enumerate() {
            let x = input_features.narrow(0, idx, 1)?.squeeze(0)?;
            if len == 0 {
                outs.push(Tensor::zeros(
                    (0usize, self.config.output_dim),
                    input_features.dtype(),
                    input_features.device(),
                )?);
                continue;
            }
            let x = x.narrow(1, 0, len)?;
            outs.push(self.forward_one(&x)?);
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0)
    }

    fn forward_one(&self, input_feature: &Tensor) -> Result<Tensor> {
        let (mel, frames) = input_feature.dims2()?;
        if mel != self.config.num_mel_bins {
            candle::bail!(
                "input_feature mel bins mismatch: expected={}, got={mel}",
                self.config.num_mel_bins
            );
        }

        let window = self
            .config
            .n_window
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("n_window overflow".to_string()))?;
        if window == 0 {
            candle::bail!("audio_config.n_window must be > 0");
        }

        let chunk_num = frames.div_ceil(window);
        let mut chunk_lengths = vec![window; chunk_num];
        if chunk_num > 0 {
            let rem = frames % window;
            if rem != 0 {
                let last = chunk_num.saturating_sub(1);
                chunk_lengths[last] = rem;
            }
        }

        let max_chunk_len = window;
        let input_t = input_feature.transpose(0, 1)?; // (frames, mel)
        let mut padded_chunks: Vec<Tensor> = Vec::with_capacity(chunk_num);
        let mut aftercnn_lens: Vec<usize> = Vec::with_capacity(chunk_num);

        let mut offset = 0usize;
        for &len in &chunk_lengths {
            if offset.saturating_add(len) > frames {
                candle::bail!("chunk split overflow: offset={offset} len={len} frames={frames}");
            }
            let chunk = input_t.narrow(0, offset, len)?; // (len, mel)
            let pad = max_chunk_len.saturating_sub(len);
            padded_chunks.push(chunk.pad_with_zeros(0, 0, pad)?);
            aftercnn_lens.push(feat_extract_output_length(len));
            offset = offset.saturating_add(len);
        }
        if offset != frames {
            candle::bail!(
                "chunk split did not consume full sequence: offset={offset} frames={frames}"
            );
        }

        if padded_chunks.is_empty() {
            return Tensor::zeros(
                (0usize, self.config.output_dim),
                input_feature.dtype(),
                input_feature.device(),
            );
        }

        // (chunk_num, max_chunk_len, mel) -> (chunk_num, mel, max_chunk_len)
        let padded_feature = Tensor::stack(&padded_chunks, 0)?.transpose(1, 2)?;
        let padded_feature = padded_feature.unsqueeze(1)?; // (chunk_num, 1, mel, max_chunk_len)

        // Convolution stack (split along chunk dimension to avoid OOM).
        let mut conv_outs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < chunk_num {
            let len = (chunk_num - start).min(self.config.conv_chunksize.max(1));
            let x = padded_feature.narrow(0, start, len)?;
            let x = gelu(&self.conv2d1.forward(&x)?)?;
            let x = gelu(&self.conv2d2.forward(&x)?)?;
            let x = gelu(&self.conv2d3.forward(&x)?)?;
            conv_outs.push(x);
            start = start.saturating_add(len);
        }

        let mut x = if conv_outs.len() == 1 {
            conv_outs
                .first()
                .ok_or_else(|| candle::Error::Msg("missing conv output".to_string()))?
                .clone()
        } else {
            Tensor::cat(&conv_outs.iter().collect::<Vec<_>>(), 0)?
        };

        let (b, c, f, t) = x.dims4()?;
        x = x.transpose(1, 3)?.transpose(2, 3)?.contiguous()?; // (b, t, c, f)
        x = x.reshape((b, t, c * f))?;
        x = self.conv_out.forward(&x)?; // (b, t, d_model)

        // Add sinusoidal positional embedding (no learned weights in checkpoint).
        let pe = self.positional_embedding.narrow(0, 0, t)?.unsqueeze(0)?;
        let pe = pe.to_dtype(x.dtype())?;
        x = x.broadcast_add(&pe)?;

        // Flatten ragged chunks by keeping only valid positions per chunk.
        let max_t = t;
        let flat = x.reshape((b * max_t, self.config.d_model))?;
        let mut idxs: Vec<u32> = Vec::with_capacity(aftercnn_lens.iter().sum());
        for (chunk_idx, &len) in aftercnn_lens.iter().enumerate() {
            if len > max_t {
                candle::bail!("aftercnn_len={len} exceeds max_t={max_t}");
            }
            let base = chunk_idx
                .checked_mul(max_t)
                .ok_or_else(|| candle::Error::Msg("index overflow".to_string()))?;
            for j in 0..len {
                let pos = base
                    .checked_add(j)
                    .ok_or_else(|| candle::Error::Msg("index overflow".to_string()))?;
                let pos_u32 = u32::try_from(pos)
                    .map_err(|_| candle::Error::Msg("index overflow".to_string()))?;
                idxs.push(pos_u32);
            }
        }
        let idx_len = idxs.len();
        let idx = Tensor::from_vec(idxs, (idx_len,), input_feature.device())?;
        let mut hidden_states = flat.index_select(&idx, 0)?;

        // Encoder transformer layers.
        let cu_seqlens = if self.use_flash_attn {
            let aftercnn_total = feat_extract_output_length(frames);
            if aftercnn_total != idx_len {
                candle::bail!("aftercnn length mismatch: expected={aftercnn_total}, got={idx_len}");
            }
            cu_seqlens_from_aftercnn_len(aftercnn_total, max_t, window, self.config.n_window_infer)?
        } else {
            vec![0usize, idx_len]
        };

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, cu_seqlens.as_slice())?;
        }

        hidden_states = self.ln_post.forward(&hidden_states)?;
        hidden_states = self.proj1.forward(&hidden_states)?;
        hidden_states = gelu(&hidden_states)?;
        hidden_states = self.proj2.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cu_seqlens_from_aftercnn_len, feature_lens_from_mask, sinusoidal_positional_embedding,
        AudioAttention, AudioTower,
    };
    use crate::config::AudioEncoderConfig;
    use candle_nn::Linear;

    #[test]
    fn test_sinusoidal_positional_embedding_shape() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let pe = sinusoidal_positional_embedding(10, 8, &device)?;
        if pe.dims() != vec![10, 8] {
            anyhow::bail!("unexpected pe dims: {:?}", pe.dims());
        }
        Ok(())
    }

    #[test]
    fn test_feature_lens_from_mask_counts_nonzero() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let mask =
            candle::Tensor::from_vec(vec![1u32, 1, 0, 0, 1, 1, 1, 0], (2usize, 4usize), &device)?;
        let lens = feature_lens_from_mask(&mask, 4)?;
        if lens != vec![2usize, 3usize] {
            anyhow::bail!("unexpected lens: {lens:?}");
        }
        Ok(())
    }

    #[test]
    fn test_audio_tower_forward_with_lens_matches_mask() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let cfg = AudioEncoderConfig {
            num_mel_bins: 4,
            encoder_layers: 1,
            encoder_attention_heads: 1,
            encoder_ffn_dim: 8,
            d_model: 4,
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_function: "gelu".to_string(),
            activation_dropout: 0.0,
            scale_embedding: false,
            initializer_range: 0.02,
            max_source_positions: 64,
            n_window: 5,
            output_dim: 6,
            n_window_infer: 20,
            conv_chunksize: 8,
            downsample_hidden_size: 2,
        };
        let vb = candle_nn::VarBuilder::zeros(candle::DType::F32, &device);
        let tower = AudioTower::load(&cfg, vb, false)?;

        let batch = 2usize;
        let mel = cfg.num_mel_bins;
        let frames = 10usize;
        let feats: Vec<f32> = (0..batch * mel * frames).map(|i| i as f32).collect();
        let input_features = candle::Tensor::from_vec(feats, (batch, mel, frames), &device)?;

        let len1 = 6usize;
        let mut mask: Vec<u32> = Vec::with_capacity(batch * frames);
        mask.extend(std::iter::repeat_n(1u32, frames));
        mask.extend(std::iter::repeat_n(1u32, len1));
        mask.extend(std::iter::repeat_n(0u32, frames - len1));
        let mask = candle::Tensor::from_vec(mask, (batch, frames), &device)?;

        let out_mask = tower.forward(&input_features, Some(&mask))?;
        let out_lens = tower.forward_with_lens(&input_features, &[frames, len1])?;

        if out_mask.dims() != out_lens.dims() {
            anyhow::bail!(
                "output dims mismatch: mask={:?} lens={:?}",
                out_mask.dims(),
                out_lens.dims()
            );
        }

        let got = out_mask.to_vec2::<f32>()?;
        let exp = out_lens.to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        for (row_g, row_e) in got.iter().zip(exp.iter()) {
            for (&a, &b) in row_g.iter().zip(row_e.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        if max_abs > 1e-6 {
            anyhow::bail!("forward mismatch: max_abs={max_abs}");
        }
        Ok(())
    }

    #[test]
    fn test_cu_seqlens_from_aftercnn_len_splits_into_windows() -> anyhow::Result<()> {
        // Mirrors the official computation:
        // window_aftercnn = max_chunk_len_aftercnn * (n_window_infer / (n_window*2))
        let aftercnn_len = 100usize;
        let max_chunk_len_aftercnn = 26usize;
        let window = 200usize; // n_window*2
        let n_window_infer = 400usize;
        let cu = cu_seqlens_from_aftercnn_len(
            aftercnn_len,
            max_chunk_len_aftercnn,
            window,
            n_window_infer,
        )?;
        if cu != vec![0usize, 52, 100] {
            anyhow::bail!("unexpected cu_seqlens: {cu:?}");
        }
        Ok(())
    }

    #[test]
    fn test_audio_attention_ignores_cu_seqlens_when_eager() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let embed_dim = 4usize;
        let seq_len = 5usize;

        let mut eye = vec![0f32; embed_dim * embed_dim];
        for i in 0..embed_dim {
            eye[i * embed_dim + i] = 1.0;
        }
        let w = candle::Tensor::from_vec(eye, (embed_dim, embed_dim), &device)?;
        let q_proj = Linear::new(w.clone(), None);
        let k_proj = Linear::new(w.clone(), None);
        let v_proj = Linear::new(w.clone(), None);
        let out_proj = Linear::new(w, None);

        let attn = AudioAttention {
            use_flash_attn: false,
            num_heads: 1,
            head_dim: embed_dim,
            scaling: (embed_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        };

        let xs: Vec<f32> = (0..seq_len * embed_dim).map(|i| i as f32).collect();
        let hidden_states = candle::Tensor::from_vec(xs, (seq_len, embed_dim), &device)?;

        let out_blocked = attn.forward(&hidden_states, &[0usize, 2, 5])?;
        let out_full = attn.forward(&hidden_states, &[0usize, 5])?;

        let got = out_blocked.to_vec2::<f32>()?;
        let exp = out_full.to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        for (row_g, row_e) in got.iter().zip(exp.iter()) {
            for (&a, &b) in row_g.iter().zip(row_e.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        if max_abs > 1e-4 {
            anyhow::bail!("segmented attention mismatch: max_abs={max_abs}");
        }
        Ok(())
    }
}
