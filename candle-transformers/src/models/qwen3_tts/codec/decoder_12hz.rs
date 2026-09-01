//! 12 Hz Mimi-based audio decoder for Qwen3-TTS.
//!
//! Converts 16-codebook codec tokens to a 24 kHz mono audio waveform.
//! The architecture is:
//!
//! 1. Dual-path RVQ decode (rvq_first + rvq_rest) → sum → pre_conv
//! 2. Mini transformer (8 layers, 512 hidden) with RoPE
//! 3. 2-stage pre-upsampling (CausalTransConv + ConvNeXtBlock, ratios 2×2)
//! 4. BigVGAN decoder blocks (upsample rates 8, 5, 4, 3)
//! 5. SnakeBeta + final causal conv → clamp to [−1, 1]
//!
//! Two decode APIs are provided:
//!
//! - [`Decoder12Hz::decode`] — batch API, requires all frames at once, produces
//!   the best quality (full-sequence transformer cross-frame context).
//! - [`Decoder12Hz::decode_frame`] / [`Decoder12Hz::new_streaming_state`] —
//!   streaming API, emits audio one frame at a time with O(N) KV-cache memory.

use std::collections::HashMap;

use candle::{DType, IndexOp, Module, Result, Tensor, D};
use super::{
    CausalConv1d, CausalConv1dState,
    CausalTransConv1d, CausalTransConv1dState,
    ConvNeXtBlock, DecoderBlock, SnakeBeta,
};

/// Configuration for the 12 Hz decoder (defaults match all released variants).
#[derive(Debug, Clone)]
pub struct Decoder12HzConfig {
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub upsampling_ratios: Vec<usize>,
    pub decoder_dim: usize,
    pub upsample_rates: Vec<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub layer_scale: f64,
}

impl Default for Decoder12HzConfig {
    fn default() -> Self {
        Self {
            codebook_dim: 512,
            latent_dim: 1024,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 1024,
            num_quantizers: 16,
            codebook_size: 2048,
            upsampling_ratios: vec![2, 2],
            decoder_dim: 1536,
            upsample_rates: vec![8, 5, 4, 3],
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            layer_scale: 0.01,
        }
    }
}

// ── Transformer layer weights (stored as raw tensors) ──────────────────────

struct TxLayerW {
    input_ln: Tensor,
    q: Tensor, k: Tensor, v: Tensor, o: Tensor,
    attn_scale: Tensor,
    post_ln: Tensor,
    gate: Tensor, up: Tensor, down: Tensor,
    mlp_scale: Tensor,
}

// ── KV cache for one decoder transformer layer ────────────────────────────

/// Growable KV cache for one decoder transformer layer.
///
/// Used by the streaming [`Decoder12HzState`] to avoid reprocessing past
/// frames on every new decode step.
#[derive(Default)]
pub struct DecoderKVCache {
    k: Option<Tensor>, // [B, heads, T_past, head_dim]
    v: Option<Tensor>,
}

impl DecoderKVCache {
    pub fn new() -> Self { Self::default() }

    /// Append a new `[B, heads, 1, head_dim]` K/V slice and return the
    /// full accumulated `[B, heads, T_past+1, head_dim]` tensors.
    fn append(&mut self, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(pk), Some(pv)) => (
                Tensor::cat(&[pk, &new_k], 2)?,
                Tensor::cat(&[pv, &new_v], 2)?,
            ),
            _ => (new_k, new_v),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

// ── Streaming state for ConvNeXtBlock ─────────────────────────────────────

/// Streaming state for one [`ConvNeXtBlock`] (holds its dwconv state).
pub struct ConvNeXtBlockState {
    pub dw: CausalConv1dState,
}

impl Default for ConvNeXtBlockState {
    fn default() -> Self { Self { dw: CausalConv1dState::default() } }
}

// ── Streaming state for UpsampleStage ─────────────────────────────────────

struct UpsampleStageState {
    tc:  CausalTransConv1dState,
    cn:  ConvNeXtBlockState,
}

// ── Streaming state for ResidualUnit ──────────────────────────────────────

struct ResidualUnitState {
    conv1: CausalConv1dState,
    conv2: CausalConv1dState,
}

// ── Streaming state for DecoderBlock ──────────────────────────────────────

struct DecoderBlockState {
    upsample: CausalTransConv1dState,
    res1: ResidualUnitState,
    res2: ResidualUnitState,
    res3: ResidualUnitState,
}

// ── Full streaming state ──────────────────────────────────────────────────

/// All mutable state needed for incremental (streaming) decoding.
///
/// Obtained via [`Decoder12Hz::new_streaming_state`]; passed mutably to
/// each call of [`Decoder12Hz::decode_frame`].
pub struct Decoder12HzState {
    /// Per-transformer-layer KV caches.
    pub kv_caches: Vec<DecoderKVCache>,
    /// Current position in the transformer (= number of frames decoded so far).
    pub offset: usize,
    // Conv state for each stage
    pre_conv:       CausalConv1dState,
    upsample_stages: Vec<UpsampleStageState>,
    decoder_init:   CausalConv1dState,
    decoder_blocks: Vec<DecoderBlockState>,
    final_conv:     CausalConv1dState,
}

// ── UpsampleStage ──────────────────────────────────────────────────────────

struct UpsampleStage {
    trans_conv: CausalTransConv1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    fn from_weights(
        tc_w: Tensor, tc_b: Tensor,
        cn_dw_w: Tensor, cn_dw_b: Tensor,
        cn_norm_w: Tensor, cn_norm_b: Tensor,
        cn_pw1_w: Tensor, cn_pw1_b: Tensor,
        cn_pw2_w: Tensor, cn_pw2_b: Tensor,
        cn_gamma: Tensor,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            trans_conv: CausalTransConv1d::from_weights(tc_w, Some(tc_b), stride)?,
            convnext: ConvNeXtBlock::from_weights(
                cn_dw_w, Some(cn_dw_b),
                cn_norm_w, cn_norm_b,
                cn_pw1_w, cn_pw1_b,
                cn_pw2_w, cn_pw2_b,
                cn_gamma,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.convnext.forward(&self.trans_conv.forward(x)?)
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut UpsampleStageState) -> Result<Tensor> {
        let after_tc = self.trans_conv.forward_with_state(x, &mut state.tc)?;
        self.convnext.forward_with_state(&after_tc, &mut state.cn)
    }
}

// ── Decoder12Hz ────────────────────────────────────────────────────────────

/// Full 12 Hz audio decoder.
pub struct Decoder12Hz {
    config: Decoder12HzConfig,
    first_codebook: Tensor,
    rest_codebooks: Vec<Tensor>,
    first_output_proj: Tensor,
    rest_output_proj: Tensor,
    pre_conv: CausalConv1d,
    input_proj_w: Tensor, input_proj_b: Tensor,
    tx_layers: Vec<TxLayerW>,
    final_norm_w: Tensor,
    output_proj_w: Tensor, output_proj_b: Tensor,
    upsample_stages: Vec<UpsampleStage>,
    decoder_init_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: CausalConv1d,
}

fn w(weights: &HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| candle::Error::Msg(format!("Missing weight: {key}")))
}

impl Decoder12Hz {
    pub fn from_weights(weights: &HashMap<String, Tensor>, config: Decoder12HzConfig) -> Result<Self> {
        // ── Codebooks (normalize by cluster_usage) ─────────────────────────
        let eps = 1e-7f32;
        let norm_cb = |sum: Tensor, usage: Tensor| -> Result<Tensor> {
            let usage_c = usage.clamp(eps, f32::MAX)?;
            sum.broadcast_div(&usage_c.unsqueeze(1)?)
        };

        let first_codebook = norm_cb(
            w(weights, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")?,
            w(weights, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")?,
        )?;

        let mut rest_codebooks = Vec::with_capacity(15);
        for i in 0..15 {
            let cb = norm_cb(
                w(weights, &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"))?,
                w(weights, &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"))?,
            )?;
            rest_codebooks.push(cb);
        }

        let first_output_proj = w(weights, "decoder.quantizer.rvq_first.output_proj.weight")?;
        let rest_output_proj  = w(weights, "decoder.quantizer.rvq_rest.output_proj.weight")?;

        // ── Pre-conv ────────────────────────────────────────────────────────
        let pre_conv = CausalConv1d::from_weights(
            w(weights, "decoder.pre_conv.conv.weight")?,
            Some(w(weights, "decoder.pre_conv.conv.bias")?),
            1,
        )?;

        // ── Pre-transformer projections ─────────────────────────────────────
        let input_proj_w = w(weights, "decoder.pre_transformer.input_proj.weight")?;
        let input_proj_b = w(weights, "decoder.pre_transformer.input_proj.bias")?;
        let output_proj_w = w(weights, "decoder.pre_transformer.output_proj.weight")?;
        let output_proj_b = w(weights, "decoder.pre_transformer.output_proj.bias")?;

        // ── Transformer layers ──────────────────────────────────────────────
        let mut tx_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("decoder.pre_transformer.layers.{i}");
            tx_layers.push(TxLayerW {
                input_ln: w(weights, &format!("{p}.input_layernorm.weight"))?,
                q: w(weights, &format!("{p}.self_attn.q_proj.weight"))?,
                k: w(weights, &format!("{p}.self_attn.k_proj.weight"))?,
                v: w(weights, &format!("{p}.self_attn.v_proj.weight"))?,
                o: w(weights, &format!("{p}.self_attn.o_proj.weight"))?,
                attn_scale: w(weights, &format!("{p}.self_attn_layer_scale.scale"))?,
                post_ln: w(weights, &format!("{p}.post_attention_layernorm.weight"))?,
                gate: w(weights, &format!("{p}.mlp.gate_proj.weight"))?,
                up:   w(weights, &format!("{p}.mlp.up_proj.weight"))?,
                down: w(weights, &format!("{p}.mlp.down_proj.weight"))?,
                mlp_scale: w(weights, &format!("{p}.mlp_layer_scale.scale"))?,
            });
        }
        let final_norm_w = w(weights, "decoder.pre_transformer.norm.weight")?;

        // ── Upsample stages ─────────────────────────────────────────────────
        let mut upsample_stages = Vec::with_capacity(config.upsampling_ratios.len());
        for (i, &stride) in config.upsampling_ratios.iter().enumerate() {
            let p = format!("decoder.upsample.{i}");
            upsample_stages.push(UpsampleStage::from_weights(
                w(weights, &format!("{p}.0.conv.weight"))?,
                w(weights, &format!("{p}.0.conv.bias"))?,
                w(weights, &format!("{p}.1.dwconv.conv.weight"))?,
                w(weights, &format!("{p}.1.dwconv.conv.bias"))?,
                w(weights, &format!("{p}.1.norm.weight"))?,
                w(weights, &format!("{p}.1.norm.bias"))?,
                w(weights, &format!("{p}.1.pwconv1.weight"))?,
                w(weights, &format!("{p}.1.pwconv1.bias"))?,
                w(weights, &format!("{p}.1.pwconv2.weight"))?,
                w(weights, &format!("{p}.1.pwconv2.bias"))?,
                w(weights, &format!("{p}.1.gamma"))?,
                stride,
            )?);
        }

        // ── Decoder init conv ────────────────────────────────────────────────
        let decoder_init_conv = CausalConv1d::from_weights(
            w(weights, "decoder.decoder.0.conv.weight")?,
            Some(w(weights, "decoder.decoder.0.conv.bias")?),
            1,
        )?;

        // ── Decoder blocks ───────────────────────────────────────────────────
        let mut decoder_blocks = Vec::with_capacity(config.upsample_rates.len());
        for (i, &rate) in config.upsample_rates.iter().enumerate() {
            let block_idx = i + 1;
            let bp = format!("decoder.decoder.{block_idx}.block");
            let load_res = |u: usize| -> Result<(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)> {
                let p = format!("{bp}.{u}");
                Ok((
                    w(weights, &format!("{p}.act1.alpha"))?,
                    w(weights, &format!("{p}.act1.beta"))?,
                    w(weights, &format!("{p}.conv1.conv.weight"))?,
                    w(weights, &format!("{p}.conv1.conv.bias"))?,
                    w(weights, &format!("{p}.act2.alpha"))?,
                    w(weights, &format!("{p}.act2.beta"))?,
                    w(weights, &format!("{p}.conv2.conv.weight"))?,
                    w(weights, &format!("{p}.conv2.conv.bias"))?,
                ))
            };
            let (r1a1a,r1a1b,r1c1w,r1c1b,r1a2a,r1a2b,r1c2w,r1c2b) = load_res(2)?;
            let (r2a1a,r2a1b,r2c1w,r2c1b,r2a2a,r2a2b,r2c2w,r2c2b) = load_res(3)?;
            let (r3a1a,r3a1b,r3c1w,r3c1b,r3a2a,r3a2b,r3c2w,r3c2b) = load_res(4)?;

            decoder_blocks.push(DecoderBlock::from_weights(
                w(weights, &format!("{bp}.0.alpha"))?,
                w(weights, &format!("{bp}.0.beta"))?,
                w(weights, &format!("{bp}.1.conv.weight"))?,
                w(weights, &format!("{bp}.1.conv.bias"))?,
                r1a1a,r1a1b,r1c1w,r1c1b,r1a2a,r1a2b,r1c2w,r1c2b,
                r2a1a,r2a1b,r2c1w,r2c1b,r2a2a,r2a2b,r2c2w,r2c2b,
                r3a1a,r3a1b,r3c1w,r3c1b,r3a2a,r3a2b,r3c2w,r3c2b,
                rate,
            )?);
        }

        let final_snake = SnakeBeta::from_weights(
            w(weights, "decoder.decoder.5.alpha")?,
            w(weights, "decoder.decoder.5.beta")?,
        )?;
        let final_conv = CausalConv1d::from_weights(
            w(weights, "decoder.decoder.6.conv.weight")?,
            Some(w(weights, "decoder.decoder.6.conv.bias")?),
            1,
        )?;

        Ok(Self {
            config,
            first_codebook,
            rest_codebooks,
            first_output_proj,
            rest_output_proj,
            pre_conv,
            input_proj_w, input_proj_b,
            tx_layers,
            final_norm_w,
            output_proj_w, output_proj_b,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
        })
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Batch decode: convert `[batch, 16, T]` codec tokens → `[batch, 1, samples]` audio.
    ///
    /// All frames must be available before calling. Produces the best quality
    /// because every transformer layer has full cross-frame context.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let device = codes.device();
        let (batch, _nq, seq) = codes.dims3()?;

        let quantized = self.rvq_decode(codes, batch, seq, device)?;

        use candle::Module;
        let hidden = self.pre_conv.forward(&quantized)?;

        let hidden = hidden.transpose(1, 2)?;
        let hidden = self.linear_3d(&hidden, &self.input_proj_w, Some(&self.input_proj_b))?;
        let hidden = self.run_transformer(hidden, seq)?;
        let hidden = self.rms_norm(&hidden, &self.final_norm_w)?;
        let hidden = self.linear_3d(&hidden, &self.output_proj_w, Some(&self.output_proj_b))?;
        let mut hidden = hidden.transpose(1, 2)?;

        for stage in &self.upsample_stages {
            hidden = stage.forward(&hidden)?;
        }

        hidden = self.decoder_init_conv.forward(&hidden)?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.final_snake.forward(&hidden)?;
        hidden = self.final_conv.forward(&hidden)?;
        hidden.clamp(-1.0f32, 1.0f32)
    }

    /// Allocate all streaming state for incremental decoding.
    ///
    /// Call once before the generation loop, then pass the returned
    /// [`Decoder12HzState`] to every [`decode_frame`](Self::decode_frame) call.
    pub fn new_streaming_state(&self) -> Decoder12HzState {
        let n_layers = self.tx_layers.len();
        let n_up     = self.upsample_stages.len();
        let n_blocks = self.decoder_blocks.len();

        Decoder12HzState {
            kv_caches: (0..n_layers).map(|_| DecoderKVCache::new()).collect(),
            offset: 0,
            pre_conv: CausalConv1dState::default(),
            upsample_stages: (0..n_up).map(|_| UpsampleStageState {
                tc: CausalTransConv1dState::default(),
                cn: ConvNeXtBlockState::default(),
            }).collect(),
            decoder_init: CausalConv1dState::default(),
            decoder_blocks: (0..n_blocks).map(|_| DecoderBlockState {
                upsample: CausalTransConv1dState::default(),
                res1: ResidualUnitState {
                    conv1: CausalConv1dState::default(),
                    conv2: CausalConv1dState::default(),
                },
                res2: ResidualUnitState {
                    conv1: CausalConv1dState::default(),
                    conv2: CausalConv1dState::default(),
                },
                res3: ResidualUnitState {
                    conv1: CausalConv1dState::default(),
                    conv2: CausalConv1dState::default(),
                },
            }).collect(),
            final_conv: CausalConv1dState::default(),
        }
    }

    /// Streaming decode: process one codec frame and return the corresponding
    /// audio chunk.
    ///
    /// `frame_codes` — `[1, 16, 1]` i64/u32 tensor (one frame, 16 codebooks).  
    /// `state` — mutable streaming state from [`new_streaming_state`](Self::new_streaming_state).
    ///
    /// Returns `[1, 1, samples_per_frame]` f32 audio.  The number of output
    /// samples per frame equals `upsample_ratios.product() * upsample_rates.product()`
    /// (2×2×8×5×4×3 = 1920 by default, i.e. ~80 ms at 24 kHz).
    pub fn decode_frame(
        &self,
        frame_codes: &Tensor,   // [1, 16, 1]
        state: &mut Decoder12HzState,
    ) -> Result<Tensor> {
        let device = frame_codes.device();
        let batch = 1usize;
        let seq   = 1usize;

        // RVQ decode for the single frame
        let quantized = self.rvq_decode(frame_codes, batch, seq, device)?; // [1, 512, 1]

        // Pre-conv (streaming)
        let hidden = self.pre_conv.forward_with_state(&quantized, &mut state.pre_conv)?; // [1, 1024, 1]

        // Pre-transformer
        let hidden = hidden.transpose(1, 2)?; // [1, 1, 1024]
        let hidden = self.linear_3d(&hidden, &self.input_proj_w, Some(&self.input_proj_b))?;
        let hidden = self.run_transformer_step(hidden, state)?; // [1, 1, 512]
        let hidden = self.rms_norm(&hidden, &self.final_norm_w)?;
        let hidden = self.linear_3d(&hidden, &self.output_proj_w, Some(&self.output_proj_b))?;
        let mut hidden = hidden.transpose(1, 2)?; // [1, latent_dim, 1]

        state.offset += 1;

        // Pre-upsample stages (streaming)
        for (stage, st) in self.upsample_stages.iter().zip(state.upsample_stages.iter_mut()) {
            hidden = stage.forward_streaming(&hidden, st)?;
        }

        // Decoder init conv (streaming)
        hidden = self.decoder_init_conv.forward_with_state(&hidden, &mut state.decoder_init)?;

        // Decoder blocks (streaming)
        for (block, bst) in self.decoder_blocks.iter().zip(state.decoder_blocks.iter_mut()) {
            hidden = block.forward_streaming(&hidden, bst)?;
        }

        // Final snake + conv (streaming)
        hidden = self.final_snake.forward(&hidden)?;
        hidden = self.final_conv.forward_with_state(&hidden, &mut state.final_conv)?;
        hidden.clamp(-1.0f32, 1.0f32)
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// RVQ decode for `[batch, 16, seq]` codes → `[batch, 512, seq]` latent.
    fn rvq_decode(
        &self,
        codes: &Tensor,
        batch: usize,
        seq: usize,
        device: &candle::Device,
    ) -> Result<Tensor> {
        let cb_size = self.config.codebook_size as i64;
        let cb_f    = cb_size as f64;

        // First quantizer — modulo to map 3072 → 2048
        let first_codes = codes.i((.., 0, ..))?.flatten_all()?;
        let codes_f  = first_codes.to_dtype(DType::F32)?;
        let quot     = codes_f.affine(1.0 / cb_f, 0.0)?.floor()?;
        let rem      = (codes_f - quot.affine(cb_f, 0.0)?)?.to_dtype(candle::DType::I64)?;
        let first_embed = self.first_codebook.index_select(&rem, 0)?
            .reshape((batch, seq, 256))?;
        let first_proj = self.conv1d_1x1(
            &first_embed.transpose(1, 2)?,
            &self.first_output_proj,
        )?;

        // Rest quantizers — fold over groups to avoid hardcoded F32 zeros
        let mut rest_embed: Option<Tensor> = None;
        for i in 0..15 {
            let c = codes.i((.., i + 1, ..))?.flatten_all()?;
            let e = self.rest_codebooks[i].index_select(&c, 0)?
                .reshape((batch, seq, 256))?;
            rest_embed = Some(match rest_embed {
                None => e,
                Some(acc) => (acc + e)?,
            });
        }
        let rest_embed = rest_embed.unwrap_or_else(|| {
            Tensor::zeros((batch, seq, 256), DType::F32, device).unwrap()
        });
        let rest_proj = self.conv1d_1x1(
            &rest_embed.transpose(1, 2)?,
            &self.rest_output_proj,
        )?;

        (first_proj + rest_proj) // [B, 512, T]
    }

    fn conv1d_1x1(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let w2 = weight.squeeze(2)?;
        let (b, in_ch, t) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.reshape((b * t, in_ch))?;
        let out = x_t.matmul(&w2.t()?)?;
        let out_ch = out.dim(1)?;
        Ok(out.reshape((b, t, out_ch))?.transpose(1, 2)?)
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, f) = x.dims3()?;
        let out = x.reshape((b * t, f))?.matmul(&weight.t()?)?;
        let out = out.reshape((b, t, out.dim(1)?))?;
        match bias {
            Some(b) => Ok(out.broadcast_add(b)?),
            None => Ok(out),
        }
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = x.broadcast_div(&(var + self.config.rms_norm_eps)?.sqrt()?)?;
        Ok(xn.broadcast_mul(weight)?)
    }

    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = self.config.head_dim;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
    }

    /// Build RoPE cos/sin for `positions` (a slice of absolute frame indices).
    fn rope_for_positions(&self, positions: &[f32], device: &candle::Device) -> Result<(Tensor, Tensor)> {
        let inv: Vec<f32> = (0..self.config.head_dim)
            .step_by(2)
            .map(|i| {
                1.0 / (self.config.rope_theta as f32)
                    .powf(i as f32 / self.config.head_dim as f32)
            })
            .collect();
        let inv_t = Tensor::from_vec(inv, (self.config.head_dim / 2,), device)?;
        let pos_t = Tensor::new(positions, device)?.unsqueeze(1)?;
        let freqs = pos_t.matmul(&inv_t.unsqueeze(0)?)?;
        let cos = freqs.cos()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = freqs.sin()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;
        Ok((cos, sin))
    }

    /// Batch transformer: full seq × seq causal attention.
    fn run_transformer(&self, mut hidden: Tensor, seq: usize) -> Result<Tensor> {
        let device = hidden.device();
        let (b, _, _) = hidden.dims3()?;

        let positions: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let (cos, sin) = self.rope_for_positions(&positions, device)?;

        // Causal mask [1, 1, seq, seq]
        let mut md = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in i + 1..seq {
                md[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(md, (seq, seq), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        for layer in &self.tx_layers {
            hidden = self.run_tx_layer_batch(&hidden, layer, &cos, &sin, &mask, b, seq)?;
        }
        Ok(hidden)
    }

    /// Single-step transformer using the KV cache in `state`.
    fn run_transformer_step(
        &self,
        mut hidden: Tensor,  // [1, 1, hidden_size]
        state: &mut Decoder12HzState,
    ) -> Result<Tensor> {
        let device = hidden.device();
        let offset = state.offset;

        // RoPE for the single new position
        let positions = [offset as f32];
        let (cos, sin) = self.rope_for_positions(&positions, device)?;

        for (layer, kv) in self.tx_layers.iter().zip(state.kv_caches.iter_mut()) {
            hidden = self.run_tx_layer_step(&hidden, layer, &cos, &sin, kv, offset)?;
        }
        Ok(hidden)
    }

    /// Batch transformer layer: full seq attention.
    fn run_tx_layer_batch(
        &self,
        hidden: &Tensor,
        l: &TxLayerW,
        cos: &Tensor, sin: &Tensor,
        mask: &Tensor,
        b: usize, seq: usize,
    ) -> Result<Tensor> {
        let nh = self.config.num_heads;
        let d  = self.config.head_dim;

        let normed = self.rms_norm(hidden, &l.input_ln)?;
        let q = self.linear_3d(&normed, &l.q, None)?.reshape((b, seq, nh, d))?.transpose(1, 2)?;
        let k = self.linear_3d(&normed, &l.k, None)?.reshape((b, seq, nh, d))?.transpose(1, 2)?;
        let v = self.linear_3d(&normed, &l.v, None)?.reshape((b, seq, nh, d))?.transpose(1, 2)?;
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;
        let scale = (d as f64).powf(-0.5);
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?
            .broadcast_add(&mask.to_dtype(q.dtype())?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out  = attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq, nh * d))?;
        let out = self.linear_3d(&out, &l.o, None)?;
        let out = out.broadcast_mul(&l.attn_scale)?;
        let hidden = (hidden + out)?;

        let normed = self.rms_norm(&hidden, &l.post_ln)?;
        let gate = self.linear_3d(&normed, &l.gate, None)?;
        let up   = self.linear_3d(&normed, &l.up, None)?;
        let mlp  = self.linear_3d(
            &(candle_nn::ops::silu(&gate)? * up)?,
            &l.down, None,
        )?;
        let mlp = mlp.broadcast_mul(&l.mlp_scale)?;
        Ok((hidden + mlp)?)
    }

    /// Single-step transformer layer: Q for new frame, K/V from cache.
    fn run_tx_layer_step(
        &self,
        hidden: &Tensor,      // [1, 1, hidden_size]
        l: &TxLayerW,
        cos: &Tensor, sin: &Tensor,  // [1, 1, 1, head_dim] — single position
        kv: &mut DecoderKVCache,
        _offset: usize,
    ) -> Result<Tensor> {
        let nh = self.config.num_heads;
        let d  = self.config.head_dim;
        let b  = 1usize;

        let normed = self.rms_norm(hidden, &l.input_ln)?;

        // Project only the new single position
        let q_new = self.linear_3d(&normed, &l.q, None)?.reshape((b, 1, nh, d))?.transpose(1, 2)?;
        let k_new = self.linear_3d(&normed, &l.k, None)?.reshape((b, 1, nh, d))?.transpose(1, 2)?;
        let v_new = self.linear_3d(&normed, &l.v, None)?.reshape((b, 1, nh, d))?.transpose(1, 2)?;

        let q_new = self.apply_rope(&q_new, cos, sin)?;
        let k_new = self.apply_rope(&k_new, cos, sin)?;

        // Append to KV cache and get full history
        let (k_full, v_full) = kv.append(k_new, v_new)?;
        let t_past = k_full.dim(2)?;

        let scale = (d as f64).powf(-0.5);
        let q = q_new.contiguous()?;
        let k = k_full.contiguous()?;
        let v = v_full.contiguous()?;

        // No mask needed: q is length 1, all past positions are visible
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let _ = t_past; // all past tokens attend freely — causal by construction
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out  = attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, 1, nh * d))?;
        let out = self.linear_3d(&out, &l.o, None)?;
        let out = out.broadcast_mul(&l.attn_scale)?;
        let hidden = (hidden + out)?;

        let normed = self.rms_norm(&hidden, &l.post_ln)?;
        let gate = self.linear_3d(&normed, &l.gate, None)?;
        let up   = self.linear_3d(&normed, &l.up, None)?;
        let mlp  = self.linear_3d(
            &(candle_nn::ops::silu(&gate)? * up)?,
            &l.down, None,
        )?;
        let mlp = mlp.broadcast_mul(&l.mlp_scale)?;
        Ok((hidden + mlp)?)
    }
}

// ── Streaming impls for ConvNeXtBlock ─────────────────────────────────────

impl ConvNeXtBlock {
    /// Streaming forward: process any `T ≥ 1` frames, updating dwconv state.
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        state: &mut ConvNeXtBlockState,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.dwconv.forward_with_state(x, &mut state.dw)?;
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma.unsqueeze(0)?.unsqueeze(0)?)?;
        let h = h.transpose(1, 2)?;
        Ok((h + residual)?)
    }
}

// ── Streaming impls for DecoderBlock ──────────────────────────────────────

impl DecoderBlock {
    pub fn forward_streaming(
        &self,
        x: &Tensor,
        state: &mut DecoderBlockState,
    ) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let h = self.upsample.forward_with_state(&h, &mut state.upsample)?;
        let h = self.res1.forward_streaming(&h, &mut state.res1)?;
        let h = self.res2.forward_streaming(&h, &mut state.res2)?;
        self.res3.forward_streaming(&h, &mut state.res3)
    }
}

// ── Streaming impls for ResidualUnit ──────────────────────────────────────

impl super::decoder_block::ResidualUnit {
    pub fn forward_streaming(
        &self,
        x: &Tensor,
        state: &mut ResidualUnitState,
    ) -> Result<Tensor> {
        let r = x;
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward_with_state(&h, &mut state.conv1)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward_with_state(&h, &mut state.conv2)?;
        Ok((h + r)?)
    }
}
