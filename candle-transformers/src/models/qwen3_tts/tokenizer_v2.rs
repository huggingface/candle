//! Qwen3-TTS Tokenizer V2 (12Hz) decoder-only implementation.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{conv1d, conv_transpose1d, layer_norm, LayerNorm, VarBuilder};
use serde::Deserialize;

use crate::models::mimi;
use crate::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};

fn build_causal_mask(
    bsz: usize,
    seq_len: usize,
    seqlen_offset: usize,
    sliding_window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let disallow = i < j || j + sliding_window < i;
            if disallow {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask, (seq_len, seq_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((seq_len, seqlen_offset), dtype, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((bsz, 1, seq_len, seq_len + seqlen_offset))?
        .to_dtype(dtype)
}

fn default_input_sample_rate() -> usize {
    24_000
}

fn default_output_sample_rate() -> usize {
    24_000
}

fn default_decode_upsample_rate() -> usize {
    1920
}

fn default_encode_downsample_rate() -> usize {
    1920
}

fn default_encoder_valid_num_quantizers() -> usize {
    16
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTokenizerV2Config {
    #[serde(default)]
    pub encoder_config: Option<serde_json::Value>,
    pub decoder_config: Qwen3TtsTokenizerV2DecoderConfig,
    #[serde(default = "default_encoder_valid_num_quantizers")]
    pub encoder_valid_num_quantizers: usize,
    #[serde(default = "default_input_sample_rate")]
    pub input_sample_rate: usize,
    #[serde(default = "default_output_sample_rate")]
    pub output_sample_rate: usize,
    #[serde(default = "default_decode_upsample_rate")]
    pub decode_upsample_rate: usize,
    #[serde(default = "default_encode_downsample_rate")]
    pub encode_downsample_rate: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTokenizerV2DecoderConfig {
    pub codebook_size: usize,
    pub hidden_size: usize,
    pub latent_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub attention_bias: bool,
    pub sliding_window: usize,
    pub intermediate_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_scale_initial_scale: f64,
    pub rms_norm_eps: f64,
    pub num_hidden_layers: usize,
    pub num_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub decoder_dim: usize,
    pub attention_dropout: f64,
    #[serde(default)]
    pub codebook_dim: Option<usize>,
}

impl Qwen3TtsTokenizerV2DecoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn codebook_dim(&self) -> usize {
        self.codebook_dim.unwrap_or(self.latent_dim)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3TtsTokenizerV2DecoderConfig, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct DecoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: candle_nn::kv_cache::ConcatKvCache,
}

impl DecoderAttention {
    fn new(
        cfg: &Qwen3TtsTokenizerV2DecoderConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: cfg.hidden_size,
            rotary,
            kv_cache: candle_nn::kv_cache::ConcatKvCache::new(2),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attn_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: candle_nn::Activation,
}

impl DecoderMlp {
    fn new(cfg: &Qwen3TtsTokenizerV2DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }
}

impl Module for DecoderMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderRmsNorm {
    inner: RmsNorm,
}

impl DecoderRmsNorm {
    fn new(hidden: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: RmsNorm::new(hidden, eps, vb)?,
        })
    }
}

impl Module for DecoderRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(hidden: usize, init_scale: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get_with_hints((hidden,), "scale", candle_nn::Init::Const(init_scale))?;
        Ok(Self { scale })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: DecoderAttention,
    mlp: DecoderMlp,
    ln1: DecoderRmsNorm,
    ln2: DecoderRmsNorm,
    attn_scale: LayerScale,
    mlp_scale: LayerScale,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3TtsTokenizerV2DecoderConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: DecoderAttention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: DecoderMlp::new(cfg, vb.pp("mlp"))?,
            ln1: DecoderRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: DecoderRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            attn_scale: LayerScale::new(
                cfg.hidden_size,
                cfg.layer_scale_initial_scale,
                vb.pp("self_attn_layer_scale"),
            )?,
            mlp_scale: LayerScale::new(
                cfg.hidden_size,
                cfg.layer_scale_initial_scale,
                vb.pp("mlp_layer_scale"),
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attn_mask, offset)?;
        let xs = (residual + xs.apply(&self.attn_scale)?)?;
        let residual = &xs;
        let xs = xs.apply(&self.ln2)?.apply(&self.mlp)?;
        Ok((residual + xs.apply(&self.mlp_scale)?)?)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
struct DecoderTransformer {
    layers: Vec<DecoderLayer>,
    norm: DecoderRmsNorm,
    rotary: Arc<RotaryEmbedding>,
    input_proj: Linear,
    output_proj: Linear,
    device: Device,
    dtype: DType,
    sliding_window: usize,
}

impl DecoderTransformer {
    fn new(cfg: &Qwen3TtsTokenizerV2DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(idx))?);
        }
        let norm = DecoderRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let input_proj = linear_b(cfg.latent_dim, cfg.hidden_size, true, vb.pp("input_proj"))?;
        let output_proj = linear_b(cfg.hidden_size, cfg.latent_dim, true, vb.pp("output_proj"))?;
        Ok(Self {
            layers,
            norm,
            rotary,
            input_proj,
            output_proj,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = inputs_embeds.dims3()?;
        let inputs_embeds = inputs_embeds.apply(&self.input_proj)?;
        let attn_mask = if seq_len > 1 {
            build_causal_mask(
                b,
                seq_len,
                seqlen_offset,
                self.sliding_window,
                &self.device,
                self.dtype,
            )?
        } else {
            Tensor::zeros((b, 1, 1, seqlen_offset + 1), self.dtype, &self.device)?
        };
        let mut xs = inputs_embeds;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, Some(&attn_mask), seqlen_offset)?;
        }
        let xs = xs.apply(&self.norm)?;
        xs.apply(&self.output_proj)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    eps: f64,
}

impl SnakeBeta {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        let beta = vb.get((channels,), "beta")?;
        Ok(Self {
            alpha,
            beta,
            eps: 1e-9,
        })
    }
}

impl Module for SnakeBeta {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let sin = (xs.broadcast_mul(&alpha)?).sin()?;
        let sin2 = (&sin * &sin)?;
        xs + sin2.broadcast_div(&(beta + self.eps)?)
    }
}

#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: candle_nn::Conv1d,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    padding: usize,
}

impl CausalConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv1dConfig {
            stride,
            dilation,
            groups,
            padding: 0,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let kernel_size = (kernel_size - 1) * dilation + 1;
        let padding = kernel_size.saturating_sub(stride);
        Ok(Self {
            conv,
            stride,
            kernel_size,
            dilation,
            padding,
        })
    }

    fn extra_padding(&self, length: usize) -> usize {
        let n_frames = (length + self.padding).saturating_sub(self.kernel_size) as f64
            / self.stride as f64
            + 1.0;
        let ideal_length =
            (n_frames.ceil() as usize - 1) * self.stride + (self.kernel_size - self.padding);
        ideal_length.saturating_sub(length)
    }
}

impl Module for CausalConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let len = xs.dim(D::Minus1)?;
        let extra = self.extra_padding(len);
        let xs = xs.pad_with_zeros(D::Minus1, self.padding, extra)?;
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct CausalTransConv1d {
    conv: candle_nn::ConvTranspose1d,
    left_pad: usize,
    right_pad: usize,
}

impl CausalTransConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::ConvTranspose1dConfig {
            stride,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        };
        let conv = conv_transpose1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let pad = kernel_size.saturating_sub(stride);
        let left_pad = pad;
        let right_pad = pad;
        Ok(Self {
            conv,
            left_pad,
            right_pad,
        })
    }
}

impl Module for CausalTransConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.conv)?;
        let len = xs.dim(D::Minus1)?;
        let end = len.saturating_sub(self.right_pad);
        xs.narrow(D::Minus1, self.left_pad, end - self.left_pad)
    }
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = CausalConv1d::new(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?;
        let norm = layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pw1 = linear_b(dim, 4 * dim, true, vb.pp("pwconv1"))?;
        let pw2 = linear_b(4 * dim, dim, true, vb.pp("pwconv2"))?;
        let gamma = vb.get((dim,), "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pw1,
            pw2,
            gamma,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.dwconv.forward(xs)?;
        let xs = xs.transpose(1, 2)?;
        let xs = xs.apply(&self.norm)?;
        let xs = xs.apply(&self.pw1)?.gelu()?.apply(&self.pw2)?;
        let xs = xs.broadcast_mul(&self.gamma)?;
        let xs = xs.transpose(1, 2)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl DecoderResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, vb.pp("act1"))?,
            conv1: CausalConv1d::new(dim, dim, 7, 1, dilation, 1, vb.pp("conv1"))?,
            act2: SnakeBeta::new(dim, vb.pp("act2"))?,
            conv2: CausalConv1d::new(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?,
        })
    }
}

impl Module for DecoderResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.act1.forward(xs)?;
        let xs = self.conv1.forward(&xs)?;
        let xs = self.act2.forward(&xs)?;
        let xs = self.conv2.forward(&xs)?;
        residual + xs
    }
}

struct DecoderBlock {
    blocks: Vec<Box<dyn Module>>,
}

impl DecoderBlock {
    fn new(
        cfg: &Qwen3TtsTokenizerV2DecoderConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_dim = cfg.decoder_dim / 2usize.pow(layer_idx as u32);
        let out_dim = cfg.decoder_dim / 2usize.pow((layer_idx + 1) as u32);
        let upsample_rate = cfg.upsample_rates[layer_idx];
        let mut blocks: Vec<Box<dyn Module>> = Vec::new();
        blocks.push(Box::new(SnakeBeta::new(in_dim, vb.pp("block.0"))?));
        blocks.push(Box::new(CausalTransConv1d::new(
            in_dim,
            out_dim,
            2 * upsample_rate,
            upsample_rate,
            vb.pp("block.1"),
        )?));
        let mut idx = 2usize;
        for dilation in [1usize, 3, 9] {
            blocks.push(Box::new(DecoderResidualUnit::new(
                out_dim,
                dilation,
                vb.pp(format!("block.{idx}")),
            )?));
            idx += 1;
        }
        Ok(Self { blocks })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = xs.clone();
        for block in self.blocks.iter() {
            out = block.forward(&out)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct EuclideanCodebook {
    cluster_usage: Tensor,
    embedding_sum: Tensor,
    epsilon: f64,
}

impl EuclideanCodebook {
    fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        let cluster_usage = vb.get((codebook_size,), "cluster_usage")?;
        let embedding_sum = vb.get((codebook_size, dim), "embedding_sum")?;
        Ok(Self {
            cluster_usage,
            embedding_sum,
            epsilon: 1e-5,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let usage = self
            .cluster_usage
            .clamp(self.epsilon as f32, f32::INFINITY)?;
        let embedding = self.embedding_sum.broadcast_div(&usage.unsqueeze(1)?)?;
        let flat = codes.flatten_all()?;
        let gathered = embedding.index_select(&flat, 0)?;
        let shape = codes.dims().to_vec();
        let mut out_shape = shape;
        out_shape.push(embedding.dim(1)?);
        gathered.reshape(out_shape)
    }
}

#[derive(Debug, Clone)]
struct VectorQuantization {
    project_out: Option<Linear>,
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    fn new(codebook_dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            project_out: None,
            codebook: EuclideanCodebook::new(codebook_dim, codebook_size, vb.pp("_codebook"))?,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut quantized = self.codebook.decode(codes)?;
        if let Some(proj) = &self.project_out {
            quantized = quantized.apply(proj)?;
        }
        quantized.transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    fn new(
        cfg: &Qwen3TtsTokenizerV2DecoderConfig,
        vb: VarBuilder,
        n_q: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_q);
        for idx in 0..n_q {
            layers.push(VectorQuantization::new(
                codebook_dim,
                cfg.codebook_size,
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_codes = codes.i(idx)?;
            let q = layer.decode(&layer_codes)?;
            sum = Some(match sum {
                Some(acc) => (acc + q)?,
                None => q,
            });
        }
        sum.ok_or_else(|| candle::Error::Msg("no quantizer layers".into()))
    }
}

#[derive(Debug, Clone)]
struct ResidualVectorQuantizer {
    output_proj: Option<candle_nn::Conv1d>,
    vq: ResidualVectorQuantization,
}

impl ResidualVectorQuantizer {
    fn new(
        cfg: &Qwen3TtsTokenizerV2DecoderConfig,
        vb: VarBuilder,
        n_q: usize,
        force_projection: bool,
    ) -> Result<Self> {
        let dimension = cfg.codebook_dim() / 2;
        let output_dimension = cfg.codebook_dim();
        let output_proj = if output_dimension == dimension && !force_projection {
            None
        } else {
            let cfg1 = candle_nn::Conv1dConfig {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            Some(conv1d(
                dimension,
                output_dimension,
                1,
                cfg1,
                vb.pp("output_proj"),
            )?)
        };
        let vq = ResidualVectorQuantization::new(cfg, vb.pp("vq"), n_q, dimension)?;
        Ok(Self { output_proj, vq })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = codes.transpose(0, 1)?;
        let mut quantized = self.vq.decode(&codes)?;
        if let Some(proj) = &self.output_proj {
            quantized = quantized.apply(proj)?;
        }
        Ok(quantized)
    }
}

#[derive(Debug, Clone)]
struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    fn new(cfg: &Qwen3TtsTokenizerV2DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let n_q = cfg.num_quantizers;
        let n_q_semantic = 1;
        let rvq_first = ResidualVectorQuantizer::new(cfg, vb.pp("rvq_first"), n_q_semantic, true)?;
        let rvq_rest =
            ResidualVectorQuantizer::new(cfg, vb.pp("rvq_rest"), n_q - n_q_semantic, true)?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q_semantic,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let first = self
            .rvq_first
            .decode(&codes.i((.., 0..self.n_q_semantic, ..))?)?;
        if codes.dim(1)? > self.n_q_semantic {
            let rest = self
                .rvq_rest
                .decode(&codes.i((.., self.n_q_semantic.., ..))?)?;
            first + rest
        } else {
            Ok(first)
        }
    }
}

struct TokenizerDecoder {
    total_upsample: usize,
    pre_transformer: DecoderTransformer,
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    upsample: Vec<Vec<Box<dyn Module>>>,
    decoder: Vec<Box<dyn Module>>,
    cfg: Qwen3TtsTokenizerV2DecoderConfig,
}

impl TokenizerDecoder {
    fn new(cfg: &Qwen3TtsTokenizerV2DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let total_upsample = cfg
            .upsample_rates
            .iter()
            .chain(cfg.upsampling_ratios.iter())
            .product();
        let pre_transformer = DecoderTransformer::new(cfg, vb.pp("pre_transformer"))?;
        let quantizer = SplitResidualVectorQuantizer::new(cfg, vb.pp("quantizer"))?;
        let pre_conv = CausalConv1d::new(
            cfg.codebook_dim(),
            cfg.latent_dim,
            3,
            1,
            1,
            1,
            vb.pp("pre_conv"),
        )?;
        let mut upsample = Vec::new();
        for (idx, factor) in cfg.upsampling_ratios.iter().enumerate() {
            let mut blocks: Vec<Box<dyn Module>> = Vec::new();
            blocks.push(Box::new(CausalTransConv1d::new(
                cfg.latent_dim,
                cfg.latent_dim,
                *factor,
                *factor,
                vb.pp(format!("upsample.{idx}.0")),
            )?));
            blocks.push(Box::new(ConvNeXtBlock::new(
                cfg.latent_dim,
                vb.pp(format!("upsample.{idx}.1")),
            )?));
            upsample.push(blocks);
        }
        let mut decoder: Vec<Box<dyn Module>> = Vec::new();
        decoder.push(Box::new(CausalConv1d::new(
            cfg.latent_dim,
            cfg.decoder_dim,
            7,
            1,
            1,
            1,
            vb.pp("decoder.0"),
        )?));
        for idx in 0..cfg.upsample_rates.len() {
            let idx_plus = idx + 1;
            decoder.push(Box::new(DecoderBlock::new(
                cfg,
                idx,
                vb.pp(format!("decoder.{idx_plus}")),
            )?));
        }
        let output_dim = cfg.decoder_dim / 2usize.pow(cfg.upsample_rates.len() as u32);
        let tail_idx = cfg.upsample_rates.len() + 1;
        decoder.push(Box::new(SnakeBeta::new(
            output_dim,
            vb.pp(format!("decoder.{tail_idx}")),
        )?));
        decoder.push(Box::new(CausalConv1d::new(
            output_dim,
            1,
            7,
            1,
            1,
            1,
            vb.pp(format!("decoder.{}", tail_idx + 1)),
        )?));
        Ok(Self {
            total_upsample,
            pre_transformer,
            quantizer,
            pre_conv,
            upsample,
            decoder,
            cfg: cfg.clone(),
        })
    }

    fn forward(&mut self, codes: &Tensor) -> Result<Tensor> {
        if codes.dim(1)? != self.cfg.num_quantizers {
            candle::bail!(
                "expected {} quantizers, got {}",
                self.cfg.num_quantizers,
                codes.dim(1)?
            );
        }
        self.pre_transformer.clear_kv_cache();
        let mut hidden = self.quantizer.decode(codes)?;
        hidden = self.pre_conv.forward(&hidden)?.transpose(1, 2)?;
        hidden = self.pre_transformer.forward(&hidden, 0)?;
        hidden = hidden.transpose(1, 2)?;
        for blocks in self.upsample.iter() {
            for block in blocks.iter() {
                hidden = block.forward(&hidden)?;
            }
        }
        let mut wav = hidden;
        for block in self.decoder.iter() {
            wav = block.forward(&wav)?;
        }
        Ok(wav.clamp(-1.0, 1.0)?)
    }

    fn chunked_decode(
        &mut self,
        codes: &Tensor,
        chunk_size: usize,
        left_context: usize,
    ) -> Result<Tensor> {
        let total_len = codes.dim(D::Minus1)?;
        let mut start = 0usize;
        let mut chunks: Vec<Tensor> = Vec::new();
        while start < total_len {
            let end = usize::min(start + chunk_size, total_len);
            let context = if start >= left_context {
                left_context
            } else {
                start
            };
            let chunk = codes.narrow(D::Minus1, start - context, end - (start - context))?;
            let wav_chunk = self.forward(&chunk)?;
            let trim = context * self.total_upsample;
            let wav_chunk = wav_chunk.narrow(D::Minus1, trim, wav_chunk.dim(D::Minus1)? - trim)?;
            chunks.push(wav_chunk);
            start = end;
        }
        Ok(Tensor::cat(&chunks.iter().collect::<Vec<_>>(), D::Minus1)?)
    }
}

pub struct Qwen3TtsTokenizerV2 {
    encoder: Option<mimi::Model>,
    decoder: TokenizerDecoder,
    cfg: Qwen3TtsTokenizerV2Config,
    device: Device,
}

impl Qwen3TtsTokenizerV2 {
    pub fn new(cfg: Qwen3TtsTokenizerV2Config, vb: VarBuilder) -> Result<Self> {
        let encoder = build_encoder(&cfg, vb.pp("encoder"))?;
        let decoder = TokenizerDecoder::new(&cfg.decoder_config, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            device: vb.device().clone(),
            cfg,
        })
    }

    pub fn output_sample_rate(&self) -> usize {
        self.cfg.output_sample_rate
    }

    pub fn decode_codes(&mut self, codes: &Tensor) -> Result<Vec<f32>> {
        let mut codes = match codes.dims().len() {
            2 => codes.unsqueeze(0)?,
            3 => codes.clone(),
            _ => candle::bail!("codes must be 2D or 3D"),
        };
        let num_q = self.cfg.decoder_config.num_quantizers;
        if codes.dim(1)? != num_q {
            if codes.dim(2)? == num_q {
                codes = codes.transpose(1, 2)?;
            } else {
                candle::bail!("codes must have num_quantizers in dim 1 or 2");
            }
        }
        let wav = self.decoder.chunked_decode(&codes, 300, 25)?;
        let wav = wav.i((0, 0))?;
        Ok(wav.to_vec1::<f32>()?)
    }

    pub fn encode_audio(
        &mut self,
        audio: &Tensor,
        padding_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| candle::Error::Msg("speech tokenizer encoder not initialized".into()))?;
        encoder.reset_state();
        let mut audio = match audio.dims().len() {
            1 => audio.unsqueeze(0)?,
            2 => audio.clone(),
            _ => candle::bail!("audio must be 1D or 2D (batch, samples)"),
        };
        if audio.device().location() != self.device.location() {
            audio = audio.to_device(&self.device)?;
        }
        let bsz = audio.dim(0)?;
        let audio_in = audio.unsqueeze(1)?; // [B, 1, T]
        let mut codes = encoder.encode(&audio_in)?;
        if codes.dim(1)? > self.cfg.encoder_valid_num_quantizers {
            codes = codes.narrow(1, 0, self.cfg.encoder_valid_num_quantizers)?;
        }
        let mut out = Vec::with_capacity(bsz);
        for idx in 0..bsz {
            let mut codes_b = codes.i(idx)?; // [Q, T]
            let total_len = codes_b.dim(1)?;
            let keep_len = if let Some(mask) = padding_mask {
                let mask = mask.i(idx)?;
                let valid = mask.to_dtype(DType::F32)?.sum_all()?.to_vec0::<f32>()? as usize;
                let mut keep =
                    (valid + self.cfg.encode_downsample_rate - 1) / self.cfg.encode_downsample_rate;
                if keep > total_len {
                    keep = total_len;
                }
                keep
            } else {
                total_len
            };
            if keep_len < total_len {
                codes_b = codes_b.narrow(1, 0, keep_len)?;
            }
            let codes_b = codes_b.transpose(0, 1)?; // [T, Q]
            out.push(codes_b);
        }
        Ok(out)
    }
}

fn build_encoder(cfg: &Qwen3TtsTokenizerV2Config, vb: VarBuilder) -> Result<Option<mimi::Model>> {
    let num_codebooks = cfg
        .encoder_config
        .as_ref()
        .and_then(|v| v.get("quantizer_n_q").and_then(|v| v.as_u64()))
        .map(|v| v as usize)
        .or(Some(cfg.encoder_valid_num_quantizers));
    let mut enc_cfg = mimi::Config::v0_1(num_codebooks);
    if let Some(enc) = cfg.encoder_config.as_ref() {
        if let Some(sample_rate) = enc.get("sample_rate").and_then(|v| v.as_f64()) {
            enc_cfg.sample_rate = sample_rate;
        }
        if let Some(frame_rate) = enc.get("frame_rate").and_then(|v| v.as_f64()) {
            enc_cfg.frame_rate = frame_rate;
        }
        if let Some(quantizer_bins) = enc.get("quantizer_bins").and_then(|v| v.as_u64()) {
            enc_cfg.quantizer_bins = quantizer_bins as usize;
        }
        if let Some(quantizer_dim) = enc.get("quantizer_dim").and_then(|v| v.as_u64()) {
            enc_cfg.quantizer_dim = quantizer_dim as usize;
        }
    }
    let encoder = mimi::Model::new(enc_cfg, vb)?;
    Ok(Some(encoder))
}
