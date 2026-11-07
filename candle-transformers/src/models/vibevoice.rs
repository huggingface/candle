//! VibeVoice model implementation for TTS and ASR.
//!
//! VibeVoice is a unified speech model from Microsoft Research supporting both
//! Text-to-Speech (TTS) and Automatic Speech Recognition (ASR). It compresses
//! audio via ultra-low frame rate tokenizers (7.5 Hz) and feeds continuous latent
//! embeddings into a Qwen2.5 LLM backbone.
//!
//! References:
//! - [VibeVoice Technical Report](https://arxiv.org/abs/2505.XXXXX)

use crate::models::qwen2;
use crate::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{conv1d, conv1d_no_bias, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

// ---------------------------------------------------------------------------
// Streaming convolution cache (for VAE decoder frame-by-frame decoding)
// ---------------------------------------------------------------------------

/// Cache for streaming causal convolution / transposed convolution.
///
/// Each conv layer is assigned a sequential index and stores the tail of its
/// previous input so that consecutive single-frame calls produce the same
/// result as a single multi-frame call.
#[derive(Debug, Clone)]
pub struct StreamingConvCache {
    /// Per-layer cached tensor (the "context" from the previous frame).
    slots: Vec<Option<Tensor>>,
    /// Counter used during a forward pass to assign each layer to the next slot.
    cursor: usize,
}

impl StreamingConvCache {
    /// Create a new empty cache with `n_layers` slots.
    pub fn new(n_layers: usize) -> Self {
        Self {
            slots: vec![None; n_layers],
            cursor: 0,
        }
    }

    /// Reset the cursor to 0 (call at the start of each frame's forward pass).
    pub fn reset_cursor(&mut self) {
        self.cursor = 0;
    }

    /// Get the next slot index and advance the cursor.
    fn next_slot(&mut self) -> usize {
        let idx = self.cursor;
        self.cursor += 1;
        idx
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

fn parse_depths(s: &str) -> Vec<usize> {
    s.split('-')
        .filter_map(|x| x.parse::<usize>().ok())
        .collect()
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenizerConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default = "default_true")]
    pub causal: bool,
    pub vae_dim: usize,
    #[serde(default)]
    pub fix_std: f64,
    #[serde(default = "default_none_str")]
    pub std_dist_type: String,
    #[serde(default = "default_depthwise")]
    pub mixer_layer: String,
    #[serde(default = "default_none_str")]
    pub conv_norm: String,
    #[serde(default = "default_constant")]
    pub pad_mode: String,
    #[serde(default = "default_true")]
    pub disable_last_norm: bool,
    #[serde(default = "default_rmsnorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_init_value: f64,
    #[serde(default = "default_n_filters")]
    pub encoder_n_filters: usize,
    pub encoder_ratios: Vec<usize>,
    pub encoder_depths: String,
    pub decoder_n_filters: Option<usize>,
    pub decoder_ratios: Option<Vec<usize>>,
    pub decoder_depths: Option<String>,
}

fn default_channels() -> usize { 1 }
fn default_true() -> bool { true }
fn default_none_str() -> String { "none".into() }
fn default_depthwise() -> String { "depthwise_conv".into() }
fn default_constant() -> String { "constant".into() }
fn default_rmsnorm() -> String { "RMSNorm".into() }
fn default_layernorm_eps() -> f64 { 1e-5 }
fn default_layer_scale() -> f64 { 1e-6 }
fn default_n_filters() -> usize { 32 }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DiffusionHeadConfig {
    pub hidden_size: usize,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_ffn_ratio")]
    pub head_ffn_ratio: f64,
    #[serde(default = "default_latent_size")]
    pub latent_size: usize,
    #[serde(default = "default_layernorm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_v_prediction")]
    pub prediction_type: String,
    #[serde(default = "default_ddpm_steps")]
    pub ddpm_num_steps: usize,
    #[serde(default = "default_ddpm_inf_steps")]
    pub ddpm_num_inference_steps: usize,
    #[serde(default = "default_cosine")]
    pub ddpm_beta_schedule: String,
}

fn default_head_layers() -> usize { 4 }
fn default_ffn_ratio() -> f64 { 3.0 }
fn default_latent_size() -> usize { 64 }
fn default_v_prediction() -> String { "v_prediction".into() }
fn default_ddpm_steps() -> usize { 1000 }
fn default_ddpm_inf_steps() -> usize { 20 }
fn default_cosine() -> String { "cosine".into() }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub acoustic_tokenizer_config: TokenizerConfig,
    pub semantic_tokenizer_config: TokenizerConfig,
    #[serde(deserialize_with = "deserialize_decoder_config")]
    pub decoder_config: qwen2::Config,
    pub diffusion_head_config: Option<DiffusionHeadConfig>,
}

// ---------------------------------------------------------------------------
// ASR-specific configuration (microsoft/VibeVoice-ASR-HF)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AsrEncoderConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    pub depths: Vec<usize>,
    pub downsampling_ratios: Vec<usize>,
    pub hidden_size: usize,       // output dim (vae_dim equivalent)
    #[serde(default = "default_n_filters")]
    pub num_filters: usize,
    #[serde(default = "default_asr_kernel")]
    pub kernel_size: usize,
    #[serde(default = "default_layernorm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_init_value: f64,
    #[serde(default)]
    pub vae_std: f64,
}

fn default_asr_kernel() -> usize { 7 }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AsrConfig {
    pub acoustic_tokenizer_encoder_config: AsrEncoderConfig,
    pub semantic_tokenizer_encoder_config: AsrEncoderConfig,
    #[serde(deserialize_with = "deserialize_text_config")]
    pub text_config: qwen2::Config,
    #[serde(default = "default_audio_bos")]
    pub audio_bos_token_id: u32,
    #[serde(default = "default_audio_eos")]
    pub audio_eos_token_id: u32,
    #[serde(default = "default_audio_token")]
    pub audio_token_id: u32,
}

fn default_audio_bos() -> u32 { 151646 }
fn default_audio_eos() -> u32 { 151647 }
fn default_audio_token() -> u32 { 151648 }

/// Custom deserializer for the nested decoder_config that handles
/// `"sliding_window": null` (VibeVoice ships this) by defaulting to
/// `max_position_embeddings` so the qwen2 causal mask works correctly.
fn fix_sliding_window(v: &mut serde_json::Value) {
    if let serde_json::Value::Object(ref mut map) = v {
        let default_sw = map
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(32768);
        match map.get("sliding_window") {
            Some(serde_json::Value::Null) | None => {
                map.insert("sliding_window".into(), serde_json::Value::Number(default_sw.into()));
            }
            _ => {}
        }
        // Handle rope_parameters.rope_theta → rope_theta (ASR config format)
        if !map.contains_key("rope_theta") {
            if let Some(rp) = map.get("rope_parameters").and_then(|v| v.as_object()) {
                if let Some(theta) = rp.get("rope_theta") {
                    map.insert("rope_theta".into(), theta.clone());
                }
            }
        }
    }
}

fn deserialize_decoder_config<'de, D>(deserializer: D) -> std::result::Result<qwen2::Config, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let mut v = serde_json::Value::deserialize(deserializer)?;
    fix_sliding_window(&mut v);
    serde_json::from_value(v).map_err(serde::de::Error::custom)
}

fn deserialize_text_config<'de, D>(deserializer: D) -> std::result::Result<qwen2::Config, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let mut v = serde_json::Value::deserialize(deserializer)?;
    fix_sliding_window(&mut v);
    serde_json::from_value(v).map_err(serde::de::Error::custom)
}

// ---------------------------------------------------------------------------
// Utility functions (adapted from encodec.rs)
// ---------------------------------------------------------------------------

fn get_extra_padding_for_conv1d(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    // Must use float arithmetic matching Python exactly (no ceil on n_frames)
    let length_f = length as f64;
    let n_frames = (length_f + padding_total as f64 - kernel_size as f64) / stride as f64 + 1.0;
    let ideal_length = (n_frames - 1.0) * stride as f64 + kernel_size as f64 - padding_total as f64;
    (ideal_length - length_f).max(0.0) as usize
}

fn pad1d_constant(xs: &Tensor, pad_l: usize, pad_r: usize) -> Result<Tensor> {
    xs.pad_with_zeros(D::Minus1, pad_l, pad_r)
}

// ---------------------------------------------------------------------------
// CausalConv1d — wraps Conv1d with causal padding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    causal: bool,
}

impl CausalConv1d {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv = if bias {
            conv1d(in_c, out_c, kernel_size, cfg, vb)?
        } else {
            conv1d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
        };
        Ok(Self { conv, kernel_size, stride, dilation, causal })
    }

    fn padding_total(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    /// Context size cached between streaming frames.
    /// Matches Python: `(kernel_size - 1) * dilation - (stride - 1)`
    fn context_size(&self) -> usize {
        let pt = self.padding_total();
        pt.saturating_sub(self.stride - 1)
    }

    /// Streaming forward: prepend cached context, run conv (no extra padding),
    /// save last `context_size` samples of the combined input.
    fn forward_streaming(&self, xs: &Tensor, cache: &mut StreamingConvCache) -> Result<Tensor> {
        let slot = cache.next_slot();
        let ctx = self.context_size();

        // Prepend cached context (or zeros on the first call)
        let input = match &cache.slots[slot] {
            Some(cached) => Tensor::cat(&[cached, xs], D::Minus1)?,
            None if ctx > 0 => {
                let dims = xs.dims();
                let zeros = Tensor::zeros(
                    &[dims[0], dims[1], ctx],
                    xs.dtype(),
                    xs.device(),
                )?;
                Tensor::cat(&[&zeros, xs], D::Minus1)?
            }
            None => xs.clone(),
        };

        // Run convolution directly (context replaces causal padding)
        let output = self.conv.forward(&input)?;

        // Save last context_size samples of the combined input
        if ctx > 0 {
            let total_len = input.dim(D::Minus1)?;
            let start = total_len.saturating_sub(ctx);
            cache.slots[slot] = Some(input.narrow(D::Minus1, start, ctx)?);
        }

        Ok(output)
    }
}

impl Module for CausalConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let padding_total = self.padding_total();
        let length = xs.dim(D::Minus1)?;
        let extra_padding =
            get_extra_padding_for_conv1d(length, self.kernel_size, self.stride, padding_total);
        if self.causal {
            let xs = pad1d_constant(xs, padding_total, extra_padding)?;
            self.conv.forward(&xs)
        } else {
            let pad_l = padding_total / 2;
            let pad_r = padding_total - pad_l + extra_padding;
            let xs = pad1d_constant(xs, pad_l, pad_r)?;
            self.conv.forward(&xs)
        }
    }
}

// ---------------------------------------------------------------------------
// CausalConvTranspose1d — wraps ConvTranspose1d with causal trimming
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    kernel_size: usize,
    stride: usize,
    causal: bool,
}

impl CausalConvTranspose1d {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = if bias {
            conv_transpose1d(in_c, out_c, kernel_size, cfg, vb)?
        } else {
            candle_nn::conv_transpose1d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
        };
        Ok(Self { conv, kernel_size, stride, causal: true })
    }
}

impl CausalConvTranspose1d {
    /// Context size for streaming cache.
    /// Matches Python: `kernel_size - 1`
    fn context_size(&self) -> usize {
        self.kernel_size - 1
    }

    /// Streaming forward: prepend cached input, run transposed conv, trim
    /// padding, return only the new output portion, save context.
    fn forward_streaming(
        &self,
        xs: &Tensor,
        cache: &mut StreamingConvCache,
        is_first: bool,
    ) -> Result<Tensor> {
        let slot = cache.next_slot();
        let ctx = self.context_size();
        let t_new = xs.dim(D::Minus1)?;

        // Prepend cached input (or empty on first call)
        let full_input = match &cache.slots[slot] {
            Some(cached) => Tensor::cat(&[cached, xs], D::Minus1)?,
            None => xs.clone(),
        };

        // Run transposed conv
        let full_output = self.conv.forward(&full_input)?;

        // Trim padding (causal: trim right)
        let padding_total = self.kernel_size - self.stride;
        let full_output = if self.causal && padding_total > 0 {
            let len = full_output.dim(D::Minus1)?;
            full_output.narrow(D::Minus1, 0, len - padding_total)?
        } else if !self.causal && padding_total > 0 {
            let pad_right = padding_total / 2;
            let pad_left = padding_total - pad_right;
            let len = full_output.dim(D::Minus1)?;
            full_output.narrow(D::Minus1, pad_left, len - pad_left - pad_right)?
        } else {
            full_output
        };

        // Extract only the new output
        let output = if is_first {
            // First chunk: return everything
            full_output
        } else {
            // Subsequent chunks: return last T_new * stride samples
            let expected = t_new * self.stride;
            let out_len = full_output.dim(D::Minus1)?;
            if out_len >= expected {
                full_output.narrow(D::Minus1, out_len - expected, expected)?
            } else {
                full_output
            }
        };

        // Save last context_size input samples
        if ctx > 0 {
            let total_len = full_input.dim(D::Minus1)?;
            if total_len >= ctx {
                cache.slots[slot] =
                    Some(full_input.narrow(D::Minus1, total_len - ctx, ctx)?);
            } else {
                cache.slots[slot] = Some(full_input);
            }
        }

        Ok(output)
    }
}

impl Module for CausalConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(xs)?;
        let padding_total = self.kernel_size - self.stride;
        if self.causal {
            // Trim right side for causal
            let len = y.dim(D::Minus1)?;
            let end = len - padding_total;
            y.narrow(D::Minus1, 0, end)
        } else {
            let pad_right = padding_total / 2;
            let pad_left = padding_total - pad_right;
            let len = y.dim(D::Minus1)?;
            y.narrow(D::Minus1, pad_left, len - pad_left - pad_right)
        }
    }
}

// ---------------------------------------------------------------------------
// ConvRmsNorm — RMSNorm on (B, C, T) tensors via transpose
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ConvRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ConvRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, C, T) → transpose to (B, T, C), apply RMSNorm, transpose back
        let xs = xs.transpose(1, 2)?; // (B, T, C)
        let (_b, _t, _c) = xs.dims3()?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_norm = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = xs_norm.broadcast_mul(&self.weight)?;
        out.transpose(1, 2) // (B, C, T)
    }
}

// ---------------------------------------------------------------------------
// RMSNorm without learnable weight (for FinalLayer)
// ---------------------------------------------------------------------------

fn rms_norm_no_weight(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    xs.broadcast_div(&(variance + eps)?.sqrt()?)
}

// ---------------------------------------------------------------------------
// TokenizerFfn — Linear → GELU → Linear (no bias)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TokenizerFfn {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

impl TokenizerFfn {
    fn new(dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(dim, ffn_dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(ffn_dim, dim, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, T, C)
        let xs = self.linear1.forward(xs)?;
        let xs = xs.gelu_erf()?;
        self.linear2.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// Block1d — depthwise conv + FFN with layer scale
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Block1d {
    norm: ConvRmsNorm,
    mixer: CausalConv1d,
    gamma: Tensor,
    ffn_norm: ConvRmsNorm,
    ffn: TokenizerFfn,
    ffn_gamma: Tensor,
}

impl Block1d {
    fn new(dim: usize, eps: f64, bias: bool, causal: bool, vb: VarBuilder) -> Result<Self> {
        let norm = ConvRmsNorm::new(dim, eps, vb.pp("norm"))?;
        // Depthwise conv: groups=dim, kernel_size=7
        let mixer = CausalConv1d::new(
            dim, dim, 7, 1, 1, dim, bias, causal,
            vb.pp("mixer").pp("conv").pp("conv").pp("conv"),
        )?;
        let gamma = vb.get(dim, "gamma")?;
        let ffn_norm = ConvRmsNorm::new(dim, eps, vb.pp("ffn_norm"))?;
        let ffn_dim = dim * 4;
        let ffn = TokenizerFfn::new(dim, ffn_dim, vb.pp("ffn"))?;
        let ffn_gamma = vb.get(dim, "ffn_gamma")?;
        Ok(Self { norm, mixer, gamma, ffn_norm, ffn, ffn_gamma })
    }
}

impl Block1d {
    fn forward_streaming(
        &self,
        xs: &Tensor,
        cache: &mut StreamingConvCache,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm.forward(xs)?;
        let xs = self.mixer.forward_streaming(&xs, cache)?;
        let gamma = self.gamma.reshape((1, (), 1))?;
        let xs = (xs.broadcast_mul(&gamma)? + residual)?;

        let residual = xs.clone();
        let h = self.ffn_norm.forward(&xs)?;
        let h = h.transpose(1, 2)?;
        let h = self.ffn.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let ffn_gamma = self.ffn_gamma.reshape((1, (), 1))?;
        h.broadcast_mul(&ffn_gamma)? + residual
    }
}

impl Module for Block1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, C, T)
        let residual = xs.clone();
        let xs = self.norm.forward(xs)?;
        let xs = self.mixer.forward(&xs)?;
        // gamma: (C,) → (1, C, 1) for broadcast
        let gamma = self.gamma.reshape((1, (), 1))?;
        let xs = (xs.broadcast_mul(&gamma)? + residual)?;

        let residual = xs.clone();
        let h = self.ffn_norm.forward(&xs)?;
        // (B, C, T) → (B, T, C) for FFN
        let h = h.transpose(1, 2)?;
        let h = self.ffn.forward(&h)?;
        // (B, T, C) → (B, C, T)
        let h = h.transpose(1, 2)?;
        let ffn_gamma = self.ffn_gamma.reshape((1, (), 1))?;
        h.broadcast_mul(&ffn_gamma)? + residual
    }
}

// ---------------------------------------------------------------------------
// TokenizerEncoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TokenizerEncoder {
    downsample_layers: Vec<CausalConv1d>,
    stages: Vec<Vec<Block1d>>,
    head: CausalConv1d,
}

impl TokenizerEncoder {
    fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let n_filters = cfg.encoder_n_filters;
        let depths = parse_depths(&cfg.encoder_depths);
        // Reverse ratios for encoder
        let ratios: Vec<usize> = cfg.encoder_ratios.iter().rev().copied().collect();
        let _n_stages = ratios.len() + 1;

        let mut downsample_layers = Vec::new();
        let mut stages = Vec::new();

        // Stem: Conv1d(channels, n_filters, kernel_size=7)
        let stem = CausalConv1d::new(
            cfg.channels, n_filters, 7, 1, 1, 1, cfg.conv_bias, cfg.causal,
            vb.pp("downsample_layers").pp("0").pp("0").pp("conv").pp("conv"),
        )?;
        downsample_layers.push(stem);

        // First stage (at n_filters channels)
        let mut stage_blocks = Vec::new();
        for j in 0..depths[0] {
            let block = Block1d::new(
                n_filters, cfg.layernorm_eps, cfg.conv_bias, cfg.causal,
                vb.pp("stages").pp(&0.to_string()).pp(&j.to_string()),
            )?;
            stage_blocks.push(block);
        }
        stages.push(stage_blocks);

        // Subsequent stages with downsampling
        for i in 0..ratios.len() {
            let in_c = n_filters * (1 << i);
            let out_c = n_filters * (1 << (i + 1));
            let ratio = ratios[i];
            let ds = CausalConv1d::new(
                in_c, out_c, ratio * 2, ratio, 1, 1, cfg.conv_bias, cfg.causal,
                vb.pp("downsample_layers").pp(&(i + 1).to_string()).pp("0").pp("conv").pp("conv"),
            )?;
            downsample_layers.push(ds);

            let stage_idx = i + 1;
            let dim = out_c;
            let mut stage_blocks = Vec::new();
            for j in 0..depths[stage_idx] {
                let block = Block1d::new(
                    dim, cfg.layernorm_eps, cfg.conv_bias, cfg.causal,
                    vb.pp("stages").pp(&stage_idx.to_string()).pp(&j.to_string()),
                )?;
                stage_blocks.push(block);
            }
            stages.push(stage_blocks);
        }

        // Head: Conv1d(final_channels, vae_dim, kernel_size=7)
        let final_channels = n_filters * (1 << ratios.len());
        let head = CausalConv1d::new(
            final_channels, cfg.vae_dim, 7, 1, 1, 1, cfg.conv_bias, cfg.causal,
            vb.pp("head").pp("conv").pp("conv"),
        )?;

        Ok(Self { downsample_layers, stages, head })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, 1, T)
        let mut xs = self.downsample_layers[0].forward(xs)?;
        xs = self.stages[0].iter().try_fold(xs, |x, block| block.forward(&x))?;

        for i in 1..self.downsample_layers.len() {
            xs = self.downsample_layers[i].forward(&xs)?;
            xs = self.stages[i].iter().try_fold(xs, |x, block| block.forward(&x))?;
        }

        let xs = self.head.forward(&xs)?;
        // (B, vae_dim, T') → (B, T', vae_dim)
        xs.transpose(1, 2)
    }
}

// ---------------------------------------------------------------------------
// TokenizerDecoder (for TTS acoustic tokenizer)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TokenizerDecoder {
    stem: CausalConv1d,
    upsample_layers: Vec<CausalConvTranspose1d>,
    stages: Vec<Vec<Block1d>>,
    head: CausalConv1d,
}

impl TokenizerDecoder {
    fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let n_filters = cfg.decoder_n_filters.unwrap_or(cfg.encoder_n_filters);
        let ratios = cfg.decoder_ratios.as_ref().unwrap_or(&cfg.encoder_ratios);
        let depths_str = cfg.decoder_depths.as_deref()
            .unwrap_or(&cfg.encoder_depths);
        let mut depths = parse_depths(depths_str);
        // When decoder_depths is not specified, the encoder depths are used
        // in reversed order (matching the mirrored encoder→decoder structure).
        if cfg.decoder_depths.is_none() {
            depths.reverse();
        }
        let n_ratios = ratios.len();

        // Stem: Conv1d(vae_dim, n_filters * 2^n_ratios, kernel_size=7)
        let final_channels = n_filters * (1 << n_ratios);
        let stem = CausalConv1d::new(
            cfg.vae_dim, final_channels, 7, 1, 1, 1, cfg.conv_bias, cfg.causal,
            vb.pp("upsample_layers").pp("0").pp("0").pp("conv").pp("conv"),
        )?;

        let mut upsample_layers = Vec::new();
        let mut stages = Vec::new();

        // First stage at final_channels
        let mut stage_blocks = Vec::new();
        for j in 0..depths[0] {
            let block = Block1d::new(
                final_channels, cfg.layernorm_eps, cfg.conv_bias, cfg.causal,
                vb.pp("stages").pp("0").pp(&j.to_string()),
            )?;
            stage_blocks.push(block);
        }
        stages.push(stage_blocks);

        // Upsample stages
        for i in 0..n_ratios {
            let in_c = n_filters * (1 << (n_ratios - i));
            let out_c = n_filters * (1 << (n_ratios - i - 1));
            let ratio = ratios[i];
            let us = CausalConvTranspose1d::new(
                in_c, out_c, ratio * 2, ratio, cfg.conv_bias,
                vb.pp("upsample_layers").pp(&(i + 1).to_string()).pp("0").pp("convtr").pp("convtr"),
            )?;
            upsample_layers.push(us);

            let stage_idx = i + 1;
            let dim = out_c;
            let mut stage_blocks = Vec::new();
            for j in 0..depths[stage_idx] {
                let block = Block1d::new(
                    dim, cfg.layernorm_eps, cfg.conv_bias, cfg.causal,
                    vb.pp("stages").pp(&stage_idx.to_string()).pp(&j.to_string()),
                )?;
                stage_blocks.push(block);
            }
            stages.push(stage_blocks);
        }

        // Head: Conv1d(n_filters, channels, kernel_size=7)
        let head = CausalConv1d::new(
            n_filters, cfg.channels, 7, 1, 1, 1, cfg.conv_bias, cfg.causal,
            vb.pp("head").pp("conv").pp("conv"),
        )?;

        Ok(Self { stem, upsample_layers, stages, head })
    }

    /// Count the total number of conv layers that need cache slots.
    fn num_cache_layers(&self) -> usize {
        // stem (1) + stages[0] blocks (each has 1 mixer conv) +
        // for each upsample: 1 convtr + stages[i+1] blocks
        // + head (1)
        let mut n = 1; // stem
        n += self.stages[0].len(); // stage 0 blocks
        for i in 0..self.upsample_layers.len() {
            n += 1; // upsample convtr
            n += self.stages[i + 1].len(); // stage blocks
        }
        n += 1; // head
        n
    }

    fn forward_streaming(
        &self,
        xs: &Tensor,
        cache: &mut StreamingConvCache,
        is_first: bool,
    ) -> Result<Tensor> {
        // xs: (B, T', vae_dim) → (B, vae_dim, T')
        let mut xs = xs.transpose(1, 2)?;

        xs = self.stem.forward_streaming(&xs, cache)?;
        for block in &self.stages[0] {
            xs = block.forward_streaming(&xs, cache)?;
        }

        for i in 0..self.upsample_layers.len() {
            xs = self.upsample_layers[i].forward_streaming(&xs, cache, is_first)?;
            for block in &self.stages[i + 1] {
                xs = block.forward_streaming(&xs, cache)?;
            }
        }

        self.head.forward_streaming(&xs, cache)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, T', vae_dim) → (B, vae_dim, T')
        let mut xs = xs.transpose(1, 2)?;

        xs = self.stem.forward(&xs)?;
        xs = self.stages[0].iter().try_fold(xs, |x, block| block.forward(&x))?;

        for i in 0..self.upsample_layers.len() {
            xs = self.upsample_layers[i].forward(&xs)?;
            xs = self.stages[i + 1].iter().try_fold(xs, |x, block| block.forward(&x))?;
        }

        self.head.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// AcousticTokenizer & SemanticTokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AcousticTokenizer {
    encoder: Option<TokenizerEncoder>,
    decoder: TokenizerDecoder,
    #[allow(dead_code)]
    vae_dim: usize,
    #[allow(dead_code)]
    fix_std: f64,
}

impl AcousticTokenizer {
    pub fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        // Encoder is optional (streaming/TTS-only models don't have one)
        let encoder = TokenizerEncoder::new(cfg, vb.pp("encoder")).ok();
        let decoder = TokenizerDecoder::new(cfg, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            vae_dim: cfg.vae_dim,
            fix_std: cfg.fix_std,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, 1, T_audio) → (B, T', vae_dim)
        match &self.encoder {
            Some(enc) => enc.forward(xs),
            None => candle::bail!("acoustic tokenizer encoder not available (TTS-only model)"),
        }
    }

    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, T', vae_dim) → (B, 1, T_audio)
        let xs = xs.to_dtype(DType::F32)?;
        self.decoder.forward(&xs)
    }

    /// Create a new streaming cache sized for this decoder.
    pub fn new_streaming_cache(&self) -> StreamingConvCache {
        StreamingConvCache::new(self.decoder.num_cache_layers())
    }

    /// Streaming decode: decode one frame using cached conv state.
    pub fn decode_streaming(
        &self,
        xs: &Tensor,
        cache: &mut StreamingConvCache,
        is_first: bool,
    ) -> Result<Tensor> {
        cache.reset_cursor();
        let xs = xs.to_dtype(DType::F32)?;
        self.decoder.forward_streaming(&xs, cache, is_first)
    }
}

#[derive(Debug, Clone)]
pub struct SemanticTokenizer {
    encoder: TokenizerEncoder,
    #[allow(dead_code)]
    vae_dim: usize,
}

impl SemanticTokenizer {
    pub fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = TokenizerEncoder::new(cfg, vb.pp("encoder"))?;
        Ok(Self {
            encoder,
            vae_dim: cfg.vae_dim,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        self.encoder.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// ASR-specific causal conv (matches HF VibeVoiceAsrCausalConv1d)
// ---------------------------------------------------------------------------

/// Causal conv1d matching the HF transformers implementation:
/// left_pad = (kernel_size - 1) * dilation - (stride - 1), NO extra right padding.
#[derive(Debug, Clone)]
struct AsrCausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl AsrCausalConv1d {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(in_c, out_c, kernel_size, cfg, vb)?;
        let left_pad = (kernel_size - 1) * dilation - (stride - 1);
        Ok(Self { conv, left_pad })
    }
}

impl Module for AsrCausalConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.pad_with_zeros(D::Minus1, self.left_pad, 0)?;
        self.conv.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// ASR-specific encoder (different weight paths from TTS encoder)
// ---------------------------------------------------------------------------

/// ConvNeXt-like block for ASR encoder (matches HF VibeVoiceAsrConvNext1dLayer).
#[derive(Debug, Clone)]
struct AsrBlock1d {
    norm: ConvRmsNorm,
    mixer: AsrCausalConv1d,
    gamma: Tensor,
    ffn_norm: ConvRmsNorm,
    ffn: TokenizerFfn,
    ffn_gamma: Tensor,
}

impl AsrBlock1d {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm = ConvRmsNorm::new(dim, eps, vb.pp("norm"))?;
        let mixer = AsrCausalConv1d::new(
            dim, dim, 7, 1, 1, dim,
            vb.pp("mixer").pp("conv"),
        )?;
        let gamma = vb.get(dim, "gamma")?;
        let ffn_norm = ConvRmsNorm::new(dim, eps, vb.pp("ffn_norm"))?;
        let ffn_dim = dim * 4;
        let ffn = TokenizerFfn::new(dim, ffn_dim, vb.pp("ffn"))?;
        let ffn_gamma = vb.get(dim, "ffn_gamma")?;
        Ok(Self { norm, mixer, gamma, ffn_norm, ffn, ffn_gamma })
    }
}

impl Module for AsrBlock1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm.forward(xs)?;
        let xs = self.mixer.forward(&xs)?;
        let gamma = self.gamma.reshape((1, (), 1))?;
        let xs = (xs.broadcast_mul(&gamma)? + residual)?;

        let residual = xs.clone();
        let h = self.ffn_norm.forward(&xs)?;
        let h = h.transpose(1, 2)?;
        let h = self.ffn.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let ffn_gamma = self.ffn_gamma.reshape((1, (), 1))?;
        h.broadcast_mul(&ffn_gamma)? + residual
    }
}

/// ASR encoder: stem{conv + stage} → conv_layers[i]{conv + stage} → head
#[derive(Debug, Clone)]
struct AsrTokenizerEncoder {
    stem_conv: AsrCausalConv1d,
    stem_stage: Vec<AsrBlock1d>,
    conv_layers: Vec<AsrCausalConv1d>,
    stages: Vec<Vec<AsrBlock1d>>,
    head: AsrCausalConv1d,
}

impl AsrTokenizerEncoder {
    fn new(cfg: &AsrEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let n_filters = cfg.num_filters;
        let ratios = &cfg.downsampling_ratios;
        let depths = &cfg.depths;

        // Stem conv at stem.conv.conv (stride=1)
        let stem_conv = AsrCausalConv1d::new(
            cfg.channels, n_filters, cfg.kernel_size, 1, 1, 1,
            vb.pp("stem").pp("conv").pp("conv"),
        )?;

        // Stem stage blocks at stem.stage.{j}
        let mut stem_stage = Vec::new();
        for j in 0..depths[0] {
            let block = AsrBlock1d::new(
                n_filters, cfg.rms_norm_eps,
                vb.pp("stem").pp("stage").pp(&j.to_string()),
            )?;
            stem_stage.push(block);
        }

        let mut conv_layers = Vec::new();
        let mut stages = Vec::new();

        for i in 0..ratios.len() {
            let in_c = n_filters * (1 << i);
            let out_c = n_filters * (1 << (i + 1));
            let ratio = ratios[i];

            // Downsample conv at conv_layers.{i}.conv.conv
            let ds = AsrCausalConv1d::new(
                in_c, out_c, ratio * 2, ratio, 1, 1,
                vb.pp("conv_layers").pp(&i.to_string()).pp("conv").pp("conv"),
            )?;
            conv_layers.push(ds);

            // Stage blocks at conv_layers.{i}.stage.{j}
            let n_blocks = depths[i + 1];
            let mut stage_blocks = Vec::new();
            for j in 0..n_blocks {
                let block = AsrBlock1d::new(
                    out_c, cfg.rms_norm_eps,
                    vb.pp("conv_layers").pp(&i.to_string()).pp("stage").pp(&j.to_string()),
                )?;
                stage_blocks.push(block);
            }
            stages.push(stage_blocks);
        }

        // Head conv at head.conv (stride=1)
        let final_channels = n_filters * (1 << ratios.len());
        let head = AsrCausalConv1d::new(
            final_channels, cfg.hidden_size, cfg.kernel_size, 1, 1, 1,
            vb.pp("head").pp("conv"),
        )?;

        Ok(Self { stem_conv, stem_stage, conv_layers, stages, head })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, 1, T)
        let mut xs = self.stem_conv.forward(xs)?;
        xs = self.stem_stage.iter().try_fold(xs, |x, block| block.forward(&x))?;

        for i in 0..self.conv_layers.len() {
            xs = self.conv_layers[i].forward(&xs)?;
            xs = self.stages[i].iter().try_fold(xs, |x, block| block.forward(&x))?;
        }

        let xs = self.head.forward(&xs)?;
        // (B, hidden_size, T') → (B, T', hidden_size)
        xs.transpose(1, 2)
    }
}

/// ASR multi-modal projector: acoustic + semantic branches, each Linear→RMSNorm→Linear
#[derive(Debug, Clone)]
struct AsrMultiModalProjector {
    acoustic_fc1: Linear,
    acoustic_norm: RmsNorm,
    acoustic_fc2: Linear,
    semantic_fc1: Linear,
    semantic_norm: RmsNorm,
    semantic_fc2: Linear,
}

impl AsrMultiModalProjector {
    fn new(
        acoustic_dim: usize,
        semantic_dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let acoustic_fc1 = linear(acoustic_dim, output_dim, vb.pp("acoustic_linear_1"))?;
        let acoustic_norm = RmsNorm::new(output_dim, 1e-6, vb.pp("acoustic_norm"))?;
        let acoustic_fc2 = linear(output_dim, output_dim, vb.pp("acoustic_linear_2"))?;
        let semantic_fc1 = linear(semantic_dim, output_dim, vb.pp("semantic_linear_1"))?;
        let semantic_norm = RmsNorm::new(output_dim, 1e-6, vb.pp("semantic_norm"))?;
        let semantic_fc2 = linear(output_dim, output_dim, vb.pp("semantic_linear_2"))?;
        Ok(Self {
            acoustic_fc1, acoustic_norm, acoustic_fc2,
            semantic_fc1, semantic_norm, semantic_fc2,
        })
    }

    fn forward_acoustic(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.acoustic_fc1.forward(xs)?;
        let xs = self.acoustic_norm.forward(&xs)?;
        self.acoustic_fc2.forward(&xs)
    }

    fn forward_semantic(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.semantic_fc1.forward(xs)?;
        let xs = self.semantic_norm.forward(&xs)?;
        self.semantic_fc2.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// SpeechConnector — Linear → RMSNorm → Linear
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(input_dim, output_dim, vb.pp("fc1"))?;
        let norm = RmsNorm::new(output_dim, 1e-6, vb.pp("norm"))?;
        let fc2 = linear(output_dim, output_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }
}

impl Module for SpeechConnector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.to_dtype(self.fc1.weight().dtype())?;
        let xs = self.fc1.forward(&xs)?;
        let xs = self.norm.forward(&xs)?;
        self.fc2.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// Diffusion Head components
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn new(hidden_size: usize, freq_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear_no_bias(freq_dim, hidden_size, vb.pp("mlp").pp("0"))?;
        let linear2 = linear_no_bias(hidden_size, hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self { linear1, linear2, frequency_embedding_size: freq_dim })
    }

    fn forward(&self, timesteps: &Tensor, device: &Device) -> Result<Tensor> {
        let half = self.frequency_embedding_size / 2;
        let freqs: Vec<f64> = (0..half)
            .map(|i| (-f64::ln(10000.0) * i as f64 / half as f64).exp())
            .collect();
        let freqs = Tensor::new(freqs, device)?;
        let t = timesteps.to_dtype(DType::F64)?;
        let args = t.unsqueeze(D::Minus1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let args = args.to_dtype(timesteps.dtype())?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        let emb = self.linear1.forward(&emb)?;
        let emb = candle_nn::Activation::Silu.forward(&emb)?;
        self.linear2.forward(&emb)
    }
}

#[derive(Debug, Clone)]
struct DiffusionFfn {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DiffusionFfn {
    fn new(embed_dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(embed_dim, ffn_dim, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(embed_dim, ffn_dim, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(ffn_dim, embed_dim, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

impl Module for DiffusionFfn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

fn modulate(xs: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let scaled = xs.broadcast_mul(&(scale + 1.0)?)?;
    scaled.broadcast_add(shift)
}

#[derive(Debug, Clone)]
struct HeadLayer {
    ffn: DiffusionFfn,
    norm: RmsNorm,
    ada_ln_linear: Linear,
}

impl HeadLayer {
    fn new(embed_dim: usize, ffn_dim: usize, cond_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let ffn = DiffusionFfn::new(embed_dim, ffn_dim, vb.pp("ffn"))?;
        let norm = RmsNorm::new(embed_dim, eps, vb.pp("norm"))?;
        let ada_ln_linear = linear_no_bias(cond_dim, 3 * embed_dim, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self { ffn, norm, ada_ln_linear })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = candle_nn::Activation::Silu.forward(cond)?;
        let modulation = self.ada_ln_linear.forward(&modulation)?;
        let chunks = modulation.chunk(3, D::Minus1)?;
        let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);
        let h = modulate(&self.norm.forward(xs)?, shift, scale)?;
        let h = self.ffn.forward(&h)?;
        xs + gate.broadcast_mul(&h)?
    }
}

#[derive(Debug, Clone)]
struct FinalLayer {
    eps: f64,
    linear: Linear,
    ada_ln_linear: Linear,
}

impl FinalLayer {
    fn new(hidden_size: usize, output_size: usize, cond_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let linear = linear_no_bias(hidden_size, output_size, vb.pp("linear"))?;
        let ada_ln_linear = linear_no_bias(cond_size, 2 * hidden_size, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self { eps, linear, ada_ln_linear })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = candle_nn::Activation::Silu.forward(cond)?;
        let modulation = self.ada_ln_linear.forward(&modulation)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = rms_norm_no_weight(xs, self.eps)?;
        let xs = modulate(&xs, shift, scale)?;
        self.linear.forward(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionHead {
    noisy_images_proj: Linear,
    cond_proj: Linear,
    t_embedder: TimestepEmbedder,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
}

impl DiffusionHead {
    pub fn new(cfg: &DiffusionHeadConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let ffn_dim = (h as f64 * cfg.head_ffn_ratio) as usize;
        let noisy_images_proj = linear_no_bias(cfg.latent_size, h, vb.pp("noisy_images_proj"))?;
        let cond_proj = linear_no_bias(h, h, vb.pp("cond_proj"))?;
        let t_embedder = TimestepEmbedder::new(h, 256, vb.pp("t_embedder"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.head_layers {
            let layer = HeadLayer::new(h, ffn_dim, h, cfg.rms_norm_eps, vb.pp("layers").pp(&i.to_string()))?;
            layers.push(layer);
        }
        let final_layer = FinalLayer::new(h, cfg.latent_size, h, cfg.rms_norm_eps, vb.pp("final_layer"))?;
        Ok(Self { noisy_images_proj, cond_proj, t_embedder, layers, final_layer })
    }

    pub fn forward(&self, noisy_images: &Tensor, timesteps: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let device = noisy_images.device();
        let x = self.noisy_images_proj.forward(noisy_images)?;
        let t = self.t_embedder.forward(timesteps, device)?;
        let c = self.cond_proj.forward(condition)?.broadcast_add(&t)?;
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &c)?;
        }
        self.final_layer.forward(&x, &c)
    }
}

// ---------------------------------------------------------------------------
// DPM-Solver Multistep Scheduler
// ---------------------------------------------------------------------------

/// DPM-Solver++ Multistep Scheduler (order 2, midpoint, `lower_order_final`).
///
/// Matches the `DPMSolverMultistepScheduler` from HuggingFace diffusers with:
///   `algorithm_type = "dpmsolver++"`, `solver_type = "midpoint"`,
///   `solver_order = 2`, `lower_order_final = True`,
///   `timestep_spacing = "linspace"`, `final_sigmas_type = "zero"`,
///   `prediction_type = "v_prediction"`, cosine beta schedule.
#[derive(Debug, Clone)]
pub struct DpmSolverScheduler {
    alphas_cumprod: Vec<f64>,
    /// DPM-solver sigmas interpolated at the timestep positions, plus a final
    /// entry of 0.0 (`final_sigmas_type = "zero"`).  Length = N + 1.
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    /// History of x0 predictions (length = solver_order = 2).
    model_outputs: Vec<Option<Tensor>>,
    step_index: usize,
    lower_order_nums: usize,
    num_inference_steps: usize,
    num_train_timesteps: usize,
    prediction_type: String,
}

impl DpmSolverScheduler {
    pub fn new(num_train_timesteps: usize, prediction_type: &str) -> Self {
        // Cosine beta schedule – matches Python's betas_for_alpha_bar("cosine")
        // followed by alphas = 1 - betas, alphas_cumprod = cumprod(alphas).
        let alpha_bar_fn = |t: f64| -> f64 {
            ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).cos().powi(2)
        };
        let max_beta: f64 = 0.999;
        let n = num_train_timesteps as f64;
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod: f64 = 1.0;
        for i in 0..num_train_timesteps {
            let t1 = i as f64 / n;
            let t2 = (i + 1) as f64 / n;
            let beta = (1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1)).min(max_beta);
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }
        Self {
            alphas_cumprod,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            model_outputs: vec![None, None],
            step_index: 0,
            lower_order_nums: 0,
            num_inference_steps: 0,
            num_train_timesteps,
            prediction_type: prediction_type.to_string(),
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        // Match Python's DPMSolverMultistepScheduler with timestep_spacing="linspace":
        //   np.linspace(0, last_timestep - 1, num_inference_steps + 1)
        //       .round()[::-1][:-1]
        let last = (self.num_train_timesteps - 1) as f64;
        let n = num_inference_steps as f64;
        let mut ts: Vec<usize> = (0..=num_inference_steps)
            .map(|i| (i as f64 * last / n).round() as usize)
            .collect();
        ts.reverse();
        ts.pop(); // drop the trailing 0 (was the first linspace point)
        self.timesteps = ts;

        // Compute sigmas = sqrt((1 - alpha_bar) / alpha_bar) for all train timesteps
        let all_sigmas: Vec<f64> = self
            .alphas_cumprod
            .iter()
            .map(|&a| ((1.0 - a) / a).sqrt())
            .collect();
        // Interpolate at the timestep positions (they are integers, so just index)
        let mut interp: Vec<f64> = self.timesteps.iter().map(|&t| all_sigmas[t]).collect();
        // final_sigmas_type = "zero"
        interp.push(0.0);
        self.sigmas = interp;

        // Reset state
        self.model_outputs = vec![None, None];
        self.step_index = 0;
        self.lower_order_nums = 0;
    }

    /// Convert a DPM-solver sigma to the diffusion (alpha_t, sigma_t) pair.
    fn sigma_to_alpha_sigma(&self, sigma: f64) -> (f64, f64) {
        let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
        let sigma_t = sigma * alpha_t;
        (alpha_t, sigma_t)
    }

    /// v_prediction → x0 conversion for DPM-Solver++.
    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let (alpha_t, sigma_t) = self.sigma_to_alpha_sigma(sigma);
        match self.prediction_type.as_str() {
            "v_prediction" => {
                // x0_pred = alpha_t * sample - sigma_t * model_output
                (sample * alpha_t)? - (model_output * sigma_t)?
            }
            "epsilon" => {
                // x0_pred = (sample - sigma_t * model_output) / alpha_t
                (sample - (model_output * sigma_t)?)? / alpha_t
            }
            "sample" => Ok(model_output.clone()),
            pt => candle::bail!("Unsupported prediction_type: {}", pt),
        }
    }

    /// First-order DPM-Solver++ update (equivalent to DDIM).
    fn first_order_update(&self, x0_pred: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma_s = self.sigmas[self.step_index];
        let sigma_t = self.sigmas[self.step_index + 1];
        let (alpha_t, sig_t) = self.sigma_to_alpha_sigma(sigma_t);
        let (_alpha_s, sig_s) = self.sigma_to_alpha_sigma(sigma_s);
        let lambda_t = (alpha_t).ln() - sig_t.ln();
        let lambda_s = self.sigma_to_alpha_sigma(sigma_s).0.ln() - sig_s.ln();
        let h = lambda_t - lambda_s;
        // x_t = (sigma_t / sigma_s) * sample - alpha_t * (exp(-h) - 1) * x0_pred
        let ratio = sig_t / sig_s;
        let coeff = alpha_t * ((-h).exp() - 1.0);
        (sample * ratio)? - (x0_pred * coeff)?
    }

    /// Second-order DPM-Solver++ midpoint update.
    fn second_order_update(
        &self,
        x0_cur: &Tensor,
        x0_prev: &Tensor,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let sigma_s0 = self.sigmas[self.step_index];
        let sigma_s1 = self.sigmas[self.step_index - 1];
        let sigma_t = self.sigmas[self.step_index + 1];
        let (alpha_t, sig_t) = self.sigma_to_alpha_sigma(sigma_t);
        let (alpha_s0, sig_s0) = self.sigma_to_alpha_sigma(sigma_s0);
        let (_alpha_s1, sig_s1) = self.sigma_to_alpha_sigma(sigma_s1);
        let lambda_t = alpha_t.ln() - sig_t.ln();
        let lambda_s0 = alpha_s0.ln() - sig_s0.ln();
        let lambda_s1 = _alpha_s1.ln() - sig_s1.ln();
        let h = lambda_t - lambda_s0;
        let h_0 = lambda_s0 - lambda_s1;
        let r0 = h_0 / h;
        // D0 = m0, D1 = (1/r0) * (m0 - m1)
        let d1 = ((x0_cur - x0_prev)? * (1.0 / r0))?;
        let ratio = sig_t / sig_s0;
        let coeff = alpha_t * ((-h).exp() - 1.0);
        // x_t = ratio * sample - coeff * D0 - 0.5 * coeff * D1   (midpoint)
        ((sample * ratio)? - (x0_cur * coeff)?)? - (&d1 * (0.5 * coeff))?
    }

    /// Perform one denoising step. Matches `scheduler.step(eps, t, speech)`.
    /// `model_output` is the raw (CFG-combined) model output (v-prediction).
    /// Returns the denoised sample.
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let n = self.timesteps.len();
        // lower_order_final: use first-order on the last step when
        // final_sigmas_type == "zero" (always true for this scheduler).
        let lower_order_final = self.step_index == n - 1;

        // Convert raw model output to x0_pred
        let x0_pred = self.convert_model_output(model_output, sample)?;

        // Shift history
        self.model_outputs[0] = self.model_outputs[1].take();
        self.model_outputs[1] = Some(x0_pred.clone());

        // Upcast sample to f64 for numerical stability (matches Python)
        let sample_f64 = sample.to_dtype(DType::F64)?;
        let x0_f64 = x0_pred.to_dtype(DType::F64)?;

        let prev_sample = if self.lower_order_nums < 1 || lower_order_final {
            // First-order (DDIM)
            self.first_order_update(&x0_f64, &sample_f64)?
        } else {
            // Second-order midpoint
            let x0_prev = self.model_outputs[0]
                .as_ref()
                .expect("second-order needs prior x0_pred")
                .to_dtype(DType::F64)?;
            self.second_order_update(&x0_f64, &x0_prev, &sample_f64)?
        };

        if self.lower_order_nums < 2 {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;

        // Cast back to original dtype
        prev_sample.to_dtype(model_output.dtype())
    }

    pub fn alpha_t(&self, t: usize) -> f64 {
        self.alphas_cumprod[t].sqrt()
    }

    pub fn sigma_t(&self, t: usize) -> f64 {
        (1.0 - self.alphas_cumprod[t]).sqrt()
    }

    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor> {
        let alpha = self.alpha_t(timestep);
        let sigma = self.sigma_t(timestep);
        (original * alpha)? + (noise * sigma)?
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }
}

// ---------------------------------------------------------------------------
// Top-level VibeVoice models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VibeVoiceModel {
    language_model: qwen2::Model,
    acoustic_tokenizer: AcousticTokenizer,
    semantic_tokenizer: SemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    speech_scaling_factor: Option<f64>,
    speech_bias_factor: Option<f64>,
    prediction_head: Option<DiffusionHead>,
}

impl VibeVoiceModel {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // qwen2::Model::new internally does vb.pp("model") to create weight
        // paths like "model.embed_tokens.weight". In VibeVoice the LLM
        // weights live under "<prefix>.language_model.*" instead. We use
        // root() to clear the path (avoiding double-prefix in Rename's inner
        // VarBuilder), then rename_f to rewrite "model.*" → "<prefix>.language_model.*".
        let cur_prefix = vb.prefix();
        let lm_vb = vb.clone().root().rename_f(move |s| {
            let mapped = if let Some(rest) = s.strip_prefix("model.") {
                format!("language_model.{rest}")
            } else {
                s.to_string()
            };
            if cur_prefix.is_empty() {
                mapped
            } else {
                format!("{cur_prefix}.{mapped}")
            }
        });
        let language_model = qwen2::Model::new(&cfg.decoder_config, lm_vb)?;
        let acoustic_tokenizer = AcousticTokenizer::new(
            &cfg.acoustic_tokenizer_config, vb.pp("acoustic_tokenizer"),
        )?;
        let semantic_tokenizer = SemanticTokenizer::new(
            &cfg.semantic_tokenizer_config, vb.pp("semantic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::new(
            cfg.acoustic_tokenizer_config.vae_dim,
            cfg.decoder_config.hidden_size,
            vb.pp("acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::new(
            cfg.semantic_tokenizer_config.vae_dim,
            cfg.decoder_config.hidden_size,
            vb.pp("semantic_connector"),
        )?;
        let prediction_head = if let Some(ref dhcfg) = cfg.diffusion_head_config {
            Some(DiffusionHead::new(dhcfg, vb.pp("prediction_head"))?)
        } else {
            None
        };

        // Try to load scaling factors from buffers (stored as BF16 0-d scalars)
        let speech_scaling_factor = vb.get((), "speech_scaling_factor").ok()
            .and_then(|t| t.to_dtype(DType::F64).ok())
            .and_then(|t| t.to_scalar::<f64>().ok());
        let speech_bias_factor = vb.get((), "speech_bias_factor").ok()
            .and_then(|t| t.to_dtype(DType::F64).ok())
            .and_then(|t| t.to_scalar::<f64>().ok());

        Ok(Self {
            language_model,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            speech_scaling_factor,
            speech_bias_factor,
            prediction_head,
        })
    }

    fn encode_speech(&self, audio: &Tensor) -> Result<Tensor> {
        // audio: (B, 1, T_samples)
        let acoustic_latents = self.acoustic_tokenizer.encode(audio)?;
        let semantic_latents = self.semantic_tokenizer.encode(audio)?;

        // Apply scaling if available
        let acoustic_latents = if let (Some(bias), Some(scale)) =
            (self.speech_bias_factor, self.speech_scaling_factor)
        {
            ((acoustic_latents + bias)? * scale)?
        } else {
            acoustic_latents
        };

        let acoustic_features = self.acoustic_connector.forward(&acoustic_latents)?;
        let semantic_features = self.semantic_connector.forward(&semantic_latents)?;
        acoustic_features + semantic_features
    }

    fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}

/// ASR-only model (microsoft/VibeVoice-ASR-HF): encodes speech → feeds to LLM → generates text
pub struct VibeVoiceASR {
    language_model: qwen2::Model,
    acoustic_encoder: AsrTokenizerEncoder,
    semantic_encoder: AsrTokenizerEncoder,
    projector: AsrMultiModalProjector,
    lm_head: Linear,
    #[allow(dead_code)]
    cfg: AsrConfig,
}

impl VibeVoiceASR {
    pub fn new(cfg: &AsrConfig, vb: VarBuilder, audio_device: &Device) -> Result<Self> {
        // Language model: qwen2::Model expects "model.*" paths internally,
        // and ASR weights are at "language_model.model.*" — so just prepend "language_model."
        let lm_vb = vb.clone().root().rename_f(|s| {
            format!("language_model.{s}")
        });
        let language_model = qwen2::Model::new(&cfg.text_config, lm_vb)?;

        // Audio encoders: place on the designated audio_device in F32 for precision
        let acoustic_encoder = AsrTokenizerEncoder::new(
            &cfg.acoustic_tokenizer_encoder_config,
            vb.pp("acoustic_tokenizer_encoder")
                .set_device(audio_device.clone())
                .to_dtype(DType::F32),
        )?;
        let semantic_encoder = AsrTokenizerEncoder::new(
            &cfg.semantic_tokenizer_encoder_config,
            vb.pp("semantic_tokenizer_encoder")
                .set_device(audio_device.clone())
                .to_dtype(DType::F32),
        )?;

        // Projector: keep on the main device (same as language_model) for efficient mixing
        let projector = AsrMultiModalProjector::new(
            cfg.acoustic_tokenizer_encoder_config.hidden_size,
            cfg.semantic_tokenizer_encoder_config.hidden_size,
            cfg.text_config.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;

        // lm_head at language_model.lm_head
        let lm_head = if vb.contains_tensor("language_model.lm_head.weight") {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("language_model").pp("lm_head"),
            )?
        } else {
            let embed_weight = vb.pp("language_model").pp("model").pp("embed_tokens").get(
                (cfg.text_config.vocab_size, cfg.text_config.hidden_size),
                "weight",
            )?;
            Linear::from_weights(embed_weight, None)
        };

        Ok(Self { language_model, acoustic_encoder, semantic_encoder, projector, lm_head, cfg: cfg.clone() })
    }

    /// Encode speech audio into continuous features for injection into the LLM.
    pub fn encode_speech(&self, audio: &Tensor) -> Result<Tensor> {
        // audio: (B, 1, T_samples)
        let acoustic_latents = self.acoustic_encoder.forward(audio)?
            .to_device(self.language_model.device())?;
        let semantic_latents = self.semantic_encoder.forward(audio)?
            .to_device(self.language_model.device())?;
        let acoustic_features = self.projector.forward_acoustic(&acoustic_latents)?;
        let semantic_features = self.projector.forward_semantic(&semantic_latents)?;
        acoustic_features + semantic_features
    }

    /// Embed token ids using the language model's embedding table.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.language_model.embed_tokens().forward(input_ids)
    }

    /// Forward pass from pre-computed embeddings (first pass with speech injection).
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let hidden_states = self.language_model.forward_embeds(inputs_embeds, seqlen_offset, None)?;
        self.lm_head.forward(&hidden_states)
    }

    /// Standard forward from token ids (autoregressive steps).
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let hidden_states = self.language_model.forward(input_ids, seqlen_offset, None)?;
        self.lm_head.forward(&hidden_states)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}

/// Full TTS model: speech encoding + diffusion-based speech generation
pub struct VibeVoiceForConditionalGeneration {
    model: VibeVoiceModel,
    lm_head: Linear,
    #[allow(dead_code)]
    cfg: Config,
}

impl VibeVoiceForConditionalGeneration {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let model = VibeVoiceModel::new(cfg, vb.pp("model"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(
                cfg.decoder_config.hidden_size,
                cfg.decoder_config.vocab_size,
                vb.pp("lm_head"),
            )?
        } else {
            // tie_word_embeddings: reuse language_model.embed_tokens
            let embed_weight = vb.pp("model").pp("language_model").pp("embed_tokens").get(
                (cfg.decoder_config.vocab_size, cfg.decoder_config.hidden_size),
                "weight",
            )?;
            Linear::from_weights(embed_weight, None)
        };
        Ok(Self { model, lm_head, cfg: cfg.clone() })
    }

    pub fn encode_speech(&self, audio: &Tensor) -> Result<Tensor> {
        self.model.encode_speech(audio)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let hidden_states = self.model.language_model.forward(input_ids, seqlen_offset, None)?;
        self.lm_head.forward(&hidden_states)
    }

    /// Returns raw LLM hidden states (before lm_head projection).
    /// Use this for diffusion conditioning in TTS mode.
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.model.language_model.forward(input_ids, seqlen_offset, None)
    }

    pub fn diffusion_head(&self) -> Option<&DiffusionHead> {
        self.model.prediction_head.as_ref()
    }

    /// Decode diffusion latents to audio waveform.
    /// Applies inverse scaling (un-normalization) before passing to the acoustic decoder,
    /// matching the Python: `scaled = latent / scaling_factor - bias_factor`.
    pub fn decode_speech(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = if let (Some(bias), Some(scale)) =
            (self.model.speech_bias_factor, self.model.speech_scaling_factor)
        {
            ((latents / scale)? - bias)?
        } else {
            latents.clone()
        };
        self.model.acoustic_tokenizer.decode(&latents)
    }

    /// Returns (speech_bias_factor, speech_scaling_factor) if available.
    pub fn speech_scaling(&self) -> (Option<f64>, Option<f64>) {
        (self.model.speech_bias_factor, self.model.speech_scaling_factor)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// ===========================================================================
// Streaming (Realtime) TTS model — VibeVoice-Realtime-0.5B
// ===========================================================================

/// Config for the streaming / realtime model.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StreamingConfig {
    pub acoustic_vae_dim: usize,
    pub acoustic_tokenizer_config: TokenizerConfig,
    #[serde(deserialize_with = "deserialize_decoder_config")]
    pub decoder_config: qwen2::Config,
    pub diffusion_head_config: DiffusionHeadConfig,
    #[serde(default = "default_tts_backbone_layers")]
    pub tts_backbone_num_hidden_layers: usize,
}

fn default_tts_backbone_layers() -> usize { 20 }

/// Binary classifier: fc1(hidden→hidden, ReLU) → fc2(hidden→1).
#[derive(Debug, Clone)]
struct BinaryClassifier {
    fc1: Linear,
    fc2: Linear,
}

impl BinaryClassifier {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, 1, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = candle_nn::Activation::Relu.forward(&self.fc1.forward(xs)?)?;
        self.fc2.forward(&xs)
    }
}

/// Inner streaming model holding both LMs and speech components.
#[derive(Debug, Clone)]
struct VibeVoiceStreamingModel {
    /// Lower Qwen2 layers (text encoding). Its final norm is Identity (unused).
    language_model: qwen2::Model,
    /// Upper Qwen2 layers (TTS). Has real final norm.
    tts_language_model: qwen2::Model,
    /// Embedding(2, hidden_size): 0 = speech token, 1 = text token.
    tts_input_types: candle_nn::Embedding,
    acoustic_tokenizer: AcousticTokenizer,
    acoustic_connector: SpeechConnector,
    prediction_head: DiffusionHead,
    speech_scaling_factor: Option<f64>,
    speech_bias_factor: Option<f64>,
}

impl VibeVoiceStreamingModel {
    fn new(cfg: &StreamingConfig, vb: VarBuilder, audio_device: &Device) -> Result<Self> {
        let full_layers = cfg.decoder_config.num_hidden_layers;
        let lm_layers = full_layers - cfg.tts_backbone_num_hidden_layers;

        // Build lower LM config (fewer layers)
        let mut lm_cfg = cfg.decoder_config.clone();
        lm_cfg.num_hidden_layers = lm_layers;

        // Build the lower language_model with prefix remapping.
        // The lower LM has no final norm in the checkpoint (Identity in Python),
        // so we redirect norm.weight to tts_language_model's norm.weight (same shape).
        let cur_prefix = vb.prefix();
        let lm_vb = vb.clone().root().rename_f(move |s| {
            let mapped = if let Some(rest) = s.strip_prefix("model.") {
                // Redirect missing norm to TTS LM's norm
                if rest == "norm.weight" {
                    "tts_language_model.norm.weight".to_string()
                } else {
                    format!("language_model.{rest}")
                }
            } else {
                s.to_string()
            };
            if cur_prefix.is_empty() {
                mapped
            } else {
                format!("{cur_prefix}.{mapped}")
            }
        });
        let mut language_model = qwen2::Model::new(&lm_cfg, lm_vb)?;
        // The Python reference replaces the lower LM's final norm with
        // nn.Identity(); replicate that by skipping the RmsNorm.
        language_model.set_skip_final_norm(true);

        // Build upper TTS LM config
        let mut tts_lm_cfg = cfg.decoder_config.clone();
        tts_lm_cfg.num_hidden_layers = cfg.tts_backbone_num_hidden_layers;

        let cur_prefix2 = vb.prefix();
        let tts_lm_vb = vb.clone().root().rename_f(move |s| {
            let mapped = if let Some(rest) = s.strip_prefix("model.") {
                format!("tts_language_model.{rest}")
            } else {
                s.to_string()
            };
            if cur_prefix2.is_empty() {
                mapped
            } else {
                format!("{cur_prefix2}.{mapped}")
            }
        });
        let tts_language_model = qwen2::Model::new(&tts_lm_cfg, tts_lm_vb)?;

        let tts_input_types = candle_nn::embedding(
            2, cfg.decoder_config.hidden_size, vb.pp("tts_input_types"),
        )?;

        let acoustic_tokenizer = AcousticTokenizer::new(
            &cfg.acoustic_tokenizer_config, vb.pp("acoustic_tokenizer").set_device(audio_device.clone()).to_dtype(DType::F32),
        )?;
        let acoustic_connector = SpeechConnector::new(
            cfg.acoustic_vae_dim, cfg.decoder_config.hidden_size,
            vb.pp("acoustic_connector").set_device(audio_device.clone()).to_dtype(DType::F32),
        )?;
        let prediction_head = DiffusionHead::new(
            &cfg.diffusion_head_config, vb.pp("prediction_head").set_device(audio_device.clone()).to_dtype(DType::F32),
        )?;

        let speech_scaling_factor = vb.get((), "speech_scaling_factor").ok()
            .and_then(|t| t.to_dtype(DType::F64).ok())
            .and_then(|t| t.to_scalar::<f64>().ok());
        let speech_bias_factor = vb.get((), "speech_bias_factor").ok()
            .and_then(|t| t.to_dtype(DType::F64).ok())
            .and_then(|t| t.to_scalar::<f64>().ok());

        Ok(Self {
            language_model,
            tts_language_model,
            tts_input_types,
            acoustic_tokenizer,
            acoustic_connector,
            prediction_head,
            speech_scaling_factor,
            speech_bias_factor,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
        self.tts_language_model.clear_kv_cache();
    }
}

/// Public streaming TTS model for VibeVoice-Realtime inference.
pub struct VibeVoiceStreaming {
    model: VibeVoiceStreamingModel,
    tts_eos_classifier: BinaryClassifier,
    cfg: StreamingConfig,
}

impl VibeVoiceStreaming {
    pub fn new(cfg: &StreamingConfig, vb: VarBuilder, audio_device: &Device) -> Result<Self> {
        let model = VibeVoiceStreamingModel::new(cfg, vb.pp("model"), audio_device)?;
        // tts_eos_classifier lives at root level (not under "model.")
        let tts_eos_classifier = BinaryClassifier::new(
            cfg.decoder_config.hidden_size, vb.pp("tts_eos_classifier"),
        )?;
        Ok(Self { model, tts_eos_classifier, cfg: cfg.clone() })
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.cfg
    }

    /// Run the lower LM on token ids. Returns hidden states.
    pub fn forward_lm(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.model.language_model.forward(input_ids, seqlen_offset, None)
    }

    /// Run the lower LM on pre-computed embeddings.
    pub fn forward_lm_embeds(
        &mut self,
        embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dtype = self.model.language_model.dtype();
        let embeds = embeds.to_dtype(dtype)?;
        self.model.language_model.forward_embeds(&embeds, seqlen_offset, None)
    }

    /// Run the upper TTS LM on pre-computed embeddings (already spliced
    /// with lm_hidden_state + tts_input_types).
    pub fn forward_tts_lm(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dtype = self.model.tts_language_model.dtype();
        let inputs_embeds = inputs_embeds.to_dtype(dtype)?;
        self.model.tts_language_model.forward_embeds(&inputs_embeds, seqlen_offset, None)
    }

    /// Get the text embedding for token ids (from the lower LM's embed_tokens).
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.language_model.embed_tokens().forward(input_ids)
    }

    /// Add tts_input_types embedding. `type_ids`: 0 = speech, 1 = text.
    pub fn add_tts_type_embedding(&self, embeds: &Tensor, type_ids: &Tensor) -> Result<Tensor> {
        let type_emb = self.model.tts_input_types.forward(type_ids)?;
        // Ensure embeds and type_emb match (e.g. if embeds was upcast/downcast)
        let embeds = embeds.to_dtype(type_emb.dtype())?;
        embeds.broadcast_add(&type_emb)
    }

    /// Run the EOS classifier on the last hidden state. Returns logit (pre-sigmoid).
    pub fn eos_logit(&self, hidden_state: &Tensor) -> Result<f32> {
        let dtype = self.model.tts_language_model.dtype();
        let logit = self.tts_eos_classifier.forward(&hidden_state.to_dtype(dtype)?)?;
        let logit = logit.to_dtype(DType::F32)?.flatten_all()?;
        logit.i(0)?.to_scalar::<f32>()
    }

    /// Pass speech latent through acoustic_connector to get embeddings for TTS LM input.
    pub fn acoustic_connector(&self, speech_latent: &Tensor) -> Result<Tensor> {
        let audio_dev = self.model.acoustic_connector.fc1.weight().device();
        let speech_audio_dev = speech_latent.to_device(audio_dev)?;
        let out = self.model.acoustic_connector.forward(&speech_audio_dev)?;
        let device = self.model.tts_input_types.embeddings().device();
        let dtype = self.model.tts_input_types.embeddings().dtype();
        out.to_device(device)?.to_dtype(dtype)
    }

    /// Diffusion head forward pass.
    pub fn prediction_head(
        &self,
        noisy: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        self.model.prediction_head.forward(noisy, timesteps, condition)
    }

    /// CFG-guided diffusion sampling of one speech latent frame.
    ///
    /// Matches the Python `sample_speech_tokens` which calls
    /// `scheduler.step(eps, t, speech)` at each timestep.
    pub fn sample_speech_tokens(
        &self,
        positive_condition: &Tensor,
        negative_condition: &Tensor,
        scheduler: &mut DpmSolverScheduler,
        cfg_scale: f64,
        _device: &Device,
        _dtype: DType,
    ) -> Result<Tensor> {
        let audio_dev = self.model.prediction_head.noisy_images_proj.weight().device();
        let pos_cond = positive_condition.to_device(audio_dev)?.to_dtype(DType::F32)?;
        let neg_cond = negative_condition.to_device(audio_dev)?.to_dtype(DType::F32)?;
        let condition = Tensor::cat(&[&pos_cond, &neg_cond], 0)?;
        let mut speech = Tensor::randn(
            0f32, 1f32, (2, self.cfg.acoustic_vae_dim), audio_dev,
        )?;

        let timesteps = scheduler.timesteps().to_vec();
        for &t in &timesteps {
            let half = speech.i(0..1)?;
            let combined = Tensor::cat(&[&half, &half], 0)?;
            let t_tensor = Tensor::full(t as f32, 2, audio_dev)?;
            let eps = self.model.prediction_head.forward(&combined, &t_tensor, &condition)?;

            // CFG: eps = uncond + cfg_scale * (cond - uncond)
            let cond_eps = eps.i(0..1)?;
            let uncond_eps = eps.i(1..2)?;
            let half_eps = (&uncond_eps + ((&cond_eps - &uncond_eps)? * cfg_scale)?)?;
            let full_eps = Tensor::cat(&[&half_eps, &half_eps], 0)?;

            // Let the scheduler do v_prediction→x0 conversion + DPM-Solver++ step
            speech = scheduler.step(&full_eps, &speech)?;
        }
        speech.i(0..1)
    }

    /// Decode a speech latent to audio waveform, applying inverse scaling.
    pub fn decode_speech(&self, latent: &Tensor) -> Result<Tensor> {
        let latent = if let (Some(bias), Some(scale)) =
            (self.model.speech_bias_factor, self.model.speech_scaling_factor)
        {
            ((latent / scale)? - bias)?
        } else {
            latent.clone()
        };
        self.model.acoustic_tokenizer.decode(&latent)
    }

    /// Create a new streaming cache for the acoustic decoder.
    pub fn new_acoustic_streaming_cache(&self) -> StreamingConvCache {
        self.model.acoustic_tokenizer.new_streaming_cache()
    }

    /// Streaming decode: decode one speech latent frame using cached conv state.
    pub fn decode_speech_streaming(
        &self,
        latent: &Tensor,
        cache: &mut StreamingConvCache,
        is_first: bool,
    ) -> Result<Tensor> {
        let latent = if let (Some(bias), Some(scale)) =
            (self.model.speech_bias_factor, self.model.speech_scaling_factor)
        {
            ((latent / scale)? - bias)?
        } else {
            latent.clone()
        };
        self.model
            .acoustic_tokenizer
            .decode_streaming(&latent, cache, is_first)
    }

    pub fn diffusion_config(&self) -> &DiffusionHeadConfig {
        &self.cfg.diffusion_head_config
    }

    /// Save TTS LM KV caches (for swapping between positive/negative CFG paths).
    pub fn save_tts_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.tts_language_model.save_kv_cache()
    }

    /// Restore TTS LM KV caches from a previously saved state.
    pub fn restore_tts_kv_cache(&mut self, cache: &[Option<(Tensor, Tensor)>]) {
        self.model.tts_language_model.restore_kv_cache(cache);
    }

    /// Restore TTS LM KV caches from raw (key, value) pairs (e.g., loaded from voice prompt).
    pub fn restore_tts_kv_cache_raw(&mut self, pairs: &[(Tensor, Tensor)]) {
        self.model.tts_language_model.set_kv_cache(pairs);
    }

    /// Save TTS LM KV caches as flat Vec of (key, value) pairs.
    pub fn save_tts_kv_cache_as_pairs(&self) -> Vec<(Tensor, Tensor)> {
        self.model
            .tts_language_model
            .save_kv_cache()
            .into_iter()
            .filter_map(|opt| opt)
            .collect()
    }

    /// Load pre-computed KV caches from a voice prompt safetensors file.
    /// Sets KV caches on both LMs (lm, tts_lm).
    /// Returns (lm_seq_len, tts_lm_seq_len) for seqlen_offset tracking.
    pub fn load_voice_prompt(
        &mut self,
        tensors: &std::collections::HashMap<String, Tensor>,
        _dtype: DType,
    ) -> Result<(usize, usize)> {
        let load_kv = |model: &mut qwen2::Model, prefix: &str, n_layers: usize| -> Result<usize> {
            let mut pairs = Vec::with_capacity(n_layers);
            let mut seq_len = 0;
            // Use model's native dtype for KV cache
            let target_dt = model.dtype();
            for i in 0..n_layers {
                let k = tensors.get(&format!("{prefix}.kv.{i}.key"))
                    .ok_or_else(|| candle::Error::Msg(format!("missing {prefix}.kv.{i}.key")))?
                    .to_dtype(target_dt)?;
                let v = tensors.get(&format!("{prefix}.kv.{i}.value"))
                    .ok_or_else(|| candle::Error::Msg(format!("missing {prefix}.kv.{i}.value")))?
                    .to_dtype(target_dt)?;
                if i == 0 {
                    seq_len = k.dim(2)?;
                }
                pairs.push((k, v));
            }
            model.set_kv_cache(&pairs);
            Ok(seq_len)
        };

        let lm_layers = self.model.language_model.num_layers();
        let tts_layers = self.model.tts_language_model.num_layers();

        let lm_seq = load_kv(&mut self.model.language_model, "lm", lm_layers)?;
        let tts_seq = load_kv(&mut self.model.tts_language_model, "tts_lm", tts_layers)?;

        Ok((lm_seq, tts_seq))
    }

    /// Get the last_hidden_state tensors from a voice prompt.
    pub fn get_voice_prompt_hidden_states(
        tensors: &std::collections::HashMap<String, Tensor>,
        dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let get = |key: &str| -> Result<Tensor> {
            tensors.get(key)
                .ok_or_else(|| candle::Error::Msg(format!("missing {key}")))?
                .to_dtype(dtype)
        };
        Ok((
            get("lm.last_hidden_state")?,
            get("tts_lm.last_hidden_state")?,
            get("neg_lm.last_hidden_state")?,
            get("neg_tts_lm.last_hidden_state")?,
        ))
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
