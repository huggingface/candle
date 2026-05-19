//! Wav2Vec2ForCTC — Candle implementation.
//!
//! Supports `Wav2Vec2ForCTC` (CTC ASR) from HuggingFace.
//! Loads directly from HF safetensors weight names.
//!
//! Architecture (do_stable_layer_norm = true):
//!   CNN feature extractor (7 layers, no-pad, GroupNorm or LayerNorm + GELU)
//!   → Feature projection  (LayerNorm → Linear)
//!   → Positional conv     (grouped Conv1d, same-pad, GELU, residual)
//!   → L × encoder layer   (pre-norm MHA + FFN)
//!   → Global LayerNorm    ← AFTER all transformer layers (critical!)
//!   → LM head             (Linear → vocab_size)
//!   → CTC greedy decode

use candle::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Wav2Vec2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// "group" = InstanceNorm on CNN layer 0 only; "layer" = LayerNorm all CNN layers.
    #[serde(default = "default_feat_extract_norm")]
    pub feat_extract_norm: String,
    pub conv_dim: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "default_num_conv_pos_embeddings")]
    pub num_conv_pos_embeddings: usize,
    #[serde(default = "default_num_conv_pos_embedding_groups")]
    pub num_conv_pos_embedding_groups: usize,
    #[serde(default = "default_do_stable_layer_norm")]
    pub do_stable_layer_norm: bool,
    #[serde(default)]
    pub pad_token_id: u32,
}

fn default_layer_norm_eps() -> f64 { 1e-5 }
fn default_feat_extract_norm() -> String { "group".to_string() }
fn default_num_conv_pos_embeddings() -> usize { 128 }
fn default_num_conv_pos_embedding_groups() -> usize { 16 }
fn default_do_stable_layer_norm() -> bool { false }

impl Wav2Vec2Config {
    pub fn num_feat_extract_layers(&self) -> usize { self.conv_dim.len() }
}

// ---------------------------------------------------------------------------
// Norm helpers on channel-first [B, C, T] tensors
// ---------------------------------------------------------------------------

/// InstanceNorm: normalise each channel over T independently.
/// Equivalent to GroupNorm(num_groups=C) on [B, C, T].
fn instance_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?;                       // [B, C, 1]
    let diff = x.broadcast_sub(&mean)?;
    let var  = diff.sqr()?.mean_keepdim(2)?;             // [B, C, 1]
    let std  = (var + eps)?.sqrt()?;
    let norm = diff.broadcast_div(&std)?;
    let w = weight.reshape((1, (), 1))?;
    let b = bias.reshape((1, (), 1))?;
    norm.broadcast_mul(&w)?.broadcast_add(&b)
}

/// Channel LayerNorm on [B, C, T]: normalise over C at each T position.
/// Transpose to [B, T, C], apply LN, transpose back.
fn layer_norm_cf(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let xt = x.transpose(1, 2)?;                         // [B, T, C]
    let mean = xt.mean_keepdim(2)?;
    let diff = xt.broadcast_sub(&mean)?;
    let var  = diff.sqr()?.mean_keepdim(2)?;
    let std  = (var + eps)?.sqrt()?;
    let norm = diff.broadcast_div(&std)?;
    let w = weight.reshape((1, 1, ()))?;
    let b = bias.reshape((1, 1, ()))?;
    norm.broadcast_mul(&w)?.broadcast_add(&b)?.transpose(1, 2)
}

fn gelu(x: &Tensor) -> Result<Tensor> { x.gelu_erf() }

// ---------------------------------------------------------------------------
// CNN feature extractor
// ---------------------------------------------------------------------------

enum ConvNorm {
    None,
    Instance { weight: Tensor, bias: Tensor },
    Layer    { weight: Tensor, bias: Tensor },
}

struct Wav2Vec2ConvLayer {
    conv: Conv1d,
    norm: ConvNorm,
}

impl Wav2Vec2ConvLayer {
    fn load(
        idx: usize,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        conv_bias: bool,
        feat_norm: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let vb_l = vb.pp(format!("conv_layers.{idx}"));
        let cfg = Conv1dConfig { stride, padding: 0, groups: 1, dilation: 1, ..Default::default() };
        let conv = if conv_bias {
            candle_nn::conv1d(in_ch, out_ch, kernel, cfg, vb_l.pp("conv"))?
        } else {
            candle_nn::conv1d_no_bias(in_ch, out_ch, kernel, cfg, vb_l.pp("conv"))?
        };

        let norm = match feat_norm {
            "layer" => {
                let w = vb_l.pp("layer_norm").get(out_ch, "weight")?;
                let b = vb_l.pp("layer_norm").get(out_ch, "bias")?;
                ConvNorm::Layer { weight: w, bias: b }
            }
            _ => {
                if idx == 0 {
                    let w = vb_l.pp("layer_norm").get(out_ch, "weight")?;
                    let b = vb_l.pp("layer_norm").get(out_ch, "bias")?;
                    ConvNorm::Instance { weight: w, bias: b }
                } else {
                    ConvNorm::None
                }
            }
        };

        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &Tensor, eps: f64) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let h = match &self.norm {
            ConvNorm::None                     => h,
            ConvNorm::Instance { weight, bias } => instance_norm(&h, weight, bias, eps)?,
            ConvNorm::Layer    { weight, bias } => layer_norm_cf(&h, weight, bias, eps)?,
        };
        gelu(&h)
    }
}

struct Wav2Vec2FeatureExtractor {
    layers: Vec<Wav2Vec2ConvLayer>,
    eps: f64,
}

impl Wav2Vec2FeatureExtractor {
    fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let vb_fe = vb.pp("wav2vec2.feature_extractor");
        let n = cfg.num_feat_extract_layers();
        let mut layers = Vec::with_capacity(n);
        let mut in_ch = 1usize;
        for i in 0..n {
            let l = Wav2Vec2ConvLayer::load(
                i, in_ch, cfg.conv_dim[i], cfg.conv_kernel[i], cfg.conv_stride[i],
                cfg.conv_bias, &cfg.feat_extract_norm, &vb_fe,
            )?;
            in_ch = cfg.conv_dim[i];
            layers.push(l);
        }
        Ok(Self { layers, eps: cfg.layer_norm_eps })
    }

    /// audio: [B, n_samples] → [B, T, C_cnn]
    fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let mut h = audio.unsqueeze(1)?;            // [B, 1, n_samples]
        for layer in &self.layers {
            h = layer.forward(&h, self.eps)?;
        }
        h.transpose(1, 2)                           // [B, T, C]
    }
}

// ---------------------------------------------------------------------------
// Feature projection
// ---------------------------------------------------------------------------

struct Wav2Vec2FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl Wav2Vec2FeatureProjection {
    fn load(cfg: &Wav2Vec2Config, cnn_out: usize, vb: VarBuilder) -> Result<Self> {
        let vb_fp = vb.pp("wav2vec2.feature_projection");
        Ok(Self {
            layer_norm: candle_nn::layer_norm(cnn_out, cfg.layer_norm_eps, vb_fp.pp("layer_norm"))?,
            projection: candle_nn::linear(cnn_out, cfg.hidden_size, vb_fp.pp("projection"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.projection.forward(&self.layer_norm.forward(x)?)
    }
}

// ---------------------------------------------------------------------------
// Positional convolutional embedding
// ---------------------------------------------------------------------------

struct Wav2Vec2PositionalConvEmbedding {
    conv: Conv1d,
    kernel_size: usize,
}

impl Wav2Vec2PositionalConvEmbedding {
    fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let h  = cfg.hidden_size;
        let k  = cfg.num_conv_pos_embeddings;
        let g  = cfg.num_conv_pos_embedding_groups;
        let pad = k / 2;
        let conv_cfg = Conv1dConfig { padding: pad, groups: g, stride: 1, dilation: 1, ..Default::default() };
        // Weight-norm must have been removed from the safetensors before loading.
        // The exported file from export_for_candle.py already has plain weights.
        let conv = candle_nn::conv1d(h, h, k, conv_cfg,
                    vb.pp("wav2vec2.encoder.pos_conv_embed.conv"))?;
        Ok(Self { conv, kernel_size: k })
    }

    /// hidden_states: [B, T, H] → [B, T, H]
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let h = hidden_states.transpose(1, 2)?;     // [B, H, T]
        let pos = self.conv.forward(&h)?;
        // Same-pad: even kernel → output is T+1, drop last element
        let pos = if self.kernel_size % 2 == 0 {
            let t = pos.dim(2)?;
            pos.narrow(2, 0, t - 1)?
        } else {
            pos
        };
        let pos = gelu(&pos)?.transpose(1, 2)?;     // [B, T, H]
        hidden_states + pos
    }
}

// ---------------------------------------------------------------------------
// Multi-head self-attention
// ---------------------------------------------------------------------------

struct Wav2Vec2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Wav2Vec2Attention {
    fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let vb_a = vb.pp("attention");
        let head_dim = h / cfg.num_attention_heads;
        Ok(Self {
            q_proj:    candle_nn::linear(h, h, vb_a.pp("q_proj"))?,
            k_proj:    candle_nn::linear(h, h, vb_a.pp("k_proj"))?,
            v_proj:    candle_nn::linear(h, h, vb_a.pp("v_proj"))?,
            out_proj:  candle_nn::linear(h, h, vb_a.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        // Project → split heads → flatten (batch, heads) for 3D matmul
        // [B, seq, H] → [B, seq, nh, hd] → [B, nh, seq, hd] → [B*nh, seq, hd]
        let reshape = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, seq, nh, hd))?
             .transpose(1, 2)?
             .reshape((b * nh, seq, hd))
        };

        let q = reshape(&self.q_proj.forward(x)?)?;
        let k = reshape(&self.k_proj.forward(x)?)?;
        let v = reshape(&self.v_proj.forward(x)?)?;

        // Scores: [B*nh, seq, seq]
        let scores = (q.matmul(&k.t()?)? * self.scale)?;
        let scores = match mask {
            Some(m) => {
                // m: [B, 1, seq, seq] → [B*nh, seq, seq]
                let m = m.broadcast_as(scores.shape())?;
                (scores + m)?
            }
            None => scores,
        };

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out  = attn.matmul(&v)?;                // [B*nh, seq, hd]

        // Merge heads: [B*nh, seq, hd] → [B, nh, seq, hd] → [B, seq, nh*hd]
        let out = out.reshape((b, nh, seq, hd))?
                     .transpose(1, 2)?
                     .contiguous()?
                     .reshape((b, seq, nh * hd))?;
        self.out_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Feed-forward network
// ---------------------------------------------------------------------------

struct Wav2Vec2FeedForward {
    intermediate_dense: Linear,
    output_dense: Linear,
}

impl Wav2Vec2FeedForward {
    fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let vb_ff = vb.pp("feed_forward");
        Ok(Self {
            intermediate_dense: candle_nn::linear(
                cfg.hidden_size, cfg.intermediate_size, vb_ff.pp("intermediate_dense"))?,
            output_dense: candle_nn::linear(
                cfg.intermediate_size, cfg.hidden_size, vb_ff.pp("output_dense"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        gelu(&self.intermediate_dense.forward(x)?)
            .and_then(|h| self.output_dense.forward(&h))
    }
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

struct Wav2Vec2EncoderLayer {
    attention:          Wav2Vec2Attention,
    feed_forward:       Wav2Vec2FeedForward,
    layer_norm:         LayerNorm,           // pre-attn LN
    final_layer_norm:   LayerNorm,           // pre-FFN LN
    do_stable_layer_norm: bool,
}

impl Wav2Vec2EncoderLayer {
    fn load(idx: usize, cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp(format!("wav2vec2.encoder.layers.{idx}"));
        let h = cfg.hidden_size;
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            attention:            Wav2Vec2Attention::load(cfg, vb_l.clone())?,
            feed_forward:         Wav2Vec2FeedForward::load(cfg, vb_l.clone())?,
            layer_norm:           candle_nn::layer_norm(h, eps, vb_l.pp("layer_norm"))?,
            final_layer_norm:     candle_nn::layer_norm(h, eps, vb_l.pp("final_layer_norm"))?,
            do_stable_layer_norm: cfg.do_stable_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if self.do_stable_layer_norm {
            // Pre-norm: LN → attn → residual; LN → FFN → residual
            let attn_out = self.attention.forward(&self.layer_norm.forward(x)?, mask)?;
            let x = (x + attn_out)?;
            let ff_out = self.feed_forward.forward(&self.final_layer_norm.forward(&x)?)?;
            &x + ff_out
        } else {
            // Post-norm: attn → residual → LN; FFN → residual → LN
            let attn_out = self.attention.forward(x, mask)?;
            let x = self.layer_norm.forward(&(x + attn_out)?)?;
            let ff_out = self.feed_forward.forward(&x)?;
            self.final_layer_norm.forward(&(&x + ff_out)?)
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

struct Wav2Vec2Encoder {
    pos_conv_embed: Wav2Vec2PositionalConvEmbedding,
    layers: Vec<Wav2Vec2EncoderLayer>,
    /// Global LayerNorm — applied AFTER all transformer layers.
    layer_norm: LayerNorm,
}

impl Wav2Vec2Encoder {
    fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let pos_conv_embed = Wav2Vec2PositionalConvEmbedding::load(cfg, vb.clone())?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Wav2Vec2EncoderLayer::load(i, cfg, vb.clone()))
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = candle_nn::layer_norm(
            cfg.hidden_size, cfg.layer_norm_eps,
            vb.pp("wav2vec2.encoder.layer_norm"))?;
        Ok(Self { pos_conv_embed, layers, layer_norm })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.pos_conv_embed.forward(x)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }
        // Global LN is LAST — outside the loop
        self.layer_norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Wav2Vec2Model
// ---------------------------------------------------------------------------

pub struct Wav2Vec2Model {
    feature_extractor:  Wav2Vec2FeatureExtractor,
    feature_projection: Wav2Vec2FeatureProjection,
    encoder:            Wav2Vec2Encoder,
}

impl Wav2Vec2Model {
    pub fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        let cnn_out = *cfg.conv_dim.last().unwrap();
        Ok(Self {
            feature_extractor:  Wav2Vec2FeatureExtractor::load(cfg, vb.clone())?,
            feature_projection: Wav2Vec2FeatureProjection::load(cfg, cnn_out, vb.clone())?,
            encoder:            Wav2Vec2Encoder::load(cfg, vb)?,
        })
    }

    /// audio: [B, n_samples] (normalised, 16kHz) → [B, T, H]
    pub fn forward(&self, audio: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let features  = self.feature_extractor.forward(audio)?;
        let projected = self.feature_projection.forward(&features)?;
        self.encoder.forward(&projected, mask)
    }
}

// ---------------------------------------------------------------------------
// Wav2Vec2ForCTC
// ---------------------------------------------------------------------------

pub struct Wav2Vec2ForCTC {
    pub wav2vec2: Wav2Vec2Model,
    pub lm_head:  Linear,
    pub config:   Wav2Vec2Config,
}

impl Wav2Vec2ForCTC {
    pub fn load(cfg: &Wav2Vec2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wav2vec2: Wav2Vec2Model::load(cfg, vb.clone())?,
            lm_head:  candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?,
            config:   cfg.clone(),
        })
    }

    /// audio: [B, n_samples] → logits: [B, T, vocab_size]
    pub fn forward(&self, audio: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.lm_head.forward(&self.wav2vec2.forward(audio, mask)?)
    }

    /// CTC greedy decode.  Returns one string per batch element.
    pub fn decode_ctc(&self, logits: &Tensor, vocab: &[String]) -> Result<Vec<String>> {
        let blank_id = self.config.pad_token_id as u32;
        let ids = logits.argmax(2)?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut results = Vec::with_capacity(ids.len());
        for seq in &ids {
            let mut text = String::new();
            let mut prev = u32::MAX;
            for &id in seq {
                if id != prev {
                    if id != blank_id {
                        if let Some(tok) = vocab.get(id as usize) {
                            if tok == "|" { text.push(' '); }
                            else if !matches!(tok.as_str(), "<unk>" | "<s>" | "</s>") {
                                text.push_str(tok);
                            }
                        }
                    }
                    prev = id;
                }
            }
            results.push(text.trim().to_string());
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Audio normalisation
// ---------------------------------------------------------------------------

/// Normalise audio to zero-mean, unit-variance per batch element.
/// audio: [B, T] → [B, T]
pub fn normalise_audio(audio: &Tensor) -> Result<Tensor> {
    let mean = audio.mean_keepdim(1)?;
    let diff = audio.broadcast_sub(&mean)?;
    let var  = diff.sqr()?.mean_keepdim(1)?;
    let std  = (var + 1e-7_f64)?.sqrt()?;
    diff.broadcast_div(&std)
}
