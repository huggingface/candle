//! Kokoro-82M text-to-speech model.
//!
//! Kokoro is a compact (82M parameter) TTS model with quality that rivals much larger models.
//! Architecture: PLBERT phoneme encoder → ProsodyPredictor → TextEncoder → ISTFTNet decoder.
//!
//! - Model card: <https://huggingface.co/hexgrad/Kokoro-82M>
//! - Reference code: <https://github.com/hexgrad/kokoro>
//!
//! # Usage
//! ```no_run
//! use candle_transformers::models::kokoro::{Config, KokoroModel};
//! use candle_nn::VarBuilder;
//! use candle::{Device, DType, Tensor};
//!
//! let cfg: Config = serde_json::from_str(include_str!("config.json")).unwrap();
//! let vb = unsafe {
//!     VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &Device::Cpu).unwrap()
//! };
//! let model = KokoroModel::new(&cfg, vb).unwrap();
//! // phoneme_ids: [1, seq_len], ref_s: [1, 256] voice embedding
//! let audio = model.forward(&phoneme_ids, &ref_s, 1.0).unwrap();
//! ```

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, ops::sigmoid, Conv1d, Conv1dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use candle_nn::{lstm, LSTMConfig, LSTM, RNN};
use serde::Deserialize;

fn leaky_relu(xs: &Tensor, slope: f64) -> Result<Tensor> {
    // max(x, slope*x) works for slope < 1: positive x → x, negative x → slope*x
    let scaled = (xs * slope)?;
    xs.maximum(&scaled)
}

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ISTFTNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

fn plbert_embedding_size_default() -> usize {
    128
}

#[derive(Debug, Clone, Deserialize)]
pub struct PLBertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub dropout: f64,
    #[serde(default = "plbert_embedding_size_default")]
    pub embedding_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub n_token: usize,
    pub hidden_dim: usize,
    pub style_dim: usize,
    pub n_mels: usize,
    pub n_layer: usize,
    pub max_dur: usize,
    pub dim_in: usize,
    pub dropout: f64,
    pub text_encoder_kernel_size: usize,
    pub istftnet: ISTFTNetConfig,
    pub plbert: PLBertConfig,
}

// ── Primitives ────────────────────────────────────────────────────────────────

/// Snake activation: x + sin²(alpha·x) / alpha
fn snake(xs: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = xs.broadcast_mul(alpha)?;
    let sin2 = ax.sin()?.sqr()?;
    xs + sin2.broadcast_mul(&alpha.recip()?)?
}

/// Per-channel instance normalization over the time axis.
/// Input: [batch, channels, time]  Output: same shape, zero-mean unit-variance per channel.
fn instance_norm_1d(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = xs.mean_keepdim(2)?;
    let diff = xs.broadcast_sub(&mean)?;
    let std = (diff.sqr()?.mean_keepdim(2)? + eps)?.sqrt()?;
    diff.broadcast_div(&std)
}

/// Load a Conv1d, handling both standard weights and weight-norm decomposed weights.
/// Weight-norm stores `weight_g` (magnitude) and `weight_v` (direction); compose on load.
fn wn_conv1d(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding,
        dilation,
        groups,
        stride: 1,
        ..Default::default()
    };
    // Try standard weight first (weight_norm removed before save — typical for Kokoro)
    if let Ok(w) = vb.get((out_ch, in_ch / groups, kernel), "weight") {
        let bias = vb.get(out_ch, "bias").ok();
        return Ok(Conv1d::new(w, bias, cfg));
    }
    // Reconstruct from weight_norm tensors
    let weight_g = vb.get((out_ch, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_ch, in_ch / groups, kernel), "weight_v")?;
    let norm = weight_v
        .reshape((out_ch, ()))?
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .reshape((out_ch, 1, 1))?;
    let weight = weight_v.broadcast_div(&norm)?.broadcast_mul(&weight_g)?;
    let bias = vb.get(out_ch, "bias").ok();
    Ok(Conv1d::new(weight, bias, cfg))
}

// ── PLBERT phoneme encoder (ALBERT-based) ─────────────────────────────────────

struct AlbertEmbeddings {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    layer_norm: LayerNorm,
}

impl AlbertEmbeddings {
    fn new(n_token: usize, cfg: &PLBertConfig, vb: VarBuilder) -> Result<Self> {
        let e = cfg.embedding_size;
        Ok(Self {
            word: embedding(n_token, e, vb.pp("word_embeddings"))?,
            position: embedding(cfg.max_position_embeddings, e, vb.pp("position_embeddings"))?,
            token_type: embedding(2, e, vb.pp("token_type_embeddings"))?,
            layer_norm: layer_norm(e, 1e-12, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, s) = ids.dims2()?;
        let dev = ids.device();
        let pos = Tensor::arange(0u32, s as u32, dev)?.unsqueeze(0)?.expand((b, s))?;
        let tt = Tensor::zeros((b, s), DType::U32, dev)?;
        let emb = (self.word.forward(ids)? + self.position.forward(&pos)? + self.token_type.forward(&tt)?)?;
        emb.apply(&self.layer_norm)
    }
}

struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl AlbertAttention {
    fn new(cfg: &PLBertConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let hd = h / heads;
        Ok(Self {
            query: linear(h, h, vb.pp("query"))?,
            key: linear(h, h, vb.pp("key"))?,
            value: linear(h, h, vb.pp("value"))?,
            dense: linear(h, h, vb.pp("dense"))?,
            layer_norm: layer_norm(h, 1e-12, vb.pp("LayerNorm"))?,
            num_heads: heads,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let project = |lin: &Linear| {
            xs.apply(lin)?
                .reshape((b, s, self.num_heads, self.head_dim))?
                .transpose(1, 2)
        };
        let q = (project(&self.query)? * self.scale)?;
        let k = project(&self.key)?;
        let v = project(&self.value)?;
        let attn = candle_nn::ops::softmax_last_dim(&q.matmul(&k.transpose(2, 3)?)?)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, ()))?.apply(&self.dense)?;
        (xs + out)?.apply(&self.layer_norm)
    }
}

struct AlbertLayer {
    attention: AlbertAttention,
    ffn: Linear,
    ffn_output: Linear,
    layer_norm: LayerNorm,
}

impl AlbertLayer {
    fn new(cfg: &PLBertConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            attention: AlbertAttention::new(cfg, vb.pp("attention"))?,
            ffn: linear(h, cfg.intermediate_size, vb.pp("ffn"))?,
            ffn_output: linear(cfg.intermediate_size, h, vb.pp("ffn_output"))?,
            layer_norm: layer_norm(h, 1e-12, vb.pp("full_layer_layer_norm"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attn = self.attention.forward(xs)?;
        let ff = attn.apply(&self.ffn)?.gelu_erf()?.apply(&self.ffn_output)?;
        (attn + ff)?.apply(&self.layer_norm)
    }
}

/// Phoneme-level ALBERT encoder. Weight-shared across all layers (ALBERT-style).
pub struct CustomAlbert {
    embeddings: AlbertEmbeddings,
    mapping_in: Linear,
    shared_layer: AlbertLayer,
    num_layers: usize,
}

impl CustomAlbert {
    pub fn new(n_token: usize, cfg: &PLBertConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: AlbertEmbeddings::new(n_token, cfg, vb.pp("embeddings"))?,
            mapping_in: linear(
                cfg.embedding_size,
                cfg.hidden_size,
                vb.pp("encoder.embedding_hidden_mapping_in"),
            )?,
            shared_layer: AlbertLayer::new(
                cfg,
                vb.pp("encoder.albert_layer_groups.0.albert_layers.0"),
            )?,
            num_layers: cfg.num_hidden_layers,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut h = self.embeddings.forward(input_ids)?.apply(&self.mapping_in)?;
        for _ in 0..self.num_layers {
            h = self.shared_layer.forward(&h)?;
        }
        Ok(h)
    }
}

// ── BiLSTM wrapper ────────────────────────────────────────────────────────────

/// Bidirectional LSTM. Both directions share the same VarBuilder prefix,
/// matching PyTorch's bidirectional LSTM state_dict layout.
struct BiLSTM {
    fwd: LSTM,
    bwd: LSTM,
}

impl BiLSTM {
    fn new(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fwd_cfg = LSTMConfig { direction: candle_nn::rnn::Direction::Forward, ..Default::default() };
        let bwd_cfg = LSTMConfig { direction: candle_nn::rnn::Direction::Backward, ..Default::default() };
        Ok(Self {
            fwd: lstm(in_dim, hidden_dim, fwd_cfg, vb.clone())?,
            bwd: lstm(in_dim, hidden_dim, bwd_cfg, vb)?,
        })
    }

    /// Input:  [batch, seq_len, in_dim]
    /// Output: [batch, seq_len, hidden_dim * 2]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let fwd_out = self.fwd.states_to_tensor(&self.fwd.seq(xs)?)?;
        let xs_rev = xs.flip(&[1usize])?;
        let bwd_out = self.bwd.states_to_tensor(&self.bwd.seq(&xs_rev)?)?.flip(&[1usize])?;
        Tensor::cat(&[fwd_out, bwd_out], 2)
    }
}

// ── TextEncoder ───────────────────────────────────────────────────────────────

/// Phoneme embedding → conv stack → BiLSTM.
/// Returns: [batch, hidden_dim, seq_len] (channel-last for conv compatibility).
pub struct TextEncoder {
    embedding: Embedding,
    convs: Vec<(Conv1d, LayerNorm)>,
    lstm: BiLSTM,
}

impl TextEncoder {
    pub fn new(
        n_token: usize,
        channels: usize,
        kernel: usize,
        n_layer: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding = embedding(n_token, channels, vb.pp("embedding"))?;
        let pad = (kernel - 1) / 2;
        let convs = (0..n_layer)
            .map(|i| {
                let cvb = vb.pp(format!("cnn.{i}"));
                let conv = wn_conv1d(channels, channels, kernel, pad, 1, 1, cvb.pp("0"))?;
                let ln = layer_norm(channels, 1e-5, cvb.pp("2"))?;
                Ok((conv, ln))
            })
            .collect::<Result<Vec<_>>>()?;
        // BiLSTM: in=channels, hidden per direction = channels/2 → total output = channels
        let lstm = BiLSTM::new(channels, channels / 2, vb.pp("lstm"))?;
        Ok(Self { embedding, convs, lstm })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // [b, s] → embedding → [b, s, ch] → transpose → [b, ch, s]
        let mut x = self.embedding.forward(input_ids)?.transpose(1, 2)?;
        for (conv, ln) in &self.convs {
            let res = x.clone();
            // Conv operates on [b, ch, s]
            x = x.apply(conv)?;
            // LayerNorm on [b, s, ch]
            x = x.transpose(1, 2)?.apply(ln)?.transpose(1, 2)?;
            x = (leaky_relu(&x, 0.2)? + res)?;
        }
        // BiLSTM expects [b, s, ch]
        x = self.lstm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        Ok(x) // [b, ch, s]
    }
}

// ── Style-conditioned norms ───────────────────────────────────────────────────

struct AdaLayerNorm {
    fc: Linear,
    channels: usize,
}

impl AdaLayerNorm {
    fn new(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { fc: linear(style_dim, channels * 2, vb.pp("fc"))?, channels })
    }

    /// xs: [b, s, ch], style: [b, style_dim] → [b, s, ch]
    fn forward(&self, xs: &Tensor, style: &Tensor) -> Result<Tensor> {
        let proj = style.apply(&self.fc)?; // [b, ch*2]
        let chunks = proj.chunk(2, 1)?;
        let (gamma, beta) = (&chunks[0], &chunks[1]);
        let gamma = (gamma.unsqueeze(1)? + 1.0)?;
        let beta = beta.unsqueeze(1)?;
        let mean = xs.mean_keepdim(2)?;
        let std = (xs.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)? + 1e-5)?.sqrt()?;
        let normed = xs.broadcast_sub(&mean)?.broadcast_div(&std)?;
        normed.broadcast_mul(&gamma)?.broadcast_add(&beta)
    }
}

pub struct AdaIN1d {
    fc: Linear,
}

impl AdaIN1d {
    pub fn new(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { fc: linear(style_dim, channels * 2, vb.pp("fc"))? })
    }

    /// xs: [b, ch, t], style: [b, style_dim] → [b, ch, t]
    pub fn forward(&self, xs: &Tensor, style: &Tensor) -> Result<Tensor> {
        let normed = instance_norm_1d(xs, 1e-5)?;
        let proj = style.apply(&self.fc)?; // [b, ch*2]
        let chunks = proj.chunk(2, 1)?;
        let (gamma, beta) = (&chunks[0], &chunks[1]);
        let gamma = (gamma.unsqueeze(2)? + 1.0)?; // [b, ch, 1]
        let beta = beta.unsqueeze(2)?;
        normed.broadcast_mul(&gamma)?.broadcast_add(&beta)
    }
}

// ── AdaIN residual block ──────────────────────────────────────────────────────

struct AdainResBlk1d {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
    shortcut: Option<Conv1d>,
    upsample: Option<usize>,
}

impl AdainResBlk1d {
    fn new(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        dilations: &[usize],
        kernel: usize,
        upsample: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n = dilations.len();
        let mut convs1 = Vec::with_capacity(n);
        let mut convs2 = Vec::with_capacity(n);
        let mut adain1 = Vec::with_capacity(n);
        let mut adain2 = Vec::with_capacity(n);
        let mut alpha1 = Vec::with_capacity(n);
        let mut alpha2 = Vec::with_capacity(n);
        for (i, &dil) in dilations.iter().enumerate() {
            let pad = (kernel - 1) * dil / 2;
            convs1.push(wn_conv1d(dim_in, dim_out, kernel, pad, dil, 1, vb.pp(format!("convs1.{i}")))?);
            convs2.push(wn_conv1d(dim_out, dim_out, kernel, pad, dil, 1, vb.pp(format!("convs2.{i}")))?);
            adain1.push(AdaIN1d::new(style_dim, dim_in, vb.pp(format!("adain1.{i}")))?);
            adain2.push(AdaIN1d::new(style_dim, dim_out, vb.pp(format!("adain2.{i}")))?);
            let k1 = format!("alpha1.{i}");
            alpha1.push(vb.get((1, dim_in, 1), &k1)?);
            let k2 = format!("alpha2.{i}");
            alpha2.push(vb.get((1, dim_out, 1), &k2)?);
        }
        let shortcut = if dim_in != dim_out {
            Some(wn_conv1d(dim_in, dim_out, 1, 0, 1, 1, vb.pp("conv1x1"))?)
        } else {
            None
        };
        Ok(Self { convs1, convs2, adain1, adain2, alpha1, alpha2, shortcut, upsample })
    }

    fn upsample_tensor(xs: &Tensor, factor: usize) -> Result<Tensor> {
        let t = xs.dim(2)?;
        xs.upsample_nearest1d(t * factor)
    }

    fn forward(&self, xs: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Shortcut: project channels and optionally upsample
        let mut res = match &self.shortcut {
            Some(c) => xs.apply(c)?,
            None => xs.clone(),
        };
        if let Some(f) = self.upsample {
            res = Self::upsample_tensor(&res, f)?;
        }

        let n = self.convs1.len();
        let mut sum = Tensor::zeros_like(&res)?;
        for i in 0..n {
            let mut xi = xs.clone();
            if let Some(f) = self.upsample {
                xi = Self::upsample_tensor(&xi, f)?;
            }
            xi = self.adain1[i].forward(&xi, style)?;
            xi = snake(&xi, &self.alpha1[i])?;
            xi = xi.apply(&self.convs1[i])?;
            xi = self.adain2[i].forward(&xi, style)?;
            xi = snake(&xi, &self.alpha2[i])?;
            xi = xi.apply(&self.convs2[i])?;
            sum = (sum + xi)?;
        }
        Ok(((res + sum * (1.0 / n as f64))? * (0.5_f64.sqrt()))?)
    }
}

// ── Duration encoder ──────────────────────────────────────────────────────────

struct DurationEncoder {
    layers: Vec<(LSTM, AdaLayerNorm)>,
    d_model: usize,
}

impl DurationEncoder {
    fn new(d_model: usize, style_dim: usize, n_layer: usize, vb: VarBuilder) -> Result<Self> {
        let layers = (0..n_layer)
            .map(|i| {
                let in_dim = if i == 0 { d_model + style_dim } else { d_model + style_dim };
                let l_cfg = LSTMConfig { layer_idx: 0, ..Default::default() };
                let l = lstm(in_dim, d_model / 2, l_cfg, vb.pp(format!("lstms.{i}.0")))?;
                let aln = AdaLayerNorm::new(style_dim, d_model / 2, vb.pp(format!("lstms.{i}.1")))?;
                Ok((l, aln))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, d_model })
    }

    /// xs: [b, s, d_model], style: [b, style_dim] → [b, s, d_model/2]
    fn forward(&self, xs: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let style_exp = style.unsqueeze(1)?.expand((b, s, style.dim(1)?))?;
        let mut x = Tensor::cat(&[xs, &style_exp], 2)?; // [b, s, d_model+style_dim]
        for (l, aln) in &self.layers {
            let states = l.seq(&x)?;
            let h = l.states_to_tensor(&states)?; // [b, s, d_model/2]
            x = aln.forward(&h, style)?;
            // Re-concat style for next layer
            x = Tensor::cat(&[&x, &style_exp], 2)?;
        }
        // Remove style concatenation from final output
        x.i((.., .., ..self.d_model / 2))
    }
}

// ── Prosody predictor ─────────────────────────────────────────────────────────

pub struct ProsodyPredictor {
    dur_enc: DurationEncoder,
    dur_lstm: LSTM,
    dur_proj: Linear,
    shared: LSTM,
    f0_blks: Vec<AdainResBlk1d>,
    n_blks: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

impl ProsodyPredictor {
    pub fn new(
        d_model: usize,
        style_dim: usize,
        n_layer: usize,
        max_dur: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dur_enc = DurationEncoder::new(d_model, style_dim, n_layer, vb.pp("text_encoder"))?;
        let l_cfg = LSTMConfig { layer_idx: 0, ..Default::default() };
        let dur_lstm = lstm(d_model / 2, d_model / 2, l_cfg.clone(), vb.pp("lstm"))?;
        let dur_proj = linear(d_model / 2, max_dur, vb.pp("duration_proj"))?;
        let shared = lstm(d_model / 2, d_model / 2, l_cfg, vb.pp("shared"))?;

        // Three AdainResBlk1d for F0 and N (energy) each
        // Block 0,1: dim_in→dim_in (no upsample); Block 2: dim_in→dim_in/2 (upsample ×2)
        let dils = [1usize, 3, 5];
        let mut f0_blks = Vec::new();
        let mut n_blks = Vec::new();
        for i in 0..3 {
            let (di, d_o, up) = if i < 2 {
                (d_model / 2, d_model / 2, None)
            } else {
                (d_model / 2, d_model / 4, Some(2usize))
            };
            f0_blks.push(AdainResBlk1d::new(di, d_o, style_dim, &dils, 3, up, vb.pp(format!("F0.{i}")))?);
            n_blks.push(AdainResBlk1d::new(di, d_o, style_dim, &dils, 3, up, vb.pp(format!("N.{i}")))?);
        }
        let f0_proj = wn_conv1d(d_model / 4, 1, 1, 0, 1, 1, vb.pp("F0_proj"))?;
        let n_proj = wn_conv1d(d_model / 4, 1, 1, 0, 1, 1, vb.pp("N_proj"))?;

        Ok(Self { dur_enc, dur_lstm, dur_proj, shared, f0_blks, n_blks, f0_proj, n_proj })
    }

    /// Returns: (durations: [1, s], f0: [1, 1, s*2], n: [1, 1, s*2])
    pub fn forward(
        &self,
        bert_enc: &Tensor, // [1, s, d_model]
        style: &Tensor,    // [1, style_dim]
        speed: f64,
    ) -> Result<(Vec<usize>, Tensor, Tensor)> {
        // Duration
        let dur_feat = self.dur_enc.forward(bert_enc, style)?; // [1, s, d_model/2]
        let dur_h = self.dur_lstm.states_to_tensor(&self.dur_lstm.seq(&dur_feat)?)?;
        let dur_logits = dur_h.apply(&self.dur_proj)?; // [1, s, max_dur]
        let dur_probs = sigmoid(&dur_logits)?;
        let dur_sums = (dur_probs.sum_keepdim(2)? / speed)?; // [1, s, 1]
        let durs_f32: Vec<f32> = dur_sums.squeeze(2)?.squeeze(0)?.to_vec1()?;
        let durations: Vec<usize> = durs_f32.iter().map(|&d| (d.round() as usize).max(1)).collect();

        // Shared LSTM for prosody features
        let shared_h = self.shared.states_to_tensor(&self.shared.seq(&dur_feat)?)?;
        // Repeat each frame by its duration → aligned feature map
        let aligned = repeat_by_durations(&shared_h, &durations)?; // [1, out_len, d_model/2]
        let feat = aligned.transpose(1, 2)?; // [1, d_model/2, out_len]

        // F0 branch
        let mut f0 = feat.clone();
        for blk in &self.f0_blks {
            f0 = blk.forward(&f0, style)?;
        }
        let f0 = f0.apply(&self.f0_proj)?; // [1, 1, out_len*2]

        // N (energy) branch
        let mut n = feat.clone();
        for blk in &self.n_blks {
            n = blk.forward(&n, style)?;
        }
        let n = n.apply(&self.n_proj)?; // [1, 1, out_len*2]

        Ok((durations, f0, n))
    }
}

fn repeat_by_durations(xs: &Tensor, durations: &[usize]) -> Result<Tensor> {
    let (b, s, d) = xs.dims3()?;
    assert_eq!(s, durations.len(), "duration len must match sequence len");
    let mut rows: Vec<Tensor> = Vec::new();
    for (i, &dur) in durations.iter().enumerate() {
        let row = xs.i((0, i, ..))?.unsqueeze(0)?; // [1, d]
        for _ in 0..dur.max(1) {
            rows.push(row.clone());
        }
    }
    if rows.is_empty() {
        return Tensor::zeros((b, 0, d), xs.dtype(), xs.device());
    }
    Tensor::stack(&rows, 0)?.unsqueeze(0) // [1, out_len, d]
}

// ── Harmonic-plus-noise source ────────────────────────────────────────────────

struct SineGen {
    sample_rate: f64,
    harmonic_num: usize,
    voiced_threshold: f64,
}

impl SineGen {
    fn new(sample_rate: f64, harmonic_num: usize) -> Self {
        Self { sample_rate, harmonic_num, voiced_threshold: 0.0 }
    }

    /// f0: [b, 1, t] in Hz → sines: [b, harmonic_num+1, t]
    fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        use std::f64::consts::PI;
        let (b, _, t) = f0.dims3()?;
        let dev = f0.device();
        let dt = f0.dtype();
        // Voiced mask
        let voiced = f0.gt(self.voiced_threshold)?.to_dtype(dt)?; // [b, 1, t]

        // Harmonic multipliers: [1, H+1, 1]
        let harmonics: Vec<f64> = (1..=(self.harmonic_num + 1)).map(|h| h as f64).collect();
        let harm_t = Tensor::from_vec(harmonics.iter().map(|&h| h as f32).collect::<Vec<_>>(),
                                      (1, self.harmonic_num + 1, 1), dev)?.to_dtype(dt)?;

        // Phase accumulation: phase[t] = 2π * f0 * t / sr
        // Simple approach: sample-independent (ignores inter-sample phase continuity)
        let t_idx = Tensor::arange(0u32, t as u32, dev)?.to_dtype(dt)?.reshape((1, 1, t))?;
        let phase = (f0.broadcast_mul(&harm_t)? * (2.0 * PI / self.sample_rate))?
            .broadcast_mul(&t_idx)?;
        let sines = (phase.sin()? * 0.1)?; // amplitude 0.1

        // Voiced sines + unvoiced noise
        let noise = Tensor::randn(0f32, 0.003f32, (b, self.harmonic_num + 1, t), dev)?
            .to_dtype(dt)?;
        let voiced_bc = voiced.broadcast_as((b, self.harmonic_num + 1, t))?;
        let unvoiced_bc = (voiced_bc.ones_like()? - &voiced_bc)?;
        let voiced_part = sines.broadcast_mul(&voiced_bc)?;
        let unvoiced_part = (noise * unvoiced_bc)?;
        voiced_part + unvoiced_part
    }
}

struct SourceModuleHnNSF {
    sine_gen: SineGen,
    l_linear: Linear,
}

impl SourceModuleHnNSF {
    fn new(sample_rate: f64, harmonic_num: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            sine_gen: SineGen::new(sample_rate, harmonic_num),
            l_linear: linear(harmonic_num + 1, 1, vb.pp("l_linear"))?,
        })
    }

    /// f0: [b, 1, t] → source: [b, 1, t]
    fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        let sines = self.sine_gen.forward(f0)?; // [b, H+1, t]
        sines.transpose(1, 2)?.apply(&self.l_linear)?.transpose(1, 2)?.tanh()
    }
}

// ── ISTFT (iDFT matrix multiply + overlap-add) ────────────────────────────────
//
// Kokoro uses n_fft=20 and hop_size=5 — small enough for a brute-force DFT approach.
// We precompute real and imaginary iDFT basis matrices at construction time.

struct ISTFT {
    n_fft: usize,
    hop_size: usize,
    /// Real part of iDFT basis: [n_bins, n_fft]
    real_basis: Tensor,
    /// Imaginary part of iDFT basis: [n_bins, n_fft]
    imag_basis: Tensor,
    /// Hann window: [n_fft]
    window: Tensor,
}

impl ISTFT {
    fn new(n_fft: usize, hop_size: usize, device: &Device, dtype: DType) -> Result<Self> {
        use std::f64::consts::PI;
        let n_bins = n_fft / 2 + 1;
        let mut real = vec![0.0f32; n_bins * n_fft];
        let mut imag_b = vec![0.0f32; n_bins * n_fft];
        for m in 0..n_bins {
            for k in 0..n_fft {
                let angle = 2.0 * PI * k as f64 * m as f64 / n_fft as f64;
                let scale = if m == 0 || m == n_fft / 2 { 1.0 } else { 2.0 };
                real[m * n_fft + k] = (scale * angle.cos() / n_fft as f64) as f32;
                imag_b[m * n_fft + k] = (-scale * angle.sin() / n_fft as f64) as f32;
            }
        }
        let window_vals: Vec<f32> = (0..n_fft)
            .map(|i| (PI * i as f64 / n_fft as f64).sin().powi(2) as f32)
            .collect();

        Ok(Self {
            n_fft,
            hop_size,
            real_basis: Tensor::from_vec(real, (n_bins, n_fft), device)?.to_dtype(dtype)?,
            imag_basis: Tensor::from_vec(imag_b, (n_bins, n_fft), device)?.to_dtype(dtype)?,
            window: Tensor::from_vec(window_vals, n_fft, device)?.to_dtype(dtype)?,
        })
    }

    /// spec_real: [b, n_bins, n_frames], spec_imag: [b, n_bins, n_frames]
    /// Returns: [b, n_samples]
    fn inverse(&self, spec_real: &Tensor, spec_imag: &Tensor) -> Result<Tensor> {
        let (b, _, n_frames) = spec_real.dims3()?;
        let n = self.n_fft;
        let hop = self.hop_size;
        let out_len = n_frames * hop + n;
        let dev = spec_real.device();
        let dt = spec_real.dtype();

        // Compute time-domain frames via iDFT matrix multiply
        // [b, n_bins, n_frames] → [b, n_frames, n_bins] → matmul [n_bins, n_fft] → [b, n_frames, n_fft]
        let sr_t = spec_real.transpose(1, 2)?; // [b, n_frames, n_bins]
        let si_t = spec_imag.transpose(1, 2)?;
        let frames = (sr_t.matmul(&self.real_basis)? - si_t.matmul(&self.imag_basis)?)?;
        // Apply Hann window: [b, n_frames, n_fft]
        let windowed = frames.broadcast_mul(&self.window.reshape((1, 1, n))?)?;

        // Overlap-add into output buffer
        let mut out = Tensor::zeros((b, out_len), dt, dev)?;
        for f in 0..n_frames {
            let start = f * hop;
            let end = start + n;
            let frame = windowed.i((.., f, ..))?.contiguous()?; // [b, n_fft]
            let current = out.i((.., start..end))?;
            out = out.slice_assign(&[0..b, start..end], &(current + frame)?)?;
        }
        Ok(out)
    }
}

// ── HiFiGAN-style ResBlock ────────────────────────────────────────────────────

struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock {
    fn new(channels: usize, kernel: usize, dils: &[usize], vb: VarBuilder) -> Result<Self> {
        let convs1 = dils.iter().enumerate()
            .map(|(i, &d)| wn_conv1d(channels, channels, kernel, (kernel - 1) * d / 2, d, 1, vb.pp(format!("convs1.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let convs2 = dils.iter().enumerate()
            .map(|(i, _)| wn_conv1d(channels, channels, kernel, (kernel - 1) / 2, 1, 1, vb.pp(format!("convs2.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let res = x.clone();
            x = leaky_relu(&x, 0.1)?.apply(c1)?;
            x = leaky_relu(&x, 0.1)?.apply(c2)?;
            x = (x + res)?;
        }
        Ok(x)
    }
}

// ── ISTFTNet Generator ────────────────────────────────────────────────────────

struct Generator {
    source: SourceModuleHnNSF,
    ups: Vec<ConvTranspose1d>,
    noise_convs: Vec<Conv1d>,
    resblocks: Vec<Vec<ResBlock>>,
    conv_post: Conv1d,
    istft: ISTFT,
    post_n_fft: usize,
    upsample_rates: Vec<usize>,
}

impl Generator {
    fn new(
        cfg: &ISTFTNetConfig,
        sample_rate: f64,
        device: &Device,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n_ups = cfg.upsample_rates.len();
        let source = SourceModuleHnNSF::new(sample_rate, 8, vb.pp("m_source"))?;

        let mut ups = Vec::with_capacity(n_ups);
        let mut noise_convs = Vec::with_capacity(n_ups);
        let mut ch = cfg.upsample_initial_channel;

        for i in 0..n_ups {
            let rate = cfg.upsample_rates[i];
            let kernel = cfg.upsample_kernel_sizes[i];
            let out_ch = ch / 2;
            let pad = (kernel - rate) / 2;
            // ConvTranspose1d weight shape: [in_ch, out_ch, kernel]
            let ct_cfg = ConvTranspose1dConfig {
                stride: rate,
                padding: pad,
                output_padding: 0,
                dilation: 1,
                groups: 1,
            };
            let ups_w_key = format!("ups.{i}.weight");
            let ups_b_key = format!("ups.{i}.bias");
            ups.push(ConvTranspose1d::new(
                vb.get((ch, out_ch, kernel), &ups_w_key)?,
                Some(vb.get(out_ch, &ups_b_key)?),
                ct_cfg,
            ));

            // Noise conv: source [b,1,T] → [b, out_ch, out_time]
            // stride matches remaining upsample factor so noise aligns with audio
            let noise_stride: usize = cfg.upsample_rates[i + 1..].iter().product::<usize>().max(1);
            let noise_pad = (noise_stride - 1) / 2;
            noise_convs.push(wn_conv1d(
                1, out_ch, noise_stride * 2, noise_pad, 1, 1,
                vb.pp(format!("noise_convs.{i}")),
            )?);
            ch = out_ch;
        }

        let n_rb = cfg.resblock_kernel_sizes.len();
        let mut resblocks: Vec<Vec<ResBlock>> = (0..n_ups).map(|_| Vec::new()).collect();
        for i in 0..n_ups * n_rb {
            let up_i = i / n_rb;
            let rb_i = i % n_rb;
            let ch_rb = cfg.upsample_initial_channel >> (up_i + 1);
            let dils = &cfg.resblock_dilation_sizes[rb_i % cfg.resblock_dilation_sizes.len()];
            let k = cfg.resblock_kernel_sizes[rb_i];
            resblocks[up_i].push(ResBlock::new(ch_rb, k, dils, vb.pp(format!("resblocks.{i}")))?);
        }

        let final_ch = cfg.upsample_initial_channel >> n_ups;
        let conv_post = wn_conv1d(final_ch, cfg.gen_istft_n_fft + 2, 7, 3, 1, 1, vb.pp("conv_post"))?;
        let istft = ISTFT::new(cfg.gen_istft_n_fft, cfg.gen_istft_hop_size, device, dtype)?;

        Ok(Self {
            source,
            ups,
            noise_convs,
            resblocks,
            conv_post,
            istft,
            post_n_fft: cfg.gen_istft_n_fft,
            upsample_rates: cfg.upsample_rates.clone(),
        })
    }

    /// x: [b, ch, t], f0: [b, 1, t] (already aligned to frame resolution)
    fn forward(&self, x_in: &Tensor, f0: &Tensor) -> Result<Tensor> {
        // Upsample F0 to full audio resolution for HnNSF source
        let total_up: usize = self.upsample_rates.iter().product();
        let t_audio = x_in.dim(2)? * total_up;
        let f0_full = f0.upsample_nearest1d(t_audio)?;
        let noise = self.source.forward(&f0_full)?; // [b, 1, t_audio]

        let mut x = x_in.clone();
        for (i, (up, noise_c)) in self.ups.iter().zip(self.noise_convs.iter()).enumerate() {
            x = leaky_relu(&x, 0.1)?.apply(up)?;
            // Align noise to current resolution
            let t_cur = x.dim(2)?;
            let noise_frac = noise.i((.., .., ..t_cur.min(noise.dim(2)?)))?;
            let noise_proj = noise_c.forward(&noise_frac)?;
            let len = t_cur.min(noise_proj.dim(2)?);
            x = (x.i((.., .., ..len))? + noise_proj.i((.., .., ..len))?)?;

            let rbs = &self.resblocks[i];
            let mut acc = Tensor::zeros_like(&x)?;
            for rb in rbs {
                acc = (acc + rb.forward(&x)?)?;
            }
            x = (acc / rbs.len() as f64)?;
        }

        x = leaky_relu(&x, 0.1)?.apply(&self.conv_post)?;
        // Split: first n_bins = magnitude tanh, next n_bins = phase
        let n_bins = self.post_n_fft / 2 + 1;
        let mag = x.i((.., ..n_bins, ..))?.tanh()?;
        let phase = x.i((.., n_bins.., ..))?.sin()?; // Use sin as phase proxy

        // Build real/imag from polar representation: real=mag*cos, imag=mag*sin
        // Since phase represents the angle, cos and sin derive from it
        let spec_real = (&mag * phase.cos()?)?;
        let spec_imag = (&mag * &phase)?;

        // ISTFT → [b, n_samples]
        self.istft.inverse(&spec_real, &spec_imag)
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────────

pub struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    generator: Generator,
}

impl Decoder {
    pub fn new(
        hidden_dim: usize,
        style_dim: usize,
        cfg: &ISTFTNetConfig,
        device: &Device,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Encoder block: [hidden_dim + 2] → 1024 (+ 2 from F0/N channels)
        let encode = AdainResBlk1d::new(
            hidden_dim + 2, 1024, style_dim, &[1], 3, None, vb.pp("encode"),
        )?;
        // 4 decode blocks: each takes 1024 + 2 (F0/N) + 64 (asr residual) input
        let decode = (0..4)
            .map(|i| {
                let (di, d_o, up) = if i < 3 {
                    (1024 + 2 + 64, 1024, None)
                } else {
                    (1024 + 2 + 64, 512, Some(2usize))
                };
                AdainResBlk1d::new(di, d_o, style_dim, &[1], 3, up, vb.pp(format!("decode.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let f0_conv = wn_conv1d(1, 1, 3, 1, 1, 1, vb.pp("F0_conv"))?;
        let n_conv = wn_conv1d(1, 1, 3, 1, 1, 1, vb.pp("N_conv"))?;
        let asr_res = wn_conv1d(hidden_dim, 64, 1, 0, 1, 1, vb.pp("asr_res.0"))?;
        let generator = Generator::new(cfg, 24_000.0, device, dtype, vb.pp("generator"))?;

        Ok(Self { encode, decode, f0_conv, n_conv, asr_res, generator })
    }

    /// asr: [1, hidden_dim, s], f0: [1, 1, s*2], n_e: [1, 1, s*2], style: [1, style_dim]
    pub fn forward(
        &self,
        asr: &Tensor,
        f0: &Tensor,
        n_e: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        let t = asr.dim(2)?;

        // Smooth F0 and N with learned conv (stride=1 keeps same length)
        let f0_s = f0.apply(&self.f0_conv)?;
        let n_s = n_e.apply(&self.n_conv)?;

        // Encode: concat [asr, f0[:t], n[:t]] → 1024
        let f0_t = f0_s.i((.., .., ..t))?;
        let n_t = n_s.i((.., .., ..t))?;
        let mut x = self.encode.forward(&Tensor::cat(&[asr, &f0_t, &n_t], 1)?, style)?;

        // ASR residual: [1, hidden_dim, t] → [1, 64, t] → upsample to 2t
        let asr_r = asr.apply(&self.asr_res)?.upsample_nearest1d(t * 2)?;

        // Decode blocks
        for (i, blk) in self.decode.iter().enumerate() {
            let t_cur = x.dim(2)?;
            let f0_cur = f0_s.i((.., .., ..t_cur.min(f0_s.dim(2)?)))?;
            let n_cur = n_s.i((.., .., ..t_cur.min(n_s.dim(2)?)))?;
            let asr_cur = asr_r.i((.., .., ..t_cur.min(asr_r.dim(2)?)))?;
            let cat = Tensor::cat(&[&x, &f0_cur, &n_cur, &asr_cur], 1)?;
            x = blk.forward(&cat, style)?;
        }

        // ISTFTNet generator: x=[1,512,2t], f0=[1,1,s*2]
        let f0_for_gen = f0_s.i((.., .., ..x.dim(2)?))?;
        self.generator.forward(&x, &f0_for_gen)
    }
}

// ── Top-level model ───────────────────────────────────────────────────────────

pub struct KokoroModel {
    bert: CustomAlbert,
    bert_encoder: Linear,
    prosody_predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
}

impl KokoroModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Ok(Self {
            bert: CustomAlbert::new(cfg.n_token, &cfg.plbert, vb.pp("bert"))?,
            bert_encoder: linear(cfg.plbert.hidden_size, cfg.hidden_dim, vb.pp("bert_encoder"))?,
            prosody_predictor: ProsodyPredictor::new(
                cfg.hidden_dim, cfg.style_dim, cfg.n_layer, cfg.max_dur,
                vb.pp("prosody_predictor"),
            )?,
            text_encoder: TextEncoder::new(
                cfg.n_token, cfg.hidden_dim, cfg.text_encoder_kernel_size,
                cfg.n_layer, vb.pp("text_encoder"),
            )?,
            decoder: Decoder::new(
                cfg.hidden_dim, cfg.style_dim, &cfg.istftnet,
                &device, dtype, vb.pp("decoder"),
            )?,
        })
    }

    /// Synthesize audio from phoneme token IDs and a voice reference embedding.
    ///
    /// - `phoneme_ids`: `[1, seq_len]` u32 tensor of phoneme indices from the vocab.
    /// - `ref_s`: `[1, 256]` f32 voice embedding. First 128 dims = acoustic style;
    ///   last 128 dims = prosody style.
    /// - `speed`: speaking rate multiplier (1.0 = normal).
    ///
    /// Returns a flat `[n_samples]` f32 tensor of 24 kHz PCM audio.
    pub fn forward(
        &self,
        phoneme_ids: &Tensor,
        ref_s: &Tensor,
        speed: f64,
    ) -> Result<Tensor> {
        // Split voice embedding
        let acoustic = ref_s.i((.., ..128))?;  // [1, 128]
        let prosody = ref_s.i((.., 128..))?;   // [1, 128]

        // BERT encoding: [1, s, 768] → [1, s, hidden_dim]
        let bert_h = self.bert.forward(phoneme_ids)?.apply(&self.bert_encoder)?;

        // Text encoding (parallel path): [1, hidden_dim, s]
        let text_feat = self.text_encoder.forward(phoneme_ids)?;

        // Prosody: durations (Vec<usize>), f0 [1,1,s*2], n [1,1,s*2]
        let (durations, f0, n_energy) =
            self.prosody_predictor.forward(&bert_h, &prosody, speed)?;

        // Align text features by predicted durations
        let text_aligned = align_text(&text_feat, &durations)?; // [1, hidden_dim, out_len]

        // Decode to audio [1, n_samples]
        let audio = self.decoder.forward(&text_aligned, &f0, &n_energy, &acoustic)?;

        audio.squeeze(0) // [n_samples]
    }
}

fn align_text(feat: &Tensor, durations: &[usize]) -> Result<Tensor> {
    let (_, ch, _) = feat.dims3()?;
    let mut cols: Vec<Tensor> = Vec::new();
    for (i, &d) in durations.iter().enumerate() {
        let col = feat.i((.., .., i..i + 1))?; // [1, ch, 1]
        for _ in 0..d.max(1) {
            cols.push(col.clone());
        }
    }
    if cols.is_empty() {
        return Tensor::zeros((1, ch, 0), feat.dtype(), feat.device());
    }
    Tensor::cat(&cols, 2) // [1, ch, out_len]
}

// ── Voice embeddings helper ───────────────────────────────────────────────────

/// Convenience wrapper to load pre-computed Kokoro voice embeddings.
///
/// Embeddings are stored as individual `.pt` (PyTorch) or `.npy` (NumPy) files
/// in the `voices/` directory of the HF repo. Use `candle_core::pickle` or the
/// `npyz` crate to load them, then pass the `[1, 256]` tensor to `KokoroModel::forward`.
pub struct VoiceEmbeddings {
    pub embeddings: std::collections::HashMap<String, Tensor>,
}

impl VoiceEmbeddings {
    /// Load all `.safetensors` voice embeddings from a directory.
    /// Each file should contain a single `[1, 256]` tensor named `"embedding"`.
    pub fn from_safetensors_dir(
        dir: &std::path::Path,
        device: &Device,
    ) -> Result<Self> {
        let mut embeddings = std::collections::HashMap::new();
        for entry in std::fs::read_dir(dir)
            .map_err(|e| candle::Error::Msg(format!("read_dir: {e}")))?
        {
            let entry = entry.map_err(|e| candle::Error::Msg(format!("dir entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                // Safety: file path is from read_dir and we own the handle lifetime.
                let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, device)? };
                if let Ok(t) = vb.get((1, 256), "embedding") {
                    embeddings.insert(name, t);
                }
            }
        }
        Ok(Self { embeddings })
    }

    /// Default voice name for a given language code.
    pub fn default_voice(language: &str) -> &'static str {
        match language {
            "ja" => "jf_alpha",
            "zh" => "zf_xiaoxiao",
            "en" => "af_heart",
            "fr" => "ff_siwis",
            "es" | "pt" => "pf_dora",
            "de" | "it" => "ef_dora",
            "ko" => "af_heart", // No dedicated Korean voice yet
            _ => "af_heart",
        }
    }
}
