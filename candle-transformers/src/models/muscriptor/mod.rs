//! MuScriptor: audio-to-MIDI music transcription.
//!
//! A decoder-only causal transformer (audiocraft/MusicGen lineage): the audio is
//! encoded as a log-mel spectrogram, projected to the model dimension and
//! prepended as prefix tokens together with two class-conditioning embeddings
//! (instrument group, dataset), after which MIDI event tokens are generated
//! autoregressively over a single stream.
//!
//! Weights: hf.co/MuScriptor/muscriptor-{small,medium,large} (CC BY-NC 4.0).

pub mod mel;
pub mod tokenizer;

use crate::generation::{LogitsProcessor, Sampling};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::kv_cache::KvCache;
use candle_nn::{layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder};

pub const SAMPLE_RATE: usize = 16000;
pub const SEGMENT_DURATION: f64 = 5.0;
pub const FRAME_RATE: usize = 100;
const N_FFT: usize = 2048;
const N_MELS: usize = 512;
const MEL_EPS: f32 = 1e-6;
const MAX_PERIOD: f32 = 10000.;
const LN_EPS: f64 = 1e-5;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub card: usize,
}

impl Config {
    pub fn small() -> Self {
        Self {
            dim: 768,
            num_heads: 12,
            num_layers: 14,
            card: 1393,
        }
    }

    pub fn medium() -> Self {
        Self {
            dim: 1024,
            num_heads: 16,
            num_layers: 24,
            card: 1395,
        }
    }

    pub fn large() -> Self {
        Self {
            dim: 1536,
            num_heads: 24,
            num_layers: 48,
            card: 1395,
        }
    }
}

/// `positions.cos() ++ positions.sin()` sinusoidal embedding, computed in f32.
/// Matches the reference `create_sin_embedding`: the exponent denominator is
/// `half_dim - 1` (not the usual `half_dim`).
fn sin_embedding(offset: usize, t: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut data = vec![0f32; t * dim];
    for pos_idx in 0..t {
        let pos = (offset + pos_idx) as f32;
        for i in 0..half {
            let period = MAX_PERIOD.powf(i as f32 / (half - 1) as f32);
            let phase = pos / period;
            data[pos_idx * dim + i] = phase.cos();
            data[pos_idx * dim + half + i] = phase.sin();
        }
    }
    Tensor::from_vec(data, (t, dim), device)
}

/// LayerNorm that dispatches to the fused kernel on Metal (a single launch
/// instead of the decomposed mean/variance op chain, which dominates decode
/// time at batch 1-4) and to the standard module elsewhere.
#[derive(Debug, Clone)]
struct Norm {
    inner: LayerNorm,
    fused: bool,
}

impl Norm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let fused = vb.device().is_metal();
        Ok(Self {
            inner: layer_norm(dim, LN_EPS, vb)?,
            fused,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match (self.fused, self.inner.bias()) {
            (true, Some(bias)) => {
                candle_nn::ops::layer_norm(xs, self.inner.weight(), bias, LN_EPS as f32)
            }
            _ => self.inner.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct StreamingMultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    kv_cache: KvCache,
    num_heads: usize,
    head_dim: usize,
}

impl StreamingMultiheadAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj_weight = vb.get((3 * dim, dim), "in_proj_weight")?;
        let in_proj = Linear::new(in_proj_weight, None);
        let out_proj = linear_no_bias(dim, dim, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            kv_cache: KvCache::new(2, 128),
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    fn reset_state(&mut self, max_seq_len: usize) {
        self.kv_cache = KvCache::new(2, max_seq_len);
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, dim) = xs.dims3()?;
        // The fused projection packs (q, k, v) along the outer dimension.
        let qkv = xs
            .apply(&self.in_proj)?
            .reshape((b, t, 3, self.num_heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = if xs.device().is_metal() {
            // Fused kernels; they read the KV cache's narrowed views through
            // their strides directly. Prefill always starts from an empty
            // cache (q_len == k_len), so plain causal alignment is correct;
            // a single decode step attends to the whole cache.
            candle_nn::ops::sdpa(&q.contiguous()?, &k, &v, None, t > 1, scale as f32, 1.0)?
        } else {
            let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn = match mask {
                None => attn,
                Some(mask) => attn.broadcast_add(mask)?,
            };
            // Softmax in f32 as torch's SDPA does for half-precision inputs.
            let dtype = attn.dtype();
            let attn = if dtype == DType::F32 {
                candle_nn::ops::softmax_last_dim(&attn)?
            } else {
                let attn = attn.to_dtype(DType::F32)?;
                candle_nn::ops::softmax_last_dim(&attn)?.to_dtype(dtype)?
            };
            attn.matmul(&v.contiguous()?)?
        };
        attn.transpose(1, 2)?
            .reshape((b, t, dim))?
            .apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct StreamingTransformerLayer {
    self_attn: StreamingMultiheadAttention,
    norm1: Norm,
    norm2: Norm,
    linear1: Linear,
    linear2: Linear,
}

impl StreamingTransformerLayer {
    fn new(dim: usize, num_heads: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: StreamingMultiheadAttention::new(dim, num_heads, vb.pp("self_attn"))?,
            norm1: Norm::new(dim, vb.pp("norm1"))?,
            norm2: Norm::new(dim, vb.pp("norm2"))?,
            linear1: linear_no_bias(dim, hidden, vb.pp("linear1"))?,
            linear2: linear_no_bias(hidden, dim, vb.pp("linear2"))?,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = (xs + self.self_attn.forward(&self.norm1.forward(xs)?, mask)?)?;
        let ff = self
            .norm2
            .forward(&xs)?
            .apply(&self.linear1)?
            // torch's F.gelu defaults to the exact erf formulation.
            .gelu_erf()?
            .apply(&self.linear2)?;
        xs + ff
    }
}

#[derive(Debug, Clone)]
struct StreamingTransformer {
    layers: Vec<StreamingTransformerLayer>,
    offset: usize,
}

impl StreamingTransformer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(StreamingTransformerLayer::new(
                cfg.dim,
                cfg.num_heads,
                4 * cfg.dim,
                vb_l.pp(i),
            )?)
        }
        Ok(Self { layers, offset: 0 })
    }

    fn reset_state(&mut self, max_seq_len: usize) {
        self.offset = 0;
        for layer in self.layers.iter_mut() {
            layer.self_attn.reset_state(max_seq_len)
        }
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_b, t, dim) = xs.dims3()?;
        let pos_emb = sin_embedding(self.offset, t, dim, xs.device())?.to_dtype(xs.dtype())?;
        let mut xs = xs.broadcast_add(&pos_emb)?;
        let mask = if t <= 1 || xs.device().is_metal() {
            // A single decode step attends to the whole cache, and the fused
            // Metal attention applies causality itself: no mask needed.
            None
        } else {
            let offset = self.offset;
            let s = offset + t;
            let mask: Vec<f32> = (0..t)
                .flat_map(|i| {
                    (0..s).map(move |j| {
                        if j <= offset + i {
                            0.
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            Some(Tensor::from_vec(mask, (t, s), xs.device())?.to_dtype(xs.dtype())?)
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref())?
        }
        self.offset += t;
        Ok(xs)
    }
}

/// Log-mel-spectrogram conditioner: mel bins are computed on the host in f32
/// and projected to the transformer dimension.
#[derive(Debug, Clone)]
pub struct MelConditioner {
    mel: mel::MelSpectrogram,
    output_proj: Linear,
    device: Device,
}

impl MelConditioner {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb_t = vb.pp("mel_spec_transform");
        let window = vb_t
            .get(N_FFT, "spectrogram.window")?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let fb = vb_t
            .get((N_FFT / 2 + 1, N_MELS), "mel_scale.fb")?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let hop = SAMPLE_RATE / FRAME_RATE;
        Ok(Self {
            mel: mel::MelSpectrogram::new(N_FFT, hop, N_MELS, window, fb),
            output_proj: linear(N_MELS, dim, vb.pp("output_proj"))?,
            device: vb.device().clone(),
        })
    }

    /// Encode one fixed-size audio segment into `[1, n_frames, dim]` (f32).
    pub fn encode(&self, samples: &[f32]) -> Result<Tensor> {
        let (mel, n_frames) = self.mel.compute(samples);
        let mel = mel.iter().map(|&v| (v + MEL_EPS).ln()).collect::<Vec<_>>();
        let mel = Tensor::from_vec(mel, (1, n_frames, N_MELS), &self.device)?;
        let embeds = mel.apply(&self.output_proj)?;
        // The reference masks frames at index >= len(samples) / hop; with
        // center-padded STFT there are len/hop + 1 frames, so the final frame
        // is always zeroed.
        let valid = samples.len() / self.mel.hop_length();
        if valid < n_frames {
            let zeros = Tensor::zeros(
                (1, n_frames - valid, embeds.dim(2)?),
                embeds.dtype(),
                embeds.device(),
            )?;
            Tensor::cat(&[embeds.narrow(1, 0, valid)?, zeros], 1)
        } else {
            Ok(embeds)
        }
    }
}

/// Class-index conditioner (instrument group / dataset name).
#[derive(Debug, Clone)]
pub struct ClassConditioner {
    embed: Embedding,
    device: Device,
}

impl ClassConditioner {
    fn new(num_classes: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let embed = candle_nn::embedding(num_classes + 1, dim, vb.pp("embed"))?;
        Ok(Self {
            embed,
            device: vb.device().clone(),
        })
    }

    /// Encode a class-id sequence into `[1, len, dim]` (f32). `None` is the
    /// null (unconditional) condition. The reference tokenizer adds 1 to the
    /// raw ids (null = -1) and its forward pass adds 1 again, so class `c`
    /// lands on embedding row `c + 2` and null on row 1.
    pub fn encode(&self, classes: Option<&[u32]>) -> Result<Tensor> {
        let ids = match classes {
            None => vec![1u32],
            Some(cs) => cs.iter().map(|c| c + 2).collect(),
        };
        let len = ids.len();
        let ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        self.embed.forward(&ids)
    }
}

fn get_with_legacy_prefix(
    vb: &VarBuilder,
    shape: (usize, usize),
    name: &str,
    legacy: &str,
) -> Result<Tensor> {
    // Published checkpoints keep the audiocraft multi-codebook naming for the
    // token embedding and output head (`emb.0.*` / `linears.0.*`).
    if vb.contains_tensor(legacy) {
        vb.get(shape, legacy)
    } else {
        vb.get(shape, name)
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum number of generated tokens per segment.
    pub max_gen_len: usize,
    /// `None` decodes greedily (the reference default).
    pub sampling: Option<Sampling>,
    pub seed: u64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_gen_len: 2000,
            sampling: None,
            seed: 299792458,
        }
    }
}

pub struct Model {
    pub mel_conditioner: MelConditioner,
    pub instrument_conditioner: ClassConditioner,
    pub dataset_conditioner: ClassConditioner,
    emb: Embedding,
    transformer: StreamingTransformer,
    out_norm: Norm,
    linear: Linear,
    card: usize,
    dtype: DType,
    device: Device,
}

impl Model {
    /// Build the model. `vb` carries the transformer compute dtype; the
    /// conditioners always run in f32 (`vb_f32`), matching the reference
    /// which keeps the conditioning pipeline in full precision.
    pub fn new(cfg: &Config, vb: VarBuilder, vb_f32: VarBuilder) -> Result<Self> {
        let conds = vb_f32.pp("condition_provider").pp("conditioners");
        let mel_conditioner = MelConditioner::new(cfg.dim, conds.pp("self_wav"))?;
        let instrument_conditioner =
            ClassConditioner::new(1000, cfg.dim, conds.pp("instrument_group"))?;
        let dataset_conditioner = ClassConditioner::new(4, cfg.dim, conds.pp("dataset_name"))?;
        let emb_weight =
            get_with_legacy_prefix(&vb, (cfg.card + 1, cfg.dim), "emb.weight", "emb.0.weight")?;
        let linear_weight = get_with_legacy_prefix(
            &vb,
            (cfg.card, cfg.dim),
            "linear.weight",
            "linears.0.weight",
        )?;
        Ok(Self {
            mel_conditioner,
            instrument_conditioner,
            dataset_conditioner,
            emb: Embedding::new(emb_weight, cfg.dim),
            transformer: StreamingTransformer::new(cfg, vb.pp("transformer"))?,
            out_norm: Norm::new(cfg.dim, vb.pp("out_norm"))?,
            linear: Linear::new(linear_weight, None),
            card: cfg.card,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// The BOS token prepended to every generated sequence.
    pub fn initial_token_id(&self) -> u32 {
        self.card as u32
    }

    /// Build the conditioning prefix for a batch of equally-sized audio
    /// segments: `[mel frames, dataset, instrument group(s)]`, in the
    /// transformer dtype, shape `[b, prefix_len, dim]`.
    pub fn build_prefix(
        &self,
        segments: &[impl AsRef<[f32]>],
        instrument_classes: Option<&[u32]>,
    ) -> Result<Tensor> {
        let mels = segments
            .iter()
            .map(|s| self.mel_conditioner.encode(s.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let mel = Tensor::cat(&mels, 0)?;
        let b = segments.len();
        // Datasets are always conditioned on the null class at inference.
        let dataset = self
            .dataset_conditioner
            .encode(None)?
            .expand((b, 1, mel.dim(2)?))?;
        let instrument = self.instrument_conditioner.encode(instrument_classes)?;
        let instrument = instrument.expand((b, instrument.dim(1)?, mel.dim(2)?))?;
        // The reference prepends conditions in front of the tokens one at a
        // time (instrument, then dataset, then mel), which yields this order.
        Tensor::cat(&[mel, dataset, instrument], 1)?.to_dtype(self.dtype)
    }

    /// One forward pass over embedded inputs; returns the last position's
    /// logits over the usable vocabulary, in f32, shape `[b, vocab]`.
    fn forward_embeds(&mut self, xs: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let out = self.transformer.forward(xs)?;
        // contiguous: the fused layer-norm kernel requires it, and narrowing
        // the time axis leaves a strided view when b > 1.
        let out = out.narrow(1, t - 1, 1)?.contiguous()?;
        let logits = self.out_norm.forward(&out)?.apply(&self.linear)?;
        // The reference sets logits at indices >= 1393 (reserved rows) to
        // -inf; restricting the head to the usable vocabulary is equivalent.
        let vocab = self.card.min(tokenizer::VOCAB_SIZE);
        logits.squeeze(1)?.narrow(1, 0, vocab)?.to_dtype(DType::F32)
    }

    /// Autoregressively generate MIDI tokens for a batch of conditioning
    /// prefixes. `on_step` receives the batch's tokens after every step
    /// (including EOS and post-EOS filler, exactly like the reference
    /// streaming loop). Returns each row's tokens truncated at EOS.
    pub fn generate(
        &mut self,
        prefix: &Tensor,
        opts: &GenerateOptions,
        mut on_step: impl FnMut(&[u32]) -> Result<()>,
    ) -> Result<Vec<Vec<u32>>> {
        let (b, prefix_len, _dim) = prefix.dims3()?;
        self.transformer
            .reset_state(prefix_len + opts.max_gen_len + 1);

        let mut logits_processor = opts
            .sampling
            .as_ref()
            .map(|sampling| LogitsProcessor::from_sampling(opts.seed, sampling.clone()));

        let bos = Tensor::full(self.initial_token_id(), (b, 1), &self.device)?;
        let bos = self.emb.forward(&bos)?;
        let mut input = Tensor::cat(&[prefix.clone(), bos], 1)?;

        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut done = vec![false; b];
        for _step in 0..opts.max_gen_len {
            if done.iter().all(|&d| d) {
                break;
            }
            let logits = self.forward_embeds(&input)?;
            // One device→host transfer per step, not one per batch row.
            let tokens = match logits_processor.as_mut() {
                None => logits.argmax(D::Minus1)?.to_vec1::<u32>()?,
                Some(lp) => {
                    let logits = logits.to_device(&Device::Cpu)?;
                    (0..b)
                        .map(|row| lp.sample(&logits.narrow(0, row, 1)?.squeeze(0)?))
                        .collect::<Result<Vec<_>>>()?
                }
            };
            on_step(&tokens)?;
            for row in 0..b {
                if !done[row] {
                    if tokens[row] == tokenizer::EOS_ID {
                        done[row] = true;
                    } else {
                        rows[row].push(tokens[row]);
                    }
                }
            }
            let next = Tensor::from_vec(tokens, (b, 1), &self.device)?;
            input = self.emb.forward(&next)?;
        }
        Ok(rows)
    }
}
