//! MusicGen text-conditioned music decoder and EnCodec pipeline.
//!
//! Architecture and interleaved codebook delay pattern follow Meta's MusicGen
//! implementation in Transformers. Inference uses causal self-attention caches
//! and independent, cached cross-attention projections of the text condition.
use crate::models::{encodec, t5};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Activation, Embedding, LayerNorm, Linear,
    VarBuilder,
};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    pub activation_function: Activation,
    pub hidden_size: usize,
    pub scale_embedding: bool,
    pub num_codebooks: usize,
    #[serde(default = "mono")]
    pub audio_channels: usize,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: Option<usize>,
    pub cross_attention_hidden_size: Option<usize>,
}
fn mono() -> usize {
    1
}
impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 24,
            ffn_dim: 4096,
            num_attention_heads: 16,
            activation_function: Activation::Gelu,
            hidden_size: 1024,
            scale_embedding: false,
            num_codebooks: 4,
            audio_channels: 1,
            pad_token_id: 2048,
            bos_token_id: 2048,
            eos_token_id: None,
            cross_attention_hidden_size: None,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0
            || self.hidden_size < 4
            || self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.num_hidden_layers == 0
            || self.ffn_dim == 0
            || self.max_position_embeddings == 0
            || ![1, 2].contains(&self.audio_channels)
            || self.num_codebooks == 0
            || !self.num_codebooks.is_multiple_of(self.audio_channels)
            || self.pad_token_id != self.vocab_size
            || self.bos_token_id != self.pad_token_id
            || self.eos_token_id.is_some()
            || self
                .cross_attention_hidden_size
                .is_some_and(|h| h != self.hidden_size)
        {
            candle::bail!("invalid or unsupported MusicGen decoder configuration");
        }
        Ok(())
    }
}

fn positions(length: usize, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let half = dim / 2;
    let scale = 10000f64.ln() / (half - 1) as f64;
    let values: Vec<f32> = (0..length)
        .flat_map(|pos| {
            let angles: Vec<_> = (0..half)
                .map(|i| pos as f64 * (-(i as f64) * scale).exp())
                .collect();
            let mut row: Vec<_> = angles
                .iter()
                .map(|v| v.cos() as f32)
                .chain(angles.iter().map(|v| v.sin() as f32))
                .collect();
            if !dim.is_multiple_of(2) {
                row.push(0.);
            }
            row
        })
        .collect();
    Tensor::from_vec(values, (length, dim), device)?.to_dtype(dtype)
}
#[derive(Debug)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    dim: usize,
    cache: Option<(Tensor, Tensor)>,
}
impl Attention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q: linear_no_bias(h, h, vb.pp("q_proj"))?,
            k: linear_no_bias(h, h, vb.pp("k_proj"))?,
            v: linear_no_bias(h, h, vb.pp("v_proj"))?,
            out: linear_no_bias(h, h, vb.pp("out_proj"))?,
            heads: cfg.num_attention_heads,
            dim: h / cfg.num_attention_heads,
            cache: None,
        })
    }
    fn forward(
        &mut self,
        x: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, h) = x.dims3()?;
        let q = x
            .apply(&self.q)?
            .reshape((b, t, self.heads, self.dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (k, v) = if let (Some(_), Some(cache)) = (context, &self.cache) {
            cache.clone()
        } else {
            let source = context.unwrap_or(x);
            let s = source.dim(1)?;
            let k = source
                .apply(&self.k)?
                .reshape((b, s, self.heads, self.dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = source
                .apply(&self.v)?
                .reshape((b, s, self.heads, self.dim))?
                .transpose(1, 2)?
                .contiguous()?;
            match (&self.cache, context) {
                (Some((pk, pv)), None) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
                _ => (k, v),
            }
        };
        self.cache = Some((k.clone(), v.clone()));
        let weights = (q.matmul(&k.transpose(2, 3)?)? / (self.dim as f64).sqrt())?;
        let weights = match mask {
            Some(mask) => weights.broadcast_add(mask)?,
            None => weights,
        };
        candle_nn::ops::softmax_last_dim(&weights)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, h))?
            .apply(&self.out)
    }
}
#[derive(Debug)]
struct Layer {
    self_attn: Attention,
    cross_attn: Attention,
    self_norm: LayerNorm,
    cross_norm: LayerNorm,
    final_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}
impl Layer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), cfg)?,
            cross_attn: Attention::load(vb.pp("encoder_attn"), cfg)?,
            self_norm: layer_norm(h, 1e-5, vb.pp("self_attn_layer_norm"))?,
            cross_norm: layer_norm(h, 1e-5, vb.pp("encoder_attn_layer_norm"))?,
            final_norm: layer_norm(h, 1e-5, vb.pp("final_layer_norm"))?,
            fc1: linear_no_bias(h, cfg.ffn_dim, vb.pp("fc1"))?,
            fc2: linear_no_bias(cfg.ffn_dim, h, vb.pp("fc2"))?,
            activation: cfg.activation_function,
        })
    }
    fn forward(&mut self, x: &Tensor, mask: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self
            .self_attn
            .forward(&x.apply(&self.self_norm)?, None, Some(mask))?)?;
        let x = match context {
            Some(context) => {
                (&x + self
                    .cross_attn
                    .forward(&x.apply(&self.cross_norm)?, Some(context), None)?)?
            }
            None => x,
        };
        &x + x
            .apply(&self.final_norm)?
            .apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)?
    }
}
#[derive(Debug)]
pub struct MusicgenForCausalLM {
    embeddings: Vec<Embedding>,
    layers: Vec<Layer>,
    norm: LayerNorm,
    heads: Vec<Linear>,
    cfg: Config,
    position: usize,
}
impl MusicgenForCausalLM {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        cfg.validate()?;
        let decoder = vb.pp("model.decoder");
        let h = cfg.hidden_size;
        Ok(Self {
            embeddings: (0..cfg.num_codebooks)
                .map(|i| {
                    embedding(
                        cfg.vocab_size + 1,
                        h,
                        decoder.pp(format!("embed_tokens.{i}")),
                    )
                })
                .collect::<Result<_>>()?,
            layers: (0..cfg.num_hidden_layers)
                .map(|i| Layer::load(decoder.pp(format!("layers.{i}")), cfg))
                .collect::<Result<_>>()?,
            norm: layer_norm(h, 1e-5, decoder.pp("layer_norm"))?,
            heads: (0..cfg.num_codebooks)
                .map(|i| linear_no_bias(h, cfg.vocab_size, vb.pp(format!("lm_heads.{i}"))))
                .collect::<Result<_>>()?,
            cfg: cfg.clone(),
            position: 0,
        })
    }
    /// Incremental forward. Call `reset` when the text condition changes.
    pub fn forward_conditioned(
        &mut self,
        ids: &Tensor,
        context: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (rows, t) = ids.dims2()?;
        let books = self.cfg.num_codebooks;
        if !rows.is_multiple_of(books)
            || rows == 0
            || t == 0
            || self.position + t > self.cfg.max_position_embeddings
        {
            candle::bail!("MusicGen input shape or context budget exceeded");
        }
        let b = rows / books;
        if let Some(context) = context {
            if context.dim(0)? != b || context.dim(2)? != self.cfg.hidden_size {
                candle::bail!("MusicGen text condition shape mismatch");
            }
        }
        let ids = ids.reshape((b, books, t))?;
        let mut x = None;
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let e = ids.i((.., i, ..))?.contiguous()?.apply(embedding)?;
            x = Some(match x {
                None => e,
                Some(x) => (x + e)?,
            });
        }
        let mut x = x.unwrap();
        if self.cfg.scale_embedding {
            x = (x * (self.cfg.hidden_size as f64).sqrt())?;
        }
        let pos = positions(
            self.position + t,
            self.cfg.hidden_size,
            x.device(),
            x.dtype(),
        )?
        .narrow(0, self.position, t)?;
        x = x.broadcast_add(&pos)?;
        let offset = self.position;
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| {
                (0..offset + t).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, t, self.position + t), x.device())?
            .to_dtype(x.dtype())?;
        for layer in &mut self.layers {
            x = layer.forward(&x, &mask, context)?;
        }
        self.position += t;
        let x = x.apply(&self.norm)?;
        Tensor::stack(
            &self
                .heads
                .iter()
                .map(|head| x.apply(head))
                .collect::<Result<Vec<_>>>()?,
            1,
        )?
        .reshape((rows, t, self.cfg.vocab_size))
    }
    /// Backwards-compatible unconditional forward.
    pub fn forward(&mut self, ids: &Tensor) -> Result<Tensor> {
        self.forward_conditioned(ids, None)
    }
    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            layer.self_attn.cache = None;
            layer.cross_attn.cache = None;
        }
    }
}
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct GenConfig {
    #[serde(rename = "decoder")]
    pub musicgen: Config,
    #[serde(rename = "text_encoder")]
    pub t5: t5::Config,
    #[serde(rename = "audio_encoder")]
    pub encodec: encodec::Config,
}
impl GenConfig {
    pub fn small() -> Self {
        Self {
            musicgen: Config::default(),
            t5: t5::Config::musicgen_small(),
            encodec: encodec::Config {
                audio_channels: 1,
                codebook_size: 2048,
                num_filters: 64,
                target_bandwidths: vec![2.2],
                upsampling_ratios: vec![8, 5, 4, 4],
                sampling_rate: 32000,
                use_causal_conv: false,
                use_conv_shortcut: false,
                pad_mode: encodec::PadMode::Reflect,
                ..Default::default()
            },
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.musicgen.validate()?;
        self.encodec.validate()?;
        if self.encodec.audio_channels != 1
            || self.encodec.normalize
            || self.encodec.chunk_length_s.is_some()
            || self.musicgen.num_codebooks / self.musicgen.audio_channels
                != self.encodec.num_quantizers()
            || self.musicgen.vocab_size != self.encodec.codebook_size
        {
            candle::bail!("unsupported MusicGen/EnCodec configuration");
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct MusicgenForConditionalGeneration {
    pub text_encoder: t5::T5EncoderModel,
    pub audio_encoder: encodec::Model,
    pub decoder: MusicgenForCausalLM,
    projection: Option<Linear>,
    cfg: GenConfig,
}
impl MusicgenForConditionalGeneration {
    pub fn load(vb: VarBuilder, cfg: GenConfig) -> Result<Self> {
        cfg.validate()?;
        let projection = if cfg.t5.d_model != cfg.musicgen.hidden_size {
            Some(linear(
                cfg.t5.d_model,
                cfg.musicgen.hidden_size,
                vb.pp("enc_to_dec_proj"),
            )?)
        } else {
            None
        };
        Ok(Self {
            text_encoder: t5::T5EncoderModel::load(vb.pp("text_encoder"), &cfg.t5)?,
            audio_encoder: encodec::Model::new(&cfg.encodec, vb.pp("audio_encoder"))?,
            decoder: MusicgenForCausalLM::load(vb.pp("decoder"), &cfg.musicgen)?,
            projection,
            cfg,
        })
    }
    pub fn config(&self) -> &GenConfig {
        &self.cfg
    }
    pub fn encode_text(&mut self, tokens: &Tensor) -> Result<Tensor> {
        let x = self.text_encoder.forward(tokens)?;
        match &self.projection {
            Some(projection) => x.apply(projection),
            None => Ok(x),
        }
    }
    /// Encode an audio prompt. Stereo channel codebooks are interleaved L/R.
    pub fn encode_audio(&self, audio: &Tensor) -> Result<Tensor> {
        let (_, channels, frames) = audio.dims3()?;
        if channels != self.cfg.musicgen.audio_channels || frames == 0 {
            candle::bail!("MusicGen reference must be nonempty and match the configured channels");
        }
        if channels == 1 {
            return self.audio_encoder.encode(audio);
        }
        let left = self
            .audio_encoder
            .encode(&audio.narrow(1, 0, 1)?.contiguous()?)?;
        let right = self
            .audio_encoder
            .encode(&audio.narrow(1, 1, 1)?.contiguous()?)?;
        let (batch, books, frames) = left.dims3()?;
        Tensor::stack(&[left, right], 2)?.reshape((batch, books * 2, frames))
    }
    pub fn decode_audio(&self, codes: &Tensor) -> Result<Tensor> {
        if self.cfg.musicgen.audio_channels == 1 {
            return self.audio_encoder.decode(codes);
        }
        let left: Vec<_> = (0..self.cfg.musicgen.num_codebooks)
            .step_by(2)
            .map(|i| codes.narrow(1, i, 1))
            .collect::<Result<_>>()?;
        let right: Vec<_> = (1..self.cfg.musicgen.num_codebooks)
            .step_by(2)
            .map(|i| codes.narrow(1, i, 1))
            .collect::<Result<_>>()?;
        Tensor::cat(
            &[
                self.audio_encoder.decode(&Tensor::cat(&left, 1)?)?,
                self.audio_encoder.decode(&Tensor::cat(&right, 1)?)?,
            ],
            1,
        )
    }
    pub fn reset(&mut self) {
        self.text_encoder.clear_kv_cache();
        self.decoder.reset();
    }
}
/// Position in the un-delayed code stream, or None for a fixed BOS/EOS pad.
/// Stereo left/right codebooks share a delay, matching Transformers' repeat_interleave.
pub fn delayed_frame(
    position: usize,
    book: usize,
    channels: usize,
    frames: usize,
) -> Option<usize> {
    let delay = book / channels;
    position
        .checked_sub(delay + 1)
        .filter(|frame| *frame < frames)
}

/// Fixed BOS/EOS padding and delayed reference tokens. Unknown positions are
/// sampled, including after the first codebook's prompt has already ended.
#[derive(Debug, Clone)]
pub struct DelayPattern {
    prefix: Vec<Vec<u32>>,
    frames: usize,
    channels: usize,
    pad: u32,
}
impl DelayPattern {
    pub fn new(
        prefix: Vec<Vec<u32>>,
        new_frames: usize,
        channels: usize,
        pad: u32,
    ) -> Result<Self> {
        if prefix.is_empty()
            || ![1, 2].contains(&channels)
            || !prefix.len().is_multiple_of(channels)
            || new_frames == 0
            || prefix.windows(2).any(|pair| pair[0].len() != pair[1].len())
            || prefix.iter().flatten().any(|code| *code >= pad)
        {
            candle::bail!("invalid MusicGen reference codebooks or continuation budget");
        }
        let frames = prefix[0]
            .len()
            .checked_add(new_frames)
            .ok_or_else(|| candle::Error::Msg("MusicGen continuation length overflow".into()))?;
        frames
            .checked_add(prefix.len() / channels)
            .ok_or_else(|| candle::Error::Msg("MusicGen delayed length overflow".into()))?;
        Ok(Self {
            prefix,
            frames,
            channels,
            pad,
        })
    }
    pub fn codebooks(&self) -> &[Vec<u32>] {
        &self.prefix
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn prefix_frames(&self) -> usize {
        self.prefix[0].len()
    }
    pub fn total_frames(&self) -> usize {
        self.frames
    }
    pub fn delayed_length(&self) -> usize {
        self.frames + self.prefix.len() / self.channels
    }
    pub fn prefill_length(&self) -> usize {
        self.prefix_frames() + 1
    }
    pub fn fixed_token(&self, position: usize, book: usize) -> Result<Option<u32>> {
        let prefix = self
            .prefix
            .get(book)
            .ok_or_else(|| candle::Error::Msg("MusicGen codebook out of range".into()))?;
        Ok(
            match delayed_frame(position, book, self.channels, self.frames) {
                None => Some(self.pad),
                Some(frame) if frame < prefix.len() => Some(prefix[frame]),
                _ => None,
            },
        )
    }
    pub fn prefill(&self) -> Result<Vec<u32>> {
        let mut values = Vec::new();
        for book in 0..self.prefix.len() {
            for position in 0..self.prefill_length() {
                values.push(self.fixed_token(position, book)?.ok_or_else(|| {
                    candle::Error::Msg("unknown token inside MusicGen prompt prefill".into())
                })?);
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/musicgen_decoder.json")).unwrap()
    }
    fn values(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn difference(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }
    #[test]
    fn conditioned_logits_match_transformers_and_incremental_cache() -> Result<()> {
        let f = fixture();
        let device = Device::Cpu;
        let cfg: Config =
            serde_json::from_value(f["config"].clone()).map_err(candle::Error::wrap)?;
        let weights = f["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                let shape = value["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                Ok((
                    name.clone(),
                    Tensor::from_vec(values(&value["values"]), shape, &device)?,
                ))
            })
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        let mut model = MusicgenForCausalLM::load(
            VarBuilder::from_tensors(weights, DType::F32, &device),
            &cfg,
        )?;
        let ids = f["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (4, 4), &device)?;
        let context = Tensor::from_vec(values(&f["context"]), (1, 3, 8), &device)?;
        let expected = Tensor::from_vec(values(&f["expected"]), (4, 4, 8), &device)?;
        let actual = model.forward_conditioned(&ids, Some(&context))?;
        let error = difference(&actual, &expected)?;
        assert!(
            error < 2e-6,
            "Transformers conditioned logits difference={error}"
        );
        model.reset();
        let mut steps = Vec::new();
        for i in 0..4 {
            steps.push(model.forward_conditioned(&ids.narrow(1, i, 1)?, Some(&context))?);
        }
        let incremental = Tensor::cat(&steps, 1)?;
        let error = difference(&actual, &incremental)?;
        assert!(error < 2e-6, "incremental cache logits difference={error}");
        model.reset();
        let unconditional = model.forward_conditioned(&ids, Some(&context.zeros_like()?))?;
        let expected = Tensor::from_vec(values(&f["expected_unconditional"]), (4, 4, 8), &device)?;
        assert!(difference(&unconditional, &expected)? < 2e-6);
        assert!(
            difference(&actual, &unconditional)? > 1e-4,
            "text condition must change logits"
        );
        // A future token must not affect earlier positions in a full prefill.
        model.reset();
        let changed = Tensor::cat(
            &[
                ids.narrow(1, 0, 3)?,
                Tensor::zeros((4, 1), DType::U32, &device)?,
            ],
            1,
        )?;
        let changed = model.forward_conditioned(&changed, Some(&context))?;
        assert!(difference(&actual.narrow(1, 0, 3)?, &changed.narrow(1, 0, 3)?)? < 2e-6);
        Ok(())
    }
    #[test]
    fn mono_and_stereo_codebook_delay_matches_transformers() {
        for case in fixture()["delay"].as_array().unwrap() {
            let n = |name: &str| case[name].as_u64().unwrap() as usize;
            let mut actual = Vec::new();
            for book in 0..n("books") {
                for position in 0..n("length") {
                    actual.push(
                        if delayed_frame(position, book, n("channels"), n("frames")).is_some() {
                            -1i64
                        } else {
                            8
                        },
                    );
                }
            }
            let expected = case["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn official_config_deserializes_and_rejects_invalid_variants() -> Result<()> {
        let mut cfg: GenConfig = serde_json::from_str(include_str!(
            "../../tests/fixtures/musicgen-small-config.json"
        ))
        .map_err(candle::Error::wrap)?;
        cfg.validate()?;
        assert_eq!(cfg.musicgen.hidden_size, 1024);
        assert_eq!(cfg.t5.d_model, 768);
        assert_eq!(cfg.encodec.sampling_rate, 32000);
        cfg.musicgen.hidden_size = 1001;
        assert!(cfg.validate().is_err());
        let mut cfg = GenConfig::small();
        cfg.musicgen.num_codebooks = 8;
        cfg.musicgen.audio_channels = 2;
        cfg.validate()?;
        cfg.musicgen.audio_channels = 3;
        assert!(cfg.validate().is_err());
        Ok(())
    }
    #[test]
    fn reference_delay_masks_preserve_mono_and_stereo_prompt_tails() -> Result<()> {
        let f = fixture();
        let continuation = &f["continuation"];
        for (channels, books, prefix_key, mask_key) in [
            (1, 4, "prefix", "mask"),
            (2, 8, "stereo_prefix", "stereo_mask"),
        ] {
            let prefix = continuation[prefix_key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            let prefix = prefix
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>();
            let pattern = DelayPattern::new(prefix, 4, channels, 8)?;
            assert_eq!(pattern.prefill_length(), 4);
            let mut actual = Vec::new();
            for book in 0..books {
                for position in 0..pattern.delayed_length() {
                    actual.push(
                        pattern
                            .fixed_token(position, book)?
                            .map_or(-1i64, i64::from),
                    );
                }
            }
            let expected = continuation[mask_key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert_eq!(pattern.fixed_token(6, books - 1)?, Some(6));
        }
        assert!(DelayPattern::new(vec![vec![1], vec![]], 4, 1, 8).is_err());
        assert!(DelayPattern::new(vec![vec![8]], 4, 1, 8).is_err());
        Ok(())
    }
}
