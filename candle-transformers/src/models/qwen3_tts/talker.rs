//! TalkerModel: autoregressive semantic token generation for Qwen3-TTS.
//!
//! The talker takes projected text embeddings and generates semantic codec tokens
//! one at a time.  It supports three prefill strategies:
//!
//! * **CustomVoice** – 9 preset speakers selected by a discrete speaker token.
//! * **VoiceDesign** – a natural-language voice description as a text prefix.
//! * **VoiceClone** – a continuous speaker embedding from a reference audio clip.

use std::collections::HashMap;
use std::str::FromStr;

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use super::config::{ParsedModelConfig, Qwen3TTSConfig};
use super::kv_cache::{AnyKVCache, CircularKVCache, KVCache, PreAllocKVCache};
use super::create_causal_mask;

// ── Token constants ──────────────────────────────────────────────────────────

pub mod special_tokens {
    pub const IM_START: u32 = 151644;
    pub const ASSISTANT: u32 = 77091;
    pub const NEWLINE: u32 = 198;
}

pub mod tts_tokens {
    pub const TTS_PAD: u32 = 151671;
    pub const TTS_BOS: u32 = 151672;
    pub const TTS_EOS: u32 = 151673;
}

pub mod codec_tokens {
    pub const CODEC_PAD: u32 = 2148;
    pub const CODEC_BOS: u32 = 2149;
    pub const CODEC_EOS: u32 = 2150;
    pub const CODEC_THINK: u32 = 2154;
    pub const CODEC_THINK_BOS: u32 = 2156;
    pub const CODEC_THINK_EOS: u32 = 2157;
    /// Total codec vocabulary size (semantic + acoustic + control tokens)
    pub const CODEC_VOCAB_SIZE: usize = 3072;
}

// ── Language / Speaker enums ─────────────────────────────────────────────────

/// Target language for TTS generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
}

impl FromStr for Language {
    type Err = candle::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "english" | "en" => Ok(Self::English),
            "chinese" | "zh" => Ok(Self::Chinese),
            "japanese" | "ja" => Ok(Self::Japanese),
            "korean" | "ko" => Ok(Self::Korean),
            "german" | "de" => Ok(Self::German),
            "french" | "fr" => Ok(Self::French),
            "russian" | "ru" => Ok(Self::Russian),
            "portuguese" | "pt" => Ok(Self::Portuguese),
            "spanish" | "es" => Ok(Self::Spanish),
            "italian" | "it" => Ok(Self::Italian),
            _ => candle::bail!("Unknown language: {}", s),
        }
    }
}

impl Language {
    pub fn token_id(self) -> u32 {
        match self {
            Self::Chinese => 2055,
            Self::English => 2050,
            Self::Japanese => 2058,
            Self::Korean => 2064,
            Self::German => 2053,
            Self::French => 2061,
            Self::Russian => 2069,
            Self::Portuguese => 2071,
            Self::Spanish => 2054,
            Self::Italian => 2070,
        }
    }
}

/// Preset speaker for CustomVoice models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Serena,
    Vivian,
    UncleFu,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
    Eric,
    Dylan,
}

impl FromStr for Speaker {
    type Err = candle::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "ryan" => Ok(Self::Ryan),
            "serena" => Ok(Self::Serena),
            "vivian" => Ok(Self::Vivian),
            "aiden" => Ok(Self::Aiden),
            "uncle_fu" | "unclefu" => Ok(Self::UncleFu),
            "ono_anna" | "onoanna" => Ok(Self::OnoAnna),
            "sohee" => Ok(Self::Sohee),
            "eric" => Ok(Self::Eric),
            "dylan" => Ok(Self::Dylan),
            _ => candle::bail!("Unknown speaker: {}", s),
        }
    }
}

impl Speaker {
    pub fn token_id(self) -> u32 {
        match self {
            Self::Serena => 3066,
            Self::Vivian => 3065,
            Self::UncleFu => 3010,
            Self::Ryan => 3061,
            Self::Aiden => 2861,
            Self::OnoAnna => 2873,
            Self::Sohee => 2864,
            Self::Eric => 2875,
            Self::Dylan => 2878,
        }
    }
}

// ── TalkerConfig ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TalkerConfig {
    pub text_vocab_size: usize,
    pub text_embed_dim: usize,
    pub hidden_size: usize,
    pub text_proj_intermediate: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub codec_vocab_size: usize,
    /// `[T, H, W]` MRoPE section, or `None` for standard RoPE.
    pub mrope_section: Option<[usize; 3]>,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            text_vocab_size: 151936,
            text_embed_dim: 2048,
            hidden_size: 1024,
            text_proj_intermediate: 2048,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            codec_vocab_size: 3072,
            mrope_section: Some([24, 20, 20]),
        }
    }
}

impl TalkerConfig {
    /// Build from a parsed HuggingFace config.json.
    pub fn from_parsed(p: &ParsedModelConfig) -> Self {
        Self {
            text_vocab_size: p.talker_text_vocab_size,
            text_embed_dim: p.talker_text_hidden_size,
            hidden_size: p.talker_hidden_size,
            text_proj_intermediate: p.talker_text_hidden_size,
            intermediate_size: p.talker_intermediate_size,
            num_hidden_layers: p.talker_num_hidden_layers,
            num_attention_heads: p.talker_num_attention_heads,
            num_key_value_heads: p.talker_num_key_value_heads,
            head_dim: p.talker_head_dim,
            rms_norm_eps: p.talker_rms_norm_eps,
            rope_theta: p.talker_rope_theta,
            max_position_embeddings: p.talker_max_position_embeddings,
            codec_vocab_size: p.talker_vocab_size,
            mrope_section: p.mrope_section,
        }
    }

    /// Config for 1.7B CustomVoice / VoiceDesign models.
    pub fn large() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 6144,
            ..Self::default()
        }
    }

    fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle::D::Minus1)?;
    let x1 = x.narrow(candle::D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(candle::D::Minus1, d / 2, d / 2)?;
    let cos = cos
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;
    let sin = sin
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;
    let rotated = Tensor::cat(
        &[
            &(x1.mul(&cos)? - x2.mul(&sin)?)?,
            &(x2.mul(&cos)? + x1.mul(&sin)?)?,
        ],
        candle::D::Minus1,
    )?;
    Ok(rotated)
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq).map(|i| i as f32).collect();
        let pos = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;
        let freqs = pos.matmul(&inv_freq.unsqueeze(0)?)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq)?;
        let sin = self.sin.i(offset..offset + seq)?;
        Ok((apply_rope(q, &cos, &sin)?, apply_rope(k, &cos, &sin)?))
    }
}

struct MRoPE {
    inv_freq: Tensor,
    device: Device,
}

impl MRoPE {
    fn new(dim: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv_freq.as_slice(), device)?,
            device: device.clone(),
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let positions: Vec<f32> = (offset..offset + seq).map(|i| i as f32).collect();
        let pos = Tensor::new(positions.as_slice(), &self.device)?;
        let freqs = pos
            .unsqueeze(1)?
            .matmul(&self.inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok((apply_rope(q, &cos, &sin)?, apply_rope(k, &cos, &sin)?))
    }
}

enum RoPEType {
    Standard(RotaryEmbedding),
    Multimodal(MRoPE),
}

impl RoPEType {
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Standard(r) => r.apply(q, k, offset),
            Self::Multimodal(r) => r.apply(q, k, offset),
        }
    }
}

// ── Attention / MLP / DecoderLayer ───────────────────────────────────────────

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(cfg: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_kv_heads();
        let d = cfg.head_dim();
        Ok(Self {
            q_proj: linear_no_bias(h, nq * d, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, nkv * d, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, nkv * d, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nq * d, h, vb.pp("o_proj"))?,
            q_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nq,
            num_kv_heads: nkv,
            head_dim: d,
            scale: 1.0 / (d as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPEType,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, s, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, s, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let (q, k) = rope.apply(&q, &k, offset)?;

        let (k, v, mask) = if let Some(cache) = kv_cache {
            let res = cache.update(&k, &v)?;
            // For circular caches window_start > 0 once the buffer wraps:
            // rebuild the causal mask to correctly reflect absolute positions.
            let mask = if res.window_start > 0 {
                let win_len = res.k.dim(2)?;
                let m = super::kv_cache::create_window_causal_mask(
                    offset, s, res.window_start, win_len, res.k.device())?
                    .to_dtype(res.k.dtype())?;
                Some(m)
            } else {
                mask.cloned()
            };
            (res.k, res.v, mask)
        } else {
            (k, v, mask.cloned())
        };

        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let attn = (q.matmul(&k.transpose(candle::D::Minus2, candle::D::Minus1)?)? * self.scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(&m.to_dtype(attn.dtype())?)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.num_heads / self.num_kv_heads;
        if n == 1 {
            return Ok(x.clone());
        }
        let (b, h, s, d) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, h, n, s, d))?
            .reshape((b, h * n, s, d))
    }
}

struct MLP {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl MLP {
    fn new(cfg: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?)
    }
}

struct DecoderLayer {
    attn: Attention,
    mlp: MLP,
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: MLP::new(cfg, vb.pp("mlp"))?,
            pre_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPEType,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.attn.forward(
            &self.pre_norm.forward(x)?,
            rope,
            mask,
            kv_cache,
            offset,
        )?;
        let h = (h + residual)?;
        let residual = &h;
        let h = self.mlp.forward(&self.post_norm.forward(&h)?)?;
        Ok((h + residual)?)
    }
}

// ── Text projection ──────────────────────────────────────────────────────────

struct TextProjection {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl TextProjection {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(
                cfg.text_embed_dim,
                cfg.text_proj_intermediate,
                vb.pp("linear_fc1"),
            )?,
            fc2: candle_nn::linear(
                cfg.text_proj_intermediate,
                cfg.hidden_size,
                vb.pp("linear_fc2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2
            .forward(&candle_nn::ops::silu(&self.fc1.forward(x)?)?)
    }
}

// ── TalkerModel ──────────────────────────────────────────────────────────────

/// Autoregressive semantic token generator.
pub struct TalkerModel {
    text_embedding: Embedding,
    text_projection: TextProjection,
    codec_embedding: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codec_head: Linear,
    rope: RoPEType,
    config: TalkerConfig,
    device: Device,
}

impl TalkerModel {
    /// Load from a flat weight map (keys prefixed with `"talker."`).
    pub fn from_weights(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let norm_w = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| candle::Error::Msg("Missing talker.model.norm.weight".into()))?;
        let hidden = norm_w.dim(0)?;
        let config = if hidden == 2048 {
            TalkerConfig::large()
        } else {
            TalkerConfig::default()
        };
        Self::from_weights_with_config(weights, config, device)
    }

    /// Load with an explicit config and F32 dtype.
    pub fn from_weights_with_config(
        weights: &HashMap<String, Tensor>,
        config: TalkerConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::from_weights_dtype(weights, config, device, DType::F32)
    }

    /// Load with explicit config and dtype (use BF16 on GPU for speed).
    pub fn from_weights_dtype(
        weights: &HashMap<String, Tensor>,
        config: TalkerConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_tensors(weights.clone(), dtype, device);
        let talker = vb.pp("talker");
        let model = talker.pp("model");
        let layer_cfg = config.to_layer_config();

        let text_embedding = embedding(
            config.text_vocab_size,
            config.text_embed_dim,
            model.pp("text_embedding"),
        )?;
        let text_projection = TextProjection::new(&config, talker.pp("text_projection"))?;
        let codec_embedding = embedding(
            config.codec_vocab_size,
            config.hidden_size,
            model.pp("codec_embedding"),
        )?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, model.pp("norm"))?;
        let codec_head = linear_no_bias(
            config.hidden_size,
            config.codec_vocab_size,
            talker.pp("codec_head"),
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| DecoderLayer::new(&layer_cfg, model.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let rope = if let Some(section) = config.mrope_section {
            let _ = section;
            RoPEType::Multimodal(MRoPE::new(config.head_dim, config.rope_theta, device)?)
        } else {
            RoPEType::Standard(RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                device,
            )?)
        };

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            rope,
            config,
            device: device.clone(),
        })
    }

    // ── Prefill variants ────────────────────────────────────────────────────

    /// Prefill for **CustomVoice**: text + discrete speaker token.
    ///
    /// Returns `(hidden_states [1, S, H], logits [1, 1, V])`.
    pub fn prefill_custom_voice(
        &self,
        text_tokens: &[u32],
        speaker: Speaker,
        language: Language,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;
        let role = self.build_role_prefix()?;
        let codec_ids = Tensor::new(
            &[
                CODEC_THINK,
                CODEC_THINK_BOS,
                language.token_id(),
                CODEC_THINK_EOS,
                speaker.token_id(),
                CODEC_PAD,
                CODEC_BOS,
            ],
            &self.device,
        )?;
        let codec_embed = self.codec_embedding.forward(&codec_ids)?.unsqueeze(0)?;
        let tts_pad = self.build_tts_pad_bos(5)?;
        let codec_hidden = tts_pad.add(&codec_embed.i((.., ..6, ..))?)?;
        let mut hidden = Tensor::cat(&[&role, &codec_hidden], 1)?;
        let codec_bos = codec_embed.i((.., 6..7, ..))?;
        if let Some(first) = self.build_first_text_combined(text_tokens, &codec_bos)? {
            hidden = Tensor::cat(&[&hidden, &first], 1)?;
        }
        self.run_prefill(hidden, kv_caches)
    }

    /// Prefill for **VoiceDesign**: instruct text prefix + role + codec (no speaker token).
    pub fn prefill_voice_design(
        &self,
        text_tokens: &[u32],
        instruct_tokens: &[u32],
        language: Language,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;
        let instruct = self.get_projected_text_embeddings(instruct_tokens)?;
        let role = self.build_role_prefix()?;
        let codec_ids = Tensor::new(
            &[
                CODEC_THINK,
                CODEC_THINK_BOS,
                language.token_id(),
                CODEC_THINK_EOS,
                CODEC_PAD,
                CODEC_BOS,
            ],
            &self.device,
        )?;
        let codec_embed = self.codec_embedding.forward(&codec_ids)?.unsqueeze(0)?;
        let tts_pad = self.build_tts_pad_bos(4)?;
        let codec_hidden = tts_pad.add(&codec_embed.i((.., ..5, ..))?)?;
        let mut hidden = Tensor::cat(&[&instruct, &role, &codec_hidden], 1)?;
        let codec_bos = codec_embed.i((.., 5..6, ..))?;
        if let Some(first) = self.build_first_text_combined(text_tokens, &codec_bos)? {
            hidden = Tensor::cat(&[&hidden, &first], 1)?;
        }
        self.run_prefill(hidden, kv_caches)
    }

    /// Prefill for **VoiceClone**: continuous speaker embedding replaces discrete token.
    pub fn prefill_voice_clone(
        &self,
        text_tokens: &[u32],
        speaker_embed: &Tensor,
        language: Language,
        icl_mode: bool,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;
        let role = self.build_role_prefix()?;
        let prefix_ids = Tensor::new(
            &[CODEC_THINK, CODEC_THINK_BOS, language.token_id(), CODEC_THINK_EOS],
            &self.device,
        )?;
        let prefix_embed = self.codec_embedding.forward(&prefix_ids)?.unsqueeze(0)?;
        let speaker = speaker_embed.reshape((1, 1, self.config.hidden_size))?;
        let suffix_ids = Tensor::new(&[CODEC_PAD, CODEC_BOS], &self.device)?;
        let suffix_embed = self.codec_embedding.forward(&suffix_ids)?.unsqueeze(0)?;
        let codec_embed = Tensor::cat(&[&prefix_embed, &speaker, &suffix_embed], 1)?;
        let tts_pad = self.build_tts_pad_bos(5)?;
        let codec_hidden = tts_pad.add(&codec_embed.i((.., ..6, ..))?)?;
        let mut hidden = Tensor::cat(&[&role, &codec_hidden], 1)?;
        if !icl_mode {
            let codec_bos = codec_embed.i((.., 6..7, ..))?;
            if let Some(first) = self.build_first_text_combined(text_tokens, &codec_bos)? {
                hidden = Tensor::cat(&[&hidden, &first], 1)?;
            }
        }
        self.run_prefill(hidden, kv_caches)
    }

    // ── Generation step ─────────────────────────────────────────────────────

    /// Single autoregressive step with a pre-built embedding input.
    ///
    /// Returns `(hidden [1, 1, H], logits [1, 1, V])`.
    pub fn generate_step_with_embed(
        &self,
        input_embed: &Tensor,
        kv_caches: &mut [AnyKVCache],
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut h = input_embed.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.rope, None, Some(&mut kv_caches[i]), offset)?;
        }
        h = self.norm.forward(&h)?;
        let logits = self.codec_head.forward(&h)?;
        Ok((h, logits))
    }

    // ── Embedding helpers ───────────────────────────────────────────────────

    pub fn get_tts_pad_embed(&self) -> Result<Tensor> {
        self.project_special(tts_tokens::TTS_PAD)
    }

    pub fn get_tts_eos_embed(&self) -> Result<Tensor> {
        self.project_special(tts_tokens::TTS_EOS)
    }

    pub fn get_codec_embedding(&self, token: u32) -> Result<Tensor> {
        let t = Tensor::new(&[token], &self.device)?;
        Ok(self.codec_embedding.forward(&t)?.unsqueeze(0)?)
    }

    pub fn get_codec_embedding_from_tensor(&self, token: &Tensor) -> Result<Tensor> {
        let t = token.flatten_all()?;
        Ok(self.codec_embedding.forward(&t)?.unsqueeze(0)?)
    }

    pub fn get_codec_embedding_batch(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.codec_embedding.forward(token_ids)?.unsqueeze(0)?)
    }

    pub fn get_projected_text_embeddings(&self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            let dtype = self.text_embedding.embeddings().dtype();
            return Tensor::zeros((1, 0, self.config.hidden_size), dtype, &self.device);
        }
        let ids_t = Tensor::new(ids, &self.device)?;
        let e = self.text_embedding.forward(&ids_t)?.unsqueeze(0)?;
        self.text_projection.forward(&e)
    }

    // ── Public accessors ────────────────────────────────────────────────────

    pub fn config(&self) -> &TalkerConfig {
        &self.config
    }

    pub fn new_kv_caches(&self, max_seq: usize) -> Vec<AnyKVCache> {
        let use_prealloc = (self.device.is_cuda() || self.device.is_metal()) && max_seq > 0;
        let dtype = self.codec_head.weight().dtype();
        (0..self.config.num_hidden_layers)
            .map(|_| {
                if use_prealloc {
                    PreAllocKVCache::new(
                        1,
                        self.config.num_key_value_heads,
                        max_seq,
                        self.config.head_dim,
                        dtype,
                        &self.device,
                    )
                    .map(AnyKVCache::PreAlloc)
                    .unwrap_or_else(|_| AnyKVCache::Concat(KVCache::new()))
                } else {
                    AnyKVCache::Concat(KVCache::new())
                }
            })
            .collect()
    }

    /// Allocate sliding-window KV caches for every transformer layer.
    ///
    /// Equivalent to `new_kv_caches` but each layer uses a [`CircularKVCache`]
    /// with a fixed `window` capacity instead of growing unboundedly.  Tokens
    /// older than `window` positions are evicted from attention, reducing
    /// memory from O(N) to O(window).
    ///
    /// Pass `prefill_len + window` as the effective context when the model has
    /// a fixed-length prefix (role / instruct embeddings) that must never be
    /// evicted.  For pure generation with no persistent prefix, `window` is
    /// the number of recent frames to retain.
    pub fn new_kv_caches_windowed(&self, window: usize) -> Vec<AnyKVCache> {
        let dtype = self.codec_head.weight().dtype();
        (0..self.config.num_hidden_layers)
            .map(|_| {
                CircularKVCache::new(
                    1,
                    self.config.num_key_value_heads,
                    window,
                    self.config.head_dim,
                    dtype,
                    &self.device,
                )
                .map(AnyKVCache::Circular)
                .unwrap_or_else(|_| AnyKVCache::Concat(KVCache::new()))
            })
            .collect()
    }

    pub fn layers_iter(&self) -> impl Iterator<Item = &DecoderLayer> {
        self.layers.iter()
    }

    pub fn rope(&self) -> &RoPEType {
        &self.rope
    }

    pub fn apply_norm(&self, h: &Tensor) -> Result<Tensor> {
        self.norm.forward(h)
    }

    pub fn apply_codec_head(&self, h: &Tensor) -> Result<Tensor> {
        self.codec_head.forward(h)
    }

    /// Build summed codec embeddings from reference codes for ICL voice cloning.
    ///
    /// `ref_codes` — shape `[T_frames, 16]` of u32 token ids (group 0 = semantic,
    /// groups 1–15 = acoustic). Each group's embedding is looked up from the
    /// talker's codec embedding table and summed into a `[1, T_frames, hidden]`
    /// tensor suitable for passing to [`build_icl_prompt`].
    ///
    /// The acoustic groups (1–15) must be embedded by the caller using
    /// `CodePredictor::embed_codes_for_group` and summed in — this function
    /// only handles the semantic group (group 0) that lives in the talker.
    /// See `sum_ref_codec_embeddings_full` in the example for the full sum.
    pub fn embed_ref_semantic_codes(&self, ref_codes: &Tensor) -> Result<Tensor> {
        // ref_codes: [T_frames, 16] u32
        let semantic = ref_codes.i((.., 0))?.contiguous()?; // [T_frames]
        self.get_codec_embedding_batch(&semantic) // [1, T_frames, hidden]
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn build_role_prefix(&self) -> Result<Tensor> {
        use special_tokens::*;
        let ids = Tensor::new(&[IM_START, ASSISTANT, NEWLINE], &self.device)?;
        let e = self.text_embedding.forward(&ids)?.unsqueeze(0)?;
        self.text_projection.forward(&e)
    }

    fn build_tts_pad_bos(&self, pad_count: usize) -> Result<Tensor> {
        use tts_tokens::*;
        let pad_id = Tensor::new(&[TTS_PAD], &self.device)?;
        let pad_proj = self
            .text_projection
            .forward(&self.text_embedding.forward(&pad_id)?.unsqueeze(0)?)?;
        let bos_id = Tensor::new(&[TTS_BOS], &self.device)?;
        let bos_proj = self
            .text_projection
            .forward(&self.text_embedding.forward(&bos_id)?.unsqueeze(0)?)?;
        let pad_exp = pad_proj.broadcast_as((1, pad_count, self.config.hidden_size))?;
        Tensor::cat(&[&pad_exp, &bos_proj], 1)
    }

    fn build_first_text_combined(
        &self,
        text_tokens: &[u32],
        codec_bos: &Tensor,
    ) -> Result<Option<Tensor>> {
        if text_tokens.is_empty() {
            return Ok(None);
        }
        let id = Tensor::new(&[text_tokens[0]], &self.device)?;
        let e = self
            .text_projection
            .forward(&self.text_embedding.forward(&id)?.unsqueeze(0)?)?;
        Ok(Some(e.add(codec_bos)?))
    }

    fn project_special(&self, token: u32) -> Result<Tensor> {
        let id = Tensor::new(&[token], &self.device)?;
        let e = self.text_embedding.forward(&id)?.unsqueeze(0)?;
        self.text_projection.forward(&e)
    }

    /// Build ICL (in-context learning) prompt for voice cloning.
    ///
    /// Aligns reference text + codec embeddings element-wise (streaming mode).
    /// The longer sequence is either truncated (codec) or the remainder becomes
    /// the trailing text embedding returned alongside the ICL embed.
    ///
    /// Returns `(icl_embed [1, T, H], trailing_text_embed [1, T_trail, H])`.
    pub fn build_icl_prompt(
        &self,
        target_text_ids: &[u32],
        ref_text_ids: &[u32],
        ref_codec_embeds: &Tensor, // [1, T_ref, hidden]
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;
        use tts_tokens::*;

        // All text: [ref_text, target_text, tts_eos] → projected
        let mut all_text_ids: Vec<u32> =
            Vec::with_capacity(ref_text_ids.len() + target_text_ids.len() + 1);
        all_text_ids.extend_from_slice(ref_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        all_text_ids.push(TTS_EOS);
        let text_embed = self.get_projected_text_embeddings(&all_text_ids)?; // [1, N_text, H]
        let n_text = text_embed.dim(1)?;

        // Codec: prepend codec_bos then ref codec frames
        let bos_id = Tensor::new(&[CODEC_BOS], &self.device)?;
        let bos_embed = self.codec_embedding.forward(&bos_id)?.unsqueeze(0)?;
        let codec_embed = Tensor::cat(&[&bos_embed, ref_codec_embeds], 1)?; // [1, T_ref+1, H]
        let n_codec = codec_embed.dim(1)?;

        let tts_pad_embed = self.get_tts_pad_embed()?;

        // Streaming: element-wise overlay — text and codec are zipped.
        if n_text > n_codec {
            let text_head = text_embed.i((.., ..n_codec, ..))?;
            let icl_embed = text_head.add(&codec_embed)?;
            let trailing = text_embed.i((.., n_codec.., ..))?;
            Ok((icl_embed, trailing))
        } else {
            let pad_count = n_codec - n_text;
            let padded_text = if pad_count > 0 {
                let pad =
                    tts_pad_embed.broadcast_as((1, pad_count, self.config.hidden_size))?;
                Tensor::cat(&[&text_embed, &pad], 1)?
            } else {
                text_embed
            };
            let icl_embed = padded_text.add(&codec_embed)?;
            Ok((icl_embed, tts_pad_embed))
        }
    }

    /// Run the transformer layers over an already-built ICL embedding, updating
    /// KV caches from `offset`, and return the last hidden state + codec logits.
    ///
    /// This is the in-context-learning prefill step for voice cloning: the ICL
    /// embed produced by [`build_icl_prompt`] is fed through the model so that
    /// the reference audio and text are available in the KV cache for generation.
    pub fn prefill_icl(
        &self,
        icl_embed: &Tensor,
        kv_caches: &mut [AnyKVCache],
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let icl_len = icl_embed.dim(1)?;
        let mask = create_causal_mask(icl_len, offset, &self.device)?;
        let mut hidden = icl_embed.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                &self.rope,
                Some(&mask.to_dtype(hidden.dtype())?),
                Some(&mut kv_caches[i]),
                offset,
            )?;
        }
        hidden = self.norm.forward(&hidden)?;
        let last = hidden.i((.., icl_len - 1..icl_len, ..))?;
        let logits = self.codec_head.forward(&last)?;
        Ok((last, logits))
    }

    pub fn run_prefill(
        &self,
        mut hidden: Tensor,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        let s = hidden.dim(1)?;
        let mask = create_causal_mask(s, 0, &self.device)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden =
                layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }
        hidden = self.norm.forward(&hidden)?;
        let last = hidden.i((.., s - 1..s, ..))?;
        let logits = self.codec_head.forward(&last)?;
        Ok((hidden, logits))
    }
}
