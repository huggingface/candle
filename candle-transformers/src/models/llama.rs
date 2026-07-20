//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::lora::LoraLinear;
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::{collections::HashMap, f32::consts::PI};

/// Rank, scaling and target projections for a LoRA adapter to inject into a
/// [`Llama`] model at load time. `target_modules` is matched against the
/// projection names `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
/// `up_proj` and `down_proj`.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    fn wants(&self, module: &str) -> bool {
        self.target_modules.iter().any(|m| m == module)
    }
}

type LoraSpec<'a> = (String, VarBuilder<'a>, LoraConfig);

/// Prefix under which a standard PEFT `PeftModel` checkpoint stores its base
/// model's weights, e.g. `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`.
/// This is the default root [`LlamaLoadConfig::with_lora_adapter`] resolves
/// LoRA tensor paths against.
pub const PEFT_ADAPTER_PREFIX: &str = "base_model.model";

/// Configuration used by [`Llama::load_with_config`] to inject one or more
/// named LoRA adapters into the base model's attention and MLP projections.
/// Building a model with an empty (default) config is equivalent to
/// [`Llama::load`].
#[derive(Clone, Default)]
pub struct LlamaLoadConfig<'a> {
    lora_adapters: Vec<LoraSpec<'a>>,
}

impl<'a> LlamaLoadConfig<'a> {
    /// Registers a named LoRA adapter, loaded from `vb`, to be injected into
    /// the projections listed in `config.target_modules`. `vb` is assumed to
    /// be rooted at the standard PEFT checkpoint layout, i.e. tensors live
    /// under [`PEFT_ADAPTER_PREFIX`] `.model.layers.{i}.self_attn.q_proj` (and
    /// so on), each holding `lora_A.weight` / `lora_B.weight`. Use
    /// [`LlamaLoadConfig::with_lora_adapter_prefixed`] if the checkpoint was
    /// saved under a different root, or if `vb` is already scoped past that
    /// prefix.
    pub fn with_lora_adapter(self, name: &str, vb: VarBuilder<'a>, config: LoraConfig) -> Self {
        self.with_lora_adapter_prefixed(name, vb, config, PEFT_ADAPTER_PREFIX)
    }

    /// Like [`LlamaLoadConfig::with_lora_adapter`], but resolves LoRA tensors
    /// under `prefix` instead of the standard [`PEFT_ADAPTER_PREFIX`]. Pass an
    /// empty string if `vb` is already scoped to the model root (i.e. tensors
    /// live directly under `model.layers.{i}...`).
    pub fn with_lora_adapter_prefixed(
        mut self,
        name: &str,
        vb: VarBuilder<'a>,
        config: LoraConfig,
        prefix: &str,
    ) -> Self {
        let vb = if prefix.is_empty() { vb } else { vb.pp(prefix) };
        self.lora_adapters.push((name.to_string(), vb, config));
        self
    }
}

/// A linear projection that is either a plain frozen layer or one augmented
/// with LoRA adapters, depending on whether the loader was asked to inject
/// adapters into it.
#[derive(Debug, Clone)]
enum Proj {
    Plain(Linear),
    Lora(LoraLinear),
}

impl Proj {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(x),
            Self::Lora(l) => l.forward(x),
        }
    }

    /// Forward pass with an optional per-row adapter assignment (see
    /// [`LoraLinear::forward_with_adapters`]). Adapters only live on the
    /// projections their config targeted, so rows selecting an adapter that
    /// is not registered on this projection fall back to the base layer here.
    fn forward_with_adapters(
        &self,
        x: &Tensor,
        assignments: Option<&[Option<&str>]>,
    ) -> Result<Tensor> {
        let (Self::Lora(l), Some(assignments)) = (self, assignments) else {
            return self.forward(x);
        };
        let names = l.adapter_names();
        let mapped: Vec<Option<&str>> = assignments
            .iter()
            .map(|a| (*a).filter(|name| names.contains(name)))
            .collect();
        l.forward_with_adapters(x, &mapped)
    }

    fn has_adapter(&self, name: &str) -> bool {
        match self {
            Self::Plain(_) => false,
            Self::Lora(l) => l.adapter_names().contains(&name),
        }
    }

    /// Activates `name` on this projection if it carries an adapter under
    /// that name, deactivates any adapter when `name` is `None`, and is a
    /// no-op otherwise (this projection simply wasn't targeted by `name`).
    fn set_active_adapter(&mut self, name: Option<&str>) -> Result<()> {
        let Self::Lora(l) = self else {
            return Ok(());
        };
        match name {
            None => l.set_active_adapter(None),
            Some(name) if l.adapter_names().contains(&name) => l.set_active_adapter(Some(name)),
            Some(_) => Ok(()),
        }
    }
}

fn load_projection(
    module: &str,
    size_in: usize,
    size_out: usize,
    vb: VarBuilder,
    lora_adapters: &[LoraSpec],
) -> Result<Proj> {
    let matching: Vec<_> = lora_adapters
        .iter()
        .filter(|(_, _, cfg)| cfg.wants(module))
        .collect();
    if matching.is_empty() {
        return Ok(Proj::Plain(linear(size_in, size_out, vb)?));
    }
    let base = candle_nn::linear_no_bias(size_in, size_out, vb)?;
    let mut lora = LoraLinear::new(base);
    for (name, adapter_vb, cfg) in matching {
        lora.load_adapter(name, adapter_vb.pp(module), cfg.rank, cfg.alpha)?;
    }
    Ok(Proj::Lora(lora))
}

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        adapters: Option<&[Option<&str>]>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward_with_adapters(x, adapters)?;
        let k = self.k_proj.forward_with_adapters(x, adapters)?;
        let v = self.v_proj.forward_with_adapters(x, adapters)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward_with_adapters(&y, adapters)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
    }

    fn has_adapter(&self, name: &str) -> bool {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
            .into_iter()
            .any(|p| p.has_adapter(name))
    }

    fn set_active_adapter(&mut self, name: Option<&str>) -> Result<()> {
        for p in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ] {
            p.set_active_adapter(name)?;
        }
        Ok(())
    }

    fn load(vb: VarBuilder, cfg: &Config, lora_adapters: &[LoraSpec]) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let lora_adapters: Vec<_> = lora_adapters
            .iter()
            .map(|(name, vb, cfg)| (name.clone(), vb.pp("self_attn"), cfg.clone()))
            .collect();
        let q_proj = load_projection("q_proj", size_in, size_q, vb.pp("q_proj"), &lora_adapters)?;
        let k_proj = load_projection("k_proj", size_in, size_kv, vb.pp("k_proj"), &lora_adapters)?;
        let v_proj = load_projection("v_proj", size_in, size_kv, vb.pp("v_proj"), &lora_adapters)?;
        let o_proj = load_projection("o_proj", size_q, size_in, vb.pp("o_proj"), &lora_adapters)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Proj,
    c_fc2: Proj,
    c_proj: Proj,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor, adapters: Option<&[Option<&str>]>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward_with_adapters(x, adapters)?)?
            * self.c_fc2.forward_with_adapters(x, adapters)?)?;
        self.c_proj.forward_with_adapters(&x, adapters)
    }

    fn has_adapter(&self, name: &str) -> bool {
        [&self.c_fc1, &self.c_fc2, &self.c_proj]
            .into_iter()
            .any(|p| p.has_adapter(name))
    }

    fn set_active_adapter(&mut self, name: Option<&str>) -> Result<()> {
        for p in [&mut self.c_fc1, &mut self.c_fc2, &mut self.c_proj] {
            p.set_active_adapter(name)?;
        }
        Ok(())
    }

    fn load(vb: VarBuilder, cfg: &Config, lora_adapters: &[LoraSpec]) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let lora_adapters: Vec<_> = lora_adapters
            .iter()
            .map(|(name, vb, cfg)| (name.clone(), vb.pp("mlp"), cfg.clone()))
            .collect();
        let c_fc1 = load_projection(
            "gate_proj",
            h_size,
            i_size,
            vb.pp("gate_proj"),
            &lora_adapters,
        )?;
        let c_fc2 = load_projection("up_proj", h_size, i_size, vb.pp("up_proj"), &lora_adapters)?;
        let c_proj = load_projection(
            "down_proj",
            i_size,
            h_size,
            vb.pp("down_proj"),
            &lora_adapters,
        )?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        adapters: Option<&[Option<&str>]>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, index_pos, block_idx, cache, adapters)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?, adapters)? + residual)?;
        Ok(x)
    }

    fn has_adapter(&self, name: &str) -> bool {
        self.attn.has_adapter(name) || self.mlp.has_adapter(name)
    }

    fn set_active_adapter(&mut self, name: Option<&str>) -> Result<()> {
        self.attn.set_active_adapter(name)?;
        self.mlp.set_active_adapter(name)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        layer_idx: usize,
        lora_adapters: &[LoraSpec],
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let block_lora: Vec<_> = lora_adapters
            .iter()
            .map(|(name, vb, cfg)| {
                (
                    name.clone(),
                    vb.pp(format!("model.layers.{layer_idx}")),
                    cfg.clone(),
                )
            })
            .collect();
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, &block_lora)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, &block_lora)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    // required by LLaVA
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }
    // required by LLaVA
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, None)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, None)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Like [`Llama::forward`], but every sequence in the batch selects its
    /// own LoRA adapter (S-LoRA / Punica style heterogeneous batching).
    ///
    /// `adapter_assignments` holds one entry per batch row: `Some(name)` to
    /// route that sequence through a registered adapter, `None` to run it on
    /// the frozen base weights only. The adapter deltas are applied through
    /// [`LoraLinear::forward_with_adapters`]' batched gather + matmul path
    /// rather than one forward per adapter group, and the result matches
    /// running each sequence separately with its own adapter. The currently
    /// active adapter (see [`Llama::set_active_adapter`]) is ignored by this
    /// method.
    pub fn forward_with_adapters(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        adapter_assignments: &[Option<&str>],
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        if adapter_assignments.len() != b_sz {
            candle::bail!(
                "forward_with_adapters: {} adapter assignments for a batch of {b_sz} sequences",
                adapter_assignments.len()
            )
        }
        // Projections silently fall back to the base weights for adapter
        // names they don't carry, so catch typos at the model level where
        // the full adapter registry is known.
        for name in adapter_assignments.iter().flatten() {
            if !self.blocks.iter().any(|b| b.has_adapter(name)) {
                candle::bail!("no such LoRA adapter: {name}")
            }
        }
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, Some(adapter_assignments))?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Self::load_with_config(vb, cfg, LlamaLoadConfig::default())
    }

    /// Like [`Llama::load`], but additionally injects the LoRA adapters
    /// listed in `load_config` into the matching attention/MLP projections.
    /// With no adapters registered this behaves exactly like `Llama::load`.
    /// Use [`Llama::set_active_adapter`] to select which (if any) of the
    /// registered adapters applies to subsequent forward passes.
    pub fn load_with_config(
        vb: VarBuilder,
        cfg: &Config,
        load_config: LlamaLoadConfig,
    ) -> Result<Self> {
        let lora_adapters = load_config.lora_adapters;
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg, i, &lora_adapters))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    /// Selects the LoRA adapter that subsequent calls to `forward` /
    /// `forward_input_embed` should use, or falls back to the frozen base
    /// weights when `name` is `None`. Switching adapters does not reload or
    /// duplicate the base model weights.
    pub fn set_active_adapter(&mut self, name: Option<&str>) -> Result<()> {
        if let Some(name) = name {
            if !self.blocks.iter().any(|b| b.has_adapter(name)) {
                candle::bail!("no such LoRA adapter: {name}")
            }
        }
        for block in &mut self.blocks {
            block.set_active_adapter(name)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Config {
        Config {
            hidden_size: 4,
            intermediate_size: 4,
            vocab_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 16,
            tie_word_embeddings: false,
        }
    }

    fn base_weights(cfg: &Config, dev: &Device) -> HashMap<String, Tensor> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let mut ts = HashMap::new();
        ts.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::ones((v, h), DType::F32, dev).unwrap(),
        );
        ts.insert(
            "model.norm.weight".to_string(),
            Tensor::ones(h, DType::F32, dev).unwrap(),
        );
        ts.insert(
            "lm_head.weight".to_string(),
            (Tensor::ones((v, h), DType::F32, dev).unwrap() * 0.1).unwrap(),
        );
        for i_layer in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i_layer}");
            ts.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::ones(h, DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::ones(h, DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::zeros((h, h), DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::zeros((h, h), DType::F32, dev).unwrap(),
            );
            // Non-zero v_proj so the attention output (and thus o_proj's
            // input) is non-zero, letting a LoRA adapter on o_proj have an
            // observable effect even with a single-token sequence.
            ts.insert(
                format!("{p}.self_attn.v_proj.weight"),
                (Tensor::ones((h, h), DType::F32, dev).unwrap() * 0.1).unwrap(),
            );
            ts.insert(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::zeros((h, h), DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.mlp.gate_proj.weight"),
                Tensor::zeros((i, h), DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.mlp.up_proj.weight"),
                Tensor::zeros((i, h), DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.mlp.down_proj.weight"),
                Tensor::zeros((h, i), DType::F32, dev).unwrap(),
            );
        }
        ts
    }

    fn lora_adapter_weights(
        cfg: &Config,
        rank: usize,
        dev: &Device,
        prefix: &str,
    ) -> HashMap<String, Tensor> {
        lora_adapter_weights_valued(cfg, rank, dev, prefix, 1.0)
    }

    fn lora_adapter_weights_valued(
        cfg: &Config,
        rank: usize,
        dev: &Device,
        prefix: &str,
        value: f64,
    ) -> HashMap<String, Tensor> {
        let h = cfg.hidden_size;
        let mut ts = HashMap::new();
        for i_layer in 0..cfg.num_hidden_layers {
            let p = if prefix.is_empty() {
                format!("model.layers.{i_layer}.self_attn.o_proj")
            } else {
                format!("{prefix}.model.layers.{i_layer}.self_attn.o_proj")
            };
            ts.insert(
                format!("{p}.lora_A.weight"),
                (Tensor::ones((rank, h), DType::F32, dev).unwrap() * value).unwrap(),
            );
            ts.insert(
                format!("{p}.lora_B.weight"),
                (Tensor::ones((h, rank), DType::F32, dev).unwrap() * value).unwrap(),
            );
        }
        ts
    }

    #[test]
    fn load_without_lora_matches_plain_load() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let vb = VarBuilder::from_tensors(base_weights(&cfg, &dev), DType::F32, &dev);
        let mut model = Llama::load(vb, &cfg)?;
        let input = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        // Should run without error and produce the expected output shape.
        let logits = model.forward(&input, 0, &mut cache)?;
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);
        // No adapters were registered, so activating one must fail.
        assert!(model.set_active_adapter(Some("missing")).is_err());
        Ok(())
    }

    #[test]
    fn lora_adapter_changes_output_only_when_active() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let base_vb = VarBuilder::from_tensors(base_weights(&cfg, &dev), DType::F32, &dev);
        // Adapter weights follow the standard PEFT `base_model.model.` layout;
        // `with_lora_adapter` should resolve them without any extra scoping.
        let adapter_vb = VarBuilder::from_tensors(
            lora_adapter_weights(&cfg, 2, &dev, PEFT_ADAPTER_PREFIX),
            DType::F32,
            &dev,
        );
        let load_config = LlamaLoadConfig::default().with_lora_adapter(
            "adapter",
            adapter_vb,
            LoraConfig {
                rank: 2,
                alpha: 4.0,
                target_modules: vec!["o_proj".to_string()],
            },
        );
        let mut model = Llama::load_with_config(base_vb, &cfg, load_config)?;

        let input = Tensor::new(&[[1u32, 2, 3]], &dev)?;

        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let base_logits = model.forward(&input, 0, &mut cache)?.to_vec2::<f32>()?;

        model.set_active_adapter(Some("adapter"))?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let adapter_logits = model.forward(&input, 0, &mut cache)?.to_vec2::<f32>()?;
        assert_ne!(base_logits, adapter_logits);

        model.set_active_adapter(None)?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let back_to_base_logits = model.forward(&input, 0, &mut cache)?.to_vec2::<f32>()?;
        assert_eq!(base_logits, back_to_base_logits);

        assert!(model.set_active_adapter(Some("missing")).is_err());
        Ok(())
    }

    #[test]
    fn lora_adapter_with_custom_prefix() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let base_vb = VarBuilder::from_tensors(base_weights(&cfg, &dev), DType::F32, &dev);
        // Adapter checkpoint saved without any wrapping prefix (tensors live
        // directly under `model.layers.{i}...`), as if already unwrapped from
        // a PeftModel. `with_lora_adapter` alone would look under
        // `base_model.model.` and fail to find these tensors.
        let adapter_vb =
            VarBuilder::from_tensors(lora_adapter_weights(&cfg, 2, &dev, ""), DType::F32, &dev);
        let load_config = LlamaLoadConfig::default().with_lora_adapter_prefixed(
            "adapter",
            adapter_vb,
            LoraConfig {
                rank: 2,
                alpha: 4.0,
                target_modules: vec!["o_proj".to_string()],
            },
            "",
        );
        let mut model = Llama::load_with_config(base_vb, &cfg, load_config)?;

        let input = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let base_logits = model.forward(&input, 0, &mut cache)?.to_vec2::<f32>()?;

        model.set_active_adapter(Some("adapter"))?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let adapter_logits = model.forward(&input, 0, &mut cache)?.to_vec2::<f32>()?;
        assert_ne!(base_logits, adapter_logits);
        Ok(())
    }

    /// A model with two adapters of different ranks and weights on o_proj:
    /// "a" (rank 2, alpha 4) and "b" (rank 1, alpha 3).
    fn multi_adapter_model(cfg: &Config, dev: &Device) -> Result<Llama> {
        let base_vb = VarBuilder::from_tensors(base_weights(cfg, dev), DType::F32, dev);
        let vb_a = VarBuilder::from_tensors(
            lora_adapter_weights_valued(cfg, 2, dev, PEFT_ADAPTER_PREFIX, 1.0),
            DType::F32,
            dev,
        );
        let vb_b = VarBuilder::from_tensors(
            lora_adapter_weights_valued(cfg, 1, dev, PEFT_ADAPTER_PREFIX, 0.5),
            DType::F32,
            dev,
        );
        let target = LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["o_proj".to_string()],
        };
        let load_config = LlamaLoadConfig::default()
            .with_lora_adapter("a", vb_a, target.clone())
            .with_lora_adapter(
                "b",
                vb_b,
                LoraConfig {
                    rank: 1,
                    alpha: 3.0,
                    ..target
                },
            );
        Llama::load_with_config(base_vb, cfg, load_config)
    }

    fn round4(rows: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        rows.into_iter()
            .map(|r| r.into_iter().map(|v| (v * 1e4).round() / 1e4).collect())
            .collect()
    }

    #[test]
    fn forward_with_adapters_matches_per_sequence_execution() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let mut model = multi_adapter_model(&cfg, &dev)?;

        // A heterogeneous batch: two different adapters, a base-only row, and
        // a repeated adapter.
        let input = Tensor::new(&[[1u32, 2, 3], [4, 5, 6], [2, 4, 6], [3, 1, 7]], &dev)?;
        let assignments = [Some("a"), None, Some("b"), Some("a")];

        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let batched = model.forward_with_adapters(&input, 0, &mut cache, &assignments)?;
        let batched = round4(batched.to_vec2::<f32>()?);

        // Reference: run every sequence on its own with its adapter active.
        for (i, assignment) in assignments.iter().enumerate() {
            model.set_active_adapter(*assignment)?;
            let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
            let row = model.forward(&input.narrow(0, i, 1)?, 0, &mut cache)?;
            assert_eq!(round4(row.to_vec2::<f32>()?), vec![batched[i].clone()]);
        }

        // The active adapter must not leak into the batched path.
        model.set_active_adapter(Some("b"))?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let rerun = model.forward_with_adapters(&input, 0, &mut cache, &assignments)?;
        assert_eq!(round4(rerun.to_vec2::<f32>()?), batched);
        Ok(())
    }

    #[test]
    fn forward_with_adapters_all_none_matches_plain_forward() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let model = multi_adapter_model(&cfg, &dev)?;
        let input = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?;

        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let batched = model.forward_with_adapters(&input, 0, &mut cache, &[None, None])?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        let plain = model.forward(&input, 0, &mut cache)?;
        assert_eq!(batched.to_vec2::<f32>()?, plain.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn forward_with_adapters_rejects_bad_assignments() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let model = multi_adapter_model(&cfg, &dev)?;
        let input = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?;

        let mut cache = Cache::new(false, DType::F32, &cfg, &dev)?;
        // Assignment count must match the batch size.
        assert!(model
            .forward_with_adapters(&input, 0, &mut cache, &[Some("a")])
            .is_err());
        // Unknown adapter names are rejected instead of silently running on
        // the base weights.
        assert!(model
            .forward_with_adapters(&input, 0, &mut cache, &[Some("a"), Some("nope")])
            .is_err());
        Ok(())
    }
}
