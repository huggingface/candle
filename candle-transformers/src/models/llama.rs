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
            use_flashinfer_attention: false,
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
    /// Additive, decode-only seam: when set, the decode step (`seq_len == 1`) attends via
    /// `candle_flashinfer_kernels::flashinfer_decode_attention` instead of the dense
    /// matmul+softmax or `flash_attn` path. Prefill (`seq_len > 1`) is unaffected, and a
    /// layer with paged KV storage attached (see `Cache::set_paged_kv`) always takes the
    /// paged path regardless of this flag. No-op unless the `flashinfer-kernels` cargo
    /// feature is enabled. Defaults to `false` everywhere.
    #[cfg_attr(not(feature = "flashinfer-kernels"), allow(dead_code))]
    pub use_flashinfer_attention: bool,
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
            use_flashinfer_attention: false,
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
            use_flashinfer_attention: false,
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

/// Physical KV storage and per-sequence block table for paged attention.
///
/// Constructed and owned by the caller (e.g. an external block allocator/eviction
/// engine such as a vLLM-style scheduler). `key_cache`/`value_cache` must be
/// contiguous: the model writes the newly computed K/V for the current forward
/// pass into the slots designated by `block_table`, then reads the full paged
/// history back through
/// [`candle_flash_attn::flash_attn_varlen_paged_windowed`]. `block_table` and
/// `seqlens_k` are read-only from the model's perspective — the caller owns
/// block allocation, eviction, and keeping them in sync with what it wants
/// written. One `PagedKvCache` covers a single transformer layer; attach one per
/// layer via [`Cache::set_paged_kv`].
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    /// `(num_blocks, page_block_size, num_kv_heads, head_dim)`, contiguous.
    pub key_cache: Tensor,
    /// `(num_blocks, page_block_size, num_kv_heads, head_dim)`, contiguous.
    pub value_cache: Tensor,
    /// `(batch_size, max_blocks)`, physical block ids per sequence.
    pub block_table: Tensor,
    /// `(batch_size + 1,)`, cumulative sequence lengths (including the tokens
    /// about to be written by this forward pass), used to drive the varlen kernel.
    pub seqlens_k: Tensor,
    pub page_block_size: usize,
}

impl PagedKvCache {
    /// Scatters the newly computed K/V (shape `(b_sz, num_kv_heads, seq_len,
    /// head_dim)`) into `key_cache`/`value_cache` at the physical slots that
    /// `block_table` designates for absolute positions
    /// `index_pos..index_pos+seq_len`.
    fn write_new_kv(&self, k: &Tensor, v: &Tensor, index_pos: usize) -> Result<()> {
        if !self.key_cache.is_contiguous() || !self.value_cache.is_contiguous() {
            candle::bail!("PagedKvCache key_cache/value_cache must be contiguous")
        }
        let (b_sz, num_kv_heads, seq_len, head_dim) = k.dims4()?;
        let (num_blocks, page_block_size, kv_heads_cache, head_dim_cache) =
            self.key_cache.dims4()?;
        if page_block_size != self.page_block_size {
            candle::bail!(
                "PagedKvCache.page_block_size ({}) does not match key_cache shape (block size {page_block_size})",
                self.page_block_size
            )
        }
        if kv_heads_cache != num_kv_heads || head_dim_cache != head_dim {
            candle::bail!(
                "PagedKvCache key/value cache shape {:?} is incompatible with new kv (heads={num_kv_heads}, head_dim={head_dim})",
                self.key_cache.dims()
            )
        }
        let block_table = self.block_table.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        if block_table.len() != b_sz {
            candle::bail!(
                "block_table batch dim ({}) does not match kv batch dim ({b_sz})",
                block_table.len()
            )
        }
        let last_pos = index_pos + seq_len - 1;
        let mut slots = Vec::with_capacity(b_sz * seq_len);
        for row in &block_table {
            let max_logical_block = last_pos / self.page_block_size;
            if max_logical_block >= row.len() {
                candle::bail!(
                    "block_table has {} blocks, but position {last_pos} needs logical block {max_logical_block}",
                    row.len()
                )
            }
            for t in 0..seq_len {
                let pos = index_pos + t;
                let logical_block = pos / self.page_block_size;
                let offset = pos % self.page_block_size;
                let physical_block = row[logical_block] as usize;
                if physical_block >= num_blocks {
                    candle::bail!(
                        "block_table references physical block {physical_block}, but key_cache only has {num_blocks} blocks"
                    )
                }
                slots.push((physical_block * self.page_block_size + offset) as u32);
            }
        }

        let device = k.device();
        let indices = Tensor::from_vec(slots, (b_sz * seq_len, 1, 1), device)?
            .broadcast_as((b_sz * seq_len, num_kv_heads, head_dim))?
            .contiguous()?;

        let k_flat =
            k.transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz * seq_len, num_kv_heads, head_dim))?;
        let v_flat =
            v.transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz * seq_len, num_kv_heads, head_dim))?;

        let key_flat =
            self.key_cache
                .reshape((num_blocks * page_block_size, num_kv_heads, head_dim))?;
        let value_flat =
            self.value_cache
                .reshape((num_blocks * page_block_size, num_kv_heads, head_dim))?;
        key_flat.scatter_set(&indices, &k_flat, 0)?;
        value_flat.scatter_set(&indices, &v_flat, 0)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    // Additive seam: caller-owned paged KV storage per transformer layer, indexed
    // by `block_idx`. `None` (the default for every layer) preserves today's
    // contiguous concat-and-narrow path byte-for-byte; existing callers that
    // never touch this are entirely unaffected. See `PagedKvCache`.
    paged_kvs: Vec<Option<PagedKvCache>>,
    // Additive seam: caller-owned, graph-replay-safe decode-position tensor per
    // transformer layer, indexed by `block_idx`. `None` (the default for every
    // layer) preserves today's host-side `narrow(0, index_pos, seq_len)` rotary
    // lookup byte-for-byte. See `Cache::set_decode_position`.
    decode_positions: Vec<Option<Tensor>>,
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
            paged_kvs: vec![None; config.num_hidden_layers],
            decode_positions: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Attaches caller-owned paged KV storage for one transformer layer.
    ///
    /// Once set, that layer's attention routes through
    /// `candle_flash_attn::flash_attn_varlen_paged_windowed` against the
    /// caller-owned `key_cache`/`value_cache`/`block_table` instead of the
    /// contiguous concat-and-narrow path. Existing callers that never call this
    /// see no behavior change.
    pub fn set_paged_kv(&mut self, block_idx: usize, paged: PagedKvCache) -> Result<()> {
        let Some(slot) = self.paged_kvs.get_mut(block_idx) else {
            candle::bail!(
                "block_idx {block_idx} out of range for {} layers",
                self.paged_kvs.len()
            )
        };
        *slot = Some(paged);
        Ok(())
    }

    /// Reverts a layer to the contiguous KV cache path.
    pub fn clear_paged_kv(&mut self, block_idx: usize) {
        if let Some(slot) = self.paged_kvs.get_mut(block_idx) {
            *slot = None;
        }
    }

    pub fn paged_kv(&self, block_idx: usize) -> Option<&PagedKvCache> {
        self.paged_kvs.get(block_idx).and_then(|p| p.as_ref())
    }

    /// Attaches a persistent, caller-owned decode-position tensor (shape `(1,)`,
    /// an integer dtype accepted by [`Tensor::index_select`], e.g. `DType::U32`)
    /// for one transformer layer.
    ///
    /// Once set, that layer's rotary embedding lookup reads `cos`/`sin` via
    /// [`Tensor::index_select`] against this device tensor instead of
    /// `narrow(0, index_pos, seq_len)`. `narrow`'s offset is computed on the
    /// host from `index_pos` and baked into the exact device pointer a CUDA
    /// graph capture (`candle_core::CudaGraph`) would record, so replaying a
    /// graph captured at one decode step would silently reuse that step's
    /// rotary embeddings forever. Reading the position from a device tensor
    /// instead keeps the captured kernel launches identical across steps: the
    /// caller updates `position`'s *contents* in place (e.g. via
    /// `Tensor::slice_set`) before each replay, while the tensor's
    /// identity/address stays fixed.
    ///
    /// Scoped to the decode step only (`seq_len == 1`, one query token) —
    /// `CausalSelfAttention::forward` returns an error if a layer with a
    /// decode position attached is called with `seq_len != 1`. Prefill always
    /// uses the existing `narrow`-based path unconditionally, so callers must
    /// not attach a decode position before the first decode step.
    ///
    /// Existing callers that never call this are entirely unaffected.
    pub fn set_decode_position(&mut self, block_idx: usize, position: Tensor) -> Result<()> {
        let Some(slot) = self.decode_positions.get_mut(block_idx) else {
            candle::bail!(
                "block_idx {block_idx} out of range for {} layers",
                self.decode_positions.len()
            )
        };
        if position.dims() != [1] {
            candle::bail!(
                "decode position tensor must have shape (1,), got {:?}",
                position.dims()
            )
        }
        *slot = Some(position);
        Ok(())
    }

    /// Reverts a layer to the host-side `narrow`-based rotary lookup.
    pub fn clear_decode_position(&mut self, block_idx: usize) {
        if let Some(slot) = self.decode_positions.get_mut(block_idx) {
            *slot = None;
        }
    }

    pub fn decode_position(&self, block_idx: usize) -> Option<&Tensor> {
        self.decode_positions
            .get(block_idx)
            .and_then(|p| p.as_ref())
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
    use_flashinfer_attention: bool,
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

#[cfg(feature = "flashinfer-kernels")]
fn flashinfer_decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    candle_flashinfer_kernels::flashinfer_decode_attention(q, k, v, softmax_scale)
}

#[cfg(not(feature = "flashinfer-kernels"))]
fn flashinfer_decode_attention(_: &Tensor, _: &Tensor, _: &Tensor, _: f32) -> Result<Tensor> {
    unimplemented!("compile with '--features flashinfer-kernels'")
}

#[cfg(feature = "flash-attn")]
#[allow(clippy::too_many_arguments)]
fn flash_attn_varlen_paged(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    page_block_size: usize,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn_varlen_paged_windowed(
        q,
        key_cache,
        value_cache,
        seqlens_q,
        seqlens_k,
        block_table,
        None,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        None,
        Some(0),
        page_block_size,
        None,
    )
}

#[cfg(not(feature = "flash-attn"))]
#[allow(clippy::too_many_arguments)]
fn flash_attn_varlen_paged(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: usize,
    _: f32,
    _: usize,
) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &Cache,
    ) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let (cos, sin) = match cache.decode_position(block_idx) {
            Some(position) => {
                if seq_len != 1 {
                    candle::bail!(
                        "Cache::set_decode_position is decode-only (seq_len must be 1), got seq_len {seq_len}"
                    )
                }
                (
                    cache.cos.index_select(position, 0)?,
                    cache.sin.index_select(position, 0)?,
                )
            }
            None => (
                cache.cos.narrow(0, index_pos, seq_len)?,
                cache.sin.narrow(0, index_pos, seq_len)?,
            ),
        };
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

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

        let q = self.apply_rotary_emb(&q, index_pos, block_idx, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, block_idx, cache)?;

        if let Some(paged) = cache.paged_kv(block_idx) {
            let v = v.contiguous()?;
            return self.forward_paged(&q, &k, &v, index_pos, b_sz, seq_len, hidden_size, paged);
        }

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

        // Additive seam: the decode step (one new query token per sequence) can attend via
        // `flashinfer_decode_attention` directly against the pre-`repeat_kv` k/v — it handles
        // grouped-query attention internally, so it must see the `num_key_value_heads`-shaped
        // cache, not the repeated one the branches below use. Prefill (`seq_len > 1`) always
        // falls through to the existing paths below.
        if self.use_flashinfer_attention && seq_len == 1 {
            let q_dec = q.squeeze(2)?.contiguous()?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            let y = flashinfer_decode_attention(&q_dec, &k, &v, softmax_scale)?;
            let y = y.reshape(&[b_sz, 1, hidden_size])?;
            return self.o_proj.forward(&y);
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
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
    }

    /// Writes the current forward pass' K/V into the caller-owned paged cache,
    /// then attends over the full paged history via
    /// `candle_flash_attn::flash_attn_varlen_paged_windowed`. `q`, `k`, `v` are
    /// `(b_sz, heads, seq_len, head_dim)`, post-RoPE for `q`/`k`.
    #[allow(clippy::too_many_arguments)]
    fn forward_paged(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        index_pos: usize,
        b_sz: usize,
        seq_len: usize,
        hidden_size: usize,
        paged: &PagedKvCache,
    ) -> Result<Tensor> {
        paged.write_new_kv(k, v, index_pos)?;

        let q = q.transpose(1, 2)?.contiguous()?.reshape((
            b_sz * seq_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let seqlens_q = Tensor::new(
            (0..=b_sz).map(|i| (i * seq_len) as u32).collect::<Vec<_>>(),
            q.device(),
        )?;
        let max_seqlen_k = paged.block_table.dim(1)? * paged.page_block_size;
        let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();

        let y = flash_attn_varlen_paged(
            &q,
            &paged.key_cache,
            &paged.value_cache,
            &seqlens_q,
            &paged.seqlens_k,
            &paged.block_table,
            seq_len,
            max_seqlen_k,
            softmax_scale,
            paged.page_block_size,
        )?
        .reshape((b_sz, seq_len, hidden_size))?;
        self.o_proj.forward(&y)
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
            use_flashinfer_attention: cfg.use_flashinfer_attention,
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
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
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
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
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
            x = block.forward(&x, index_pos, block_idx, cache)?;
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
            x = block.forward(&x, index_pos, block_idx, cache)?;
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
            use_flashinfer_attention: false,
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
                Tensor::ones((rank, h), DType::F32, dev).unwrap(),
            );
            ts.insert(
                format!("{p}.lora_B.weight"),
                Tensor::ones((h, rank), DType::F32, dev).unwrap(),
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

    // Distinct from `tiny_config()` above: uses 2 layers so paged-cache tests can
    // exercise per-layer attach/detach independently.
    fn paged_test_config() -> Config {
        Config {
            num_hidden_layers: 2,
            ..tiny_config()
        }
    }

    #[test]
    fn paged_kv_seam_is_additive_by_default() -> Result<()> {
        let device = Device::Cpu;
        let cfg = paged_test_config();
        let cache = Cache::new(true, DType::F32, &cfg, &device)?;
        // Every layer defaults to the contiguous path: nothing to opt into paged
        // attention unless a caller explicitly attaches a `PagedKvCache`.
        for block_idx in 0..cfg.num_hidden_layers {
            assert!(cache.paged_kv(block_idx).is_none());
        }
        assert!(cache.use_kv_cache);
        Ok(())
    }

    #[test]
    fn set_and_clear_paged_kv() -> Result<()> {
        let device = Device::Cpu;
        let cfg = paged_test_config();
        let mut cache = Cache::new(true, DType::F32, &cfg, &device)?;

        let num_blocks = 2;
        let page_block_size = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let paged = PagedKvCache {
            key_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &device,
            )?,
            value_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &device,
            )?,
            block_table: Tensor::from_vec(vec![0u32, 1u32], (1, 2), &device)?,
            seqlens_k: Tensor::new(&[0u32, 4u32], &device)?,
            page_block_size,
        };
        cache.set_paged_kv(0, paged)?;
        assert!(cache.paged_kv(0).is_some());
        assert!(cache.paged_kv(1).is_none());

        cache.clear_paged_kv(0);
        assert!(cache.paged_kv(0).is_none());

        // Out-of-range layers are rejected rather than silently ignored.
        let paged = PagedKvCache {
            key_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &device,
            )?,
            value_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &device,
            )?,
            block_table: Tensor::from_vec(vec![0u32, 1u32], (1, 2), &device)?,
            seqlens_k: Tensor::new(&[0u32, 4u32], &device)?,
            page_block_size,
        };
        assert!(cache.set_paged_kv(cfg.num_hidden_layers, paged).is_err());
        Ok(())
    }

    #[test]
    fn paged_kv_cache_writes_land_in_the_slots_the_block_table_designates() -> Result<()> {
        let device = Device::Cpu;
        // 2 physical blocks of 4 slots each, 1 kv head, head_dim 2.
        let num_blocks = 2;
        let page_block_size = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let key_cache = Tensor::zeros(
            (num_blocks, page_block_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )?;
        let value_cache = Tensor::zeros(
            (num_blocks, page_block_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )?;
        // Logical block 0 -> physical block 1, logical block 1 -> physical block 0.
        let block_table = Tensor::from_vec(vec![1u32, 0u32], (1, 2), &device)?;
        let paged = PagedKvCache {
            key_cache: key_cache.clone(),
            value_cache: value_cache.clone(),
            block_table,
            seqlens_k: Tensor::new(&[0u32, 5u32], &device)?,
            page_block_size,
        };

        // 3 new tokens at absolute positions 2, 3, 4 (b_sz=1).
        // pos 2 -> logical block 0, offset 2 -> physical slot 1*4+2 = 6
        // pos 3 -> logical block 0, offset 3 -> physical slot 1*4+3 = 7
        // pos 4 -> logical block 1, offset 0 -> physical slot 0*4+0 = 0
        let index_pos = 2;
        let seq_len = 3;
        let k = Tensor::from_vec(
            vec![10f32, 11., 20., 21., 30., 31.],
            (1, num_kv_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::from_vec(
            vec![-10f32, -11., -20., -21., -30., -31.],
            (1, num_kv_heads, seq_len, head_dim),
            &device,
        )?;

        paged.write_new_kv(&k, &v, index_pos)?;

        let key_flat = paged
            .key_cache
            .reshape((num_blocks * page_block_size, num_kv_heads, head_dim))?
            .i((.., 0, ..))?
            .to_vec2::<f32>()?;
        let value_flat = paged
            .value_cache
            .reshape((num_blocks * page_block_size, num_kv_heads, head_dim))?
            .i((.., 0, ..))?
            .to_vec2::<f32>()?;

        assert_eq!(key_flat[6], [10., 11.]);
        assert_eq!(key_flat[7], [20., 21.]);
        assert_eq!(key_flat[0], [30., 31.]);
        assert_eq!(value_flat[6], [-10., -11.]);
        assert_eq!(value_flat[7], [-20., -21.]);
        assert_eq!(value_flat[0], [-30., -31.]);

        // Untouched slots stay zero.
        for slot in [1, 2, 3, 4, 5] {
            assert_eq!(key_flat[slot], [0., 0.]);
            assert_eq!(value_flat[slot], [0., 0.]);
        }
        Ok(())
    }

    #[test]
    fn paged_kv_cache_rejects_block_table_overflow() -> Result<()> {
        let device = Device::Cpu;
        let num_blocks = 1;
        let page_block_size = 2;
        let num_kv_heads = 1;
        let head_dim = 2;
        let paged = PagedKvCache {
            key_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &device,
            )?,
            value_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &device,
            )?,
            // Only one logical block available.
            block_table: Tensor::from_vec(vec![0u32], (1, 1), &device)?,
            seqlens_k: Tensor::new(&[0u32, 2u32], &device)?,
            page_block_size,
        };
        // Position 2 needs logical block 1, which doesn't exist in the table.
        let k = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &device)?;
        let v = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &device)?;
        assert!(paged.write_new_kv(&k, &v, 2).is_err());
        Ok(())
    }

    // Requires a CUDA device and the `flash-attn` feature (which compiles the real
    // `flash_attn_varlen_paged_windowed` kernel); not runnable on the CPU-only sandbox
    // this crate is normally developed in. Exercises the actual GPU code path end to
    // end: writes new K/V into a caller-owned paged cache and checks the attention
    // output against the long-established dense (non-flash) causal path.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn paged_attention_matches_dense_causal_attention_on_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        let num_attention_heads = 2;
        let num_key_value_heads = 2;
        let head_dim = 64;
        let hidden_size = num_attention_heads * head_dim;
        let kv_size = num_key_value_heads * head_dim;
        let seq_len = 17;
        let page_block_size = 32;

        let new_linear = |out_dim: usize, in_dim: usize| -> Result<Linear> {
            let w = Tensor::randn(0f32, 0.02, (out_dim, in_dim), &device)?.to_dtype(dtype)?;
            Ok(Linear::from_weights(w, None))
        };
        let attn = CausalSelfAttention {
            q_proj: Proj::Plain(new_linear(hidden_size, hidden_size)?),
            k_proj: Proj::Plain(new_linear(kv_size, hidden_size)?),
            v_proj: Proj::Plain(new_linear(kv_size, hidden_size)?),
            o_proj: Proj::Plain(new_linear(hidden_size, hidden_size)?),
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            use_flash_attn: false,
            use_flashinfer_attention: false,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
        };

        let mut cfg = paged_test_config();
        cfg.hidden_size = hidden_size;
        cfg.num_attention_heads = num_attention_heads;
        cfg.num_key_value_heads = num_key_value_heads;
        cfg.num_hidden_layers = 1;
        // `paged_test_config()`'s base `max_position_embeddings` (16) is too small
        // for this test's `seq_len` (17); the dense reference path narrows the
        // RoPE cos/sin cache to `max_position_embeddings` rows.
        cfg.max_position_embeddings = DEFAULT_MAX_SEQ_LEN;

        let x = Tensor::randn(0f32, 1., (1, seq_len, hidden_size), &device)?.to_dtype(dtype)?;

        // Ground truth: the long-established dense (non-flash) causal softmax path.
        let mut dense_cache = Cache::new(true, dtype, &cfg, &device)?;
        let y_dense = attn.forward(&x, 0, 0, &mut dense_cache)?;

        // Candidate: paged cache sized to hold the whole sequence in a single block,
        // routed through `flash_attn_varlen_paged_windowed`.
        let key_cache = Tensor::zeros(
            (1, page_block_size, num_key_value_heads, head_dim),
            dtype,
            &device,
        )?;
        let value_cache = Tensor::zeros(
            (1, page_block_size, num_key_value_heads, head_dim),
            dtype,
            &device,
        )?;
        let paged = PagedKvCache {
            key_cache,
            value_cache,
            block_table: Tensor::from_vec(vec![0u32], (1, 1), &device)?,
            seqlens_k: Tensor::new(&[0u32, seq_len as u32], &device)?,
            page_block_size,
        };
        let mut paged_cache = Cache::new(true, dtype, &cfg, &device)?;
        paged_cache.set_paged_kv(0, paged)?;
        let y_paged = attn.forward(&x, 0, 0, &mut paged_cache)?;

        let diff = y_dense
            .to_dtype(DType::F32)?
            .sub(&y_paged.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(diff < 0.1, "paged vs dense max abs diff {diff}");
        Ok(())
    }

    #[test]
    fn decode_position_seam_is_additive_by_default() -> Result<()> {
        let device = Device::Cpu;
        let cfg = paged_test_config();
        let cache = Cache::new(true, DType::F32, &cfg, &device)?;
        // Every layer defaults to the host-side narrow path: nothing changes
        // unless a caller explicitly attaches a decode-position tensor.
        for block_idx in 0..cfg.num_hidden_layers {
            assert!(cache.decode_position(block_idx).is_none());
        }
        Ok(())
    }

    #[test]
    fn set_and_clear_decode_position() -> Result<()> {
        let device = Device::Cpu;
        let cfg = paged_test_config();
        let mut cache = Cache::new(true, DType::F32, &cfg, &device)?;

        cache.set_decode_position(0, Tensor::new(&[3u32], &device)?)?;
        assert!(cache.decode_position(0).is_some());
        assert!(cache.decode_position(1).is_none());

        cache.clear_decode_position(0);
        assert!(cache.decode_position(0).is_none());

        // Out-of-range layers are rejected rather than silently ignored.
        assert!(cache
            .set_decode_position(cfg.num_hidden_layers, Tensor::new(&[3u32], &device)?)
            .is_err());

        // Anything other than a single-element tensor is rejected: the seam
        // is decode-only (one query token per layer per step).
        assert!(cache
            .set_decode_position(0, Tensor::new(&[3u32, 4u32], &device)?)
            .is_err());
        Ok(())
    }

    fn tiny_causal_self_attention(cfg: &Config, device: &Device) -> Result<CausalSelfAttention> {
        let hidden_size = cfg.hidden_size;
        let kv_size = (hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let new_linear = |out_dim: usize, in_dim: usize| -> Result<Linear> {
            let w = Tensor::randn(0f32, 0.02, (out_dim, in_dim), device)?;
            Ok(Linear::from_weights(w, None))
        };
        Ok(CausalSelfAttention {
            q_proj: new_linear(hidden_size, hidden_size)?,
            k_proj: new_linear(kv_size, hidden_size)?,
            v_proj: new_linear(kv_size, hidden_size)?,
            o_proj: new_linear(hidden_size, hidden_size)?,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: hidden_size / cfg.num_attention_heads,
            use_flash_attn: false,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }

    #[test]
    fn decode_position_matches_narrow_based_rotary_lookup() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let attn = tiny_causal_self_attention(&cfg, &device)?;
        let x = Tensor::randn(0f32, 1., (1, 1, cfg.hidden_size), &device)?;
        let index_pos = 3;

        // Baseline: today's host-side narrow-based rotary lookup.
        let mut narrow_cache = Cache::new(false, DType::F32, &cfg, &device)?;
        let y_narrow = attn.forward(&x, index_pos, 0, &mut narrow_cache)?;

        // Candidate: device-tensor decode-position gather, same absolute position.
        let mut gather_cache = Cache::new(false, DType::F32, &cfg, &device)?;
        gather_cache.set_decode_position(0, Tensor::new(&[index_pos as u32], &device)?)?;
        let y_gather = attn.forward(&x, index_pos, 0, &mut gather_cache)?;

        assert_eq!(y_narrow.to_vec3::<f32>()?, y_gather.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn decode_position_rejects_multi_token_forward() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let attn = tiny_causal_self_attention(&cfg, &device)?;
        // Prefill-shaped input (seq_len > 1) must not be routed through the
        // decode-only gather path.
        let x = Tensor::randn(0f32, 1., (1, 3, cfg.hidden_size), &device)?;
        let mut cache = Cache::new(false, DType::F32, &cfg, &device)?;
        cache.set_decode_position(0, Tensor::new(&[0u32], &device)?)?;
        assert!(attn.forward(&x, 0, 0, &mut cache).is_err());
        Ok(())
    }

    // Requires a CUDA device (feature-gated behind flash-attn, same as the paged-
    // attention GPU test above, since that's the GPU feature this crate's CI
    // compiles under); not runnable on the CPU-only sandbox this crate is
    // normally developed in. Exercises the actual property issue #12 is about:
    // a CudaGraph capturing a decode step must replay against the *current*
    // contents of the attached decode-position tensor, not the position that
    // was live at capture time.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn decode_position_stays_correct_across_cuda_graph_replay() -> Result<()> {
        use candle::CudaGraph;

        let device = Device::new_cuda(0)?;
        let Device::Cuda(cuda_device) = &device else {
            unreachable!()
        };
        let dtype = DType::F32;

        let cfg = tiny_config();
        let attn = tiny_causal_self_attention(&cfg, &device)?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        // Shape (b_sz, heads, seq_len=1, head_dim): what `apply_rotary_emb` sees
        // for a single decode step's query. Capture that call directly instead
        // of routing through the full attention `forward`: with one query
        // attending only to itself and no cached history, softmax over a
        // single key is trivially 1.0 and the attention *output* is
        // position-invariant regardless of rotary correctness, so it can't
        // distinguish a graph stuck on a stale position from a correct one.
        let q_in = Tensor::randn(0f32, 1., (1, cfg.num_attention_heads, 1, head_dim), &device)?;

        let position = Tensor::new(&[0u32], &device)?;
        let mut cache = Cache::new(false, dtype, &cfg, &device)?;
        cache.set_decode_position(0, position.clone())?;

        // `out` is allocated before capture and written in place via `slice_set`,
        // per `CudaGraph::capture`'s contract: a tensor allocated inside the
        // capture closure would be a graph-owned allocation, unreadable outside
        // replay. The warm-up run must itself hold `enable_cuda_graph_htod_cache`:
        // that guard is what makes the *first* upload of a given host constant
        // (here, `index_select`'s non-contiguous-layout parameter vector) populate
        // the cache instead of performing a plain upload; `CudaGraph::capture` only
        // enables the guard for its own call, so a warm-up run outside of any guard
        // never seeds the cache, and capture then fails with a cache-miss error.
        let out = Tensor::zeros((1, cfg.num_attention_heads, 1, head_dim), dtype, &device)?;
        {
            let _warmup_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
            let rotated = attn.apply_rotary_emb(&q_in, 0, 0, &cache)?;
            out.slice_set(&rotated, 0, 0)?;
        }
        device.synchronize()?;

        let (graph, ()) = CudaGraph::capture(cuda_device, || {
            let rotated = attn.apply_rotary_emb(&q_in, 0, 0, &cache)?;
            out.slice_set(&rotated, 0, 0)?;
            Ok(())
        })?;

        // Replaying at the captured position (0) must match the dense narrow-based
        // path at position 0.
        graph.replay()?;
        device.synchronize()?;
        let narrow_cache_0 = Cache::new(false, dtype, &cfg, &device)?;
        let expected_0 = attn.apply_rotary_emb(&q_in, 0, 0, &narrow_cache_0)?;
        assert_eq!(
            out.flatten_all()?.to_vec1::<f32>()?,
            expected_0.flatten_all()?.to_vec1::<f32>()?
        );

        // Update the position tensor's *contents* in place (its identity/address
        // is unchanged) and replay again: the graph must pick up position 5, not
        // silently keep replaying position 0.
        position.slice_set(&Tensor::new(&[5u32], &device)?, 0, 0)?;
        graph.replay()?;
        device.synchronize()?;
        let narrow_cache_5 = Cache::new(false, dtype, &cfg, &device)?;
        let expected_5 = attn.apply_rotary_emb(&q_in, 5, 0, &narrow_cache_5)?;
        assert_eq!(
            out.flatten_all()?.to_vec1::<f32>()?,
            expected_5.flatten_all()?.to_vec1::<f32>()?
        );

        // Sanity check that positions 0 and 5 actually produce different rotary
        // embeddings; otherwise the assertions above wouldn't distinguish a
        // graph that's still stuck on the captured position from one that isn't.
        assert_ne!(
            expected_0.flatten_all()?.to_vec1::<f32>()?,
            expected_5.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "flashinfer-kernels"))]
mod flashinfer_attention_tests {
    use super::*;

    fn tiny_config(use_flashinfer_attention: bool) -> Config {
        Config {
            hidden_size: 8,
            intermediate_size: 8,
            vocab_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            use_flash_attn: false,
            use_flashinfer_attention,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 16,
            tie_word_embeddings: false,
        }
    }

    fn weights(cfg: &Config, dev: &Device) -> HashMap<String, Tensor> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let mut ts = HashMap::new();
        ts.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::rand(0f32, 1f32, (v, h), dev).unwrap(),
        );
        ts.insert(
            "model.norm.weight".to_string(),
            Tensor::ones(h, DType::F32, dev).unwrap(),
        );
        ts.insert(
            "lm_head.weight".to_string(),
            Tensor::rand(0f32, 1f32, (v, h), dev).unwrap(),
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
            let size_q = h;
            let size_kv = (h / cfg.num_attention_heads) * cfg.num_key_value_heads;
            ts.insert(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::rand(-1f32, 1f32, (size_q, h), dev).unwrap(),
            );
            ts.insert(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::rand(-1f32, 1f32, (size_kv, h), dev).unwrap(),
            );
            ts.insert(
                format!("{p}.self_attn.v_proj.weight"),
                Tensor::rand(-1f32, 1f32, (size_kv, h), dev).unwrap(),
            );
            ts.insert(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::rand(-1f32, 1f32, (size_q, h), dev).unwrap(),
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

    // The decode-step flashinfer seam is purely additive: with the flag unset the
    // forward pass must be byte-for-byte identical to the pre-existing dense path,
    // and with it set the numerically-equivalent flashinfer kernel must reproduce
    // the same logits (up to floating-point tolerance) for a single-token decode
    // step following a multi-token prefill.
    #[test]
    fn flashinfer_decode_matches_dense_path() -> Result<()> {
        let dev = Device::Cpu;
        let cfg_dense = tiny_config(false);
        let cfg_flashinfer = tiny_config(true);
        let ts = weights(&cfg_dense, &dev);

        let vb = VarBuilder::from_tensors(ts.clone(), DType::F32, &dev);
        let model_dense = Llama::load(vb, &cfg_dense)?;
        let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
        let model_flashinfer = Llama::load(vb, &cfg_flashinfer)?;

        let prefill = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        let mut cache_dense = Cache::new(true, DType::F32, &cfg_dense, &dev)?;
        let mut cache_flashinfer = Cache::new(true, DType::F32, &cfg_flashinfer, &dev)?;
        model_dense.forward(&prefill, 0, &mut cache_dense)?;
        model_flashinfer.forward(&prefill, 0, &mut cache_flashinfer)?;

        let decode = Tensor::new(&[[4u32]], &dev)?;
        let dense_logits = model_dense
            .forward(&decode, 3, &mut cache_dense)?
            .to_vec2::<f32>()?;
        let flashinfer_logits = model_flashinfer
            .forward(&decode, 3, &mut cache_flashinfer)?
            .to_vec2::<f32>()?;

        assert_eq!(dense_logits.len(), flashinfer_logits.len());
        for (a, b) in dense_logits[0].iter().zip(flashinfer_logits[0].iter()) {
            assert!((a - b).abs() < 1e-4, "dense={a} flashinfer={b}");
        }
        Ok(())
    }
}
