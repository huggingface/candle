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

/// Physical K/V storage and per-sequence block table for paged attention on
/// a single transformer layer, in the layout
/// `candle_flash_attn::flash_attn_varlen_paged_windowed` expects.
/// Constructed and owned by the caller — e.g. a downstream block
/// allocator/scheduler — and handed to [`Cache::new_paged`] as one entry per
/// transformer layer, in `Config::num_hidden_layers` order.
///
/// `Llama::forward` only writes the newly computed K/V for the tokens at
/// `index_pos..index_pos + seq_len` into the slots `block_table` indicates,
/// and reads `block_table` / `seqlens_k` to attend over the pool. Block
/// allocation, eviction and admission stay a caller concern.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    /// `(num_blocks, page_block_size, num_key_value_heads, head_dim)`.
    pub key_cache: Tensor,
    /// `(num_blocks, page_block_size, num_key_value_heads, head_dim)`.
    pub value_cache: Tensor,
    /// `(batch_size, max_blocks)`, physical block ids (`u32`), one row per
    /// sequence in the batch, in the same order as the model's input batch.
    pub block_table: Tensor,
    /// `(batch_size + 1,)`, cumulative per-sequence context length in
    /// tokens (`u32`), covering the tokens this `forward` call is about to
    /// write (e.g. for a uniform batch, `[0, L, 2*L, .., batch_size*L]` with
    /// `L = index_pos + seq_len`). The flash-attn path reads this as-is; the
    /// dense fallback (used when `Config::use_flash_attn` is `false`)
    /// instead assumes a uniform `index_pos + seq_len` context length across
    /// the batch, matching the contiguous cache path.
    pub seqlens_k: Tensor,
    /// Number of tokens stored per block in `key_cache` / `value_cache`.
    pub page_block_size: usize,
}

impl PagedKvCache {
    /// Writes `k`/`v` (already rotary-embedded for `k`), shaped `(b_sz,
    /// num_key_value_heads, seq_len, head_dim)`, into this layer's pool at
    /// the slots `block_table` indicates for `index_pos..index_pos+seq_len`.
    fn write(&self, k: &Tensor, v: &Tensor, index_pos: usize) -> Result<()> {
        let (b_sz, _num_kv_heads, seq_len, _head_dim) = k.dims4()?;
        let page_block_size = self.page_block_size;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let block_table = self.block_table.to_dtype(DType::U32)?.to_vec2::<u32>()?;

        let mut t = 0usize;
        while t < seq_len {
            let pos = index_pos + t;
            let logical_block = pos / page_block_size;
            let offset = pos % page_block_size;
            let run = (page_block_size - offset).min(seq_len - t);
            for (b, row) in block_table.iter().enumerate().take(b_sz) {
                let physical_block = row[logical_block] as usize;
                let k_run = k.i((b, t..t + run))?.contiguous()?;
                let v_run = v.i((b, t..t + run))?.contiguous()?;
                self.key_cache
                    .i(physical_block)?
                    .slice_set(&k_run, 0, offset)?;
                self.value_cache
                    .i(physical_block)?
                    .slice_set(&v_run, 0, offset)?;
            }
            t += run;
        }
        Ok(())
    }

    /// Gathers `kv_len` tokens of history per sequence out of the pool via
    /// `block_table`, for the dense (non flash-attn) fallback path. Returns
    /// `(k, v)` shaped `(b_sz, num_key_value_heads, kv_len, head_dim)`.
    fn gather(&self, b_sz: usize, kv_len: usize) -> Result<(Tensor, Tensor)> {
        let page_block_size = self.page_block_size;
        let num_logical_blocks = kv_len.div_ceil(page_block_size);
        let block_table = self.block_table.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut ks = Vec::with_capacity(b_sz);
        let mut vs = Vec::with_capacity(b_sz);
        for row in block_table.iter().take(b_sz) {
            let k_blocks: Vec<_> = row[..num_logical_blocks]
                .iter()
                .map(|&pb| self.key_cache.i(pb as usize))
                .collect::<Result<_>>()?;
            let v_blocks: Vec<_> = row[..num_logical_blocks]
                .iter()
                .map(|&pb| self.value_cache.i(pb as usize))
                .collect::<Result<_>>()?;
            ks.push(Tensor::cat(&k_blocks, 0)?.narrow(0, 0, kv_len)?);
            vs.push(Tensor::cat(&v_blocks, 0)?.narrow(0, 0, kv_len)?);
        }
        let k = Tensor::stack(&ks, 0)?.transpose(1, 2)?.contiguous()?;
        let v = Tensor::stack(&vs, 0)?.transpose(1, 2)?.contiguous()?;
        Ok((k, v))
    }
}

#[derive(Debug, Clone)]
enum KvStore {
    Contiguous(Vec<Option<(Tensor, Tensor)>>),
    Paged(Vec<PagedKvCache>),
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kv_store: KvStore,
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
            kv_store: KvStore::Contiguous(vec![None; config.num_hidden_layers]),
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Like [`Cache::new`], but backs the KV storage with caller-owned
    /// [`PagedKvCache`] pools instead of the contiguous concat-and-narrow
    /// path — one entry per transformer layer, in `config.num_hidden_layers`
    /// order. When `Config::use_flash_attn` is set, `Llama::forward` attends
    /// over the pools via
    /// `candle_flash_attn::flash_attn_varlen_paged_windowed`; otherwise it
    /// falls back to gathering the referenced blocks into a dense tensor for
    /// a plain masked matmul (works without the `flash-attn` feature, e.g.
    /// for CPU testing). Block allocation, eviction and admission remain the
    /// caller's responsibility — this only wires the attention/cache call
    /// site.
    pub fn new_paged(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
        layers: Vec<PagedKvCache>,
    ) -> Result<Self> {
        if layers.len() != config.num_hidden_layers {
            candle::bail!(
                "expected {} paged KV cache layers, got {}",
                config.num_hidden_layers,
                layers.len()
            )
        }
        let mut cache = Self::new(use_kv_cache, dtype, config, device)?;
        cache.kv_store = KvStore::Paged(layers);
        Ok(cache)
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

#[cfg(feature = "flash-attn")]
#[allow(clippy::too_many_arguments)]
fn flash_attn_paged(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_right: Option<usize>,
    page_block_size: usize,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn_varlen_paged_windowed(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        block_table,
        None,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        None,
        window_size_right,
        page_block_size,
        None,
    )
}

#[cfg(not(feature = "flash-attn"))]
#[allow(clippy::too_many_arguments)]
fn flash_attn_paged(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _block_table: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    _softmax_scale: f32,
    _window_size_right: Option<usize>,
    _page_block_size: usize,
) -> Result<Tensor> {
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
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // Only needed by the dense (non flash-attn) paths below; computed
        // once, ahead of borrowing `cache.kv_store` mutably, since building
        // it requires `&mut cache` too (for the memoized mask cache).
        let mask = if !self.use_flash_attn && seq_len > 1 {
            Some(cache.mask(seq_len, index_pos)?)
        } else {
            None
        };

        let use_kv_cache = cache.use_kv_cache;
        let y = match &mut cache.kv_store {
            KvStore::Contiguous(kvs) => self.forward_contiguous(
                q,
                k,
                v,
                seq_len,
                use_kv_cache,
                &mut kvs[block_idx],
                mask.as_ref(),
            )?,
            KvStore::Paged(layers) => self.forward_paged(
                q,
                k,
                v,
                index_pos,
                seq_len,
                b_sz,
                &layers[block_idx],
                mask.as_ref(),
            )?,
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    /// Today's contiguous concat-and-narrow KV cache path, unchanged in
    /// behavior from before [`PagedKvCache`] was added.
    #[allow(clippy::too_many_arguments)]
    fn forward_contiguous(
        &self,
        q: Tensor,
        mut k: Tensor,
        mut v: Tensor,
        seq_len: usize,
        use_kv_cache: bool,
        kv: &mut Option<(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if use_kv_cache {
            if let Some((cache_k, cache_v)) = kv {
                k = Tensor::cat(&[&*cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[&*cache_v, &v], 2)?.contiguous()?;
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
            *kv = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    masked_fill(&att, &mask.broadcast_as(att.shape())?, f32::NEG_INFINITY)?
                }
            };

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)
        }
    }

    /// Additive paged-attention path: writes this step's K/V into the
    /// caller-owned [`PagedKvCache`] pool, then either attends via the
    /// paged flash-attn kernel or, without the `flash-attn` feature /
    /// `Config::use_flash_attn`, falls back to gathering the referenced
    /// blocks into a dense tensor for a plain masked matmul.
    #[allow(clippy::too_many_arguments)]
    fn forward_paged(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        index_pos: usize,
        seq_len: usize,
        b_sz: usize,
        paged: &PagedKvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        paged.write(&k, &v, index_pos)?;

        if self.use_flash_attn {
            let device = q.device().clone();
            let q = q
                .transpose(1, 2)?
                .reshape((b_sz * seq_len, self.num_attention_heads, self.head_dim))?
                .contiguous()?;
            let seqlens_q: Vec<u32> = (0..=b_sz as u32).map(|i| i * seq_len as u32).collect();
            let seqlens_q = Tensor::new(seqlens_q, &device)?;
            let seqlens_k = paged.seqlens_k.to_dtype(DType::U32)?;
            let max_seqlen_k = seqlens_k
                .to_vec1::<u32>()?
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .max()
                .unwrap_or(0);
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            let window_size_right = if seq_len > 1 { Some(0) } else { None };
            let y = flash_attn_paged(
                &q,
                &paged.key_cache,
                &paged.value_cache,
                &seqlens_q,
                &seqlens_k,
                &paged.block_table,
                seq_len,
                max_seqlen_k,
                softmax_scale,
                window_size_right,
                paged.page_block_size,
            )?;
            y.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        } else {
            let kv_len = index_pos + seq_len;
            let (k, v) = paged.gather(b_sz, kv_len)?;
            let k = self.repeat_kv(k)?;
            let v = self.repeat_kv(v)?;
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    masked_fill(&att, &mask.broadcast_as(att.shape())?, f32::NEG_INFINITY)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)
        }
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

    /// Unlike [`base_weights`] (uniform embeddings, zeroed q/k projections),
    /// this gives every token id a distinct embedding and makes q/k/v/o
    /// projections the identity, so attention output actually depends on
    /// *which* positions the KV cache reconstructs and in what order —
    /// letting [`paged_cache_matches_contiguous_cache`] catch addressing
    /// bugs that a uniform-embedding fixture would miss.
    fn distinct_weights(cfg: &Config, dev: &Device) -> HashMap<String, Tensor> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let mut ts = HashMap::new();
        let embed: Vec<f32> = (0..v * h)
            .map(|idx| (idx / h) as f32 * 0.1 + (idx % h) as f32 * 0.01)
            .collect();
        ts.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(embed, (v, h), dev).unwrap(),
        );
        ts.insert(
            "model.norm.weight".to_string(),
            Tensor::ones(h, DType::F32, dev).unwrap(),
        );
        ts.insert(
            "lm_head.weight".to_string(),
            (Tensor::ones((v, h), DType::F32, dev).unwrap() * 0.1).unwrap(),
        );
        let identity: Vec<f32> = (0..h * h)
            .map(|idx| if idx / h == idx % h { 1.0 } else { 0.0 })
            .collect();
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
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                ts.insert(
                    format!("{p}.self_attn.{proj}.weight"),
                    Tensor::from_vec(identity.clone(), (h, h), dev).unwrap(),
                );
            }
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

    /// End-to-end regression test for the additive `Cache::new_paged` seam
    /// (candle issue #8): with `Config::use_flash_attn` off, the paged path
    /// falls back to gathering blocks into a dense tensor, so this runs on
    /// CPU without the `flash-attn` feature. A 3-token prefill followed by
    /// two single-token decode steps against a `PagedKvCache` (block size 2,
    /// identity block table) must reproduce the contiguous cache's logits
    /// exactly, proving the block/offset write-and-gather addressing lines
    /// up with the plain concat-and-narrow path.
    #[test]
    fn paged_cache_matches_contiguous_cache() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let vb = VarBuilder::from_tensors(distinct_weights(&cfg, &dev), DType::F32, &dev);
        let model = Llama::load(vb, &cfg)?;

        let prompt = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        let step1 = Tensor::new(&[[4u32]], &dev)?;
        let step2 = Tensor::new(&[[5u32]], &dev)?;

        let mut contiguous_cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let contiguous_prompt = model
            .forward(&prompt, 0, &mut contiguous_cache)?
            .to_vec2::<f32>()?;
        let contiguous_step1 = model
            .forward(&step1, 3, &mut contiguous_cache)?
            .to_vec2::<f32>()?;
        let contiguous_step2 = model
            .forward(&step2, 4, &mut contiguous_cache)?
            .to_vec2::<f32>()?;

        let page_block_size = 2usize;
        let num_blocks = 3usize;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let paged_layer = PagedKvCache {
            key_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &dev,
            )?,
            value_cache: Tensor::zeros(
                (num_blocks, page_block_size, num_kv_heads, head_dim),
                DType::F32,
                &dev,
            )?,
            block_table: Tensor::new(&[[0u32, 1, 2]], &dev)?,
            // Unused by the dense fallback, which derives context length
            // from `index_pos + seq_len` (see `PagedKvCache::seqlens_k`).
            seqlens_k: Tensor::new(&[0u32, 0], &dev)?,
            page_block_size,
        };
        let mut paged_cache = Cache::new_paged(true, DType::F32, &cfg, &dev, vec![paged_layer])?;
        let paged_prompt = model
            .forward(&prompt, 0, &mut paged_cache)?
            .to_vec2::<f32>()?;
        let paged_step1 = model
            .forward(&step1, 3, &mut paged_cache)?
            .to_vec2::<f32>()?;
        let paged_step2 = model
            .forward(&step2, 4, &mut paged_cache)?
            .to_vec2::<f32>()?;

        assert_eq!(contiguous_prompt, paged_prompt);
        assert_eq!(contiguous_step1, paged_step1);
        assert_eq!(contiguous_step2, paged_step2);
        Ok(())
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
}
