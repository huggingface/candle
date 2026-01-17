//! GraniteMoeHybrid is a Long Context Transformer Language Model.
//!
//! A high performance transformer model optimized for efficient processing
//! of very long context sequences

use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::iter::repeat_n;
use std::{collections::HashMap, f32::consts::PI};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum GraniteMoeHybridRopeType {
    #[serde(rename = "granite")]
    Granite,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GraniteMoeHybridRopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: GraniteMoeHybridRopeType,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GraniteMoeHybridConfig {
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
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<GraniteMoeHybridRopeConfig>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub layer_types: Vec<GraniteMoeHybridLayerType>,
    #[serde(default = "default_one")]
    pub attention_multiplier: f32,
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_one")]
    pub residual_multiplier: f32,
    #[serde(default = "default_one")]
    pub logits_scaling: f32,
    #[serde(default)]
    pub shared_intermediate_size: Option<usize>,
}

impl GraniteMoeHybridConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

fn default_one() -> f32 {
    1.0
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteMoeHybridLayerType {
    #[default]
    Attention,
    Mamba,
}

impl GraniteMoeHybridConfig {
    pub fn into_config(self, use_flash_attn: bool) -> GraniteMoeHybridInternalConfig {
        let layer_types = if self.layer_types.is_empty() {
            vec![GraniteMoeHybridLayerType::Attention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        };
        let shared_intermediate_size = self
            .shared_intermediate_size
            .unwrap_or(self.intermediate_size);
        GraniteMoeHybridInternalConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            shared_intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            use_flash_attn,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            layer_types,
            attention_multiplier: self.attention_multiplier,
            embedding_multiplier: self.embedding_multiplier,
            residual_multiplier: self.residual_multiplier,
            logits_scaling: self.logits_scaling,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybridInternalConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<GraniteMoeHybridRopeConfig>,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<GraniteMoeHybridLayerType>,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybridCache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &GraniteMoeHybridInternalConfig) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl GraniteMoeHybridCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &GraniteMoeHybridInternalConfig,
        device: &Device,
    ) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(GraniteMoeHybridRopeConfig {
                rope_type: GraniteMoeHybridRopeType::Default,
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

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mut mask: Vec<u8> = Vec::with_capacity(t * t);
            (0..t).for_each(|i| {
                mask.extend(repeat_n(0, i + 1));
                mask.extend(repeat_n(1, t - i - 1));
            });
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
    attention_multiplier: f32,
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
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &GraniteMoeHybridCache,
    ) -> Result<Tensor> {
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
        cache: &mut GraniteMoeHybridCache,
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
            flash_attn(&q, &k, &v, self.attention_multiplier, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = q
                .matmul(&k.t()?)?
                .affine(self.attention_multiplier as f64, 0.)?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
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

    fn load(vb: VarBuilder, cfg: &GraniteMoeHybridInternalConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
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
            attention_multiplier: cfg.attention_multiplier,
        })
    }
}

/// Utility function to fill elements of a tensor based on a boolean mask.
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

// A simple feed forward network with a gated activation
// (GeLU, SiLU, etc.). The goal is to add non-linearity and
// increase the model's capacity to learn complex patterns.
#[derive(Debug, Clone)]
struct MultiLayerPercepton {
    input_linear: Linear,
    output_linear: Linear,
    span: tracing::Span,
}

impl MultiLayerPercepton {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let projected = self.input_linear.forward(x)?;
        let chunks = projected.chunk(2, D::Minus1)?;
        let (left, right) = (&chunks[0], &chunks[1]);
        let gated = (candle_nn::ops::silu(left)? * right)?;
        self.output_linear.forward(&gated)
    }

    fn load(vb: VarBuilder, cfg: &GraniteMoeHybridInternalConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let inter_size = cfg.shared_intermediate_size;
        let input_linear = linear(h_size, inter_size * 2, vb.pp("shared_mlp.input_linear"))?;
        let output_linear = linear(inter_size, h_size, vb.pp("shared_mlp.output_linear"))?;
        Ok(Self {
            input_linear,
            output_linear,
            span,
        })
    }
}

// A Block is a actually a Transformer layer, consisting of
// a self-attention mechanism followed by a feed-forward neural network (MLP).
#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    multi_layer_percepton: MultiLayerPercepton,
    span: tracing::Span,
    residual_scale: f32,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut GraniteMoeHybridCache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let attn = self.attn.forward(&x, index_pos, block_idx, cache)?;
        let attn = scale_tensor(attn, self.residual_scale)?;
        let x = (attn + residual)?;
        let residual = &x;
        let multi_layer_percepton_out = self
            .multi_layer_percepton
            .forward(&self.rms_2.forward(&x)?)?;
        let multi_layer_percepton_out =
            scale_tensor(multi_layer_percepton_out, self.residual_scale)?;
        let x = (multi_layer_percepton_out + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &GraniteMoeHybridInternalConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let multi_layer_percepton = MultiLayerPercepton::load(vb.clone(), cfg)?;
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
            multi_layer_percepton,
            span,
            residual_scale: cfg.residual_multiplier,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GraniteMoeHybrid {
    word_token_embedding: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    logits_scale: f32,
    embedding_scale: f32,
}

impl GraniteMoeHybrid {
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut GraniteMoeHybridCache,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let x = self.word_token_embedding.forward(x)?;
        let x = scale_tensor(x, self.embedding_scale)?;
        let x = self
            .blocks
            .iter()
            .enumerate()
            .try_fold(x, |x, (block_idx, block)| {
                block.forward(&x, index_pos, block_idx, cache)
            })?;
        // Final normalization
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        // Project to vocabulary size
        let logits = x.matmul(&self.word_token_embedding.embeddings().t()?)?;
        let logits = logits.to_dtype(DType::F32)?;
        // Scale the logits if needed (that's also different from Granite 1)
        let scaled_logits = if (self.logits_scale - 1.0).abs() < f32::EPSILON {
            logits
        } else {
            logits.affine(self.logits_scale as f64, 0.)?
        };

        Ok(scaled_logits)
    }

    pub fn load(vb: VarBuilder, cfg: &GraniteMoeHybridInternalConfig) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            candle::bail!(
                "layer_types length {} does not match num_hidden_layers {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }
        let blocks = cfg
            .layer_types
            .iter()
            .enumerate()
            .map(|(idx, layer_ty)| match layer_ty {
                GraniteMoeHybridLayerType::Attention => {
                    Block::load(vb.pp(format!("model.layers.{idx}")), cfg)
                }
                GraniteMoeHybridLayerType::Mamba => {
                    // TODO: Not supprting Mamba layers (blocks) for now,
                    // so we only iterate over attention layers.
                    candle::bail!(
                        "mamba layers are not yet supported in GraniteMoeHybrid inference"
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            word_token_embedding: wte,
            blocks,
            ln_f,
            logits_scale: if cfg.logits_scaling == 0.0 {
                1.0
            } else {
                1.0 / cfg.logits_scaling
            },
            embedding_scale: cfg.embedding_multiplier,
        })
    }
}

fn scale_tensor(tensor: Tensor, scale: f32) -> Result<Tensor> {
    if (scale - 1.0).abs() < f32::EPSILON {
        Ok(tensor)
    } else {
        tensor.affine(scale as f64, 0.)
    }
}
