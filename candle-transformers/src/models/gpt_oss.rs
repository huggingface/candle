use crate::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, VarBuilder};
use std::borrow::Cow;
use std::sync::Arc;

const QK_MXFP4: usize = 32;
const SWIGLU_ALPHA: f64 = 1.702;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub factor: f64,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

fn default_attention_bias() -> bool {
    true
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    150_000.0
}

fn default_swiglu_limit() -> f64 {
    7.0
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub initial_context_length: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f64,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub rope_scaling_factor: Option<f64>,
    #[serde(default)]
    pub rope_ntk_alpha: Option<f64>,
    #[serde(default)]
    pub rope_ntk_beta: Option<f64>,
    #[serde(alias = "num_local_experts", alias = "num_experts")]
    pub num_local_experts: usize,
    #[serde(alias = "num_experts_per_tok", alias = "experts_per_token")]
    pub num_experts_per_tok: usize,
}

impl Config {
    fn layer_type(&self, layer_idx: usize) -> LayerType {
        if self.layer_types.is_empty() {
            if layer_idx.is_multiple_of(2) {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        } else {
            self.layer_types[layer_idx]
        }
    }

    fn rope_factor(&self) -> f64 {
        if let Some(rope_scaling) = &self.rope_scaling {
            if rope_scaling.factor > 0.0 {
                return rope_scaling.factor;
            }
        }
        self.rope_scaling_factor.unwrap_or(1.0)
    }

    fn rope_ntk_alpha(&self) -> f64 {
        if let Some(rope_scaling) = &self.rope_scaling {
            if let Some(beta_slow) = rope_scaling.beta_slow {
                return beta_slow;
            }
        }
        self.rope_ntk_alpha.unwrap_or(1.0)
    }

    fn rope_ntk_beta(&self) -> f64 {
        if let Some(rope_scaling) = &self.rope_scaling {
            if let Some(beta_fast) = rope_scaling.beta_fast {
                return beta_fast;
            }
        }
        self.rope_ntk_beta.unwrap_or(32.0)
    }

    fn rope_initial_context_length(&self) -> usize {
        self.initial_context_length
            .or_else(|| {
                self.rope_scaling
                    .as_ref()
                    .and_then(|r| r.original_max_position_embeddings)
            })
            .unwrap_or(4096)
    }

    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings.unwrap_or_else(|| {
            (self.rope_initial_context_length() as f64 * self.rope_factor()).round() as usize
        })
    }

    fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 {
            candle::bail!("num_hidden_layers must be > 0")
        }
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers {
            candle::bail!(
                "layer_types len ({}) must match num_hidden_layers ({})",
                self.layer_types.len(),
                self.num_hidden_layers
            )
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            candle::bail!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads
            )
        }
        if !self.hidden_size.is_multiple_of(QK_MXFP4) {
            candle::bail!(
                "hidden_size ({}) must be divisible by {} for MXFP4 expert weights",
                self.hidden_size,
                QK_MXFP4
            )
        }
        if !self.intermediate_size.is_multiple_of(QK_MXFP4) {
            candle::bail!(
                "intermediate_size ({}) must be divisible by {} for MXFP4 expert weights",
                self.intermediate_size,
                QK_MXFP4
            )
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let base = cfg.rope_theta;
        let head_dim = cfg.head_dim;
        let d_half = head_dim / 2;
        let scaling_factor = cfg.rope_factor();
        let ntk_alpha = cfg.rope_ntk_alpha();
        let ntk_beta = cfg.rope_ntk_beta();
        let initial_ctx = cfg.rope_initial_context_length() as f64;

        let mut concentration = 1.0f64;
        let inv_freq: Vec<f32> = if scaling_factor > 1.0 {
            concentration = 0.1 * scaling_factor.ln() + 1.0;
            let low = d_half as f64 * (initial_ctx / (ntk_beta * 2.0 * std::f64::consts::PI)).ln()
                / base.ln();
            let high = d_half as f64
                * (initial_ctx / (ntk_alpha * 2.0 * std::f64::consts::PI)).ln()
                / base.ln();

            (0..d_half)
                .map(|i| {
                    let freq = base.powf((2 * i) as f64 / head_dim as f64);
                    let interpolation = 1.0 / (scaling_factor * freq);
                    let extrapolation = 1.0 / freq;
                    let ramp = (i as f64 - low) / (high - low);
                    let mask = 1.0 - ramp.clamp(0.0, 1.0);
                    (interpolation * (1.0 - mask) + extrapolation * mask) as f32
                })
                .collect()
        } else {
            (0..d_half)
                .map(|i| (1.0 / base.powf((2 * i) as f64 / head_dim as f64)) as f32)
                .collect()
        };

        let max_seq_len = cfg.max_seq_len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, d_half), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let concentration = concentration as f32;
        let cos = freqs
            .cos()?
            .affine(concentration as f64, 0.)?
            .to_dtype(dtype)?;
        let sin = freqs
            .sin()?
            .affine(concentration as f64, 0.)?
            .to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sinks: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    scale: f64,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Attention {
    fn new(cfg: &Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let hidden_size = num_heads * head_dim;

        Ok(Self {
            q_proj: linear_b(
                cfg.hidden_size,
                num_heads * head_dim,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(
                hidden_size,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            sinks: vb.get((num_heads,), "sinks")?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size,
            scale: (head_dim as f64).sqrt().recip(),
            rotary,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        if let Some(mask) = mask {
            let mask = if mask.dtype() != scores.dtype() {
                mask.to_dtype(scores.dtype())?
            } else {
                mask.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }

        let sinks = self
            .sinks
            .to_dtype(scores.dtype())?
            .reshape((1, self.num_heads, 1, 1))?
            .broadcast_as((b, self.num_heads, l, 1))?;
        let logits = Tensor::cat(&[&scores, &sinks], D::Minus1)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let attn_weights = probs.narrow(D::Minus1, 0, scores.dim(D::Minus1)?)?;
        let attn_weights = attn_weights.to_dtype(v.dtype())?;
        let y = attn_weights.matmul(&v)?;
        y.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

fn pack_mxfp4_blocks(blocks: &[u8], scales: &[u8]) -> Vec<u8> {
    fn reorder_block_pairwise_to_ggml(src: &[u8]) -> [u8; 16] {
        debug_assert_eq!(src.len(), 16);
        let mut dst = [0u8; 16];
        for k in 0..16 {
            let lo = if k % 2 == 0 {
                src[k / 2] & 0x0f
            } else {
                (src[k / 2] >> 4) & 0x0f
            };
            let k2 = k + 16;
            let hi = if k2 % 2 == 0 {
                src[k2 / 2] & 0x0f
            } else {
                (src[k2 / 2] >> 4) & 0x0f
            };
            dst[k] = lo | (hi << 4);
        }
        dst
    }

    let mut packed = Vec::with_capacity(scales.len() * 17);
    for (block_idx, &e) in scales.iter().enumerate() {
        packed.push(e);
        let start = block_idx * 16;
        let reordered = reorder_block_pairwise_to_ggml(&blocks[start..start + 16]);
        packed.extend_from_slice(&reordered);
    }
    packed
}

fn load_mxfp4_expert_qtensor(
    num_experts: usize,
    rows: usize,
    cols: usize,
    vb_u8: VarBuilder,
    blocks_name: &str,
    scales_name: &str,
    device: &Device,
) -> Result<Arc<QTensor>> {
    let blocks = vb_u8.get_unchecked_dtype(blocks_name, DType::U8)?;
    let scales = vb_u8.get_unchecked_dtype(scales_name, DType::U8)?;
    let (b_e, b_rows, b_col_blocks, b_block_bytes) = blocks.dims4()?;
    let (s_e, s_rows, s_col_blocks) = scales.dims3()?;
    if b_e != num_experts || s_e != num_experts {
        candle::bail!(
            "{blocks_name}/{scales_name}: expected num_experts={num_experts}, got {b_e}/{s_e}"
        )
    }
    if b_rows != rows || s_rows != rows {
        candle::bail!("{blocks_name}/{scales_name}: expected rows={rows}, got {b_rows}/{s_rows}")
    }
    if b_col_blocks * QK_MXFP4 != cols {
        candle::bail!(
            "{blocks_name}: expected cols={cols}, got {} blocks of {QK_MXFP4}",
            b_col_blocks
        )
    }
    if s_col_blocks != b_col_blocks {
        candle::bail!(
            "{blocks_name}/{scales_name}: mismatched block count {b_col_blocks} != {s_col_blocks}"
        )
    }
    if b_block_bytes != 16 {
        candle::bail!("{blocks_name}: expected last dim 16, got {b_block_bytes}")
    }

    let blocks = blocks.flatten_all()?.to_vec1::<u8>()?;
    let scales = scales.flatten_all()?.to_vec1::<u8>()?;
    if blocks.len() != scales.len() * 16 {
        candle::bail!(
            "{blocks_name}/{scales_name}: invalid flattened sizes {} and {}",
            blocks.len(),
            scales.len()
        )
    }
    let packed = pack_mxfp4_blocks(&blocks, &scales);
    let storage = QStorage::from_data(Cow::Owned(packed), device, GgmlDType::MXFP4)?;
    let q = QTensor::new(storage, (num_experts, rows, cols))?;
    Ok(Arc::new(q))
}

#[derive(Debug, Clone)]
struct Mlp {
    router: Linear,
    gate_up_proj: Arc<QTensor>,
    down_proj: Arc<QTensor>,
    gate_up_bias: Tensor,
    down_bias: Tensor,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    swiglu_limit: f64,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, vb_u8: VarBuilder) -> Result<Self> {
        let router = linear_b(
            cfg.hidden_size,
            cfg.num_local_experts,
            true,
            vb.pp("router"),
        )?;
        let experts_vb = vb.pp("experts");
        let experts_u8_vb = vb_u8.pp("experts");
        let gate_up_proj = load_mxfp4_expert_qtensor(
            cfg.num_local_experts,
            2 * cfg.intermediate_size,
            cfg.hidden_size,
            experts_u8_vb.clone(),
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            vb.device(),
        )?;
        let down_proj = load_mxfp4_expert_qtensor(
            cfg.num_local_experts,
            cfg.hidden_size,
            cfg.intermediate_size,
            experts_u8_vb,
            "down_proj_blocks",
            "down_proj_scales",
            vb.device(),
        )?;
        let gate_up_bias = experts_vb
            .get(
                (cfg.num_local_experts, 2 * cfg.intermediate_size),
                "gate_up_proj_bias",
            )?
            .to_dtype(DType::F32)?;
        let down_bias = experts_vb
            .get((cfg.num_local_experts, cfg.hidden_size), "down_proj_bias")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            router,
            gate_up_proj,
            down_proj,
            gate_up_bias,
            down_bias,
            num_experts_per_tok: cfg.num_experts_per_tok,
            intermediate_size: cfg.intermediate_size,
            swiglu_limit: cfg.swiglu_limit,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden_dim) = x.dims3()?;
        let x = x.reshape(((), hidden_dim))?;
        let num_tokens = x.dim(0)?;
        let x_f32 = x.to_dtype(DType::F32)?;

        let router_logits = self.router.forward(&x_f32)?;
        let topk_ids = router_logits
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let topk_logits = router_logits.gather(&topk_ids, D::Minus1)?;
        let topk_weights = candle_nn::ops::softmax_last_dim(&topk_logits.to_dtype(DType::F32)?)?;

        let gate_up = self.gate_up_proj.indexed_moe_forward(&x_f32, &topk_ids)?;
        let ids_flat = topk_ids.flatten_all()?;
        let gate_up_bias = self.gate_up_bias.index_select(&ids_flat, 0)?.reshape((
            num_tokens,
            self.num_experts_per_tok,
            2 * self.intermediate_size,
        ))?;
        let gate_up = gate_up.broadcast_add(&gate_up_bias)?;

        let gate_up = gate_up.reshape((
            num_tokens,
            self.num_experts_per_tok,
            self.intermediate_size,
            2,
        ))?;
        let gate = gate_up.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let up = gate_up.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
        let gate = gate.clamp(f32::NEG_INFINITY, self.swiglu_limit as f32)?;
        let up = up.clamp(-(self.swiglu_limit as f32), self.swiglu_limit as f32)?;
        let glu = (&gate * candle_nn::ops::sigmoid(&gate.affine(SWIGLU_ALPHA, 0.)?)?)?;
        let down_input = (glu * up.affine(1., 1.)?)?;

        let down = self.down_proj.indexed_moe_forward(&down_input, &topk_ids)?;
        let down_bias = self.down_bias.index_select(&ids_flat, 0)?.reshape((
            num_tokens,
            self.num_experts_per_tok,
            hidden_dim,
        ))?;
        let down = down.broadcast_add(&down_bias)?;
        let down = down.broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?;
        let out = down.sum(D::Minus2)?;
        out.reshape((b, seq_len, hidden_dim))?.to_dtype(x.dtype())
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    layer_type: LayerType,
}

impl DecoderLayer {
    fn new(
        layer_idx: usize,
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
        vb_u8: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"), vb_u8.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            layer_type: cfg.layer_type(layer_idx),
        })
    }

    fn uses_sliding_window(&self) -> bool {
        self.layer_type == LayerType::SlidingAttention
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: Option<usize>,
    dtype: DType,
    device: Device,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let vb_u8 = vb.clone().to_dtype(DType::U8).set_device(Device::Cpu);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        let vb_u8_l = vb_u8.pp("model.layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                layer_idx,
                cfg,
                rotary.clone(),
                vb_l.pp(layer_idx),
                vb_u8_l.pp(layer_idx),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            sliding_window: cfg.sliding_window,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    fn causal_mask(
        &self,
        batch: usize,
        tgt: usize,
        offset: usize,
        sliding_window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() && tgt == 1 {
            return Ok(None);
        }
        if let Some(sw) = sliding_window {
            if tgt == 1 && offset + 1 <= sw {
                return Ok(None);
            }
        }

        let src = tgt + offset;
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                let abs_i = i + offset;
                (0..src).map(move |j| {
                    let past_ok = j <= abs_i;
                    let sw_ok = match sliding_window {
                        Some(sw) => (abs_i as isize - j as isize) < sw as isize,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.0
                    } else {
                        minf
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (batch, 1, tgt, src), &self.device)?;
        Ok(Some(mask.to_dtype(self.dtype)?))
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let full_mask = self.causal_mask(b, l, offset, None)?;
        let sliding_mask = if self.layers.iter().any(DecoderLayer::uses_sliding_window) {
            self.causal_mask(b, l, offset, self.sliding_window)?
        } else {
            None
        };

        for layer in &mut self.layers {
            let mask = if layer.uses_sliding_window() {
                sliding_mask.as_ref()
            } else {
                full_mask.as_ref()
            };
            h = layer.forward(&h, mask, offset)?;
        }
        self.norm.forward(&h)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_mxfp4_blocks, Config, LayerType};

    fn decode_pairwise_block(src: &[u8; 16]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..16 {
            out[2 * i] = src[i] & 0x0f;
            out[2 * i + 1] = (src[i] >> 4) & 0x0f;
        }
        out
    }

    fn decode_ggml_block(src: &[u8; 16]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..16 {
            out[i] = src[i] & 0x0f;
            out[i + 16] = (src[i] >> 4) & 0x0f;
        }
        out
    }

    #[test]
    fn parse_config_hf_style() {
        let cfg: Config = serde_json::from_str(
            r#"{
                "model_type":"gpt_oss",
                "vocab_size":201088,
                "hidden_size":2880,
                "intermediate_size":2880,
                "num_hidden_layers":2,
                "num_attention_heads":64,
                "num_key_value_heads":8,
                "head_dim":64,
                "attention_bias":true,
                "max_position_embeddings":131072,
                "initial_context_length":4096,
                "sliding_window":128,
                "tie_word_embeddings":false,
                "rope_theta":150000.0,
                "rms_norm_eps":1e-5,
                "swiglu_limit":7.0,
                "num_local_experts":32,
                "num_experts_per_tok":4,
                "layer_types":["sliding_attention","full_attention"],
                "rope_scaling":{
                    "factor":32.0,
                    "beta_fast":32.0,
                    "beta_slow":1.0,
                    "original_max_position_embeddings":4096
                }
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.num_local_experts, 32);
        assert_eq!(cfg.num_experts_per_tok, 4);
        assert_eq!(cfg.layer_types[0], LayerType::SlidingAttention);
        assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
        cfg.validate().unwrap();
    }

    #[test]
    fn pack_mxfp4_layout() {
        let scales = vec![127u8, 130u8];
        let blocks = vec![
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44,
            0x33, 0x22, 0x11, 0x00,
        ];
        let packed = pack_mxfp4_blocks(&blocks, &scales);
        assert_eq!(packed.len(), 34);
        assert_eq!(packed[0], scales[0]);
        assert_eq!(packed[17], scales[1]);

        let src0: [u8; 16] = blocks[0..16].try_into().unwrap();
        let src1: [u8; 16] = blocks[16..32].try_into().unwrap();
        let dst0: [u8; 16] = packed[1..17].try_into().unwrap();
        let dst1: [u8; 16] = packed[18..34].try_into().unwrap();

        assert_eq!(decode_pairwise_block(&src0), decode_ggml_block(&dst0));
        assert_eq!(decode_pairwise_block(&src1), decode_ggml_block(&dst1));
    }
}
