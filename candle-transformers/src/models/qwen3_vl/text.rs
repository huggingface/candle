use std::sync::{Arc, Mutex};

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    embedding, kv_cache::IncrementalKvCache, linear, linear_b, rms_norm, Activation, Embedding,
    Linear, Module, RmsNorm, VarBuilder,
};

use super::config::TextConfig;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _qh, seq_len, _n_embd) = q.dims4()?;

        let rope = candle_nn::rotary_emb::rope;

        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.sin.narrow(0, *offset, seq_len)?;
            let q_embed = rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            let k_embed = rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

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
    rotary_emb: Arc<RotaryEmbedding>,
    n_kv_groups: usize,
    softmax_scale: f64,
    kv_cache: Arc<Mutex<IncrementalKvCache>>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let q_proj = linear_b(hidden_sz, num_heads * cfg.head_dim, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            false,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            false,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(num_heads * cfg.head_dim, hidden_sz, false, vb.pp("o_proj"))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_emb,
            n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: Arc::new(Mutex::new(IncrementalKvCache::new(
                2,
                cfg.max_position_embeddings,
            ))),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let (k, v) = self
            .kv_cache
            .lock()
            .expect("Need a lock because of the deepstack injection")
            .append(&k, &v)?;

        let k = crate::utils::repeat_kv(k, self.n_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.n_kv_groups)?.contiguous()?;

        let mut attn_output = {
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offsets)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Qwen3VLTextModel {
    embed_tokens: Embedding,
    pub(super) norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    lm_head: Linear,
    pub(super) dtype: DType,
    pub(super) num_attn_heads: usize,
}

impl Qwen3VLTextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            vb.device(),
            vb_m.dtype(),
        )?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
            )?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
            dtype: vb.dtype(),
            num_attn_heads: cfg.num_attention_heads,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
            )?;

            // Integrate DeepStack visual features when provided.
            if let (Some(visual_pos_masks), Some(deepstack)) =
                (visual_pos_masks, deepstack_visual_embeds)
            {
                if i < deepstack.len() {
                    xs = self.deepstack_process(xs, visual_pos_masks, &deepstack[i])?;
                }
            }
        }

        xs = xs.apply(&self.norm)?;

        self.lm_head
            .forward(&xs)?
            .i((.., seq_len - 1, ..))?
            .contiguous()
    }

    fn deepstack_process(
        &self,
        hidden_states: Tensor,
        visual_pos_masks: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();

        let mask = visual_pos_masks.to_device(device)?.to_dtype(DType::F32)?;
        let mask_flat = mask.flatten_all()?;

        let masked_count = mask_flat.sum_all()?.to_scalar::<f32>()? as usize;
        let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

        if masked_count == 0 {
            if visual_embeds.dim(0)? != 0 {
                candle::bail!(
                    "DeepStack visual embeds ({}) provided but mask is empty",
                    visual_embeds.dim(0)?
                );
            }
            return Ok(hidden_states);
        }

        if visual_embeds.dim(0)? != masked_count {
            candle::bail!(
                "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
                visual_embeds.dim(0)?,
                masked_count
            );
        }

        let (batch, seq, hidden) = hidden_states.dims3()?;
        let total_positions = batch * seq;
        let mut hidden_flat = hidden_states.reshape((total_positions, hidden))?;

        let prefix = mask_flat.cumsum(0)?;
        let rank = (prefix - &mask_flat)?.mul(&mask_flat)?;
        let rank_u32 = rank.to_dtype(DType::U32)?;

        let positions = Tensor::arange(0u32, total_positions as u32, device)?;
        let positions_f32 = positions.to_dtype(DType::F32)?;
        let masked_positions = positions_f32.mul(&mask_flat)?;

        let mut position_per_rank = Tensor::zeros((masked_count,), DType::F32, device)?;
        position_per_rank = position_per_rank.scatter_add(&rank_u32, &masked_positions, 0)?;
        let position_per_rank = position_per_rank.to_dtype(DType::U32)?;

        let linear_index = position_per_rank.unsqueeze(1)?.repeat((1, hidden))?;

        hidden_flat = hidden_flat.scatter_add(&linear_index, &visual_embeds, 0)?;
        hidden_flat.reshape((batch, seq, hidden))
    }
}
