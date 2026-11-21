use std::sync::Arc;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, kv_cache::ConcatKvCache, linear, linear_b, rms_norm, Activation, Embedding, Linear,
    Module, RmsNorm, VarBuilder,
};

use super::config::TextConfig;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    mrope_section: Option<[usize; 3]>,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        mrope_section: Vec<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mrope_section = match mrope_section.as_slice() {
            [] => None,
            [t, h, w] => {
                let expected = head_dim / 2;
                let Some(sum) = t.checked_add(*h).and_then(|sum| sum.checked_add(*w)) else {
                    candle::bail!("mrope_section {:?} overflows usize", mrope_section);
                };
                if sum != expected {
                    candle::bail!(
                        "mrope_section {:?} sums to {} but head_dim / 2 = {}",
                        mrope_section,
                        sum,
                        expected
                    );
                }
                Some([*t, *h, *w])
            }
            _ => {
                candle::bail!(
                    "mrope_section must be empty or have 3 entries, got {}",
                    mrope_section.len()
                )
            }
        };
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

        Ok(Self {
            cos,
            sin,
            mrope_section,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.build_cos_sin(position_ids)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let q_embed = apply_rotation(q, &cos, &sin)?;
        let k_embed = apply_rotation(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    fn build_cos_sin(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (three, batch, seq_len) = position_ids.dims3()?;
        if three != 3 {
            candle::bail!("position_ids axis 0 must be 3, got {three}");
        }
        let half_dim = self.head_dim / 2;
        let Some(mrope_section) = self.mrope_section else {
            let pos = position_ids.i(0)?.flatten_all()?;
            let cos = self
                .cos
                .index_select(&pos, 0)?
                .reshape((batch, seq_len, half_dim))?;
            let sin = self
                .sin
                .index_select(&pos, 0)?
                .reshape((batch, seq_len, half_dim))?;
            let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
            let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
            return Ok((cos, sin));
        };
        let cos_per_axis = self.gather_axis(position_ids, batch, seq_len, &self.cos)?;
        let sin_per_axis = self.gather_axis(position_ids, batch, seq_len, &self.sin)?;
        let cos = apply_interleaved_mrope(&cos_per_axis, mrope_section, half_dim)?;
        let sin = apply_interleaved_mrope(&sin_per_axis, mrope_section, half_dim)?;
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        Ok((cos, sin))
    }

    fn gather_axis(
        &self,
        position_ids: &Tensor,
        batch: usize,
        seq_len: usize,
        table: &Tensor,
    ) -> Result<[Tensor; 3]> {
        let half_dim = self.head_dim / 2;
        let gather = |axis: usize| -> Result<Tensor> {
            let pos = position_ids.i(axis)?.flatten_all()?;
            table
                .index_select(&pos, 0)?
                .reshape((batch, seq_len, half_dim))
        };
        Ok([gather(0)?, gather(1)?, gather(2)?])
    }
}

fn apply_interleaved_mrope(
    per_axis: &[Tensor; 3],
    mrope_section: [usize; 3],
    half_dim: usize,
) -> Result<Tensor> {
    let mut axis_for: Vec<usize> = vec![0usize; half_dim];
    for (dim, offset) in [(1usize, 1usize), (2usize, 2usize)] {
        let length = mrope_section[dim].saturating_mul(3);
        let mut idx = offset;
        while idx < length && idx < half_dim {
            axis_for[idx] = dim;
            idx += 3;
        }
    }
    let mut bands: Vec<Tensor> = Vec::new();
    let mut run_start = 0usize;
    while run_start < half_dim {
        let current_axis = axis_for[run_start];
        let mut run_end = run_start + 1;
        while run_end < half_dim && axis_for[run_end] == current_axis {
            run_end += 1;
        }
        bands.push(per_axis[current_axis].narrow(D::Minus1, run_start, run_end - run_start)?);
        run_start = run_end;
    }
    Tensor::cat(&bands, D::Minus1)
}

fn apply_rotation(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
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
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    softmax_scale: f64,
    kv_cache: ConcatKvCache,
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
            num_kv_groups: num_heads / num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_emb,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;

        let (q, k) = self.rotary_emb.forward(&q, &k, position_ids)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
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

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, position_ids)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
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

        let mrope_section = cfg
            .rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_default();
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            mrope_section,
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
        &mut self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                position_ids,
            )?;

            if let (Some(visual_pos_masks), Some(deepstack)) =
                (visual_pos_masks, deepstack_visual_embeds)
            {
                if i < deepstack.len() {
                    xs = Self::deepstack_process(xs, visual_pos_masks, &deepstack[i])?;
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

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageGrid {
    pub grid_h: usize,
    pub grid_w: usize,
}

fn position_id(pos: usize) -> Result<u32> {
    if pos > u32::MAX as usize {
        candle::bail!("position id {pos} exceeds u32::MAX");
    }
    Ok(pos as u32)
}

fn offset_position_id(base: usize, offset: usize) -> Result<u32> {
    let Some(pos) = base.checked_add(offset) else {
        candle::bail!("position id overflow");
    };
    position_id(pos)
}

/// Build 3D M-RoPE position IDs for a prefill that may contain text
/// and image tokens. Returns a `(3, batch, seq_len)` U32 tensor and
/// the highest position used in each batch row (decode for row `b`
/// should resume at `max_positions[b] + 1`).
///
/// `image_pad_spans[b]` lists `(start, end_exclusive)` ranges holding
/// image-pad placeholders in batch `b`, in order. `image_grids[b][k]`
/// is the grid for the k-th image in batch `b`; `grid_h * grid_w`
/// must equal the matching span length.
pub fn compute_mrope_position_ids(
    batch: usize,
    seq_len: usize,
    image_pad_spans: &[Vec<(usize, usize)>],
    image_grids: &[Vec<ImageGrid>],
    device: &Device,
) -> Result<(Tensor, Vec<usize>)> {
    if image_pad_spans.len() != batch {
        candle::bail!(
            "image_pad_spans has {} entries but batch is {batch}",
            image_pad_spans.len()
        );
    }
    if image_grids.len() != batch {
        candle::bail!(
            "image_grids has {} entries but batch is {batch}",
            image_grids.len()
        );
    }
    let mut pos_t = vec![0u32; batch * seq_len];
    let mut pos_h = vec![0u32; batch * seq_len];
    let mut pos_w = vec![0u32; batch * seq_len];
    let mut max_per_batch: Vec<usize> = vec![0; batch];
    for (b, spans) in image_pad_spans.iter().enumerate() {
        let grids = &image_grids[b];
        if spans.len() != grids.len() {
            candle::bail!(
                "batch {b} has {} image spans but {} grids",
                spans.len(),
                grids.len()
            );
        }
        let mut prev_end = 0usize;
        for (span_idx, (&(span_start, span_end), &grid)) in
            spans.iter().zip(grids.iter()).enumerate()
        {
            if span_start >= span_end {
                candle::bail!(
                    "batch {b} span {span_idx} has invalid range {:?}",
                    (span_start, span_end)
                );
            }
            if span_end > seq_len {
                candle::bail!(
                    "batch {b} span {span_idx} ends at {span_end} but seq_len is {seq_len}"
                );
            }
            if span_start < prev_end {
                candle::bail!(
                    "batch {b} spans must be sorted and non-overlapping; \
                     span {} ends at {prev_end} but span {span_idx} starts at {span_start}",
                    span_idx - 1,
                );
            }
            if grid.grid_h == 0 || grid.grid_w == 0 {
                candle::bail!(
                    "batch {b} grid {span_idx} has invalid shape {}x{}",
                    grid.grid_h,
                    grid.grid_w
                );
            }
            let Some(expected_len) = grid.grid_h.checked_mul(grid.grid_w) else {
                candle::bail!(
                    "batch {b} grid {span_idx} shape {}x{} overflows",
                    grid.grid_h,
                    grid.grid_w
                );
            };
            let span_len = span_end - span_start;
            if span_len != expected_len {
                candle::bail!(
                    "batch {b} span {:?} length {span_len} does not match grid {}x{} = {expected_len}",
                    (span_start, span_end),
                    grid.grid_h,
                    grid.grid_w,
                );
            }
            prev_end = span_end;
        }
        let row_start = b * seq_len;
        let mut counter: usize = 0;
        let mut span_iter = spans.iter().enumerate().peekable();
        let mut s = 0usize;
        while s < seq_len {
            if let Some(&(grid_idx, &(span_start, span_end))) = span_iter.peek() {
                if s == span_start {
                    let grid = grids[grid_idx];
                    let span_len = span_end - span_start;
                    let offset = position_id(counter)?;
                    for vision_idx in 0..span_len {
                        let token_idx = row_start + span_start + vision_idx;
                        let h_pos = offset_position_id(counter, vision_idx / grid.grid_w)?;
                        let w_pos = offset_position_id(counter, vision_idx % grid.grid_w)?;
                        pos_t[token_idx] = offset;
                        pos_h[token_idx] = h_pos;
                        pos_w[token_idx] = w_pos;
                    }
                    let span_max = grid.grid_h.max(grid.grid_w);
                    let Some(next_counter) = counter.checked_add(span_max) else {
                        candle::bail!("batch {b} position counter overflow");
                    };
                    counter = next_counter;
                    max_per_batch[b] = max_per_batch[b].max(counter.saturating_sub(1));
                    s = span_end;
                    span_iter.next();
                    continue;
                }
            }
            let token_idx = row_start + s;
            let c = position_id(counter)?;
            pos_t[token_idx] = c;
            pos_h[token_idx] = c;
            pos_w[token_idx] = c;
            max_per_batch[b] = max_per_batch[b].max(counter);
            let Some(next_counter) = counter.checked_add(1) else {
                candle::bail!("batch {b} position counter overflow");
            };
            counter = next_counter;
            s += 1;
        }
    }
    let to_tensor =
        |v: Vec<u32>| -> Result<Tensor> { Tensor::from_vec(v, (batch, seq_len), device) };
    let pos = Tensor::stack(
        &[to_tensor(pos_t)?, to_tensor(pos_h)?, to_tensor(pos_w)?],
        0,
    )?;
    Ok((pos, max_per_batch))
}

/// Build a `(3, batch, 1)` U32 position-id tensor where each batch
/// row carries `t = h = w = positions[b]`. For incremental decode,
/// pass `positions[b] = previous_max + 1`.
pub fn single_token_position_ids(positions: &[usize], device: &Device) -> Result<Tensor> {
    if positions.is_empty() {
        candle::bail!("single_token_position_ids needs at least one batch row");
    }
    let batch = positions.len();
    let values: Vec<u32> = positions
        .iter()
        .map(|&p| position_id(p))
        .collect::<Result<_>>()?;
    let t = Tensor::from_vec(values.clone(), (batch, 1), device)?;
    let h = Tensor::from_vec(values.clone(), (batch, 1), device)?;
    let w = Tensor::from_vec(values, (batch, 1), device)?;
    Tensor::stack(&[t, h, w], 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_position_ids_pure_text() -> Result<()> {
        let device = Device::Cpu;
        let (ids, max_per_batch) =
            compute_mrope_position_ids(1, 5, &[Vec::new()], &[Vec::new()], &device)?;
        assert_eq!(ids.dims(), &[3, 1, 5]);
        assert_eq!(max_per_batch, vec![4]);
        for axis in 0..3 {
            let row: Vec<u32> = ids.i(axis)?.i(0)?.to_vec1()?;
            assert_eq!(row, vec![0, 1, 2, 3, 4], "axis {axis}");
        }
        Ok(())
    }

    #[test]
    fn mrope_position_ids_image_in_middle() -> Result<()> {
        let device = Device::Cpu;
        let grid = ImageGrid {
            grid_h: 2,
            grid_w: 3,
        };
        let (ids, max_per_batch) =
            compute_mrope_position_ids(1, 9, &[vec![(2, 8)]], &[vec![grid]], &device)?;
        assert_eq!(ids.dims(), &[3, 1, 9]);
        let t: Vec<u32> = ids.i(0)?.i(0)?.to_vec1()?;
        let h: Vec<u32> = ids.i(1)?.i(0)?.to_vec1()?;
        let w: Vec<u32> = ids.i(2)?.i(0)?.to_vec1()?;
        assert_eq!(&t[..2], &[0, 1]);
        assert_eq!(&h[..2], &[0, 1]);
        assert_eq!(&w[..2], &[0, 1]);
        assert_eq!(&t[2..8], &[2, 2, 2, 2, 2, 2]);
        assert_eq!(&h[2..8], &[2, 2, 2, 3, 3, 3]);
        assert_eq!(&w[2..8], &[2, 3, 4, 2, 3, 4]);
        assert_eq!(t[8], 5);
        assert_eq!(h[8], 5);
        assert_eq!(w[8], 5);
        assert_eq!(max_per_batch, vec![5]);
        Ok(())
    }

    #[test]
    fn mrope_position_ids_per_batch_layouts_independent() -> Result<()> {
        let device = Device::Cpu;
        let grid_a = ImageGrid {
            grid_h: 1,
            grid_w: 2,
        };
        let grid_b = ImageGrid {
            grid_h: 2,
            grid_w: 2,
        };
        let (ids, max_per_batch) = compute_mrope_position_ids(
            2,
            5,
            &[vec![(1, 3)], vec![(0, 4)]],
            &[vec![grid_a], vec![grid_b]],
            &device,
        )?;
        assert_eq!(ids.dims(), &[3, 2, 5]);
        assert_eq!(max_per_batch, vec![4, 2]);
        Ok(())
    }

    #[test]
    fn mrope_position_ids_span_length_mismatch_errs() {
        let device = Device::Cpu;
        let grid = ImageGrid {
            grid_h: 2,
            grid_w: 3,
        };
        let result = compute_mrope_position_ids(1, 4, &[vec![(0, 4)]], &[vec![grid]], &device);
        assert!(result.is_err());
    }

    #[test]
    fn mrope_position_ids_rejects_malformed_spans() {
        let device = Device::Cpu;
        let grid = ImageGrid {
            grid_h: 1,
            grid_w: 2,
        };

        let reversed = compute_mrope_position_ids(1, 5, &[vec![(3, 2)]], &[vec![grid]], &device);
        assert!(reversed.is_err());

        let out_of_bounds =
            compute_mrope_position_ids(1, 5, &[vec![(3, 6)]], &[vec![grid]], &device);
        assert!(out_of_bounds.is_err());

        let overlapping =
            compute_mrope_position_ids(1, 5, &[vec![(1, 3), (2, 4)]], &[vec![grid, grid]], &device);
        assert!(overlapping.is_err());
    }

    #[test]
    fn mrope_position_ids_rejects_empty_grid() {
        let device = Device::Cpu;
        let grid = ImageGrid {
            grid_h: 0,
            grid_w: 2,
        };
        let result = compute_mrope_position_ids(1, 2, &[vec![(0, 2)]], &[vec![grid]], &device);
        assert!(result.is_err());
    }

    #[test]
    fn single_token_position_ids_per_batch() -> Result<()> {
        let device = Device::Cpu;
        let ids = single_token_position_ids(&[7, 11], &device)?;
        assert_eq!(ids.dims(), &[3, 2, 1]);
        for axis in 0..3 {
            assert_eq!(ids.i(axis)?.i(0)?.i(0)?.to_scalar::<u32>()?, 7);
            assert_eq!(ids.i(axis)?.i(1)?.i(0)?.to_scalar::<u32>()?, 11);
        }
        Ok(())
    }

    #[test]
    fn rope_table_construction_matches_legacy_for_1d_mode() -> Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(10_000f32, 64, 32, Vec::new(), &device, DType::F32)?;
        let legacy_cos = rope.cos.narrow(0, 7, 1)?;
        let pos = Tensor::from_vec(vec![7u32; 3], (3, 1, 1), &device)?;
        let (cos, _sin) = rope.build_cos_sin(&pos)?;
        let cos_half = cos.narrow(D::Minus1, 0, 32)?.squeeze(0)?.squeeze(0)?;
        let cos_half_vec: Vec<f32> = cos_half.to_vec1()?;
        let legacy_vec: Vec<f32> = legacy_cos.squeeze(0)?.to_vec1()?;
        assert_eq!(cos_half_vec, legacy_vec);
        Ok(())
    }

    #[test]
    fn interleaved_mrope_axis_pattern_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let head_dim_half = 16usize;
        let make = |base: f32| -> Result<Tensor> {
            let v: Vec<f32> = (0..head_dim_half).map(|i| base + i as f32).collect();
            Tensor::from_vec(v, (1usize, 1, head_dim_half), &device)
        };
        let per_axis = [make(100.0)?, make(200.0)?, make(300.0)?];
        let section = [8usize, 4, 4];
        let out = apply_interleaved_mrope(&per_axis, section, head_dim_half)?;
        let row: Vec<f32> = out.i(0)?.i(0)?.to_vec1()?;
        let expected: Vec<f32> = vec![
            100.0, 201.0, 302.0, 103.0, 204.0, 305.0, 106.0, 207.0, 308.0, 109.0, 210.0, 311.0,
            112.0, 113.0, 114.0, 115.0,
        ];
        assert_eq!(row, expected);
        Ok(())
    }

    #[test]
    fn interleaved_mrope_equals_1d_when_all_axes_equal() -> Result<()> {
        let device = Device::Cpu;
        let mrope_section = vec![24usize, 20, 20];
        let head_dim = 128usize;
        let rope =
            RotaryEmbedding::new(10_000f32, head_dim, 64, mrope_section, &device, DType::F32)?;
        let rope_1d =
            RotaryEmbedding::new(10_000f32, head_dim, 64, Vec::new(), &device, DType::F32)?;
        let pos = Tensor::from_vec(vec![5u32, 5u32, 5u32], (3, 1, 1), &device)?;
        let (cos_mrope, _) = rope.build_cos_sin(&pos)?;
        let (cos_1d, _) = rope_1d.build_cos_sin(&pos)?;
        let mrope_vec: Vec<f32> = cos_mrope.squeeze(0)?.squeeze(0)?.to_vec1()?;
        let one_d_vec: Vec<f32> = cos_1d.squeeze(0)?.squeeze(0)?.to_vec1()?;
        assert_eq!(mrope_vec, one_d_vec);
        Ok(())
    }
}
