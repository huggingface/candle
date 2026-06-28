//! Gemma 4 audio encoder (Conformer-based).
//!
//! SSCP conv projection + conformer blocks with chunked attention,
//! relative position embeddings, light conv1d, and clippable linears.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv2d, Conv2dConfig, VarBuilder};

use super::config::Gemma4AudioConfig;

// ── RmsNorm (standard, no +1 offset for audio) ─────────────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

// ── LayerNorm (for SSCP conv blocks) ────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f64,
    dim: usize,
}

impl LayerNorm {
    fn new(dim: usize, eps: f64) -> Self {
        Self { eps, dim }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?;
        let var = (x.sqr()?.sum_keepdim(D::Minus1)? / self.dim as f64)?;
        let x = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x.to_dtype(x_dtype)
    }
}

// ── SSCP Conv Blocks ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SSCPConvBlock {
    conv: Conv2d,
    norm: LayerNorm,
    manual_padding: (usize, usize, usize, usize), // (f_left, f_right, t_top, t_bottom)
    time_stride: usize,
    #[allow(dead_code)]
    out_channels: usize,
}

impl SSCPConvBlock {
    fn new(
        cfg: &Gemma4AudioConfig,
        idx: usize,
        input_freq_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channels = if idx == 0 {
            1
        } else {
            cfg.sscp_conv_channel_size[idx - 1]
        };
        let out_channels = cfg.sscp_conv_channel_size[idx];
        let kernel_t = cfg.sscp_conv_kernel_size[idx][0];
        let _kernel_f = cfg.sscp_conv_kernel_size[idx][1];
        let stride_t = cfg.sscp_conv_stride_size[idx][0];
        let _stride_f = cfg.sscp_conv_stride_size[idx][1];

        // Semicausal padding
        let half = kernel_t / 2;
        let (pad_t_top, pad_t_bottom) = (half, half);
        let pad_f_left = 1;
        let pad_f_right = 1;

        let _ = input_freq_dim; // used for future freq-dim tracking

        let conv = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_t, // assumes kernel_t == kernel_f
            Conv2dConfig {
                stride: stride_t,
                padding: 0,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv"),
        )?;
        let norm = LayerNorm::new(out_channels, cfg.rms_norm_eps);

        Ok(Self {
            conv,
            norm,
            manual_padding: (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom),
            time_stride: stride_t,
            out_channels,
        })
    }

    fn forward(
        &self,
        audio_encodings: &Tensor,
        audio_mel_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Zero out padded positions
        let valid_mask = audio_mel_mask
            .eq(0.0)?
            .unsqueeze(1)?
            .unsqueeze(D::Minus1)?
            .to_dtype(audio_encodings.dtype())?;
        let audio_encodings = audio_encodings.broadcast_mul(&valid_mask)?;

        // Manual padding
        let audio_encodings = audio_encodings
            .pad_with_zeros(D::Minus1, self.manual_padding.0, self.manual_padding.1)?
            .pad_with_zeros(D::Minus2, self.manual_padding.2, self.manual_padding.3)?;

        let audio_encodings = self.conv.forward(&audio_encodings)?;

        // Subsample mask
        let t_out = audio_encodings.dim(2)?;
        let output_mask = subsample_mask(audio_mel_mask, self.time_stride, t_out)?;

        // Norm: permute to (b, t, f, c), norm on c, then back
        let x = audio_encodings.permute((0, 2, 3, 1))?;
        let x = self.norm.forward(&x)?;
        let x = x.permute((0, 3, 1, 2))?.relu()?;
        Ok((x, output_mask))
    }
}

fn subsample_mask(mask: &Tensor, stride: usize, target_len: usize) -> Result<Tensor> {
    let mask_len = mask.dim(1)?;
    let indices: Vec<u32> = (0..target_len)
        .map(|i| (i * stride).min(mask_len - 1) as u32)
        .collect();
    let indices = Tensor::from_vec(indices, target_len, mask.device())?;
    mask.index_select(&indices, 1)
}

// ── SubSampleConvProjection ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SubSampleConvProjection {
    conv_0: SSCPConvBlock,
    conv_1: SSCPConvBlock,
    input_proj_linear: candle_nn::Linear,
}

impl SubSampleConvProjection {
    fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        let mut current_f = cfg.input_feat_size;
        let mut f_out_dims = Vec::new();

        for i in 0..2 {
            let kernel_w = cfg.sscp_conv_kernel_size[i][1];
            let stride_w = cfg.sscp_conv_stride_size[i][1];
            let f_in_padded = current_f + 2; // pad_f_left + pad_f_right
            let f_out = (f_in_padded - kernel_w) / stride_w + 1;
            f_out_dims.push(f_out);
            current_f = f_out;
        }

        let conv_0 = SSCPConvBlock::new(cfg, 0, cfg.input_feat_size, vb.pp("layer0"))?;
        let conv_1 = SSCPConvBlock::new(cfg, 1, f_out_dims[0], vb.pp("layer1"))?;

        let final_c_out = cfg.sscp_conv_channel_size[1];
        let final_f_out = f_out_dims[1];
        let input_proj_linear = candle_nn::linear_no_bias(
            final_c_out * final_f_out,
            cfg.hidden_size,
            vb.pp("input_proj_linear"),
        )?;

        Ok(Self {
            conv_0,
            conv_1,
            input_proj_linear,
        })
    }

    fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = audio_mel.unsqueeze(1)?;
        let (x, mask) = self.conv_0.forward(&x, audio_mel_mask)?;
        let (x, mask) = self.conv_1.forward(&x, &mask)?;

        let (b, c_out, t_out, f_out) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((b, t_out, f_out * c_out))?;
        Ok((self.input_proj_linear.forward(&x)?, mask))
    }
}

// ── Relative Position Embedding ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RelativePositionEmbedding {
    pos_proj: candle_nn::Linear,
    inv_timescales: Tensor,
    pos_indices: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl RelativePositionEmbedding {
    fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.conf_num_attention_heads;
        let channels = cfg.hidden_size;
        let head_dim = channels / num_heads;
        let max_backward = cfg.conf_attention_context_left.saturating_sub(1);
        let max_forward = cfg.conf_attention_context_right;
        let num_timescales = channels / 2;

        let pos_proj =
            candle_nn::linear_no_bias(channels, num_heads * head_dim, vb.pp("relative_k_proj"))?;

        let min_timescale = 1.0_f64;
        let max_timescale = 10_000.0_f64;
        let log_timescale_increment =
            (max_timescale / min_timescale).ln() / num_timescales.saturating_sub(1).max(1) as f64;
        let inv_timescales = Tensor::from_vec(
            (0..num_timescales)
                .map(|i| (min_timescale * (-log_timescale_increment * i as f64).exp()) as f32)
                .collect::<Vec<_>>(),
            (1, 1, num_timescales),
            vb.device(),
        )?;

        let pos_values: Vec<i64> = (-(max_forward as i64)..=max_backward as i64)
            .rev()
            .collect();
        let span = pos_values.len();
        let pos_indices = Tensor::from_vec(pos_values, (1, span), vb.device())?;

        Ok(Self {
            pos_proj,
            inv_timescales,
            pos_indices,
            num_heads,
            head_dim,
        })
    }

    fn get_timing_signal(&self, position: &Tensor, dtype: DType) -> Result<Tensor> {
        let position = position.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
        let inv_timescales = self.inv_timescales.to_device(position.device())?;
        let scaled_time = position.broadcast_mul(&inv_timescales)?;
        let sin_emb = scaled_time.sin()?;
        let cos_emb = scaled_time.cos()?;
        Tensor::cat(&[sin_emb, cos_emb], D::Minus1)?.to_dtype(dtype)
    }

    fn forward(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        let (batch_size, num_query_blocks, query_block_size, _num_heads, head_dim) =
            queries.dims5()?;
        let key_context_size = keys.dim(2)?;
        let max_span_plus_1 = self.pos_indices.dim(1)?;

        let pos_indices = self.pos_indices.to_device(queries.device())?;
        let sin_emb_timing = self.get_timing_signal(&pos_indices, queries.dtype())?;
        let projected_sin_emb = self.pos_proj.forward(&sin_emb_timing)?;
        let sin_emb = projected_sin_emb
            .reshape((1, max_span_plus_1, self.num_heads, self.head_dim))?
            .squeeze(0)?;

        // term_ac: query * key^T
        let queries_p = queries.transpose(1, 3)?.transpose(2, 3)?.contiguous()?;
        let keys_p_t = keys
            .transpose(1, 3)?
            .transpose(2, 3)?
            .transpose(3, 4)?
            .contiguous()?;

        let queries_3d = queries_p.reshape((
            batch_size * self.num_heads * num_query_blocks,
            query_block_size,
            head_dim,
        ))?;
        let keys_3d = keys_p_t.reshape((
            batch_size * self.num_heads * num_query_blocks,
            head_dim,
            key_context_size,
        ))?;
        let term_ac = queries_3d.matmul(&keys_3d)?.reshape((
            batch_size,
            self.num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        ))?;

        // term_bd: query * sin_emb^T (relative position bias)
        let q_transposed = queries.transpose(1, 3)?.transpose(2, 3)?;
        let s_transposed = sin_emb.transpose(0, 2)?.transpose(0, 1)?;
        let q_reshaped = q_transposed.reshape((
            batch_size * self.num_heads,
            num_query_blocks * query_block_size,
            head_dim,
        ))?;
        let s_broadcast = s_transposed
            .unsqueeze(0)?
            .broadcast_as((batch_size, self.num_heads, head_dim, max_span_plus_1))?
            .reshape((batch_size * self.num_heads, head_dim, max_span_plus_1))?
            .contiguous()?;
        let term_bd_unshifted = q_reshaped.contiguous()?.matmul(&s_broadcast)?.reshape((
            batch_size,
            self.num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        ))?;

        // Relative shift
        let pad_amount = (key_context_size + 1) - max_span_plus_1;
        let term_bd_padded = term_bd_unshifted.pad_with_zeros(D::Minus1, 0, pad_amount)?;
        let term_bd_reshaped = term_bd_padded.reshape((
            batch_size,
            self.num_heads,
            num_query_blocks,
            query_block_size * (key_context_size + 1),
        ))?;
        let term_bd_sliced =
            term_bd_reshaped.narrow(D::Minus1, 0, query_block_size * key_context_size)?;
        let term_bd = term_bd_sliced.reshape((
            batch_size,
            self.num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        ))?;

        term_ac.broadcast_add(&term_bd)
    }
}

// ── Conformer Attention ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ConformerAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    post: candle_nn::Linear,
    relative_position_embedding: RelativePositionEmbedding,
    per_dim_scale_softplus: Tensor,
    pre_attn_norm: RmsNorm,
    post_norm: RmsNorm,
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_past_horizon: usize,
    max_future_horizon: usize,
    context_size: usize,
    q_scale: f64,
    k_scale: f64,
    softcap: f64,
    invalid_logits_value: f64,
    local_causal_valid_mask: Tensor,
    gradient_clipping: f64,
    hidden_size: usize,
}

impl ConformerAttention {
    fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.conf_num_attention_heads;
        let hidden_size = cfg.hidden_size;
        let head_dim = hidden_size / num_heads;
        let chunk_size = cfg.conf_attention_chunk_size;
        let max_past_horizon = cfg.conf_attention_context_left.saturating_sub(1);
        let max_future_horizon = cfg.conf_attention_context_right;
        let context_size = chunk_size + max_past_horizon + max_future_horizon;

        let attn_vb = vb.pp("self_attn");
        let relative_position_embedding = RelativePositionEmbedding::new(cfg, attn_vb.clone())?;
        let per_dim_scale = attn_vb.get(head_dim, "per_dim_scale")?;
        let q_proj =
            candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, attn_vb.pp("q_proj"))?;
        let k_proj =
            candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, attn_vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, attn_vb.pp("v_proj"))?;
        let post = candle_nn::linear_no_bias(hidden_size, hidden_size, attn_vb.pp("post"))?;

        let pre_attn_norm = RmsNorm::new(hidden_size, cfg.rms_norm_eps, vb.pp("norm_pre_attn"))?;
        let post_norm = RmsNorm::new(hidden_size, cfg.rms_norm_eps, vb.pp("norm_post_attn"))?;

        let q_scale = (head_dim as f64).powf(-0.5) / 2.0_f64.ln();
        let k_scale = (1.0_f64 + std::f64::consts::E).ln() / 2.0_f64.ln();

        // Build local causal valid mask
        let mut mask_vec = vec![0u8; chunk_size * context_size];
        for i in 0..chunk_size {
            for j in 0..context_size {
                let lower = j >= i;
                let upper =
                    (j as isize) <= (i as isize + (max_past_horizon + max_future_horizon) as isize);
                if lower && upper {
                    mask_vec[i * context_size + j] = 1;
                }
            }
        }
        let local_causal_valid_mask =
            Tensor::from_vec(mask_vec, (chunk_size, context_size), vb.device())?
                .to_dtype(DType::U8)?;

        let per_dim_scale_softplus = {
            let ones = Tensor::ones_like(&per_dim_scale)?.to_dtype(DType::F32)?;
            let exp_scale = per_dim_scale.to_dtype(DType::F32)?.exp()?;
            ones.broadcast_add(&exp_scale)?.log()?
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            post,
            relative_position_embedding,
            per_dim_scale_softplus,
            pre_attn_norm,
            post_norm,
            num_heads,
            head_dim,
            chunk_size,
            max_past_horizon,
            max_future_horizon,
            context_size,
            q_scale,
            k_scale,
            softcap: cfg.conf_attention_logit_cap,
            invalid_logits_value: cfg.conf_attention_invalid_logits_value,
            local_causal_valid_mask,
            gradient_clipping: cfg.gradient_clipping,
            hidden_size: cfg.hidden_size,
        })
    }

    fn convert_to_block(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (b, t) = (dims[0], dims[1]);
        let num_blocks = t.div_ceil(self.chunk_size);
        let padding_len = num_blocks * self.chunk_size - t;
        let x = if padding_len > 0 {
            x.pad_with_zeros(1, 0, padding_len)?
        } else {
            x.clone()
        };
        let mut new_shape = vec![b, num_blocks, self.chunk_size];
        new_shape.extend_from_slice(&dims[2..]);
        x.reshape(new_shape)
    }

    fn extract_block_context(&self, x: &Tensor) -> Result<Tensor> {
        let pad_left = self.max_past_horizon;
        let pad_right = self.max_future_horizon + self.chunk_size - 1;
        let x = x.pad_with_zeros(1, pad_left, pad_right)?;
        let frame_len = self.context_size;
        let frame_step = self.chunk_size;
        let time_dim = x.dim(1)?;
        let num_windows = (time_dim - frame_len) / frame_step + 1;

        let mut windows = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            let start_idx = i * frame_step;
            windows.push(x.narrow(1, start_idx, frame_len)?);
        }
        Tensor::stack(&windows, 1)
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = x.clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.pre_attn_norm.forward(&x)?;

        let q = self.q_proj.forward(&x)?.to_dtype(DType::F32)?;
        let k = self.k_proj.forward(&x)?.to_dtype(DType::F32)?;
        let v = self.v_proj.forward(&x)?.to_dtype(DType::F32)?;

        let (b, t, _) = x.dims3()?;

        let q = q.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, t, self.num_heads, self.head_dim))?;

        let per_dim_scale = self
            .per_dim_scale_softplus
            .to_device(x.device())?
            .to_dtype(DType::F32)?;

        // Scale Q and K
        let q = q
            .affine(self.q_scale, 0.0)?
            .broadcast_mul(&per_dim_scale.reshape((1, 1, 1, self.head_dim))?)?;
        let k = k.affine(self.k_scale, 0.0)?;

        // Convert to blocks
        let query_blocks = self.convert_to_block(&q)?;
        let key_blocks = self.extract_block_context(&k)?;
        let value_blocks = self.extract_block_context(&v)?;
        let num_query_blocks = query_blocks.dim(1)?;

        // Ensure key/value blocks match context_size
        let key_blocks = if key_blocks.dim(2)? != self.context_size {
            if key_blocks.dim(2)? < self.context_size {
                key_blocks.pad_with_zeros(2, 0, self.context_size - key_blocks.dim(2)?)?
            } else {
                key_blocks.narrow(2, 0, self.context_size)?
            }
        } else {
            key_blocks
        };
        let value_blocks = if value_blocks.dim(2)? != self.context_size {
            if value_blocks.dim(2)? < self.context_size {
                value_blocks.pad_with_zeros(2, 0, self.context_size - value_blocks.dim(2)?)?
            } else {
                value_blocks.narrow(2, 0, self.context_size)?
            }
        } else {
            value_blocks
        };

        // Align block counts
        let key_blocks = if key_blocks.dim(1)? > num_query_blocks {
            key_blocks.narrow(1, 0, num_query_blocks)?
        } else {
            key_blocks
        };
        let value_blocks = if value_blocks.dim(1)? > num_query_blocks {
            value_blocks.narrow(1, 0, num_query_blocks)?
        } else {
            value_blocks
        };

        // Build validity mask from input mask + causality
        let original_valid = mask.eq(0.0)?.to_dtype(DType::U8)?;
        let extracted_valid = self.extract_block_context(&original_valid)?;
        let extracted_valid = if extracted_valid.rank() == 4 {
            extracted_valid.reshape((b, num_query_blocks, self.context_size))?
        } else {
            extracted_valid
        };
        let extracted_valid = if extracted_valid.dim(D::Minus1)? != self.context_size {
            if extracted_valid.dim(D::Minus1)? < self.context_size {
                extracted_valid.pad_with_zeros(
                    D::Minus1,
                    0,
                    self.context_size - extracted_valid.dim(D::Minus1)?,
                )?
            } else {
                extracted_valid.narrow(D::Minus1, 0, self.context_size)?
            }
        } else {
            extracted_valid
        };

        let cond_input = extracted_valid.unsqueeze(1)?.unsqueeze(3)?;
        let cond_causal = self
            .local_causal_valid_mask
            .to_device(x.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let final_cond = cond_input
            .to_dtype(DType::U8)?
            .broadcast_mul(&cond_causal.to_dtype(DType::U8)?)?;

        // Relative position logits
        let logits = self
            .relative_position_embedding
            .forward(&query_blocks, &key_blocks)?;
        let logits = ((logits / self.softcap)?.tanh()? * self.softcap)?;

        // Broadcast mask to logits shape
        let final_cond = final_cond.broadcast_as(logits.shape())?;

        let invalid_logits = Tensor::new(self.invalid_logits_value as f32, logits.device())?
            .broadcast_as(logits.shape())?;
        let masked_logits = final_cond.where_cond(&logits, &invalid_logits)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&masked_logits.to_dtype(DType::F32)?)?;

        // Weighted sum of values
        let (b_dim, n_dim, u_dim, w_dim, c_dim) = probabilities.dims5()?;
        let h_dim = value_blocks.dim(D::Minus1)?;
        let probs_p = probabilities.permute((0, 2, 1, 3, 4))?.reshape((
            b_dim * u_dim * n_dim,
            w_dim,
            c_dim,
        ))?;
        let vals_p = value_blocks.permute((0, 1, 3, 2, 4))?.reshape((
            b_dim * u_dim * n_dim,
            c_dim,
            h_dim,
        ))?;
        let context_vectors = probs_p
            .matmul(&vals_p)?
            .reshape((b_dim, u_dim, n_dim, w_dim, h_dim))?
            .permute((0, 1, 3, 2, 4))?
            .reshape((
                b,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            ))?
            .narrow(1, 0, t)?;

        let context_vectors = context_vectors.reshape((b, t, self.hidden_size))?;
        let out = self
            .post
            .forward(&context_vectors)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        residual.broadcast_add(&self.post_norm.forward(&out)?)
    }
}

// ── Conformer FeedForward ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ConformerFeedForward {
    scale: f64,
    pre_layer_norm: RmsNorm,
    ffw_layer_1: candle_nn::Linear,
    ffw_layer_2: candle_nn::Linear,
    post_layer_norm: RmsNorm,
    gradient_clipping: f64,
}

impl ConformerFeedForward {
    fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: cfg.conf_residual_weight,
            pre_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_layer_norm"),
            )?,
            ffw_layer_1: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.hidden_size * 4,
                vb.pp("ffw_layer_1"),
            )?,
            ffw_layer_2: candle_nn::linear_no_bias(
                cfg.hidden_size * 4,
                cfg.hidden_size,
                vb.pp("ffw_layer_2"),
            )?,
            post_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_layer_norm"),
            )?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = x.clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.pre_layer_norm.forward(&x)?;
        let x = candle_nn::ops::silu(&self.ffw_layer_1.forward(&x)?)?;
        let x = self
            .ffw_layer_2
            .forward(&x)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.post_layer_norm.forward(&x)?;
        residual.broadcast_add(&(x * self.scale)?)
    }
}

// ── Conformer LightConv1d ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ConformerLightConv1d {
    pre_layer_norm: RmsNorm,
    depthwise_conv1d: Conv1d,
    conv_norm: RmsNorm,
    linear_start: candle_nn::Linear,
    linear_end: candle_nn::Linear,
    causal_padding: usize,
    gradient_clipping: f64,
}

impl ConformerLightConv1d {
    fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            pre_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_layer_norm"),
            )?,
            linear_start: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.hidden_size * 2,
                vb.pp("linear_start"),
            )?,
            depthwise_conv1d: candle_nn::conv1d_no_bias(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.conf_conv_kernel_size,
                candle_nn::Conv1dConfig {
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: cfg.hidden_size,
                    cudnn_fwd_algo: None,
                },
                vb.pp("depthwise_conv1d"),
            )?,
            conv_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("conv_norm"))?,
            linear_end: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.hidden_size,
                vb.pp("linear_end"),
            )?,
            causal_padding: cfg.conf_conv_kernel_size - 1,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, audio_encodings: &Tensor) -> Result<Tensor> {
        let residual = audio_encodings;
        let x = self.pre_layer_norm.forward(audio_encodings)?;
        let x = self.linear_start.forward(&x)?;
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let x = (x1 * candle_nn::ops::sigmoid(&x2)?)?;
        let x = x.transpose(D::Minus1, D::Minus2)?;
        let x = x.pad_with_zeros(D::Minus1, self.causal_padding, 0)?;
        let x = self
            .depthwise_conv1d
            .forward(&x.to_dtype(DType::F32)?)?
            .to_dtype(audio_encodings.dtype())?
            .transpose(D::Minus2, D::Minus1)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.conv_norm.forward(&x)?;
        let x = candle_nn::ops::silu(&x)?;
        let x = self.linear_end.forward(&x)?;
        residual.broadcast_add(&x)
    }
}

// ── ConformerBlock ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ConformerBlock {
    ffw_layer_start: ConformerFeedForward,
    attention: ConformerAttention,
    lconv1d: ConformerLightConv1d,
    ffw_layer_end: ConformerFeedForward,
    norm: RmsNorm,
    gradient_clipping: f64,
}

impl ConformerBlock {
    fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ffw_layer_start: ConformerFeedForward::new(cfg, vb.pp("feed_forward1"))?,
            attention: ConformerAttention::new(cfg, vb.clone())?,
            lconv1d: ConformerLightConv1d::new(cfg, vb.pp("lconv1d"))?,
            ffw_layer_end: ConformerFeedForward::new(cfg, vb.pp("feed_forward2"))?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_out"))?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, audio_encodings: &Tensor, audio_mel_mask: &Tensor) -> Result<Tensor> {
        let x = self.ffw_layer_start.forward(audio_encodings)?;
        let x = self.attention.forward(&x, audio_mel_mask)?;
        let x = self.lconv1d.forward(&x)?;
        let x = self
            .ffw_layer_end
            .forward(&x)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        self.norm.forward(&x)
    }
}

// ── AudioModel (public) ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AudioModel {
    subsample_conv_projection: SubSampleConvProjection,
    conformer: Vec<ConformerBlock>,
    conf_reduction_factor: usize,
    output_proj: Option<candle_nn::Linear>,
}

impl AudioModel {
    pub fn new(cfg: &Gemma4AudioConfig, vb: VarBuilder) -> Result<Self> {
        let subsample_conv_projection =
            SubSampleConvProjection::new(cfg, vb.pp("subsample_conv_projection"))?;
        let mut conformer = Vec::with_capacity(cfg.conf_num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.conf_num_hidden_layers {
            conformer.push(ConformerBlock::new(cfg, vb_layers.pp(i))?);
        }
        let output_proj = if let Some(output_dim) = cfg.output_proj_dims {
            Some(candle_nn::linear(
                cfg.hidden_size,
                output_dim,
                vb.pp("output_proj"),
            )?)
        } else {
            None
        };
        Ok(Self {
            subsample_conv_projection,
            conformer,
            conf_reduction_factor: cfg.conf_reduction_factor,
            output_proj,
        })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mut audio_encodings, mut current_mask) = self
            .subsample_conv_projection
            .forward(audio_mel, audio_mel_mask)?;

        for block in &self.conformer {
            audio_encodings = block.forward(&audio_encodings, &current_mask)?;
        }

        // Reduction factor subsampling
        if self.conf_reduction_factor > 1 {
            let stride = self.conf_reduction_factor;
            let enc_len = audio_encodings.dim(1)?;
            let reduced_len = enc_len.div_ceil(stride);
            let indices: Vec<u32> = (0..reduced_len)
                .map(|i| (i * stride).min(enc_len - 1) as u32)
                .collect();
            let indices = Tensor::from_vec(indices, reduced_len, audio_encodings.device())?;
            audio_encodings = audio_encodings.index_select(&indices, 1)?;
            current_mask = current_mask.index_select(&indices, 1)?;
        }

        if let Some(ref output_proj) = self.output_proj {
            audio_encodings = output_proj.forward(&audio_encodings)?;
        }

        // Align mask length
        let enc_len = audio_encodings.dim(1)?;
        let mask_len = current_mask.dim(1)?;
        if mask_len != enc_len {
            if enc_len < mask_len {
                current_mask = current_mask.narrow(1, 0, enc_len)?;
            } else {
                current_mask = current_mask.pad_with_zeros(1, 0, enc_len - mask_len)?;
            }
        }

        // Zero out invalid positions
        let valid_mask = current_mask.eq(0.0)?;
        let zeros = Tensor::zeros_like(&audio_encodings)?;
        let audio_encodings = valid_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(audio_encodings.shape())?
            .where_cond(&audio_encodings, &zeros)?;

        Ok((audio_encodings, current_mask))
    }
}
