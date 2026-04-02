//! Gemma 4 vision encoder (VisionTower).
//!
//! Patch-based ViT with 2D RoPE, clippable linears, pooling,
//! 4 norms per layer, and optional standardization.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, Linear, VarBuilder};

use super::config::Gemma4VisionConfig;

// ── RmsNorm (Gemma-style) ───────────────────────────────────────────────────

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
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

/// Pure RMS normalization without learned weight (V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── 2D Vision Rotary Embedding ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct VisionRotaryEmbedding {
    inv_freq: Tensor,
    ndim: usize,
}

impl VisionRotaryEmbedding {
    fn new(head_dim: usize, theta: f64, ndim: usize, device: &Device) -> Result<Self> {
        let dim_per_dim = head_dim / ndim;
        let half = dim_per_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta.powf(2.0 * i as f64 / dim_per_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, 1, half), device)?;
        Ok(Self { inv_freq, ndim })
    }

    fn forward(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        // positions: [b, num_patches, 2]
        let inv_freq = self.inv_freq.to_dtype(DType::F32)?;
        let mut emb_parts = Vec::with_capacity(self.ndim);
        for d in 0..self.ndim {
            let pos_d = positions.i((.., .., d))?.to_dtype(DType::F32)?;
            let pos_d = pos_d.unsqueeze(D::Minus1)?;
            let freqs_d = pos_d.broadcast_mul(&inv_freq)?;
            let emb_d = Tensor::cat(&[&freqs_d, &freqs_d], D::Minus1)?;
            emb_parts.push(emb_d);
        }
        let full_emb = Tensor::cat(&emb_parts, D::Minus1)?;
        let cos = full_emb.cos()?;
        let sin = full_emb.sin()?;
        Ok((cos, sin))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_2d_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, ndim: usize) -> Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let dim_per_dim = head_dim / ndim;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let mut parts = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let x_part = x.narrow(D::Minus1, d * dim_per_dim, dim_per_dim)?;
        let cos_part = cos.narrow(D::Minus1, d * dim_per_dim, dim_per_dim)?;
        let sin_part = sin.narrow(D::Minus1, d * dim_per_dim, dim_per_dim)?;
        let rotated =
            (x_part.broadcast_mul(&cos_part)? + rotate_half(&x_part)?.broadcast_mul(&sin_part)?)?;
        parts.push(rotated);
    }
    Tensor::cat(&parts, D::Minus1)
}

// ── PatchEmbedder ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PatchEmbedder {
    input_proj: Linear,
    position_embedding_table: Tensor,
    patch_size: usize,
}

impl PatchEmbedder {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let ps = cfg.patch_size;
        let input_proj =
            candle_nn::linear_no_bias(ps * ps * 3, cfg.hidden_size, vb.pp("input_proj"))?;
        let position_embedding_table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        Ok(Self {
            input_proj,
            position_embedding_table,
            patch_size: ps,
        })
    }

    fn forward(&self, pixel_values: &Tensor, patch_positions: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = pixel_values.dims4()?;
        let ps = self.patch_size;
        let ph = h / ps;
        let pw = w / ps;

        // Patchify: (b, c, ph, ps, pw, ps) -> permute(0,2,4,3,5,1) -> (b, ph*pw, ps*ps*c)
        let patches = pixel_values
            .reshape((b, c, ph, ps, pw, ps))?
            .permute((0, 2, 4, 3, 5, 1))?
            .reshape((b, ph * pw, ps * ps * c))?
            .contiguous()?;

        // Scale to [-1, 1]
        let patches = ((patches - 0.5)? * 2.0)?;

        // Linear projection
        let patches = self.input_proj.forward(&patches)?;

        // Position embeddings via index_select
        let clamped_pos = patch_positions.clamp(0i64, i64::MAX)?;
        let n = clamped_pos.dim(1)?;
        let hidden = self.position_embedding_table.dim(2)?;

        let pos_emb_0 = {
            let pos_d = clamped_pos.i((.., .., 0usize))?;
            let table_d = self.position_embedding_table.i(0)?;
            let flat_idx = pos_d.flatten_all()?.to_dtype(DType::U32)?;
            table_d
                .index_select(&flat_idx, 0)?
                .reshape((b, n, hidden))?
        };
        let pos_emb_1 = {
            let pos_d = clamped_pos.i((.., .., 1usize))?;
            let table_d = self.position_embedding_table.i(1)?;
            let flat_idx = pos_d.flatten_all()?.to_dtype(DType::U32)?;
            table_d
                .index_select(&flat_idx, 0)?
                .reshape((b, n, hidden))?
        };
        let pos_emb = (pos_emb_0 + pos_emb_1)?;

        patches + pos_emb
    }
}

// ── VisionAttention ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rms_norm_eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_proj = candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rms_norm_eps: cfg.rms_norm_eps,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let mut q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;

        // Q/K norms and V norm
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        let v = v_norm(&v, self.rms_norm_eps)?.transpose(1, 2)?;

        // Transpose to (b, heads, seq, head_dim) for RoPE
        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;

        // Apply 2D RoPE
        q = apply_2d_rope(&q, cos, sin, 2)?;
        k = apply_2d_rope(&k, cos, sin, 2)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // GQA
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Scaled dot-product attention (scale = 1.0 since Q is already normalized)
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

// ── VisionMlp ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: cfg.hidden_activation,
        })
    }
}

impl Module for VisionMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── VisionEncoderLayer ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl VisionEncoderLayer {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = VisionMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (residual + xs)?;

        let residual = xs.clone();
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        residual + xs
    }
}

// ── VisionPooler ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct VisionPooler {
    hidden_size: usize,
    default_output_length: usize,
}

impl VisionPooler {
    fn new(cfg: &Gemma4VisionConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            default_output_length: cfg.default_output_length,
        }
    }

    /// Average pool patches by spatial position into `output_length` bins.
    fn avg_pool_by_positions(
        &self,
        x: &Tensor,
        patch_positions: &Tensor,
        output_length: usize,
    ) -> Result<Tensor> {
        let (b, num_patches, dim) = x.dims3()?;
        let k = ((num_patches as f64 / output_length as f64).sqrt()) as i64;
        let k_sq = k * k;
        let device = x.device();

        let clamped = patch_positions.clamp(0i64, i64::MAX)?;
        let pos_x = clamped.i((.., .., 0usize))?.to_dtype(DType::F32)?;
        let pos_y = clamped.i((.., .., 1usize))?.to_dtype(DType::F32)?;

        let max_x = (pos_x.max_keepdim(D::Minus1)? + 1.0)?;

        let kf = k as f64;
        let kx = (pos_x / kf)?.floor()?;
        let ky = (pos_y / kf)?.floor()?;
        let stride = (max_x / kf)?.floor()?;
        let kernel_idxs = (kx + stride.broadcast_mul(&ky)?)?.to_dtype(DType::U32)?;

        let original_dtype = x.dtype();
        let x_scaled = (x.to_dtype(DType::F32)? / k_sq as f64)?;
        let idx_expanded = kernel_idxs
            .unsqueeze(2)?
            .broadcast_as(&[b, num_patches, dim])?
            .contiguous()?;
        let output = Tensor::zeros((b, output_length, dim), DType::F32, device)?
            .scatter_add(&idx_expanded, &x_scaled, 1)?
            .to_dtype(original_dtype)?;

        // Scale by sqrt(hidden_size)
        Ok((output * (self.hidden_size as f64).sqrt())?)
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        patch_positions: &Tensor,
        output_length: Option<usize>,
    ) -> Result<Tensor> {
        let output_length = output_length.unwrap_or(self.default_output_length);
        if hidden_states.dim(1)? == output_length {
            Ok((hidden_states.clone() * (self.hidden_size as f64).sqrt())?)
        } else {
            self.avg_pool_by_positions(hidden_states, patch_positions, output_length)
        }
    }
}

// ── VisionTower (public) ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VisionTower {
    patch_embedder: PatchEmbedder,
    encoder_layers: Vec<VisionEncoderLayer>,
    pooler: VisionPooler,
    rotary_emb: VisionRotaryEmbedding,
    std_bias: Option<Tensor>,
    std_scale: Option<Tensor>,
    patch_size: usize,
    pooling_kernel_size: usize,
}

impl VisionTower {
    pub fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedder = PatchEmbedder::new(cfg, vb.pp("patch_embedder"))?;

        let mut encoder_layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_enc = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            encoder_layers.push(VisionEncoderLayer::new(cfg, vb_enc.pp(i))?);
        }

        let pooler = VisionPooler::new(cfg);
        let rotary_emb =
            VisionRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta(), 2, vb.device())?;

        let std_bias = if cfg.standardize {
            Some(vb.get(cfg.hidden_size, "std_bias")?)
        } else {
            None
        };
        let std_scale = if cfg.standardize {
            Some(vb.get(cfg.hidden_size, "std_scale")?)
        } else {
            None
        };

        Ok(Self {
            patch_embedder,
            encoder_layers,
            pooler,
            rotary_emb,
            std_bias,
            std_scale,
            patch_size: cfg.patch_size,
            pooling_kernel_size: cfg.pooling_kernel_size,
        })
    }

    /// Encode a single image at its natural resolution.
    fn encode_single(&self, pv: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let (_, _, h, w) = pv.dims4()?;
        let ph = h / self.patch_size;
        let pw = w / self.patch_size;
        let num_patches = ph * pw;

        // Build position IDs [b, num_patches, 2] -> (col, row)
        let mut pos_data = Vec::with_capacity(num_patches * 2);
        for row in 0..ph {
            for col in 0..pw {
                pos_data.push(col as i64);
                pos_data.push(row as i64);
            }
        }
        let positions =
            Tensor::from_vec(pos_data, (1, num_patches, 2), &Device::Cpu)?.to_device(device)?;

        let embeds = self.patch_embedder.forward(pv, &positions)?;

        // 2D RoPE
        let (cos, sin) = self.rotary_emb.forward(&positions)?;
        let cos = cos.to_dtype(dtype)?;
        let sin = sin.to_dtype(dtype)?;

        let mut hidden_states = embeds;
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin)?;
        }

        // Pool
        let k = self.pooling_kernel_size;
        let output_length = num_patches / (k * k);
        let pooled = self
            .pooler
            .forward(&hidden_states, &positions, Some(output_length))?;

        pooled.squeeze(0)
    }

    /// Encode a batch of images (each may have different sizes).
    pub fn forward(&self, pixel_values_list: &[Tensor]) -> Result<Tensor> {
        let device = pixel_values_list[0].device().clone();
        let dtype = pixel_values_list[0].dtype();

        let mut all_tokens = Vec::with_capacity(pixel_values_list.len());
        for pv in pixel_values_list {
            let tokens = self.encode_single(pv, &device, dtype)?;
            all_tokens.push(tokens);
        }
        let mut hidden_states = Tensor::cat(&all_tokens, 0)?;

        if let (Some(ref std_bias), Some(ref std_scale)) = (&self.std_bias, &self.std_scale) {
            let std_bias = std_bias
                .to_device(hidden_states.device())?
                .to_dtype(hidden_states.dtype())?;
            let std_scale = std_scale
                .to_device(hidden_states.device())?
                .to_dtype(hidden_states.dtype())?;
            hidden_states = (hidden_states.broadcast_sub(&std_bias)?).broadcast_mul(&std_scale)?;
        }

        hidden_states.unsqueeze(0)
    }
}
