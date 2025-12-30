//! PaddleOCR-VL Vision Encoder.
//!
//! NaViT-style dynamic resolution visual encoder with 2D rotary position embeddings.

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, linear_b, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use std::cell::RefCell;
use std::collections::HashMap;

use super::config::VisionConfig;

/// Default maximum number of cached position embeddings.
const DEFAULT_POS_EMBED_CACHE_SIZE: usize = 16;

/// LFU (Least Frequently Used) cache for interpolated position embeddings.
///
/// Caches interpolated position embeddings keyed by (height, width) grid dimensions.
/// Uses frequency-based eviction when cache is full: the least frequently accessed
/// entry is evicted first. This matches PyTorch's caching behavior.
struct PosEmbedCache {
    /// Cached embeddings: (height, width) -> tensor
    cache: HashMap<(usize, usize), Tensor>,
    /// Access frequency for each key
    frequency: HashMap<(usize, usize), usize>,
    /// Maximum cache size
    max_size: usize,
}

impl PosEmbedCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            frequency: HashMap::with_capacity(max_size),
            max_size,
        }
    }

    /// Get a cached embedding, incrementing its access frequency.
    fn get(&mut self, key: (usize, usize)) -> Option<Tensor> {
        if let Some(tensor) = self.cache.get(&key) {
            *self.frequency.entry(key).or_insert(0) += 1;
            Some(tensor.clone())
        } else {
            None
        }
    }

    /// Insert an embedding into the cache, evicting LFU entry if full.
    fn insert(&mut self, key: (usize, usize), tensor: Tensor) {
        // If already in cache, just update
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.cache.entry(key) {
            e.insert(tensor);
            *self.frequency.entry(key).or_insert(0) += 1;
            return;
        }

        // Evict LFU entry if at capacity
        if self.cache.len() >= self.max_size {
            if let Some((&lfu_key, _)) = self.frequency.iter().min_by_key(|(_, &freq)| freq) {
                self.cache.remove(&lfu_key);
                self.frequency.remove(&lfu_key);
            }
        }

        // Insert new entry
        self.cache.insert(key, tensor);
        self.frequency.insert(key, 1);
    }

    /// Clear all cached embeddings.
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.cache.clear();
        self.frequency.clear();
    }
}

/// Patch embedding using Conv2d with interpolated position embedding.
///
/// Weight names:
/// - embeddings.patch_embedding.{weight,bias}
/// - embeddings.position_embedding.weight (base 27×27 grid for interpolation)
/// - embeddings.packing_position_embedding.weight (fallback, 32768 positions)
///
/// For dynamic resolution images, the base position embedding grid is bilinearly
/// interpolated to match the actual patch grid size. Interpolated embeddings are
/// cached with LFU eviction to avoid redundant computation.
struct PatchEmbedding {
    patch_embedding: candle_nn::Conv2d,
    position_embedding: Tensor, // (num_positions, hidden_size) where num_positions = (image_size/patch_size)^2
    #[allow(dead_code)]
    packing_position_embedding: candle_nn::Embedding, // Fallback, kept for weight loading
    base_grid_size: usize,      // sqrt(num_positions), typically 27 for 384/14
    hidden_size: usize,
    /// Cache for interpolated position embeddings (LFU eviction)
    pos_embed_cache: RefCell<PosEmbedCache>,
}

impl PatchEmbedding {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        // Weight: embeddings.patch_embedding (with bias)
        let patch_embedding = candle_nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;

        // Weight: embeddings.position_embedding (base grid for interpolation)
        // Shape: (num_positions, hidden_size) where num_positions = (image_size/patch_size)^2
        let base_grid_size = cfg.image_size / cfg.patch_size;
        let num_positions = base_grid_size * base_grid_size;
        let position_embedding = vb
            .pp("position_embedding")
            .get((num_positions, cfg.hidden_size), "weight")?;

        // Weight: embeddings.packing_position_embedding (32768 positions) - kept for compatibility
        let packing_position_embedding =
            candle_nn::embedding(32768, cfg.hidden_size, vb.pp("packing_position_embedding"))?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            packing_position_embedding,
            base_grid_size,
            hidden_size: cfg.hidden_size,
            pos_embed_cache: RefCell::new(PosEmbedCache::new(DEFAULT_POS_EMBED_CACHE_SIZE)),
        })
    }

    /// Bilinearly interpolate position embeddings to match target grid size.
    ///
    /// Takes the base position embedding grid (e.g., 27×27) and interpolates it
    /// to the target size (e.g., 72×58) using bilinear interpolation.
    ///
    /// This matches PyTorch's nn.functional.interpolate with mode='bilinear', align_corners=False.
    /// Results are cached with LFU eviction to avoid redundant computation.
    fn interpolate_pos_encoding(&self, target_h: usize, target_w: usize) -> Result<Tensor> {
        let cache_key = (target_h, target_w);

        // Check cache first
        if let Some(cached) = self.pos_embed_cache.borrow_mut().get(cache_key) {
            return Ok(cached);
        }

        let device = self.position_embedding.device();
        let dtype = self.position_embedding.dtype();
        let base_h = self.base_grid_size;
        let base_w = self.base_grid_size;

        // If target matches base, just reshape and return (also cache it)
        if target_h == base_h && target_w == base_w {
            let result = self
                .position_embedding
                .reshape((1, target_h * target_w, self.hidden_size))?
                .to_dtype(dtype)?;
            self.pos_embed_cache
                .borrow_mut()
                .insert(cache_key, result.clone());
            return Ok(result);
        }

        // Reshape position embedding to (base_h, base_w, hidden)
        let pos_embed = self.position_embedding.to_dtype(DType::F32)?.reshape((
            base_h,
            base_w,
            self.hidden_size,
        ))?;

        // Compute scale factors (align_corners=False style)
        let scale_h = base_h as f64 / target_h as f64;
        let scale_w = base_w as f64 / target_w as f64;

        // Build interpolated output
        let mut output_data = Vec::with_capacity(target_h * target_w * self.hidden_size);

        for ty in 0..target_h {
            for tx in 0..target_w {
                // Source coordinates (align_corners=False: map center to center)
                let sy = (ty as f64 + 0.5) * scale_h - 0.5;
                let sx = (tx as f64 + 0.5) * scale_w - 0.5;

                // Clamp to valid range
                let sy = sy.max(0.0).min((base_h - 1) as f64);
                let sx = sx.max(0.0).min((base_w - 1) as f64);

                // Integer and fractional parts
                let sy0 = sy.floor() as usize;
                let sx0 = sx.floor() as usize;
                let sy1 = (sy0 + 1).min(base_h - 1);
                let sx1 = (sx0 + 1).min(base_w - 1);
                let fy = (sy - sy0 as f64) as f32;
                let fx = (sx - sx0 as f64) as f32;

                // Bilinear weights
                let w00 = (1.0 - fy) * (1.0 - fx);
                let w01 = (1.0 - fy) * fx;
                let w10 = fy * (1.0 - fx);
                let w11 = fy * fx;

                // Get the 4 corner embeddings
                let e00: Vec<f32> = pos_embed.i((sy0, sx0))?.to_vec1()?;
                let e01: Vec<f32> = pos_embed.i((sy0, sx1))?.to_vec1()?;
                let e10: Vec<f32> = pos_embed.i((sy1, sx0))?.to_vec1()?;
                let e11: Vec<f32> = pos_embed.i((sy1, sx1))?.to_vec1()?;

                // Interpolate each dimension
                for d in 0..self.hidden_size {
                    let val = w00 * e00[d] + w01 * e01[d] + w10 * e10[d] + w11 * e11[d];
                    output_data.push(val);
                }
            }
        }

        // Create output tensor and cache it
        let result = Tensor::from_vec(
            output_data,
            (1, target_h * target_w, self.hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        self.pos_embed_cache
            .borrow_mut()
            .insert(cache_key, result.clone());
        Ok(result)
    }

    /// Forward pass with interpolated position embeddings for dynamic resolution.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Input: (batch, channels, height, width)
        // Output: (batch, num_patches, hidden_size)
        let xs = self.patch_embedding.forward(xs)?;
        let (batch, hidden, h, w) = xs.dims4()?;
        let num_patches = h * w;

        // Reshape to (batch, num_patches, hidden)
        let xs = xs.reshape((batch, hidden, num_patches))?.transpose(1, 2)?;

        // Get interpolated position embedding for this grid size
        let pos_embed = self.interpolate_pos_encoding(h, w)?;

        // Broadcast add position embedding to each batch
        xs.broadcast_add(&pos_embed)
    }
}

/// 2D Rotary Position Embedding for vision.
struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.0;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

/// Vision MLP block.
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: candle_nn::Activation,
}

impl VisionMlp {
    fn new(
        dim: usize,
        hidden_dim: usize,
        act: candle_nn::Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            fc1: linear_b(dim, hidden_dim, true, vb.pp("fc1"))?,
            fc2: linear_b(hidden_dim, dim, true, vb.pp("fc2"))?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.apply(&self.act)?;
        self.fc2.forward(&xs)
    }
}

/// Vision self-attention with 2D RoPE.
/// Weight names:
/// - self_attn.q_proj.{weight,bias}
/// - self_attn.k_proj.{weight,bias}
/// - self_attn.v_proj.{weight,bias}
/// - self_attn.out_proj.{weight,bias}
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = dim / num_heads;
        Ok(Self {
            q_proj: linear_b(dim, dim, true, vb.pp("q_proj"))?,
            k_proj: linear_b(dim, dim, true, vb.pp("k_proj"))?,
            v_proj: linear_b(dim, dim, true, vb.pp("v_proj"))?,
            out_proj: linear_b(dim, dim, true, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        self.forward_impl(xs, cu_seqlens, cos, sin, None)
    }

    /// Forward pass with optional debug tensor export.
    fn forward_with_debug(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        exports: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        self.forward_impl(xs, cu_seqlens, cos, sin, Some(exports))
    }

    fn forward_impl(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        mut exports: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;

        // Separate Q, K, V projections
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Export Q, K, V before reshape
        if let Some(ref mut exp) = exports {
            exp.insert("attn_q_proj".to_string(), q.to_dtype(DType::F32)?);
            exp.insert("attn_k_proj".to_string(), k.to_dtype(DType::F32)?);
            exp.insert("attn_v_proj".to_string(), v.to_dtype(DType::F32)?);
        }

        // Reshape to (seq_len, num_heads, head_dim)
        let mut q = q.reshape((seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((seq_len, self.num_heads, self.head_dim))?;
        let mut v = v.reshape((seq_len, self.num_heads, self.head_dim))?;

        // Convert to f32 for precision in RoPE
        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;

        // Export cos/sin and Q/K before RoPE
        if let Some(ref mut exp) = exports {
            exp.insert("rope_cos".to_string(), cos.clone());
            exp.insert("rope_sin".to_string(), sin.clone());
            exp.insert("q_before_rope".to_string(), q.clone());
            exp.insert("k_before_rope".to_string(), k.clone());
        }

        // Apply 2D RoPE
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;

        // Export Q/K after RoPE
        if let Some(ref mut exp) = exports {
            exp.insert("q_after_rope".to_string(), q.clone());
            exp.insert("k_after_rope".to_string(), k.clone());
        }

        // Process each image sequence separately (variable length)
        let mut outputs = Vec::new();
        let mut first_chunk_attn_weights = None;

        for (chunk_idx, window) in cu_seqlens.windows(2).enumerate() {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let mut chunk_out = {
                let q = q_chunk.unsqueeze(0)?;
                let k = k_chunk.unsqueeze(0)?;
                let v = v_chunk.unsqueeze(0)?;

                let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
                // Compute softmax in F32 for numerical stability (matches PyTorch's
                // softmax(..., dtype=torch.float32).to(query.dtype) pattern)
                let original_dtype = attn_weights.dtype();
                let attn_weights = if original_dtype != DType::F32 {
                    let attn_weights = attn_weights.to_dtype(DType::F32)?;
                    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                    attn_weights.to_dtype(original_dtype)?
                } else {
                    candle_nn::ops::softmax_last_dim(&attn_weights)?
                };

                // Capture first chunk's attention weights for debugging
                if chunk_idx == 0 && exports.is_some() {
                    first_chunk_attn_weights = Some(attn_weights.clone());
                }

                attn_weights.matmul(&v)?
            };

            chunk_out = chunk_out.squeeze(0)?.transpose(0, 1)?;
            chunk_out = chunk_out.reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out.to_dtype(xs.dtype())?);
        }

        // Export attention weights (first chunk only, as full would be huge)
        if let Some(ref mut exp) = exports {
            if let Some(attn_w) = first_chunk_attn_weights {
                exp.insert("attn_weights_chunk0".to_string(), attn_w);
            }
        }

        let attn_output = Tensor::cat(&outputs, 0)?;

        // Export before out_proj
        if let Some(ref mut exp) = exports {
            exp.insert(
                "attn_output_before_proj".to_string(),
                attn_output.to_dtype(DType::F32)?,
            );
        }

        self.out_proj.forward(&attn_output)
    }
}

/// Vision encoder block (pre-norm transformer).
/// Weight names:
/// - layer_norm1.{weight,bias}
/// - layer_norm2.{weight,bias}
/// - self_attn.{q,k,v,out}_proj.{weight,bias}
/// - mlp.fc1.{weight,bias}
/// - mlp.fc2.{weight,bias}
struct VisionBlock {
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    self_attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, norm_cfg, vb.pp("layer_norm1"))?,
            layer_norm2: layer_norm(cfg.hidden_size, norm_cfg, vb.pp("layer_norm2"))?,
            self_attn: VisionAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: VisionMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.hidden_act,
                vb.pp("mlp"),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.layer_norm1.forward(xs)?;
        let attn_out = self.self_attn.forward(&normed, cu_seqlens, cos, sin)?;
        let xs_att = xs.add(&attn_out)?;
        let mlp_out = self.mlp.forward(&self.layer_norm2.forward(&xs_att)?)?;
        xs_att.add(&mlp_out)
    }

    /// Forward pass with debug tensor export for attention internals.
    fn forward_with_debug(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        exports: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let normed = self.layer_norm1.forward(xs)?;
        exports.insert(
            "layer0_after_norm1".to_string(),
            normed.to_dtype(DType::F32)?,
        );

        let attn_out = self
            .self_attn
            .forward_with_debug(&normed, cu_seqlens, cos, sin, exports)?;
        exports.insert(
            "layer0_attn_output".to_string(),
            attn_out.to_dtype(DType::F32)?,
        );

        let xs_att = xs.add(&attn_out)?;
        exports.insert(
            "layer0_after_attn_residual".to_string(),
            xs_att.to_dtype(DType::F32)?,
        );

        let normed2 = self.layer_norm2.forward(&xs_att)?;
        exports.insert(
            "layer0_after_norm2".to_string(),
            normed2.to_dtype(DType::F32)?,
        );

        let mlp_out = self.mlp.forward(&normed2)?;
        exports.insert(
            "layer0_mlp_output".to_string(),
            mlp_out.to_dtype(DType::F32)?,
        );

        xs_att.add(&mlp_out)
    }
}

/// Projector (mlp_AR) - Vision-to-Text bridge.
///
/// Projects vision features to text model dimension with 2×2 spatial merging.
/// Weight names: mlp_AR.pre_norm, mlp_AR.linear_1, mlp_AR.linear_2
///
/// The spatial merge gathers 2×2 patches from the image grid:
/// ```text
/// Input patches (raster order):     Merged output:
/// [0,  1,  2,  3]                   [0+1+4+5,   2+3+6+7]
/// [4,  5,  6,  7]        ->         [8+9+12+13, 10+11+14+15]
/// [8,  9,  10, 11]
/// [12, 13, 14, 15]
/// ```
pub struct Projector {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    spatial_merge_size: usize,
    hidden_size: usize,
}

impl Projector {
    pub fn new(cfg: &VisionConfig, text_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let norm_cfg = LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        Ok(Self {
            pre_norm: layer_norm(cfg.hidden_size, norm_cfg, vb.pp("pre_norm"))?,
            linear_1: linear_b(
                merged_hidden_size,
                merged_hidden_size,
                true,
                vb.pp("linear_1"),
            )?,
            linear_2: linear_b(
                merged_hidden_size,
                text_hidden_size,
                true,
                vb.pp("linear_2"),
            )?,
            spatial_merge_size: cfg.spatial_merge_size,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Forward pass with proper 2×2 spatial merge.
    ///
    /// Implements the einops pattern: "(t h m1 w m2) d -> (t h w) (m1 m2 d)"
    /// where m1=m2=spatial_merge_size (typically 2).
    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let normed = self.pre_norm.forward(xs)?;

        let grid = grid_thw.to_vec2::<u32>()?;
        let m = self.spatial_merge_size;

        let mut merged_features = Vec::new();
        let mut offset = 0usize;

        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let seq_len = t * h * w;

            // Extract this image's features
            let features = normed.narrow(0, offset, seq_len)?;
            offset += seq_len;

            // Reshape to (t, h, w, hidden)
            let features = features.reshape((t, h, w, self.hidden_size))?;

            // Merged dimensions
            let h_merged = h / m;
            let w_merged = w / m;

            // Gather 2×2 blocks: for each merged position, collect m×m patches
            // and concatenate their features
            let mut blocks = Vec::with_capacity(t * h_merged * w_merged);

            for ti in 0..t {
                for hi in 0..h_merged {
                    for wi in 0..w_merged {
                        // Collect m×m patches at this merged position
                        let mut patch_features = Vec::with_capacity(m * m);
                        for mi in 0..m {
                            for mj in 0..m {
                                let patch = features.i((ti, hi * m + mi, wi * m + mj))?;
                                patch_features.push(patch);
                            }
                        }
                        // Concatenate patch features: (m*m, hidden) -> (m*m * hidden,)
                        let block = Tensor::cat(&patch_features, 0)?;
                        blocks.push(block);
                    }
                }
            }

            // Stack all blocks: (t * h_merged * w_merged, merged_hidden)
            let merged = Tensor::stack(&blocks, 0)?;
            merged_features.push(merged);
        }

        // Concatenate all images
        let merged = Tensor::cat(&merged_features, 0)?;

        // Apply MLP
        let xs = self.linear_1.forward(&merged)?;
        let xs = xs.gelu()?;
        self.linear_2.forward(&xs)
    }

    /// Forward pass returning separate embeddings for each image.
    ///
    /// Unlike `forward()` which concatenates all image features, this method
    /// returns a `Vec<Tensor>` where each tensor contains the embeddings for
    /// one image. This enables the text model to inject each image's embeddings
    /// at the correct positions in multi-image scenarios.
    ///
    /// # Arguments
    /// * `xs` - Vision encoder output of shape (total_patches, hidden_size)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3)
    ///
    /// # Returns
    /// Vector of tensors, one per image, each of shape (num_merged_patches, text_hidden_size)
    pub fn forward_multi(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Vec<Tensor>> {
        let normed = self.pre_norm.forward(xs)?;

        let grid = grid_thw.to_vec2::<u32>()?;
        let m = self.spatial_merge_size;

        let mut result = Vec::with_capacity(grid.len());
        let mut offset = 0usize;

        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let seq_len = t * h * w;

            // Extract this image's features
            let features = normed.narrow(0, offset, seq_len)?;
            offset += seq_len;

            // Reshape to (t, h, w, hidden)
            let features = features.reshape((t, h, w, self.hidden_size))?;

            // Merged dimensions
            let h_merged = h / m;
            let w_merged = w / m;

            // Gather 2×2 blocks
            let mut blocks = Vec::with_capacity(t * h_merged * w_merged);

            for ti in 0..t {
                for hi in 0..h_merged {
                    for wi in 0..w_merged {
                        let mut patch_features = Vec::with_capacity(m * m);
                        for mi in 0..m {
                            for mj in 0..m {
                                let patch = features.i((ti, hi * m + mi, wi * m + mj))?;
                                patch_features.push(patch);
                            }
                        }
                        let block = Tensor::cat(&patch_features, 0)?;
                        blocks.push(block);
                    }
                }
            }

            // Stack all blocks: (t * h_merged * w_merged, merged_hidden)
            let merged = Tensor::stack(&blocks, 0)?;

            // Apply MLP
            let xs = self.linear_1.forward(&merged)?;
            let xs = xs.gelu()?;
            let projected = self.linear_2.forward(&xs)?;

            result.push(projected);
        }

        Ok(result)
    }
}

/// PaddleOCR-VL Vision Model.
///
/// NaViT-style encoder with 2D RoPE, supporting dynamic image resolutions.
/// Weight structure:
/// - embeddings.patch_embedding, embeddings.position_embedding
/// - encoder.layers.{i}.*
/// - post_layernorm
pub struct VisionModel {
    embeddings: PatchEmbedding,
    encoder_layers: Vec<VisionBlock>,
    post_layernorm: LayerNorm,
    projector: Projector,
    rotary_pos_emb: VisionRotaryEmbedding,
    hidden_size: usize,
    patch_size: usize,
}

impl VisionModel {
    pub fn new(
        vision_cfg: &VisionConfig,
        text_hidden_size: usize,
        vb: VarBuilder,
        projector_vb: VarBuilder,
    ) -> Result<Self> {
        // Embeddings: embeddings.patch_embedding, embeddings.position_embedding
        let embeddings = PatchEmbedding::new(vision_cfg, vb.pp("embeddings"))?;

        // Encoder layers: encoder.layers.{i}.*
        let mut encoder_layers = Vec::with_capacity(vision_cfg.num_hidden_layers);
        let vb_encoder = vb.pp("encoder").pp("layers");
        for i in 0..vision_cfg.num_hidden_layers {
            encoder_layers.push(VisionBlock::new(vision_cfg, vb_encoder.pp(i))?);
        }

        // Post layer norm: post_layernorm
        let norm_cfg = LayerNormConfig {
            eps: vision_cfg.layer_norm_eps,
            ..Default::default()
        };
        let post_layernorm = layer_norm(vision_cfg.hidden_size, norm_cfg, vb.pp("post_layernorm"))?;

        // Projector is separate at mlp_AR
        let projector = Projector::new(vision_cfg, text_hidden_size, projector_vb)?;

        let head_dim = vision_cfg.head_dim();
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;

        Ok(Self {
            embeddings,
            encoder_layers,
            post_layernorm,
            projector,
            rotary_pos_emb,
            hidden_size: vision_cfg.hidden_size,
            patch_size: vision_cfg.patch_size,
        })
    }

    /// Compute 2D rotary position embeddings for variable-size grids.
    ///
    /// For each patch position, computes (row_embed, col_embed) based on its
    /// 2D coordinates in the image grid. Uses raster order: position i has
    /// row = i // width, col = i % width.
    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.rotary_pos_emb.inv_freq.device();
        let grid = grid_thw.to_vec2::<u32>()?;

        // Find max grid dimension to build frequency table
        let max_hw = grid
            .iter()
            .flat_map(|v| v[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let freq_table = self.rotary_pos_emb.make_embeds(max_hw)?;

        // Build position indices using simple raster order
        // Reference: image_pids = arange(t*h*w) % (h*w)
        //            h_ids = image_pids // w
        //            w_ids = image_pids % w
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;

            // For each temporal frame, patches are in raster order
            for _ in 0..t {
                for pos in 0..(h * w) {
                    let row = (pos / w) as i64;
                    let col = (pos % w) as i64;
                    rows.push(row);
                    cols.push(col);
                }
            }
        }

        let total_tokens = rows.len();
        let rows = Tensor::from_vec(rows, (total_tokens,), device)?;
        let cols = Tensor::from_vec(cols, (total_tokens,), device)?;

        // Get row and column frequency embeddings
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;

        // Stack and reshape: (tokens, 2, dim/2) -> (tokens, dim)
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
    }

    /// Build cumulative sequence lengths for variable-length attention.
    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|v| v[0] as usize).sum::<usize>() + 1);
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    /// Forward pass for vision encoder.
    ///
    /// # Arguments
    /// * `pixel_values` - Image tensor of shape (batch, channels, height, width)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3) containing [temporal, height, width]
    ///
    /// # Returns
    /// Projected vision features of shape (total_patches / merge_factor, text_hidden_size)
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        self.forward_with_debug(pixel_values, grid_thw, false)
    }

    /// Forward pass with optional debug output.
    pub fn forward_with_debug(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        debug: bool,
    ) -> Result<Tensor> {
        let dtype = pixel_values.dtype();

        // Get patch embeddings
        let hidden_states = self.embeddings.forward(pixel_values)?;
        let hidden_states = hidden_states.reshape(((), self.hidden_size))?;

        if debug {
            let hs_f32 = hidden_states.to_dtype(DType::F32)?;
            let first_10: Vec<f32> = hs_f32.i(0)?.narrow(0, 0, 10)?.to_vec1()?;
            eprintln!("DEBUG vision encoder:");
            eprintln!(
                "  patch_embedding+pos output shape: {:?}",
                hidden_states.dims()
            );
            eprintln!("  embeddings[0,:10]: {:?}", first_10);
            let mean = hs_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!("  embeddings mean: {:.6}", mean);
        }

        // Compute rotary embeddings
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        // Pass through encoder layers
        let mut hidden_states = hidden_states;
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;

            if debug && (i == 0 || i == 13 || i == 26) {
                let hs_f32 = hidden_states.to_dtype(DType::F32)?;
                let first_10: Vec<f32> = hs_f32.i(0)?.narrow(0, 0, 10)?.to_vec1()?;
                let mean = hs_f32.mean_all()?.to_scalar::<f32>()?;
                eprintln!(
                    "  after layer {}: mean={:.6}, [0,:10]={:?}",
                    i, mean, first_10
                );
            }
        }

        // Apply post layer norm
        let hidden_states = self.post_layernorm.forward(&hidden_states)?;

        if debug {
            let hs_f32 = hidden_states.to_dtype(DType::F32)?;
            let first_10: Vec<f32> = hs_f32.i(0)?.narrow(0, 0, 10)?.to_vec1()?;
            let mean = hs_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!(
                "  after post_layernorm: mean={:.6}, [0,:10]={:?}",
                mean, first_10
            );
        }

        // Project to text model dimension with proper 2×2 spatial merging
        let output = self.projector.forward(&hidden_states, grid_thw)?;

        if debug {
            let out_f32 = output.to_dtype(DType::F32)?;
            let first_10: Vec<f32> = out_f32.i(0)?.narrow(0, 0, 10)?.to_vec1()?;
            let mean = out_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!(
                "  projector output: shape={:?}, mean={:.6}, [0,:10]={:?}",
                output.dims(),
                mean,
                first_10
            );
        }

        output.to_dtype(dtype)
    }

    /// Forward pass for multiple images, returning separate embeddings for each.
    ///
    /// # Arguments
    /// * `pixel_values` - Batched image tensor of shape (num_images, channels, height, width)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3)
    ///
    /// # Returns
    /// Vector of tensors, one per image, each of shape (num_merged_patches, text_hidden_size)
    pub fn forward_multi(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Vec<Tensor>> {
        let dtype = pixel_values.dtype();

        // Get patch embeddings
        let hidden_states = self.embeddings.forward(pixel_values)?;
        let hidden_states = hidden_states.reshape(((), self.hidden_size))?;

        // Compute rotary embeddings
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        // Pass through encoder layers
        let mut hidden_states = hidden_states;
        for layer in self.encoder_layers.iter() {
            hidden_states = layer.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
        }

        // Apply post layer norm
        let hidden_states = self.post_layernorm.forward(&hidden_states)?;

        // Project to text model dimension, returning separate tensors per image
        let outputs = self.projector.forward_multi(&hidden_states, grid_thw)?;

        // Convert each output to target dtype
        outputs.into_iter().map(|t| t.to_dtype(dtype)).collect()
    }

    /// Forward pass with tensor export for substitution testing.
    ///
    /// Returns a HashMap of checkpoint tensors that can be saved for comparison
    /// with the PyTorch reference implementation.
    pub fn forward_with_export(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, HashMap<String, Tensor>)> {
        let dtype = pixel_values.dtype();
        let mut exports: HashMap<String, Tensor> = HashMap::new();

        // Export patchified pixel values to match PyTorch format: (num_patches, 3, 14, 14)
        // Input is (batch, channels, height, width), output is (num_patches, channels, patch, patch)
        let (batch, channels, height, width) = pixel_values.dims4()?;
        let h_patches = height / self.patch_size;
        let w_patches = width / self.patch_size;
        let patchified = pixel_values
            .reshape((
                batch,
                channels,
                h_patches,
                self.patch_size,
                w_patches,
                self.patch_size,
            ))?
            .permute((0, 2, 4, 1, 3, 5))? // (batch, h_patches, w_patches, channels, patch_size, patch_size)
            .reshape((
                h_patches * w_patches,
                channels,
                self.patch_size,
                self.patch_size,
            ))?;
        exports.insert("pixel_values".to_string(), patchified.to_dtype(DType::F32)?);

        // 1. Patch embedding (before position embedding)
        let patch_out = self.embeddings.patch_embedding.forward(pixel_values)?;
        let (batch, hidden, h, w) = patch_out.dims4()?;
        let num_patches = h * w;
        let patch_out = patch_out
            .reshape((batch, hidden, num_patches))?
            .transpose(1, 2)?;
        exports.insert(
            "patch_embedding_output".to_string(),
            patch_out.to_dtype(DType::F32)?,
        );

        // 2. Add position embedding (use interpolated 2D position embeddings, same as forward())
        // NOTE: The packing_position_embedding is a fallback; we must use interpolate_pos_encoding
        // to match the regular forward path which uses bilinear interpolation of the 27×27 base grid.
        let pos_embed = self.embeddings.interpolate_pos_encoding(h, w)?;
        let hidden_states = patch_out.broadcast_add(&pos_embed)?;
        let hidden_states = hidden_states.reshape(((), self.hidden_size))?;
        exports.insert(
            "embeddings_output".to_string(),
            hidden_states.to_dtype(DType::F32)?,
        );

        // Compute rotary embeddings
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        // Export RoPE embeddings for comparison
        exports.insert("rope_pos_emb_raw".to_string(), rotary_pos_emb.clone());

        // Pass through encoder layers with checkpoints
        // Layer 0 gets detailed debug export
        let mut hidden_states = hidden_states;
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            if i == 0 {
                // Use debug forward for layer 0 to capture attention internals
                hidden_states = layer.forward_with_debug(
                    &hidden_states,
                    &cu_seqlens,
                    &cos,
                    &sin,
                    &mut exports,
                )?;
                exports.insert(
                    "layer_0_output".to_string(),
                    hidden_states.to_dtype(DType::F32)?,
                );
            } else {
                hidden_states = layer.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
                if i == 13 || i == 26 {
                    exports.insert(
                        format!("layer_{}_output", i),
                        hidden_states.to_dtype(DType::F32)?,
                    );
                }
            }
        }

        // Apply post layer norm
        let hidden_states = self.post_layernorm.forward(&hidden_states)?;
        exports.insert(
            "post_layernorm_output".to_string(),
            hidden_states.to_dtype(DType::F32)?,
        );

        // Project to text model dimension
        let output = self.projector.forward(&hidden_states, grid_thw)?;
        exports.insert("projector_output".to_string(), output.to_dtype(DType::F32)?);

        Ok((output.to_dtype(dtype)?, exports))
    }
}
