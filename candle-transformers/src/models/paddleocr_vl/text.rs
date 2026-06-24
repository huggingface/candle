//! PaddleOCR-VL Text Model.
//!
//! ERNIE-4.5-0.3B based decoder with RMSNorm, GQA, and M-RoPE (Multimodal RoPE).
//!
//! M-RoPE uses 3D position IDs (temporal, height, width) for vision tokens,
//! allowing the model to encode spatial structure of images.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_b, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

use super::config::TextConfig;

/// Multimodal Rotary Position Embedding (M-RoPE).
///
/// Unlike standard 1D RoPE, M-RoPE supports 3D position IDs for vision tokens:
/// - Temporal position (for video frames, always 0 for images)
/// - Height position (row in the image grid)
/// - Width position (column in the image grid)
///
/// Text tokens use the same position for all 3 dimensions (equivalent to 1D RoPE).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Precomputed cos values for all positions: [max_seq_len, head_dim/2]
    cos: Tensor,
    /// Precomputed sin values for all positions: [max_seq_len, head_dim/2]
    sin: Tensor,
    /// M-RoPE section sizes: [temporal, height, width]
    mrope_section: Vec<usize>,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;

        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        // Compute cos/sin for all positions
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            mrope_section: cfg.mrope_section.clone(),
            head_dim: dim,
        })
    }

    /// Apply Multimodal RoPE with 3D position IDs.
    ///
    /// This follows the PyTorch implementation where:
    /// 1. Compute cos/sin for each of the 3 position dimensions (temporal, height, width)
    /// 2. Split the head_dim into sections based on mrope_section
    /// 3. Use temporal positions for first section, height for second, width for third
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, kv_heads, seq_len, head_dim]
    /// * `position_ids` - 3D position IDs [3, batch, seq_len] where dim 0 is [temporal, height, width]
    pub fn apply_multimodal_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // position_ids: [3, batch, seq_len]
        let (three, _batch, _seq_len) = position_ids.dims3()?;
        assert_eq!(three, 3, "position_ids must have 3 dimensions");

        // Compute cos/sin for each position dimension
        // Each returns [batch, seq_len, head_dim] with cos/sin of (inv_freq * position)
        let (cos_3d, sin_3d) = self.compute_3d_rope_embeddings(position_ids)?;
        // cos_3d/sin_3d: [3, batch, seq_len, head_dim]

        // Apply mrope_section to select appropriate bands from each dimension
        // mrope_section = [16, 24, 24] splits head_dim=128 into [16, 24, 24, 64] chunks
        // where 64 is the remainder. Chunk i uses dimension i % 3.
        let (cos, sin) = self.apply_mrope_sections(&cos_3d, &sin_3d)?;
        // cos/sin: [batch, seq_len, head_dim]

        // Reshape for broadcasting: [batch, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        // Apply RoPE to q and k
        let q_embed = self.apply_rope_to_tensor(q, &cos, &sin)?;
        let k_embed = self.apply_rope_to_tensor(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    /// Compute cos/sin embeddings for 3D position IDs.
    /// position_ids: [3, batch, seq_len]
    /// Returns: (cos, sin) each with shape [3, batch, seq_len, head_dim]
    fn compute_3d_rope_embeddings(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (three, batch, seq_len) = position_ids.dims3()?;
        let half_dim = self.head_dim / 2;

        // For each of the 3 dimensions, gather cos/sin based on positions
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();

        for dim_idx in 0..three {
            let pos = position_ids.i(dim_idx)?; // [batch, seq_len]
            let pos_flat = pos.flatten_all()?; // [batch * seq_len]

            // Gather from precomputed cos/sin
            let cos_gathered = self.cos.index_select(&pos_flat, 0)?; // [batch*seq_len, half_dim]
            let sin_gathered = self.sin.index_select(&pos_flat, 0)?;

            // Reshape to [batch, seq_len, half_dim]
            let cos_dim = cos_gathered.reshape((batch, seq_len, half_dim))?;
            let sin_dim = sin_gathered.reshape((batch, seq_len, half_dim))?;

            // Duplicate to full head_dim: [batch, seq_len, head_dim]
            let cos_full = Tensor::cat(&[&cos_dim, &cos_dim], D::Minus1)?;
            let sin_full = Tensor::cat(&[&sin_dim, &sin_dim], D::Minus1)?;

            cos_parts.push(cos_full);
            sin_parts.push(sin_full);
        }

        // Stack to [3, batch, seq_len, head_dim]
        let cos_3d = Tensor::stack(&cos_parts, 0)?;
        let sin_3d = Tensor::stack(&sin_parts, 0)?;

        Ok((cos_3d, sin_3d))
    }

    /// Apply mrope_section to select bands from each dimension.
    ///
    /// PyTorch behavior: `cos.split(mrope_section * 2, dim=-1)` where `* 2` is **list repetition**!
    /// In Python: `[16, 24, 24] * 2 = [16, 24, 24, 16, 24, 24]` (6 chunks totaling 128)
    ///
    /// Then `[m[i % 3] for i, m in enumerate(splits)]` selects from the 3D position embeddings:
    /// - chunk 0 (dims 0-15):    from temporal (i=0, i%3=0)
    /// - chunk 1 (dims 16-39):   from height (i=1, i%3=1)
    /// - chunk 2 (dims 40-63):   from width (i=2, i%3=2)
    /// - chunk 3 (dims 64-79):   from temporal (i=3, i%3=0)
    /// - chunk 4 (dims 80-103):  from height (i=4, i%3=1)
    /// - chunk 5 (dims 104-127): from width (i=5, i%3=2)
    ///
    /// Final layout: [T:16, H:24, W:24, T:16, H:24, W:24]
    fn apply_mrope_sections(&self, cos_3d: &Tensor, sin_3d: &Tensor) -> Result<(Tensor, Tensor)> {
        // cos_3d/sin_3d: [3, batch, seq_len, head_dim]
        // mrope_section = [16, 24, 24]
        //
        // In Python: mrope_section * 2 = [16, 24, 24, 16, 24, 24] (list repetition!)
        // This creates 6 splits, cycling through temporal/height/width twice
        let mut sections_repeated: Vec<usize> = Vec::new();
        sections_repeated.extend_from_slice(&self.mrope_section);
        sections_repeated.extend_from_slice(&self.mrope_section);
        // sections_repeated = [16, 24, 24, 16, 24, 24]

        // Split the head_dim and take from appropriate dimension (i % 3)
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        let mut offset = 0;

        for (i, &sec_size) in sections_repeated.iter().enumerate() {
            let dim_idx = i % 3; // Cycles: temporal(0), height(1), width(2), temporal(0), ...
                                 // Take slice from dimension dim_idx at the current offset
            let cos_slice = cos_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec_size)?;
            let sin_slice = sin_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec_size)?;
            cos_parts.push(cos_slice);
            sin_parts.push(sin_slice);
            offset += sec_size;
        }

        // Concatenate along head_dim: [batch, seq_len, head_dim]
        let cos = Tensor::cat(&cos_parts, D::Minus1)?;
        let sin = Tensor::cat(&sin_parts, D::Minus1)?;

        Ok((cos, sin))
    }

    /// Apply rotary embedding to a tensor.
    /// x: [batch, heads, seq_len, head_dim]
    /// cos/sin: [batch, 1, seq_len, head_dim]
    fn apply_rope_to_tensor(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;

        // rotate_half: split x into two halves and rotate
        let head_dim = x.dim(D::Minus1)?;
        let half_dim = head_dim / 2;

        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        // rotate_half gives [-x2, x1]
        let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        // Apply: x * cos + rotate_half(x) * sin
        x.broadcast_mul(cos)? + x_rotated.broadcast_mul(sin)?
    }

    /// Apply Multimodal RoPE with export of intermediate tensors for debugging.
    pub fn apply_multimodal_rotary_emb_with_export(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor, std::collections::HashMap<String, Tensor>)> {
        use std::collections::HashMap;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        let (three, _batch, _seq_len) = position_ids.dims3()?;
        assert_eq!(three, 3, "position_ids must have 3 dimensions");

        // Export position_ids
        tensors.insert("position_ids".to_string(), position_ids.clone());

        // Compute cos/sin for each position dimension
        let (cos_3d, sin_3d) = self.compute_3d_rope_embeddings(position_ids)?;
        tensors.insert("cos_3d".to_string(), cos_3d.clone());
        tensors.insert("sin_3d".to_string(), sin_3d.clone());

        // Apply mrope_section to select appropriate bands
        let (cos, sin) = self.apply_mrope_sections(&cos_3d, &sin_3d)?;
        tensors.insert("cos_after_mrope".to_string(), cos.clone());
        tensors.insert("sin_after_mrope".to_string(), sin.clone());

        // Export specific position for debugging (position 947 if available)
        let seq_len = cos.dim(1)?;
        if seq_len > 947 {
            tensors.insert("cos_pos947".to_string(), cos.i((.., 947, ..))?.squeeze(1)?);
            tensors.insert("sin_pos947".to_string(), sin.i((.., 947, ..))?.squeeze(1)?);
        }

        // Reshape for broadcasting: [batch, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        // Apply RoPE to q and k
        let q_embed = self.apply_rope_to_tensor(q, &cos, &sin)?;
        let k_embed = self.apply_rope_to_tensor(k, &cos, &sin)?;

        Ok((q_embed, k_embed, tensors))
    }
}

/// Image grid specification for multi-image M-RoPE position computation.
#[derive(Debug, Clone)]
pub struct ImageGrid {
    /// Grid height (number of patches in height dimension, after spatial merge)
    pub grid_h: usize,
    /// Grid width (number of patches in width dimension, after spatial merge)
    pub grid_w: usize,
}

/// Compute 3D M-RoPE position IDs for multi-image multimodal input.
///
/// This function creates position IDs of shape [3, batch, seq_len] for inputs
/// containing multiple images. Each image's tokens get 2D spatial positions,
/// while text tokens get sequential 1D positions.
///
/// # Position Layout
/// ```text
/// Text tokens: all 3 dims same (t=h=w=pos)
/// Image tokens: 2D grid positions offset by preceding text
///   - pos_t = offset (temporal = 0 for images)
///   - pos_h = row_in_grid + offset
///   - pos_w = col_in_grid + offset
/// ```
///
/// # Arguments
/// * `input_ids` - Token IDs of shape (batch, seq_len)
/// * `image_token_id` - The token ID used for image placeholders
/// * `image_grids` - Grid dimensions for each image (in order of appearance)
/// * `device` - Device to create tensors on
///
/// # Returns
/// Position IDs tensor of shape [3, batch, seq_len]
pub fn compute_mrope_position_ids_multi(
    input_ids: &Tensor,
    image_token_id: u32,
    image_grids: &[ImageGrid],
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    // Create position IDs for all 3 dimensions
    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];

    for b in 0..batch {
        let batch_start = b * seq_len;

        // Find all image token ranges
        let mut image_ranges: Vec<(usize, usize)> = Vec::new(); // (start, end) exclusive
        let mut in_image = false;
        let mut image_start = 0usize;

        for s in 0..seq_len {
            let token_id = input_ids_vec[batch_start + s];
            if token_id == image_token_id {
                if !in_image {
                    in_image = true;
                    image_start = s;
                }
            } else if in_image {
                image_ranges.push((image_start, s));
                in_image = false;
            }
        }
        // Handle case where image tokens extend to end of sequence
        if in_image {
            image_ranges.push((image_start, seq_len));
        }

        // Verify we have the right number of image ranges
        if image_ranges.len() != image_grids.len() {
            return Err(candle::Error::Msg(format!(
                "Mismatch: found {} image ranges but {} grids provided",
                image_ranges.len(),
                image_grids.len()
            )));
        }

        // Compute positions
        let mut current_pos = 0i64;
        let mut range_idx = 0usize;

        for s in 0..seq_len {
            let idx = batch_start + s;

            // Check if we're at the start of an image range
            if range_idx < image_ranges.len() && s == image_ranges[range_idx].0 {
                // Process entire image range
                let (img_start, img_end) = image_ranges[range_idx];
                let grid = &image_grids[range_idx];
                let num_vision_tokens = grid.grid_h * grid.grid_w;

                // Verify token count matches grid
                let actual_tokens = img_end - img_start;
                if actual_tokens != num_vision_tokens {
                    return Err(candle::Error::Msg(format!(
                        "Image {} has {} tokens but grid {}x{} = {} expected",
                        range_idx, actual_tokens, grid.grid_h, grid.grid_w, num_vision_tokens
                    )));
                }

                // Assign spatial positions to vision tokens
                let offset = current_pos;
                for vision_idx in 0..num_vision_tokens {
                    let token_s = img_start + vision_idx;
                    let token_idx = batch_start + token_s;

                    let t_pos = 0i64; // Temporal is 0 for images
                    let h_pos = (vision_idx / grid.grid_w) as i64;
                    let w_pos = (vision_idx % grid.grid_w) as i64;

                    pos_t[token_idx] = t_pos + offset;
                    pos_h[token_idx] = h_pos + offset;
                    pos_w[token_idx] = w_pos + offset;
                }

                // Update current_pos to max position in this image + 1
                let max_h = (grid.grid_h - 1) as i64;
                let max_w = (grid.grid_w - 1) as i64;
                current_pos = offset + max_h.max(max_w) + 1;

                range_idx += 1;
                continue;
            }

            // Skip if we're inside an image range (already processed)
            if range_idx > 0 {
                let prev_range = image_ranges[range_idx - 1];
                if s >= prev_range.0 && s < prev_range.1 {
                    continue;
                }
            }
            if range_idx < image_ranges.len() {
                let curr_range = image_ranges[range_idx];
                if s >= curr_range.0 && s < curr_range.1 {
                    continue;
                }
            }

            // Text token: all dimensions same
            pos_t[idx] = current_pos;
            pos_h[idx] = current_pos;
            pos_w[idx] = current_pos;
            current_pos += 1;
        }
    }

    // Create tensors and stack
    let pos_t = Tensor::from_vec(pos_t, (batch, seq_len), device)?;
    let pos_h = Tensor::from_vec(pos_h, (batch, seq_len), device)?;
    let pos_w = Tensor::from_vec(pos_w, (batch, seq_len), device)?;

    Tensor::stack(&[pos_t, pos_h, pos_w], 0)
}

/// Compute 3D M-RoPE position IDs for multimodal input.
///
/// This function creates position IDs of shape [3, batch, seq_len] following PyTorch's
/// get_rope_index() algorithm:
/// - Text tokens before vision: all 3 dims same, starting from 0
/// - Vision tokens: (temporal + offset, height + offset, width + offset)
/// - Text tokens after vision: all 3 dims same, continuing from max vision position + 1
///
/// For vision tokens, positions encode the 2D spatial structure offset by preceding text.
pub fn compute_mrope_position_ids(
    input_ids: &Tensor,
    image_token_id: u32,
    grid_h: usize,
    grid_w: usize,
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    // Create position IDs for all 3 dimensions
    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];

    for b in 0..batch {
        // Find the first image token position
        let batch_start = b * seq_len;
        let mut first_image_pos = None;
        for s in 0..seq_len {
            if input_ids_vec[batch_start + s] == image_token_id {
                first_image_pos = Some(s);
                break;
            }
        }

        // Compute positions following PyTorch's algorithm
        let num_vision_tokens = grid_h * grid_w;

        // Text tokens before vision get sequential positions
        let text_before = first_image_pos.unwrap_or(seq_len);
        for s in 0..text_before {
            let idx = batch_start + s;
            pos_t[idx] = s as i64;
            pos_h[idx] = s as i64;
            pos_w[idx] = s as i64;
        }

        // Vision tokens: (temporal, height, width) + text_before offset
        let offset = text_before as i64;
        let mut vision_idx = 0usize;
        let mut max_vision_pos = offset - 1; // Will be updated

        for s in text_before..seq_len {
            let idx = batch_start + s;
            let token_id = input_ids_vec[idx];

            if token_id == image_token_id && vision_idx < num_vision_tokens {
                // Vision token: spatial position + offset
                let t_pos = 0i64; // Temporal is 0 for images
                let h_pos = (vision_idx / grid_w) as i64;
                let w_pos = (vision_idx % grid_w) as i64;

                pos_t[idx] = t_pos + offset;
                pos_h[idx] = h_pos + offset;
                pos_w[idx] = w_pos + offset;

                // Track max position for text tokens that follow
                max_vision_pos = max_vision_pos
                    .max(pos_t[idx])
                    .max(pos_h[idx])
                    .max(pos_w[idx]);

                vision_idx += 1;
            } else {
                // Text token after vision: continue from max_vision_pos + 1
                max_vision_pos += 1;
                pos_t[idx] = max_vision_pos;
                pos_h[idx] = max_vision_pos;
                pos_w[idx] = max_vision_pos;
            }
        }
    }

    // Create tensors and stack
    let pos_t = Tensor::from_vec(pos_t, (batch, seq_len), device)?;
    let pos_h = Tensor::from_vec(pos_h, (batch, seq_len), device)?;
    let pos_w = Tensor::from_vec(pos_w, (batch, seq_len), device)?;

    Tensor::stack(&[pos_t, pos_h, pos_w], 0)
}

/// Grid specification for video input.
///
/// Unlike images which have only spatial dimensions (h, w),
/// video has temporal (t), height (h), and width (w) dimensions.
#[derive(Debug, Clone)]
pub struct VideoGrid {
    /// Number of temporal frames (after any temporal patching)
    pub grid_t: usize,
    /// Number of height patches (after spatial merge)
    pub grid_h: usize,
    /// Number of width patches (after spatial merge)
    pub grid_w: usize,
}

/// Compute 3D M-RoPE position IDs for video input.
///
/// Unlike multi-image (where t=0 for all images), video uses sequential
/// temporal positions (t=frame_index) to encode temporal relationships
/// between frames.
///
/// Position encoding pattern for video with grid_t=3, grid_h=2, grid_w=2:
/// ```text
/// t_index = [0,0,0,0, 1,1,1,1, 2,2,2,2]  // Temporal: repeats for h*w per frame
/// h_index = [0,0,1,1, 0,0,1,1, 0,0,1,1]  // Height: repeats w times per t
/// w_index = [0,1,0,1, 0,1,0,1, 0,1,0,1]  // Width: cycles fastest
/// ```
///
/// # Arguments
/// * `input_ids` - Token IDs of shape (batch, seq_len)
/// * `video_token_id` - The token ID used for video placeholders (different from image_token_id!)
/// * `video_grid` - Grid dimensions for the video (temporal, height, width)
/// * `second_per_grid_t` - Time interval per temporal grid unit (= temporal_patch_size / fps)
/// * `tokens_per_second` - Temporal position scaling factor (use 2 for video, matching HuggingFace)
/// * `device` - Device to create tensors on
///
/// # Returns
/// Position IDs tensor of shape [3, batch, seq_len]
pub fn compute_mrope_position_ids_video(
    input_ids: &Tensor,
    video_token_id: u32,
    video_grid: &VideoGrid,
    second_per_grid_t: f32,
    tokens_per_second: usize,
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    let grid_t = video_grid.grid_t;
    let grid_h = video_grid.grid_h;
    let grid_w = video_grid.grid_w;
    let num_vision_tokens = grid_t * grid_h * grid_w;

    // Create position IDs for all 3 dimensions
    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];

    for b in 0..batch {
        let batch_start = b * seq_len;

        // Find the video token range
        let mut video_start = None;
        let mut video_end = None;
        let mut in_video = false;

        for s in 0..seq_len {
            let token_id = input_ids_vec[batch_start + s];
            if token_id == video_token_id {
                if !in_video {
                    in_video = true;
                    video_start = Some(s);
                }
            } else if in_video {
                video_end = Some(s);
                break;
            }
        }
        // Handle case where video tokens extend to end of sequence
        if in_video && video_end.is_none() {
            video_end = Some(seq_len);
        }

        // Verify video token count matches grid
        if let (Some(start), Some(end)) = (video_start, video_end) {
            let actual_tokens = end - start;
            if actual_tokens != num_vision_tokens {
                return Err(candle::Error::Msg(format!(
                    "Video has {} tokens but grid {}x{}x{} = {} expected",
                    actual_tokens, grid_t, grid_h, grid_w, num_vision_tokens
                )));
            }
        }

        // Compute positions
        let mut current_pos = 0i64;
        let video_range = video_start.zip(video_end);

        for s in 0..seq_len {
            let idx = batch_start + s;

            // Check if we're at the start of the video range
            if let Some((v_start, v_end)) = video_range {
                if s == v_start {
                    // Process entire video range with 3D positions
                    let offset = current_pos;

                    for vision_idx in 0..num_vision_tokens {
                        let token_s = v_start + vision_idx;
                        let token_idx = batch_start + token_s;

                        // 3D position: t uses temporal scaling for proper frame spacing
                        // Formula: t_pos = frame_index * second_per_grid_t * tokens_per_second
                        // This matches HuggingFace Qwen2-VL processor behavior
                        let frame_index = vision_idx / (grid_h * grid_w);
                        let t_pos = (frame_index as f32
                            * second_per_grid_t
                            * tokens_per_second as f32) as i64;
                        let spatial_idx = vision_idx % (grid_h * grid_w);
                        let h_pos = (spatial_idx / grid_w) as i64;
                        let w_pos = (spatial_idx % grid_w) as i64;

                        pos_t[token_idx] = t_pos + offset;
                        pos_h[token_idx] = h_pos + offset;
                        pos_w[token_idx] = w_pos + offset;
                    }

                    // Update current_pos to max position in video + 1
                    // max_t also needs temporal scaling to match the scaled positions
                    let max_t =
                        ((grid_t - 1) as f32 * second_per_grid_t * tokens_per_second as f32) as i64;
                    let max_h = (grid_h - 1) as i64;
                    let max_w = (grid_w - 1) as i64;
                    current_pos = offset + max_t.max(max_h).max(max_w) + 1;

                    continue;
                }

                // Skip if we're inside the video range (already processed)
                if s > v_start && s < v_end {
                    continue;
                }
            }

            // Text token: all dimensions same
            pos_t[idx] = current_pos;
            pos_h[idx] = current_pos;
            pos_w[idx] = current_pos;
            current_pos += 1;
        }
    }

    // Create tensors and stack
    let pos_t = Tensor::from_vec(pos_t, (batch, seq_len), device)?;
    let pos_h = Tensor::from_vec(pos_h, (batch, seq_len), device)?;
    let pos_w = Tensor::from_vec(pos_w, (batch, seq_len), device)?;

    Tensor::stack(&[pos_t, pos_h, pos_w], 0)
}

/// Gated MLP block (SwiGLU-style).
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_b(hidden_sz, intermediate_sz, cfg.use_bias, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_sz, intermediate_sz, cfg.use_bias, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_sz, hidden_sz, cfg.use_bias, vb.pp("down_proj"))?;
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

    /// Forward with intermediate tensor export for debugging.
    fn forward_with_export(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        use std::collections::HashMap;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        // gate_proj: hidden_size -> intermediate_size
        let gate_out = self.gate_proj.forward(xs)?;
        tensors.insert("gate_proj_out".to_string(), gate_out.clone());

        // Activation (SiLU)
        let gate_act = gate_out.apply(&self.act_fn)?;
        tensors.insert("gate_act_out".to_string(), gate_act.clone());

        // up_proj: hidden_size -> intermediate_size
        let up_out = self.up_proj.forward(xs)?;
        tensors.insert("up_proj_out".to_string(), up_out.clone());

        // Element-wise multiplication
        let mul_out = (&gate_act * &up_out)?;
        tensors.insert("gate_up_mul".to_string(), mul_out.clone());

        // down_proj: intermediate_size -> hidden_size
        let output = self.down_proj.forward(&mul_out)?;
        tensors.insert("down_proj_out".to_string(), output.clone());

        Ok((output, tensors))
    }
}

/// Multi-head attention with Grouped Query Attention (GQA).
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    softmax_scale: f64,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            hidden_sz,
            num_heads * head_dim,
            cfg.use_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            hidden_sz,
            num_kv_heads * head_dim,
            cfg.use_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden_sz,
            num_kv_heads * head_dim,
            cfg.use_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            hidden_sz,
            cfg.use_bias,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache: None,
            softmax_scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Forward with 3D M-RoPE.
    fn forward_with_mrope(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply M-RoPE (3D position IDs)
        let (query_states, key_states) = self.rotary_emb.apply_multimodal_rotary_emb(
            &query_states,
            &key_states,
            position_ids,
        )?;

        self.compute_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            b_sz,
            q_len,
        )
    }

    /// Shared attention computation.
    fn compute_attention(
        &mut self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        attention_mask: Option<&Tensor>,
        b_sz: usize,
        q_len: usize,
    ) -> Result<Tensor> {
        // KV cache handling
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Repeat KV heads for GQA (matches PyTorch's repeat_kv)
        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        // Compute attention (matches eager_attention_forward_ernie)
        let attn_output = {
            // attn_weights = query @ key^T * scaling
            let attn_weights =
                (query_states.matmul(&key_states.transpose(2, 3)?)? * self.softmax_scale)?;

            // Apply causal mask
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            // Softmax in F32 for stability (matches PyTorch's softmax(..., dtype=torch.float32).to(query.dtype))
            let original_dtype = attn_weights.dtype();
            let attn_weights = if original_dtype != DType::F32 {
                let attn_weights = attn_weights.to_dtype(DType::F32)?;
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.to_dtype(original_dtype)?
            } else {
                candle_nn::ops::softmax_last_dim(&attn_weights)?
            };
            // attn_output = attn_weights @ value
            attn_weights.matmul(&value_states)?
        };

        // attn_output.transpose(1, 2).contiguous().reshape(...)
        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    /// Forward with 3D M-RoPE and export attention intermediates (for debugging).
    /// Matches PyTorch's Ernie4_5Attention.forward + eager_attention_forward_ernie exactly.
    pub fn forward_with_mrope_export(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        use std::collections::HashMap;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        let (b_sz, q_len, _) = xs.dims3()?;

        // Q, K, V projections (matches: query_states = self.q_proj(hidden_states))
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        // Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
        // matches: .view(hidden_shape).transpose(1, 2)
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        tensors.insert("q_pre_rope".to_string(), query_states.clone());
        tensors.insert("k_pre_rope".to_string(), key_states.clone());
        tensors.insert("v".to_string(), value_states.clone());

        // Apply M-RoPE with export (matches: apply_multimodal_rotary_pos_emb)
        let (query_states, key_states, rope_tensors) = self
            .rotary_emb
            .apply_multimodal_rotary_emb_with_export(&query_states, &key_states, position_ids)?;

        // Merge RoPE tensors with prefix
        for (k, v) in rope_tensors {
            tensors.insert(format!("rope_{}", k), v);
        }

        tensors.insert("q_post_rope".to_string(), query_states.clone());
        tensors.insert("k_post_rope".to_string(), key_states.clone());

        // No KV cache during prefill
        // Repeat KV heads for GQA (matches: repeat_kv in eager_attention_forward_ernie)
        let key_states_repeated =
            crate::utils::repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
        let value_states_repeated =
            crate::utils::repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;

        tensors.insert("k_repeated".to_string(), key_states_repeated.clone());
        tensors.insert("v_repeated".to_string(), value_states_repeated.clone());

        // Attention scores: Q @ K^T * scaling (matches: torch.matmul(query, key_states.transpose(2, 3)) * scaling)
        let attn_weights_pre =
            (query_states.matmul(&key_states_repeated.transpose(2, 3)?)? * self.softmax_scale)?;
        // Skip exporting full attention matrices - too large ([1, 16, 1357, 1357])
        // Just export a slice for verification: last row of attention for each head
        let seq_len = attn_weights_pre.dim(2)?;
        let attn_last_row = attn_weights_pre.narrow(2, seq_len - 1, 1)?;
        tensors.insert("attn_weights_last_row".to_string(), attn_last_row);

        // Apply mask (matches: attn_weights = attn_weights + causal_mask)
        let attn_weights_masked = match attention_mask {
            None => attn_weights_pre,
            Some(mask) => attn_weights_pre.broadcast_add(mask)?,
        };

        // Softmax (matches: softmax(..., dtype=torch.float32).to(query.dtype))
        let original_dtype = attn_weights_masked.dtype();
        let attn_weights = if original_dtype != DType::F32 {
            let attn_weights = attn_weights_masked.to_dtype(DType::F32)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.to_dtype(original_dtype)?
        } else {
            candle_nn::ops::softmax_last_dim(&attn_weights_masked)?
        };
        // Export last row of softmax attention weights
        let attn_softmax_last_row = attn_weights.narrow(2, seq_len - 1, 1)?;
        tensors.insert(
            "attn_weights_softmax_last_row".to_string(),
            attn_softmax_last_row,
        );

        // Attention output (matches: torch.matmul(attn_weights, value_states))
        let attn_output = attn_weights.matmul(&value_states_repeated)?;
        tensors.insert("attn_output_pre_transpose".to_string(), attn_output.clone());

        // Reshape (matches: .transpose(1, 2).contiguous())
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            q_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection (matches: self.o_proj(attn_output))
        let output = self.o_proj.forward(&attn_output)?;
        tensors.insert("attn_output".to_string(), output.clone());

        Ok((output, tensors))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Decoder layer with pre-norm architecture.
struct DecoderLayer {
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

    /// Forward with 3D M-RoPE.
    fn forward_with_mrope(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_mrope(&xs, attention_mask, position_ids)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    /// Forward with 3D M-RoPE and export attention intermediates.
    fn forward_with_mrope_export(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        use std::collections::HashMap;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        let residual = xs;
        tensors.insert("layer_input".to_string(), xs.clone());

        let xs = self.input_layernorm.forward(xs)?;
        tensors.insert("post_input_layernorm".to_string(), xs.clone());

        let (attn_out, attn_tensors) =
            self.self_attn
                .forward_with_mrope_export(&xs, attention_mask, position_ids)?;

        // Merge attention tensors with prefix
        for (k, v) in attn_tensors {
            tensors.insert(format!("attn_{}", k), v);
        }

        let xs = (attn_out + residual)?;
        tensors.insert("post_attn_residual".to_string(), xs.clone());

        let residual = &xs;
        let post_norm = xs.apply(&self.post_attention_layernorm)?;
        tensors.insert("post_attention_layernorm".to_string(), post_norm.clone());

        // Use MLP forward with export to capture intermediate values
        let (mlp_out, mlp_tensors) = self.mlp.forward_with_export(&post_norm)?;

        // Merge MLP tensors with prefix
        for (k, v) in mlp_tensors {
            tensors.insert(format!("mlp_{}", k), v);
        }

        tensors.insert("mlp_output".to_string(), mlp_out.clone());

        let output = (residual + mlp_out)?;
        tensors.insert("layer_output".to_string(), output.clone());

        Ok((output, tensors))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// PaddleOCR-VL Text Model (ERNIE-4.5 based).
pub struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    pub dtype: DType,
    pub hidden_size: usize,
    device: Device,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, vb.device(), vb.dtype())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
        })
    }

    /// Get token embeddings.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Prepare causal attention mask.
    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    /// Forward pass with embeddings using 3D M-RoPE.
    ///
    /// This method is used for all forward passes (both prefill and generation).
    /// M-RoPE must always be used to maintain consistency with the prefill positions.
    pub fn forward_embeds_with_mrope(
        &mut self,
        mut xs: Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        // Create causal attention mask for prefill
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_sz, seq_len, 0)?)
        };

        for layer in self.layers.iter_mut() {
            xs = layer.forward_with_mrope(&xs, attention_mask.as_ref(), position_ids)?;
        }

        xs = xs.apply(&self.norm)?;

        // Only compute logits for last token
        self.lm_head
            .forward(&xs)?
            .i((.., seq_len - 1, ..))?
            .contiguous()
    }

    /// Clear all KV caches.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    /// Forward pass with M-RoPE and tensor export for debugging.
    ///
    /// Captures intermediate tensors at key checkpoints for comparison with PyTorch.
    /// Layer 1 exports detailed attention intermediates for GQA repeat_kv debugging.
    pub fn forward_embeds_with_mrope_export(
        &mut self,
        mut xs: Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        use std::collections::HashMap;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let (b_sz, seq_len, _) = xs.dims3()?;

        // Causal attention mask
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_causal_attention_mask(b_sz, seq_len, 0)?;
            tensors.insert("causal_mask".to_string(), mask.clone());
            Some(mask)
        };

        tensors.insert("layer0_input".to_string(), xs.clone());

        // Forward through ALL layers, capturing each output
        // Layer 1 gets detailed attention export for debugging
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == 1 {
                // Layer 1: export all attention intermediates
                let (layer_out, layer_tensors) =
                    layer.forward_with_mrope_export(&xs, attention_mask.as_ref(), position_ids)?;
                xs = layer_out;
                // Add layer 1 tensors with prefix
                for (k, v) in layer_tensors {
                    tensors.insert(format!("layer1_{}", k), v);
                }
            } else {
                xs = layer.forward_with_mrope(&xs, attention_mask.as_ref(), position_ids)?;
            }
            // Capture EVERY layer output for detailed comparison
            tensors.insert(format!("layer_{}_output", i), xs.clone());
        }

        // Final layer norm
        xs = xs.apply(&self.norm)?;
        tensors.insert("final_hidden_state".to_string(), xs.clone());

        // LM head - compute full logits
        let logits = self.lm_head.forward(&xs)?;
        tensors.insert("logits".to_string(), logits.clone());

        Ok((logits, tensors))
    }
}
