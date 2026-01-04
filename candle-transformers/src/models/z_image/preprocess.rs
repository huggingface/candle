//! Input preprocessing utilities for Z-Image
//!
//! Provides padding and mask construction to convert variable-length inputs
//! into fixed-shape batch tensors.

use candle::{DType, Device, Result, Tensor};

use super::transformer::SEQ_MULTI_OF;

/// Preprocessed inputs structure
#[derive(Debug, Clone)]
pub struct PreparedInputs {
    /// Latent tensor (B, C, 1, H, W)
    pub latents: Tensor,
    /// Padded caption features (B, max_text_len, dim)
    pub cap_feats: Tensor,
    /// Caption attention mask (B, max_text_len), 1=valid, 0=padding
    pub cap_mask: Tensor,
    /// Original text lengths for each sample
    pub text_lengths: Vec<usize>,
}

/// Compute padding length to align to SEQ_MULTI_OF
#[inline]
pub fn compute_padding_len(ori_len: usize) -> usize {
    (SEQ_MULTI_OF - (ori_len % SEQ_MULTI_OF)) % SEQ_MULTI_OF
}

/// Pad variable-length text embeddings to uniform length
///
/// # Arguments
/// * `text_embeddings` - Variable-length text embeddings, each of shape (seq_len, dim)
/// * `pad_value` - Padding value (typically 0.0)
/// * `device` - Device
///
/// # Returns
/// * Padded tensor (B, max_len, dim)
/// * Attention mask (B, max_len), 1=valid, 0=padding
/// * Original lengths
pub fn pad_text_embeddings(
    text_embeddings: &[Tensor],
    pad_value: f32,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<usize>)> {
    if text_embeddings.is_empty() {
        candle::bail!("text_embeddings cannot be empty");
    }

    let batch_size = text_embeddings.len();
    let dim = text_embeddings[0].dim(1)?;
    let dtype = text_embeddings[0].dtype();

    // Compute max length and align to SEQ_MULTI_OF
    let lengths: Vec<usize> = text_embeddings
        .iter()
        .map(|t| t.dim(0))
        .collect::<Result<Vec<_>>>()?;
    let max_len = *lengths.iter().max().unwrap();
    let padded_len = max_len + compute_padding_len(max_len);

    // Build padded tensor and mask
    let mut padded_list = Vec::with_capacity(batch_size);
    let mut mask_list = Vec::with_capacity(batch_size);

    for (i, emb) in text_embeddings.iter().enumerate() {
        let seq_len = lengths[i];
        let pad_len = padded_len - seq_len;

        // Pad embedding
        let padded = if pad_len > 0 {
            let padding = Tensor::full(pad_value, (pad_len, dim), device)?.to_dtype(dtype)?;
            Tensor::cat(&[emb, &padding], 0)?
        } else {
            emb.clone()
        };
        padded_list.push(padded);

        // Create mask: 1 for valid, 0 for padding
        let valid = Tensor::ones((seq_len,), DType::U8, device)?;
        let mask = if pad_len > 0 {
            let invalid = Tensor::zeros((pad_len,), DType::U8, device)?;
            Tensor::cat(&[&valid, &invalid], 0)?
        } else {
            valid
        };
        mask_list.push(mask);
    }

    // Stack into batch
    let cap_feats = Tensor::stack(&padded_list, 0)?;
    let cap_mask = Tensor::stack(&mask_list, 0)?;

    Ok((cap_feats, cap_mask, lengths))
}

/// Prepare all inputs, converting variable-length inputs to fixed-shape batch tensors
///
/// # Arguments
/// * `latents` - Latent tensor (B, C, H, W)
/// * `text_embeddings` - Variable-length text embeddings, each of shape (seq_len, cap_feat_dim)
/// * `device` - Device
///
/// # Returns
/// PreparedInputs containing all preprocessed tensors
pub fn prepare_inputs(
    latents: &Tensor,
    text_embeddings: &[Tensor],
    device: &Device,
) -> Result<PreparedInputs> {
    // Latents: (B, C, H, W) -> (B, C, 1, H, W) add frame dimension
    let latents = latents.unsqueeze(2)?;

    // Pad text embeddings
    let (cap_feats, cap_mask, text_lengths) = pad_text_embeddings(text_embeddings, 0.0, device)?;

    Ok(PreparedInputs {
        latents,
        cap_feats,
        cap_mask,
        text_lengths,
    })
}

/// Create attention mask for a single sample
/// Useful for testing or simplified scenarios
pub fn create_attention_mask(
    valid_len: usize,
    total_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let valid = Tensor::ones((valid_len,), DType::U8, device)?;
    if valid_len < total_len {
        let invalid = Tensor::zeros((total_len - valid_len,), DType::U8, device)?;
        Tensor::cat(&[&valid, &invalid], 0)
    } else {
        Ok(valid)
    }
}

/// Create a batch of uniform text embeddings
///
/// # Arguments
/// * `text_embedding` - Single text embedding (seq_len, dim)
/// * `batch_size` - Number of copies to create
///
/// # Returns
/// Batched text embeddings (batch_size, seq_len, dim)
pub fn batch_text_embedding(text_embedding: &Tensor, batch_size: usize) -> Result<Tensor> {
    let (seq_len, dim) = text_embedding.dims2()?;
    text_embedding
        .unsqueeze(0)?
        .broadcast_as((batch_size, seq_len, dim))?
        .contiguous()
}

/// Create a batch of uniform masks
///
/// # Arguments
/// * `mask` - Single mask (seq_len,)
/// * `batch_size` - Number of copies to create
///
/// # Returns
/// Batched masks (batch_size, seq_len)
pub fn batch_mask(mask: &Tensor, batch_size: usize) -> Result<Tensor> {
    let seq_len = mask.dim(0)?;
    mask.unsqueeze(0)?
        .broadcast_as((batch_size, seq_len))?
        .contiguous()
}
