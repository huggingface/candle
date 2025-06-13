//! Helper utils for common Candle use cases.

use candle::{Device, Result, Tensor};
use tokenizers::{Encoding, Tokenizer};

/// Helper function to get a single candle Device. This will not perform any multi device mapping.
/// Will prioritize Metal or CUDA (if Candle is compiled with those features) and fallback to CPU.
///
/// ## Example  
/// ```
/// let device = candle_core::utils::device(true, false).unwrap();
/// ```
pub fn device(use_cpu: bool, quiet: bool) -> Result<Device> {
    if use_cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        if !quiet {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                println!("Running on CPU, to run on GPU (metal), build with `--features metal`");
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!("Running on CPU, to run on GPU, build with `--features cuda`");
            }
        }

        Ok(Device::Cpu)
    }
}

/// Perform L2 normalization.
///
/// ## Example  
/// ```
/// use candle_core::{Tensor, utils::normalize_l2};
///
/// let device = candle_core::utils::device(true, false).unwrap();
/// let x = Tensor::new(&[[0f32, 1.], [2., 3.]], &device).unwrap();
/// let normalized_x = normalize_l2(&x).unwrap();
/// ```
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let squared = &v.sqr()?;
    let summed = &squared.sum_keepdim(1)?;
    let norms = &summed.sqrt()?;
    v.broadcast_div(norms)
}

/// Encodes the input strings into tokens, IDs, and type IDs.
///
/// ## Example  
/// ```ignore
/// use candle_helpers::encode_tokens;
///
/// let device = candle_core::utils::device(true, false).unwrap();
/// let (tokens, token_ids, token_type_ids) = encode_tokens(inputs, &tokenizer, &device).unwrap();
/// ```
pub fn encode_tokens(
    inputs: &[&str],
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Vec<Encoding>, Tensor, Tensor)> {
    let tokens = tokenizer
        .encode_batch(inputs.to_owned(), true)
        .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {}", e)))?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<Result<Vec<_>>>()?;
    let token_ids = Tensor::stack(&token_ids, 0)?;

    let token_type_ids = Tensor::zeros(token_ids.dims(), token_ids.dtype(), device)?;

    Ok((tokens, token_ids, token_type_ids))
}

/// Builds the attention mask from the token encodings.
///
/// ## Example  
/// ```ignore
/// use candle_helpers::build_attention_mask;
///
/// let device = candle_core::utils::device(true, false).unwrap();
/// let attention_mask = build_attention_mask(&tokens, &device).unwrap();
/// ```
pub fn build_attention_mask(tokens: &[Encoding], device: &Device) -> Result<Tensor> {
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;

    Ok(attention_mask)
}
