//! Tokenization utilities for working with tokenizers.

use candle::{Device, Result, Tensor};
use tokenizers::{Encoding, Tokenizer};

/// Encodes input strings into tokens, token IDs, and type IDs for model input.
///
/// This function handles batch encoding of text inputs and converts them into
/// the tensor format expected by models.
///
/// # Example  
/// ```ignore
/// use candle_utils::tokenization::encode_tokens;
///
/// let device = candle_utils::get_device(true, false).unwrap();
/// let inputs = ["Hello world", "How are you?"];
/// let (tokens, token_ids, token_type_ids) = encode_tokens(&inputs, &tokenizer, &device).unwrap();
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

/// Builds attention mask tensor from tokenizer encodings.
///
/// The attention mask indicates which tokens should be attended to (1) and
/// which should be ignored (0), typically used to mask padding tokens.
///
/// # Example  
/// ```ignore
/// use candle_utils::tokenization::build_attention_mask;
///
/// let device = candle_utils::get_device(true, false).unwrap();
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
