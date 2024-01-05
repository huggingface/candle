use candle::{Device, DType, Result, Tensor};
use candle::IndexOp;
use candle::Shape;
use candle_nn as nn;
use candle_nn::embedding;
use candle_nn::VarBuilder;

use crate::bart_config::Config;
use crate::bart_decoder::{BartDecoder, masked_fill};
use crate::bart_encoder::BartEncoder;
use crate::DEVICE;
use crate::layer_state::LayerState;

pub const DTYPE: DType = DType::F32;

pub struct BartModel {
    pub config: Config,
    pub(crate) encoder: BartEncoder,
    decoder: BartDecoder,
    pub(crate) embeddings: nn::Embedding,
    pad_token_id: usize,
}

impl BartModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let pad_token_id = config.pad_token_id.unwrap_or(1);

        let embeddings = embedding(config.vocab_size, config.d_model, vb.pp("shared"))?;

        let encoder = BartEncoder::load(vb.pp("encoder"), config)?;

        let decoder = BartDecoder::load(vb.pp("decoder"), config)?;

        Ok(BartModel {
            config: config.clone(),
            encoder,
            decoder,
            embeddings,
            pad_token_id,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<BartModelOutput> {
        let calc_decoder_input_ids = if decoder_input_ids.is_none() {
            Some(_shift_tokens_right(
                input_ids.unwrap(),
                self.pad_token_id as i64,
                self.config.decoder_start_token_id.unwrap() as f64,
            )?)
        } else {
            None
        };

        let decoder_input_ids =
            decoder_input_ids.unwrap_or_else(|| calc_decoder_input_ids.as_ref().unwrap());

        let calc_encoder_output = if encoder_output.is_none() {
            Some(self.encoder.forward_t(
                input_ids.unwrap(),
                attention_mask,
                &self.embeddings,
                train,
            )?)
        } else {
            None
        };

        let (calc_hidden_states, all_encoder_hidden_states, all_encoder_attentions) =
            if let Some(calc_encoder_output) = calc_encoder_output {
                (
                    Some(calc_encoder_output.hidden_state),
                    calc_encoder_output.all_hidden_states,
                    calc_encoder_output.all_attentions,
                )
            } else {
                (None, None, None)
            };

        let encoder_output = encoder_output.unwrap_or_else(|| calc_hidden_states.as_ref().unwrap());

        let decoder_output = self.decoder.forward_t(
            decoder_input_ids,
            encoder_output,
            attention_mask,
            decoder_attention_mask,
            &self.embeddings,
            layer_states,
            train,
        )?;

        Ok(BartModelOutput {
            decoder_output: decoder_output.hidden_state,
            encoder_hidden_state: calc_hidden_states,
            cache: decoder_output.next_decoder_cache,
            all_decoder_hidden_states: decoder_output.all_hidden_states,
            all_decoder_attentions: decoder_output.all_attentions,
            all_encoder_hidden_states,
            all_encoder_attentions,
        })
    }
}

pub struct BartModelOutput {
    /// Hidden state of the last layer of the decoder, or logits for a custom head
    /// module after the decoder (e.g. for classification or language modeling tasks)
    pub decoder_output: Tensor,
    /// Hidden state for the last layer of the encoder if they are calculated (not provided), otherwise None
    pub encoder_hidden_state: Option<Tensor>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
    /// Hidden states for all layers of the decoder
    pub all_decoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all layers of the decoder
    pub all_decoder_attentions: Option<Vec<Tensor>>,
    /// Hidden states for all layers of the encoder
    pub all_encoder_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all layers of the encoder
    pub all_encoder_attentions: Option<Vec<Tensor>>,
}

/// https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/bart/modeling_bart.py#L73
fn _shift_tokens_right(
    input_ids: &Tensor,
    pad_token_id: i64,
    decoder_start_token_id: f64,
) -> Result<Tensor> {
    let (n_rows, n_cols) = input_ids.shape().dims2()?;

    let shifted_input_ids = input_ids.i((.., ..n_cols - 1))?;

    let first_col = tensor_n((n_rows, 1), decoder_start_token_id, DType::I64, DEVICE)?;

    let shifted_input_ids = Tensor::cat(&[&first_col, &shifted_input_ids], 1)?;
    let mask = shifted_input_ids.eq(-100i64)?;
    let fill = tensor_n(
        shifted_input_ids.shape(),
        pad_token_id as f64,
        DType::I64,
        DEVICE,
    )?;

    let shifted_input_ids = masked_fill(&shifted_input_ids, &mask, &fill)?;
    // shifted_input_ids[:, 0] = decoder_start_token_id
    // shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()

    Ok(shifted_input_ids)
}

pub fn tensor_n<S: Into<Shape>>(shape: S, n: f64, dtype: DType, device: &Device) -> Result<Tensor> {
    (Tensor::ones(shape, DType::F64, DEVICE)? * n)?
        .to_dtype(dtype)?
        .to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift_tokens_right() -> Result<()> {
        let input_ids = Tensor::ones((3, 3), DType::I64, DEVICE)?;

        let (_, cols) = input_ids.shape().dims2()?;

        let expected_dim: (usize, usize) = (3, 2);

        assert_eq!(
            input_ids.i((.., ..cols - 1))?.shape().dims2()?,
            expected_dim
        );

        let data: Vec<i64> = vec![5, 1, 1, 5, 1, 1, 5, 1, 1];
        let expected = Tensor::from_vec(data, (3, 3), DEVICE)?;

        let result = _shift_tokens_right(&input_ids, 5, 5f64)?;
        assert_eq!(result.to_vec2::<i64>()?, expected.to_vec2::<i64>()?);

        Ok(())
    }

    #[test]
    fn test_n_tensor() -> Result<()> {
        let result = tensor_n((5, 5), 1f64, DType::I64, DEVICE)?;
        let expected = Tensor::ones((5, 5), DType::I64, DEVICE)?;

        assert_eq!(result.to_vec2::<i64>()?, expected.to_vec2::<i64>()?);

        let result = tensor_n((7, 5), 1f64, DType::F64, DEVICE)?;
        let expected = Tensor::ones((7, 5), DType::F64, DEVICE)?;

        assert_eq!(result.to_vec2::<f64>()?, expected.to_vec2::<f64>()?);
        Ok(())
    }
}
