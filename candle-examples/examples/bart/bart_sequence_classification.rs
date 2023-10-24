use candle::{Device, Result, Tensor};
use candle::Error;
use candle::IndexOp;
use candle_nn as nn;
use candle_nn::linear;
use candle_nn::Dropout;
use candle_nn::VarBuilder;

use crate::bart_config::Config;
use crate::BartModel;
use crate::DEVICE;
use crate::layer_state::LayerState;

pub struct BartClassificationHead {
    dense: nn::Linear,
    dropout: Dropout,
    out_proj: nn::Linear,
}

impl BartClassificationHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_labels = config
            .id2label
            .as_ref()
            .ok_or_else(|| Error::Msg("num_labels not provided in configuration".to_string()))?
            .len();

        let embed_dim = config.d_model;

        let dense = linear(embed_dim, embed_dim, vb.pp("dense"))?;

        let dropout = Dropout::new(config.classif_dropout.unwrap_or(0.0));

        let out_proj = linear(embed_dim, num_labels, vb.pp("out_proj"))?;

        Ok(BartClassificationHead {
            dense,
            dropout,
            out_proj,
        })
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        self.dropout
            .forward(
                &self
                    .dropout
                    .forward(&x, train)?
                    .apply(&self.dense)?
                    .tanh()?,
                train,
            )?
            .apply(&self.out_proj)
    }
}

pub struct BartForSequenceClassification {
    base_model: BartModel,
    classification_head: BartClassificationHead,
    eos_token_id: i64,
}

// https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/bart/modeling_bart.py#L1483
impl BartForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let base_model = BartModel::load(vb.pp("model"), &config)?;

        let classification_head =
            BartClassificationHead::load(vb.pp("classification_head"), &config)?;
        let eos_token_id = config.eos_token_id.unwrap_or(3) as i64;

        Ok(BartForSequenceClassification {
            base_model,
            classification_head,
            eos_token_id,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_output: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<BartModelOutput> {
        let bart_model_output = self.base_model.forward_t(
            Some(input_ids),
            attention_mask,
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask,
            None,
            train,
        )?;

        let eos_mask = input_ids.eq(self.eos_token_id)?;
        let reshape = eos_mask.sum(1)?;
        let input = bart_model_output.decoder_output.permute([2, 0, 1])?;

        let (rows, cols) = bart_model_output.decoder_output.shape().dims2()?;
        let reshape_cols = rows as i64 * reshape.i((0, 0))?.to_scalar::<i64>()?;

        let last_dim = bart_model_output.decoder_output.shape().dims2()?.1;

        let selected = masked_select(input, &eos_mask, DEVICE)?
            .reshape(((), reshape_cols as usize))?
            .transpose(0, 1)?
            .reshape((rows, (), last_dim))?
            .reshape((1, ()))?;

        let sentence_representation = masked_select(
            bart_model_output.decoder_output.permute([2, 0, 1])?,
            &eos_mask,
            DEVICE,
        )?
        .reshape(((), reshape_cols as usize))?
        .transpose(0, 1)?
        .reshape((rows, (), last_dim))?
        .reshape((1, ()))?;

        let logits = self
            .classification_head
            .forward_t(&sentence_representation, train)?;

        Ok(BartModelOutput {
            decoder_output: logits,
            encoder_hidden_state: bart_model_output.encoder_hidden_state,
            cache: None,
            all_decoder_hidden_states: bart_model_output.all_decoder_hidden_states,
            all_decoder_attentions: bart_model_output.all_decoder_attentions,
            all_encoder_hidden_states: bart_model_output.all_encoder_hidden_states,
            all_encoder_attentions: bart_model_output.all_encoder_attentions,
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

fn masked_select(tensor: Tensor, mask: &Tensor, device: &Device) -> Result<Tensor> {
    assert_eq!(mask.dims().len(), 2);
    let mut data: Vec<i64> = Vec::default();

    let (n_rows, n_cols) = mask.shape().dims2()?;
    for row_index in 0..n_rows {
        let row = mask.i(row_index)?;
        for col_index in 0..n_cols {
            let mask_value = mask.i((row_index, col_index))?.to_scalar::<u8>()?;

            if mask_value == 1u8 {
                let value = tensor.i((row_index, col_index))?.to_vec0::<i64>()?;
                data.push(value);
            }
        }
    }

    let data: Vec<i64> = vec![2, 4, 1, 5];
    let length = data.len();
    Tensor::from_vec(data, (length, 1), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_select() -> Result<()> {
        let data: Vec<i64> = vec![0, 0, 2, 0, 4, 1, 5, 0, 0];
        let input = Tensor::from_vec(data, (3, 3), DEVICE)?;
        let mask = input.ge(1i64)?;
        let result = masked_select(input, &mask, DEVICE)?;

        let data: Vec<i64> = vec![2, 4, 1, 5];
        let length = data.len();
        let expected = Tensor::from_vec(data, (length, 1), DEVICE)?;

        assert_eq!(result.to_vec2::<i64>()?, expected.to_vec2::<i64>()?);
        Ok(())
    }
}
