use crate::models::{bart, swin};
use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Donut configuration combining encoder and decoder configs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DonutConfig {
    pub encoder: swin::Config,
    pub decoder: bart::BartConfig,
}

impl DonutConfig {
    pub fn image_height(&self) -> usize {
        self.encoder.image_size.height
    }

    pub fn image_width(&self) -> usize {
        self.encoder.image_size.width
    }
}

/// Donut model combining Swin encoder and BART decoder.
pub struct DonutModel {
    encoder: swin::SwinEncoder,
    decoder: bart::BartForCausalLM,
}

impl DonutModel {
    pub fn load(config: &DonutConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = swin::SwinEncoder::new(&config.encoder, vb.clone())?;
        let decoder = bart::BartForCausalLM::new(&config.decoder, vb)?;

        Ok(Self { encoder, decoder })
    }

    pub fn encode(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.encoder.forward(pixel_values)
    }

    pub fn decode(
        &mut self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let seq_len = decoder_input_ids.dim(1)?;
        let device = decoder_input_ids.device();

        // Create causal mask for decoder
        let attn_mask = if past_kv_len > 0 && seq_len == 1 {
            None
        } else {
            Some(bart::BartForCausalLM::create_causal_mask(seq_len, device)?)
        };

        self.decoder.forward(
            decoder_input_ids,
            Some(encoder_output),
            past_kv_len,
            attn_mask.as_ref(),
        )
    }
}
