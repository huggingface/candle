//! Colpali Model for text/image similarity scoring.
//!
//! Colpali combines a vision encoder with an efficient LM for retrieving content.
//!

use candle::{BackendStorage, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::paligemma;
use candle_nn::{linear, Linear};

pub struct Model<B: BackendStorage> {
    pub model: paligemma::Model<B>,
    pub custom_text_projection: Linear<B>,
}

impl<B: BackendStorage> Model<B> {
    pub fn new(config: &paligemma::Config, vb: VarBuilder<B>) -> Result<Self> {
        let model = paligemma::Model::new(config, vb.pp("model"))?;
        let custom_text_projection = linear(
            config.text_config.hidden_size,
            128,
            vb.pp("custom_text_proj"),
        )?;

        Ok(Self {
            model,
            custom_text_projection,
        })
    }

    pub fn forward_images(
        &mut self,
        pixel_values: &Tensor<B>,
        input_ids: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let outputs = self
            .model
            .setup_without_projection(pixel_values, input_ids)?;
        let outputs = self.custom_text_projection.forward(&outputs)?;
        let outputs = outputs.broadcast_div(&outputs.sqr()?.sum_keepdim(2)?.sqrt()?)?;
        Ok(outputs)
    }

    pub fn forward_text(&mut self, input_ids: &Tensor<B>) -> Result<Tensor<B>> {
        let outputs = self.model.forward_without_projection(input_ids)?;
        let outputs = self.custom_text_projection.forward(&outputs)?;
        let outputs = outputs.broadcast_div(&outputs.sqr()?.sum_keepdim(2)?.sqrt()?)?;
        Ok(outputs)
    }
}
