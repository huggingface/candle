use super::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use super::distilbert::{DistilBertModel, Config};

use candle::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

pub struct DistilBertForMaskedLM {
    pub distilbert: DistilBertModel,
    vocab_transform: Linear,
    vocab_layer_norm: LayerNorm,
    vocab_projector: Linear,
    span: tracing::Span,
}

impl DistilBertForMaskedLM {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let distilbert = DistilBertModel::load(vb.pp("distilbert"), config)?;
        
        // The prediction head components
        let vocab_transform = linear(config.dim, config.dim, vb.pp("vocab_transform"))?;
        let vocab_layer_norm = layer_norm(config.dim, 1e-12, vb.pp("vocab_layer_norm"))?;
        
        // distil_bert_uncased uses the word embeddings for the vocab projector weight, but has a seperate vocab_projector bias
        let vocab_projector_weight_vb = vb.pp("distilbert.embeddings.word_embeddings");
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let ws = vocab_projector_weight_vb.get_with_hints((config.vocab_size, config.dim), "weight", init_ws)?;
        let bound = 1. / (config.dim as f64).sqrt();
        let init_bs = candle_nn::Init::Uniform {
            lo: -bound,
            up: bound,
        };

        let vocab_projector_bias_vb = vb.pp("vocab_projector");
        let bs = vocab_projector_bias_vb.get_with_hints(config.vocab_size, "bias", init_bs)?;

        let vocab_projector = Linear::from_weights(ws, Some(bs));

        Ok(Self {
            distilbert,
            vocab_transform,
            vocab_layer_norm,
            vocab_projector,
            span: tracing::span!(tracing::Level::TRACE, "masked_lm"),
        })
    }
    
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        masked_lm_positions: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let sequence_output = self.distilbert.forward(input_ids, attention_mask)?;
        
        let prediction_logits = self.prediction_head_forward(&sequence_output)?;
        
        Ok(prediction_logits)
    }
    
    // The prediction head transforms the hidden states into vocabulary predictions
    fn prediction_head_forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let transformed = self.vocab_transform.forward(hidden_states)?;
        
        let transformed = transformed.gelu()?;
        
        let normalized = self.vocab_layer_norm.forward(&transformed)?;
        let logits = self.vocab_projector.forward(&normalized)?;
        
        Ok(logits)
    }
    
    pub fn compute_loss(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        labels: &Tensor,
    ) -> Result<Tensor> {
        let logits = self.forward(input_ids, attention_mask, None)?;
        let (batch_size, seq_length, vocab_size) = logits.dims3()?;
        let logits = logits.reshape(&[batch_size * seq_length, vocab_size])?;
        let labels = labels.reshape(&[batch_size * seq_length])?;
        let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
        
        Ok(loss)
    }
}