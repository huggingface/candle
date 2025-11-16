use candle::{Device, Module, Result, Tensor};
use candle_nn::{Dropout, VarBuilder};

use crate::models::debertav2::{Config, DebertaV2ContextPooler, DebertaV2Model, StableDropout};

// https://huggingface.co/naver/provence-reranker-debertav3-v1/blob/421f9139ad3f5ed919d9b04dd4ff02c10301dac9/modeling_provence.py
pub struct ProvenceOutput {
    pub compression_logits: Tensor,
    pub ranking_scores: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

pub struct ProvenceModel {
    pub device: Device,
    deberta: DebertaV2Model,
    pooler: DebertaV2ContextPooler,
    dropout: StableDropout,
    classifier: candle_nn::Linear,
    token_dropout: Dropout,
    token_classifier: candle_nn::Linear,
}

impl ProvenceModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // TODO: okay to hardcode?
        // For ranking (single score)
        let num_labels = 1;

        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let pooler = DebertaV2ContextPooler::load(vb.clone(), config)?;
        let output_dim = pooler.output_dim()?;

        let base_dropout = config.cls_dropout.unwrap_or(config.hidden_dropout_prob);

        // RANKING LAYER (on pooled output)
        let dropout = StableDropout::new(base_dropout);
        let classifier = candle_nn::linear(output_dim, num_labels, vb.root().pp("classifier"))?;

        // COMPRESSION LAYER (on token embeddings)
        let token_dropout = Dropout::new(base_dropout as f32);
        let token_classifier =
            candle_nn::linear(config.hidden_size, 1, vb.root().pp("classifier"))?;

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            pooler,
            dropout,
            classifier,
            token_dropout,
            token_classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<Tensor>,
    ) -> Result<ProvenceOutput> {
        let encoder_layer = self.deberta.forward(input_ids, None, attention_mask)?;

        // Ranking
        let pooled_output = self.pooler.forward(&encoder_layer)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        let ranking_logits = self.classifier.forward(&pooled_output)?;
        let ranking_scores = ranking_logits.squeeze(1)?;

        // Compression
        let token_output = self.token_dropout.forward(&encoder_layer, false)?;
        let compression_logits = self.token_classifier.forward(&token_output)?;

        Ok(ProvenceOutput {
            compression_logits,
            ranking_scores,
            // TODO: implement
            hidden_states: None,
            attentions: None,
        })
    }
}
