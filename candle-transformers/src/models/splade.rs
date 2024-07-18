use std::collections::HashMap;

use super::bert::{BertModel, BertPredictionHead, Config};
use candle_nn::VarBuilder;

use candle::{Device, Result, Tensor};

struct SpladePredictionHead {
    head: BertPredictionHead,
}

pub struct SpladeModel {
    bert: BertModel,
    head: BertPredictionHead,
    pub device: Device,
    span: tracing::Span,
}

impl SpladeModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let bert = BertModel::load(vb.pp("bert"), config)?;
        let head = BertPredictionHead::load(vb.clone(), config)?;

        Ok(Self {
            bert,
            head,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bert_output = self.bert.forward(input_ids, token_type_ids)?;

        let predictions = self.head.forward(&bert_output)?;
        let beans = (1. + predictions.relu()?)?.log()?.sum(1)?;
        Ok(beans)
    }

    pub fn sparse_forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        filter: Option<f32>,
    ) -> Result<HashMap<u32, f32>> {
        let dense_terms = self.forward(input_ids, token_type_ids)?;

        let filter = filter.unwrap_or(1.0);

        let value_map: HashMap<u32, f32> = dense_terms
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .enumerate()
            .filter(|&(idx, v)| v > filter && (1996 <= idx) && (idx < 29612))
            .map(|(idx, v)| {
                // Ensure idx (of type usize) is within u32 range
                let idx_u32 = idx as u32; // Directly cast usize to u32
                (idx_u32, v)
            })
            .collect();
        Ok(value_map)
    }
}
