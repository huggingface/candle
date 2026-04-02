//! Gemma4 multimodal embedder: projects modality features into language model space.
//!
//! Simply: RMSNorm (no learnable scale) + linear projection.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// Bare RMS normalization without learnable parameters.
#[derive(Debug, Clone)]
struct BareRmsNorm {
    eps: f64,
}

impl BareRmsNorm {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for BareRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let hidden_size = x.dim(D::Minus1)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalEmbedder {
    embedding_projection: candle_nn::Linear,
    embedding_pre_projection_norm: BareRmsNorm,
}

impl MultimodalEmbedder {
    pub fn new(
        multimodal_hidden_size: usize,
        text_hidden_size: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding_projection = candle_nn::linear_no_bias(
            multimodal_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;
        let embedding_pre_projection_norm = BareRmsNorm::new(eps);
        Ok(Self {
            embedding_projection,
            embedding_pre_projection_norm,
        })
    }

    pub fn forward(&self, soft_features: &Tensor) -> Result<Tensor> {
        let normed = self.embedding_pre_projection_norm.forward(soft_features)?;
        self.embedding_projection.forward(&normed)
    }
}
