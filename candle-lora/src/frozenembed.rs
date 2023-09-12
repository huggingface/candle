use candle::{Result, Tensor};
use candle_nn::Embedding;

use crate::EmbeddingLayerLike;

/// Embedding, but with a `new` implementation that ensures the embeddings are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenEmbedding {
    embed: Embedding,
}

impl FrozenEmbedding {
    pub(crate) fn new(embeddings: &Tensor, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            embed: Embedding::new(embeddings.detach()?, hidden_size),
        })
    }

    pub(crate) fn new_from_embed(old: &dyn EmbeddingLayerLike) -> Result<Self> {
        Self::new(old.embeddings(), old.hidden_size())
    }
}

impl crate::Module for FrozenEmbedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        self.embed.forward(indexes)
    }
}

impl EmbeddingLayerLike for FrozenEmbedding {
    fn embeddings(&self) -> &Tensor {
        self.embed.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.embed.hidden_size()
    }
}
