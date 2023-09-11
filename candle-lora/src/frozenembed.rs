use candle::{Result, Tensor};

use crate::EmbeddingLayerLike;

/// Embedding, but with a `new` implementation that ensures the embeddings are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenEmbedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl FrozenEmbedding {
    pub(crate) fn new(embeddings: &Tensor, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            embeddings: embeddings.detach()?,
            hidden_size,
        })
    }

    pub(crate) fn new_from_embed(old: &dyn EmbeddingLayerLike) -> Result<Self> {
        Self::new(old.embeddings(), old.hidden_size())
    }
}

impl crate::Module for FrozenEmbedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

impl EmbeddingLayerLike for FrozenEmbedding {
    fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
