//! Embedding Layer.
use candle::{BackendStorage, Result, Tensor};

#[derive(Clone, Debug)]
pub struct Embedding<B: BackendStorage> {
    embeddings: Tensor<B>,
    hidden_size: usize,
}

impl<B: BackendStorage> Embedding<B> {
    pub fn new(embeddings: Tensor<B>, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor<B> {
        &self.embeddings
    }

    /// Get the hidden size of the embedding matrix
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl<B: BackendStorage> crate::Module<B> for Embedding<B> {
    fn forward(&self, indexes: &Tensor<B>) -> Result<Tensor<B>> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

pub fn embedding<B: BackendStorage>(
    in_size: usize,
    out_size: usize,
    vb: crate::VarBuilder<B>,
) -> Result<Embedding<B>> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        crate::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}
