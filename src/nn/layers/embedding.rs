use crate::{Result, Tensor};

/// TODO
#[derive(Clone)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        todo!("embedding")
        // self.weight.select(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let weights = Tensor::zeros((3, 2), DType::f32, &Device::Cpu);
        let embedding = Embedding::new(weights);
        let out = embedding.forward(&[0, 1]).unwrap();
    }

    #[test]
    fn test_embedding_errors() {
        let weights = Tensor::zeros(vec![3, 2]);
        let embedding = Embedding::new(weights);
        let mut out = Tensor::zeros(vec![2, 2]);
        assert!(embedding.forward(&[3], &mut out).is_err());
    }
}
