use crate::{Result, Tensor};

pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Tensor::embedding(ids, &self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_embedding() {
        let weights =
            Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], (3, 2), &Device::Cpu).unwrap();
        let input_ids = Tensor::new(&[2, 1], &Device::Cpu).unwrap();
        let embedding = Embedding::new(weights);
        let out = embedding.forward(&input_ids).unwrap();
        assert_eq!(
            &out.storage_data::<f32>().unwrap()[..],
            &[4.0, 5.0, 2.0, 3.0]
        );

        // Invalid index
        let input_ids = Tensor::new(&[3, 1], &Device::Cpu).unwrap();
        assert!(embedding.forward(&input_ids).is_err());
    }
}
