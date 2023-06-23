#![allow(unreachable_patterns, dead_code)]
use crate::{Result, Tensor};

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, epsilon: f32) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    pub fn forward(&self, tensor: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(tensor.clone())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::cpu::f32::Tensor;
//
//     #[test]
//     fn test_layer_norm() {
//         let mut zeros = Tensor::zeros(vec![3, 2]);
//         let weights = Tensor::zeros(vec![2]);
//         let bias = Tensor::zeros(vec![2]);
//
//         let linear = LayerNorm::new(weights, bias, 1e-5);
//
//         linear.forward(&mut zeros).unwrap();
//     }
// }
