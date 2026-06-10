//! Expert output combination strategies
//!
//! After sparse expert dispatch, outputs must be merged back per token.
//! Different strategies offer different tradeoffs:

use candle::{Result, Tensor};
use super::super::{Linear, VarBuilder, Module};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombineMode {
    /// Equal weighting of active experts (default)
    Uniform,
    /// Scalar-weighted by routing score
    Scalar,
    /// Learned linear combination (adds parameters)
    Learned,
}

/// Strategy trait for combining expert outputs
pub trait CombineStrategy: Send + Sync {
    fn combine(
        &self,
        expert_outputs: &HashMap<usize, Tensor>,
        expert_weights: &Tensor,
    ) -> Result<Tensor>;
}

/// Uniform: all active experts contribute equally
pub struct UniformCombine {
    k_active: usize,
}

impl UniformCombine {
    pub fn new(k_active: usize) -> Self {
        Self { k_active }
    }
}

impl CombineStrategy for UniformCombine {
    fn combine(
        &self,
        expert_outputs: &HashMap<usize, Tensor>,
        expert_weights: &Tensor,
    ) -> Result<Tensor> {
        // Weights are ignored for uniform - just average active experts
        let batch_size = expert_weights.dim(0)?;
        
        // Get all outputs and reconstruct per-token averages
        let weight_vec: Vec<u32> = expert_weights.to_vec1()?;
        
        let mut result = Vec::with_capacity(batch_size * 512); // Assume d_model=512
        for b in 0..batch_size {
            let mut token_result = Vec::new();
            
            for k in 0..self.k_active {
                let expert_id = weight_vec[b * self.k_active + k] as usize;
                if let Some(output) = expert_outputs.get(&expert_id) {
                    let output_vec: Vec<f32> = output.to_vec1()?;
                    if token_result.is_empty() {
                        token_result = output_vec;
                    } else {
                        // Accumulate
                        for (i, v) in output_vec.iter().enumerate() {
                            token_result[i] += v;
                        }
                    }
                }
            }
            
            // Divide by k_active (count of active experts that contributed)
            for v in &mut token_result {
                *v /= self.k_active as f32;
            }
            
            result.extend(token_result);
        }
        
        let d_model = 512; // We need to track this properly
        let out = Tensor::from_slice(&result, (batch_size, d_model), &expert_weights.device())?;
        Ok(out)
    }
}

/// Scalar-weighted: combine by routing scores
pub struct ScalarCombine {
    _k_active: usize,
}

impl ScalarCombine {
    pub fn new(k_active: usize) -> Self {
        Self { _k_active: k_active }
    }
}

impl CombineStrategy for ScalarCombine {
    fn combine(
        &self,
        expert_outputs: &HashMap<usize, Tensor>,
        expert_weights: &Tensor,
    ) -> Result<Tensor> {
        let weight_vec: Vec<f32> = expert_weights.to_vec1()?;
        let batch_size = expert_weights.dim(0)?;
        
        // Infer d_model from first output
        let first_output = expert_outputs.values().next()
            .ok_or_else(|| candle::Error::Msg("No expert outputs".to_string()))?;
        let d_model = first_output.dim(1)?;
        
        let k_active = weight_vec.len() / batch_size;
        
        let mut result = Vec::with_capacity(batch_size * d_model);
        
        for b in 0..batch_size {
            let mut weighted_sum = vec![0.0f32; d_model];
            let mut weight_sum = 0.0f32;
            
            for k in 0..k_active {
                let expert_id = weight_vec[b * k_active + k] as usize;
                let weight = weight_vec[b * k_active + k]; // same value, used as both id and weight... fix this
                
                if let Some(output) = expert_outputs.get(&expert_id) {
                    let output_vec: Vec<f32> = output.to_vec1()?;
                    for (i, v) in output_vec.iter().enumerate() {
                        weighted_sum[i] += v * weight;
                    }
                    weight_sum += weight;
                }
            }
            
            // Normalize
            if weight_sum > 0.0 {
                for v in &mut weighted_sum {
                    *v /= weight_sum;
                }
            }
            
            result.extend(weighted_sum);
        }
        
        let out = Tensor::from_slice(&result, (batch_size, d_model), &expert_weights.device())?;
        Ok(out)
    }
}

/// Learned linear combination
pub struct LearnedCombine {
    linear: Linear,
    k_active: usize,
}

impl LearnedCombine {
    pub fn new(d_model: usize, k_active: usize, vb: VarBuilder) -> Result<Self> {
        // Input: k_active * d_model (stacked expert outputs)
        // Output: d_model (learned combination)
        let linear = super::super::linear(k_active * d_model, d_model, vb)?;
        Ok(Self { linear, k_active })
    }
}

impl CombineStrategy for LearnedCombine {
    fn combine(
        &self,
        expert_outputs: &HashMap<usize, Tensor>,
        expert_weights: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = expert_weights.dim(0)?;
        let weight_vec: Vec<u32> = expert_weights.to_vec1()?;
        let k_active = weight_vec.len() / batch_size;
        
        // Infer d_model
        let first_output = expert_outputs.values().next()
            .ok_or_else(|| candle::Error::Msg("No expert outputs".to_string()))?;
        let d_model = first_output.dim(1)?;
        
        // Stack expert outputs per token
        let mut stacked = Vec::with_capacity(batch_size * k_active * d_model);
        
        for b in 0..batch_size {
            let mut token_stack = Vec::with_capacity(k_active * d_model);
            
            for k in 0..k_active {
                let expert_id = weight_vec[b * k_active + k] as usize;
                if let Some(output) = expert_outputs.get(&expert_id) {
                    token_stack.extend(output.to_vec1::<f32>()?);
                } else {
                    // Zero fill for missing expert
                    token_stack.extend(vec![0.0f32; d_model]);
                }
            }
            
            stacked.extend(token_stack);
        }
        
        let input = Tensor::from_slice(&stacked, (batch_size, k_active * d_model), &expert_weights.device())?;
        let output = self.linear.forward(&input)?;
        
        Ok(output)
    }
}

/// Factory for creating combine strategies
pub struct CombineFactory;

impl CombineFactory {
    pub fn create(
        mode: CombineMode,
        k_active: usize,
        d_model: usize,
        vb: VarBuilder,
    ) -> Result<Box<dyn CombineStrategy>> {
        match mode {
            CombineMode::Uniform => Ok(Box::new(UniformCombine::new(k_active))),
            CombineMode::Scalar => Ok(Box::new(ScalarCombine::new(k_active))),
            CombineMode::Learned => Ok(Box::new(LearnedCombine::new(d_model, k_active, vb)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    
    #[test]
    fn test_uniform_combine() {
        let strategy = UniformCombine::new(2);
        
        let mut outputs = HashMap::new();
        outputs.insert(0, Tensor::randn(0.0, 1.0, (4, 512), &Device::Cpu).unwrap());
        outputs.insert(1, Tensor::randn(0.0, 1.0, (4, 512), &Device::Cpu).unwrap());
        
        let weights = Tensor::new(&[[0u32, 1], [0u32, 1], [1u32, 2], [1u32, 2]], &Device::Cpu).unwrap();
        
        let result = strategy.combine(&outputs, &weights);
        assert!(result.is_ok());
    }
}