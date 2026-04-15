//! Expert Bank — sparse FFN expert collection
//!
//! Contains n_experts independent Feed-Forward Networks.
//! Each expert: W_up (d_model → d_ff) → act → W_down (d_ff → d_model).
//!
//! Forward pass sparse-dispatches only the k_active experts per token.

use candle::{Result, Tensor};
use super::super::{Linear, VarBuilder, Module};
use std::collections::HashMap;

/// Expert Bank — collection of independent FFN experts
///
/// # Expert Structure
/// ```
/// input [d_model]
///   ↓ W_up [d_model, d_ff]
///   ↓
///   act (SiLU/GELU)
///   ↓
///   ↓ W_down [d_ff, d_model]
/// output [d_model]
/// ```
pub struct ExpertBank {
    experts: Vec<Expert>,
    n_experts: usize,
    k_active: usize,
    d_model: usize,
    d_ff: usize,
}

struct Expert {
    up: Linear,
    down: Linear,
    activation: Activation,
}

#[derive(Clone, Copy)]
enum Activation {
    Silu,
    Gelu,
}

impl Expert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up = self.up.forward(xs)?;
        let activated = match self.activation {
            Activation::Silu => up.silu()?,
            Activation::Gelu => up.gelu()?,
        };
        self.down.forward(&activated)
    }
}

impl ExpertBank {
    /// Create a new expert bank with n_experts
    pub fn new(
        n_experts: usize,
        k_active: usize,
        d_model: usize,
        d_ff: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut experts = Vec::with_capacity(n_experts);
        
        for i in 0..n_experts {
            let vb_i = vb.pp(format!("expert_{}", i));
            let up = super::super::linear(d_model, d_ff, vb_i.pp("up"))?;
            let down = super::super::linear(d_ff, d_model, vb_i.pp("down"))?;
            experts.push(Expert {
                up,
                down,
                activation: Activation::Silu,
            });
        }
        
        Ok(Self {
            experts,
            n_experts,
            k_active,
            d_model,
            d_ff,
        })
    }

    /// Forward pass with sparse expert dispatch
    ///
    /// # Arguments
    /// * `xs` - Input tokens [batch*seq, d_model]
    /// * `expert_ids` - Selected expert indices [batch*seq, k_active]
    ///
    /// # Returns
    /// Map of expert_id → output tensor [active_count * d_model]
    pub fn forward(&self, xs: &Tensor, expert_ids: &Tensor) -> Result<HashMap<usize, Tensor>> {
        let (_batch, d_model) = xs.dims2()?;
        assert_eq!(d_model, self.d_model, "Input dim mismatch");
        
        let expert_ids_vec: Vec<u32> = expert_ids.to_vec1()?;
        
        // Collect which experts are active
        let mut active_experts: Vec<usize> = expert_ids_vec
            .iter()
            .map(|&id| id as usize)
            .filter(|&id| id < self.n_experts)
            .collect();
        active_experts.sort();
        active_experts.dedup();
        
        // Dispatch to active experts
        let mut outputs: HashMap<usize, Tensor> = HashMap::new();
        
        for &expert_id in &active_experts {
            // Get indices for this expert
            let indices: Vec<usize> = expert_ids_vec
                .iter()
                .enumerate()
                .filter(|(_, &id)| id as usize == expert_id)
                .map(|(i, _)| i)
                .collect();
            
            if indices.is_empty() {
                continue;
            }
            
            // Gather tokens for this expert
            let mut gathered = Vec::with_capacity(indices.len() * d_model);
            for &idx in &indices {
                let token = xs.get(idx)?;
                let token_vec: Vec<f32> = token.to_vec1()?;
                gathered.extend(token_vec);
            }
            
            let input = Tensor::from_slice(
                &gathered,
                (indices.len(), d_model),
                xs.device(),
            )?;
            
            // Forward through expert
            let output = self.experts[expert_id].forward(&input)?;
            outputs.insert(expert_id, output);
        }
        
        Ok(outputs)
    }

    /// Get number of experts
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Get active expert count
    pub fn k_active(&self) -> usize {
        self.k_active
    }

    /// Get expert dimensions
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.n_experts, self.d_model, self.d_ff)
    }

    /// Total parameters in the expert bank
    pub fn total_params(&self) -> usize {
        // Each expert: d_model * d_ff (up) + d_ff * d_model (down) = 2 * d_model * d_ff
        self.n_experts * 2 * self.d_model * self.d_ff
    }
}

/// Individual expert wrapper for inspection
pub struct ExpertView<'a> {
    bank: &'a ExpertBank,
    index: usize,
}

impl<'a> ExpertView<'a> {
    pub fn new(bank: &'a ExpertBank, index: usize) -> Self {
        Self { bank, index }
    }
    
    pub fn weight_up(&self) -> Result<&Tensor> {
        Ok(self.bank.experts[self.index].up.weight())
    }
    
    pub fn weight_down(&self) -> Result<&Tensor> {
        Ok(self.bank.experts[self.index].down.weight())
    }
    
    pub fn forward_single(&self, xs: &Tensor) -> Result<Tensor> {
        self.bank.experts[self.index].forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_expert_bank_creation() {
        let bank = ExpertBank::new(64, 2, 512, 2048, VarBuilder::from_device(Device::Cpu)).unwrap();
        assert_eq!(bank.n_experts(), 64);
        assert_eq!(bank.total_params(), 64 * 2 * 512 * 2048);
    }
    
    #[test]
    fn test_expert_forward() {
        let bank = ExpertBank::new(8, 2, 16, 64, VarBuilder::from_device(Device::Cpu)).unwrap();
        
        // Create dummy input [batch=2, d_model=16]
        let xs = Tensor::randn(0.0, 1.0, (2, 16), &Device::Cpu).unwrap();
        
        // Select expert 0 and 1
        let expert_ids = Tensor::new(&[[0u32, 1], [2u32, 3]], xs.device()).unwrap();
        
        let outputs = bank.forward(&xs, &expert_ids).unwrap();
        assert!(outputs.contains_key(&0));
        assert!(outputs.contains_key(&1));
        assert!(outputs.contains_key(&2));
        assert!(outputs.contains_key(&3));
    }
}