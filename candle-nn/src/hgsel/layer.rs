//! HGSELLayer — main entry point for HGSEL sparse MoE
//!
//! Replaces standard MLP with a sparse expert bank routed by multi-hash.

use super::{MultiHashRouter, ExpertBank, CombineMode, CombineFactory, CombineStrategy};
use candle::{Result, Tensor, Module};
use std::collections::HashMap;

/// HGSEL Layer — replaces standard MLP with sparse expert bank
pub struct HGSELLayer {
    router: MultiHashRouter,
    expert_bank: ExpertBank,
    combine: Box<dyn CombineStrategy>,
    k_active: usize,
    n_experts: usize,
    layer_id: usize,
    expert_load_ema: HashMap<usize, f32>,
    ema_decay: f32,
}

impl HGSELLayer {
    /// Create a new HGSELLayer
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_experts: usize,
        k_active: usize,
        n_hashes: usize,
        combine_mode: CombineMode,
        layer_id: usize,
        vb: crate::VarBuilder,
    ) -> Result<Self> {
        let router = MultiHashRouter::new(n_experts, k_active, n_hashes, d_model, layer_id);
        let expert_bank = ExpertBank::new(n_experts, k_active, d_model, d_ff, vb.clone())?;
        let combine = CombineFactory::create(combine_mode, k_active, d_model, vb)?;
        
        let expert_load_ema = (0..n_experts)
            .map(|i| (i, 1.0 / n_experts as f32))
            .collect();
        
        Ok(Self {
            router,
            expert_bank,
            combine,
            k_active,
            n_experts,
            layer_id,
            expert_load_ema,
            ema_decay: 0.99,
        })
    }

    /// Set salt — positive = more uniform, negative = more concentrated
    pub fn set_salt(&mut self, salt: f32) {
        self.router.set_salt(salt);
    }

    pub fn salt(&self) -> f32 {
        self.router.salt()
    }

    /// Forward pass with sparse expert routing
    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (original_shape, xs) = self._prepare_input(xs)?;
        
        let (expert_ids, expert_weights) = self.router.forward(&xs)?;
        let expert_outputs = self.expert_bank.forward(&xs, &expert_ids)?;
        let output = self.combine.combine(&expert_outputs, &expert_weights)?;
        
        self._update_load_ema(&expert_ids);
        
        self._restore_shape(output, original_shape)
    }

    /// Forward with routing diagnostics
    pub fn forward_with_info(&mut self, xs: &Tensor) -> Result<(Tensor, HGSELDiagnostics)> {
        let (original_shape, xs_flat) = self._prepare_input(xs)?;
        
        let (expert_ids, expert_weights) = self.router.forward(&xs_flat)?;
        let expert_outputs = self.expert_bank.forward(&xs_flat, &expert_ids)?;
        let output = self.combine.combine(&expert_outputs, &expert_weights)?;
        
        let output = self._restore_shape(output, original_shape.clone())?;
        
        let diagnostics = HGSELDiagnostics {
            selected_experts: expert_ids.to_vec1::<u32>()?,
            expert_weights: expert_weights.to_vec1::<f32>()?,
            routing_entropy: self.routing_entropy(),
            expert_load_ema: self.expert_load_ema.clone(),
            batch_tokens: xs_flat.dim(0)?,
        };
        
        Ok((output, diagnostics))
    }

    /// Get routing entropy (0 = concentrated, 1 = uniform)
    pub fn routing_entropy(&self) -> f32 {
        let loads: Vec<f32> = (0..self.n_experts)
            .map(|i| *self.expert_load_ema.get(&i).unwrap_or(&0.0))
            .collect();
        let sum: f32 = loads.iter().sum();
        if sum == 0.0 { return 0.5; }
        let probs: Vec<f32> = loads.iter().map(|l| l / sum).collect();
        let entropy = -probs.iter()
            .filter(|p| **p > 1e-8)
            .map(|p| p * p.ln())
            .sum::<f32>();
        let max_entropy = (self.n_experts as f32).ln();
        (entropy / max_entropy).max(0.0).min(1.0)
    }

    /// Get expert load statistics
    pub fn expert_load_stats(&self) -> ExpertLoadStats {
        let loads: Vec<f32> = (0..self.n_experts)
            .map(|i| *self.expert_load_ema.get(&i).unwrap_or(&0.0))
            .collect();
        let mean = loads.iter().sum::<f32>() / loads.len() as f32;
        let variance = loads.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / loads.len() as f32;
        ExpertLoadStats {
            mean,
            std: variance.sqrt(),
            min: loads.iter().cloned().fold(f32::INFINITY, f32::min),
            max: loads.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            entropy: self.routing_entropy(),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        let uniform = 1.0 / self.n_experts as f32;
        self.expert_load_ema = (0..self.n_experts)
            .map(|i| (i, uniform))
            .collect();
    }

    fn _prepare_input(&self, xs: &Tensor) -> Result<(Option<Vec<usize>>, Tensor)> {
        let dims = xs.dims();
        match dims.as_ref() {
            [b, s, d] => {
                let flat = xs.reshape(((*b) * (*s), *d))?;
                Ok((Some(vec![*b, *s, *d]), flat))
            }
            [_n, _d] => Ok((None, xs.clone())),
            _ => candle::bail!("HGSEL input must be [batch, seq, d] or [batch*seq, d]"),
        }
    }

    fn _restore_shape(&self, xs: Tensor, shape: Option<Vec<usize>>) -> Result<Tensor> {
        match shape {
            Some(ref v) if v.len() == 3 => xs.reshape((v[0], v[1], v[2])),
            _ => Ok(xs),
        }
    }

    fn _update_load_ema(&mut self, expert_ids: &Tensor) {
        let ids = match expert_ids.to_vec1::<u32>() {
            Ok(v) => v,
            Err(_) => return,
        };
        
        let mut counts: HashMap<usize, f32> = HashMap::new();
        for &id in &ids {
            let id = id as usize;
            if id < self.n_experts {
                *counts.entry(id).or_insert(0.0) += 1.0;
            }
        }
        
        let n = ids.len() as f32;
        if n == 0.0 { return; }
        for (i, count) in counts {
            let normalized = count / n;
            let current = *self.expert_load_ema.get(&i).unwrap_or(&0.0);
            let updated = self.ema_decay * current + (1.0 - self.ema_decay) * normalized;
            self.expert_load_ema.insert(i, updated);
        }
    }
}

/// Diagnostics returned by forward_with_info
#[derive(Debug, Clone)]
pub struct HGSELDiagnostics {
    pub selected_experts: Vec<u32>,
    pub expert_weights: Vec<f32>,
    pub routing_entropy: f32,
    pub expert_load_ema: HashMap<usize, f32>,
    pub batch_tokens: usize,
}

/// Expert load statistics
#[derive(Debug, Clone)]
pub struct ExpertLoadStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub entropy: f32,
}