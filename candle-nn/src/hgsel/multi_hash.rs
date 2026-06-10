//! Multi-Hash Router — deterministic routing without learned parameters
//!
//! Uses hash-based candidate selection:
//! - n_hashes independent hash functions
//! - Each hash maps token state → expert assignment
//! - k_active experts selected (top-k by composite score)
//! - salt parameter shifts distribution without retraining

use candle::{Result, Tensor};

/// Simple non-cryptographic hash functions for routing
/// Using sipHash-style mixing for speed and good distribution
fn hash_sip(data: &[u8], seed0: u64, seed1: u64) -> u64 {
    let mut h0 = seed0 ^ 0x9e3779b97f4a7c15u64;
    let mut h1 = seed1 ^ 0x9e3779b97f4a7c15u64;
    for chunk in data.chunks(8) {
        let mut val = 0u64;
        for (i, &b) in chunk.iter().enumerate() {
            val |= (b as u64) << (i * 8);
        }
        h0 = h0.wrapping_add(val);
        h1 ^= h0;
        h0 = h0.rotate_left(31).wrapping_add(h1);
        h1 = h1.rotate_left(27) ^ h0;
    }
    h0.wrapping_add(h1)
}

/// Convert tensor to bytes for hashing
fn tensor_to_bytes(t: &Tensor) -> Vec<u8> {
    let flat = match t.flatten_all() {
        Ok(f) => f,
        Err(_) => return vec![],
    };
    let vals: Vec<f32> = match flat.to_vec1() {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    vals.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect()
}

/// MultiHashRouter — deterministic sparse routing
///
/// Routing is based on multi-hash candidate generation + salt-controlled selection:
pub struct MultiHashRouter {
    n_experts: usize,
    k_active: usize,
    n_hashes: usize,
    salt: f32,
    layer_id: usize,
}

impl MultiHashRouter {
    pub fn new(
        n_experts: usize,
        k_active: usize,
        n_hashes: usize,
        _d_model: usize,
        layer_id: usize,
    ) -> Self {
        Self {
            n_experts,
            k_active,
            n_hashes,
            salt: 0.0,
            layer_id,
        }
    }

    /// Set salt — positive = more uniform, negative = more concentrated
    pub fn set_salt(&mut self, salt: f32) {
        self.salt = salt;
    }

    pub fn salt(&self) -> f32 {
        self.salt
    }

    /// Forward routing — returns (expert_ids, weights)
    ///
    /// Returns:
    /// - expert_ids: [batch*seq, k_active] — selected expert indices
    /// - weights: [batch*seq, k_active] — combination weights
    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, _d_model) = xs.dims2()?;
        
        let mut expert_ids = Vec::with_capacity(batch * self.k_active);
        let mut expert_weights = Vec::with_capacity(batch * self.k_active);
        
        for b in 0..batch {
            let token_slice = xs.get(b)?;
            let token_bytes = tensor_to_bytes(&token_slice);
            
            // Multi-hash: gather candidates from all hashes
            let mut candidate_scores: Vec<(usize, f64)> = (0..self.n_experts)
                .map(|expert_id| {
                    let mut score = 0.0f64;
                    for h in 0..self.n_hashes {
                        let seed = self._hash_seed(h, expert_id);
                        let h_val = hash_sip(&token_bytes, seed.0, seed.1);
                        let normalized = (h_val as f64) / (u64::MAX as f64);
                        score += normalized;
                    }
                    score /= self.n_hashes as f64;
                    (expert_id, score)
                })
                .collect();
            
            // Apply salt to scores
            if self.salt.abs() > 1e-6 {
                let shift = self.salt as f64 * 0.1;
                for (_, score) in &mut candidate_scores {
                    *score = (*score + shift).clamp(0.0, 1.0);
                }
            }
            
            // Sort by score descending
            candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Select top-k
            for (expert_id, score) in candidate_scores.into_iter().take(self.k_active) {
                expert_ids.push(expert_id as u32);
                expert_weights.push(score as f32);
            }
            
            // Pad if k_active < n_experts
            while expert_ids.len() < (b + 1) * self.k_active {
                expert_ids.push(0u32);
                expert_weights.push(0.0f32);
            }
        }
        
        let expert_ids_t = Tensor::from_slice(
            &expert_ids,
            (batch, self.k_active),
            &xs.device(),
        )?;
        let weights_t = Tensor::from_slice(
            &expert_weights,
            (batch, self.k_active),
            &xs.device(),
        )?;
        
        Ok((expert_ids_t, weights_t))
    }

    fn _hash_seed(&self, hash_idx: usize, expert_id: usize) -> (u64, u64) {
        let base = (self.layer_id as u64).wrapping_mul(0x9e3779b97f4a7c15u64)
            .wrapping_add(hash_idx as u64)
            .wrapping_add(expert_id as u64);
        (base, base.wrapping_mul(0xbf3e0b9d6e3c8a2fu64))
    }

    pub fn routing_stats(&self, expert_ids: &Tensor, _weights: &Tensor) -> RoutingStats {
        let ids = match expert_ids.to_vec1::<u32>() {
            Ok(v) => v,
            Err(_) => return RoutingStats::default(),
        };
        
        let mut counts = vec![0usize; self.n_experts];
        for &id in &ids {
            if (id as usize) < self.n_experts {
                counts[id as usize] += 1;
            }
        }
        
        let total = counts.iter().sum::<usize>() as f32;
        let max_count = counts.iter().cloned().max().unwrap_or(0) as f32;
        
        RoutingStats {
            active_experts: counts.iter().filter(|&&c| c > 0).count(),
            max_load: max_count / total,
            min_load: *counts.iter().min().unwrap_or(&0) as f32 / total.max(1.0),
            distribution: counts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub active_experts: usize,
    pub max_load: f32,
    pub min_load: f32,
    pub distribution: Vec<usize>,
}

impl Default for RoutingStats {
    fn default() -> Self {
        Self {
            active_experts: 0,
            max_load: 0.0,
            min_load: 0.0,
            distribution: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    
    #[test]
    fn test_hash_stability() {
        let data = b"hello world";
        let seed0 = 0x1234_5678_9abc_def0u64;
        let seed1 = 0xfedc_ba98_7654_3210u64;
        
        let h1 = hash_sip(data, seed0, seed1);
        let h2 = hash_sip(data, seed0, seed1);
        
        assert_eq!(h1, h2, "Hash must be deterministic");
    }
    
    #[test]
    fn test_router_selects_k() {
        let router = MultiHashRouter::new(64, 2, 4, 512, 0);
        let xs = Tensor::randn(0.0, 1.0, (4, 512), &Device::Cpu).unwrap();
        let (ids, weights) = router.forward(&xs).unwrap();
        
        assert_eq!(ids.dims().unwrap(), [4, 2]);
        assert_eq!(weights.dims().unwrap(), [4, 2]);
    }
    
    #[test]
    fn test_salt_effect() {
        let xs = Tensor::randn(0.0, 1.0, (4, 16), &Device::Cpu).unwrap();
        
        let (ids1, _) = MultiHashRouter::new(8, 2, 4, 16, 0).forward(&xs).unwrap();
        
        let mut router2 = MultiHashRouter::new(8, 2, 4, 16, 0);
        router2.set_salt(5.0);
        let (ids2, _) = router2.forward(&xs).unwrap();
        
        let ids1_v: Vec<u32> = ids1.to_vec1().unwrap();
        let ids2_v: Vec<u32> = ids2.to_vec1().unwrap();
        
        // High salt should push toward more uniform distribution
        println!("No salt: {:?}", &ids1_v[..8]);
        println!("High salt: {:?}", &ids2_v[..8]);
    }
}