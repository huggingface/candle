//! High-performance GPU-native quantized KV cache for language models.
//!
//! This module provides a `QuantizedKvCache` that implements memory-efficient KV storage
//! using 1-4 bit quantization (QJL, PolarQuant, TurboQuant) with high-speed GPU-vectorized dequantization.

use candle::{Device, Result, Tensor};
use crate::quant_kv::{
    fwht,
    polar_quant::{
        polar_attention_scores_vectorized, polar_quantize_tensor, PolarQuantConfig, PolarQuantTensors,
    },
    qjl::{
        qjl_attention_scores_vectorized, qjl_quantize_tensor, QjlConfig, QjlTensors,
    },
    turbo_quant::{
        turbo_attention_scores_vectorized, turbo_quantize_tensor, TurboQuantConfig, TurboQuantTensors,
    },
};

/// Supported quantization algorithms.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantAlgorithm {
    Qjl(QjlConfig),
    PolarQuant(PolarQuantConfig),
    TurboQuant(TurboQuantConfig),
}

/// A quantized key cache stored on the GPU.
#[derive(Debug, Clone)]
pub enum QuantKCache {
    Qjl(QjlTensors),
    PolarQuant(PolarQuantTensors),
    TurboQuant(TurboQuantTensors),
}

/// A standard value cache stored as full precision on the GPU.
#[derive(Debug, Clone)]
pub struct ValueCache {
    pub data: Vec<Tensor>,
    pub cur_len: usize,
}

impl ValueCache {
    pub fn new() -> Self {
        Self { data: Vec::new(), cur_len: 0 }
    }
}

/// A high-performance quantized KV cache.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    pub k_cache: QuantKCache,
    pub v_cache: ValueCache,
    pub algorithm: QuantAlgorithm,
    pub max_seq_len: usize,
}

impl QuantizedKvCache {
    pub fn new(
        num_heads: usize,
        max_seq_len: usize,
        dim: usize,
        algorithm: QuantAlgorithm,
        device: &Device,
    ) -> Result<Self> {
        let k_cache = match &algorithm {
            QuantAlgorithm::Qjl(cfg) => QuantKCache::Qjl(QjlTensors::new(num_heads, max_seq_len, cfg.dim / 8, device)?),
            QuantAlgorithm::PolarQuant(_) => QuantKCache::PolarQuant(PolarQuantTensors::new(num_heads, max_seq_len, dim, device)?),
            QuantAlgorithm::TurboQuant(_) => QuantKCache::TurboQuant(TurboQuantTensors::new(num_heads, max_seq_len, dim, device)?),
        };

        Ok(Self {
            k_cache,
            v_cache: ValueCache::new(),
            algorithm,
            max_seq_len,
        })
    }

    pub fn current_seq_len(&self) -> usize {
        match &self.k_cache {
            QuantKCache::Qjl(t) => t.cur_len,
            QuantKCache::PolarQuant(t) => t.cur_len,
            QuantKCache::TurboQuant(t) => t.cur_len,
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let k_dims = k.dims();
        let seq_len = k_dims[k_dims.len() - 2];
        
        // Quantize and push Key
        match &mut self.k_cache {
            QuantKCache::Qjl(t) => {
                let cfg = match &self.algorithm { QuantAlgorithm::Qjl(cfg) => cfg, _ => unreachable!() };
                let (bits, norms) = qjl_quantize_tensor(k, cfg)?;
                t.sign_bits.push(bits);
                t.norms.push(norms);
                t.cur_len += seq_len;
            },
            QuantKCache::PolarQuant(t) => {
                let cfg = match &self.algorithm { QuantAlgorithm::PolarQuant(cfg) => cfg, _ => unreachable!() };
                let (l1, ln, r) = polar_quantize_tensor(k, cfg)?;
                t.level1_codes.push(l1);
                t.leveln_codes.push(ln);
                t.radii.push(r);
                t.cur_len += seq_len;
            },
            QuantKCache::TurboQuant(t) => {
                let cfg = match &self.algorithm { QuantAlgorithm::TurboQuant(cfg) => cfg, _ => unreachable!() };
                let ((l1, ln, r), (bits, norms)) = turbo_quantize_tensor(k, cfg)?;
                t.mse_tensors.level1_codes.push(l1);
                t.mse_tensors.leveln_codes.push(ln);
                t.mse_tensors.radii.push(r);
                t.qjl_tensors.sign_bits.push(bits);
                t.qjl_tensors.norms.push(norms);
                t.mse_tensors.cur_len += seq_len;
                t.qjl_tensors.cur_len += seq_len;
                t.cur_len += seq_len;
            }
        }

        // Push Value
        self.v_cache.data.push(v.clone());
        self.v_cache.cur_len += seq_len;
        
        Ok(())
    }

    pub fn attention_scores(&self, q: &Tensor) -> Result<Tensor> {
        match &self.k_cache {
            QuantKCache::Qjl(t) => {
                let cfg = match &self.algorithm { QuantAlgorithm::Qjl(cfg) => cfg, _ => unreachable!() };
                qjl_attention_scores_vectorized(q, t, cfg)
            },
            QuantKCache::PolarQuant(t) => {
                let cfg = match &self.algorithm { QuantAlgorithm::PolarQuant(cfg) => cfg, _ => unreachable!() };
                polar_attention_scores_vectorized(q, t, cfg)
            },
            QuantKCache::TurboQuant(t) => {
                let cfg = match &self.algorithm { QuantAlgorithm::TurboQuant(cfg) => cfg, _ => unreachable!() };
                turbo_attention_scores_vectorized(q, t, cfg)
            }
        }
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        // Implementation of full K dequantization if needed, 
        // but typically models use vectorized scores.
        Ok(None)
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        if self.v_cache.data.is_empty() {
            Ok(None)
        } else {
            let cat_dim = self.v_cache.data[0].dims().len() - 2;
            Tensor::cat(&self.v_cache.data, cat_dim).map(Some)
        }
    }
}
