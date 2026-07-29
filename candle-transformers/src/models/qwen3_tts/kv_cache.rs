//! KV cache implementations for Qwen3-TTS autoregressive generation.

use candle::{DType, Device, Result, Tensor};

/// Simple concatenation-based KV cache.
pub struct KVCache {
    pub(crate) k: Option<Tensor>,
    pub(crate) v: Option<Tensor>,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
        let k = if let Some(prev) = &self.k {
            Tensor::cat(&[prev, k], 2)?
        } else {
            k.clone()
        };
        self.k = Some(k.clone());
        Ok(k)
    }

    pub fn update_v(&mut self, v: &Tensor) -> Result<Tensor> {
        let v = if let Some(prev) = &self.v {
            Tensor::cat(&[prev, v], 2)?
        } else {
            v.clone()
        };
        self.v = Some(v.clone());
        Ok(v)
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Pre-allocated fixed-size KV cache using `slice_set` for in-place writes.
pub struct PreAllocKVCache {
    k_buf: Tensor,
    v_buf: Tensor,
    current_len: usize,
    max_seq: usize,
    num_heads: usize,
    head_dim: usize,
}

impl PreAllocKVCache {
    pub fn new(
        batch: usize,
        num_heads: usize,
        max_seq: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (batch, num_heads, max_seq, head_dim);
        Ok(Self {
            k_buf: Tensor::zeros(shape, dtype, device)?,
            v_buf: Tensor::zeros(shape, dtype, device)?,
            current_len: 0,
            max_seq,
            num_heads,
            head_dim,
        })
    }

    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq = k.dim(2)?;
        let new_len = self.current_len + new_seq;
        if new_len > self.max_seq {
            candle::bail!(
                "KV cache overflow: current={} + new={} > max={}",
                self.current_len,
                new_seq,
                self.max_seq
            );
        }
        let k_c = k.contiguous()?;
        let v_c = v.contiguous()?;
        self.k_buf.slice_set(&k_c, 2, self.current_len)?;
        self.v_buf.slice_set(&v_c, 2, self.current_len)?;
        self.current_len = new_len;
        let k_view = self.k_buf.narrow(2, 0, new_len)?;
        let v_view = self.v_buf.narrow(2, 0, new_len)?;
        Ok((k_view, v_view))
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn len(&self) -> usize {
        self.current_len
    }

    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }
}

/// Unified KV cache: concat-based or pre-allocated.
pub enum AnyKVCache {
    Concat(KVCache),
    PreAlloc(PreAllocKVCache),
}

impl AnyKVCache {
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            AnyKVCache::Concat(c) => {
                let k = c.update_k(k)?;
                let v = c.update_v(v)?;
                Ok((k, v))
            }
            AnyKVCache::PreAlloc(c) => c.update(k, v),
        }
    }

    pub fn reset(&mut self) {
        match self {
            AnyKVCache::Concat(c) => c.reset(),
            AnyKVCache::PreAlloc(c) => c.reset(),
        }
    }
}
