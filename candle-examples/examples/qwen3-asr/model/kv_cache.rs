//! KV cache implementation for autoregressive generation.
//!
//! This is a minimal, Candle-friendly cache for transformer self-attention key/value
//! tensors. It mirrors the conceptual API of HuggingFace "dynamic cache".

use candle::{Result, Tensor};

/// Per-layer key/value cache entry.
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Cached key tensor: (batch, num_kv_heads, seq_len, head_dim)
    pub key: Tensor,
    /// Cached value tensor: (batch, num_kv_heads, seq_len, head_dim)
    pub value: Tensor,
}

impl KVCacheEntry {
    pub fn new(key: Tensor, value: Tensor) -> Self {
        Self { key, value }
    }

    pub fn seq_len(&self) -> Result<usize> {
        self.key.dim(2)
    }

    pub fn update(&mut self, new_key: &Tensor, new_value: &Tensor) -> Result<(Tensor, Tensor)> {
        self.key = Tensor::cat(&[&self.key, new_key], 2)?;
        self.value = Tensor::cat(&[&self.value, new_value], 2)?;
        Ok((self.key.clone(), self.value.clone()))
    }
}

/// Dynamic KV-cache that grows as generation progresses.
#[derive(Debug, Clone, Default)]
pub struct KVCache {
    entries: Vec<Option<KVCacheEntry>>,
    seq_len: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            seq_len: 0,
        }
    }

    pub fn with_num_layers(num_layers: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
            seq_len: 0,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn get(&self, layer_idx: usize) -> Option<&KVCacheEntry> {
        self.entries.get(layer_idx).and_then(|e| e.as_ref())
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        while self.entries.len() <= layer_idx {
            self.entries.push(None);
        }

        let new_len = key.dim(2)?;

        match &mut self.entries[layer_idx] {
            Some(entry) => {
                let result = entry.update(key, value)?;
                if layer_idx == 0 {
                    self.seq_len = self.seq_len.saturating_add(new_len);
                }
                Ok(result)
            }
            None => {
                self.entries[layer_idx] = Some(KVCacheEntry::new(key.clone(), value.clone()));
                if layer_idx == 0 {
                    self.seq_len = new_len;
                }
                Ok((key.clone(), value.clone()))
            }
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.seq_len = 0;
    }

    /// The next position index (0-based) to fill when appending.
    pub fn cache_position(&self) -> usize {
        self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::KVCache;
    use candle::{DType, Device, Tensor};

    #[test]
    fn test_kv_cache_basic() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut cache = KVCache::with_num_layers(2);

        if !cache.is_empty() {
            anyhow::bail!("expected empty cache");
        }
        if cache.seq_len() != 0 {
            anyhow::bail!("expected seq_len=0");
        }

        let key1 = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;
        let value1 = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;

        let (k, v) = cache.update(0, &key1, &value1)?;
        if k.dims() != [1, 4, 5, 64] {
            anyhow::bail!("unexpected key dims: {:?}", k.dims());
        }
        if v.dims() != [1, 4, 5, 64] {
            anyhow::bail!("unexpected value dims: {:?}", v.dims());
        }
        if cache.seq_len() != 5 {
            anyhow::bail!("expected seq_len=5, got {}", cache.seq_len());
        }

        let key2 = Tensor::zeros((1, 4, 1, 64), DType::F32, &device)?;
        let value2 = Tensor::zeros((1, 4, 1, 64), DType::F32, &device)?;

        let (k, v) = cache.update(0, &key2, &value2)?;
        if k.dims() != [1, 4, 6, 64] {
            anyhow::bail!("unexpected key dims after append: {:?}", k.dims());
        }
        if v.dims() != [1, 4, 6, 64] {
            anyhow::bail!("unexpected value dims after append: {:?}", v.dims());
        }
        if cache.seq_len() != 6 {
            anyhow::bail!("expected seq_len=6, got {}", cache.seq_len());
        }

        Ok(())
    }

    #[test]
    fn test_kv_cache_clear() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut cache = KVCache::with_num_layers(1);

        let key = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;
        let value = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;

        cache.update(0, &key, &value)?;
        if cache.is_empty() {
            anyhow::bail!("expected non-empty cache after update");
        }

        cache.clear();
        if !cache.is_empty() {
            anyhow::bail!("expected empty cache after clear");
        }
        if cache.seq_len() != 0 {
            anyhow::bail!("expected seq_len=0 after clear");
        }

        Ok(())
    }
}
