//! KV cache for autoregressive generation.

use candle::{Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;

#[derive(Debug, Clone, Default)]
pub struct KVCache {
    entries: Vec<Option<ConcatKvCache>>,
    seq_len: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
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
            Some(cache) => {
                let result = cache.append(key, value)?;
                if layer_idx == 0 {
                    self.seq_len = self.seq_len.saturating_add(new_len);
                }
                Ok(result)
            }
            None => {
                let mut cache = ConcatKvCache::new(2);
                let result = cache.append(key, value)?;
                self.entries[layer_idx] = Some(cache);
                if layer_idx == 0 {
                    self.seq_len = new_len;
                }
                Ok(result)
            }
        }
    }
}
