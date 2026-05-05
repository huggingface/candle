//! KV cache for autoregressive generation.

use candle::{Result, Tensor};

#[derive(Debug, Clone, Default)]
pub struct KVCache {
    entries: Vec<Option<(Tensor, Tensor)>>,
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
            Some((k, v)) => {
                *k = Tensor::cat(&[&*k, key], 2)?;
                *v = Tensor::cat(&[&*v, value], 2)?;
                if layer_idx == 0 {
                    self.seq_len = self.seq_len.saturating_add(new_len);
                }
                Ok((k.clone(), v.clone()))
            }
            None => {
                self.entries[layer_idx] = Some((key.clone(), value.clone()));
                if layer_idx == 0 {
                    self.seq_len = new_len;
                }
                Ok((key.clone(), value.clone()))
            }
        }
    }
}
