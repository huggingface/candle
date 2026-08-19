//! Paged KV Cache for Candle — Phase 1.
//!
//! Uses per-layer allocators and block tables. Each layer has its own flat pool:
//! [num_blocks * block_size, num_kv_heads, head_dim].
//!
//! Optimizations:
//!   1. Cached GPU indices tensor (rebuilt only on block boundary crossing).
//!   2. Large block_size (64): fewer blocks, better coalescing, less H2D.
//!   3. Fast-path: single-block sequences use narrow() (zero-copy view).

use candle::{DType, Device, Result, Tensor};
use std::sync::Mutex;

use super::block::{BlockAllocator, BlockTable};

#[derive(Debug, Clone)]
pub struct PagedCacheConfig {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub device: Device,
}

#[derive(Debug, Clone)]
struct LayerPool {
    /// Flattened: [num_blocks * block_size, num_kv_heads, head_dim]
    k: Tensor,
    v: Tensor,
}

#[derive(Debug, Clone)]
struct CachedIndices {
    tensor: Tensor,
    seq_len: usize,
}

#[derive(Debug)]
pub struct PagedKvCache {
    pools: Vec<LayerPool>,
    allocators: Vec<Mutex<BlockAllocator>>,
    /// tables[layer][seq_id]
    tables: Vec<Vec<BlockTable>>,
    /// cached_indices[layer][seq_id]
    cached_indices: Vec<Vec<Option<CachedIndices>>>,
    config: PagedCacheConfig,
}

impl PagedKvCache {
    pub fn new(config: PagedCacheConfig) -> Result<Self> {
        let flat_len = config.num_blocks * config.block_size;
        let mut pools = Vec::with_capacity(config.num_layers);
        let mut allocators = Vec::with_capacity(config.num_layers);
        let mut tables = Vec::with_capacity(config.num_layers);
        let mut cached_indices = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            let k = Tensor::zeros(
                (flat_len, config.num_kv_heads, config.head_dim),
                config.dtype,
                &config.device,
            )?;
            let v = Tensor::zeros(
                (flat_len, config.num_kv_heads, config.head_dim),
                config.dtype,
                &config.device,
            )?;
            pools.push(LayerPool { k, v });
            allocators.push(Mutex::new(BlockAllocator::new(config.num_blocks)));
            tables.push(Vec::new());
            cached_indices.push(Vec::new());
        }

        Ok(Self {
            pools,
            allocators,
            tables,
            cached_indices,
            config,
        })
    }

    /// Add a new sequence. Returns its sequence ID.
    pub fn add_sequence(&mut self) -> usize {
        let seq_id = self.tables[0].len();
        for layer in 0..self.config.num_layers {
            self.tables[layer].push(BlockTable::new(self.config.block_size));
            self.cached_indices[layer].push(None);
        }
        seq_id
    }

    /// COW fork for beam search. O(num_layers * num_blocks) ref bumps, no tensor copies.
    pub fn fork_sequence(&mut self, seq_id: usize) -> Result<usize> {
        let new_id = self.tables[0].len();
        for layer in 0..self.config.num_layers {
            let mut alloc = self.allocators[layer].lock().unwrap();
            let new_table = self.tables[layer][seq_id].fork(&mut alloc);
            drop(alloc);
            self.tables[layer].push(new_table);
            self.cached_indices[layer].push(None);
        }
        Ok(new_id)
    }

    /// Reset a sequence (drop its blocks back to the free lists).
    pub fn reset_sequence(&mut self, seq_id: usize) {
        for layer in 0..self.config.num_layers {
            let mut alloc = self.allocators[layer].lock().unwrap();
            let table = &mut self.tables[layer][seq_id];
            for block in table.block_ids() {
                alloc.free(*block);
            }
            *table = BlockTable::new(self.config.block_size);
            self.cached_indices[layer][seq_id] = None;
        }
    }

    /// Append a single token's K and V projections to the cache.
    ///
    /// `k_new`, `v_new`: shape [batch=1, num_kv_heads, 1, head_dim]
    pub fn append_kv(
        &mut self,
        seq_id: usize,
        layer: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        let mut alloc = self.allocators[layer].lock().unwrap();
        let table = &mut self.tables[layer][seq_id];
        let (block_id, offset) = table
            .append(&mut alloc)
            .ok_or_else(|| candle::Error::Msg("Paged KV cache OOM".to_string()))?;
        drop(alloc);

        let flat_index = block_id.0 * self.config.block_size + offset;

        // k_new: [1, num_kv_heads, 1, head_dim]
        // -> squeeze batch and seq dims -> [num_kv_heads, head_dim]
        // -> unsqueeze to [1, num_kv_heads, head_dim] for slice_set
        let k_src = k_new.squeeze(0)?.squeeze(1)?.unsqueeze(0)?;
        let v_src = v_new.squeeze(0)?.squeeze(1)?.unsqueeze(0)?;

        let pool = &mut self.pools[layer];
        pool.k.slice_set(&k_src, 0, flat_index)?;
        pool.v.slice_set(&v_src, 0, flat_index)?;

        // Invalidate cached GPU indices only when we cross a block boundary.
        if offset == 0 && table.seq_len() > 1 {
            self.cached_indices[layer][seq_id] = None;
        }

        Ok(())
    }

    /// Gather the full K and V tensors for a sequence and layer.
    /// Returns: [seq_len, num_kv_heads, head_dim]
    pub fn gather_kv(&mut self, seq_id: usize, layer: usize) -> Result<(Tensor, Tensor)> {
        let table = &self.tables[layer][seq_id];
        let seq_len = table.seq_len();

        if seq_len == 0 {
            let shape = (0usize, self.config.num_kv_heads, self.config.head_dim);
            let k = Tensor::zeros(shape, self.config.dtype, &self.config.device)?;
            let v = Tensor::zeros(shape, self.config.dtype, &self.config.device)?;
            return Ok((k, v));
        }

        // OPTIMIZATION 3: Fast path for single-block sequences.
        // narrow() returns a view, zero copy.
        if table.num_blocks() == 1 {
            let block_id = table.block_ids()[0].0;
            let start = block_id * self.config.block_size;
            let pool = &self.pools[layer];
            let k = pool.k.narrow(0, start, seq_len)?;
            let v = pool.v.narrow(0, start, seq_len)?;
            return Ok((k, v));
        }

        // OPTIMIZATION 1: Reuse cached indices if seq_len hasn't grown.
        let indices = self.get_or_build_indices(seq_id, layer)?;

        let pool = &self.pools[layer];
        let k = pool.k.index_select(&indices, 0)?;
        let v = pool.v.index_select(&indices, 0)?;

        Ok((k, v))
    }

    fn get_or_build_indices(&mut self, seq_id: usize, layer: usize) -> Result<Tensor> {
        let table = &self.tables[layer][seq_id];
        let seq_len = table.seq_len();

        if let Some(ref cached) = self.cached_indices[layer][seq_id] {
            if cached.seq_len == seq_len {
                return Ok(cached.tensor.clone());
            }
        }

        let mut flat_indices = Vec::with_capacity(seq_len);
        for (block_idx, block_id) in table.block_ids().iter().enumerate() {
            let is_last = block_idx == table.num_blocks() - 1;
            let tokens_here = if is_last {
                seq_len - block_idx * self.config.block_size
            } else {
                self.config.block_size
            };
            let base = (block_id.0 * self.config.block_size) as i64;
            for off in 0..tokens_here {
                flat_indices.push(base + off as i64);
            }
        }

        let tensor = Tensor::new(flat_indices, &self.config.device)?;
        self.cached_indices[layer][seq_id] = Some(CachedIndices {
            tensor: tensor.clone(),
            seq_len,
        });
        Ok(tensor)
    }

    pub fn current_seq_len(&self, seq_id: usize) -> usize {
        self.tables[0][seq_id].seq_len()
    }

    pub fn num_blocks(&self, seq_id: usize, layer: usize) -> usize {
        self.tables[layer][seq_id].num_blocks()
    }

    pub fn num_free_blocks(&self, layer: usize) -> usize {
        self.allocators[layer].lock().unwrap().num_free()
    }
}