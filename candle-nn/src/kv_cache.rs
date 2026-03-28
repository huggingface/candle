//! Cache Implementations
//!
use candle::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Cache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    all_data: Option<Tensor>,
    dim: usize,
    current_seq_len: usize,
    grow_by: usize,
    max_seq_len: usize,
}

impl Cache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            grow_by: max_seq_len,
            max_seq_len,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.dim, 0, self.current_seq_len)?),
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad)
        };
        let ad = self.all_data.as_mut().unwrap();
        while self.current_seq_len + seq_len > self.max_seq_len {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.grow_by;
            let next_ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            *ad = Tensor::cat(&[&*ad, &next_ad], self.dim)?;
            self.max_seq_len += self.grow_by;
        }
        ad.slice_set(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = Cache::new(dim, max_seq_len);
        let v = Cache::new(dim, max_seq_len);
        Self { k, v }
    }

    pub fn k_cache(&self) -> &Cache {
        &self.k
    }

    pub fn v_cache(&self) -> &Cache {
        &self.v
    }

    pub fn k_cache_mut(&mut self) -> &mut Cache {
        &mut self.k
    }

    pub fn v_cache_mut(&mut self) -> &mut Cache {
        &mut self.v
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

#[derive(Debug, Clone)]
pub struct RotatingCache {
    all_data: Option<Tensor>,
    dim: usize,
    // `offset` is the current write index in the buffer
    offset: usize,
    // The total size of the sequence seen so far.
    current_seq_len: usize,
    // max_seq_len is the size of the rotating buffer, it is actually allowed for the full
    // sequence to grow past this limit.
    max_seq_len: usize,
}

impl RotatingCache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            offset: 0,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => {
                if self.current_seq_len >= self.max_seq_len {
                    Some(d.clone())
                } else {
                    Some(d.narrow(self.dim, 0, self.current_seq_len)?)
                }
            }
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad)
        };
        let ad = self.all_data.as_mut().unwrap();

        self.current_seq_len += seq_len;
        if seq_len >= self.max_seq_len {
            let to_copy = src
                .narrow(self.dim, seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?;
            ad.slice_set(&to_copy, self.dim, 0)?;
            self.offset = 0;
            // Here we return `src` rather than `ad` so that all the past can be used.
            Ok(src.clone())
        } else {
            let rem_len = self.max_seq_len - self.offset;
            if seq_len <= rem_len {
                ad.slice_set(&src.contiguous()?, self.dim, self.offset)?;
                self.offset = (self.offset + seq_len) % self.max_seq_len;
            } else {
                // We have to make two copies here as we go over the boundary of the cache.
                if rem_len > 0 {
                    let src1 = src.narrow(self.dim, 0, rem_len)?.contiguous()?;
                    ad.slice_set(&src1, self.dim, self.offset)?;
                }
                let src2 = src
                    .narrow(self.dim, rem_len, seq_len - rem_len)?
                    .contiguous()?;
                ad.slice_set(&src2, self.dim, 0)?;
                self.offset = seq_len - rem_len;
            }
            if self.current_seq_len >= self.max_seq_len {
                Ok(ad.clone())
            } else {
                Ok(ad.narrow(self.dim, 0, self.current_seq_len)?)
            }
        }
    }

    fn get_mask_abs(&self, size1: usize, size2: usize, device: &Device) -> Result<Tensor> {
        let context = self.max_seq_len;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    u8::from(size1 + j > size2 + i || size1 + j + context < size2 + i)
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), device)
    }

    fn get_mask_rel(&self, size1: usize, size2: usize, device: &Device) -> Result<Tensor> {
        let context = self.max_seq_len;
        let upd_offset = (self.offset + size1) % self.max_seq_len;
        let mask: Vec<_> = (0..size1)
            .flat_map(|pos_src| {
                // The absolute position of the elements that will get added to the cache.
                let pos_src = self.current_seq_len + pos_src;
                (0..size2).map(move |pos_cache_rel| {
                    // The absolute position of the cache elements after the addition.
                    let pos_cache = self.current_seq_len + size1 + pos_cache_rel - upd_offset;
                    let pos_cache = if pos_cache_rel < upd_offset {
                        pos_cache
                    } else {
                        pos_cache - self.max_seq_len
                    };
                    u8::from(pos_cache > pos_src || pos_cache + context < pos_src)
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        if seq_len <= self.max_seq_len {
            let upd_offset = (self.offset + seq_len) % self.max_seq_len;
            let cache_out_len = (self.current_seq_len + seq_len).min(self.max_seq_len);
            (0..cache_out_len)
                .map(|i| {
                    let pos_cache = self.current_seq_len + seq_len + i - upd_offset;
                    if i < upd_offset {
                        pos_cache
                    } else {
                        pos_cache - self.max_seq_len
                    }
                })
                .collect()
        } else {
            (self.current_seq_len..(self.current_seq_len + seq_len)).collect()
        }
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &Device) -> Result<Option<Tensor>> {
        let mask = if seq_len == 1 {
            None
        } else {
            let mask = if seq_len < self.max_seq_len {
                let cache_out_len = (self.current_seq_len + seq_len).min(self.max_seq_len);
                self.get_mask_rel(seq_len, cache_out_len, device)?
            } else {
                self.get_mask_abs(seq_len, seq_len, device)?
            };
            Some(mask)
        };
        Ok(mask)
    }
}

#[derive(Debug, Clone)]
pub struct RotatingKvCache {
    k: RotatingCache,
    v: RotatingCache,
}

impl RotatingKvCache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, max_seq_len);
        let v = RotatingCache::new(dim, max_seq_len);
        Self { k, v }
    }

    pub fn k_cache(&self) -> &RotatingCache {
        &self.k
    }

    pub fn v_cache(&self) -> &RotatingCache {
        &self.v
    }

    pub fn k_cache_mut(&mut self) -> &mut RotatingCache {
        &mut self.k
    }

    pub fn v_cache_mut(&mut self) -> &mut RotatingCache {
        &mut self.v
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let out_k = self.k.append(k)?;
        let out_v = self.v.append(v)?;
        Ok((out_k, out_v))
    }

    pub fn offset(&self) -> usize {
        self.k.offset()
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &Device) -> Result<Option<Tensor>> {
        self.k.attn_mask(seq_len, device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        self.k.positions(seq_len)
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

#[derive(Debug, Clone)]
pub struct IndicesAndMask {
    indices: Tensor,
    mask: Tensor,
}

impl IndicesAndMask {
    pub fn mask(&self) -> &Tensor {
        &self.mask
    }
}

#[derive(Debug, Clone)]
pub struct ScatteredKvCache {
    k: Tensor,
    v: Tensor,
    context: usize,
}

impl ScatteredKvCache {
    pub fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        iam: &IndicesAndMask,
    ) -> Result<(Tensor, Tensor)> {
        if self.context <= k.dim(2)? {
            return Ok((k.clone(), v.clone()));
        }
        let indices = iam.indices.unsqueeze(2)?.unsqueeze(1)?;
        let indices = indices.broadcast_as(k.shape())?.contiguous()?;
        self.k.scatter_set(&indices, k, 2)?;
        self.v.scatter_set(&indices, v, 2)?;
        Ok((self.k.clone(), self.v.clone()))
    }

    pub fn k(&self) -> &Tensor {
        &self.k
    }

    pub fn v(&self) -> &Tensor {
        &self.v
    }
}

#[derive(Debug, Clone)]
pub struct ScatteredCacheBuilder {
    context: usize,
    // The current position in the stream, this can be larger than context.
    positions: Vec<usize>,
    // The index where the next element will be stored.
    indices: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl ScatteredCacheBuilder {
    pub fn new(batch_size: usize, context: usize, dtype: DType, device: &Device) -> Result<Self> {
        let positions = vec![0; batch_size];
        let indices = vec![0; batch_size];
        Ok(Self {
            positions,
            indices,
            context,
            dtype,
            device: device.clone(),
        })
    }

    pub fn make_cache(&self, num_heads: usize, head_dim: usize) -> Result<ScatteredKvCache> {
        let batch_size = self.batch_size();
        let shape = (batch_size, num_heads, self.context, head_dim);
        let k = Tensor::zeros(shape, self.dtype, self.device())?;
        let v = Tensor::zeros(shape, self.dtype, self.device())?;
        Ok(ScatteredKvCache {
            k,
            v,
            context: self.context,
        })
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn reset(&mut self) {
        self.positions.fill(0);
        self.indices.fill(0);
    }

    pub fn batch_size(&self) -> usize {
        self.positions.len()
    }

    pub fn reset_batch_index(&mut self, batch_index: usize) {
        self.positions[batch_index] = 0;
        self.indices[batch_index] = 0;
    }

    #[allow(clippy::needless_range_loop)]
    pub fn indices_and_mask(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask> {
        // mask shape is (b, h, t, k)
        let context = self.context;
        if self.context <= seq_len {
            return self.indices_and_mask_abs(seq_len, batch_mask);
        }
        let mut attention_masks = Vec::with_capacity(self.batch_size());
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let masks: Vec<Vec<f32>> = vec![vec![0.0; context]; seq_len];
                let indices = vec![self.indices[batch_i] as u32; seq_len];
                attention_masks.push(masks);
                cache_indices.push(indices);
            } else {
                let start_index = self.indices[batch_i];
                let start_pos = self.positions[batch_i];
                let mut masks: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
                let mut indices = Vec::with_capacity(seq_len);
                let mut all_pos = vec![usize::MAX; context];
                if start_pos < context {
                    for i in 0..start_pos {
                        all_pos[i] = i;
                    }
                } else {
                    let offset = start_pos - start_index;
                    for i in 0..context {
                        all_pos[i] = if i < start_index {
                            i + offset
                        } else {
                            i + offset - context
                        };
                    }
                }
                for seq_i in 0..seq_len {
                    let index = self.indices[batch_i];
                    all_pos[index] = seq_i + start_pos;
                    indices.push(index as u32);
                    self.indices[batch_i] += 1;
                    self.positions[batch_i] += 1;
                    if self.indices[batch_i] >= self.context {
                        self.indices[batch_i] = 0;
                    }
                }

                for seq_i in 0..seq_len {
                    let my_pos = seq_i + start_pos;
                    let mask = all_pos
                        .iter()
                        .map(|&pos| {
                            if pos <= my_pos {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f32>>();
                    masks.push(mask);
                }

                attention_masks.push(masks);
                cache_indices.push(indices);
            }
        }
        // Flattening the attention mask then using Tensor::from_vec rather using Tensor::new ends
        // up being almost 10x faster with candle 0.9.0. This has been fixed in candle 0.9.1.
        let attention_masks = attention_masks
            .into_iter()
            .flat_map(|m| m.into_iter().flatten())
            .collect::<Vec<f32>>();
        let mask = Tensor::from_vec(attention_masks, ((), 1, seq_len, context), self.device())?
            .to_dtype(self.dtype)?;
        let indices = Tensor::new(cache_indices, self.device())?;
        Ok(IndicesAndMask { indices, mask })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(clippy::needless_range_loop)]
    fn indices_and_mask_abs(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask> {
        let mask = self.get_mask_abs(seq_len, seq_len)?;
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let indices = vec![self.indices[batch_i] as u32; seq_len];
                cache_indices.push(indices);
            } else {
                let mut indices = Vec::with_capacity(seq_len);
                for _ in 0..seq_len {
                    let index = self.indices[batch_i];
                    indices.push(index as u32);
                    self.indices[batch_i] += 1;
                    self.positions[batch_i] += 1;
                    if self.indices[batch_i] >= self.context {
                        self.indices[batch_i] = 0;
                    }
                }
                cache_indices.push(indices);
            }
        }
        let indices = Tensor::new(cache_indices, self.device())?;
        Ok(IndicesAndMask { indices, mask })
    }

    fn get_mask_abs(&self, size1: usize, size2: usize) -> Result<Tensor> {
        let context = self.context;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    if size1 + j > size2 + i || size1 + j + context < size2 + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), self.device())
    }
}

/// KV-Cache using concatenation for append operations
///
/// This implementation uses `Tensor::cat` instead of `slice_set` for updates,
/// providing significant GPU performance improvements for autoregressive generation.
///
/// # When to Use
///
/// **Recommended for:**
/// - GPU inference (CUDA, Metal)
/// - Autoregressive generation (token-by-token decoding)
///
/// **Use `KvCache` instead for:**
/// - CPU-only inference
/// - When you need fixed memory allocation upfront
///
/// # Example
///
/// ```ignore
/// use candle_nn::kv_cache::ConcatKvCache;
///
/// let mut cache = ConcatKvCache::new(2); // dim=2 for sequence dimension
///
/// // First token (prefill)
/// let k1 = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let v1 = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let (k, v) = cache.append(&k1, &v1)?;
///
/// // Subsequent tokens (decode)
/// let k_new = Tensor::randn(0f32, 1., (1, 8, 1, 64), &device)?;
/// let v_new = Tensor::randn(0f32, 1., (1, 8, 1, 64), &device)?;
/// let (k, v) = cache.append(&k_new, &v_new)?;
/// ```
#[derive(Debug, Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl ConcatKvCache {
    /// Create a new empty concatenation-based KV-cache
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    ///   - For attention with shape `[batch, heads, seq, head_dim]`, use `dim=2`
    ///   - For attention with shape `[batch, seq, heads, head_dim]`, use `dim=1`
    ///
    /// # Example
    /// ```ignore
    /// // For standard transformer attention: [B, H, S, D]
    /// let cache = ConcatKvCache::new(2);
    /// ```
    pub fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,
        }
    }

    /// Get current sequence length in the cache
    ///
    /// Returns 0 if the cache is empty.
    pub fn current_seq_len(&self) -> usize {
        self.k
            .as_ref()
            .and_then(|k| k.dims().get(self.dim).copied())
            .unwrap_or(0)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    /// Get the concatenation dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append key and value tensors to the cache
    ///
    /// This is the core operation that uses optimized concatenation kernels.
    ///
    /// # Arguments
    /// * `k` - Key tensor to append (shape: [..., seq_len, ...])
    /// * `v` - Value tensor to append (shape: [..., seq_len, ...])
    ///
    /// # Returns
    /// Tuple of `(full_k, full_v)` containing all cached keys and values,
    /// including the newly appended data.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Ensure inputs are contiguous for optimal concatenation performance
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // Update K cache using concatenation.
        // Detach inputs and outputs to prevent BackpropOp chain accumulation:
        // Tensor::cat stores both operands in a BackpropOp::new2 node. If either
        // input has track_op()==true (which KV projections always do — they derive
        // from model weights), the old cache buffer is retained by the op graph
        // even after the new buffer is written. Over N tokens this holds N
        // full-buffer copies in memory regardless of cache strategy.
        // Detaching is safe: KV caches are inference-only, no gradient flows
        // through them.
        self.k = Some(match &self.k {
            None => k.detach(),
            Some(k_cache) => {
                Tensor::cat(&[&k_cache.detach(), &k.detach()], self.dim)?.detach()
            }
        });

        // Update V cache using concatenation
        self.v = Some(match &self.v {
            None => v.detach(),
            Some(v_cache) => Tensor::cat(&[&v_cache.detach(), &v.detach()], self.dim)?.detach(),
        });

        Ok((
            self.k.as_ref().unwrap().clone(),
            self.v.as_ref().unwrap().clone(),
        ))
    }

    /// Reset the cache (clear all stored keys and values)
    ///
    /// After calling this, `is_empty()` will return `true` and
    /// `current_seq_len()` will return 0.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Get reference to current K cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn k(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    /// Get reference to current V cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }

    /// Get mutable reference to K cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn k_mut(&mut self) -> Option<&mut Tensor> {
        self.k.as_mut()
    }

    /// Get mutable reference to V cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn v_mut(&mut self) -> Option<&mut Tensor> {
        self.v.as_mut()
    }

    /// Get owned K and V tensors, consuming the cache
    ///
    /// Returns `None` if the cache is empty.
    pub fn into_inner(self) -> Option<(Tensor, Tensor)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}

/// Default maximum sequence length for [`PreallocKvCache`].
///
/// 512 tokens covers most chat turns. Increase for long-context workloads.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Fixed-capacity KV cache that pre-allocates buffers once and writes via
/// `slice_scatter`. Produces **zero net allocation growth** over arbitrarily
/// many tokens once constructed.
///
/// # Memory model
///
/// | Event       | Allocator activity                                       |
/// |-------------|----------------------------------------------------------|
/// | `new()`     | 2× `(1, H, max_seq, D)` alloc — happens once at init    |
/// | `append()`  | `slice_scatter` same-size alloc + free (uniform size)    |
/// | `reset()`   | counter = 0 — zero dealloc, O(1)                        |
///
/// Because every `slice_scatter` produces the same allocation size, the CUDA
/// and Metal allocators can always satisfy requests from previously freed blocks.
/// This eliminates the fragmentation cascade that causes OOM after ~37 tokens
/// with [`ConcatKvCache`] on unified-memory platforms (CUDA SM121, Apple Silicon).
///
/// # BackpropOp note
///
/// `slice_scatter` stores both operands in a `BackpropOp::new2` node. Without
/// `.detach()`, every append chains the prior buffer in the op graph, leaking
/// O(N²) memory over N tokens. All inputs and outputs are detached here because
/// KV caches are inference-only — no gradient should flow through them.
///
/// # Usage
///
/// ```rust,no_run
/// use candle::{DType, Device};
/// use candle_nn::kv_cache::PreallocKvCache;
///
/// let device = Device::Cpu;
/// let mut cache = PreallocKvCache::new(8, 64, 512, DType::F32, &device).unwrap();
///
/// // Autoregressive generation loop:
/// // for _token in 0..100 {
/// //     let new_k = /* ... */;
/// //     let new_v = /* ... */;
/// //     let (full_k, full_v) = cache.append(&new_k, &new_v).unwrap();
/// //     // full_k/full_v are the accumulated sequence up to this token.
/// // }
///
/// // Between conversations — O(1), no deallocation:
/// cache.reset();
/// ```
///
/// # See also
///
/// - [`ConcatKvCache`] — the growing-allocation baseline
/// - Candle issues: #2950, #2271, #1599, #3197
/// - Candle PRs: #3188 (proposed `KvCache` trait), #3143 (`ConcatKvCache`)
#[derive(Debug, Clone)]
pub struct PreallocKvCache {
    k_buf: Tensor,
    v_buf: Tensor,
    current_pos: usize,
    max_seq_len: usize,
}

impl PreallocKvCache {
    /// Allocate KV buffers for one attention layer.
    ///
    /// # Arguments
    ///
    /// - `num_kv_heads` — number of KV attention heads (may be fewer than
    ///   query heads in grouped-query attention)
    /// - `head_dim` — per-head feature dimension
    /// - `max_seq_len` — maximum tokens this cache will hold. Use
    ///   [`DEFAULT_MAX_SEQ_LEN`] (512) for typical chat workloads.
    /// - `dtype` / `device` — must match the model's weight dtype and device
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (1, num_kv_heads, max_seq_len, head_dim);
        let k_buf = Tensor::zeros(shape, dtype, device)?;
        let v_buf = Tensor::zeros(shape, dtype, device)?;
        Ok(Self {
            k_buf,
            v_buf,
            current_pos: 0,
            max_seq_len,
        })
    }

    /// Append new K/V tensors and return the accumulated sequence so far.
    ///
    /// `new_k` and `new_v` must have shape `(batch, num_kv_heads, seq_len, head_dim)`.
    /// Returns `(full_k, full_v)` of shape `(batch, num_kv_heads, current_seq_len, head_dim)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `current_seq_len + seq_len > max_seq_len`. Call
    /// [`reset`](Self::reset) between conversations to reuse the buffer.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = new_k.dim(2)?;
        let end_pos = self.current_pos + seq_len;
        if end_pos > self.max_seq_len {
            candle::bail!(
                "PreallocKvCache: sequence length {end_pos} exceeds max_seq_len {}. \
                 Call reset() between conversations or increase max_seq_len.",
                self.max_seq_len
            );
        }
        // Detach inputs and outputs to prevent BackpropOp chain accumulation.
        // See struct-level documentation for details.
        self.k_buf = self
            .k_buf
            .slice_scatter(&new_k.detach(), 2, self.current_pos)?
            .detach();
        self.v_buf = self
            .v_buf
            .slice_scatter(&new_v.detach(), 2, self.current_pos)?
            .detach();
        self.current_pos = end_pos;
        // Return a view of the written prefix only. Stale data beyond
        // current_pos is invisible to callers and will be overwritten on reset.
        let k_active = self.k_buf.narrow(2, 0, self.current_pos)?.detach();
        let v_active = self.v_buf.narrow(2, 0, self.current_pos)?.detach();
        Ok((k_active, v_active))
    }

    /// Reset the cache to empty without deallocating the underlying buffers.
    ///
    /// This is O(1). Stale data beyond the new `current_seq_len` of 0 is
    /// invisible to callers via `narrow` and will be overwritten on next use.
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    /// Number of tokens currently in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.current_pos
    }

    /// Maximum tokens this cache can hold before returning an error.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::IndexOp;

    #[test]
    fn test_scattered_kv_cache() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ScatteredCacheBuilder::new(2, 5, DType::F32, &device)?;
        let inf = f32::INFINITY;

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[0], [0]]);
        assert_eq!(
            mask,
            [[[0.0, -inf, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[1], [0]]);
        assert_eq!(
            mask,
            [[[0.0, 0.0, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        let iam = cache.indices_and_mask(3, &[false, true])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[2, 2, 2], [0, 1, 2]]);
        assert_eq!(
            mask,
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
                [
                    [0.0, -inf, -inf, -inf, -inf],
                    [0.0, 0.0, -inf, -inf, -inf],
                    [0.0, 0.0, 0.0, -inf, -inf]
                ]
            ]
        );

        let iam = cache.indices_and_mask(3, &[true, true])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[2, 3, 4], [3, 4, 0]]);
        assert_eq!(
            mask,
            [
                [
                    [0.0, 0.0, 0.0, -inf, -inf],
                    [0.0, 0.0, 0.0, 0.0, -inf],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
                [
                    [-inf, 0.0, 0.0, 0.0, -inf],
                    [-inf, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            ]
        );

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[0], [1]]);
        assert_eq!(
            mask,
            [[[0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        let iam = cache.indices_and_mask(2, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[1, 2], [1, 1]]);
        assert_eq!(
            mask,
            [
                [[0.0, 0.0, -inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
            ]
        );

        Ok(())
    }

    #[test]
    fn test_concat_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(2);

        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);

        // First append
        let k1 = Tensor::zeros((1, 8, 3, 64), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 8, 3, 64), DType::F32, &device)?;
        let (k, v) = cache.append(&k1, &v1)?;

        assert_eq!(k.dims(), &[1, 8, 3, 64]);
        assert_eq!(v.dims(), &[1, 8, 3, 64]);
        assert_eq!(cache.current_seq_len(), 3);
        assert!(!cache.is_empty());

        // Second append
        let k2 = Tensor::zeros((1, 8, 2, 64), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 8, 2, 64), DType::F32, &device)?;
        let (k, v) = cache.append(&k2, &v2)?;

        assert_eq!(k.dims(), &[1, 8, 5, 64]); // 3 + 2
        assert_eq!(v.dims(), &[1, 8, 5, 64]);
        assert_eq!(cache.current_seq_len(), 5);

        Ok(())
    }

    #[test]
    fn test_concat_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(2);

        let k = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        let v = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        cache.append(&k, &v)?;

        assert_eq!(cache.current_seq_len(), 10);

        cache.reset();

        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.k().is_none());
        assert!(cache.v().is_none());

        Ok(())
    }

    #[test]
    fn test_concat_cache_multiple_appends() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(2);

        // Simulate autoregressive generation
        let k_prefill = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        let v_prefill = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        cache.append(&k_prefill, &v_prefill)?;

        assert_eq!(cache.current_seq_len(), 10);

        // Decode phase: append one token at a time
        for i in 1..=5 {
            let k_token = Tensor::zeros((1, 8, 1, 64), DType::F32, &device)?;
            let v_token = Tensor::zeros((1, 8, 1, 64), DType::F32, &device)?;
            let (k, v) = cache.append(&k_token, &v_token)?;
            assert_eq!(k.dims()[2], 10 + i);
            assert_eq!(v.dims()[2], 10 + i);
        }

        assert_eq!(cache.current_seq_len(), 15);

        Ok(())
    }

    #[test]
    fn test_concat_cache_different_dim() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(1); // Concatenate on dim 1 instead of 2

        let k1 = Tensor::zeros((1, 3, 8, 64), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 3, 8, 64), DType::F32, &device)?;
        let (k, _v) = cache.append(&k1, &v1)?;

        assert_eq!(k.dims(), &[1, 3, 8, 64]);

        let k2 = Tensor::zeros((1, 2, 8, 64), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 2, 8, 64), DType::F32, &device)?;
        let (k, _v) = cache.append(&k2, &v2)?;

        assert_eq!(k.dims(), &[1, 5, 8, 64]); // Concatenated on dim 1
        assert_eq!(cache.current_seq_len(), 5);

        Ok(())
    }

    // ── PreallocKvCache tests ───────────────────────────────────────────

    #[test]
    fn test_prealloc_new() -> Result<()> {
        let c = PreallocKvCache::new(4, 64, 128, DType::F32, &Device::Cpu)?;
        assert_eq!(c.current_seq_len(), 0);
        assert_eq!(c.max_seq_len(), 128);
        Ok(())
    }

    #[test]
    fn test_prealloc_append_accumulates() -> Result<()> {
        let device = Device::Cpu;
        let mut c = PreallocKvCache::new(2, 8, 64, DType::F32, &device)?;
        for i in 1..=10usize {
            let k = Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?;
            let v = Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?;
            let (ok, ov) = c.append(&k, &v)?;
            assert_eq!(c.current_seq_len(), i);
            assert_eq!(ok.dim(2)?, i);
            assert_eq!(ov.dim(2)?, i);
        }
        Ok(())
    }

    #[test]
    fn test_prealloc_data_integrity() -> Result<()> {
        let device = Device::Cpu;
        let mut c = PreallocKvCache::new(1, 2, 16, DType::F32, &device)?;
        let k0 = Tensor::zeros((1, 1, 1, 2), DType::F32, &device)?;
        let k1 = Tensor::ones((1, 1, 1, 2), DType::F32, &device)?;
        c.append(&k0, &k0)?;
        let (out_k, _) = c.append(&k1, &k1)?;
        let vals: Vec<f32> = out_k.flatten_all()?.to_vec1()?;
        assert_eq!(vals, vec![0.0, 0.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_prealloc_reset_reusable() -> Result<()> {
        let device = Device::Cpu;
        let mut c = PreallocKvCache::new(2, 4, 32, DType::F32, &device)?;
        for _ in 0..16 {
            c.append(
                &Tensor::zeros((1, 2, 1, 4), DType::F32, &device)?,
                &Tensor::zeros((1, 2, 1, 4), DType::F32, &device)?,
            )?;
        }
        assert_eq!(c.current_seq_len(), 16);
        c.reset();
        assert_eq!(c.current_seq_len(), 0);
        let (ok, _) = c.append(
            &Tensor::ones((1, 2, 1, 4), DType::F32, &device)?,
            &Tensor::ones((1, 2, 1, 4), DType::F32, &device)?,
        )?;
        assert_eq!(ok.dim(2)?, 1);
        Ok(())
    }

    #[test]
    fn test_prealloc_overflow_error() -> Result<()> {
        let device = Device::Cpu;
        let mut c = PreallocKvCache::new(1, 4, 3, DType::F32, &device)?;
        c.append(
            &Tensor::zeros((1, 1, 2, 4), DType::F32, &device)?,
            &Tensor::zeros((1, 1, 2, 4), DType::F32, &device)?,
        )?;
        let result = c.append(
            &Tensor::zeros((1, 1, 2, 4), DType::F32, &device)?,
            &Tensor::zeros((1, 1, 2, 4), DType::F32, &device)?,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_seq_len"));
        Ok(())
    }

    #[test]
    fn test_prealloc_outputs_detached() -> Result<()> {
        // BackpropOp regression: outputs must not track ops.
        let device = Device::Cpu;
        let var_map = crate::VarMap::new();
        let vb = crate::VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let k_var = vb.get((1, 2, 1, 4), "k")?;
        assert!(k_var.track_op(), "test setup: var must track ops");
        let v_var = vb.get((1, 2, 1, 4), "v")?;

        let mut c = PreallocKvCache::new(2, 4, 64, DType::F32, &device)?;
        for _ in 0..10 {
            let (ok, ov) = c.append(&k_var, &v_var)?;
            assert!(!ok.track_op(), "output K must not track ops");
            assert!(!ov.track_op(), "output V must not track ops");
        }
        Ok(())
    }
}
