//! Cache Implementations
//!
use crate::turboquant::{TurboQuant, TurboQuantized};
use crate::turboquant_mse::TurboMseQuantized;
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
        // Update K cache using concatenation
        self.k = Some(match &self.k {
            None => k.clone(),
            Some(k_cache) => {
                // Concatenate along the sequence dimension
                // GPU kernel for cat is highly optimized:
                // - Fused allocation + copy
                // - Coalesced memory access
                // - Single kernel launch
                Tensor::cat(&[k_cache, &k], self.dim)?
            }
        });

        // Update V cache using concatenation
        self.v = Some(match &self.v {
            None => v.clone(),
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?,
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

/// KV cache that stores key/value embeddings using TurboQuant compression.
///
/// This is the quantized equivalent of [`ConcatKvCache`]. Each K/V vector is
/// stored as U8 centroid indices + F32 norm, providing ~3.7× memory reduction
/// vs F32 (at head_dim=128, 4-bit quantization). Uses `Tensor::cat` for updates,
/// matching `ConcatKvCache`'s GPU-friendly concatenation strategy.
///
/// This is a **lossy** cache — [`append`](Self::append) returns dequantized
/// approximations of the original vectors. The type system makes this explicit:
/// models must opt in by constructing a `QuantizedKvCache` instead of a plain cache.
///
/// # When to Use
///
/// **Recommended for:**
/// - Long-context inference where KV memory is the bottleneck
/// - GPU inference with `head_dim` ≥ 32
///
/// **Use [`ConcatKvCache`] instead for:**
/// - When lossless K/V storage is required
/// - Short sequences where memory is not a concern
///
/// **Use [`QuantizedPreAllocKvCache`] instead for:**
/// - CPU inference or very long sequences (avoids repeated concatenation)
///
/// # Example
///
/// ```ignore
/// use candle::{DType, Tensor};
/// use candle_nn::kv_cache::QuantizedKvCache;
///
/// // Create a 4-bit quantized cache for attention with [B, H, S, D] layout
/// let mut cache = QuantizedKvCache::new(2, 64, 4, DType::F32, &device)?;
///
/// // First token (prefill)
/// let k1 = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let v1 = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let (k, v) = cache.append(&k1, &v1)?;
///
/// // Subsequent tokens (decode) — same API as ConcatKvCache
/// let k_new = Tensor::randn(0f32, 1., (1, 8, 1, 64), &device)?;
/// let v_new = Tensor::randn(0f32, 1., (1, 8, 1, 64), &device)?;
/// let (k, v) = cache.append(&k_new, &v_new)?;
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    k_quantizer: TurboQuant,
    v_quantizer: TurboQuant,
    k_indices: Option<Tensor>,
    k_norms: Option<Tensor>,
    k_qjl_signs: Option<Tensor>,
    k_residual_norms: Option<Tensor>,
    v_indices: Option<Tensor>,
    v_norms: Option<Tensor>,
    v_qjl_signs: Option<Tensor>,
    v_residual_norms: Option<Tensor>,
    k_dequantized: Option<Tensor>,
    v_dequantized: Option<Tensor>,
    dim: usize,
    dtype: DType,
}

impl QuantizedKvCache {
    /// Create a new TurboQuant-compressed KV cache.
    ///
    /// Allocates two independent [`TurboQuant`](crate::turboquant::TurboQuant)
    /// quantizers (one for keys, one for values), each with their own random
    /// rotation matrix and QJL projection. The cache starts empty.
    ///
    /// Uses the full TurboQuant algorithm (MSE + QJL correction) for unbiased
    /// inner product estimation, which improves attention score accuracy.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    ///   - For attention with shape `[batch, heads, seq, head_dim]`, use `dim=2`
    ///   - For attention with shape `[batch, seq, heads, head_dim]`, use `dim=1`
    /// * `head_dim` - Attention head dimension (d). Should be ≥ 32 for good quantization.
    /// * `bit_width` - Quantization bits per coordinate (b). Typically 2–4.
    ///   Higher values give better accuracy at the cost of more memory.
    /// * `dtype` - Data type for dequantized output tensors (e.g., `DType::F32`).
    /// * `device` - Device for tensor allocation.
    ///
    /// # Example
    /// ```ignore
    /// // For standard transformer attention: [B, H, S, D] with 4-bit quantization
    /// let cache = QuantizedKvCache::new(2, 128, 4, DType::F32, &device)?;
    /// ```
    pub fn new(
        dim: usize,
        head_dim: usize,
        bit_width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        Ok(Self {
            k_quantizer: TurboQuant::new(head_dim, bit_width, internal_dtype, device)?,
            v_quantizer: TurboQuant::new(head_dim, bit_width, internal_dtype, device)?,
            k_indices: None,
            k_norms: None,
            k_qjl_signs: None,
            k_residual_norms: None,
            v_indices: None,
            v_norms: None,
            v_qjl_signs: None,
            v_residual_norms: None,
            k_dequantized: None,
            v_dequantized: None,
            dim,
            dtype,
        })
    }

    /// Append key/value tensors to the cache.
    ///
    /// Quantizes the input tensors via TurboQuant (MSE + QJL), concatenates the
    /// quantized representations with existing cache along `dim`, then dequantizes
    /// only the newly appended tokens and concatenates with the cached dequantized
    /// output. This reduces total dequantization work from O(S²) to O(S) across
    /// an autoregressive generation of length S.
    ///
    /// # Arguments
    /// * `k` - Key tensor, shape `[batch, heads, seq_len, head_dim]`
    /// * `v` - Value tensor, shape `[batch, heads, seq_len, head_dim]`
    ///
    /// # Returns
    /// Tuple of `(full_k, full_v)` containing dequantized approximations of all
    /// cached keys and values, including the newly appended data.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if k.dims().len() != 4 {
            candle::bail!(
                "QuantizedKvCache expects 4D tensors [B,H,S,D], got {}D",
                k.dims().len()
            );
        }

        let kq = turbo_quantize_4d(&self.k_quantizer, &k)?;
        let vq = turbo_quantize_4d(&self.v_quantizer, &v)?;

        // Dequantize only the newly appended tokens
        let new_k_deq = turbo_dequantize_4d(
            &self.k_quantizer,
            &kq.indices,
            &kq.norms,
            &kq.qjl_signs,
            &kq.residual_norms,
            self.dtype,
        )?;
        let new_v_deq = turbo_dequantize_4d(
            &self.v_quantizer,
            &vq.indices,
            &vq.norms,
            &vq.qjl_signs,
            &vq.residual_norms,
            self.dtype,
        )?;

        // Concatenate quantized representations along sequence dimension
        macro_rules! cat_or_init {
            ($field:expr, $new:expr, $dim:expr) => {
                $field = Some(match &$field {
                    Some(prev) => Tensor::cat(&[prev, &$new], $dim)?,
                    None => $new,
                });
            };
        }
        cat_or_init!(self.k_indices, kq.indices, self.dim);
        cat_or_init!(self.k_norms, kq.norms, self.dim);
        cat_or_init!(self.k_qjl_signs, kq.qjl_signs, self.dim);
        cat_or_init!(self.k_residual_norms, kq.residual_norms, self.dim);
        cat_or_init!(self.v_indices, vq.indices, self.dim);
        cat_or_init!(self.v_norms, vq.norms, self.dim);
        cat_or_init!(self.v_qjl_signs, vq.qjl_signs, self.dim);
        cat_or_init!(self.v_residual_norms, vq.residual_norms, self.dim);

        // Incrementally build dequantized output by concatenating with cached
        let full_k = match &self.k_dequantized {
            Some(prev) => Tensor::cat(&[prev, &new_k_deq], self.dim)?,
            None => new_k_deq,
        };
        let full_v = match &self.v_dequantized {
            Some(prev) => Tensor::cat(&[prev, &new_v_deq], self.dim)?,
            None => new_v_deq,
        };

        self.k_dequantized = Some(full_k.clone());
        self.v_dequantized = Some(full_v.clone());

        Ok((full_k, full_v))
    }

    /// Reset the cache (clear all stored keys and values).
    pub fn reset(&mut self) {
        self.k_indices = None;
        self.k_norms = None;
        self.k_qjl_signs = None;
        self.k_residual_norms = None;
        self.v_indices = None;
        self.v_norms = None;
        self.v_qjl_signs = None;
        self.v_residual_norms = None;
        self.k_dequantized = None;
        self.v_dequantized = None;
    }

    /// Get current sequence length in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.k_indices
            .as_ref()
            .and_then(|t| t.dims().get(self.dim).copied())
            .unwrap_or(0)
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k_indices.is_none()
    }

    /// Get the concatenation dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// 4D quantization result for TurboQuant (MSE indices + QJL signs + norms).
struct Quantized4D {
    indices: Tensor,
    norms: Tensor,
    qjl_signs: Tensor,
    residual_norms: Tensor,
}

/// Quantize a 4D `[B,H,S,D]` tensor using full TurboQuant (MSE + QJL).
fn turbo_quantize_4d(quantizer: &TurboQuant, x: &Tensor) -> Result<Quantized4D> {
    let x = x.contiguous()?;
    let (b, h, s, d) = x.dims4()?;
    let flat = x.reshape((b * h * s, d))?;
    let q = quantizer.quantize(&flat)?;
    let d_packed = d.div_ceil(8);
    Ok(Quantized4D {
        indices: q.polar.indices.to_dtype(DType::U8)?.reshape((b, h, s, d))?,
        norms: q.polar.norms.reshape((b, h, s, 1))?,
        qjl_signs: q.qjl_signs.reshape((b, h, s, d_packed))?,
        residual_norms: q.residual_norms.reshape((b, h, s, 1))?,
    })
}

/// Dequantize 4D TurboQuant data back to a tensor.
fn turbo_dequantize_4d(
    quantizer: &TurboQuant,
    indices: &Tensor,
    norms: &Tensor,
    qjl_signs: &Tensor,
    residual_norms: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    let (b, h, s, d) = indices.dims4()?;
    let n = b * h * s;
    let d_packed = qjl_signs.dims()[3];
    let q = TurboQuantized {
        polar: TurboMseQuantized {
            indices: indices.reshape((n, d))?.to_dtype(DType::U32)?,
            norms: norms.reshape((n,))?,
        },
        qjl_signs: qjl_signs.reshape((n, d_packed))?,
        residual_norms: residual_norms.reshape((n,))?,
    };
    let flat = quantizer.dequantize(&q)?;
    flat.reshape((b, h, s, d))?.to_dtype(dtype)
}
///
/// This is the quantized equivalent of [`KvCache`]. Uses four [`Cache`] instances
/// to store U8 indices and F32 norms for both keys and values. Pre-allocation
/// avoids repeated tensor concatenation, providing better performance than
/// [`QuantizedKvCache`] for long sequences on CPU.
///
/// # When to Use
///
/// **Recommended for:**
/// - CPU inference with long sequences
/// - When you need fixed memory allocation upfront
///
/// **Use [`QuantizedKvCache`] instead for:**
/// - GPU inference (concatenation kernels are faster than slice_set)
///
/// **Use [`QuantizedRotatingKvCache`] instead for:**
/// - Bounded-memory generation with a sliding window
#[derive(Debug, Clone)]
pub struct QuantizedPreAllocKvCache {
    k_quantizer: TurboQuant,
    v_quantizer: TurboQuant,
    k_idx: Cache,
    k_nrm: Cache,
    k_signs: Cache,
    k_rnrm: Cache,
    v_idx: Cache,
    v_nrm: Cache,
    v_signs: Cache,
    v_rnrm: Cache,
    k_dequantized: Option<Tensor>,
    v_dequantized: Option<Tensor>,
    dtype: DType,
}

impl QuantizedPreAllocKvCache {
    /// Create a new pre-allocated TurboQuant-compressed KV cache.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    /// * `head_dim` - Attention head dimension (d). Should be ≥ 32 for good quantization.
    /// * `bit_width` - Quantization bits per coordinate (b). Typically 2–4.
    /// * `max_seq_len` - Maximum sequence length for pre-allocation.
    /// * `dtype` - Data type for dequantized output tensors.
    /// * `device` - Device for tensor allocation.
    pub fn new(
        dim: usize,
        head_dim: usize,
        bit_width: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        Ok(Self {
            k_quantizer: TurboQuant::new(head_dim, bit_width, internal_dtype, device)?,
            v_quantizer: TurboQuant::new(head_dim, bit_width, internal_dtype, device)?,
            k_idx: Cache::new(dim, max_seq_len),
            k_nrm: Cache::new(dim, max_seq_len),
            k_signs: Cache::new(dim, max_seq_len),
            k_rnrm: Cache::new(dim, max_seq_len),
            v_idx: Cache::new(dim, max_seq_len),
            v_nrm: Cache::new(dim, max_seq_len),
            v_signs: Cache::new(dim, max_seq_len),
            v_rnrm: Cache::new(dim, max_seq_len),
            k_dequantized: None,
            v_dequantized: None,
            dtype,
        })
    }

    /// Append key/value tensors to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if k.dims().len() != 4 {
            candle::bail!(
                "QuantizedPreAllocKvCache expects 4D tensors [B,H,S,D], got {}D",
                k.dims().len()
            );
        }

        let kq = turbo_quantize_4d(&self.k_quantizer, &k)?;
        let vq = turbo_quantize_4d(&self.v_quantizer, &v)?;

        // Dequantize only the newly appended tokens
        let dim = self.k_idx.dim();
        let new_k_deq = turbo_dequantize_4d(
            &self.k_quantizer,
            &kq.indices,
            &kq.norms,
            &kq.qjl_signs,
            &kq.residual_norms,
            self.dtype,
        )?;
        let new_v_deq = turbo_dequantize_4d(
            &self.v_quantizer,
            &vq.indices,
            &vq.norms,
            &vq.qjl_signs,
            &vq.residual_norms,
            self.dtype,
        )?;

        self.k_idx.append(&kq.indices)?;
        self.k_nrm.append(&kq.norms)?;
        self.k_signs.append(&kq.qjl_signs)?;
        self.k_rnrm.append(&kq.residual_norms)?;
        self.v_idx.append(&vq.indices)?;
        self.v_nrm.append(&vq.norms)?;
        self.v_signs.append(&vq.qjl_signs)?;
        self.v_rnrm.append(&vq.residual_norms)?;

        // Incrementally build dequantized output by concatenating with cached
        let full_k = match &self.k_dequantized {
            Some(prev) => Tensor::cat(&[prev, &new_k_deq], dim)?,
            None => new_k_deq,
        };
        let full_v = match &self.v_dequantized {
            Some(prev) => Tensor::cat(&[prev, &new_v_deq], dim)?,
            None => new_v_deq,
        };

        self.k_dequantized = Some(full_k.clone());
        self.v_dequantized = Some(full_v.clone());

        Ok((full_k, full_v))
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.k_idx.reset();
        self.k_nrm.reset();
        self.k_signs.reset();
        self.k_rnrm.reset();
        self.v_idx.reset();
        self.v_nrm.reset();
        self.v_signs.reset();
        self.v_rnrm.reset();
        self.k_dequantized = None;
        self.v_dequantized = None;
    }

    /// Get current sequence length in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.k_idx.current_seq_len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k_idx.current_seq_len() == 0
    }
}

/// KV cache that stores quantized key/value embeddings in a rotating buffer.
///
/// This is the quantized equivalent of [`RotatingKvCache`]. Uses four
/// [`RotatingCache`] instances to store U8 indices and F32 norms for both keys
/// and values. The rotating buffer caps memory usage at `max_seq_len` positions,
/// making this suitable for long-context generation with bounded memory.
///
/// # When to Use
///
/// **Recommended for:**
/// - Streaming / long-context generation with a fixed memory budget
/// - Models that use sliding-window attention
///
/// **Use [`QuantizedKvCache`] instead for:**
/// - When all past positions must be retained
#[derive(Debug, Clone)]
pub struct QuantizedRotatingKvCache {
    k_quantizer: TurboQuant,
    v_quantizer: TurboQuant,
    k_idx: RotatingCache,
    k_nrm: RotatingCache,
    k_signs: RotatingCache,
    k_rnrm: RotatingCache,
    v_idx: RotatingCache,
    v_nrm: RotatingCache,
    v_signs: RotatingCache,
    v_rnrm: RotatingCache,
    k_dequantized: Option<Tensor>,
    v_dequantized: Option<Tensor>,
    dtype: DType,
}

impl QuantizedRotatingKvCache {
    /// Create a new rotating TurboQuant-compressed KV cache.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    /// * `head_dim` - Attention head dimension (d). Should be ≥ 32 for good quantization.
    /// * `bit_width` - Quantization bits per coordinate (b). Typically 2–4.
    /// * `max_seq_len` - Maximum sequence length (window size) for the rotating buffer.
    /// * `dtype` - Data type for dequantized output tensors.
    /// * `device` - Device for tensor allocation.
    pub fn new(
        dim: usize,
        head_dim: usize,
        bit_width: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let internal_dtype = if dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        Ok(Self {
            k_quantizer: TurboQuant::new(head_dim, bit_width, internal_dtype, device)?,
            v_quantizer: TurboQuant::new(head_dim, bit_width, internal_dtype, device)?,
            k_idx: RotatingCache::new(dim, max_seq_len),
            k_nrm: RotatingCache::new(dim, max_seq_len),
            k_signs: RotatingCache::new(dim, max_seq_len),
            k_rnrm: RotatingCache::new(dim, max_seq_len),
            v_idx: RotatingCache::new(dim, max_seq_len),
            v_nrm: RotatingCache::new(dim, max_seq_len),
            v_signs: RotatingCache::new(dim, max_seq_len),
            v_rnrm: RotatingCache::new(dim, max_seq_len),
            k_dequantized: None,
            v_dequantized: None,
            dtype,
        })
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &Device) -> Result<Option<Tensor>> {
        self.k_idx.attn_mask(seq_len, device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        self.k_idx.positions(seq_len)
    }

    /// Current write offset in the rotating buffer.
    pub fn offset(&self) -> usize {
        self.k_idx.offset()
    }

    /// Append key/value tensors to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if k.dims().len() != 4 {
            candle::bail!(
                "QuantizedRotatingKvCache expects 4D tensors [B,H,S,D], got {}D",
                k.dims().len()
            );
        }

        let dim = self.k_idx.dim();
        let seq_len = k.dim(dim)?;
        let prev_total = self.k_idx.current_seq_len();
        let max_seq = self.k_idx.max_seq_len();

        let kq = turbo_quantize_4d(&self.k_quantizer, &k)?;
        let vq = turbo_quantize_4d(&self.v_quantizer, &v)?;

        let ki = self.k_idx.append(&kq.indices)?;
        let kn = self.k_nrm.append(&kq.norms)?;
        let ks = self.k_signs.append(&kq.qjl_signs)?;
        let kr = self.k_rnrm.append(&kq.residual_norms)?;

        let vi = self.v_idx.append(&vq.indices)?;
        let vn = self.v_nrm.append(&vq.norms)?;
        let vs = self.v_signs.append(&vq.qjl_signs)?;
        let vr = self.v_rnrm.append(&vq.residual_norms)?;

        let (full_k, full_v) = if prev_total + seq_len <= max_seq {
            // No wrapping — dequantize only the new tokens and concatenate
            let new_k_deq = turbo_dequantize_4d(
                &self.k_quantizer,
                &kq.indices,
                &kq.norms,
                &kq.qjl_signs,
                &kq.residual_norms,
                self.dtype,
            )?;
            let new_v_deq = turbo_dequantize_4d(
                &self.v_quantizer,
                &vq.indices,
                &vq.norms,
                &vq.qjl_signs,
                &vq.residual_norms,
                self.dtype,
            )?;
            let fk = match &self.k_dequantized {
                Some(prev) => Tensor::cat(&[prev, &new_k_deq], dim)?,
                None => new_k_deq,
            };
            let fv = match &self.v_dequantized {
                Some(prev) => Tensor::cat(&[prev, &new_v_deq], dim)?,
                None => new_v_deq,
            };
            (fk, fv)
        } else {
            // Wrapping occurred — must dequantize the full window
            let fk = turbo_dequantize_4d(&self.k_quantizer, &ki, &kn, &ks, &kr, self.dtype)?;
            let fv = turbo_dequantize_4d(&self.v_quantizer, &vi, &vn, &vs, &vr, self.dtype)?;
            (fk, fv)
        };

        self.k_dequantized = Some(full_k.clone());
        self.v_dequantized = Some(full_v.clone());

        Ok((full_k, full_v))
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.k_idx.reset();
        self.k_nrm.reset();
        self.k_signs.reset();
        self.k_rnrm.reset();
        self.v_idx.reset();
        self.v_nrm.reset();
        self.v_signs.reset();
        self.v_rnrm.reset();
        self.k_dequantized = None;
        self.v_dequantized = None;
    }

    /// Get current sequence length in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.k_idx.current_seq_len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k_idx.current_seq_len() == 0
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

    #[test]
    fn test_quantized_kv_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq) = (1, 8, 10);

        let mut cache = QuantizedKvCache::new(2, head_dim, 4, DType::F32, &device)?;

        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;

        let (ck, cv) = cache.append(&k, &v)?;
        assert_eq!(ck.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cv.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cache.current_seq_len(), seq);

        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_incremental() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads) = (1, 4);

        let mut cache = QuantizedKvCache::new(2, head_dim, 4, DType::F32, &device)?;

        let k0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        let v0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        cache.append(&k0, &v0)?;
        assert_eq!(cache.current_seq_len(), 8);

        for expected in 9..=11 {
            let k1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let v1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let (ck, _) = cache.append(&k1, &v1)?;
            assert_eq!(ck.dims(), &[batch, heads, expected, head_dim]);
        }

        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_attention_quality() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq) = (1, 4, 16);

        let q = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores_exact = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let mut cache = QuantizedKvCache::new(2, head_dim, 4, DType::F32, &device)?;
        let (ck, _) = cache.append(&k, &v)?;
        let scores_quant = (q.matmul(&ck.transpose(2, 3)?)? * scale)?;

        let err = scores_exact
            .broadcast_sub(&scores_quant)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        let mag = scores_exact.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let rel = err / (mag + 1e-10);
        assert!(rel < 0.1, "Relative attention error: {rel:.4}");

        Ok(())
    }

    #[test]
    fn test_quantized_prealloc_basic() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq) = (1, 8, 10);
        let mut cache = QuantizedPreAllocKvCache::new(2, head_dim, 4, 512, DType::F32, &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let (ck, cv) = cache.append(&k, &v)?;
        assert_eq!(ck.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cv.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cache.current_seq_len(), seq);
        Ok(())
    }

    #[test]
    fn test_quantized_prealloc_incremental() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads) = (1, 4);
        let mut cache = QuantizedPreAllocKvCache::new(2, head_dim, 4, 512, DType::F32, &device)?;
        let k0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        let v0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        cache.append(&k0, &v0)?;
        assert_eq!(cache.current_seq_len(), 8);
        for expected in 9..=11 {
            let k1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let v1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let (ck, _) = cache.append(&k1, &v1)?;
            assert_eq!(ck.dims(), &[batch, heads, expected, head_dim]);
        }
        Ok(())
    }

    #[test]
    fn test_quantized_rotating_basic() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads) = (1, 4);
        let max_seq = 32;
        let mut cache =
            QuantizedRotatingKvCache::new(2, head_dim, 4, max_seq, DType::F32, &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, 10, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, 10, head_dim), &device)?;
        let (ck, cv) = cache.append(&k, &v)?;
        assert_eq!(ck.dims(), &[batch, heads, 10, head_dim]);
        assert_eq!(cv.dims(), &[batch, heads, 10, head_dim]);
        assert_eq!(cache.current_seq_len(), 10);
        Ok(())
    }

    #[test]
    fn test_quantized_rotating_wrap() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 32;
        let (batch, heads) = (1, 2);
        let max_seq = 8;
        let mut cache =
            QuantizedRotatingKvCache::new(2, head_dim, 4, max_seq, DType::F32, &device)?;
        // Fill past capacity
        for _ in 0..12 {
            let k = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let v = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            cache.append(&k, &v)?;
        }
        assert_eq!(cache.current_seq_len(), 12);
        // After wrap, returned data should be max_seq_len wide
        let k = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let (ck, _) = cache.append(&k, &v)?;
        assert_eq!(ck.dims()[2], max_seq); // capped at window size
        Ok(())
    }
}
