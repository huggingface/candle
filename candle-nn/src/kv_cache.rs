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
        // Detach inputs to break BackpropOp chain - KV caches are inference-only.
        let k = k.contiguous()?.detach();
        let v = v.contiguous()?.detach();

        self.k = Some(match &self.k {
            None => k,
            Some(k_cache) => Tensor::cat(&[k_cache, &k], self.dim)?.detach(),
        });

        self.v = Some(match &self.v {
            None => v,
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?.detach(),
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

    // ---- QuantizedKvCache tests ----

    #[test]
    fn test_quantized_kv_cache_roundtrip() -> Result<()> {
        // Quantise + dequantise should recover the original values within INT8 tolerance.
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut cache = QuantizedKvCache::new(2, 0); // no sinks

        let k = Tensor::from_vec(
            vec![-1.0f32, 0.5, 0.0, 1.0, -0.5, 0.25, 0.75, -0.25],
            (1, 1, 2, 4),
            &device,
        )?;
        let v = k.neg()?;
        let (k_out, v_out) = cache.append(&k, &v)?;

        assert_eq!(k_out.dims(), k.dims());
        assert_eq!(v_out.dims(), v.dims());

        // Max quantisation error for symmetric INT8 is scale/127 * 0.5 ≈ 0.004 per entry.
        let k_err = (k_out.to_dtype(dtype)? - k.to_dtype(dtype)?)?.abs()?.max(0)?.max(0)?.max(0)?.max(0)?;
        let k_err: f32 = k_err.to_vec0()?;
        assert!(k_err < 0.02, "k roundtrip error {k_err} exceeds tolerance");
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_sinks_full_precision() -> Result<()> {
        // The first n_sink_tokens must be returned without quantisation loss.
        let device = Device::Cpu;
        let n_sink = 4;
        let mut cache = QuantizedKvCache::new(2, n_sink);

        // Append exactly n_sink tokens — they should all go into the sink.
        let k_sink = Tensor::randn(0f32, 1., (1, 4, n_sink, 64), &device)?;
        let v_sink = Tensor::randn(0f32, 1., (1, 4, n_sink, 64), &device)?;
        let (k_out, v_out) = cache.append(&k_sink, &v_sink)?;

        let diff_k = (k_out - k_sink)?.abs()?.max(0)?.max(0)?.max(0)?.max(0)?;
        let diff_k: f32 = diff_k.to_vec0()?;
        assert_eq!(diff_k, 0.0, "sink tokens should be lossless; got diff={diff_k}");

        let diff_v = (v_out - v_sink)?.abs()?.max(0)?.max(0)?.max(0)?.max(0)?;
        let diff_v: f32 = diff_v.to_vec0()?;
        assert_eq!(diff_v, 0.0, "sink tokens should be lossless; got diff={diff_v}");
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_mixed_sink_and_bulk() -> Result<()> {
        // Append more tokens than n_sink; first n_sink stay full-precision,
        // rest are quantised. The combined output shape must be correct.
        let device = Device::Cpu;
        let n_sink = 4;
        let total = 10;
        let mut cache = QuantizedKvCache::new(2, n_sink);

        let k = Tensor::randn(0f32, 1., (1, 4, total, 64), &device)?;
        let v = Tensor::randn(0f32, 1., (1, 4, total, 64), &device)?;
        let (k_out, v_out) = cache.append(&k, &v)?;

        assert_eq!(k_out.dims(), &[1, 4, total, 64]);
        assert_eq!(v_out.dims(), &[1, 4, total, 64]);
        assert_eq!(cache.current_seq_len(), total);
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_incremental_append() -> Result<()> {
        // Append in multiple steps and check growing sequence length.
        let device = Device::Cpu;
        let mut cache = QuantizedKvCache::new(2, 2);

        for step in 1..=5usize {
            let k = Tensor::randn(0f32, 1., (1, 2, 1, 32), &device)?;
            let v = Tensor::randn(0f32, 1., (1, 2, 1, 32), &device)?;
            let (k_out, _) = cache.append(&k, &v)?;
            assert_eq!(k_out.dim(2)?, step);
        }
        assert_eq!(cache.current_seq_len(), 5);
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = QuantizedKvCache::new(2, 2);
        let k = Tensor::randn(0f32, 1., (1, 2, 4, 32), &device)?;
        let v = k.clone();
        cache.append(&k, &v)?;
        assert_eq!(cache.current_seq_len(), 4);
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);
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
}

/// Stores a single tensor with shape `(S, H_kv, 2*D)` where the first D
/// elements are K and the next D are V. This sequence-first layout means:
/// - The kernel reads `kv[pos * H_kv * 2D + head * 2D .. + 2D]` directly
/// - No transpose needed before the flash kernel
/// - K and V for the same position share cache lines
///
/// Input K and V have shape `(B, H_kv, S, D)` (standard attention format)
/// and are transposed+interleaved on append.
#[derive(Debug, Clone)]
pub struct InterleavedKvCache {
    kv: Option<Tensor>,
    head_dim: usize,
}

impl InterleavedKvCache {
    pub fn new(head_dim: usize) -> Self {
        Self { kv: None, head_dim }
    }

    pub fn current_seq_len(&self) -> usize {
        self.kv
            .as_ref()
            .and_then(|kv| kv.dims().first().copied())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.kv.is_none()
    }

    /// Append K, V of shape `(1, H_kv, S_new, D)`. Batch must be 1.
    /// Returns the full cache `(S_total, H_kv, 2*D)`.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // Detach to break BackpropOp chain - KV caches are inference-only.
        let k_seq = k.squeeze(0)?.transpose(0, 1)?.contiguous()?.detach();
        let v_seq = v.squeeze(0)?.transpose(0, 1)?.contiguous()?.detach();
        let kv_new = Tensor::cat(&[&k_seq, &v_seq], 2)?.detach();

        self.kv = Some(match &self.kv {
            None => kv_new,
            Some(kv_cache) => Tensor::cat(&[kv_cache, &kv_new], 0)?.detach(),
        });

        Ok(self.kv.as_ref().unwrap().clone())
    }

    /// Get the raw interleaved KV tensor `(S, H_kv, 2*D)`.
    pub fn kv(&self) -> Option<&Tensor> {
        self.kv.as_ref()
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn reset(&mut self) {
        self.kv = None;
    }
}

// Work with KV cache directly in interleaved space.
#[derive(Debug, Clone)]
pub struct RawInterleavedKvCache {
    buf: Vec<f32>,
    h_kv: usize,
    d: usize,
    pos_stride: usize,
    len: usize,
}

impl RawInterleavedKvCache {
    /// Create a new cache with space for `max_seq` positions.
    pub fn new(h_kv: usize, d: usize, max_seq: usize) -> Self {
        let pos_stride = h_kv * 2 * d;
        Self {
            buf: vec![0f32; max_seq * pos_stride],
            h_kv,
            d,
            pos_stride,
            len: 0,
        }
    }

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Write one position of K and V into the cache.
    ///
    /// `k_flat` and `v_flat` are flat `(H_kv * D)` slices from the projection output.
    /// They are interleaved per-head into the buffer: `[H0_K, H0_V, H1_K, H1_V, ...]`.
    pub fn write_kv(&mut self, k_flat: &[f32], v_flat: &[f32]) {
        let pos = self.len;
        let base = pos * self.pos_stride;
        let d = self.d;

        // Grow buffer if needed
        if base + self.pos_stride > self.buf.len() {
            self.buf.resize(self.buf.len() * 2, 0.0);
        }

        for h in 0..self.h_kv {
            let k_src = h * d;
            let v_src = h * d;
            let dst = base + h * 2 * d;
            self.buf[dst..dst + d].copy_from_slice(&k_flat[k_src..k_src + d]);
            self.buf[dst + d..dst + 2 * d].copy_from_slice(&v_flat[v_src..v_src + d]);
        }

        self.len += 1;
    }

    /// Write multiple positions of K and V (for prefill).
    ///
    /// `k_flat` is `(S, H_kv * D)` row-major, `v_flat` same.
    pub fn write_kv_batch(&mut self, k_flat: &[f32], v_flat: &[f32], seq_len: usize) {
        let hd = self.h_kv * self.d;
        for s in 0..seq_len {
            let k_row = &k_flat[s * hd..(s + 1) * hd];
            let v_row = &v_flat[s * hd..(s + 1) * hd];
            self.write_kv(k_row, v_row);
        }
    }

    /// Get the active portion of the cache as a slice: `(len * H_kv * 2 * D)` elements.
    pub fn data(&self) -> &[f32] {
        &self.buf[..self.len * self.pos_stride]
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn h_kv(&self) -> usize {
        self.h_kv
    }

    pub fn d(&self) -> usize {
        self.d
    }
}

/// INT8-quantised KV cache following TurboQuant (Google Research, ICLR 2024).
///
/// Keys and values beyond the attention-sink window are quantised to `U8`
/// (symmetric INT8 with a +128 bias) with per-token per-head scales, reducing
/// KV-cache memory by ~4× vs BF16/F16. The first `n_sink_tokens` positions are
/// kept in full precision because initial tokens accumulate disproportionately
/// large attention scores ("attention sinks") and quantising them degrades
/// output quality.
///
/// ## Quantisation scheme
///
/// For each tensor `x` of shape `(B, H, S, D)`:
/// ```text
/// scale = clamp(max(|x|, dim=D), min=1e-6)   // shape (B, H, S, 1)
/// q     = clip(round(x / scale) + 128, 0, 255) // U8, same shape as x
/// x'    = (q.f32() - 128) * scale             // dequant, f32
/// ```
///
/// ## Memory layout
/// * `k_sink` / `v_sink` – full-precision tensors, shape `(B, H, n_sink, D)`
/// * `k_q` / `v_q`       – U8 tensors, shape `(B, H, seq_len-n_sink, D)`
/// * `k_scale` / `v_scale` – F32 scales, shape `(B, H, seq_len-n_sink, 1)`
///
/// ## Example
/// ```ignore
/// use candle_nn::kv_cache::QuantizedKvCache;
///
/// let mut cache = QuantizedKvCache::new(/*dim=*/2, /*n_sink_tokens=*/4);
///
/// let k = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let v = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let (k_out, v_out) = cache.append(&k, &v)?;
/// // k_out / v_out are dequantised F32 tensors ready for attention.
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    // Attention sinks – full precision, shape (B, H, n_sink, D).
    k_sink: Option<Tensor>,
    v_sink: Option<Tensor>,
    // Quantised bulk – U8, shape (B, H, bulk_len, D).
    k_q: Option<Tensor>,
    v_q: Option<Tensor>,
    // Per-token per-head scales – F32, shape (B, H, bulk_len, 1).
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    // Dimension along which the sequence grows (typically 2).
    dim: usize,
    n_sink_tokens: usize,
    current_seq_len: usize,
}

impl QuantizedKvCache {
    /// Create a new cache.
    ///
    /// * `dim` – the sequence dimension of the K/V tensors (usually 2 for
    ///   tensors with shape `[batch, heads, seq, head_dim]`).
    /// * `n_sink_tokens` – number of leading tokens to keep at full precision.
    ///   The TurboQuant paper uses 4.
    pub fn new(dim: usize, n_sink_tokens: usize) -> Self {
        Self {
            k_sink: None,
            v_sink: None,
            k_q: None,
            v_q: None,
            k_scale: None,
            v_scale: None,
            dim,
            n_sink_tokens,
            current_seq_len: 0,
        }
    }

    /// Number of sink tokens kept at full precision.
    pub fn n_sink_tokens(&self) -> usize {
        self.n_sink_tokens
    }

    /// Total sequence length stored (sinks + quantised bulk).
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// `true` when no tokens have been appended yet.
    pub fn is_empty(&self) -> bool {
        self.current_seq_len == 0
    }

    /// Quantise `x` to U8 with per-token per-head symmetric INT8 encoding.
    ///
    /// Returns `(q_u8, scale_f32)` where `scale` has shape `(B, H, S, 1)`.
    fn quantize(x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute per-token per-head max(|x|) → shape (B, H, S, 1).
        let scale = x
            .abs()?
            .max_keepdim(x.rank() - 1)?
            .clamp(1e-6f64, f64::INFINITY)?
            .to_dtype(DType::F32)?;
        // Normalise and map to [0, 255] symmetric around 128.
        let x_f32 = x.to_dtype(DType::F32)?;
        let q = ((x_f32.broadcast_div(&scale)? + 128f64)?.clamp(0f64, 255f64)?)
            .round()?
            .to_dtype(DType::U8)?;
        Ok((q, scale))
    }

    /// Dequantise `q_u8` using `scale_f32` back to the original dtype.
    fn dequantize(q: &Tensor, scale: &Tensor, dtype: DType) -> Result<Tensor> {
        let q_f32 = q.to_dtype(DType::F32)?;
        let x_f32 = (q_f32 - 128f64)?.broadcast_mul(scale)?;
        x_f32.to_dtype(dtype)
    }

    fn append_quantized(
        existing_q: &mut Option<Tensor>,
        existing_scale: &mut Option<Tensor>,
        new_q: Tensor,
        new_scale: Tensor,
        dim: usize,
    ) -> Result<()> {
        *existing_q = Some(match existing_q.take() {
            None => new_q,
            Some(q) => Tensor::cat(&[&q, &new_q], dim)?,
        });
        *existing_scale = Some(match existing_scale.take() {
            None => new_scale,
            Some(s) => Tensor::cat(&[&s, &new_scale], dim)?,
        });
        Ok(())
    }

    /// Append `k` and `v` tensors, storing sinks at full precision and the
    /// rest quantised to U8. Returns the full dequantised `(K, V)` pair ready
    /// for attention.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = k.dtype();
        let seq_len = k.dim(self.dim)?;

        let sink_deficit = self.n_sink_tokens.saturating_sub(self.current_seq_len);
        let sink_new = sink_deficit.min(seq_len);
        let bulk_new = seq_len - sink_new;

        // --- Handle sink portion ---
        if sink_new > 0 {
            let k_s = k.narrow(self.dim, 0, sink_new)?;
            let v_s = v.narrow(self.dim, 0, sink_new)?;
            self.k_sink = Some(match self.k_sink.take() {
                None => k_s,
                Some(prev) => Tensor::cat(&[&prev, &k_s], self.dim)?,
            });
            self.v_sink = Some(match self.v_sink.take() {
                None => v_s,
                Some(prev) => Tensor::cat(&[&prev, &v_s], self.dim)?,
            });
        }

        // --- Handle bulk (quantised) portion ---
        if bulk_new > 0 {
            let k_b = k.narrow(self.dim, sink_new, bulk_new)?;
            let v_b = v.narrow(self.dim, sink_new, bulk_new)?;
            let (k_q, k_s) = Self::quantize(&k_b)?;
            let (v_q, v_s) = Self::quantize(&v_b)?;
            Self::append_quantized(&mut self.k_q, &mut self.k_scale, k_q, k_s, self.dim)?;
            Self::append_quantized(&mut self.v_q, &mut self.v_scale, v_q, v_s, self.dim)?;
        }

        self.current_seq_len += seq_len;

        // --- Reconstruct full tensors for attention ---
        let k_out = self.current_k(dtype)?;
        let v_out = self.current_v(dtype)?;
        Ok((k_out, v_out))
    }

    /// Return the full dequantised key tensor at `dtype`.
    pub fn current_k(&self, dtype: DType) -> Result<Tensor> {
        match (&self.k_sink, &self.k_q, &self.k_scale) {
            (Some(sink), Some(q), Some(scale)) => {
                let bulk = Self::dequantize(q, scale, dtype)?;
                let sink = sink.to_dtype(dtype)?;
                Tensor::cat(&[&sink, &bulk], self.dim)
            }
            (Some(sink), None, _) => sink.to_dtype(dtype),
            (None, Some(q), Some(scale)) => Self::dequantize(q, scale, dtype),
            _ => candle::bail!("QuantizedKvCache: empty cache"),
        }
    }

    /// Return the full dequantised value tensor at `dtype`.
    pub fn current_v(&self, dtype: DType) -> Result<Tensor> {
        match (&self.v_sink, &self.v_q, &self.v_scale) {
            (Some(sink), Some(q), Some(scale)) => {
                let bulk = Self::dequantize(q, scale, dtype)?;
                let sink = sink.to_dtype(dtype)?;
                Tensor::cat(&[&sink, &bulk], self.dim)
            }
            (Some(sink), None, _) => sink.to_dtype(dtype),
            (None, Some(q), Some(scale)) => Self::dequantize(q, scale, dtype),
            _ => candle::bail!("QuantizedKvCache: empty cache"),
        }
    }

    /// Reset the cache, clearing all stored tensors.
    pub fn reset(&mut self) {
        self.k_sink = None;
        self.v_sink = None;
        self.k_q = None;
        self.v_q = None;
        self.k_scale = None;
        self.v_scale = None;
        self.current_seq_len = 0;
    }
}

/// Apply interleaved RoPE in-place on a flat `(H, D)` slice.
///
/// Pairs adjacent elements `(2i, 2i+1)` with frequency `cos[i], sin[i]`.
/// `cos` and `sin` are `(D/2,)` for the current position.
pub fn rope_i_inplace(data: &mut [f32], cos: &[f32], sin: &[f32], num_heads: usize, d: usize) {
    let half_d = d / 2;
    for h in 0..num_heads {
        let base = h * d;
        for i in 0..half_d {
            let a = data[base + 2 * i];
            let b = data[base + 2 * i + 1];
            data[base + 2 * i] = a * cos[i] - b * sin[i];
            data[base + 2 * i + 1] = b * cos[i] + a * sin[i];
        }
    }
}
