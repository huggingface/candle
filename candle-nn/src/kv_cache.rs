//! Cache Implementations
//!
use crate::turboquant::{MseQuantized, TurboQuantMse};
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
#[derive(Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
    // Optional TurboQuant quantization for compressed KV storage.
    // When set, K/V are stored quantized and dequantized on retrieval.
    quant: Option<ConcatKvQuant>,
}

/// Internal quantized storage for ConcatKvCache.
#[derive(Clone)]
struct ConcatKvQuant {
    k_quantizer: TurboQuantMse,
    v_quantizer: TurboQuantMse,
    k_indices: Option<Tensor>,
    k_norms: Option<Tensor>,
    v_indices: Option<Tensor>,
    v_norms: Option<Tensor>,
    batch_size: usize,
    num_heads: usize,
    orig_dtype: DType,
}

impl std::fmt::Debug for ConcatKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConcatKvCache")
            .field("dim", &self.dim)
            .field("seq_len", &self.current_seq_len())
            .field("quantized", &self.quant.is_some())
            .finish()
    }
}

impl ConcatKvCache {
    /// Create a new empty concatenation-based KV-cache (unquantized).
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    ///   - For attention with shape `[batch, heads, seq, head_dim]`, use `dim=2`
    ///   - For attention with shape `[batch, seq, heads, head_dim]`, use `dim=1`
    pub fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,
            quant: None,
        }
    }

    /// Create a new KV-cache with TurboQuant compression enabled.
    ///
    /// When quantization is enabled, K/V vectors are stored as compact U8
    /// indices + F32 norms (~3.7× compression vs F32 at head_dim=128, 4-bit).
    /// The `append` method transparently quantizes on write and dequantizes
    /// on read — no model code changes needed.
    ///
    /// # Arguments
    /// * `dim` - Concatenation dimension (typically 2 for `[B, H, S, D]`).
    /// * `head_dim` - Attention head dimension.
    /// * `bit_width` - Quantization bits per coordinate (typically 2–4).
    /// * `dtype` - Data type of K/V tensors.
    /// * `device` - Device for tensor allocation.
    ///
    /// # Example
    /// ```ignore
    /// // 4-bit quantized KV cache — same append() API as unquantized
    /// let cache = ConcatKvCache::quantized(2, 64, 4, DType::F32, &device)?;
    /// ```
    pub fn quantized(
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
            k: None,
            v: None,
            dim,
            quant: Some(ConcatKvQuant {
                k_quantizer: TurboQuantMse::new(head_dim, bit_width, internal_dtype, device)?,
                v_quantizer: TurboQuantMse::new(head_dim, bit_width, internal_dtype, device)?,
                k_indices: None,
                k_norms: None,
                v_indices: None,
                v_norms: None,
                batch_size: 0,
                num_heads: 0,
                orig_dtype: dtype,
            }),
        })
    }

    /// Whether this cache uses TurboQuant compression.
    pub fn is_quantized(&self) -> bool {
        self.quant.is_some()
    }

    /// Get current sequence length in the cache
    pub fn current_seq_len(&self) -> usize {
        if let Some(q) = &self.quant {
            if q.batch_size > 0 && q.num_heads > 0 {
                if let Some(indices) = &q.k_indices {
                    return indices.dims()[0] / (q.batch_size * q.num_heads);
                }
            }
            return 0;
        }
        self.k
            .as_ref()
            .and_then(|k| k.dims().get(self.dim).copied())
            .unwrap_or(0)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        if self.quant.is_some() {
            return self.quant.as_ref().is_none_or(|q| q.k_indices.is_none());
        }
        self.k.is_none()
    }

    /// Get the concatenation dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append key and value tensors to the cache
    ///
    /// For unquantized caches, this concatenates tensors directly.
    /// For quantized caches, this quantizes the new data, appends the
    /// compressed representation, and returns dequantized full tensors.
    ///
    /// Either way, the returned `(full_k, full_v)` tensors contain all
    /// cached data and can be used directly for attention computation.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.quant.is_some() {
            return self.append_quantized(k, v);
        }

        let k = k.contiguous()?;
        let v = v.contiguous()?;

        self.k = Some(match &self.k {
            None => k.clone(),
            Some(k_cache) => Tensor::cat(&[k_cache, &k], self.dim)?,
        });
        self.v = Some(match &self.v {
            None => v.clone(),
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?,
        });

        Ok((
            self.k.as_ref().unwrap().clone(),
            self.v.as_ref().unwrap().clone(),
        ))
    }

    fn append_quantized(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let dims = k.dims();
        if dims.len() != 4 {
            candle::bail!(
                "ConcatKvCache quantized mode expects 4D tensors [B,H,S,D], got {}D",
                dims.len()
            );
        }
        let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);

        let q = self.quant.as_mut().unwrap();
        q.batch_size = b;
        q.num_heads = h;
        q.orig_dtype = k.dtype();

        let n = b * h * s;
        let k_flat = k.reshape((n, d))?;
        let v_flat = v.reshape((n, d))?;

        let k_q = q.k_quantizer.quantize(&k_flat)?;
        let v_q = q.v_quantizer.quantize(&v_flat)?;

        let k_idx = k_q.indices.to_dtype(DType::U8)?;
        let v_idx = v_q.indices.to_dtype(DType::U8)?;

        q.k_indices = Some(cat_or_set(&q.k_indices, &k_idx)?);
        q.k_norms = Some(cat_or_set(&q.k_norms, &k_q.norms)?);
        q.v_indices = Some(cat_or_set(&q.v_indices, &v_idx)?);
        q.v_norms = Some(cat_or_set(&q.v_norms, &v_q.norms)?);

        let full_k = dequant(
            &q.k_quantizer,
            q.k_indices.as_ref().unwrap(),
            q.k_norms.as_ref().unwrap(),
            b,
            h,
            q.orig_dtype,
        )?;
        let full_v = dequant(
            &q.v_quantizer,
            q.v_indices.as_ref().unwrap(),
            q.v_norms.as_ref().unwrap(),
            b,
            h,
            q.orig_dtype,
        )?;

        Ok((full_k, full_v))
    }

    /// Reset the cache (clear all stored keys and values)
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        if let Some(q) = &mut self.quant {
            q.k_indices = None;
            q.k_norms = None;
            q.v_indices = None;
            q.v_norms = None;
        }
    }

    /// Get reference to current K cache data
    ///
    /// Returns `None` if the cache is empty.
    /// Note: for quantized caches, this returns the last dequantized snapshot;
    /// prefer using the return value of `append()` instead.
    pub fn k(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    /// Get reference to current V cache data
    pub fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }

    /// Get mutable reference to K cache data
    pub fn k_mut(&mut self) -> Option<&mut Tensor> {
        self.k.as_mut()
    }

    /// Get mutable reference to V cache data
    pub fn v_mut(&mut self) -> Option<&mut Tensor> {
        self.v.as_mut()
    }

    /// Get owned K and V tensors, consuming the cache
    pub fn into_inner(self) -> Option<(Tensor, Tensor)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}

fn cat_or_set(existing: &Option<Tensor>, new: &Tensor) -> Result<Tensor> {
    match existing {
        Some(prev) => Tensor::cat(&[prev, new], 0),
        None => Ok(new.clone()),
    }
}

fn dequant(
    quantizer: &TurboQuantMse,
    indices: &Tensor,
    norms: &Tensor,
    batch_size: usize,
    num_heads: usize,
    dtype: DType,
) -> Result<Tensor> {
    let indices_u32 = indices.to_dtype(DType::U32)?;
    let q = MseQuantized {
        indices: indices_u32,
        norms: norms.clone(),
    };
    let flat = quantizer.dequantize(&q)?;
    let n_total = flat.dims()[0];
    let head_dim = flat.dims()[1];
    let seq_len = n_total / (batch_size * num_heads);
    flat.reshape((batch_size, num_heads, seq_len, head_dim))?
        .to_dtype(dtype)
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
    fn test_concat_cache_quantized_basic() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq) = (1, 8, 10);

        let mut cache = ConcatKvCache::quantized(2, head_dim, 4, DType::F32, &device)?;
        assert!(cache.is_quantized());
        assert!(cache.is_empty());

        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;

        let (ck, cv) = cache.append(&k, &v)?;
        assert_eq!(ck.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cv.dims(), &[batch, heads, seq, head_dim]);
        assert_eq!(cache.current_seq_len(), seq);
        assert!(!cache.is_empty());

        Ok(())
    }

    #[test]
    fn test_concat_cache_quantized_incremental() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads) = (1, 4);

        let mut cache = ConcatKvCache::quantized(2, head_dim, 4, DType::F32, &device)?;

        // Prefill: 8 tokens
        let k0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        let v0 = Tensor::randn(0f32, 1f32, (batch, heads, 8, head_dim), &device)?;
        cache.append(&k0, &v0)?;
        assert_eq!(cache.current_seq_len(), 8);

        // Decode: 1 token at a time
        for expected_len in 9..=11 {
            let k1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let v1 = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
            let (ck, cv) = cache.append(&k1, &v1)?;
            assert_eq!(ck.dims(), &[batch, heads, expected_len, head_dim]);
            assert_eq!(cv.dims(), &[batch, heads, expected_len, head_dim]);
            assert_eq!(cache.current_seq_len(), expected_len);
        }

        Ok(())
    }

    #[test]
    fn test_concat_cache_quantized_attention_quality() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let (batch, heads, seq_len) = (1, 4, 16);

        let q = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq_len, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq_len, head_dim), &device)?;

        // Unquantized attention scores
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores_exact = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Quantized (4-bit) attention scores
        let mut cache = ConcatKvCache::quantized(2, head_dim, 4, DType::F32, &device)?;
        let (ck, _cv) = cache.append(&k, &v)?;
        let scores_quant = (q.matmul(&ck.transpose(2, 3)?)? * scale)?;

        let score_err = scores_exact
            .broadcast_sub(&scores_quant)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        let score_mag = scores_exact.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let rel_err = score_err / (score_mag + 1e-10);

        assert!(
            rel_err < 0.1,
            "Quantized attention score relative error too high: {rel_err:.4}"
        );

        Ok(())
    }

    #[test]
    fn test_concat_cache_quantized_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::quantized(2, 32, 3, DType::F32, &device)?;

        let k = Tensor::randn(0f32, 1f32, (1, 2, 5, 32), &device)?;
        let v = Tensor::randn(0f32, 1f32, (1, 2, 5, 32), &device)?;
        cache.append(&k, &v)?;
        assert!(!cache.is_empty());

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);

        Ok(())
    }
}
