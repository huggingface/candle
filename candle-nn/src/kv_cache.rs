//! Cache Implementations
//!
use candle::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Cache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reseted (as in it will not share
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
        if self.current_seq_len + seq_len > self.max_seq_len {
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

    /// Returns the positions corresponding to all the elements that will be retured
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

    /// Returns the positions corresponding to all the elements that will be retured
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
        // up being almost 10x faster with candle 0.9.0. The slowness seems to be on the CPU
        // copies, to be further investigated.
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
}
