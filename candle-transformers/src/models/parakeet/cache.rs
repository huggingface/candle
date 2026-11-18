use candle::{Result, Tensor};

pub trait CacheLike {
    fn offset(&self) -> usize;
    fn update_and_fetch_kv(&mut self, keys: Tensor, values: Tensor) -> Result<(Tensor, Tensor)>;
    fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor>;
}

#[derive(Debug, Clone, Default)]
pub struct ConformerCache {
    pub keys: Option<Tensor>,
    pub values: Option<Tensor>,
    pub conv: Option<Tensor>,
    pub offset: usize,
    step: usize,
}

impl ConformerCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            conv: None,
            offset: 0,
            step: 256,
        }
    }

    pub fn update_and_fetch_kv(
        &mut self,
        keys: Tensor,
        values: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, s, _) = keys.dims4()?;
        if let (Some(k), Some(v)) = (&self.keys, &self.values) {
            let new_k = Tensor::cat(&[k, &keys], 2)?;
            let new_v = Tensor::cat(&[v, &values], 2)?;
            self.keys = Some(new_k);
            self.values = Some(new_v);
        } else {
            self.keys = Some(keys);
            self.values = Some(values);
        }
        self.offset += s;
        Ok((
            self.keys.as_ref().unwrap().clone(),
            self.values.as_ref().unwrap().clone(),
        ))
    }

    pub fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        if padding == 0 {
            return Ok(x.clone());
        }
        let (_, s, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let conv_cache = if let Some(cache) = &self.conv {
            cache.clone()
        } else {
            let zeros = Tensor::zeros((x.dims3()?.0, padding, x.dims3()?.2), dtype, device)?;
            zeros
        };

        let tokens_to_cache = padding.min(s);
        let cache_update = x.narrow(1, s - tokens_to_cache, tokens_to_cache)?;
        let new_cache = if tokens_to_cache < padding {
            let trimmed = conv_cache.narrow(1, tokens_to_cache, padding - tokens_to_cache)?;
            Tensor::cat(&[&trimmed, &cache_update], 1)?
        } else {
            cache_update
        };

        self.conv = Some(new_cache.clone());
        let mut result = Tensor::cat(&[&new_cache, x], 1)?;
        result = result.pad_with_zeros(candle::D::Minus2, 0, padding)?;
        Ok(result)
    }
}

impl CacheLike for ConformerCache {
    fn offset(&self) -> usize {
        self.offset
    }

    fn update_and_fetch_kv(&mut self, keys: Tensor, values: Tensor) -> Result<(Tensor, Tensor)> {
        ConformerCache::update_and_fetch_kv(self, keys, values)
    }

    fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        ConformerCache::update_and_fetch_conv(self, x, padding)
    }
}

#[derive(Debug, Clone)]
pub struct RotatingConformerCache {
    pub keys: Option<Tensor>,
    pub values: Option<Tensor>,
    pub conv: Option<Tensor>,
    pub offset: usize,
    capacity: usize,
    cache_drop_size: usize,
}

impl RotatingConformerCache {
    pub fn new(capacity: usize, cache_drop_size: usize) -> Self {
        Self {
            keys: None,
            values: None,
            conv: None,
            offset: 0,
            capacity,
            cache_drop_size,
        }
    }

    pub fn update_and_fetch_kv(
        &mut self,
        keys: Tensor,
        values: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, s, _) = keys.dims4()?;
        let drop = self.cache_drop_size.min(s);
        let to_cache = s.saturating_sub(drop).min(self.capacity);

        let new_kv = if to_cache > 0 {
            let start = s - drop - to_cache;
            let k_chunk = keys.narrow(2, start, to_cache)?;
            let v_chunk = values.narrow(2, start, to_cache)?;
            if let (Some(k), Some(v)) = (&self.keys, &self.values) {
                let k_cat = Tensor::cat(&[k, &k_chunk], 2)?;
                let v_cat = Tensor::cat(&[v, &v_chunk], 2)?;
                let k_trim = if k_cat.dims4()?.2 > self.capacity {
                    let start = k_cat.dims4()?.2 - self.capacity;
                    k_cat.narrow(2, start, self.capacity)?
                } else {
                    k_cat
                };
                let v_trim = if v_cat.dims4()?.2 > self.capacity {
                    let start = v_cat.dims4()?.2 - self.capacity;
                    v_cat.narrow(2, start, self.capacity)?
                } else {
                    v_cat
                };
                (k_trim, v_trim)
            } else {
                (k_chunk, v_chunk)
            }
        } else {
            (
                self.keys
                    .clone()
                    .unwrap_or_else(|| keys.narrow(2, 0, 0).unwrap()),
                self.values
                    .clone()
                    .unwrap_or_else(|| values.narrow(2, 0, 0).unwrap()),
            )
        };

        self.keys = Some(new_kv.0.clone());
        self.values = Some(new_kv.1.clone());
        self.offset += to_cache;

        let k_out = if let Some(k) = &self.keys {
            Tensor::cat(&[k, &keys], 2)?
        } else {
            keys
        };
        let v_out = if let Some(v) = &self.values {
            Tensor::cat(&[v, &values], 2)?
        } else {
            values
        };

        Ok((k_out, v_out))
    }

    pub fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        if padding == 0 {
            return Ok(x.clone());
        }

        let (_, s, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let conv_cache = if let Some(cache) = &self.conv {
            cache.clone()
        } else {
            Tensor::zeros((x.dims3()?.0, padding, x.dims3()?.2), dtype, device)?
        };

        let mut new_cache = conv_cache;
        if s > self.cache_drop_size {
            let tokens_to_cache = padding.min(s - self.cache_drop_size);
            let cache_update = x.narrow(1, s - tokens_to_cache, tokens_to_cache)?;
            new_cache = if tokens_to_cache < padding {
                let trimmed = new_cache.narrow(1, tokens_to_cache, padding - tokens_to_cache)?;
                Tensor::cat(&[&trimmed, &cache_update], 1)?
            } else {
                cache_update
            };
        }

        self.conv = Some(new_cache.clone());
        let mut result = Tensor::cat(&[&new_cache, x], 1)?;
        result = result.pad_with_zeros(candle::D::Minus2, 0, padding)?;
        Ok(result)
    }
}

impl CacheLike for RotatingConformerCache {
    fn offset(&self) -> usize {
        self.offset
    }

    fn update_and_fetch_kv(&mut self, keys: Tensor, values: Tensor) -> Result<(Tensor, Tensor)> {
        RotatingConformerCache::update_and_fetch_kv(self, keys, values)
    }

    fn update_and_fetch_conv(&mut self, x: &Tensor, padding: usize) -> Result<Tensor> {
        RotatingConformerCache::update_and_fetch_conv(self, x, padding)
    }
}
