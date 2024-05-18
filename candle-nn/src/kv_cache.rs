use candle::{DType, Device, Result, Shape, Tensor};

#[derive(Debug, Clone)]
pub struct Cache {
    all_data: Tensor,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
}

impl Cache {
    pub fn new<S: Into<Shape>, D: candle::shape::Dim>(
        dim: D,
        shape: S,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let max_seq_len = shape.dims()[dim];
        let all_data = Tensor::zeros(shape, dtype, dev)?;
        Ok(Self {
            all_data,
            dim,
            current_seq_len: 0,
            max_seq_len,
        })
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

    pub fn all_data(&self) -> &Tensor {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<Tensor> {
        self.all_data.narrow(self.dim, 0, self.current_seq_len)
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        if self.current_seq_len + seq_len > self.max_seq_len {
            candle::bail!(
                "kv-cache: above max-seq-len {}+{seq_len}>{}",
                self.current_seq_len,
                self.max_seq_len
            )
        }
        self.all_data
            .slice_set(src, self.dim, self.current_seq_len)?;
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
    pub fn new<S: Into<Shape>, D: candle::shape::Dim>(
        dim: D,
        shape: S,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let k = Cache::new(dim, &shape, dtype, dev)?;
        let v = Cache::new(dim, &shape, dtype, dev)?;
        Ok(Self { k, v })
    }

    pub fn k(&self) -> Result<Tensor> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<Tensor> {
        self.v.current_data()
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let k = self.k.current_data()?;
        let v = self.v.current_data()?;
        Ok((k, v))
    }
}
