use candle::{DType, Result, Tensor};

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }
}

impl candle::Forward for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let (_bsize, _seq_len, hidden_size) = x.shape().r3()?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum(&[2])? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = ((&x * &x)?.sum(&[2])? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
}
