use candle::Tensor;

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let w = match x.dims() {
            &[bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}
