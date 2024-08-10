use candle::{Module, Result, Tensor};
use candle_nn as nn;

pub struct Qkv {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

pub struct Mlp {
    fc1: nn::Linear,
    act: nn::Activation,
    fc2: nn::Linear,
}

impl Mlp {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let fc1 = nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let act = nn::Activation::GeluPytorchTanh;
        let fc2 = nn::linear(hidden_features, in_features, vb.pp("fc2"))?;

        Ok(Self { fc1, act, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

pub struct QkvOnlyAttnProjections {
    qkv: nn::Linear,
    head_dim: usize,
}

impl QkvOnlyAttnProjections {
    pub fn new(dim: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        // {'dim': 1536, 'num_heads': 24}
        let head_dim = dim / num_heads;
        let qkv = nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        Ok(Self { qkv, head_dim })
    }

    pub fn pre_attention(&self, x: &Tensor) -> Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        split_qkv(&qkv, self.head_dim)
    }
}

pub struct AttnProjections {
    head_dim: usize,
    qkv: nn::Linear,
    proj: nn::Linear,
}

impl AttnProjections {
    pub fn new(dim: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = nn::linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            head_dim,
            qkv,
            proj,
        })
    }

    pub fn pre_attention(&self, x: &Tensor) -> Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        split_qkv(&qkv, self.head_dim)
    }

    pub fn post_attention(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

fn split_qkv(qkv: &Tensor, head_dim: usize) -> Result<Qkv> {
    let (batch_size, seq_len, _) = qkv.dims3()?;
    let qkv = qkv.reshape((batch_size, seq_len, 3, (), head_dim))?;
    let q = qkv.get_on_dim(2, 0)?;
    let q = q.reshape((batch_size, seq_len, ()))?;
    let k = qkv.get_on_dim(2, 1)?;
    let k = k.reshape((batch_size, seq_len, ()))?;
    let v = qkv.get_on_dim(2, 2)?;
    Ok(Qkv { q, k, v })
}
