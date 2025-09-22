use candle::{BackendStorage, Module, Result, Tensor};
use candle_nn as nn;

pub struct Qkv<B: BackendStorage> {
    pub q: Tensor<B>,
    pub k: Tensor<B>,
    pub v: Tensor<B>,
}

pub struct Mlp<B: BackendStorage> {
    fc1: nn::Linear<B>,
    act: nn::Activation,
    fc2: nn::Linear<B>,
}

impl<B: BackendStorage> Mlp<B> {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        vb: candle_nn::VarBuilder<B>,
    ) -> Result<Self> {
        let fc1 = nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let act = nn::Activation::GeluPytorchTanh;
        let fc2 = nn::linear(hidden_features, in_features, vb.pp("fc2"))?;

        Ok(Self { fc1, act, fc2 })
    }
}

impl<B: BackendStorage> Module<B> for Mlp<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let x = self.fc1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

pub struct QkvOnlyAttnProjections<B: BackendStorage> {
    qkv: nn::Linear<B>,
    head_dim: usize,
}

impl<B: BackendStorage> QkvOnlyAttnProjections<B> {
    pub fn new(dim: usize, num_heads: usize, vb: nn::VarBuilder<B>) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        Ok(Self { qkv, head_dim })
    }

    pub fn pre_attention(&self, x: &Tensor<B>) -> Result<Qkv<B>> {
        let qkv = self.qkv.forward(x)?;
        split_qkv(&qkv, self.head_dim)
    }
}

pub struct AttnProjections<B: BackendStorage> {
    head_dim: usize,
    qkv: nn::Linear<B>,
    ln_k: Option<candle_nn::RmsNorm<B>>,
    ln_q: Option<candle_nn::RmsNorm<B>>,
    proj: nn::Linear<B>,
}

impl<B: BackendStorage> AttnProjections<B> {
    pub fn new(dim: usize, num_heads: usize, vb: nn::VarBuilder<B>) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = nn::linear(dim, dim, vb.pp("proj"))?;
        let (ln_k, ln_q) = if vb.contains_tensor("ln_k.weight") {
            let ln_k = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_k"))?;
            let ln_q = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_q"))?;
            (Some(ln_k), Some(ln_q))
        } else {
            (None, None)
        };
        Ok(Self {
            head_dim,
            qkv,
            proj,
            ln_k,
            ln_q,
        })
    }

    pub fn pre_attention(&self, x: &Tensor<B>) -> Result<Qkv<B>> {
        let qkv = self.qkv.forward(x)?;
        let Qkv { q, k, v } = split_qkv(&qkv, self.head_dim)?;
        let q = match self.ln_q.as_ref() {
            None => q,
            Some(l) => {
                let (b, t, h) = q.dims3()?;
                l.forward(&q.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        let k = match self.ln_k.as_ref() {
            None => k,
            Some(l) => {
                let (b, t, h) = k.dims3()?;
                l.forward(&k.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        Ok(Qkv { q, k, v })
    }

    pub fn post_attention(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        self.proj.forward(x)
    }
}

fn split_qkv<B: BackendStorage>(qkv: &Tensor<B>, head_dim: usize) -> Result<Qkv<B>> {
    let (batch_size, seq_len, _) = qkv.dims3()?;
    let qkv = qkv.reshape((batch_size, seq_len, 3, (), head_dim))?;
    let q = qkv.get_on_dim(2, 0)?;
    let q = q.reshape((batch_size, seq_len, ()))?;
    let k = qkv.get_on_dim(2, 1)?;
    let k = k.reshape((batch_size, seq_len, ()))?;
    let v = qkv.get_on_dim(2, 2)?;
    Ok(Qkv { q, k, v })
}
