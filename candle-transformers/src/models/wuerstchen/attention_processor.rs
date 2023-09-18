#![allow(unused)]
use candle::{Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

#[derive(Debug)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    scale: f64,
}

impl Attention {
    pub fn new(query_dim: usize, heads: usize, dim_head: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(query_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(query_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            scale,
        })
    }
}
