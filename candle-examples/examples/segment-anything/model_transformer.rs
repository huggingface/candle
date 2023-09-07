use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    internal_dim: usize,
}

impl Attention {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let internal_dim = embedding_dim / downsample_rate;
        let q_proj = candle_nn::linear(embedding_dim, internal_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(embedding_dim, internal_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embedding_dim, internal_dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(internal_dim, embedding_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            internal_dim,
        })
    }
}
