use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    internal_dim: usize,
    num_heads: usize,
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
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        x.reshape((b, n, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_heads, n_tokens, c_per_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, n_tokens, n_heads * c_per_head))
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(q)?;
        let k = self.k_proj.forward(k)?;
        let v = self.v_proj.forward(v)?;

        let q = self.separate_heads(&q)?;
        let k = self.separate_heads(&k)?;
        let v = self.separate_heads(&v)?;

        let (_, _, _, c_per_head) = q.dims4()?;
        let attn = (q.matmul(&k.t()?)? / (c_per_head as f64).sqrt())?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        self.recombine_heads(&out)?.apply(&self.out_proj)
    }
}

#[derive(Debug)]
struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: LayerNorm,
    cross_attn_token_to_image: Attention,
    norm2: LayerNorm,
    mlp: crate::MlpBlock,
    norm3: LayerNorm,
    norm4: LayerNorm,
    cross_attn_image_to_token: Attention,
    skip_first_layer_pe: bool,
}

impl TwoWayAttentionBlock {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        skip_first_layer_pe: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(embedding_dim, num_heads, 1, vb.pp("self_attn"))?;
        let norm1 = layer_norm(embedding_dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(embedding_dim, 1e-5, vb.pp("norm2"))?;
        let norm3 = layer_norm(embedding_dim, 1e-5, vb.pp("norm3"))?;
        let norm4 = layer_norm(embedding_dim, 1e-5, vb.pp("norm4"))?;
        let self_attn = Attention::new(embedding_dim, num_heads, 1, vb.pp("self_attn"))?;
        let cross_attn_token_to_image = Attention::new(
            embedding_dim,
            num_heads,
            2,
            vb.pp("cross_attn_token_to_image"),
        )?;
        let cross_attn_image_to_token = Attention::new(
            embedding_dim,
            num_heads,
            2,
            vb.pp("cross_attn_image_to_token"),
        )?;
        // TODO: use relu in this mlp
        let mlp = crate::MlpBlock::new(embedding_dim, mlp_dim, vb.pp("mlp"))?;
        Ok(Self {
            self_attn,
            norm1,
            cross_attn_image_to_token,
            norm2,
            mlp,
            norm3,
            norm4,
            cross_attn_token_to_image,
            skip_first_layer_pe,
        })
    }
}
