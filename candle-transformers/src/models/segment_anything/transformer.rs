use candle::{Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
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
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        x.reshape((b, n, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_heads, n_tokens, c_per_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, n_tokens, n_heads * c_per_head))
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(&q.contiguous()?)?;
        let k = self.k_proj.forward(&k.contiguous()?)?;
        let v = self.v_proj.forward(&v.contiguous()?)?;

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
    mlp: super::MlpBlock,
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
        let mlp = super::MlpBlock::new(
            embedding_dim,
            mlp_dim,
            candle_nn::Activation::Relu,
            vb.pp("mlp"),
        )?;
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

    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Self attention block
        let queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = (queries + query_pe)?;
            let attn_out = self.self_attn.forward(&q, &q, queries)?;
            (queries + attn_out)?
        };
        let queries = self.norm1.forward(&queries)?;

        // Cross attention block, tokens attending to image embedding
        let q = (&queries + query_pe)?;
        let k = (keys + key_pe)?;
        let attn_out = self.cross_attn_token_to_image.forward(&q, &k, keys)?;
        let queries = (&queries + attn_out)?;
        let queries = self.norm2.forward(&queries)?;

        // MLP block
        let mlp_out = self.mlp.forward(&queries);
        let queries = (queries + mlp_out)?;
        let queries = self.norm3.forward(&queries)?;

        // Cross attention block, image embedding attending to tokens
        let q = (&queries + query_pe)?;
        let k = (keys + key_pe)?;
        let attn_out = self.cross_attn_image_to_token.forward(&k, &q, &queries)?;
        let keys = (keys + attn_out)?;
        let keys = self.norm4.forward(&keys)?;

        Ok((queries, keys))
    }
}

#[derive(Debug)]
pub struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNorm,
}

impl TwoWayTransformer {
    pub fn new(
        depth: usize,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let layer =
                TwoWayAttentionBlock::new(embedding_dim, num_heads, mlp_dim, i == 0, vb_l.pp(i))?;
            layers.push(layer)
        }
        let final_attn_token_to_image = Attention::new(
            embedding_dim,
            num_heads,
            2,
            vb.pp("final_attn_token_to_image"),
        )?;
        let norm_final_attn = layer_norm(embedding_dim, 1e-5, vb.pp("norm_final_attn"))?;
        Ok(Self {
            layers,
            final_attn_token_to_image,
            norm_final_attn,
        })
    }

    pub fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let image_embedding = image_embedding.flatten_from(2)?.permute((0, 2, 1))?;
        let image_pe = image_pe.flatten_from(2)?.permute((0, 2, 1))?;

        let mut queries = point_embedding.clone();
        let mut keys = image_embedding;

        for layer in self.layers.iter() {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, &image_pe)?
        }

        let q = (&queries + point_embedding)?;
        let k = (&keys + image_pe)?;
        let attn_out = self.final_attn_token_to_image.forward(&q, &k, &keys)?;
        let queries = (queries + attn_out)?.apply(&self.norm_final_attn)?;

        Ok((queries, keys))
    }
}
