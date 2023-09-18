use candle::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

// A simplified version of:
// https://github.com/huggingface/diffusers/blob/119ad2c3dc8a8fb8446a83f4bf6f20929487b47f/src/diffusers/models/attention_processor.py#L38
#[derive(Debug)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
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
            heads,
        })
    }

    fn batch_to_head_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((b_size / self.heads, self.heads, seq_len, dim))?
            .permute((0, 2, 1, 3))?
            .reshape((b_size / self.heads, seq_len, dim * self.heads))
    }

    fn head_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((b_size, seq_len, self.heads, dim / self.heads))?
            .permute((0, 2, 1, 3))?
            .reshape((b_size * self.heads, seq_len, dim / self.heads))
    }

    fn get_attention_scores(&self, query: &Tensor, key: &Tensor) -> Result<Tensor> {
        let attn_probs = (query.matmul(&key.t()?)? * self.scale)?;
        candle_nn::ops::softmax_last_dim(&attn_probs)
    }

    pub fn forward(&self, xs: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let (b_size, channel, h, w) = xs.dims4()?;
        let xs = xs.reshape((b_size, channel, h * w))?.t()?;

        let query = self.to_q.forward(&xs)?;
        let key = self.to_k.forward(encoder_hidden_states)?;
        let value = self.to_v.forward(encoder_hidden_states)?;

        let query = self.head_to_batch_dim(&query)?;
        let key = self.head_to_batch_dim(&key)?;
        let value = self.head_to_batch_dim(&value)?;

        let attn_prs = self.get_attention_scores(&query, &key)?;
        let xs = attn_prs.matmul(&value)?;
        let xs = self.batch_to_head_dim(&xs)?;

        self.to_out
            .forward(&xs)?
            .t()?
            .reshape((b_size, channel, h, w))
    }
}
