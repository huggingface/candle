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
    use_flash_attn: bool,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl Attention {
    pub fn new(
        query_dim: usize,
        heads: usize,
        dim_head: usize,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
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
            use_flash_attn,
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

        let xs = if self.use_flash_attn {
            let init_dtype = query.dtype();
            let q = query
                .to_dtype(candle::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let k = key
                .to_dtype(candle::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let v = value
                .to_dtype(candle::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            flash_attn(&q, &k, &v, self.scale as f32, false)?
                .transpose(1, 2)?
                .squeeze(0)?
                .to_dtype(init_dtype)?
        } else {
            let attn_prs = self.get_attention_scores(&query, &key)?;
            attn_prs.matmul(&value)?
        };
        let xs = self.batch_to_head_dim(&xs)?;

        self.to_out
            .forward(&xs)?
            .t()?
            .reshape((b_size, channel, h, w))
    }
}
