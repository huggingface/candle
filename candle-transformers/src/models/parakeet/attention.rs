use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{ops::softmax_last_dim, Linear, Module, VarBuilder};

use crate::models::parakeet::cache::CacheLike;

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_head: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn load(n_head: usize, n_feat: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let q_proj = if bias {
            candle_nn::linear(n_feat, n_feat, vb.pp("linear_q"))?
        } else {
            candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_q"))?
        };
        let k_proj = if bias {
            candle_nn::linear(n_feat, n_feat, vb.pp("linear_k"))?
        } else {
            candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_k"))?
        };
        let v_proj = if bias {
            candle_nn::linear(n_feat, n_feat, vb.pp("linear_v"))?
        } else {
            candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_v"))?
        };
        let out_proj = if bias {
            candle_nn::linear(n_feat, n_feat, vb.pp("linear_out"))?
        } else {
            candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_out"))?
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_head,
            head_dim: n_feat / n_head,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut dyn CacheLike>,
    ) -> Result<Tensor> {
        let q = self.q_proj.forward(q)?;
        let k = self.k_proj.forward(k)?;
        let v = self.v_proj.forward(v)?;
        let (b, tq, _) = q.dims3()?;
        let (_, tk, _) = k.dims3()?;

        let q = q
            .reshape((b, tq, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((b, tk, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b, tk, self.n_head, self.head_dim))?
            .transpose(1, 2)?;

        if let Some(cache) = cache {
            let (k_cached, v_cached) = cache.update_and_fetch_kv(k, v)?;
            k = k_cached;
            v = v_cached;
        }

        let scale = (self.head_dim as f64).powf(-0.5);
        let q = (&q * scale)?;
        let k_t = k.transpose(2, 3)?;
        let mut scores = q.matmul(&k_t)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let attn = softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, tq, self.n_head * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

#[derive(Debug, Clone)]
pub struct RelPositionMultiHeadAttention {
    inner: MultiHeadAttention,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPositionMultiHeadAttention {
    pub fn load(n_head: usize, n_feat: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let inner = MultiHeadAttention::load(n_head, n_feat, bias, vb.clone())?;
        let linear_pos = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?;
        let pos_bias_u = vb.get((n_head, n_feat / n_head), "pos_bias_u")?;
        let pos_bias_v = vb.get((n_head, n_feat / n_head), "pos_bias_v")?;
        Ok(Self {
            inner,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
        })
    }

    fn rel_shift(x: Tensor) -> Result<Tensor> {
        let (b, h, tq, pos_len) = x.dims4()?;
        let x = x.pad_with_zeros(D::Minus1, 1, 0)?;
        let x = x.reshape((b, h, pos_len + 1, tq))?;
        let x = x.narrow(2, 1, pos_len)?;
        x.reshape((b, h, tq, pos_len))
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        pos_emb: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut dyn CacheLike>,
    ) -> Result<Tensor> {
        let q_proj = self.inner.q_proj.forward(q)?;
        let k_proj = self.inner.k_proj.forward(k)?;
        let v_proj = self.inner.v_proj.forward(v)?;
        let p_proj = self.linear_pos.forward(pos_emb)?;

        let (b, tq, _) = q_proj.dims3()?;
        let (_, tk, _) = k_proj.dims3()?;
        let (_, pos_len, _) = p_proj.dims3()?;

        let q = q_proj.reshape((b, tq, self.inner.n_head, self.inner.head_dim))?;
        let q_u = (&q
            + &self
                .pos_bias_u
                .reshape((1, 1, self.inner.n_head, self.inner.head_dim))?)?
            .transpose(1, 2)?;
        let q_v = (&q
            + &self
                .pos_bias_v
                .reshape((1, 1, self.inner.n_head, self.inner.head_dim))?)?
            .transpose(1, 2)?;

        let mut k = k_proj
            .reshape((b, tk, self.inner.n_head, self.inner.head_dim))?
            .transpose(1, 2)?;
        let mut v = v_proj
            .reshape((b, tk, self.inner.n_head, self.inner.head_dim))?
            .transpose(1, 2)?;
        let p = p_proj
            .reshape((b, pos_len, self.inner.n_head, self.inner.head_dim))?
            .transpose(1, 2)?;

        if let Some(cache) = cache {
            let (k_cached, v_cached) = cache.update_and_fetch_kv(k, v)?;
            k = k_cached;
            v = v_cached;
        }

        let scale = (self.inner.head_dim as f64).powf(-0.5);
        let k_t = k.transpose(2, 3)?;
        let mut matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = {
            let p_t = p.transpose(2, 3)?;
            let bd = q_v.matmul(&p_t)?;
            Self::rel_shift(bd)?
        };
        let mut matrix_bd = matrix_bd;
        let k_len = k.dims4()?.2;
        if matrix_bd.dims4()?.3 > k_len {
            matrix_bd = matrix_bd.narrow(3, 0, k_len)?;
        }

        matrix_ac = (&matrix_ac * scale)?;
        matrix_bd = (&matrix_bd * scale)?;
        let mut scores = matrix_ac.broadcast_add(&matrix_bd)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }

        let attn = softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, tq, self.inner.n_head * self.inner.head_dim))?;
        self.inner.out_proj.forward(&out)
    }
}

#[derive(Debug, Clone)]
pub struct RelPositionalEncoding {
    d_model: usize,
    max_len: usize,
    scale: f64,
    pe: Tensor,
}

impl RelPositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, scale_input: bool, device: &Device) -> Result<Self> {
        let scale = if scale_input {
            (d_model as f64).sqrt()
        } else {
            1.0
        };
        let pe = Self::build_pe(d_model, max_len, device)?;
        Ok(Self {
            d_model,
            max_len,
            scale,
            pe,
        })
    }

    fn build_pe(d_model: usize, max_len: usize, device: &Device) -> Result<Tensor> {
        let len = 2 * max_len - 1;
        let mut data = vec![0f32; len * d_model];
        let mut div_term = Vec::with_capacity(d_model / 2);
        for i in 0..(d_model / 2) {
            let exp = -((10000.0f64).ln() / d_model as f64) * (2 * i) as f64;
            div_term.push(exp.exp() as f32);
        }
        for idx in 0..len {
            let position = (max_len - 1) as i32 - idx as i32;
            let pos_f = position as f32;
            for i in 0..(d_model / 2) {
                let v = pos_f * div_term[i];
                data[idx * d_model + 2 * i] = v.sin();
                data[idx * d_model + 2 * i + 1] = v.cos();
            }
        }
        Tensor::from_vec(data, (1, len, d_model), device)?.to_dtype(DType::F32)
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _) = x.dims3()?;
        let input_len = seq_len + offset;
        if input_len > self.max_len {
            self.max_len = input_len + 1;
            self.pe = Self::build_pe(self.d_model, self.max_len, x.device())?;
        }
        let x = (x * self.scale)?;
        let buffer_len = self.pe.dims3()?.1;
        let center = buffer_len / 2;
        let start_idx = center.saturating_sub(input_len - 1);
        let end_idx = (center + input_len).min(buffer_len);
        let pos_emb = self.pe.narrow(1, start_idx, end_idx - start_idx)?;
        Ok((x, pos_emb))
    }
}
