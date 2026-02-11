//! RWKV v7 model implementation.
//!
//! The [RWKV model](https://wiki.rwkv.com/) is a recurrent neural network model
//! with performance on par with transformer architectures. Several variants are
//! available, candle implements the v5, v6 and v7 versions and can be used with
//! Eagle 7B([blog post](https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers)).
//!
//! Key characteristics:
//! - Linear attention mechanism
//! - Time-mixing for temporal dependencies
//! - Group normalization
//! - Feed forward gating
//! - State recycling for efficient inference
//!
//! # Example
//!
//! ```bash
//! cargo run --example rwkv --release -- \
//!   --prompt "The smallest prime is "
//!
//! > avx: true, neon: false, simd128: false, f16c: true
//! > temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
//! > The smallest prime is ϕ(2) = 2.
//! > The smallest composite is ϕ(3) = 3.
//! > The smallest perfect number is ϕ(5) = 5.
//! > The smallest perfect square is ϕ(4) = 4.
//! > The smallest perfect cube is ϕ(6) = 6.
//! ```

use super::with_tracing::{layer_norm, linear_no_bias as linear, LayerNorm, Linear};
use candle::{Result, Tensor, D};
use candle_nn::{
    embedding, Embedding, GroupNorm, Module, VarBuilder,
};

pub use crate::models::rwkv_v5::{Config, State, Tokenizer};

#[derive(Debug, Clone)]
struct SelfAttention {
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor,
    r_k: Tensor,
    w0: Tensor,
    w1: Tensor,
    w2: Tensor,
    a0: Tensor,
    a1: Tensor,
    a2: Tensor,
    g1: Tensor,
    g2: Tensor,
    v0: Option<Tensor>,
    v1: Option<Tensor>,
    v2: Option<Tensor>,
    k_k: Tensor,
    k_a: Tensor,
    receptance: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ln_x: candle_nn::GroupNorm,
    layer_id: usize,
    n_head: usize,
    head_size: usize,
}

impl SelfAttention {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let attn_hidden_size = cfg.attention_hidden_size;

        let receptance = linear(hidden_size, attn_hidden_size, vb.pp("receptance"))?;
        let key = linear(hidden_size, attn_hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, attn_hidden_size, vb.pp("value"))?;
        let output = linear(attn_hidden_size, hidden_size, vb.pp("output"))?;

        let vb_x = vb.pp("ln_x");
        let ln_x_weight = vb_x.get(hidden_size, "weight")?;
        let ln_x_bias = vb_x.get(hidden_size, "bias")?;

        let n_head = cfg.hidden_size / cfg.head_size;
        let head_size = cfg.head_size;
        let ln_x = GroupNorm::new(ln_x_weight, ln_x_bias, hidden_size, n_head, 64e-5)?;

        let x_r = vb.get((1, 1, cfg.hidden_size), "x_r")?;
        let x_w = vb.get((1, 1, cfg.hidden_size), "x_w")?;
        let x_k = vb.get((1, 1, cfg.hidden_size), "x_k")?;
        let x_v = vb.get((1, 1, cfg.hidden_size), "x_v")?;
        let x_a = vb.get((1, 1, cfg.hidden_size), "x_a")?;
        let x_g = vb.get((1, 1, cfg.hidden_size), "x_g")?;
        let r_k = vb.get((n_head, head_size), "r_k")?;
        let w0 = vb.get((1, 1, cfg.hidden_size), "w0")?;
        let w1 = vb.get((cfg.hidden_size, 64), "w1")?;
        let w2 = vb.get((64, cfg.hidden_size), "w2")?;
        let a0 = vb.get((1, 1, cfg.hidden_size), "a0")?;
        let a1 = vb.get((cfg.hidden_size, 64), "a1")?;
        let a2 = vb.get((64, cfg.hidden_size), "a2")?;
        let g1 = vb.get((cfg.hidden_size, 128), "g1")?;
        let g2 = vb.get((128, cfg.hidden_size), "g2")?;

        let v0 = if layer_id == 0 {
            None
        } else {
            Some(vb.get((1, 1, cfg.hidden_size), "v0")?)
        };
        let v1 = if layer_id == 0 {
            None
        } else {
            Some(vb.get((cfg.hidden_size, 32), "v1")?)
        };
        let v2 = if layer_id == 0 {
            None
        } else {
            Some(vb.get((32, cfg.hidden_size), "v2")?)
        };

        let k_k = vb.get((1, 1, cfg.hidden_size), "k_k")?;
        let k_a = vb.get((1, 1, cfg.hidden_size), "k_a")?;

        Ok(Self {
            key,
            value,
            receptance,
            output,
            ln_x,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            r_k,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            g1,
            g2,
            v0,
            v1,
            v2,
            k_k,
            k_a,
            layer_id,
            n_head,
            head_size,
        })
    }

    fn normalize_l2(&self, kk: &Tensor) -> Result<Tensor> {
        let eps = 1e-12; // Standard PyTorch epsilon for F.normalize

        // 1. Compute the L2 norm: sqrt(sum(x^2)) along the last dimension (-1)
        // keepdim=true is essential to allow broadcasting during division
        let norm = kk.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;

        // 2. Clamp to a minimum value (epsilon) to avoid division by zero
        let norm = norm.maximum(&Tensor::new(&[[[eps]]], kk.device())?.to_dtype(kk.dtype())?)?;

        // 3. Divide the original tensor by its norm
        kk.broadcast_div(&norm)
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        v_first: Option<&Tensor>,
        state: &mut State,
    ) -> Result<(Tensor, Tensor)> {
        let shifted = state.per_layer[self.layer_id].extract_key_value.clone();
        let shifted = if shifted.rank() == 2 {
            shifted.unsqueeze(1)?
        } else {
            shifted
        };
        state.per_layer[self.layer_id].extract_key_value = xs.clone();

        let xx = (&shifted - xs)?;

        let (xr, xw, xk, xv, xa, xg) = (
            (xs + (&xx * &self.x_r)?)?,
            (xs + (&xx * &self.x_w)?)?,
            (xs + (&xx * &self.x_k)?)?,
            (xs + (&xx * &self.x_v)?)?,
            (xs + (&xx * &self.x_a)?)?,
            (xs + (&xx * &self.x_g)?)?,
        );
        let r = self.receptance.forward(&xr)?;
        let w = (&self.w0
            + xw.broadcast_matmul(&self.w1)?
                .tanh()?
                .broadcast_matmul(&self.w2)?)?;
        let w = (candle_nn::ops::sigmoid(&w)? * -0.606531)?.exp()?;
        let k = self.key.forward(&xk)?;
        let v = self.value.forward(&xv)?;

        let (v, v_first_curr) = if self.layer_id == 0 {
            let v_first = v.clone();
            (v, v_first)
        } else {
            let v_first = v_first.unwrap();
            let v0 = &self.v0.clone().unwrap();
            let v1 = &self.v1.clone().unwrap();
            let v2 = &self.v2.clone().unwrap();
            let v_new = (&v
                + (v_first - &v)?
                    * candle_nn::ops::sigmoid(
                        &(v0 + &xv.broadcast_matmul(v1)?.broadcast_matmul(v2)?)?,
                    )?)?;
            (v_new, v_first.clone())
        };

        let a = candle_nn::ops::sigmoid(
            &(&self.a0 + xa.broadcast_matmul(&self.a1)?.broadcast_matmul(&self.a2)?)?,
        )?;
        let g =
            candle_nn::ops::sigmoid(&xg.broadcast_matmul(&self.g1)?)?.broadcast_matmul(&self.g2)?;

        let kk = (&k * &self.k_k)?;

        // kk = F.normalize(kk, dim=-1, p=2.0)
        let kk = self.normalize_l2(&kk)?;

        let k = (k * (1.0 + (((&a - 1.0)?) * &self.k_a)?)?)?;

        let vk = v
            .reshape((self.n_head, self.head_size, ()))?
            .matmul(&k.reshape((self.n_head, (), self.head_size))?)?;
        let ab = kk
            .neg()?
            .reshape((self.n_head, self.head_size, ()))?
            .matmul(&(kk * a)?.reshape((self.n_head, (), self.head_size))?)?;

        let st = &state.per_layer[self.layer_id].linear_attention.squeeze(0)?;

        let w = w.reshape((self.n_head, (), self.head_size))?;

        let st = (st.broadcast_mul(&w)? + (st.matmul(&ab))? + vk)?;
        state.per_layer[self.layer_id].linear_attention = st.clone();
        let xs = st.matmul(&r.reshape((self.n_head, self.head_size, ()))?)?;

        let xs = xs.reshape((1, (), 1))?;

        let xs = self.ln_x.forward(&xs)?.reshape((1, 1, ()))?;

        let r = r.reshape(((), self.n_head, self.head_size))?;
        let k = k.reshape(((), self.n_head, self.head_size))?;
        let v = v.reshape(((), self.n_head, self.head_size))?;
        let rkv = (&r * &k.broadcast_mul(&self.r_k)?)?
            .sum_keepdim(D::Minus1)?
            .broadcast_mul(&v)?;
        let xs = ((xs + rkv.reshape((1, 1, ()))?)? * &g)?;

        let xs = self.output.forward(&xs)?;
        Ok((xs, v_first_curr))
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    x_k: Tensor,
    key: Linear,
    value: Linear,
    layer_id: usize,
}

impl FeedForward {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let x_k = vb.get((1, 1, cfg.hidden_size), "x_k")?;
        let key = linear(cfg.hidden_size, cfg.hidden_size * 4, vb.pp("key"))?;
        let value = linear(cfg.hidden_size * 4, cfg.hidden_size, vb.pp("value"))?;
        Ok(Self {
            x_k,
            key,
            value,
            layer_id,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let shifted = state.per_layer[self.layer_id].feed_forward.clone();
        let shifted = if shifted.rank() == 2 {
            shifted.unsqueeze(1)?
        } else {
            shifted
        };
        let xx = (&shifted - xs)?;
        state.per_layer[self.layer_id].feed_forward = xs.clone();
        let xk = (xs + &xx * &self.x_k)?;
        let k = self.key.forward(&xk)?.relu()?.powf(2.0)?;
        let out = self.value.forward(&k)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct Block {
    pre_ln: Option<LayerNorm>,
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl Block {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln1 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln2"))?;
        let pre_ln = if layer_id == 0 {
            let ln = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("pre_ln"))?;
            Some(ln)
        } else {
            None
        };
        let attention = SelfAttention::new(layer_id, cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(layer_id, cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        v_first: Option<&Tensor>,
        state: &mut State,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let xs = match self.pre_ln.as_ref() {
            None => xs.clone(),
            Some(pre_ln) => xs.apply(pre_ln)?,
        };
        let (attention, v_first_out) =
            self.attention
                .forward(&xs.apply(&self.ln1)?, v_first, state)?;
        let xs = (xs + attention)?;
        let feed_forward = self.feed_forward.forward(&xs.apply(&self.ln2)?, state)?;
        let xs = (xs + feed_forward)?;
        Ok((xs, Some(v_first_out)))
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head: Linear,
    rescale_every: usize,
    layers_are_rescaled: bool,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("rwkv");
        let embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embeddings"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_m.pp("blocks");
        for block_index in 0..cfg.num_hidden_layers {
            let block = Block::new(block_index, cfg, vb_b.pp(block_index))?;
            blocks.push(block)
        }
        let ln_out = layer_norm(cfg.hidden_size, 1e-5, vb_m.pp("ln_out"))?;
        let head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            blocks,
            ln_out,
            head,
            rescale_every: cfg.rescale_every,
            layers_are_rescaled: false, // This seem to only happen for the f16/bf16 dtypes.
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let mut xs = xs.apply(&self.embeddings)?;
        let mut v_first: Option<Tensor> = None;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            (xs, v_first) = block.forward(&xs, v_first.as_ref(), state)?;
            if self.layers_are_rescaled && (block_idx + 1) % self.rescale_every == 0 {
                xs = (xs / 2.)?
            }
        }
        let xs = xs.apply(&self.ln_out)?.apply(&self.head)?;
        state.pos += 1;
        Ok(xs)
    }
}
