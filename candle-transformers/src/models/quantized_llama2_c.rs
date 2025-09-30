//! Quantized Llama2 model implementation.
//!
//! This provides an 8-bit quantized implementation of Meta's LLaMA2 language model
//! for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Decoder-only transformer architecture
//! - RoPE position embeddings
//! - Grouped Query Attention
//! - 8-bit quantization of weights
//!
//! References:
//! - [LLaMA2 Paper](https://arxiv.org/abs/2307.09288)
//! - [LLaMA2 Technical Report](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
//!

use super::llama2_c::{Cache, Config};
use crate::quantized_nn::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
pub use crate::quantized_var_builder::VarBuilder;
use candle::{quantized::QuantizedBackend, DType, IndexOp, Module, Result, Tensor, D};

fn silu<QB: QuantizedBackend>(xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

#[derive(Debug, Clone)]
struct CausalSelfAttention<QB: QuantizedBackend> {
    q_proj: Linear<QB>,
    k_proj: Linear<QB>,
    v_proj: Linear<QB>,
    o_proj: Linear<QB>,
    n_head: usize,
    n_key_value_head: usize,
    head_dim: usize,
}

impl<QB: QuantizedBackend> CausalSelfAttention<QB> {
    fn apply_rotary_emb(
        &self,
        x: &Tensor<QB::Storage>,
        index_pos: usize,
        cache: &Cache<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>> {
        let (b_sz, seq_len, h, n_embd) = x.dims4()?;
        let cos = cache.cos.i(index_pos..index_pos + seq_len)?;
        let sin = cache.sin.i(index_pos..index_pos + seq_len)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let cos = cos.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let sin = sin.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let x = x.reshape((b_sz, seq_len, h, n_embd / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let dst0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let dst1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        let rope = Tensor::cat(&[&dst0, &dst1], D::Minus1)?.reshape((b_sz, seq_len, h, n_embd))?;
        Ok(rope)
    }

    fn forward(
        &self,
        x: &Tensor<QB::Storage>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_key_value_head, self.head_dim))?;
        let mut v = v.reshape((b_sz, seq_len, self.n_key_value_head, self.head_dim))?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 1)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 1)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len <= 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            masked_fill::<QB>(&att, &mask, f32::NEG_INFINITY)?
        };
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let n_rep = self.n_head / self.n_key_value_head;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, seq_len, n_kv_head, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(3)?
                .expand((b_sz, seq_len, n_kv_head, n_rep, head_dim))?
                .reshape((b_sz, seq_len, n_kv_head * n_rep, head_dim))?;
            Ok(x)
        }
    }

    fn load(vb: VarBuilder<QB>, cfg: &Config) -> Result<Self> {
        let size_in = cfg.dim;
        let size_q = (cfg.dim / cfg.n_heads) * cfg.n_heads;
        let size_kv = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_head: cfg.n_heads,
            n_key_value_head: cfg.n_kv_heads,
            head_dim: cfg.dim / cfg.n_heads,
        })
    }
}

fn masked_fill<QB: QuantizedBackend>(
    on_false: &Tensor<QB::Storage>,
    mask: &Tensor<QB::Storage>,
    on_true: f32,
) -> Result<Tensor<QB::Storage>> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp<QB: QuantizedBackend> {
    c_fc1: Linear<QB>,
    c_fc2: Linear<QB>,
    c_proj: Linear<QB>,
}

impl<QB: QuantizedBackend> Mlp<QB> {
    fn new(c_fc1: Linear<QB>, c_fc2: Linear<QB>, c_proj: Linear<QB>) -> Self {
        Self {
            c_fc1,
            c_fc2,
            c_proj,
        }
    }

    fn forward(&self, x: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let x = (silu::<QB>(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder<QB>, cfg: &Config) -> Result<Self> {
        let h_size = cfg.dim;
        let i_size = cfg.hidden_dim;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

#[derive(Debug, Clone)]
struct Block<QB: QuantizedBackend> {
    rms_1: RmsNorm<QB>,
    attn: CausalSelfAttention<QB>,
    rms_2: RmsNorm<QB>,
    mlp: Mlp<QB>,
}

impl<QB: QuantizedBackend> Block<QB> {
    fn new(
        rms_1: RmsNorm<QB>,
        attn: CausalSelfAttention<QB>,
        rms_2: RmsNorm<QB>,
        mlp: Mlp<QB>,
    ) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(
        &self,
        x: &Tensor<QB::Storage>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder<QB>, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let input_layernorm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct QLlama<QB: QuantizedBackend> {
    wte: Embedding<QB>,
    blocks: Vec<Block<QB>>,
    ln_f: RmsNorm<QB>,
    lm_head: Linear<QB>,
    pub config: Config,
}

impl<QB: QuantizedBackend> QLlama<QB> {
    pub fn forward(
        &self,
        x: &Tensor<QB::Storage>,
        index_pos: usize,
        cache: &mut Cache<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let (_b_sz, _seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder<QB>, cfg: Config) -> Result<Self> {
        let wte = Embedding::new(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.dim, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.n_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), &cfg).unwrap())
            .collect();
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            config: cfg,
        })
    }
}
