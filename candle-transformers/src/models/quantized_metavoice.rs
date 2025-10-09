//! Quantized MetaVoice model implementation.
//!
//! MetaVoice is a conditional text-to-speech model based on a transformer architecture.
//! This implementation provides quantization for reduced memory and compute.
//!
//! Key characteristics:
//! - Transformer-based autoregressive decoder
//! - Speaker conditioning
//! - Support for 8-bit quantization
//! - Key-value caching for efficient inference
//! - RMS normalization layers
//!
//! References:
//! - [MetaVoice Code](https://github.com/metavoiceio/metavoice)
//!

use crate::quantized_nn::{linear_b, Embedding, Linear, RmsNorm};
pub use crate::quantized_var_builder::VarBuilder;

use crate::models::metavoice::repeat_interleave;
use candle::{Module, Result, Tensor, D};

pub mod transformer {
    use candle::quantized::QuantizedBackend;

    use super::*;

    type Config = crate::models::metavoice::transformer::Config;

    #[derive(Debug, Clone)]
    struct FeedForward<QB: QuantizedBackend> {
        w1: Linear<QB>,
        w2: Linear<QB>,
        w3: Linear<QB>,
        span: tracing::Span,
    }

    impl<QB: QuantizedBackend> FeedForward<QB> {
        fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
            let i_size = cfg.intermediate_size();
            let w1 = linear_b(cfg.dim, i_size, false, vb.pp("swiglu.w1"))?;
            let w2 = linear_b(i_size, cfg.dim, false, vb.pp("w2"))?;
            let w3 = linear_b(cfg.dim, i_size, false, vb.pp("swiglu.w3"))?;
            Ok(Self {
                w1,
                w2,
                w3,
                span: tracing::span!(tracing::Level::TRACE, "feed-forward"),
            })
        }
    }

    impl<QB: QuantizedBackend> Module<QB::Storage> for FeedForward<QB>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
            let _enter = self.span.enter();
            let swiglu = (candle_nn::ops::silu(&xs.apply(&self.w1)?)? * xs.apply(&self.w3))?;
            swiglu.apply(&self.w2)
        }
    }

    type KVCache<QB> = (
        Tensor<<QB as QuantizedBackend>::Storage>,
        Tensor<<QB as QuantizedBackend>::Storage>,
    );
    #[derive(Debug, Clone)]
    struct Attention<QB: QuantizedBackend> {
        wqkv: Linear<QB>,
        wo: Linear<QB>,
        dim: usize,
        kv_size: usize,
        n_local_heads: usize,
        head_dim: usize,
        n_head: usize,
        kv_cache: Option<KVCache<QB>>,
        span: tracing::Span,
    }

    impl<QB: QuantizedBackend> Attention<QB> {
        fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
            let n_local_heads = cfg.n_local_heads();
            let head_dim = cfg.head_dim();
            let total_head_dim = (cfg.n_head + 2 * n_local_heads) * head_dim;
            let wqkv = linear_b(cfg.dim, total_head_dim, false, vb.pp("wqkv"))?;
            let wo = linear_b(cfg.dim, cfg.dim, false, vb.pp("wo"))?;
            Ok(Self {
                wqkv,
                wo,
                dim: cfg.dim,
                kv_size: n_local_heads * head_dim,
                n_local_heads,
                head_dim,
                n_head: cfg.n_head,
                kv_cache: None,
                span: tracing::span!(tracing::Level::TRACE, "attention"),
            })
        }

        fn forward(
            &mut self,
            xs: &Tensor<QB::Storage>,
            _pos: usize,
            mask: &Tensor<QB::Storage>,
        ) -> Result<Tensor<QB::Storage>>
        where
            Linear<QB>: Module<QB::Storage>,
        {
            let _enter = self.span.enter();
            let (b_sz, seqlen, _) = xs.dims3()?;

            let qkv = xs.apply(&self.wqkv)?;
            let q = qkv.narrow(D::Minus1, 0, self.dim)?;
            let k = qkv.narrow(D::Minus1, self.dim, self.kv_size)?;
            let v = qkv.narrow(D::Minus1, self.dim + self.kv_size, self.kv_size)?;
            let q = q
                .reshape((b_sz, seqlen, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b_sz, seqlen, self.n_local_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seqlen, self.n_local_heads, self.head_dim))?
                .transpose(1, 2)?;

            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k, &k], 2)?;
                    let v = Tensor::cat(&[prev_v, &v], 2)?;
                    (k, v)
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));

            let k = repeat_interleave(&k, self.n_head / self.n_local_heads, 1)?;
            let v = repeat_interleave(&v, self.n_head / self.n_local_heads, 1)?;

            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = attn_weights.broadcast_add(mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, seqlen, self.dim))?
                .apply(&self.wo)
        }

        fn clear_kv_cache(&mut self) {
            self.kv_cache = None
        }
    }

    #[derive(Debug, Clone)]
    struct Block<QB: QuantizedBackend> {
        attention: Attention<QB>,
        feed_forward: FeedForward<QB>,
        ffn_norm: RmsNorm<QB>,
        attention_norm: RmsNorm<QB>,
        span: tracing::Span,
    }

    impl<QB: QuantizedBackend> Block<QB> {
        fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
            let attention = Attention::new(cfg, vb.pp("attention"))?;
            let feed_forward = FeedForward::new(cfg, vb.pp("feed_forward"))?;
            let ffn_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("ffn_norm"))?;
            let attention_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("attention_norm"))?;
            Ok(Self {
                attention,
                feed_forward,
                ffn_norm,
                attention_norm,
                span: tracing::span!(tracing::Level::TRACE, "block"),
            })
        }

        fn forward(
            &mut self,
            xs: &Tensor<QB::Storage>,
            pos: usize,
            mask: &Tensor<QB::Storage>,
        ) -> Result<Tensor<QB::Storage>>
        where
            Linear<QB>: Module<QB::Storage>,
        {
            let _enter = self.span.enter();
            let hs = xs.apply(&self.attention_norm)?;
            let hs = (xs + self.attention.forward(&hs, pos, mask))?;
            &hs + hs.apply(&self.ffn_norm)?.apply(&self.feed_forward)
        }

        fn clear_kv_cache(&mut self) {
            self.attention.clear_kv_cache()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Model<QB: QuantizedBackend> {
        tok_embeddings: Embedding<QB>,
        pos_embeddings: Embedding<QB>,
        speaker_cond_pos: Linear<QB>,
        layers: Vec<Block<QB>>,
        norm: RmsNorm<QB>,
        output: Linear<QB>,
        spk_cond_mask: Tensor<QB::Storage>,
        span: tracing::Span,
    }

    impl<QB: QuantizedBackend> Model<QB> {
        pub fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
            let tok_embeddings = Embedding::new(cfg.vocab_size, cfg.dim, vb.pp("tok_embeddings"))?;
            let pos_embeddings = Embedding::new(cfg.block_size, cfg.dim, vb.pp("pos_embeddings"))?;
            let speaker_cond_pos = linear_b(
                cfg.speaker_emb_dim,
                cfg.dim,
                false,
                vb.pp("speaker_cond_pos"),
            )?;
            let mut layers = Vec::with_capacity(cfg.n_layer);
            let vb_l = vb.pp("layers");
            for layer_idx in 0..cfg.n_layer {
                let layer = Block::new(cfg, vb_l.pp(layer_idx))?;
                layers.push(layer)
            }
            let norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("norm"))?;
            let output = linear_b(cfg.dim, cfg.vocab_size, false, vb.pp("output"))?;
            let spk_cond_mask = Tensor::cat(
                &[
                    Tensor::ones((1, 1, cfg.dim), candle::DType::F32, vb.device())?,
                    Tensor::zeros((1, 1, cfg.dim), candle::DType::F32, vb.device())?,
                ],
                0,
            )?;
            Ok(Self {
                tok_embeddings,
                pos_embeddings,
                speaker_cond_pos,
                layers,
                norm,
                output,
                spk_cond_mask,
                span: tracing::span!(tracing::Level::TRACE, "qtransformer"),
            })
        }

        pub fn clear_kv_cache(&mut self) {
            for layer in self.layers.iter_mut() {
                layer.clear_kv_cache()
            }
        }

        pub fn forward(
            &mut self,
            xs: &Tensor<QB::Storage>,
            spk_emb: &Tensor<QB::Storage>,
            pos: usize,
        ) -> Result<Tensor<QB::Storage>>
        where
            Linear<QB>: Module<QB::Storage>,
        {
            let _enter = self.span.enter();
            let (_b_sz, seqlen) = xs.dims2()?;
            let mask: Vec<_> = (0..seqlen)
                .flat_map(|i| (0..seqlen).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
                .collect();
            let mask = Tensor::from_slice(&mask, (1, 1, seqlen, seqlen), xs.device())?;
            let input_pos = Tensor::arange(pos as u32, (pos + seqlen) as u32, xs.device())?;
            let tok_embeddings = xs.apply(&self.tok_embeddings)?;
            let pos_embeddings = input_pos.apply(&self.pos_embeddings)?;
            let mut xs = tok_embeddings
                .broadcast_add(&pos_embeddings)?
                .broadcast_add(
                    &spk_emb
                        .apply(&self.speaker_cond_pos)?
                        .broadcast_mul(&self.spk_cond_mask)?,
                )?;
            let mask = mask.to_dtype(xs.dtype())?;
            for layer in self.layers.iter_mut() {
                xs = layer.forward(&xs, pos, &mask)?
            }
            xs.narrow(1, seqlen - 1, 1)?
                .contiguous()?
                .apply(&self.norm)?
                .apply(&self.output)
        }
    }
}
