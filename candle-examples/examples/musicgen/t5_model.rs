// T5 Text Encoder
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

use crate::nn::{embedding, linear, Dropout, Embedding, HiddenAct, Linear, VarBuilder};
use anyhow::Result;
use candle::{DType, Tensor, D};
use candle_nn::Module;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    vocab_size: usize,
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    num_layers: usize,
    num_decoder_layers: Option<usize>,
    num_heads: usize,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    dropout_rate: f64,
    layer_norm_epsilon: f64,
    initializer_factor: f64,
    feed_forward_proj: HiddenAct,
    is_decoder: bool,
    is_encoder_decoder: bool,
    use_cache: bool,
    pad_token_id: usize,
    eos_token_id: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: HiddenAct::Relu,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
        }
    }
}

impl Config {
    // https://huggingface.co/facebook/musicgen-small/blob/495da4ad086b3416a27c6187f9239f9fd96f3962/config.json#L184
    pub fn musicgen_small() -> Self {
        Self {
            d_ff: 3072,
            d_kv: 64,
            d_model: 768,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: HiddenAct::Relu,
            initializer_factor: 1.0,
            is_decoder: false,
            is_encoder_decoder: true,
            layer_norm_epsilon: 1e-6,
            num_decoder_layers: Some(12),
            num_heads: 12,
            num_layers: 12,
            pad_token_id: 0,
            relative_attention_max_distance: 128,
            relative_attention_num_buckets: 32,
            use_cache: true,
            vocab_size: 32128,
        }
    }
}

#[derive(Debug)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let xs2_f32 = (&xs_f32 * &xs_f32)?;
        let sum_xs2_f32 = xs2_f32.sum_keepdim(D::Minus1)?;
        let variance = xs2_f32.broadcast_div(&sum_xs2_f32)?;
        let xs = (xs / (variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5DenseActDense {
    wi: Linear,
    wo: Linear,
    dropout: Dropout,
    act: HiddenAct,
}

impl T5DenseActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi = linear(cfg.d_model, cfg.d_ff, false, vb.pp("wi"))?;
        let wo = linear(cfg.d_ff, cfg.d_model, false, vb.pp("wo"))?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            wi,
            wo,
            dropout,
            act: HiddenAct::Relu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.wi.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.dropout.forward(&xs)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5LayerFF {
    dense_relu_dense: T5DenseActDense,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5LayerFF {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        // is_gated_act is not supported.
        let dense_relu_dense = T5DenseActDense::load(vb.pp("DenseReluDense"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            dense_relu_dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.layer_norm.forward(xs)?;
        let ys = self.dense_relu_dense.forward(&ys)?;
        let xs = (xs + self.dropout.forward(&ys)?)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
}

impl T5Attention {
    fn load(h: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = linear(cfg.d_model, inner_dim, false, vb.pp("q"))?;
        let k = linear(cfg.d_model, inner_dim, false, vb.pp("k"))?;
        let v = linear(cfg.d_model, inner_dim, false, vb.pp("v"))?;
        let o = linear(inner_dim, cfg.d_model, false, vb.pp("o"))?;
        let relative_attention_bias = if h {
            let emb = embedding(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // TODO: Apply the mask(s)?
        // TODO: kv caching.
        let (b_sz, seq_len) = (xs.dim(0)?, xs.dim(1)?);
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let scores = q.matmul(&k.t()?)?;
        // TODO: position_bias_masked
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok(attn_output)
    }
}

#[derive(Debug)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5LayerSelfAttention {
    fn load(h: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, vb.pp("SelfAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            self_attention,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let normed_xs = self.layer_norm.forward(xs)?;
        let ys = self.self_attention.forward(&normed_xs)?;
        let ys = (xs + ys)?;
        Ok(ys)
    }
}

#[derive(Debug)]
struct T5LayerCrossAttention {}

impl T5LayerCrossAttention {
    fn load(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        todo!()
    }

    fn forward(&self, _xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(has_relative_attention_bias: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn = T5LayerSelfAttention::load(has_relative_attention_bias, vb.pp("0"), cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(&ff_i.to_string()), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.self_attn.forward(xs)?;
        // TODO: clamp for f16?
        if let Some(cross_attn) = &self.cross_attn {
            xs = cross_attn.forward(&xs)?;
            // TODO: clamp for f16?
        }
        let xs = self.ff.forward(&xs)?;
        // TODO: clamp for f16?
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5Stack {
    fn load(vb: VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, vb.pp(&format!("block.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
            dropout,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let (_b_sz, _seq_len) = input_embeds.dims2()?;

        let mut hidden_states = self.dropout.forward(&input_embeds)?;
        for block in self.block.iter() {
            hidden_states = block.forward(&hidden_states)?
        }
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

#[derive(Debug)]
pub struct T5EncoderModel {
    shared: Arc<Embedding>,
    encoder: T5Stack,
}

impl T5EncoderModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let shared = embedding(cfg.vocab_size, cfg.d_model, vb.pp("shared"))?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(vb.pp("encoder"), &shared, cfg)?;
        Ok(Self { shared, encoder })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let encoder_outputs = self.encoder.forward(input_ids)?;
        Ok(encoder_outputs)
    }
}
