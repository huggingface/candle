#![allow(unused)]
use crate::models::with_tracing::Linear;
use candle::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Config {
    pub num_layers: usize,
    pub padded_vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub kv_channels: usize,
    pub num_attention_heads: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
    pub rmsnorm: bool,
    pub apply_residual_connection_post_layernorm: bool,
    pub post_layer_norm: bool,
    pub add_bias_linear: bool,
    pub add_qkv_bias: bool,
    pub bias_dropout_fusion: bool,
    pub multi_query_attention: bool,
    pub multi_query_group_num: usize,
    pub apply_query_key_layer_scaling: bool,
    pub attention_softmax_in_fp32: bool,
    pub fp32_residual_connection: bool,
}

impl Config {
    fn glm3_6b() -> Self {
        Self {
            num_layers: 28,
            padded_vocab_size: 65024,
            hidden_size: 4096,
            ffn_hidden_size: 13696,
            kv_channels: 128,
            num_attention_heads: 32,
            seq_length: 8192,
            layernorm_epsilon: 1e-5,
            rmsnorm: true,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            add_bias_linear: false,
            add_qkv_bias: true,
            bias_dropout_fusion: true,
            multi_query_attention: true,
            multi_query_group_num: 2,
            apply_query_key_layer_scaling: true,
            attention_softmax_in_fp32: true,
            fp32_residual_connection: false,
        }
    }
}

fn linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        crate::models::with_tracing::linear(in_dim, out_dim, vb)
    } else {
        crate::models::with_tracing::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cache: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device();
        let rotary_dim = cfg.kv_channels;
        let n_elem = rotary_dim / 2;
        let inv_freq: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / 10_000f64.powf(i as f64 / n_elem as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.seq_length as u32, dev)?
            .to_dtype(dtype)?
            .reshape((cfg.seq_length, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cache = Tensor::cat(&[&freqs.cos()?, &freqs.sin()?], D::Minus1)?;
        Ok(Self { cache })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct CoreAttention {}

impl CoreAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    query_key_value: Linear,
    core_attention: CoreAttention,
    dense: Linear,
}

impl SelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let projection_size = cfg.kv_channels * cfg.num_attention_heads;
        let hidden_size_per_attention_head = projection_size / cfg.num_attention_heads;
        let qkv_hidden_size = if cfg.multi_query_attention {
            projection_size + 2 * hidden_size_per_attention_head * cfg.multi_query_group_num
        } else {
            3 * projection_size
        };
        let query_key_value = linear(
            cfg.hidden_size,
            qkv_hidden_size,
            cfg.add_bias_linear || cfg.add_qkv_bias,
            vb.pp("query_key_value"),
        )?;
        let core_attention = CoreAttention::new(cfg, vb.pp("core_attention"))?;
        let dense = linear(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense"),
        )?;
        Ok(Self {
            query_key_value,
            core_attention,
            dense,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
struct MLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense_h_to_4h = linear(
            cfg.hidden_size,
            cfg.ffn_hidden_size * 2,
            cfg.add_bias_linear,
            vb.pp("dense_h_to_4h"),
        )?;
        let dense_4h_to_h = linear(
            cfg.ffn_hidden_size * 2,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense_h_to_4h"),
        )?;
        Ok(Self {
            dense_4h_to_h,
            dense_h_to_4h,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense_h_to_4h)?
            .apply(&candle_nn::Activation::Swiglu)?
            .apply(&self.dense_4h_to_h)
    }
}

#[derive(Debug, Clone)]
struct Block {
    input_layernorm: candle_nn::LayerNorm,
    self_attention: SelfAttention,
    post_attention_layernorm: candle_nn::LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = if cfg.rmsnorm {
            candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("input_layernorm"),
            )?
            .into_inner()
        } else {
            candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("input_layernorm"),
            )?
        };
        let post_attention_layernorm = if cfg.rmsnorm {
            candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?
            .into_inner()
        } else {
            candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?
        };
        let self_attention = SelfAttention::new(cfg, vb.pp("self_attention"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    layers: Vec<Block>,
    final_layernorm: Option<candle_nn::LayerNorm>,
}

impl Transformer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Embedding {
    word_embeddings: candle_nn::Embedding,
    fp32_residual_connection: bool,
}

impl Embedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.padded_vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        Ok(Self {
            word_embeddings,
            fp32_residual_connection: cfg.fp32_residual_connection,
        })
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.word_embeddings.forward(xs)?.transpose(0, 1)?; // b,s,h -> s,b,h
        if self.fp32_residual_connection {
            xs.to_dtype(candle::DType::F32)
        } else {
            xs.contiguous()
        }
    }
}

#[derive(Debug, Clone)]
struct Model {
    embedding: Embedding,
    encoder: Transformer,
    output_layer: Linear,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("transformer");
        let embedding = Embedding::new(cfg, vb.pp("embedding"))?;
        let encoder = Transformer::new(cfg, vb.pp("encoder"))?;
        let output_layer = linear(
            cfg.hidden_size,
            cfg.padded_vocab_size,
            false,
            vb.pp("output_layer"),
        )?;
        Ok(Self {
            embedding,
            encoder,
            output_layer,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
