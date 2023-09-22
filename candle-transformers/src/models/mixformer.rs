#![allow(unused)]
/// MixFormer model.
/// https://huggingface.co/microsoft/phi-1_5
/// https://arxiv.org/abs/2309.05463
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};

// https://huggingface.co/microsoft/phi-1_5/blob/main/configuration_mixformer_sequential.py
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    n_inner: Option<usize>,
    n_head: usize,
    rotary_dim: usize,
    activation_function: Activation,
    layer_norm_epsilon: f64,
    tie_word_embeddings: bool,
    pad_vocab_size_multiple: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50304,
            n_positions: 2048,
            n_embd: 1024,
            n_layer: 20,
            n_inner: None,
            n_head: 16,
            rotary_dim: usize::min(32, 1024 / 16),
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
            tie_word_embeddings: false,
            pad_vocab_size_multiple: 64,
        }
    }
}

#[derive(Debug)]
struct Embedding {
    wte: candle_nn::Embedding,
}

impl Embedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        Ok(Self { wte })
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.wte.forward(xs)
    }
}

#[derive(Debug)]
struct RotaryEmbedding {}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    act: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_inner = cfg.n_inner.unwrap_or(4 * cfg.n_embd);
        let fc1 = candle_nn::linear(cfg.n_embd, n_inner, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(n_inner, cfg.n_embd, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: cfg.activation_function,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

#[derive(Debug)]
struct SelfAttention {
    causal: bool,
    softmax_scale: f64,
}

#[derive(Debug)]
struct CrossAttention {
    causal: bool,
    softmax_scale: f64,
}

#[derive(Debug)]
struct CausalLMHead {
    ln: candle_nn::LayerNorm,
    linear: candle_nn::Linear,
}

impl CausalLMHead {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln"))?;
        let linear = candle_nn::linear(cfg.n_embd, cfg.vocab_size, vb.pp("linear"))?;
        Ok(Self { ln, linear })
    }
}

impl Module for CausalLMHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.ln)?
            .apply(&self.linear)?
            .to_dtype(DType::F32)
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MHA {
    wqkv: candle_nn::Linear,
    out_proj: candle_nn::Linear,
}

impl MHA {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let op_size = cfg.n_embd;
        let wqkv = candle_nn::linear(cfg.n_embd, 3 * op_size, vb.pp("Wqkv"))?;
        let out_proj = candle_nn::linear(op_size, cfg.n_embd, vb.pp("out_proj"))?;
        Ok(Self { wqkv, out_proj })
    }
}

impl Module for MHA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug)]
struct ParallelBlock {
    ln: candle_nn::LayerNorm,
    mixer: MHA,
    mlp: MLP,
}

impl ParallelBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln"))?;
        let mixer = MHA::new(cfg, vb.pp("mixer"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self { ln, mixer, mlp })
    }
}

impl Module for ParallelBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.ln)?;
        let attn_outputs = self.mixer.forward(&xs)?;
        let feed_forward_hidden_states = self.mlp.forward(&xs)?;
        attn_outputs + feed_forward_hidden_states + residual
    }
}

#[derive(Debug)]
pub struct MixFormerSequentialForCausalLM {
    embedding: Embedding,
    blocks: Vec<ParallelBlock>,
    head: CausalLMHead,
}

impl MixFormerSequentialForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

impl Module for MixFormerSequentialForCausalLM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
