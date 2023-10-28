#![allow(unused)]
use super::with_tracing::{linear, linear_no_bias, Embedding, Linear};
use candle::{Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub decoder_vocab_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_attention_heads: usize,
    pub use_cache: bool,
    pub is_encoder_decoder: bool,
    pub activation_function: candle_nn::Activation,
    pub d_model: usize,
    pub decoder_start_token_id: usize,
    pub scale_embedding: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub forced_eos_token_id: usize,
    pub share_encoder_decoder_embeddings: bool,
}

#[derive(Debug, Clone)]
struct SinusoidalPositionalEmbedding {
    weight: Tensor,
}

impl SinusoidalPositionalEmbedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    v_proj: Linear,
    k_proj: Linear,
    out_proj: Linear,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    activation_fn: candle_nn::Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    activation_fn: candle_nn::Activation,
    encoder_attn: Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Decoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Model {
    shared: Embedding,
    encoder: Encoder,
    decoder: Decoder,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct MTModel {
    model: Model,
    lm_head: Linear,
}

impl MTModel {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}
