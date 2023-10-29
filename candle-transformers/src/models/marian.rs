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
    emb: Embedding,
}

impl SinusoidalPositionalEmbedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device();
        let dtype = vb.dtype();
        let num_positions = cfg.max_position_embeddings;
        let dim = cfg.d_model;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, num_positions as u32, dev)?
            .to_dtype(dtype)?
            .reshape((num_positions, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let weights = Tensor::cat(&[&sin, &cos], 1)?;
        let emb = Embedding::from_weights(weights)?;
        Ok(Self { emb })
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl Attention {
    fn new(cfg: &Config, is_decoder: bool, vb: VarBuilder) -> Result<Self> {
        let _attention_heads = if is_decoder {
            cfg.decoder_attention_heads
        } else {
            cfg.encoder_attention_heads
        };
        let embed_dim = cfg.d_model;
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
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
        let self_attn = Attention::new(cfg, true, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            activation_fn: cfg.activation_function,
            fc1,
            fc2,
            final_layer_norm,
        })
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
        let self_attn = Attention::new(cfg, true, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn = Attention::new(cfg, true, vb.pp("encoder_attn"))?;
        let encoder_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(cfg.d_model, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.decoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            activation_fn: cfg.activation_function,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(cfg: &Config, embed_tokens: &Embedding, vb: VarBuilder) -> Result<Self> {
        let embed_positions = SinusoidalPositionalEmbedding::new(cfg, vb.pp("embed_positions"))?;
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.encoder_layers {
            let layer = EncoderLayer::new(cfg, vb_l.pp(idx))?;
            layers.push(layer)
        }
        Ok(Self {
            embed_tokens: embed_tokens.clone(),
            embed_positions,
            layers,
        })
    }
}

#[derive(Debug, Clone)]
struct Decoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    fn new(cfg: &Config, embed_tokens: &Embedding, vb: VarBuilder) -> Result<Self> {
        let embed_positions = SinusoidalPositionalEmbedding::new(cfg, vb.pp("embed_positions"))?;
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.decoder_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(idx))?;
            layers.push(layer)
        }
        Ok(Self {
            embed_tokens: embed_tokens.clone(),
            embed_positions,
            layers,
        })
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
        let shared = Embedding::new(cfg.vocab_size, cfg.d_model, vb.pp("shared"))?;
        let encoder = Encoder::new(cfg, &shared, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, &shared, vb.pp("decoder"))?;
        Ok(Self {
            shared,
            encoder,
            decoder,
        })
    }
}

#[derive(Debug, Clone)]
struct MTModel {
    model: Model,
    lm_head: Linear,
}

impl MTModel {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let target_vocab_size = cfg.decoder_vocab_size.unwrap_or(cfg.vocab_size);
        let lm_head = linear_no_bias(cfg.d_model, target_vocab_size, vb.pp("lm_head"))?;
        let model = Model::new(cfg, vb.pp("model"))?;
        Ok(Self { model, lm_head })
    }
}
