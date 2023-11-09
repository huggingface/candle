use crate::models::vit::{Config, Embeddings, Encoder};
use candle::{Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TrOCRConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub hidden_size: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub activation_function: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub dropout: f64,
    pub attention_dropout: f64,
    pub activation_dropout: f64,
    pub decoder_start_token_id: u32,
    pub init_std: f64,
    pub decoder_layerdrop: f64,
    pub use_cache: bool,
    pub scale_embedding: bool,
    pub use_learned_position_embeddings: bool,
    pub layernorm_embedding: bool,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: u32,
    pub num_attention_heads: usize,
    pub decoder_vocab_size: Option<usize>,
}

impl Default for TrOCRConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 1024,
            hidden_size: 768,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: candle_nn::Activation::Gelu,
            max_position_embeddings: 512,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: 2,
            init_std: 0.02,
            decoder_layerdrop: 0.0,
            use_cache: true,
            scale_embedding: false,
            use_learned_position_embeddings: true,
            layernorm_embedding: true,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            num_attention_heads: 12,
            decoder_vocab_size: Some(50265),
        }
    }
}

#[derive(Debug, Clone)]
struct TrOCRLearnedPositionalEmbedding {
    offset: usize,
    weights: Embedding,
}

impl TrOCRLearnedPositionalEmbedding {
    fn load(vb: VarBuilder, cfg: &TrOCRConfig) -> Result<Self> {
        let offset: usize = 2;
        let num_embeddings = cfg.max_position_embeddings;
        let embedding_dim = cfg.d_model;
        let weights = embedding(num_embeddings + offset, embedding_dim, vb)?;

        Ok(Self { offset, weights })
    }

    fn forward(&mut self, input_ids: &Tensor, past_key_values_length: u32) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;

        let mut positions = Tensor::arange(
            past_key_values_length,
            seq_len as u32 + past_key_values_length,
            input_ids.device(),
        )?
        .expand((b_sz, seq_len))?;

        positions =
            positions.broadcast_add(&Tensor::new(self.offset as u32, input_ids.device())?)?;
        self.weights.forward(&positions)
    }
}

#[derive(Debug, Clone)]
struct TrOCRAttention {
    head_dim: usize,
    num_heads: usize,
    is_decoder: bool,
    scaling: f64,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl TrOCRAttention {
    fn load(
        vb: VarBuilder,
        cfg: &TrOCRConfig,
        kdim: Option<usize>,
        vdim: Option<usize>,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_heads = cfg.decoder_attention_heads;
        let head_dim = embed_dim / num_heads;
        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);

        let k_proj = linear_no_bias(kdim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(vdim, embed_dim, vb.pp("v_proj"))?;
        let q_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("q_proj"))?;

        let out_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            head_dim,
            num_heads,
            is_decoder: true,
            scaling: 1. / (head_dim as f64).sqrt(),
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            kv_cache: None,
        })
    }

    fn _shape(&self, tensor: &Tensor, bsz: usize) -> Result<Tensor> {
        tensor
            .reshape((bsz, (), self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (xs.apply(&self.q_proj)? * self.scaling)?;
        let (key_states, value_states) = match kv_states {
            None => {
                let key_states = self._shape(&xs.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&xs.apply(&self.v_proj)?, b_sz)?;
                if self.is_decoder {
                    let kv_states = match &self.kv_cache {
                        None => (key_states, value_states),
                        Some((p_key_states, p_value_states)) => {
                            let key_states = Tensor::cat(&[p_key_states, &key_states], 2)?;
                            let value_states = Tensor::cat(&[p_value_states, &value_states], 2)?;
                            (key_states, value_states)
                        }
                    };
                    self.kv_cache = Some(kv_states.clone());
                    kv_states
                } else {
                    (key_states, value_states)
                }
            }
            Some(kv_states) => {
                let key_states = self._shape(&kv_states.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&kv_states.apply(&self.v_proj)?, b_sz)?;
                (key_states, value_states)
            }
        };
        let proj_shape = (b_sz * self.num_heads, (), self.head_dim);
        let query_states = self._shape(&query_states, b_sz)?.reshape(proj_shape)?;
        let key_states = key_states.reshape(proj_shape)?;
        let value_states = value_states.reshape(proj_shape)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let attn_weights = match attn_mask {
            None => attn_weights,
            Some(attn_mask) => attn_weights.broadcast_add(attn_mask)?,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&value_states)?;
        attn_output
            .reshape((b_sz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.head_dim * self.num_heads))?
            .apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct TrOCRDecoderLayer {
    self_attn: TrOCRAttention,
    activation_fn: candle_nn::Activation,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: TrOCRAttention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl TrOCRDecoderLayer {
    fn load(vb: VarBuilder, cfg: &TrOCRConfig) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let self_attn = TrOCRAttention::load(vb.pp("self_attn"), cfg, None, None)?;
        let self_attn_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn = TrOCRAttention::load(
            vb.pp("encoder_attn"),
            cfg,
            Some(cfg.hidden_size),
            Some(cfg.hidden_size),
        )?;
        let encoder_attn_layer_norm =
            layer_norm(embed_dim, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear_no_bias(embed_dim, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(cfg.decoder_ffn_dim, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;
        let activation_fn = candle_nn::Activation::Gelu;

        Ok(Self {
            self_attn,
            activation_fn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.self_attn.forward(xs, None, Some(attention_mask))?;
        let xs = (xs + residual)?;
        let mut xs = self.self_attn_layer_norm.forward(&xs)?;

        if let Some(encoder_hidden_states) = &encoder_hidden_states {
            let residual = xs.clone();
            let encoder_attention_mask = attention_mask.clone(); // TODO
            xs = self.encoder_attn.forward(
                &xs,
                Some(encoder_hidden_states),
                Some(&encoder_attention_mask),
            )?;
            xs = (xs + residual)?;
            xs = self.encoder_attn_layer_norm.forward(&xs)?
        }

        let residual = xs.clone();
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        let xs = self.final_layer_norm.forward(&xs)?;

        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct TrOCRDecoder {
    layers: Vec<TrOCRDecoderLayer>,
    embed_scale: Option<f64>,
    embed_tokens: Embedding,
    embed_positions: TrOCRLearnedPositionalEmbedding,
}

impl TrOCRDecoder {
    fn new(cfg: &TrOCRConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("decoder.model.decoder");

        let embed_tokens = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;
        let embed_positions = TrOCRLearnedPositionalEmbedding::load(vb.pp("embed_positions"), cfg)?;
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.decoder_layers {
            let layer = TrOCRDecoderLayer::load(vb_l.pp(idx), cfg)?;
            layers.push(layer)
        }
        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };

        Ok(Self {
            layers,
            embed_scale,
            embed_tokens,
            embed_positions,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let embed_pos = self.embed_positions.forward(xs, past_kv_len as u32)?;
        let xs = xs.apply(&self.embed_tokens)?;

        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };

        let mut xs = xs.broadcast_add(&embed_pos)?;

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask, encoder_xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct TrOCREncoder {
    embeddings: Embeddings,
    encoder: Encoder,
    layernorm: LayerNorm,
}

impl TrOCREncoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_v = vb.pp("encoder");

        let embeddings = Embeddings::new(cfg, false, vb_v.pp("embeddings"))?;

        let encoder = Encoder::new(cfg, vb_v.pp("encoder"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb_v.pp("layernorm"))?;

        Ok(Self {
            embeddings,
            encoder,
            layernorm,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(xs, None, false)?;
        let encoder_outputs = self.encoder.forward(&embedding_output)?;

        self.layernorm.forward(&encoder_outputs)
    }
}

#[derive(Debug, Clone)]
pub struct TrOCRForCausalLM {
    decoder: TrOCRDecoder,
    output_projection: Linear,
}

impl TrOCRForCausalLM {
    pub fn new(decoder_cfg: &TrOCRConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = TrOCRDecoder::new(decoder_cfg, vb.clone())?;
        let output_projection =
            candle_nn::Linear::new(decoder.embed_tokens.embeddings().clone(), None);
        Ok(Self {
            decoder,
            output_projection,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let xs = self
            .decoder
            .forward(xs, encoder_xs, past_kv_len, attn_mask)?;
        let xs = xs.apply(&self.output_projection)?;

        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct TrOCRModel {
    encoder: TrOCREncoder,
    decoder: TrOCRForCausalLM,
}

impl TrOCRModel {
    pub fn new(encoder_cfg: &Config, decoder_cfg: &TrOCRConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = TrOCREncoder::new(encoder_cfg, vb.clone())?;
        let decoder = TrOCRForCausalLM::new(decoder_cfg, vb)?;
        Ok(Self { encoder, decoder })
    }

    pub fn encoder(&mut self) -> &mut TrOCREncoder {
        &mut self.encoder
    }

    pub fn decoder(&mut self) -> &mut TrOCRForCausalLM {
        &mut self.decoder
    }

    pub fn decode(
        &mut self,
        xs: &Tensor,
        encoder_xs: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), xs.device())?;

        self.decoder
            .forward(xs, Some(encoder_xs), past_kv_len, &mask)
    }
}
