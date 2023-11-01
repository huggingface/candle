use super::with_tracing::{linear, Embedding, Linear};
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
    pub decoder_start_token_id: u32,
    pub scale_embedding: bool,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub forced_eos_token_id: u32,
    pub share_encoder_decoder_embeddings: bool,
}

impl Config {
    // https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-fr-en/blob/main/config.json
    pub fn opus_mt_tc_big_fr_en() -> Self {
        Self {
            activation_function: candle_nn::Activation::Relu,
            d_model: 1024,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            decoder_layers: 6,
            decoder_start_token_id: 53016,
            decoder_vocab_size: Some(53017),
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            encoder_layers: 6,
            eos_token_id: 43311,
            forced_eos_token_id: 43311,
            is_encoder_decoder: true,
            max_position_embeddings: 1024,
            pad_token_id: 53016,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            use_cache: true,
            vocab_size: 53017,
        }
    }

    // https://huggingface.co/Helsinki-NLP/opus-mt-fr-en/blob/main/config.json
    pub fn opus_mt_fr_en() -> Self {
        Self {
            activation_function: candle_nn::Activation::Swish,
            d_model: 512,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 2048,
            decoder_layers: 6,
            decoder_start_token_id: 59513,
            decoder_vocab_size: Some(59514),
            encoder_attention_heads: 8,
            encoder_ffn_dim: 2048,
            encoder_layers: 6,
            eos_token_id: 0,
            forced_eos_token_id: 0,
            is_encoder_decoder: true,
            max_position_embeddings: 512,
            pad_token_id: 59513,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            use_cache: true,
            vocab_size: 59514,
        }
    }
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
        let weights = Tensor::cat(&[&sin, &cos], 1)?.contiguous()?;
        let emb = Embedding::from_weights(weights)?;
        Ok(Self { emb })
    }

    fn forward(&self, input_ids: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        Tensor::arange(
            past_kv_len as u32,
            (past_kv_len + seq_len) as u32,
            input_ids.device(),
        )?
        .apply(&self.emb)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scaling: f64,
    num_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    is_decoder: bool,
}

impl Attention {
    fn new(cfg: &Config, is_decoder: bool, vb: VarBuilder) -> Result<Self> {
        let num_heads = if is_decoder {
            cfg.decoder_attention_heads
        } else {
            cfg.encoder_attention_heads
        };
        let embed_dim = cfg.d_model;
        let head_dim = embed_dim / num_heads;
        let scaling = (head_dim as f64).powf(-0.5);
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scaling,
            num_heads,
            head_dim,
            kv_cache: None,
            is_decoder,
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

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None
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

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = (self.self_attn.forward(xs, None, None)? + residual)?
            .apply(&self.self_attn_layer_norm)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)?;
        (xs + residual)?.apply(&self.final_layer_norm)
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache()
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
        let encoder_attn_layer_norm =
            layer_norm(cfg.d_model, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
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

    fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = (self.self_attn.forward(xs, None, Some(attn_mask))? + residual)?
            .apply(&self.self_attn_layer_norm)?;
        let xs = match encoder_xs {
            None => xs,
            Some(encoder_xs) => {
                let residual = &xs;
                let xs = self.encoder_attn.forward(&xs, Some(encoder_xs), None)?;
                (residual + xs)?.apply(&self.encoder_attn_layer_norm)?
            }
        };
        let residual = &xs;
        let xs = xs
            .apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)?;
        let xs = (xs + residual)?.apply(&self.final_layer_norm)?;
        Ok(xs)
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
        self.encoder_attn.reset_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<EncoderLayer>,
    embed_scale: Option<f64>,
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
        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };
        Ok(Self {
            embed_tokens: embed_tokens.clone(),
            embed_positions,
            layers,
            embed_scale,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let xs = xs.apply(&self.embed_tokens)?;
        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };
        let embed_pos = self
            .embed_positions
            .forward(&xs, past_kv_len)?
            .unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&embed_pos)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }

    pub fn reset_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_kv_cache()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    embed_tokens: Embedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<DecoderLayer>,
    embed_scale: Option<f64>,
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
        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };
        Ok(Self {
            embed_tokens: embed_tokens.clone(),
            embed_positions,
            layers,
            embed_scale,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let xs = xs.apply(&self.embed_tokens)?;
        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };
        let embed_pos = self
            .embed_positions
            .forward(&xs, past_kv_len)?
            .unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&embed_pos)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, encoder_xs, attn_mask)?;
        }
        Ok(xs)
    }

    pub fn reset_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_kv_cache()
        }
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

    fn reset_kv_cache(&mut self) {
        self.encoder.reset_kv_cache();
        self.decoder.reset_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct MTModel {
    model: Model,
    lm_head: Linear,
    final_logits_bias: Tensor,
}

impl MTModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let target_vocab_size = cfg.decoder_vocab_size.unwrap_or(cfg.vocab_size);
        let final_logits_bias = vb.get((1, target_vocab_size), "final_logits_bias")?;
        let model = Model::new(cfg, vb.pp("model"))?;
        let lm_head = Linear::from_weights(model.shared.embeddings().clone(), None);
        Ok(Self {
            model,
            lm_head,
            final_logits_bias,
        })
    }

    pub fn encoder(&mut self) -> &mut Encoder {
        &mut self.model.encoder
    }

    pub fn decoder(&mut self) -> &mut Decoder {
        &mut self.model.decoder
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
        self.model
            .decoder
            .forward(xs, Some(encoder_xs), past_kv_len, &mask)?
            .apply(&self.lm_head)?
            .broadcast_add(&self.final_logits_bias)
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }
}
