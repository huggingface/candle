use candle::{ DType, Device, IndexOp, Module, Result, Tensor };
use candle_nn::{ layer_norm, Activation, LayerNorm, VarBuilder };
use std::sync::Arc;
use std::time::Instant;

use super::with_tracing::{ linear, linear_no_bias, Linear };

#[derive(Debug, Clone, Copy)]
pub enum EmbedDim {
    Dim256,
    Dim768,
    Dim1024,
    Dim2048,
    Dim4096,
    Dim6144,
    Dim8192,
}

impl Default for EmbedDim {
    fn default() -> Self {
        Self::Dim1024
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct EmbedHead {
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    // pub pad_token_id: usize,
    // pub hidden_dropout_prob: f64,
    // pub attention_probs_dropout_prob: f64,
    pub norm_eps: f64,
    // pub initializer_range: f64,
    // pub position_embedding_type: String,
    pub scaling_factor: f64,
    pub rope_theta: f64,
    // pub use_memory_efficient_attention: bool,
    // pub unpad_inputs: bool,
    // pub layer_norm_type: String,
    // pub logn_attention_scale: bool,
    // pub logn_attention_clip1: bool,
    pub activation_fn: Activation,
    pub embed_head: EmbedHead,
}

impl Config {
    pub fn new_400m(embed_dim: EmbedDim) -> Self {
        let embed_head = EmbedHead {
            in_features: 1024,
            out_features: match embed_dim {
                EmbedDim::Dim256 => 256,
                EmbedDim::Dim768 => 768,
                EmbedDim::Dim1024 => 1024,
                EmbedDim::Dim2048 => 2048,
                EmbedDim::Dim4096 => 4096,
                EmbedDim::Dim6144 => 6144,
                EmbedDim::Dim8192 => 8192,
            },
        };

        Self {
            vocab_size: 30528,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            // pad_token_id: 0,
            // hidden_dropout_prob: 0.1,
            // attention_probs_dropout_prob: 0.0,
            norm_eps: 1e-12,
            // initializer_range: 0.02,
            // position_embedding_type: "rope".to_string(),
            scaling_factor: 2.0,
            rope_theta: 160000.0,
            // use_memory_efficient_attention: true,
            // unpad_inputs: false,
            // layer_norm_type: "layer_norm".to_string(),
            // logn_attention_scale: false,
            // logn_attention_clip1: false,
            activation_fn: Activation::Gelu,
            embed_head,
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    // _scaling_factor: f64,
    // _mixed_b: Option<f64>,
    // _dim: usize,
    // _base: f64,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let scaling_factor = cfg.scaling_factor; // Can be configured in Config if needed
        let base = cfg.rope_theta;

        // Calculate scaled position embeddings
        let scaled_max_seq_len = ((max_seq_len as f64) * scaling_factor) as usize;

        // Calculate inv_freq with NTK scaling
        let inv_freq: Vec<_> = (0..dim / 2)
            .map(|i| {
                // Apply base scaling
                let base = base * scaling_factor;
                let freq = 1.0 / base.powf((2.0 * (i as f64)) / (dim as f64));

                // Apply fixed NTK scaling
                let freq = freq / scaling_factor.powf(2.0 / (dim as f64));

                freq as f32
            })
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;

        // Calculate position embeddings with scaled sequence length
        let t = Tensor::arange(0u32, scaled_max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((scaled_max_seq_len, 1))?;

        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

        Ok(Self {
            sin: emb.sin()?,
            cos: emb.cos()?,
            // _scaling_factor: scaling_factor,
            // _mixed_b: None,
            // _dim: dim,
            // _base: base,
        })
    }
}

#[derive(Debug, Clone)]
enum NormType {
    LayerNorm(candle_nn::LayerNorm),
    // RmsNorm(RmsNorm),
}

impl NormType {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(ln) => ln.forward(x),
            // Self::RmsNorm(rms) => rms.forward(x),
        }
    }
}

#[derive(Debug)]
pub struct Embeddings {
    word_embeddings: candle_nn::Embedding,
    // position_embeddings: Option<candle_nn::Embedding>,
    token_type_embeddings: candle_nn::Embedding,
    layer_norm: LayerNorm,
    // _padding_idx: usize,
    // _position_embedding_type: String,
    rotary_emb: Arc<RotaryEmbedding>,
    position_ids: Tensor,
}

impl Embeddings {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings")
        )?;

        // let position_embeddings = if cfg.position_embedding_type == "absolute" {
        //     Some(
        //         candle_nn::embedding(
        //             cfg.max_position_embeddings,
        //             cfg.hidden_size,
        //             vb.pp("position_embeddings")
        //         )?
        //     )
        // } else {
        //     None
        // };

        let token_type_embeddings = candle_nn::embedding(
            cfg.type_vocab_size,
            cfg.hidden_size,
            vb.pp("token_type_embeddings")
        )?;
        //  if cfg.type_vocab_size > 0 {
        //     Some(
        //         candle_nn::embedding(
        //             cfg.type_vocab_size,
        //             cfg.hidden_size,
        //             vb.pp("token_type_embeddings")
        //         )?
        //     )
        // } else {
        //     None
        // };

        //if cfg.layer_norm_type == "layer_norm" {
            let weight = vb
                .pp("LayerNorm")
                .get_with_hints(cfg.hidden_size, "weight", candle_nn::Init::Const(1.0))?;
            let bias = vb
                .pp("LayerNorm")
                .get_with_hints(cfg.hidden_size, "bias", candle_nn::Init::Const(0.0))?;
        let layer_norm = candle_nn::LayerNorm::new(weight, bias, cfg.norm_eps);
        // } else {
        //     NormType::RmsNorm(
        //         RmsNorm::new(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?
        //     )
        // };

        // let rotary_emb = if cfg.position_embedding_type == "rope" {
        //     Some(Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?))
        // } else {
        //     None
        // };
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        // let position_ids = if cfg.position_embedding_type == "absolute" {
        //     Some(Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?)
        // } else {
        //     None
        // };
        let position_ids = Tensor::arange(0u32, cfg.max_position_embeddings as u32, word_embeddings.embeddings().device())?;

        Ok(Self {
            word_embeddings,
            // position_embeddings,
            token_type_embeddings,
            layer_norm,
            // _padding_idx: cfg.pad_token_id,
            // _position_embedding_type: cfg.position_embedding_type.clone(),
            rotary_emb,
            position_ids,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        // token_type_ids: Option<&Tensor>,
        // position_ids: Option<&Tensor>,
        // inputs_embeds: Option<&Tensor>,
        // unpad_inputs: bool,
        // attention_mask: Option<&Tensor>
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (batch_size, seq_length) = input_ids.dims2()?;

        let mut embeddings = self.word_embeddings.forward(input_ids)?;
        // match inputs_embeds {
        //     Some(e) => e.clone(),
        //     None => self.word_embeddings.forward(input_ids)?,
        // };

        // Get position_ids first
        // let position_ids = if let Some(ids) = position_ids {
        //     ids.clone()
        // } else {
            // Get device from input_ids which is always available
            // let device = input_ids.device();

            // Initialize position_ids if None
            // if self.position_ids.is_none() {
            //     self.position_ids = Some(Tensor::arange(0u32, seq_length as u32, device)?);
            // }

            // // Now check if we need to extend it
            // if seq_length > self.position_ids.as_ref().unwrap().dim(0)? {
            //     self.position_ids = Some(Tensor::arange(0u32, seq_length as u32, device)?);
            // }

        // let position_ids = 
        /*if unpad_inputs {
                // For now, just use the same position IDs as padded case since we don't have lengths
                self.position_ids
                    .as_ref()
                    .unwrap()
                    .narrow(0, 0, seq_length)?
                    .expand((batch_size, seq_length))?
            } else {*/
                // self.position_ids
                //     .as_ref()
                //     .unwrap()
                //     .narrow(0, 0, seq_length)?
                //     .expand((batch_size, seq_length))?;
            // };
        // };

        let position_ids = self.position_ids
            .as_ref()
            .narrow(0, 0, seq_length)?
            .expand((batch_size, seq_length))?;


        let rope_embeds = {
            // Get the cos and sin for this sequence length
            let cos = self.rotary_emb.cos.narrow(0, 0, seq_length)?; // [seq_len, head_dim]
            let sin = self.rotary_emb.sin.narrow(0, 0, seq_length)?; // [seq_len, head_dim]

            // Index using position_ids if needed
            let position_ids = position_ids.flatten_all()?;
            let cos = cos.index_select(&position_ids, 0)?; // Use index_select instead of i()
            let sin = sin.index_select(&position_ids, 0)?; // Use index_select instead of i()

            Some((cos, sin))
        };
        // // Get rotary embeddings if using RoPE
        // let rope_embeds = if let Some(rotary) = &self.rotary_emb {
            
        // } else {
        //     None
        // };

        // Handle token type embeddings
        embeddings = embeddings.add(&self.token_type_embeddings.forward(&position_ids.zeros_like()?)?).unwrap();
        // if let Some(token_emb) = &self.token_type_embeddings {
            // let token_type_ids = if let Some(ids) = token_type_ids {
            //     ids.clone()
            // } else {
                // position_ids.zeros_like()? // Use mul(0) equivalent
            // };
            // if unpad_inputs {
            //     todo!("Implement unpadded case");
            // } else {
                // embeddings = embeddings.add(&token_emb.forward(&position_ids.zeros_like()?)?).unwrap();
            // }
        // }

        // Handle absolute position embeddings
        // if let Some(pos_emb) = &self.position_embeddings {
        //     let position_embeddings = pos_emb.forward(&position_ids)?;
        //     embeddings = embeddings.add(&position_embeddings)?;
        // }

        let embeddings = self.layer_norm.forward(&embeddings)?;

        Ok((embeddings, rope_embeds))
    }
}

#[derive(Debug)]
struct NewAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    // _use_memory_efficient_attention: bool,
}

impl NewAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = hidden_sz / num_heads;

        let qkv_proj = linear(hidden_sz, 3 * num_heads * head_dim, vb.pp("qkv_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            head_dim,
            hidden_size: hidden_sz,
            // _use_memory_efficient_attention: cfg.use_memory_efficient_attention,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        rope_embeds: Option<&(Tensor, Tensor)>,
        // _attention_scale: Option<&Tensor>
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        // QKV projection
        let qkv = self.qkv_proj.forward(hidden_states)?;

        // Split into Q, K, V and reshape to match PyTorch shapes
        let qkv = qkv.reshape((b_sz, seq_len, 3, self.num_heads, self.head_dim))?;

        // Get Q, K, V with shape [batch, seq_len, num_heads, head_dim]
        let query_states = qkv.i((.., .., 0, .., ..))?.contiguous()?;
        let key_states = qkv.i((.., .., 1, .., ..))?.contiguous()?;
        let value_states = qkv.i((.., .., 2, .., ..))?.contiguous()?;

        // Apply RoPE if provided
        let (query_states, key_states) = if let Some((cos, sin)) = rope_embeds {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin)?
        } else {
            (query_states, key_states)
        };

        // Transpose for attention computation [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        let query_states = query_states.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.transpose(1, 2)?.contiguous()?;

        // For key, we want to transpose the last two dimensions for the matmul
        // Is this equivalent to PyTorch's transpose(-1, -2)?
        let key_states_t = key_states.transpose(2, 3)?.contiguous()?;

        // Prepare tensors for batched matmul using matmul
        // Reshape tensors to merge batch and head dimensions
        let bsz = b_sz;
        let nh = self.num_heads;
        let s_len = seq_len;
        let h_dim = self.head_dim;

        // Reshape tensors to [batch_size * num_heads, seq_len, head_dim]
        let query_states_reshaped = query_states.reshape((bsz * nh, s_len, h_dim))?;
        let key_states_t_reshaped = key_states_t.reshape((bsz * nh, h_dim, s_len))?;

        // Perform batched matmul using matmul
        // The matmul should handle batch dimensions if tensors are 3D
        let attn_weights = query_states_reshaped.matmul(&key_states_t_reshaped)?;

        // Reshape attn_weights back to [batch_size, num_heads, seq_len, seq_len]
        let attn_weights = attn_weights.reshape((bsz, nh, s_len, s_len))?;

        // Scale attention scores
        let scale = 1f32 / (self.head_dim as f32).sqrt();

        let scale_tensor = Tensor::new(scale, attn_weights.device())?
            .to_dtype(attn_weights.dtype())?
            .broadcast_as(attn_weights.shape())?;

        // Multiply the attention weights by the scalar tensor
        let attn_weights = attn_weights.mul(&scale_tensor)?;

        // Apply attention mask
        let mut attn_weights = if let Some(bias) = attention_bias {
            // let attn_weights = attn_weights.broadcast_add(bias)?;
            attn_weights.broadcast_add(bias)?
        } else {
            attn_weights
        };

        // Normalize attention scores
        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Reshape value_states for batched matmul
        let value_states_reshaped = value_states.reshape((bsz * nh, s_len, h_dim))?;

        // Reshape attn_weights to [batch_size * num_heads, seq_len, seq_len]
        let attn_weights_reshaped = attn_weights.reshape((bsz * nh, s_len, s_len))?;

        // Compute attention output
        let attn_output = attn_weights_reshaped.matmul(&value_states_reshaped)?;

        // Reshape attn_output back to [batch_size, num_heads, seq_len, head_dim]
        let attn_output = attn_output.reshape((bsz, nh, s_len, h_dim))?;

        // Transpose back to [batch_size, seq_len, num_heads, head_dim]
        let attn_output = attn_output.transpose(1, 2)?;

        // Project to final dimension
        let attn_output = attn_output.reshape((b_sz, seq_len, self.hidden_size))?;
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Debug)]
struct NewGatedMLP {
    up_gate_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl NewGatedMLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let act_fn = cfg.activation_fn;
        let up_gate_proj = linear_no_bias(hidden_sz, intermediate_size * 2, vb.pp("up_gate_proj"))?;
        let down_proj = linear(intermediate_size, hidden_sz, vb.pp("down_proj"))?;

        Ok(Self {
            up_gate_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for NewGatedMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_gate = self.up_gate_proj.forward(xs)?;

        // Get the dimensions
        let (_batch_size, _seq_len, hidden_dim) = up_gate.dims3()?;
        let split_size = hidden_dim / 2;

        // Split along the last dimension (hidden_dim)
        let up_states = up_gate.narrow(2, 0, split_size)?;
        let gate = up_gate.narrow(2, split_size, split_size)?;

        // Apply activation to gate and multiply
        let gate = gate.apply(&self.act_fn)?;

        let gated_states = up_states.mul(&gate)?;

        // Project back to hidden dimension
        let output = self.down_proj.forward(&gated_states)?;

        Ok(output)
    }
}

#[derive(Debug)]
struct NewLayer {
    attention: NewAttention,
    mlp: NewGatedMLP,
    attn_ln: NormType,
    mlp_ln: NormType,
}

impl NewLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = NewAttention::new(cfg, vb.pp("attention"))?;
        let mlp = NewGatedMLP::new(cfg, vb.pp("mlp"))?;

        // let ln_eps = cfg.layer_norm_eps;

        // Use LayerNorm or RmsNorm based on config
        let (attn_ln, mlp_ln) = {
            let attn_ln = layer_norm(
                cfg.hidden_size,
                candle_nn::LayerNormConfig { eps: cfg.norm_eps, ..Default::default() },
                vb.pp("attn_ln")
            )?;
            let mlp_ln = layer_norm(
                cfg.hidden_size,
                candle_nn::LayerNormConfig { eps: cfg.norm_eps, ..Default::default() },
                vb.pp("mlp_ln")
            )?;
            (NormType::LayerNorm(attn_ln), NormType::LayerNorm(mlp_ln))
        };
        //  else 
        // {
        //     let attn_ln = RmsNorm::new(cfg.hidden_size, ln_eps, vb.pp("attn_ln"))?;
        //     let mlp_ln = RmsNorm::new(cfg.hidden_size, ln_eps, vb.pp("mlp_ln"))?;
        //     (NormType::RmsNorm(attn_ln), NormType::RmsNorm(mlp_ln))
        // };

        Ok(Self {
            attention,
            mlp,
            attn_ln,
            mlp_ln,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        rope_embeds: Option<&(Tensor, Tensor)>,
        // attention_scale: Option<&Tensor>
    ) -> Result<Tensor> {
        // Store original input
        let original = hidden_states;

        // Use normalized states for attention
        let hidden_states = self.attention.forward(
            original,
            attention_bias,
            rope_embeds,
            // attention_scale
        )?;

        let hidden_states = original.add(&hidden_states)?;

        // Apply layer norm
        let hidden_states = self.attn_ln.forward(&hidden_states)?;

        // Store residual
        let residual = &hidden_states;

        // Pass through MLP
        let hidden_states = self.mlp.forward(&hidden_states)?;

        // Add residual connection
        let hidden_states = residual.add(&hidden_states)?;

        // Final layer norm
        self.mlp_ln.forward(&hidden_states)
    }
}

#[derive(Debug)]
struct NewEncoder {
    layers: Vec<NewLayer>,
}

impl NewEncoder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layer");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(NewLayer::new(cfg, vb_l.pp(layer_idx))?);
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        rope_embeds: Option<&(Tensor, Tensor)>,
        // attention_scale: Option<&Tensor>
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in self.layers.iter() {
            hidden_states = layer.forward(
                &hidden_states,
                attention_bias,
                rope_embeds,
                // attention_scale
            )?;
        }

        Ok(hidden_states)
    }
}

#[derive(Debug)]
pub struct NewModel {
    embeddings: Embeddings,
    encoder: NewEncoder,
    device: Device,
    dtype: DType,
    // config: Config,
}

impl NewModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("new");
        let embeddings = Embeddings::new(cfg, vb_m.pp("embeddings"))?;
        let encoder = NewEncoder::new(cfg, vb_m.pp("encoder"))?;
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            // config: cfg.clone(),
        })
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = attn_mask.dims2()?;
        let mask = attn_mask
            .unsqueeze(1)
            ? // [b_sz, 1, seq_len]
            .unsqueeze(2)
            ? // [b_sz, 1, 1, seq_len]
            .broadcast_as((b_sz, 1, 1, seq_len))?; // [b_sz, 1, 1, seq_len]

        // Use a large negative value for mask instead of -0.0
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;

        mask.where_cond(&on_true, &on_false)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        // token_type_ids: Option<&Tensor>,
        // position_ids: Option<&Tensor>
    ) -> Result<Tensor> {
        let (_, seq_length) = input_ids.dims2()?;

        // Get attention mask if not provided
        // let attention_mask = mask;

        // Prepare attention bias
        let attention_bias = if seq_length <= 1 {
            None
        } else {
            Some(self.prepare_attention_mask(attention_mask)?)
        };

        // Get embeddings and rotary embeddings
        let (hidden_states, rope_embeds) = self.embeddings.forward(
            input_ids,
            // token_type_ids,
            // position_ids,
            // None,
            // self.config.unpad_inputs,
            // Some(&attention_mask)
        )?;

        // Compute attention scale if needed
        // let attention_scale = if self.config.logn_attention_scale {
        //     let scale =
        //         attention_mask.sum_keepdim(1)?.log()? /
        //         (self.config.max_position_embeddings as f64).ln();
        //     if self.config.logn_attention_clip1 {
        //         let scale = scale?;
        //         Some(scale.maximum(&Tensor::new(1f64, &self.device)?)?)
        //     } else {
        //         Some(scale?)
        //     }
        // } else {
        //     None
        // };

        // Forward through encoder
        let hidden_states = self.encoder.forward(
            &hidden_states,
            attention_bias.as_ref(),
            rope_embeds.as_ref(),
            // attention_scale.as_ref()
        )?;

        Ok(hidden_states)
    }
}

// Optional pooler implementation
// #[derive(Debug)]
// pub struct NewPooler {
//     dense: Linear,
// }

// impl NewPooler {
//     pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
//         Ok(Self { dense })
//     }

//     pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let first_token = hidden_states.i((.., 0, ..))?;
//         let pooled = self.dense.forward(&first_token)?;
//         pooled.tanh()
//     }
// }

// // Complete model with pooler
// #[derive(Debug)]
// pub struct NewModelWithPooler {
//     model: NewModel,
//     pooler: Option<NewPooler>,
// }

// impl NewModelWithPooler {
//     pub fn new(cfg: &Config, vb: VarBuilder, add_pooling_layer: bool) -> Result<Self> {
//         let vb_m = vb.pp("new");
//         let model = NewModel::new(cfg, vb_m.pp("model"))?;
//         let pooler = if add_pooling_layer {
//             Some(NewPooler::new(cfg, vb.pp("new").pp("pooler"))?)
//         } else {
//             None
//         };
//         Ok(Self { model, pooler })
//     }

//     pub fn forward(
//         &mut self,
//         input_ids: &Tensor,
//         attention_mask: Option<&Tensor>,
//         token_type_ids: Option<&Tensor>,
//         position_ids: Option<&Tensor>
//     ) -> Result<(Tensor, Option<Tensor>)> {
//         let hidden_states = self.model.forward(
//             input_ids,
//             attention_mask,
//             token_type_ids,
//             position_ids
//         )?;

//         let pooled_output = match &self.pooler {
//             Some(pooler) => Some(pooler.forward(&hidden_states)?),
//             None => None,
//         };

//         Ok((hidden_states, pooled_output))
//     }
// }

#[derive(Debug)]
pub struct EmbeddingModel {
    base_model: NewModel,
    lm_head: Linear,
}

impl EmbeddingModel {
    pub fn new(cfg: &Config, base_vb: VarBuilder, embed_vb: VarBuilder) -> Result<Self> {
        let base_model = NewModel::new(cfg, base_vb)?;
        let lm_head = linear(
            cfg.embed_head.in_features,
            cfg.embed_head.out_features,
            embed_vb.pp("linear")
        )?;

        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = self.base_model.forward(input_ids, mask)?;//, None, None)?;
        let x = self.pool(&x, mask)?;
        self.lm_head.forward(&x.to_dtype(DType::F32)?)
    }

    pub fn forward_norm(&mut self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = self.forward(input_ids, mask)?;
        x.broadcast_div(&x.sqr()?.sum_keepdim(1)?.sqrt()?)
    }
    fn pool(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mask = mask.to_dtype(x.dtype())?;
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let mask_expanded = mask.unsqueeze(2)?.broadcast_as((batch_size, seq_len, hidden_dim))?; // [B_Sz, Seq_len, Hidden_dim]
        let x = x.mul(&mask_expanded)?;
        let sum_mask = mask.sum(1)?.unsqueeze(1)?.expand((batch_size, hidden_dim))?;
        x.sum(1)? / sum_mask
    }
}

pub fn time_run<F, T>(f: F) -> (T, std::time::Duration) where F: FnOnce() -> T {
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor
) -> Result<(Tensor, Tensor)> {
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;

    let (batch_size, seq_len, num_heads, head_dim) = q.dims4()?;
    let half_dim = head_dim / 2;

    // Reshape q and k to split the head dim for rotation
    let q_split = q.chunk(2, 3)?; // Split along head_dim
    let k_split = k.chunk(2, 3)?;

    let q1 = &q_split[0];
    let q2 = &q_split[1];
    let k1 = &k_split[0];
    let k2 = &k_split[1];

    // Handle cos/sin for the sequence length we have
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;

    // Reshape cos/sin to match the dimensions we need
    let cos = cos
        .reshape((seq_len, head_dim))?
        .chunk(2, 1)?
        [0].reshape((seq_len, 1, half_dim))?
        .broadcast_as((seq_len, num_heads, half_dim))?
        .unsqueeze(0)?
        .broadcast_as((batch_size, seq_len, num_heads, half_dim))?;

    let sin = sin
        .reshape((seq_len, head_dim))?
        .chunk(2, 1)?
        [0].reshape((seq_len, 1, half_dim))?
        .broadcast_as((seq_len, num_heads, half_dim))?
        .unsqueeze(0)?
        .broadcast_as((batch_size, seq_len, num_heads, half_dim))?;

    // Apply rotation using the formulas:
    // q = q * cos + rotate_half(q) * sin
    // k = k * cos + rotate_half(k) * sin
    let q_out = Tensor::cat(
        &[&q1.mul(&cos)?.sub(&q2.mul(&sin)?)?, &q2.mul(&cos)?.add(&q1.mul(&sin)?)?],
        3
    )?;

    let k_out = Tensor::cat(
        &[&k1.mul(&cos)?.sub(&k2.mul(&sin)?)?, &k2.mul(&cos)?.add(&k1.mul(&sin)?)?],
        3
    )?;

    Ok((q_out, k_out))
}
