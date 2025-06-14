use crate::models::with_tracing::{linear, Linear, linear_no_bias};
use candle::{DType, Device, IndexOp, Module, D};
use candle::{Result, Tensor};
use candle_nn::{ Conv2dConfig, LayerNorm, LayerNormConfig, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::models::deepseek2::NonZeroOp;

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Idefic3VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,

    num_attention_heads: usize,
    num_channels: usize,
    pub patch_size: usize,
    image_size: usize,
    attention_dropout: f32,
    layer_norm_eps: f64,
    hidden_act: candle_nn::Activation,
    initializer_range: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Idefics3TextConfig {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Idefics3Config {
    pub vision_config: Idefic3VisionConfig,
    pub text_config: Idefics3TextConfig,
}

pub struct Idefics3VisionEmbeddings {
    patch_size: usize,
    patch_embeddings: candle_nn::Conv2d,
    num_patches_per_side: usize,
    position_embeddings: candle_nn::Embedding,
}

impl Idefics3VisionEmbeddings {
    pub fn load(config: &Idefic3VisionConfig, vs: candle_nn::VarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let image_size = config.image_size;
        let patch_size = config.patch_size;
        let num_patches_per_side = image_size / patch_size;
        let num_patches = num_patches_per_side * num_patches_per_side;
        let num_position = num_patches;
        let patch_embeddings = candle_nn::conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            Conv2dConfig {
                stride: config.patch_size,
                padding: 0,
                groups: 1,
                dilation: 1,
                cudnn_fwd_algo: None,
            },
            vs.pp("patch_embedding"),
        )?;
        let position_embeddings = candle_nn::embedding(
            num_position,
            embed_dim,
            vs.pp("position_embedding"),
        )?;
        Ok(Self {
            patch_size,
            patch_embeddings,
            num_patches_per_side,
            position_embeddings,
        })
    }

    pub fn forward(
        &self,
        pixel_values: Tensor,
        patch_attention_mask: Tensor,
        device: &Device,
    ) -> Result<Tensor> {
        let batch_size = pixel_values.dims()[0];
        let max_im_h = pixel_values.dims()[2];
        let max_im_w = pixel_values.dims()[3];

        let patch_embeds = self.patch_embeddings.forward(&pixel_values)?;
        let embeddings = patch_embeds.flatten_from(2)?.transpose(1, 2)?;
        let (max_nb_patchs_h, max_nb_patchs_w) =
            (max_im_h / self.patch_size, max_im_w / self.patch_size);
        let boundaries = Tensor::arange_step(
            1.0 / self.num_patches_per_side as f64,
            1.0,
            1.0 / self.num_patches_per_side as f64,
            device,
        )?
        .to_vec1::<f64>()?;
        let mut position_ids = Tensor::zeros(
            (batch_size, max_nb_patchs_h * max_nb_patchs_w),
            DType::I64,
            device,
        )?;

        for batch_idx in 0..batch_size {
            let p_attn_mask = patch_attention_mask.get(batch_idx)?;
            let nb_patches_h = p_attn_mask.get_on_dim(1, 0)?.sum_all()?.to_scalar::<u8>()?;
            let nb_patches_w = p_attn_mask.get_on_dim(1, 1)?.sum_all()?.to_scalar::<u8>()?;

            let fractional_coords_h =
                Tensor::arange_step(0., 1. - 1e-6, 1. / nb_patches_h as f64, device)?
                    .to_vec1::<f64>()?;
            let fractional_coords_w =
                Tensor::arange_step(0., 1. - 1e-6, 1. / nb_patches_w as f64, device)?
                    .to_vec1::<f64>()?;

            let bucket_coords_h = bucketize(&fractional_coords_h, &boundaries, true);
            let bucket_coords_w = bucketize(&fractional_coords_w, &boundaries, true);

            let bucket_coords_h_tensor =
                Tensor::from_vec(bucket_coords_h.clone(), (bucket_coords_h.len(),), device)?;
            let bucket_coords_w_tensor =
                Tensor::from_vec(bucket_coords_w.clone(), (bucket_coords_w.len(),), device)?;

            let pos_ids = (bucket_coords_h_tensor.unsqueeze(1)?
                * (self.num_patches_per_side as f64))?
                .broadcast_add(&bucket_coords_w_tensor)?
                .flatten_from(0)?;

            let p_attn_mask_flat = p_attn_mask.flatten_from(0)?;
            // Use tensor operations to find indices where mask is 1
            let indices = p_attn_mask_flat
                .to_dtype(DType::F32)?
                .eq(1.0)?
                .nonzero()?
                .squeeze(1)?;

            let positions = pos_ids.gather(&indices, 0)?;
            position_ids = position_ids.slice_assign(
                &[batch_idx..batch_idx + 1, 0..positions.dims()[0]],
                &positions.unsqueeze(0)?,
            )?;
        }
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        let embeddings = embeddings.add(&position_embeddings)?;

        Ok(embeddings)
    }
}

struct Idefics3VisionAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    dropout: f32,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    is_causal: bool,
    use_flash_attn: bool,
}

impl Idefics3VisionAttention {
    pub fn load(
        config: &Idefic3VisionConfig,
        use_flash_attn: bool,
        vs: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);
        let k_proj = linear(embed_dim, embed_dim, vs.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vs.pp("v_proj"))?;
        let q_proj = linear(embed_dim, embed_dim, vs.pp("q_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vs.pp("out_proj"))?;
        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scale,
            dropout: config.attention_dropout,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            is_causal: false,
            use_flash_attn,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, embed_dim) = hidden_states.dims3()?;
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * self.scale as f64)?;
        let attn_weights = attn_weights.broadcast_add(&attention_mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_weights = candle_nn::ops::dropout(&attn_weights, self.dropout)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, embed_dim))?;
        let attn_output = self.out_proj.forward(&attn_output)?;
        Ok((attn_output, attn_weights))
    }
}

struct Idefics3VisionMLP {
    activation_fn: candle_nn::Activation,
    fc1: Linear,
    fc2: Linear,
}

impl Idefics3VisionMLP {
    pub fn load(config: &Idefic3VisionConfig, vs: candle_nn::VarBuilder) -> Result<Self> {
        let activation_fn = config.hidden_act;
        let fc1 = linear(config.hidden_size, config.intermediate_size, vs.pp("fc1"))?;
        let fc2 = linear(config.intermediate_size, config.hidden_size, vs.pp("fc2"))?;
        Ok(Self {
            activation_fn,
            fc1,
            fc2,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = self.activation_fn.forward(&hidden_states)?;
        let hidden_states = self.fc2.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

struct Idefics3SimpleMLP {
    proj: Linear,
}

impl Idefics3SimpleMLP {
    pub fn load(config: Idefics3Config, vs: candle_nn::VarBuilder) -> Result<Self> {
        let proj = linear(
            config.vision_config.hidden_size,
            config.vision_config.hidden_size,
            vs.pp("proj"),
        )?;
        Ok(Self { proj })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.proj.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

struct Idefics3EncoderLayer {
    self_attn: Idefics3VisionAttention,
    layer_norm1: candle_nn::LayerNorm,
    layer_norm2: candle_nn::LayerNorm,
    mlp: Idefics3VisionMLP,
}

impl Idefics3EncoderLayer {
    pub fn load(
        config: &Idefic3VisionConfig,
        use_flash_attn: bool,
        vs: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let self_attn = Idefics3VisionAttention::load(config, use_flash_attn, vs.pp("self_attn"))?;
        let layer_norm1 = candle_nn::layer_norm(
            config.hidden_size,
            LayerNormConfig::from(config.layer_norm_eps),
            vs.pp("layer_norm1"),
        )?;
        let layer_norm2 = candle_nn::layer_norm(
            config.hidden_size,
            LayerNormConfig::from(config.layer_norm_eps),
            vs.pp("layer_norm2"),
        )?;
        let mlp = Idefics3VisionMLP::load(config, vs.pp("mlp"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            layer_norm2,
            mlp,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let residual = hidden_states;

        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let (hidden_states, attn_weights) =
            self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = hidden_states.add(&residual).unwrap();

        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = hidden_states.add(&residual).unwrap();
        Ok((hidden_states, attn_weights))
    }
}

struct Idefics3Encoder {
    layers: Vec<Idefics3EncoderLayer>,
}

impl Idefics3Encoder {
    pub fn load(
        config: &Idefic3VisionConfig,
        use_flash_attn: bool,
        vs: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                Idefics3EncoderLayer::load(config, use_flash_attn, vs.pp(format!("layers.{}", i)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, input_embeds: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = input_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?.0;
        }
        Ok(hidden_states)
    }
}

struct Idefics3RMSNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl Idefics3RMSNorm {
    pub fn load(config: &Idefic3VisionConfig, vs: candle_nn::VarBuilder) -> Result<Self> {
        let weight = vs.get_with_hints(config.hidden_size, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, variance_epsilon: config.layer_norm_eps })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let variance = hidden_states.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let hidden_states = (hidden_states * (variance + self.variance_epsilon))?.sqrt()?.recip()?;
        let hidden_states = (hidden_states * &self.weight)?;
        Ok(hidden_states)
    }
}

pub struct Idefics3VisionTransformer{
    embeddings: Idefics3VisionEmbeddings,
    encoder: Idefics3Encoder,
    post_layernorm: LayerNorm,
    use_flash_attn: bool,
}

impl Idefics3VisionTransformer{
    pub fn load(config: &Idefics3Config, use_flash_attn: bool, vs: candle_nn::VarBuilder) -> Result<Self> {
        let embeddings = Idefics3VisionEmbeddings::load(&config.vision_config, vs.pp("embeddings"))?;
        let encoder = Idefics3Encoder::load(&config.vision_config, use_flash_attn, vs.pp("encoder"))?;
        let post_layernorm = candle_nn::layer_norm(config.vision_config.hidden_size, LayerNormConfig::from(config.vision_config.layer_norm_eps), vs.pp("post_layernorm"))?;
        Ok(Self { embeddings, encoder, post_layernorm, use_flash_attn })
    }
}




fn bucketize(inputs: &[f64], boundaries: &[f64], right: bool) -> Vec<i64> {
    // Pre-allocate with capacity for better performance
    let mut result = Vec::with_capacity(inputs.len());

    // Use binary search to find the bucket for each input
    // This is O(log n) instead of O(n) for each input
    for &input in inputs {
        let bucket = match boundaries.binary_search_by(|&boundary| {
            if input < boundary || (!right && input == boundary) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        result.push(bucket as i64);
    }

    result
}
