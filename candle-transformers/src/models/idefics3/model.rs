use crate::models::llama::{self, Cache, LlamaBase};
use crate::models::with_tracing::{linear, linear_no_bias, Embedding, Linear};
use candle::{DType, Device, Module, D};
use candle::{Result, Tensor};
use candle_nn::{Conv2dConfig, LayerNorm, LayerNormConfig};
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

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3Config {
    pub vision_config: Idefic3VisionConfig,
    pub text_config: llama::LlamaConfig,
    pub scale_factor: Option<usize>,
    pub image_token_id: usize,
}

pub struct Idefics3VisionEmbeddings {
    patch_size: usize,
    patch_embeddings: candle_nn::Conv2d,
    num_patches_per_side: usize,
    position_embeddings: Embedding,
}

impl Idefics3VisionEmbeddings {
    pub fn load(config: &Idefic3VisionConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
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
            vb.pp("patch_embedding"),
        )?;
        let position_embeddings =
            Embedding::new(num_position, embed_dim, vb.pp("position_embedding"))?;
        Ok(Self {
            patch_size,
            patch_embeddings,
            num_patches_per_side,
            position_embeddings,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        patch_attention_mask: &Tensor,
        device: &Device,
    ) -> Result<Tensor> {
        let batch_size = pixel_values.dims()[0];
        let max_im_h = pixel_values.dims()[2];
        let max_im_w = pixel_values.dims()[3];

        let patch_embeds = self.patch_embeddings.forward(pixel_values)?;
        let embeddings = patch_embeds.flatten_from(2)?.transpose(1, 2)?;
        let (max_nb_patchs_h, max_nb_patchs_w) =
            (max_im_h / self.patch_size, max_im_w / self.patch_size);
        let boundaries = Tensor::arange_step(
            1.0 / self.num_patches_per_side as f64,
            1.0,
            1.0 / self.num_patches_per_side as f64,
            &Device::Cpu,
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
    num_heads: usize,
    head_dim: usize,
    scale: f32,
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
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            num_heads,
            head_dim,
            scale,
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
        attention_mask: &Option<Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, q_len, embed_dim) = hidden_states.dims3()?;
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let scale = 1f32 / (self.head_dim as f32).sqrt();
            let attn_output = flash_attn(&q, &k, &v, scale, self.is_causal)?.transpose(1, 2)?;
            let attn_output = attn_output
                .transpose(1, 2)?
                .reshape((batch_size, q_len, embed_dim))?;
            let attn_output = self.out_proj.forward(&attn_output)?;
            return Ok((attn_output, None));
        }
        let shape = (batch_size, q_len, self.num_heads, self.head_dim);

        let query_states = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale as f64)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // The original implementation upcasts to f32 but candle_nn::ops::softmax should handle this properly.
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_outputs = attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?
            .apply(&self.out_proj)?;
        Ok((attn_outputs, None))
    }
}

struct Idefics3VisionMLP {
    activation_fn: candle_nn::Activation,
    fc1: Linear,
    fc2: Linear,
}

impl Idefics3VisionMLP {
    pub fn load(config: &Idefic3VisionConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let activation_fn = config.hidden_act;
        let fc1 = linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
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
    pub fn load(config: &Idefics3Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(
            config.vision_config.hidden_size * (config.scale_factor.unwrap_or(2).pow(2)),
            config.text_config.hidden_size,
            vb.pp("proj"),
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
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let self_attn = Idefics3VisionAttention::load(config, use_flash_attn, vb.pp("self_attn"))?;
        let layer_norm1 = candle_nn::layer_norm(
            config.hidden_size,
            LayerNormConfig::from(config.layer_norm_eps),
            vb.pp("layer_norm1"),
        )?;
        let layer_norm2 = candle_nn::layer_norm(
            config.hidden_size,
            LayerNormConfig::from(config.layer_norm_eps),
            vb.pp("layer_norm2"),
        )?;
        let mlp = Idefics3VisionMLP::load(config, vb.pp("mlp"))?;
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
        attention_mask: &Option<Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let residual = hidden_states;

        let hidden_states = hidden_states.apply(&self.layer_norm1)?;
        let (hidden_states, attn_weights) =
            self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = hidden_states.add(residual)?;

        let residual = hidden_states.clone();
        let hidden_states = hidden_states.apply(&self.layer_norm2)?;
        let hidden_states = self.mlp.forward(&hidden_states)?.add(&residual)?;
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
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                Idefics3EncoderLayer::load(config, use_flash_attn, vb.pp(format!("layers.{}", i)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    pub fn forward(
        &self,
        input_embeds: &Tensor,
        attention_mask: &Option<Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = input_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?.0;
        }
        Ok(hidden_states)
    }
}

struct Idefics3Connector {
    modaliity_projection: Idefics3SimpleMLP,
    scale_factor: Option<usize>,
}

impl Idefics3Connector {
    pub fn load(config: &Idefics3Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let modaliity_projection = Idefics3SimpleMLP::load(config, vb.pp("modality_projection"))?;
        Ok(Self {
            modaliity_projection,
            scale_factor: config.scale_factor,
        })
    }

    pub fn pixel_shuffle(&self, x: &Tensor, scale_factor: Option<usize>) -> Result<Tensor> {
        let scale_factor = scale_factor.unwrap_or(2);
        let (b_sz, seq, embed_dim) = x.dims3()?;
        let height = (seq as f64).sqrt() as usize;
        let width = height;

        let x = x.reshape((b_sz, height, width, embed_dim))?;
        let x = x.reshape((b_sz, height, width / scale_factor, embed_dim * scale_factor))?;
        let x = x.permute((0, 2, 1, 3))?;
        let x = x.reshape((
            b_sz,
            width / scale_factor,
            height / scale_factor,
            embed_dim * scale_factor * scale_factor,
        ))?;
        let x = x.permute((0, 2, 1, 3))?;
        let x = x.reshape((
            b_sz,
            seq / (scale_factor * scale_factor),
            embed_dim * scale_factor * scale_factor,
        ))?;
        Ok(x)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pixel_shuffle(x, self.scale_factor)?;
        let x = self.modaliity_projection.forward(&x)?;
        Ok(x)
    }
}

pub struct Idefics3VisionTransformer {
    embeddings: Idefics3VisionEmbeddings,
    encoder: Idefics3Encoder,
    post_layernorm: LayerNorm,
}

impl Idefics3VisionTransformer {
    pub fn load(
        config: &Idefics3Config,
        use_flash_attn: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let embeddings =
            Idefics3VisionEmbeddings::load(&config.vision_config, vb.pp("embeddings"))?;
        let encoder =
            Idefics3Encoder::load(&config.vision_config, use_flash_attn, vb.pp("encoder"))?;
        let post_layernorm = candle_nn::layer_norm(
            config.vision_config.hidden_size,
            LayerNormConfig::from(config.vision_config.layer_norm_eps),
            vb.pp("post_layernorm"),
        )?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        patch_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let patch_attention_mask = if let Some(patch_attention_mask) = patch_attention_mask {
            patch_attention_mask.clone()
        } else {
            Tensor::ones(
                (
                    pixel_values.dims()[0],
                    pixel_values.dims()[2],
                    pixel_values.dims()[3],
                ),
                DType::F32,
                pixel_values.device(),
            )?
        };
        let hidden_states =
            self.embeddings
                .forward(pixel_values, &patch_attention_mask, pixel_values.device())?;
        let patch_attention_mask = patch_attention_mask.flatten_from(1)?;
        let patch_attention_mask =
            prepare_4d_attention_mask(&patch_attention_mask, pixel_values.dtype(), None)?;

        let hidden_states = self
            .encoder
            .forward(&hidden_states, &Some(patch_attention_mask))?;
        let hidden_states = self.post_layernorm.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct Idefics3Model {
    vision_model: Idefics3VisionTransformer,
    connector: Idefics3Connector,
    text_model: LlamaBase,
    image_token_id: usize,
    use_flash_attn: bool,
    config: Idefics3Config,
    dtype: DType,
}

impl Idefics3Model {
    pub fn load(
        config: &Idefics3Config,
        use_flash_attn: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let vision_model =
            Idefics3VisionTransformer::load(config, use_flash_attn, vb.pp("vision_model"))?;
        let connector = Idefics3Connector::load(config, vb.pp("connector"))?;
        let text_model = LlamaBase::load(
            vb.pp("text_model"),
            &config.text_config.clone().into_config(use_flash_attn),
        )?;

        let image_token_id = config.image_token_id;
        Ok(Self {
            vision_model,
            connector,
            text_model,
            image_token_id,
            use_flash_attn,
            config: config.clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: &Option<Tensor>,
        pixel_attention_mask: &Option<Tensor>,
    ) -> Result<Tensor> {
        if let (Some(pixel_values), Some(pixel_attention_mask)) =
            (pixel_values, pixel_attention_mask)
        {
            let pixel_values = pixel_values.to_dtype(self.dtype)?;
            let input_embeds = self.text_model.embed(input_ids)?;
            let (bsz, num_images, channels, height, width) = pixel_values.dims5()?;
            let pixel_values = pixel_values.reshape((bsz * num_images, channels, height, width))?;
            let pixel_attention_mask =
                pixel_attention_mask.reshape((bsz * num_images, height, width))?;

            let nb_values_per_image =
                pixel_values.dims()[1] * pixel_values.dims()[2] * pixel_values.dims()[3];
            let real_image_inds = pixel_values
                .eq(0.0)?
                .sum((3, 2, 1))?
                .ne(nb_values_per_image as f64)?;

            let indices = real_image_inds
                .to_dtype(DType::F32)?
                .eq(1.0)?
                .nonzero()?
                .squeeze(1)?;

            let pixel_values = pixel_values.index_select(&indices, 0)?;
            let pixel_attention_mask = pixel_attention_mask.index_select(&indices, 0)?;

            let patches_subgrid = unfold(
                &pixel_attention_mask,
                self.config.vision_config.patch_size,
                self.config.vision_config.patch_size,
                1,
            );

            let patches_subgrid = unfold(
                &patches_subgrid?,
                self.config.vision_config.patch_size,
                self.config.vision_config.patch_size,
                2,
            )?;

            let patch_attention_mask = patches_subgrid.sum((D::Minus1, D::Minus2))?.ge(0.0)?;

            let image_hidden_states = self
                .vision_model
                .forward(&pixel_values, Some(&patch_attention_mask))?;
            let image_hidden_states = self.connector.forward(&image_hidden_states)?;

            let new_input_embeds =
                self.inputs_merger(input_ids, &input_embeds, &image_hidden_states)?;

            let output = self.text_model.forward_input_embed(
                &new_input_embeds,
                0,
                &mut Cache::new(
                    false,
                    pixel_values.dtype(),
                    &self
                        .config
                        .text_config
                        .clone()
                        .into_config(self.use_flash_attn),
                    pixel_values.device(),
                )?,
            )?;
            Ok(output)
        } else {
            self.text_model.forward(
                input_ids,
                0,
                &mut Cache::new(
                    false,
                    self.dtype,
                    &self
                        .config
                        .text_config
                        .clone()
                        .into_config(self.use_flash_attn),
                    input_ids.device(),
                )?,
            )
        }
    }

    fn inputs_merger(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        image_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let (num_images, seq_len, vision_hidden_size) = image_hidden_states.dims3()?;
        let (bsz, text_seq_len, embed_dim) = input_embeds.dims3()?;

        let input_embeds_reshaped = input_embeds.reshape((bsz * text_seq_len, embed_dim))?;

        let input_ids = input_ids.flatten_from(0)?;
        let special_image_token_indices = input_ids.eq(self.image_token_id as f64)?;

        let image_hidden_states =
            image_hidden_states.reshape((num_images * seq_len, vision_hidden_size))?;
        let special_image_token_indices = special_image_token_indices
            .nonzero()?
            .repeat((1, embed_dim))?;

        let new_input_embeds =
            input_embeds_reshaped.scatter(&special_image_token_indices, &image_hidden_states, 0).unwrap();
        let new_input_embeds = new_input_embeds.reshape((bsz, text_seq_len, embed_dim))?;

        Ok(new_input_embeds)
    }
}

pub struct ColIdefics3Model {
    model: Idefics3Model,
    linear: Linear,
}

impl ColIdefics3Model {
    pub fn load(
        config: &Idefics3Config,
        use_flash_attn: bool,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let model = Idefics3Model::load(config, use_flash_attn, vb.pp("model"))?;
        let dim = 128;
        let linear = linear(model.config.text_config.hidden_size, dim, vb.pp("linear"))?;
        Ok(Self { model, linear })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        pixel_values: &Option<Tensor>,
        pixel_attention_mask: &Option<Tensor>,
    ) -> Result<Tensor> {
        let output = self
            .model
            .forward(input_ids, pixel_values, pixel_attention_mask)?;
        let proj = self.linear.forward(&output)?;
        let proj = proj.broadcast_div(&proj.sqr()?.sum_keepdim(2)?.sqrt()?)?;
        let proj = proj.broadcast_mul(&attention_mask.unsqueeze(2)?.to_dtype(proj.dtype())?)?;

        Ok(proj)
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

fn unfold(
    x: &candle::Tensor,
    size: usize,
    step: usize,
    dim: usize,
) -> candle::Result<candle::Tensor> {
    let x_shape = x.shape().dims().to_vec();
    let len = x_shape[dim];
    let num = (len - size) / step + 1;

    let mut idx_data = Vec::with_capacity(num * size);
    for i in 0..num {
        let base = i * step;
        for j in 0..size {
            idx_data.push((base + j) as i64);
        }
    }

    let mut perm: Vec<usize> = (0..x_shape.len()).filter(|&i| i != dim).collect();
    perm.push(dim);
    let x = x.permute(perm)?;

    let mut inv_perm: Vec<usize> = (0..x_shape.len()).collect();
    let moved_element = inv_perm.remove(0);
    inv_perm.insert(dim, moved_element);
    inv_perm.push(x_shape.len());

    let mut idx_shape = vec![num];
    for (i, _) in x_shape.iter().enumerate() {
        if i != dim {
            idx_shape.push(x_shape[i]);
        }
    }
    idx_shape.push(size);

    let idx = candle::Tensor::from_vec(idx_data, &[num, size], x.device())?;

    let mut reshape_dims = vec![num];
    for i in 0..x_shape.len() {
        if i != dim {
            reshape_dims.push(1);
        }
    }
    reshape_dims.push(size);

    let reshape_dims: &[usize] = &reshape_dims;
    let idx = idx
        .reshape(reshape_dims)?
        .broadcast_as(&idx_shape[..])?
        .contiguous()?;

    let mut repeat_dims = vec![1; x_shape.len()];
    repeat_dims[0] = num;
    let x = x.unsqueeze(0)?.repeat(repeat_dims)?;

    let x = x.gather(&idx, x_shape.len())?.permute(inv_perm)?;

    Ok(x)
}

// Global attention mask calculated from padded token inputs
fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?;

    let inverted_mask = (1.0 - expanded_mask)?;

    (inverted_mask * f32::MIN as f64)?.to_dtype(dtype)
}
