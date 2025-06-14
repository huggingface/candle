use candle::{DType, Device, IndexOp, Module};
use candle::{Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::models::deepseek2::NonZeroOp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Idefic3VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,

    num_attention_heads: usize,
    num_channels: usize,
    patch_size: usize,
    image_size: usize,
    attention_dropout: f64,
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
    embed_dim: usize,
    image_size: usize,
    patch_size: usize,
    patch_embeddings: candle_nn::Conv2d,
    num_patches_per_side: usize,
    num_patches: usize,
    num_position: usize,
    position_embeddings: candle_nn::Embedding,
}

impl Idefics3VisionEmbeddings {
    pub fn load(config: Idefic3VisionConfig, vs: candle_nn::VarBuilder) -> Result<Self> {
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
            vs.pp("model.vision_model.embeddings.patch_embedding"),
        )?;
        let position_embeddings = candle_nn::embedding(
            num_position,
            embed_dim,
            vs.pp("model.vision_model.embeddings.position_embedding"),
        )?;
        Ok(Self {
            embed_dim,
            image_size,
            patch_size,
            patch_embeddings,
            num_patches_per_side,
            num_patches,
            num_position,
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
        let num_channels = pixel_values.dims()[1];
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

        println!("Number of patches: {}", self.num_patches);
        println!("Number of patches per side: {}", self.num_patches_per_side);

        for batch_idx in 0..batch_size {
            let p_attn_mask = patch_attention_mask.get(batch_idx)?;
            let nb_patches_h = p_attn_mask
                .get_on_dim(1, 0)?
                .sum_all()?
                .to_scalar::<i64>()?;
            let nb_patches_w = p_attn_mask
                .get_on_dim(1, 1)?
                .sum_all()?
                .to_scalar::<i64>()?;

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
                .broadcast_add(&bucket_coords_w_tensor)?.flatten_from(0)?;

            let p_attn_mask_flat = p_attn_mask.flatten_from(0)?;
            // Use tensor operations to find indices where mask is 1
            let indices = p_attn_mask_flat
                .to_dtype(DType::F32)?
                .eq(1.0)?
                .nonzero()?
                .squeeze(1)?
                .to_vec1::<u32>()?;
            
            println!("Pos IDs: {}", pos_ids);
            // println!("Indices: {:?}", indices);
            // let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), &device)?;
            // println!("Position IDs: {:?}", position_ids.shape());
            // position_ids = position_ids.slice_assign(&[batch_idx..batch_idx+1, 0..pos_ids.dims()[0]], &pos_ids.unsqueeze(0)?).unwrap();
            // println!("Position IDs: {}", position_ids.get(batch_idx).unwrap());
            // println!("P Attn Mask Flat: {}", p_attn_mask_flat);
            // println!("Position IDs: {}", position_ids.get(batch_idx)?);
            // position_ids.slice_assign(&[batch_idx..batch_idx+1, p_attn_mask_flat], &pos_ids)?;
        }

        Ok(embeddings)
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
