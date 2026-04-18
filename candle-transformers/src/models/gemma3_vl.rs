//! Gemma 3 Vision-Language (4B/12B/27B) inference implementation.
//!
//! See ["Gemma 3 Technical Report"](https://storage.googleapis.com/deepmind-media/gemma/Gemma3-Report.pdf)

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::gemma3;
use super::siglip;

fn default_mm_tokens_per_image() -> usize {
    256
}
fn default_image_token_index() -> usize {
    262144
}
fn default_boi_token_index() -> usize {
    255999
}
fn default_eoi_token_index() -> usize {
    256000
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub text_config: gemma3::Config,
    pub vision_config: siglip::VisionConfig,
    #[serde(default = "default_mm_tokens_per_image")]
    pub mm_tokens_per_image: usize,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: usize,
    #[serde(default = "default_boi_token_index")]
    pub boi_token_index: usize,
    #[serde(default = "default_eoi_token_index")]
    pub eoi_token_index: usize,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct MultiModalProjector {
    mm_soft_emb_norm: RmsNorm,
    mm_input_projection_weight: Tensor,
    patches_per_image: usize,
    kernel_size: usize,
}

impl MultiModalProjector {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vis_hidden = cfg.vision_config.hidden_size;
        let text_hidden = cfg.text_config.hidden_size;

        let mm_soft_emb_norm = RmsNorm::new(
            vis_hidden,
            cfg.vision_config.layer_norm_eps,
            vb.pp("mm_soft_emb_norm"),
        )?;
        let mm_input_projection_weight =
            vb.get((vis_hidden, text_hidden), "mm_input_projection_weight")?;

        let patches_per_image = cfg.vision_config.image_size / cfg.vision_config.patch_size;
        let tokens_per_side = (cfg.mm_tokens_per_image as f64).sqrt() as usize;
        let kernel_size = patches_per_image / tokens_per_side;

        Ok(Self {
            mm_soft_emb_norm,
            mm_input_projection_weight,
            patches_per_image,
            kernel_size,
        })
    }

    fn forward(&self, vision_outputs: &Tensor) -> Result<Tensor> {
        let (batch, _num_patches, vis_hidden) = vision_outputs.dims3()?;
        let p = self.patches_per_image;
        let k = self.kernel_size;

        let x = vision_outputs.transpose(1, 2)?;
        let x = x.reshape((batch, vis_hidden, p, p))?;
        let x = x.avg_pool2d_with_stride(k, k)?;
        let x = x.flatten(2, 3)?;
        let x = x.transpose(1, 2)?;

        let x = self.mm_soft_emb_norm.forward(&x)?;
        x.matmul(&self.mm_input_projection_weight)
    }
}

fn broadcast_embed_to_mask(embeds: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len) = mask.dims2()?;
    let hidden = embeds.dim(D::Minus1)?;
    let mask_f32 = mask.to_dtype(DType::F32)?;

    let zeros = Tensor::zeros((b_sz, seq_len, hidden), embeds.dtype(), embeds.device())?;

    if b_sz == 1 {
        let num_tokens = mask_f32.sum_all()?.to_scalar::<f32>()? as usize;
        if num_tokens == 0 {
            return Ok(zeros);
        }
        let embed_len = embeds.dim(0)?;
        if embed_len >= seq_len {
            return embeds.narrow(0, 0, seq_len)?.unsqueeze(0);
        }
        let padding = Tensor::zeros(
            (seq_len - embed_len, hidden),
            embeds.dtype(),
            embeds.device(),
        )?;
        let padded = Tensor::cat(&[embeds, &padding], 0)?;
        return padded.unsqueeze(0);
    }

    Ok(zeros)
}

#[derive(Debug, Clone)]
pub struct Model {
    vision_tower: siglip::VisionModel,
    multi_modal_projector: MultiModalProjector,
    language_model: gemma3::Model,
    image_token_index: usize,
}

impl Model {
    pub fn new(use_flash_attn: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vision_tower = siglip::VisionModel::new(
            &cfg.vision_config,
            false,
            vb.pp("vision_tower.vision_model"),
        )?;

        let multi_modal_projector = MultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;

        let language_model =
            gemma3::Model::new(use_flash_attn, &cfg.text_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            image_token_index: cfg.image_token_index,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.language_model.forward(input_ids, seqlen_offset)
    }

    pub fn forward_with_image(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let image_mask = input_ids
            .to_dtype(DType::F32)?
            .eq(self.image_token_index as f64)?;

        let clamped_ids = input_ids.clamp(
            0u32,
            (self.language_model.embed_tokens().embeddings().dim(0)? - 1) as u32,
        )?;
        let mut input_embeds = self.language_model.embed_tokens().forward(&clamped_ids)?;

        let vision_outputs = self.vision_tower.forward(pixel_values)?;
        let image_embeds = self
            .multi_modal_projector
            .forward(&vision_outputs)?
            .to_dtype(input_embeds.dtype())?;

        let image_embeds_flat = image_embeds.squeeze(0)?;
        let mask_expanded = image_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(input_embeds.shape())?
            .to_dtype(input_embeds.dtype())?;
        let image_embeds_broadcast = broadcast_embed_to_mask(&image_embeds_flat, &image_mask)?;
        input_embeds = ((mask_expanded.clone() * image_embeds_broadcast)?
            + ((1.0 - mask_expanded)? * input_embeds)?)?;

        self.language_model
            .forward_embeds(&input_embeds, seqlen_offset, Some(&image_mask))
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}
