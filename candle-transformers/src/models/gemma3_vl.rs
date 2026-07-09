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
        x.broadcast_matmul(&self.mm_input_projection_weight)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    vision_tower: siglip::VisionModel,
    multi_modal_projector: MultiModalProjector,
    language_model: gemma3::Model,
    image_token_index: usize,
    boi_token_index: usize,
    eoi_token_index: usize,
    mm_tokens_per_image: usize,
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
            boi_token_index: cfg.boi_token_index,
            eoi_token_index: cfg.eoi_token_index,
            mm_tokens_per_image: cfg.mm_tokens_per_image,
        })
    }

    pub fn image_token_index(&self) -> usize {
        self.image_token_index
    }

    pub fn dtype(&self) -> DType {
        self.language_model.dtype()
    }

    pub fn expand_image_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        let boi = self.boi_token_index as u32;
        let eoi = self.eoi_token_index as u32;
        let img = self.image_token_index as u32;
        let n = self.mm_tokens_per_image;

        let mut result = Vec::with_capacity(tokens.len() + n);
        let mut i = 0;
        while i < tokens.len() {
            result.push(tokens[i]);
            if tokens[i] == boi && i + 1 < tokens.len() && tokens[i + 1] == eoi {
                result.extend(std::iter::repeat(img).take(n));
            }
            i += 1;
        }
        result
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
        let input_ids_vec = input_ids.squeeze(0)?.to_vec1::<u32>()?;
        let img_token = self.image_token_index as u32;

        let mask_vec: Vec<u8> = input_ids_vec
            .iter()
            .map(|&t| if t == img_token { 1u8 } else { 0u8 })
            .collect();
        let image_mask = Tensor::from_vec(mask_vec, (1, input_ids_vec.len()), input_ids.device())?;

        let clamped_ids = input_ids.clamp(
            0u32,
            (self.language_model.embed_tokens().embeddings().dim(0)? - 1) as u32,
        )?;
        let mut input_embeds = self.language_model.embed_tokens().forward(&clamped_ids)?;

        let hidden_size = self.language_model.embed_tokens().embeddings().dim(1)?;
        input_embeds = (input_embeds * (hidden_size as f64).sqrt())?;

        let pixel_values = pixel_values.to_dtype(input_embeds.dtype())?;
        let vision_outputs = self.vision_tower.forward(&pixel_values)?;
        let image_embeds = self
            .multi_modal_projector
            .forward(&vision_outputs)?
            .to_dtype(input_embeds.dtype())?;

        if let Some(start) = input_ids_vec.iter().position(|&t| t == img_token) {
            let num_image = image_embeds.dim(1)?;
            let seq_len = input_embeds.dim(1)?;
            let before = input_embeds.narrow(1, 0, start)?;
            let after_start = start + num_image;
            let after = input_embeds.narrow(1, after_start, seq_len - after_start)?;
            input_embeds = Tensor::cat(&[&before, &image_embeds, &after], 1)?;
        }

        self.language_model
            .forward_embeds(input_embeds, seqlen_offset, Some(&image_mask))
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}
