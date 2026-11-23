pub mod config;
mod projector;
mod vision;

use crate::models::lfm2::{Cache, Model as Lfm2};

use candle::{Result, Tensor};

use candle_nn::VarBuilder;

pub use config::Lfm2VlConfig;
use projector::Lfm2MultiModalProjector;
use vision::Siglip2VisionModel;

/// Image inputs from the processor: `pixel_values` `(N, max_num_patches, C * patch_size^2)`,
/// `spatial_shapes` `(h, w)` in patches per image, `pixel_attention_mask` `(N, max_num_patches)`.
pub type ImageInputs<'a> = (&'a Tensor, &'a [(usize, usize)], &'a Tensor);

pub struct Lfm2VlModel {
    vision_model: Siglip2VisionModel,
    text_model: Lfm2,
    projector: Lfm2MultiModalProjector,
    image_token_id: u32,
}

impl Lfm2VlModel {
    pub fn new(cfg: &Lfm2VlConfig, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let vision_model = Siglip2VisionModel::new(
            &cfg.vision_config,
            vb_m.pp("vision_tower").pp("vision_model"),
        )?;

        let text_cfg = cfg.text_config.clone().into_config(use_flash_attn);
        let text_model =
            Lfm2::new_with_vbs(&text_cfg, vb_m.pp("language_model"), vb.pp("lm_head"))?;

        let projector = Lfm2MultiModalProjector::new(cfg, vb_m.pp("multi_modal_projector"))?;

        Ok(Self {
            vision_model,
            text_model,
            projector,
            image_token_id: cfg.image_token_id,
        })
    }

    /// Returns one row per `<image>` token, `(sum of h/2 * w/2 over images, text hidden_size)`,
    /// in the same order as the image tokens in the prompt.
    fn get_image_features(
        &self,
        pixel_values: &Tensor,
        spatial_shapes: &[(usize, usize)],
        pixel_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let last_hidden_state =
            self.vision_model
                .forward(pixel_values, spatial_shapes, pixel_attention_mask)?;

        // Images have different sizes, so each one is unpadded and projected on its own.
        // Padding is always at the end, so the real patches are the first `h * w` rows.
        let image_features = spatial_shapes
            .iter()
            .enumerate()
            .map(|(i, &(h, w))| {
                last_hidden_state
                    .get(i)? // (max_num_patches, vision hidden)
                    .narrow(0, 0, h * w)? // (h * w, vision hidden)
                    .reshape((1, h, w, ()))? // (1, h, w, vision hidden)
                    .apply(&self.projector)? // (1, h / 2, w / 2, text hidden)
                    .squeeze(0)?
                    .flatten(0, 1) // (h / 2 * w / 2, text hidden)
            })
            .collect::<Result<Vec<_>>>()?;

        Tensor::cat(&image_features, 0)
    }

    /// Replaces the embedding of each `<image>` token with the next row of `image_features`,
    /// like `masked_scatter` in transformers. Only batch size 1 is supported.
    fn merge_image_features(
        &self,
        input_ids: &Tensor,
        text_embeds: &Tensor,
        image_features: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = text_embeds.dims3()?;
        if batch_size != 1 {
            candle::bail!("lfm2-vl only supports batch size 1, got {batch_size}")
        }
        let ids = input_ids.squeeze(0)?.to_vec1::<u32>()?;

        // `combined` stacks the text rows then the image rows; each prompt position picks
        // its own text row, or the next image row when it holds an image token.
        let mut next_image_row = seq_len;
        let indices: Vec<u32> = ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| {
                if id == self.image_token_id {
                    next_image_row += 1;
                    (next_image_row - 1) as u32
                } else {
                    pos as u32
                }
            })
            .collect();

        let n_image_tokens = next_image_row - seq_len;
        let n_image_features = image_features.dim(0)?;
        if n_image_tokens != n_image_features {
            candle::bail!(
                "image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}"
            )
        }

        let combined = Tensor::cat(
            &[
                &text_embeds.squeeze(0)?,
                &image_features.to_dtype(text_embeds.dtype())?,
            ],
            0,
        )?;
        let indices = Tensor::new(indices, text_embeds.device())?;
        combined.index_select(&indices, 0)?.unsqueeze(0)
    }

    /// Returns the logits of the last position. Pass `images` on the first call (the prompt
    /// with its image tokens) and `None` for the following generated tokens.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        images: Option<ImageInputs>,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let embeds = self.text_model.embed_tokens(input_ids)?;
        let embeds = match images {
            Some((pixel_values, spatial_shapes, pixel_attention_mask)) => {
                let image_features =
                    self.get_image_features(pixel_values, spatial_shapes, pixel_attention_mask)?;
                self.merge_image_features(input_ids, &embeds, &image_features)?
            }
            None => embeds,
        };
        self.text_model.forward_embeds(&embeds, index_pos, cache)
    }
}
