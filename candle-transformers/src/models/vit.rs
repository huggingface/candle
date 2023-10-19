#![allow(unused)]
use crate::models::with_tracing::{conv2d, linear, linear_no_bias, Conv2d, Linear};
use candle::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/configuration_vit.py
pub struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: candle_nn::Activation,
    layer_norm_eps: f64,
    image_size: usize,
    patch_size: usize,
    num_channels: usize,
    qkv_bias: bool,
}

impl Config {
    // https://huggingface.co/google/vit-base-patch16-224/blob/main/config.json
    pub fn vit_base_patch16_224() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: true,
        }
    }
}

struct PatchEmbeddings {
    num_patches: usize,
    projection: Conv2d,
}

impl PatchEmbeddings {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let image_size = cfg.image_size;
        let patch_size = cfg.patch_size;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let projection = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            patch_size,
            conv_cfg,
            vb.pp("projection"),
        )?;
        Ok(Self {
            num_patches,
            projection,
        })
    }

    fn forward(&self, pixel_values: &Tensor, interpolate_pos_encoding: bool) -> Result<Tensor> {
        let (b_size, num_channels, height, width) = pixel_values.dims4()?;
        self.projection
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)
    }
}

struct Embeddings {
    cls_token: Tensor,
    mask_token: Option<Tensor>,
    patch_embeddings: PatchEmbeddings,
    position_embeddings: Tensor,
    hidden_size: usize,
}

impl Embeddings {
    fn new(cfg: &Config, use_mask_token: bool, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let cls_token = vb.get((1, 1, hidden_size), "cls_token")?;
        let mask_token = if use_mask_token {
            Some(vb.get((1, 1, hidden_size), "mask_token")?)
        } else {
            None
        };
        let patch_embeddings = PatchEmbeddings::new(cfg, vb.pp("patch_embeddings"))?;
        let num_patches = patch_embeddings.num_patches;
        let position_embeddings =
            vb.get((1, num_patches + 1, hidden_size), "position_embeddings")?;
        Ok(Self {
            cls_token,
            mask_token,
            patch_embeddings,
            position_embeddings,
            hidden_size,
        })
    }

    fn interpolate_pos_encoding(
        &self,
        embeddings: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        todo!()
    }

    fn forward(
        &self,
        pixel_values: &Tensor,
        bool_masked_pos: Option<&Tensor>,
        interpolate_pos_encoding: bool,
    ) -> Result<Tensor> {
        let (b_size, num_channels, height, width) = pixel_values.dims4()?;
        let embeddings = self
            .patch_embeddings
            .forward(pixel_values, interpolate_pos_encoding)?;
        let embeddings = match (bool_masked_pos, &self.mask_token) {
            (None, _) => embeddings,
            (Some(_), None) => candle::bail!("bool_masked_pos set without mask_token"),
            (Some(bool_masked_pos), Some(mask_tokens)) => {
                let seq_len = embeddings.dim(1)?;
                let mask_tokens = mask_tokens.broadcast_as((b_size, seq_len, self.hidden_size))?;
                let mask = bool_masked_pos
                    .unsqueeze(D::Minus1)?
                    .to_dtype(mask_tokens.dtype())?;
                ((mask_tokens * &mask)? - (embeddings * (mask - 1.)?)?)?
            }
        };
        let cls_tokens = self.cls_token.broadcast_as((b_size, 1, self.hidden_size))?;
        let embeddings = Tensor::cat(&[&cls_tokens, &embeddings], 1)?;
        if interpolate_pos_encoding {
            let pos = self.interpolate_pos_encoding(&embeddings, height, width)?;
            embeddings.broadcast_add(&pos)
        } else {
            embeddings.broadcast_add(&self.position_embeddings)
        }
    }
}

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl SelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let num_attention_heads = cfg.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let linear = |name| {
            if cfg.qkv_bias {
                linear(cfg.hidden_size, all_head_size, vb.pp(name))
            } else {
                linear_no_bias(cfg.hidden_size, all_head_size, vb.pp(name))
            }
        };
        let query = linear("query")?;
        let key = linear("key")?;
        let value = linear("value")?;
        Ok(Self {
            query,
            key,
            value,
            num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        xs.reshape((
            b_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .permute((0, 2, 1, 3))
    }

    fn forward(
        &self,
        xs: &Tensor,
        bool_masked_pos: Option<&Tensor>,
        interpolate_pos_encoding: bool,
    ) -> Result<Tensor> {
        let query = self.query.forward(xs)?;
        let key = self.key.forward(xs)?;
        let value = self.value.forward(xs)?;

        let query = self.transpose_for_scores(&query)?;
        let key = self.transpose_for_scores(&key)?;
        let value = self.transpose_for_scores(&value)?;

        let attention_scores =
            (query.matmul(&key.t()?)? / f64::sqrt(self.attention_head_size as f64))?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        attention_probs
            .matmul(&value)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .flatten_from(D::Minus2)
    }
}
