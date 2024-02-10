use crate::models::with_tracing::{conv2d, linear, linear_no_bias, Conv2d, Linear};
use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/configuration_vit.py
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub qkv_bias: bool,
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

    pub fn microsoft_trocr_base_handwritten() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 384,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: false,
        }
    }
}

#[derive(Debug, Clone)]
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
}

impl Module for PatchEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (_b_size, _num_channels, _height, _width) = pixel_values.dims4()?;
        self.projection
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
pub struct Embeddings {
    cls_token: Tensor,
    mask_token: Option<Tensor>,
    patch_embeddings: PatchEmbeddings,
    position_embeddings: Tensor,
    hidden_size: usize,
}

impl Embeddings {
    pub fn new(cfg: &Config, use_mask_token: bool, vb: VarBuilder) -> Result<Self> {
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
        _embeddings: &Tensor,
        _height: usize,
        _width: usize,
    ) -> Result<Tensor> {
        todo!()
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        bool_masked_pos: Option<&Tensor>,
        interpolate_pos_encoding: bool,
    ) -> Result<Tensor> {
        let (b_size, _num_channels, height, width) = pixel_values.dims4()?;
        let embeddings = self.patch_embeddings.forward(pixel_values)?;
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

#[derive(Debug, Clone)]
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
}

impl Module for SelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let query = self.query.forward(xs)?;
        let key = self.key.forward(xs)?;
        let value = self.value.forward(xs)?;

        let query = self.transpose_for_scores(&query)?.contiguous()?;
        let key = self.transpose_for_scores(&key)?.contiguous()?;
        let value = self.transpose_for_scores(&value)?.contiguous()?;

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

#[derive(Debug, Clone)]
struct SelfOutput {
    dense: Linear,
}

impl SelfOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl Module for SelfOutput {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    attention: SelfAttention,
    output: SelfOutput,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = SelfAttention::new(cfg, vb.pp("attention"))?;
        let output = SelfOutput::new(cfg, vb.pp("output"))?;
        Ok(Self { attention, output })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.attention)?.apply(&self.output)
    }
}

#[derive(Debug, Clone)]
struct Intermediate {
    dense: Linear,
    intermediate_act_fn: candle_nn::Activation,
}

impl Intermediate {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Intermediate {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)?.apply(&self.intermediate_act_fn)
    }
}

#[derive(Debug, Clone)]
struct Output {
    dense: Linear,
}

impl Output {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)? + input_tensor
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attention: Attention,
    intermediate: Intermediate,
    output: Output,
    layernorm_before: LayerNorm,
    layernorm_after: LayerNorm,
}

impl Layer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = Attention::new(cfg, vb.pp("attention"))?;
        let intermediate = Intermediate::new(cfg, vb.pp("intermediate"))?;
        let output = Output::new(cfg, vb.pp("output"))?;
        let h_sz = cfg.hidden_size;
        let layernorm_before = layer_norm(h_sz, cfg.layer_norm_eps, vb.pp("layernorm_before"))?;
        let layernorm_after = layer_norm(h_sz, cfg.layer_norm_eps, vb.pp("layernorm_after"))?;
        Ok(Self {
            attention,
            intermediate,
            output,
            layernorm_after,
            layernorm_before,
        })
    }
}

impl Module for Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs.apply(&self.layernorm_before)?.apply(&self.attention)? + xs)?;
        let ys = xs.apply(&self.layernorm_after)?.apply(&self.intermediate)?;
        self.output.forward(&ys, &xs)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    layers: Vec<Layer>,
}

impl Encoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("layer");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = Layer::new(cfg, vb.pp(i))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = xs.apply(layer)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embeddings,
    encoder: Encoder,
    layernorm: LayerNorm,
    // no need for pooling layer for image classification
    classifier: Linear,
}

impl Model {
    pub fn new(cfg: &Config, num_labels: usize, vb: VarBuilder) -> Result<Self> {
        let vb_v = vb.pp("vit");
        let embeddings = Embeddings::new(cfg, false, vb_v.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb_v.pp("encoder"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb_v.pp("layernorm"))?;
        let classifier = linear(cfg.hidden_size, num_labels, vb.pp("classifier"))?;
        Ok(Self {
            embeddings,
            encoder,
            layernorm,
            classifier,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(xs, None, false)?;
        let encoder_outputs = self.encoder.forward(&embedding_output)?;
        encoder_outputs
            .i((.., 0, ..))?
            .apply(&self.layernorm)?
            .apply(&self.classifier)
    }
}
