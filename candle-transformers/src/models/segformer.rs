//! Segformer model implementation for semantic segmentation and image classification.
//!
//! Segformer is a transformer-based model designed for vision tasks. It uses a hierarchical
//! structure that progressively generates features at different scales.
//!
//! Key characteristics:
//! - Efficient self-attention with sequence reduction
//! - Hierarchical feature generation
//! - Mix-FFN for local and global feature interaction
//! - Lightweight all-MLP decode head
//!
//! References:
//! - [SegFormer Paper](https://arxiv.org/abs/2105.15203)
//! - [Model Card](https://huggingface.co/nvidia/mit-b0)
//!

use crate::models::with_tracing::{conv2d, linear, Conv2d, Linear};
use candle::{BackendStorage, Context, Module, ModuleT, Result, Tensor, D};
use candle_nn::{conv2d_no_bias, layer_norm, Activation, Conv2dConfig, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/segformer/configuration_segformer.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub id2label: HashMap<String, String>,
    pub num_channels: usize,
    pub num_encoder_blocks: usize,
    pub depths: Vec<usize>,
    pub sr_ratios: Vec<usize>,
    pub hidden_sizes: Vec<usize>,
    pub patch_sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub num_attention_heads: Vec<usize>,
    pub mlp_ratios: Vec<usize>,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
    pub decoder_hidden_size: usize,
}

#[derive(Debug, Clone)]
struct SegformerOverlapPatchEmbeddings<B: BackendStorage> {
    projection: Conv2d<B>,
    layer_norm: candle_nn::LayerNorm<B>,
}

impl<B: BackendStorage> SegformerOverlapPatchEmbeddings<B> {
    fn new(
        config: &Config,
        patch_size: usize,
        stride: usize,
        num_channels: usize,
        hidden_size: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let projection = conv2d(
            num_channels,
            hidden_size,
            patch_size,
            Conv2dConfig {
                stride,
                padding: patch_size / 2,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        let layer_norm =
            candle_nn::layer_norm(hidden_size, config.layer_norm_eps, vb.pp("layer_norm"))?;
        Ok(Self {
            projection,
            layer_norm,
        })
    }
}

impl<B: BackendStorage> Module<B> for SegformerOverlapPatchEmbeddings<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let embeddings = self.projection.forward(x)?;
        let shape = embeddings.shape();
        // [B, C, H, W] -> [B, H * W, C]
        let embeddings = embeddings.flatten_from(2)?.transpose(1, 2)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        // [B, H * W, C] -> [B, C, H, W]
        let embeddings = embeddings.transpose(1, 2)?.reshape(shape)?;
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
struct SegformerEfficientSelfAttention<B: BackendStorage> {
    num_attention_heads: usize,
    attention_head_size: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    sr: Option<Conv2d<B>>,
    layer_norm: Option<layer_norm::LayerNorm<B>>,
}

impl<B: BackendStorage> SegformerEfficientSelfAttention<B> {
    fn new(
        config: &Config,
        hidden_size: usize,
        num_attention_heads: usize,
        sequence_reduction_ratio: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        if hidden_size % num_attention_heads != 0 {
            candle::bail!(
                "The hidden size {} is not a multiple of the number of attention heads {}",
                hidden_size,
                num_attention_heads
            )
        }
        let attention_head_size = hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let (sr, layer_norm) = if sequence_reduction_ratio > 1 {
            (
                Some(conv2d(
                    hidden_size,
                    hidden_size,
                    sequence_reduction_ratio,
                    Conv2dConfig {
                        stride: sequence_reduction_ratio,
                        ..Default::default()
                    },
                    vb.pp("sr"),
                )?),
                Some(candle_nn::layer_norm(
                    hidden_size,
                    config.layer_norm_eps,
                    vb.pp("layer_norm"),
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            num_attention_heads,
            attention_head_size,
            query,
            key,
            value,
            sr,
            layer_norm,
        })
    }

    fn transpose_for_scores(&self, hidden_states: Tensor<B>) -> Result<Tensor<B>> {
        let (batch, seq_length, _) = hidden_states.shape().dims3()?;
        let new_shape = &[
            batch,
            seq_length,
            self.num_attention_heads,
            self.attention_head_size,
        ];
        let hidden_states = hidden_states.reshape(new_shape)?;
        let hidden_states = hidden_states.permute((0, 2, 1, 3))?;
        Ok(hidden_states)
    }
}

impl<B: BackendStorage> Module<B> for SegformerEfficientSelfAttention<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // [B, C, H, W] -> [B, H * W, C]
        let hidden_states = x.flatten_from(2)?.permute((0, 2, 1))?;
        let query = self
            .transpose_for_scores(self.query.forward(&hidden_states)?)?
            .contiguous()?;
        let hidden_states = if let (Some(sr), Some(layer_norm)) = (&self.sr, &self.layer_norm) {
            let hidden_states = sr.forward(x)?;
            // [B, C, H, W] -> [B, H * W, C]
            let hidden_states = hidden_states.flatten_from(2)?.permute((0, 2, 1))?;
            layer_norm.forward(&hidden_states)?
        } else {
            // already [B, H * W, C]
            hidden_states
        };
        // standard self-attention
        let key = self
            .transpose_for_scores(self.key.forward(&hidden_states)?)?
            .contiguous()?;
        let value = self
            .transpose_for_scores(self.value.forward(&hidden_states)?)?
            .contiguous()?;
        let attention_scores =
            (query.matmul(&key.t()?)? / f64::sqrt(self.attention_head_size as f64))?;
        let attention_scores = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let result = attention_scores.matmul(&value)?;
        let result = result.permute((0, 2, 1, 3))?.contiguous()?;
        result.flatten_from(D::Minus2)
    }
}

#[derive(Debug, Clone)]
struct SegformerSelfOutput<B: BackendStorage> {
    dense: Linear<B>,
}

impl<B: BackendStorage> SegformerSelfOutput<B> {
    fn new(hidden_size: usize, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl<B: BackendStorage> Module<B> for SegformerSelfOutput<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        self.dense.forward(x)
    }
}

#[derive(Debug, Clone)]
struct SegformerAttention<B: BackendStorage> {
    attention: SegformerEfficientSelfAttention<B>,
    output: SegformerSelfOutput<B>,
}

impl<B: BackendStorage> SegformerAttention<B> {
    fn new(
        config: &Config,
        hidden_size: usize,
        num_attention_heads: usize,
        sequence_reduction_ratio: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let attention = SegformerEfficientSelfAttention::new(
            config,
            hidden_size,
            num_attention_heads,
            sequence_reduction_ratio,
            vb.pp("self"),
        )?;
        let output = SegformerSelfOutput::new(hidden_size, vb.pp("output"))?;
        Ok(Self { attention, output })
    }
}

impl<B: BackendStorage> Module<B> for SegformerAttention<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let attention_output = self.attention.forward(x)?;
        self.output.forward(&attention_output)
    }
}

#[derive(Debug, Clone)]
struct SegformerDWConv<B: BackendStorage> {
    dw_conv: Conv2d<B>,
}

impl<B: BackendStorage> SegformerDWConv<B> {
    fn new(dim: usize, vb: VarBuilder<B>) -> Result<Self> {
        let dw_conv = conv2d(
            dim,
            dim,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                groups: dim,
                ..Default::default()
            },
            vb.pp("dwconv"),
        )?;
        Ok(Self { dw_conv })
    }
}

impl<B: BackendStorage> Module<B> for SegformerDWConv<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        self.dw_conv.forward(x)
    }
}

#[derive(Debug, Clone)]
struct SegformerMixFFN<B: BackendStorage> {
    dense1: Linear<B>,
    dw_conv: SegformerDWConv<B>,
    act: Activation,
    dense2: Linear<B>,
}

impl<B: BackendStorage> SegformerMixFFN<B> {
    fn new(
        config: &Config,
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let dense1 = linear(in_features, hidden_features, vb.pp("dense1"))?;
        let dw_conv = SegformerDWConv::new(hidden_features, vb.pp("dwconv"))?;
        let act = config.hidden_act;
        let dense2 = linear(hidden_features, out_features, vb.pp("dense2"))?;
        Ok(Self {
            dense1,
            dw_conv,
            act,
            dense2,
        })
    }
}

impl<B: BackendStorage> Module<B> for SegformerMixFFN<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let (batch, _, height, width) = x.shape().dims4()?;
        let hidden_states = self
            .dense1
            .forward(&x.flatten_from(2)?.permute((0, 2, 1))?)?;
        let channels = hidden_states.dim(2)?;
        let hidden_states = self.dw_conv.forward(
            &hidden_states
                .permute((0, 2, 1))?
                .reshape((batch, channels, height, width))?,
        )?;
        let hidden_states = self.act.forward(&hidden_states)?;
        let hidden_states = self
            .dense2
            .forward(&hidden_states.flatten_from(2)?.permute((0, 2, 1))?)?;
        let channels = hidden_states.dim(2)?;
        hidden_states
            .permute((0, 2, 1))?
            .reshape((batch, channels, height, width))
    }
}

#[derive(Debug, Clone)]
struct SegformerLayer<B: BackendStorage> {
    layer_norm_1: candle_nn::LayerNorm<B>,
    attention: SegformerAttention<B>,
    layer_norm_2: candle_nn::LayerNorm<B>,
    mlp: SegformerMixFFN<B>,
}

impl<B: BackendStorage> SegformerLayer<B> {
    fn new(
        config: &Config,
        hidden_size: usize,
        num_attention_heads: usize,
        sequence_reduction_ratio: usize,
        mlp_ratio: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let layer_norm_1 = layer_norm(hidden_size, config.layer_norm_eps, vb.pp("layer_norm_1"))?;
        let attention = SegformerAttention::new(
            config,
            hidden_size,
            num_attention_heads,
            sequence_reduction_ratio,
            vb.pp("attention"),
        )?;
        let layer_norm_2 = layer_norm(hidden_size, config.layer_norm_eps, vb.pp("layer_norm_2"))?;
        let mlp = SegformerMixFFN::new(
            config,
            hidden_size,
            hidden_size * mlp_ratio,
            hidden_size,
            vb.pp("mlp"),
        )?;
        Ok(Self {
            layer_norm_1,
            attention,
            layer_norm_2,
            mlp,
        })
    }
}

impl<B: BackendStorage> Module<B> for SegformerLayer<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let shape = x.shape().dims4()?;
        // [B, C, H, W] -> [B, H * W, C]
        let hidden_states = x.flatten_from(2)?.permute((0, 2, 1))?;
        let layer_norm_output = self.layer_norm_1.forward(&hidden_states)?;
        let layer_norm_output = layer_norm_output.permute((0, 2, 1))?.reshape(shape)?;
        // attention takes in [B, C, H, W] in order to properly do conv2d (and output [B, H * W, C])
        let attention_output = self.attention.forward(&layer_norm_output)?;
        let hidden_states = (attention_output + hidden_states)?;
        let layer_norm_output = self.layer_norm_2.forward(&hidden_states)?;
        let mlp_output = self
            .mlp
            .forward(&layer_norm_output.permute((0, 2, 1))?.reshape(shape)?)?;
        hidden_states.permute((0, 2, 1))?.reshape(shape)? + mlp_output
    }
}

#[derive(Debug, Clone)]
struct SegformerEncoder<B: BackendStorage> {
    /// config file
    config: Config,
    /// a list of embeddings
    patch_embeddings: Vec<SegformerOverlapPatchEmbeddings<B>>,
    /// a list of attention blocks, each consisting of layers
    blocks: Vec<Vec<SegformerLayer<B>>>,
    /// a final list of layer norms
    layer_norms: Vec<candle_nn::LayerNorm<B>>,
}

impl<B: BackendStorage> SegformerEncoder<B> {
    fn new(config: Config, vb: VarBuilder<B>) -> Result<Self> {
        let mut patch_embeddings = Vec::with_capacity(config.num_encoder_blocks);
        let mut blocks = Vec::with_capacity(config.num_encoder_blocks);
        let mut layer_norms = Vec::with_capacity(config.num_encoder_blocks);
        for i in 0..config.num_encoder_blocks {
            let patch_size = config.patch_sizes[i];
            let stride = config.strides[i];
            let hidden_size = config.hidden_sizes[i];
            let num_channels = if i == 0 {
                config.num_channels
            } else {
                config.hidden_sizes[i - 1]
            };
            patch_embeddings.push(SegformerOverlapPatchEmbeddings::new(
                &config,
                patch_size,
                stride,
                num_channels,
                hidden_size,
                vb.pp(format!("patch_embeddings.{i}")),
            )?);
            let mut layers = Vec::with_capacity(config.depths[i]);
            for j in 0..config.depths[i] {
                let sequence_reduction_ratio = config.sr_ratios[i];
                let num_attention_heads = config.num_attention_heads[i];
                let mlp_ratio = config.mlp_ratios[i];
                layers.push(SegformerLayer::new(
                    &config,
                    hidden_size,
                    num_attention_heads,
                    sequence_reduction_ratio,
                    mlp_ratio,
                    vb.pp(format!("block.{i}.{j}")),
                )?);
            }
            blocks.push(layers);
            layer_norms.push(layer_norm(
                hidden_size,
                config.layer_norm_eps,
                vb.pp(format!("layer_norm.{i}")),
            )?);
        }
        Ok(Self {
            config,
            patch_embeddings,
            blocks,
            layer_norms,
        })
    }
}

impl<B: BackendStorage> ModuleWithHiddenStates<B> for SegformerEncoder<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Vec<Tensor<B>>> {
        let mut all_hidden_states = Vec::with_capacity(self.config.num_encoder_blocks);
        let mut hidden_states = x.clone();
        for i in 0..self.config.num_encoder_blocks {
            hidden_states = self.patch_embeddings[i].forward(&hidden_states)?;
            for layer in &self.blocks[i] {
                hidden_states = layer.forward(&hidden_states)?;
            }
            let shape = hidden_states.shape().dims4()?;
            hidden_states =
                self.layer_norms[i].forward(&hidden_states.flatten_from(2)?.permute((0, 2, 1))?)?;
            hidden_states = hidden_states.permute((0, 2, 1))?.reshape(shape)?;
            all_hidden_states.push(hidden_states.clone());
        }
        Ok(all_hidden_states)
    }
}

#[derive(Debug, Clone)]
struct SegformerModel<B: BackendStorage> {
    encoder: SegformerEncoder<B>,
}

impl<B: BackendStorage> SegformerModel<B> {
    fn new(config: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let encoder = SegformerEncoder::new(config.clone(), vb.pp("encoder"))?;
        Ok(Self { encoder })
    }
}

impl<B: BackendStorage> ModuleWithHiddenStates<B> for SegformerModel<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Vec<Tensor<B>>> {
        self.encoder.forward(x)
    }
}

#[derive(Debug, Clone)]
struct SegformerMLP<B: BackendStorage> {
    proj: Linear<B>,
}

impl<B: BackendStorage> SegformerMLP<B> {
    fn new(config: &Config, input_dim: usize, vb: VarBuilder<B>) -> Result<Self> {
        let proj = linear(input_dim, config.decoder_hidden_size, vb.pp("proj"))?;
        Ok(Self { proj })
    }
}

impl<B: BackendStorage> Module for SegformerMLP<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        self.proj.forward(x)
    }
}

#[derive(Debug, Clone)]
struct SegformerDecodeHead<B: BackendStorage> {
    linear_c: Vec<SegformerMLP<B>>,
    linear_fuse: candle_nn::Conv2d<B>,
    batch_norm: candle_nn::BatchNorm<B>,
    classifier: candle_nn::Conv2d<B>,
}

impl<B: BackendStorage> SegformerDecodeHead<B> {
    fn new(config: &Config, num_labels: usize, vb: VarBuilder<B>) -> Result<Self> {
        let mut linear_c = Vec::with_capacity(config.num_encoder_blocks);
        for i in 0..config.num_encoder_blocks {
            let hidden_size = config.hidden_sizes[i];
            linear_c.push(SegformerMLP::new(
                config,
                hidden_size,
                vb.pp(format!("linear_c.{i}")),
            )?);
        }
        let linear_fuse = conv2d_no_bias(
            config.decoder_hidden_size * config.num_encoder_blocks,
            config.decoder_hidden_size,
            1,
            Conv2dConfig::default(),
            vb.pp("linear_fuse"),
        )?;
        let batch_norm = candle_nn::batch_norm(
            config.decoder_hidden_size,
            config.layer_norm_eps,
            vb.pp("batch_norm"),
        )?;
        let classifier = conv2d_no_bias(
            config.decoder_hidden_size,
            num_labels,
            1,
            Conv2dConfig::default(),
            vb.pp("classifier"),
        )?;
        Ok(Self {
            linear_c,
            linear_fuse,
            batch_norm,
            classifier,
        })
    }

    fn forward(&self, encoder_hidden_states: &[Tensor<B>]) -> Result<Tensor<B>> {
        if encoder_hidden_states.len() != self.linear_c.len() {
            candle::bail!(
                "The number of encoder hidden states {} is not equal to the number of linear layers {}",
                encoder_hidden_states.len(),
                self.linear_c.len()
            )
        }
        // most fine layer
        let (_, _, upsample_height, upsample_width) = encoder_hidden_states[0].shape().dims4()?;
        let mut hidden_states = Vec::with_capacity(self.linear_c.len());
        for (hidden_state, mlp) in encoder_hidden_states.iter().zip(&self.linear_c) {
            let (batch, _, height, width) = hidden_state.shape().dims4()?;
            let hidden_state = mlp.forward(&hidden_state.flatten_from(2)?.permute((0, 2, 1))?)?;
            let hidden_state = hidden_state.permute((0, 2, 1))?.reshape((
                batch,
                hidden_state.dim(2)?,
                height,
                width,
            ))?;
            let hidden_state = hidden_state.upsample_nearest2d(upsample_height, upsample_width)?;
            hidden_states.push(hidden_state);
        }
        hidden_states.reverse();
        let hidden_states = Tensor::cat(&hidden_states, 1)?;
        let hidden_states = self.linear_fuse.forward(&hidden_states)?;
        let hidden_states = self.batch_norm.forward_t(&hidden_states, false)?;
        let hidden_states = hidden_states.relu()?;
        self.classifier.forward(&hidden_states)
    }
}

trait ModuleWithHiddenStates<B: BackendStorage> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Vec<Tensor<B>>>;
}

#[derive(Debug, Clone)]
pub struct SemanticSegmentationModel<B: BackendStorage> {
    segformer: SegformerModel<B>,
    decode_head: SegformerDecodeHead<B>,
}

impl<B: BackendStorage> SemanticSegmentationModel<B> {
    pub fn new(config: &Config, num_labels: usize, vb: VarBuilder<B>) -> Result<Self> {
        let segformer = SegformerModel::new(config, vb.pp("segformer"))?;
        let decode_head = SegformerDecodeHead::new(config, num_labels, vb.pp("decode_head"))?;
        Ok(Self {
            segformer,
            decode_head,
        })
    }
}

impl<B: BackendStorage> Module<B> for SemanticSegmentationModel<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let hidden_states = self.segformer.forward(xs)?;
        self.decode_head.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct ImageClassificationModel<B: BackendStorage> {
    segformer: SegformerModel<B>,
    classifier: Linear<B>,
}

impl<B: BackendStorage> ImageClassificationModel<B> {
    pub fn new(config: &Config, num_labels: usize, vb: VarBuilder<B>) -> Result<Self> {
        let segformer = SegformerModel::new(config, vb.pp("segformer"))?;
        let classifier = linear(config.decoder_hidden_size, num_labels, vb.pp("classifier"))?;
        Ok(Self {
            segformer,
            classifier,
        })
    }
}

impl<B: BackendStorage> Module<B> for ImageClassificationModel<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let all_hidden_states = self.segformer.forward(x)?;
        let hidden_states = all_hidden_states.last().context("no last")?;
        let hidden_states = hidden_states.flatten_from(2)?.permute((0, 2, 1))?;
        let mean = hidden_states.mean(1)?;
        self.classifier.forward(&mean)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_config_json_load() {
        let raw_json = r#"{
            "architectures": [
              "SegformerForImageClassification"
            ],
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout_prob": 0.1,
            "decoder_hidden_size": 256,
            "depths": [
              2,
              2,
              2,
              2
            ],
            "downsampling_rates": [
              1,
              4,
              8,
              16
            ],
            "drop_path_rate": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_sizes": [
              32,
              64,
              160,
              256
            ],
            "image_size": 224,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-06,
            "mlp_ratios": [
              4,
              4,
              4,
              4
            ],
            "model_type": "segformer",
            "num_attention_heads": [
              1,
              2,
              5,
              8
            ],
            "num_channels": 3,
            "num_encoder_blocks": 4,
            "patch_sizes": [
              7,
              3,
              3,
              3
            ],
            "sr_ratios": [
              8,
              4,
              2,
              1
            ],
            "strides": [
              4,
              2,
              2,
              2
            ],
            "torch_dtype": "float32",
            "transformers_version": "4.12.0.dev0"
          }"#;
        let config: Config = serde_json::from_str(raw_json).unwrap();
        assert_eq!(vec![4, 2, 2, 2], config.strides);
        assert_eq!(1e-6, config.layer_norm_eps);
    }
}
