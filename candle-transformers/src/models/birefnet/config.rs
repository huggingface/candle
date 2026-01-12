//! Configuration structures for BiRefNet model

use serde::Deserialize;

/// Configuration loaded from JSON file
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct BiRefNetConfig {
    pub bb_pretrained: bool,
}

/// Main configuration for BiRefNet model
#[derive(Debug, Clone)]
pub struct Config {
    /// Backbone type (Swin-L or Swin-B)
    pub backbone: BackboneType,
    /// Lateral channels from backbone [1536, 768, 384, 192] for Swin-L
    pub lateral_channels: Vec<usize>,

    /// Whether to use decoder input processing
    pub dec_ipt: bool,
    /// Whether to split patches for decoder input
    pub dec_ipt_split: bool,
    /// Number of context features for multi-scale skip connections
    pub cxt_num: usize,
    /// Multi-scale input fusion method
    pub mul_scl_ipt: MultiScaleInputType,
    /// Decoder attention type
    pub dec_att: DecoderAttentionType,
    /// Squeeze block type
    pub squeeze_block: SqueezeBlockType,
    /// Decoder block type
    pub dec_blk: DecoderBlockType,

    /// Whether to use multi-scale supervision
    pub ms_supervision: bool,
    /// Whether to output reference (for training)
    pub out_ref: bool,
}

/// Backbone architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneType {
    /// Swin Transformer Large
    SwinV1L,
    /// Swin Transformer Base
    SwinV1B,
}

/// Multi-scale input fusion method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiScaleInputType {
    /// No multi-scale input
    None,
    /// Add multi-scale features
    Add,
    /// Concatenate multi-scale features
    Cat,
}

/// Decoder attention module type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderAttentionType {
    /// No attention
    None,
    /// Standard ASPP
    ASPP,
    /// ASPP with Deformable Convolution
    ASPPDeformable,
}

/// Squeeze block type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqueezeBlockType {
    /// No squeeze block
    None,
    /// Basic decoder block x1
    BasicDecBlkX1,
}

/// Decoder block type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderBlockType {
    /// Basic decoder block
    BasicDecBlk,
    /// Residual block
    ResBlk,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backbone: BackboneType::SwinV1L,
            lateral_channels: vec![1536, 768, 384, 192],
            dec_ipt: true,
            dec_ipt_split: true,
            cxt_num: 3,
            mul_scl_ipt: MultiScaleInputType::Cat,
            dec_att: DecoderAttentionType::ASPPDeformable,
            squeeze_block: SqueezeBlockType::BasicDecBlkX1,
            dec_blk: DecoderBlockType::BasicDecBlk,
            ms_supervision: true,
            out_ref: true,
        }
    }
}
