//! Basic building blocks for BiRefNet decoder
//!
//! This module contains BasicDecBlk, BasicLatBlk, and SimpleConvs.

use candle::{Module, Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, ModuleT, VarBuilder};

use super::aspp::{ASPPDeformable, ASPP};
use super::config::DecoderAttentionType;

/// Decoder attention module enum
#[allow(clippy::upper_case_acronyms, clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum DecAttention {
    Aspp(ASPP),
    AsppDeformable(ASPPDeformable),
}

impl Module for DecAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            DecAttention::Aspp(aspp) => aspp.forward(xs),
            DecAttention::AsppDeformable(aspp) => aspp.forward(xs),
        }
    }
}

/// Basic Decoder Block
#[derive(Debug, Clone)]
pub struct BasicDecBlk {
    conv_in: Conv2d,
    bn_in: Option<BatchNorm>,
    dec_att: Option<DecAttention>,
    conv_out: Conv2d,
    bn_out: Option<BatchNorm>,
}

impl BasicDecBlk {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        dec_att_type: DecoderAttentionType,
        use_bn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inter_channels = 64;

        let conv_in = candle_nn::conv2d(
            in_channels,
            inter_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_in"),
        )?;

        let bn_in = if use_bn {
            Some(candle_nn::batch_norm(inter_channels, 1e-5, vb.pp("bn_in"))?)
        } else {
            None
        };

        let dec_att = match dec_att_type {
            DecoderAttentionType::ASPP => Some(DecAttention::Aspp(ASPP::new(
                inter_channels,
                None,
                use_bn,
                vb.pp("dec_att"),
            )?)),
            DecoderAttentionType::ASPPDeformable => {
                // Default parallel_block_sizes = [1, 3, 7]
                Some(DecAttention::AsppDeformable(ASPPDeformable::new(
                    inter_channels,
                    None,
                    vec![1, 3, 7],
                    use_bn,
                    vb.pp("dec_att"),
                )?))
            }
            DecoderAttentionType::None => None,
        };

        let conv_out = candle_nn::conv2d(
            inter_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;

        let bn_out = if use_bn {
            Some(candle_nn::batch_norm(out_channels, 1e-5, vb.pp("bn_out"))?)
        } else {
            None
        };

        Ok(Self {
            conv_in,
            bn_in,
            dec_att,
            conv_out,
            bn_out,
        })
    }
}

impl Module for BasicDecBlk {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv_in.forward(xs)?;
        let xs = if let Some(bn) = &self.bn_in {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };
        let xs = xs.relu()?;

        let xs = if let Some(dec_att) = &self.dec_att {
            dec_att.forward(&xs)?
        } else {
            xs
        };

        let xs = self.conv_out.forward(&xs)?;
        let xs = if let Some(bn) = &self.bn_out {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };

        Ok(xs)
    }
}

/// Basic Lateral Block
#[derive(Debug, Clone)]
pub struct BasicLatBlk {
    conv: Conv2d,
}

impl BasicLatBlk {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = candle_nn::conv2d(
            in_channels,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }
}

impl Module for BasicLatBlk {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Simple Convolutions block for decoder input
///
/// Note: This class intentionally does not include activation functions.
/// The output is fused with other features, and activation is applied later.
#[derive(Debug, Clone)]
pub struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        inter_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = candle_nn::conv2d(
            in_channels,
            inter_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv_out = candle_nn::conv2d(
            inter_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;
        Ok(Self { conv1, conv_out })
    }
}

impl Module for SimpleConvs {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.conv1.forward(xs)?)
    }
}
