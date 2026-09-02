// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use candle::{streaming, Module, Result, StreamTensor, StreamingModule, Tensor};
use candle_nn::VarBuilder;

use super::conv::{StreamableConv1d, StreamableConvTranspose1d};

#[derive(Debug, Clone)]
pub struct Config {
    pub dimension: usize,
    pub channels: usize,
    pub causal: bool,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub activation: candle_nn::Activation,
    pub norm: super::conv::Norm,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: super::conv::PadMode,
    pub true_skip: bool,
    pub compress: usize,
    pub lstm: usize,
    pub disable_norm_outer_blocks: usize,
    pub final_activation: Option<candle_nn::Activation>,
}

#[derive(Debug, Clone)]
pub struct SeaNetResnetBlock {
    block: Vec<StreamableConv1d>,
    shortcut: Option<StreamableConv1d>,
    activation: candle_nn::Activation,
    skip_op: candle::StreamingBinOp,
    span: tracing::Span,
}

impl SeaNetResnetBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        k_sizes_and_dilations: &[(usize, usize)],
        activation: candle_nn::Activation,
        norm: Option<super::conv::Norm>,
        causal: bool,
        pad_mode: super::conv::PadMode,
        compress: usize,
        true_skip: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut block = Vec::with_capacity(k_sizes_and_dilations.len());
        let hidden = dim / compress;
        let vb_b = vb.pp("block");
        for (i, (k_size, dilation)) in k_sizes_and_dilations.iter().enumerate() {
            let in_c = if i == 0 { dim } else { hidden };
            let out_c = if i == k_sizes_and_dilations.len() - 1 {
                dim
            } else {
                hidden
            };
            let c = StreamableConv1d::new(
                in_c,
                out_c,
                /* k_size */ *k_size,
                /* stride */ 1,
                /* dilation */ *dilation,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ causal,
                /* norm */ norm,
                /* pad_mode */ pad_mode,
                vb_b.pp(2 * i + 1),
            )?;
            block.push(c)
        }
        let shortcut = if true_skip {
            None
        } else {
            let c = StreamableConv1d::new(
                dim,
                dim,
                /* k_size */ 1,
                /* stride */ 1,
                /* dilation */ 1,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ causal,
                /* norm */ norm,
                /* pad_mode */ pad_mode,
                vb.pp("shortcut"),
            )?;
            Some(c)
        };
        Ok(Self {
            block,
            shortcut,
            activation,
            skip_op: streaming::StreamingBinOp::new(streaming::BinOp::Add, candle::D::Minus1),
            span: tracing::span!(tracing::Level::TRACE, "sea-resnet"),
        })
    }
}

impl Module for SeaNetResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut ys = xs.clone();
        for block in self.block.iter() {
            ys = ys.apply(&self.activation)?.apply(block)?;
        }
        match self.shortcut.as_ref() {
            None => ys + xs,
            Some(shortcut) => ys + xs.apply(shortcut),
        }
    }
}

impl StreamingModule for SeaNetResnetBlock {
    fn reset_state(&mut self) {
        for block in self.block.iter_mut() {
            block.reset_state()
        }
        if let Some(shortcut) = self.shortcut.as_mut() {
            shortcut.reset_state()
        }
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let _enter = self.span.enter();
        let mut ys = xs.clone();
        for block in self.block.iter_mut() {
            ys = block.step(&ys.apply(&self.activation)?)?;
        }
        match self.shortcut.as_ref() {
            None => self.skip_op.step(&ys, xs),
            Some(shortcut) => self.skip_op.step(&ys, &xs.apply(shortcut)?),
        }
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    residuals: Vec<SeaNetResnetBlock>,
    downsample: StreamableConv1d,
}

#[derive(Debug, Clone)]
pub struct SeaNetEncoder {
    init_conv1d: StreamableConv1d,
    activation: candle_nn::Activation,
    layers: Vec<EncoderLayer>,
    final_conv1d: StreamableConv1d,
    span: tracing::Span,
}

impl SeaNetEncoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.lstm > 0 {
            candle::bail!("seanet lstm is not supported")
        }
        let n_blocks = 2 + cfg.ratios.len();
        let mut mult = 1usize;
        let init_norm = if cfg.disable_norm_outer_blocks >= 1 {
            None
        } else {
            Some(cfg.norm)
        };
        let mut layer_idx = 0;
        let vb = vb.pp("layers");
        let init_conv1d = StreamableConv1d::new(
            cfg.channels,
            mult * cfg.n_filters,
            cfg.kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ init_norm,
            /* pad_mode */ cfg.pad_mode,
            vb.pp(layer_idx),
        )?;
        layer_idx += 1;
        let mut layers = Vec::with_capacity(cfg.ratios.len());

        for (i, &ratio) in cfg.ratios.iter().rev().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks >= i + 2 {
                None
            } else {
                Some(cfg.norm)
            };
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let resnet_block = SeaNetResnetBlock::new(
                    mult * cfg.n_filters,
                    &[
                        (cfg.residual_kernel_size, cfg.dilation_base.pow(j as u32)),
                        (1, 1),
                    ],
                    cfg.activation,
                    norm,
                    cfg.causal,
                    cfg.pad_mode,
                    cfg.compress,
                    cfg.true_skip,
                    vb.pp(layer_idx),
                )?;
                residuals.push(resnet_block);
                layer_idx += 1;
            }
            let downsample = StreamableConv1d::new(
                mult * cfg.n_filters,
                mult * cfg.n_filters * 2,
                /* k_size */ ratio * 2,
                /* stride */ ratio,
                /* dilation */ 1,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ true,
                /* norm */ norm,
                /* pad_mode */ cfg.pad_mode,
                vb.pp(layer_idx + 1),
            )?;
            layer_idx += 2;
            let layer = EncoderLayer {
                downsample,
                residuals,
            };
            layers.push(layer);
            mult *= 2
        }

        let final_norm = if cfg.disable_norm_outer_blocks >= n_blocks {
            None
        } else {
            Some(cfg.norm)
        };
        let final_conv1d = StreamableConv1d::new(
            mult * cfg.n_filters,
            cfg.dimension,
            cfg.last_kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ final_norm,
            /* pad_mode */ cfg.pad_mode,
            vb.pp(layer_idx + 1),
        )?;
        Ok(Self {
            init_conv1d,
            activation: cfg.activation,
            layers,
            final_conv1d,
            span: tracing::span!(tracing::Level::TRACE, "sea-encoder"),
        })
    }
}

impl Module for SeaNetEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.apply(&self.init_conv1d)?;
        for layer in self.layers.iter() {
            for residual in layer.residuals.iter() {
                xs = xs.apply(residual)?
            }
            xs = xs.apply(&self.activation)?.apply(&layer.downsample)?;
        }
        xs.apply(&self.activation)?.apply(&self.final_conv1d)
    }
}

impl StreamingModule for SeaNetEncoder {
    fn reset_state(&mut self) {
        self.init_conv1d.reset_state();
        self.layers.iter_mut().for_each(|v| {
            v.residuals.iter_mut().for_each(|v| v.reset_state());
            v.downsample.reset_state()
        });
        self.final_conv1d.reset_state();
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let _enter = self.span.enter();
        let mut xs = self.init_conv1d.step(xs)?;
        for layer in self.layers.iter_mut() {
            for residual in layer.residuals.iter_mut() {
                xs = residual.step(&xs)?;
            }
            xs = layer.downsample.step(&xs.apply(&self.activation)?)?;
        }
        self.final_conv1d.step(&xs.apply(&self.activation)?)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    upsample: StreamableConvTranspose1d,
    residuals: Vec<SeaNetResnetBlock>,
}

#[derive(Debug, Clone)]
pub struct SeaNetDecoder {
    init_conv1d: StreamableConv1d,
    activation: candle_nn::Activation,
    layers: Vec<DecoderLayer>,
    final_conv1d: StreamableConv1d,
    final_activation: Option<candle_nn::Activation>,
    span: tracing::Span,
}

impl SeaNetDecoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.lstm > 0 {
            candle::bail!("seanet lstm is not supported")
        }
        let n_blocks = 2 + cfg.ratios.len();
        let mut mult = 1 << cfg.ratios.len();
        let init_norm = if cfg.disable_norm_outer_blocks == n_blocks {
            None
        } else {
            Some(cfg.norm)
        };
        let mut layer_idx = 0;
        let vb = vb.pp("layers");
        let init_conv1d = StreamableConv1d::new(
            cfg.dimension,
            mult * cfg.n_filters,
            cfg.kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ init_norm,
            /* pad_mode */ cfg.pad_mode,
            vb.pp(layer_idx),
        )?;
        layer_idx += 1;
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks + i + 1 >= n_blocks {
                None
            } else {
                Some(cfg.norm)
            };
            let upsample = StreamableConvTranspose1d::new(
                mult * cfg.n_filters,
                mult * cfg.n_filters / 2,
                /* k_size */ ratio * 2,
                /* stride */ ratio,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ true,
                /* norm */ norm,
                vb.pp(layer_idx + 1),
            )?;
            layer_idx += 2;

            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let resnet_block = SeaNetResnetBlock::new(
                    mult * cfg.n_filters / 2,
                    &[
                        (cfg.residual_kernel_size, cfg.dilation_base.pow(j as u32)),
                        (1, 1),
                    ],
                    cfg.activation,
                    norm,
                    cfg.causal,
                    cfg.pad_mode,
                    cfg.compress,
                    cfg.true_skip,
                    vb.pp(layer_idx),
                )?;
                residuals.push(resnet_block);
                layer_idx += 1;
            }
            let layer = DecoderLayer {
                upsample,
                residuals,
            };
            layers.push(layer);
            mult /= 2
        }
        let final_norm = if cfg.disable_norm_outer_blocks >= 1 {
            None
        } else {
            Some(cfg.norm)
        };
        let final_conv1d = StreamableConv1d::new(
            cfg.n_filters,
            cfg.channels,
            cfg.last_kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ final_norm,
            /* pad_mode */ cfg.pad_mode,
            vb.pp(layer_idx + 1),
        )?;
        Ok(Self {
            init_conv1d,
            activation: cfg.activation,
            layers,
            final_conv1d,
            final_activation: cfg.final_activation,
            span: tracing::span!(tracing::Level::TRACE, "sea-decoder"),
        })
    }
}

impl Module for SeaNetDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.apply(&self.init_conv1d)?;
        for layer in self.layers.iter() {
            xs = xs.apply(&self.activation)?.apply(&layer.upsample)?;
            for residual in layer.residuals.iter() {
                xs = xs.apply(residual)?
            }
        }
        let xs = xs.apply(&self.activation)?.apply(&self.final_conv1d)?;
        let xs = match self.final_activation.as_ref() {
            None => xs,
            Some(act) => xs.apply(act)?,
        };
        Ok(xs)
    }
}

impl StreamingModule for SeaNetDecoder {
    fn reset_state(&mut self) {
        self.init_conv1d.reset_state();
        self.layers.iter_mut().for_each(|v| {
            v.residuals.iter_mut().for_each(|v| v.reset_state());
            v.upsample.reset_state()
        });
        self.final_conv1d.reset_state();
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let _enter = self.span.enter();
        let mut xs = self.init_conv1d.step(xs)?;
        for layer in self.layers.iter_mut() {
            xs = layer.upsample.step(&xs.apply(&self.activation)?)?;
            for residual in layer.residuals.iter_mut() {
                xs = residual.step(&xs)?;
            }
        }
        let xs = self.final_conv1d.step(&xs.apply(&self.activation)?)?;
        let xs = match self.final_activation.as_ref() {
            None => xs,
            Some(act) => xs.apply(act)?,
        };
        Ok(xs)
    }
}
