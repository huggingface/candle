// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use candle::{Module, Result, StreamTensor, StreamingModule, Tensor, D};
use candle_nn::{Conv1d, VarBuilder};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Norm {
    WeightNorm,
    SpectralNorm,
    TimeGroupNorm,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Reflect,
    Replicate,
}

// Applies weight norm for inference by recomputing the weight tensor. This
// does not apply to training.
// https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = if vb.contains_tensor("weight") {
        vb.get((out_c, in_c, kernel_size), "weight")?
    } else {
        let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
        let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?
    };
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

#[derive(Debug, Clone)]
pub struct NormConv1d {
    conv: Conv1d,
    norm: Option<candle_nn::GroupNorm>,
    span: tracing::Span,
}

impl NormConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_c: usize,
        out_c: usize,
        k_size: usize,
        causal: bool,
        norm: Option<Norm>,
        bias: bool,
        cfg: candle_nn::Conv1dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = match norm {
            None | Some(Norm::TimeGroupNorm) => {
                if bias {
                    candle_nn::conv1d(in_c, out_c, k_size, cfg, vb.pp("conv"))?
                } else {
                    candle_nn::conv1d_no_bias(in_c, out_c, k_size, cfg, vb.pp("conv"))?
                }
            }
            Some(Norm::WeightNorm) => {
                conv1d_weight_norm(in_c, out_c, k_size, bias, cfg, vb.pp("conv"))?
            }
            Some(Norm::SpectralNorm) => candle::bail!("SpectralNorm is not supported yet."),
        };
        let norm = match norm {
            None | Some(Norm::WeightNorm) | Some(Norm::SpectralNorm) => None,
            Some(Norm::TimeGroupNorm) => {
                if causal {
                    candle::bail!("GroupNorm doesn't support causal evaluation.")
                }
                let norm = candle_nn::group_norm(1, out_c, 1e-5, vb.pp("norm"))?;
                Some(norm)
            }
        };
        Ok(Self {
            conv,
            norm,
            span: tracing::span!(tracing::Level::TRACE, "norm-conv1d"),
        })
    }
}

impl Module for NormConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = xs.apply(&self.conv)?;
        match self.norm.as_ref() {
            None => Ok(xs),
            Some(norm) => xs.apply(norm),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormConvTranspose1d {
    ws: Tensor,
    bs: Option<Tensor>,
    k_size: usize,
    stride: usize,
    groups: usize,
    norm: Option<candle_nn::GroupNorm>,
    span: tracing::Span,
}

impl NormConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_c: usize,
        out_c: usize,
        k_size: usize,
        causal: bool,
        norm: Option<Norm>,
        bias: bool,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("conv");
        let bs = if bias {
            Some(vb.get(out_c, "bias")?)
        } else {
            None
        };
        let ws = match norm {
            None | Some(Norm::TimeGroupNorm) => vb.get((in_c, out_c / groups, k_size), "weight")?,
            Some(Norm::WeightNorm) => {
                if vb.contains_tensor("weight") {
                    vb.get((in_c, out_c, k_size), "weight")?
                } else {
                    let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
                    let weight_v = vb.get((in_c, out_c, k_size), "weight_v")?;
                    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
                    weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?
                }
            }
            Some(Norm::SpectralNorm) => candle::bail!("SpectralNorm is not supported yet."),
        };
        let (ws, groups) = if groups == out_c && in_c == out_c {
            let eye = Tensor::eye(out_c, ws.dtype(), ws.device())?;
            let ws = ws
                .repeat((1, out_c, 1))?
                .mul(&eye.unsqueeze(2)?.repeat((1, 1, k_size))?)?;
            (ws, 1)
        } else {
            (ws, groups)
        };
        let norm = match norm {
            None | Some(Norm::WeightNorm) | Some(Norm::SpectralNorm) => None,
            Some(Norm::TimeGroupNorm) => {
                if causal {
                    candle::bail!("GroupNorm doesn't support causal evaluation.")
                }
                let norm = candle_nn::group_norm(1, out_c, 1e-5, vb.pp("norm"))?;
                Some(norm)
            }
        };
        Ok(Self {
            ws,
            bs,
            k_size,
            stride,
            groups,
            norm,
            span: tracing::span!(tracing::Level::TRACE, "norm-conv-tr1d"),
        })
    }
}

impl Module for NormConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // conv-transpose1d seems to be broken on metal after enough iterations. Causing
        // the following error:
        // _status < MTLCommandBufferStatusCommitted >
        // -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
        // This is now fixed in candle.
        let xs = Tensor::conv_transpose1d(xs, &self.ws, 0, 0, self.stride, 1, self.groups)?;
        let xs = match &self.bs {
            None => xs,
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                xs.broadcast_add(&bias)?
            }
        };
        match self.norm.as_ref() {
            None => Ok(xs),
            Some(norm) => xs.apply(norm),
        }
    }
}

fn get_extra_padding_for_conv1d(
    xs: &Tensor,
    k_size: usize,
    stride: usize,
    padding_total: usize,
) -> Result<usize> {
    let len = xs.dim(D::Minus1)?;
    let n_frames = (len + padding_total).saturating_sub(k_size) as f64 / stride as f64 + 1.0;
    let ideal_len =
        ((n_frames.ceil() as usize - 1) * stride + k_size).saturating_sub(padding_total);
    Ok(ideal_len.saturating_sub(len))
}

fn pad1d(xs: &Tensor, pad_l: usize, pad_r: usize, mode: PadMode) -> Result<Tensor> {
    match mode {
        PadMode::Constant => xs.pad_with_zeros(D::Minus1, pad_l, pad_r),
        PadMode::Reflect => candle::bail!("pad-mode 'reflect' is not supported"),
        PadMode::Replicate => xs.pad_with_same(D::Minus1, pad_l, pad_r),
    }
}

fn unpad1d(xs: &Tensor, unpad_l: usize, unpad_r: usize) -> Result<Tensor> {
    let len = xs.dim(D::Minus1)?;
    if len < unpad_l + unpad_r {
        candle::bail!("unpad1d: tensor len {len} is too low, {unpad_l} + {unpad_r}")
    }
    xs.narrow(D::Minus1, unpad_l, len - (unpad_l + unpad_r))
}

#[derive(Debug, Clone)]
pub struct StreamableConv1d {
    conv: NormConv1d,
    causal: bool,
    pad_mode: PadMode,
    state_prev_xs: StreamTensor,
    left_pad_applied: bool,
    kernel_size: usize,
    span: tracing::Span,
}

impl StreamableConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_c: usize,
        out_c: usize,
        k_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        norm: Option<Norm>,
        pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
        };
        let conv = NormConv1d::new(in_c, out_c, k_size, causal, norm, bias, cfg, vb)?;
        if k_size < stride {
            candle::bail!("kernel-size {k_size} is smaller than stride {stride}")
        }
        Ok(Self {
            conv,
            causal,
            pad_mode,
            state_prev_xs: StreamTensor::empty(),
            left_pad_applied: false,
            kernel_size: k_size,
            span: tracing::span!(tracing::Level::TRACE, "streamable-conv1d"),
        })
    }
}

impl Module for StreamableConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_b, _t, _c) = xs.dims3()?;
        let k_size = self.conv.conv.weight().dim(D::Minus1)?;
        let conv_cfg = self.conv.conv.config();
        // Effective kernel size with dilations.
        let k_size = (k_size - 1) * conv_cfg.dilation + 1;
        let padding_total = k_size - conv_cfg.stride;
        let extra_padding =
            get_extra_padding_for_conv1d(xs, k_size, conv_cfg.stride, padding_total)?;
        let xs = if self.causal {
            pad1d(xs, padding_total, extra_padding, self.pad_mode)?
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            pad1d(
                xs,
                padding_left,
                padding_right + extra_padding,
                self.pad_mode,
            )?
        };
        xs.apply(&self.conv)
    }
}

impl StreamingModule for StreamableConv1d {
    fn reset_state(&mut self) {
        self.state_prev_xs.reset();
        self.left_pad_applied = false;
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let _enter = self.span.enter();
        let xs = match xs.as_option() {
            None => return Ok(().into()),
            Some(xs) => xs.clone(),
        };
        let xs = if self.left_pad_applied {
            xs
        } else {
            self.left_pad_applied = true;
            let k_size = self.conv.conv.weight().dim(D::Minus1)?;
            let conv_cfg = self.conv.conv.config();
            let k_size = (k_size - 1) * conv_cfg.dilation + 1;
            let padding_total = k_size - conv_cfg.stride;
            pad1d(&xs, padding_total, 0, self.pad_mode)?
        };
        let cfg = self.conv.conv.config();
        let stride = cfg.stride;
        let dilation = cfg.dilation;
        let kernel = (self.kernel_size - 1) * dilation + 1;
        let xs = StreamTensor::cat2(&self.state_prev_xs, &xs.into(), D::Minus1)?;
        let seq_len = xs.seq_len(D::Minus1)?;
        let num_frames = (seq_len + stride).saturating_sub(kernel) / stride;
        if num_frames > 0 {
            let offset = num_frames * stride;
            self.state_prev_xs = xs.narrow(D::Minus1, offset, seq_len - offset)?;
            let in_l = (num_frames - 1) * stride + kernel;
            let xs = xs.narrow(D::Minus1, 0, in_l)?;
            // We apply the underlying convtr directly rather than through forward so as
            // not to apply any padding here.
            xs.apply(&self.conv.conv)
        } else {
            self.state_prev_xs = xs;
            Ok(StreamTensor::empty())
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamableConvTranspose1d {
    convtr: NormConvTranspose1d,
    causal: bool,
    state_prev_ys: StreamTensor,
    kernel_size: usize,
    span: tracing::Span,
}

impl StreamableConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_c: usize,
        out_c: usize,
        k_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        norm: Option<Norm>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let convtr =
            NormConvTranspose1d::new(in_c, out_c, k_size, causal, norm, bias, stride, groups, vb)?;
        Ok(Self {
            convtr,
            causal,
            kernel_size: k_size,
            state_prev_ys: StreamTensor::empty(),
            span: tracing::span!(tracing::Level::TRACE, "streamable-conv-tr1d"),
        })
    }
}

impl Module for StreamableConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let k_size = self.convtr.k_size;
        let stride = self.convtr.stride;
        let padding_total = k_size.saturating_sub(stride);
        let xs = xs.apply(&self.convtr)?;
        if self.causal {
            // This corresponds to trim_right_ratio = 1.
            unpad1d(&xs, 0, padding_total)
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            unpad1d(&xs, padding_left, padding_right)
        }
    }
}

impl StreamingModule for StreamableConvTranspose1d {
    fn reset_state(&mut self) {
        self.state_prev_ys.reset()
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let _enter = self.span.enter();
        let xs = match xs.as_option() {
            Some(xs) => xs,
            None => return Ok(StreamTensor::empty()),
        };
        let stride = self.convtr.stride;
        // We apply the underlying convtr directly rather than through forward so as
        // not to apply any padding here.
        let ys = self.convtr.forward(xs)?;
        let ot = ys.dim(D::Minus1)?;
        let ys = match self.state_prev_ys.as_option() {
            None => ys,
            Some(prev_ys) => {
                let pt = prev_ys.dim(D::Minus1)?;
                // Remove the bias as it will be applied multiple times.
                let prev_ys = match &self.convtr.bs {
                    None => prev_ys.clone(),
                    Some(bias) => {
                        let bias = bias.reshape((1, (), 1))?;
                        prev_ys.broadcast_sub(&bias)?
                    }
                };
                let ys1 = (ys.narrow(D::Minus1, 0, pt)? + prev_ys)?;
                let ys2 = ys.narrow(D::Minus1, pt, ot - pt)?;
                Tensor::cat(&[ys1, ys2], D::Minus1)?
            }
        };
        let invalid_steps = self.kernel_size - stride;
        let (ys, prev_ys) = StreamTensor::from(ys).split(D::Minus1, ot - invalid_steps)?;
        self.state_prev_ys = prev_ys;
        Ok(ys)
    }
}

#[derive(Debug, Clone)]
pub struct ConvDownsample1d {
    conv: StreamableConv1d,
}

impl ConvDownsample1d {
    pub fn new(
        stride: usize,
        dim: usize,
        causal: bool,
        learnt: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        if !learnt {
            candle::bail!("only learnt=true is supported")
        }
        let conv = StreamableConv1d::new(
            /* in_c */ dim,
            /* out_c */ dim,
            /* k_size_c */ 2 * stride,
            /* stride */ stride,
            /* dilation */ 1,
            /* groups */ 1, // channel_wise = false
            /* bias */ false,
            /* causal */ causal,
            /* norm */ None,
            /* pad_mode */ PadMode::Replicate,
            vb,
        )?;
        Ok(Self { conv })
    }
}

impl Module for ConvDownsample1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv)
    }
}

impl StreamingModule for ConvDownsample1d {
    fn reset_state(&mut self) {
        self.conv.reset_state()
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        self.conv.step(xs)
    }
}

#[derive(Debug, Clone)]
pub struct ConvTrUpsample1d {
    convtr: StreamableConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn new(
        stride: usize,
        dim: usize,
        causal: bool,
        learnt: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        if !learnt {
            candle::bail!("only learnt=true is supported")
        }
        let convtr = StreamableConvTranspose1d::new(
            dim,
            dim,
            /* k_size */ 2 * stride,
            /* stride */ stride,
            /* groups */ dim,
            /* bias */ false,
            /* causal */ causal,
            /* norm */ None,
            vb,
        )?;
        Ok(Self { convtr })
    }
}

impl Module for ConvTrUpsample1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.convtr)
    }
}

impl StreamingModule for ConvTrUpsample1d {
    fn reset_state(&mut self) {
        self.convtr.reset_state()
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        self.convtr.step(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::IndexOp;

    fn run_conv1d(
        k_size: usize,
        stride: usize,
        dilation: usize,
        step_size: usize,
        len: usize,
        bias: bool,
    ) -> Result<()> {
        // TODO: We should ensure for the seed to be constant when running these tests.
        let dev = &candle::Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle::DType::F32, dev);
        let conv1d = StreamableConv1d::new(
            /* in_c */ 2,
            /* out_c */ 3,
            /* k_size */ k_size,
            /* stride */ stride,
            /* dilation */ dilation,
            /* groups */ 1,
            /* bias */ bias,
            /* causal */ true,
            /* norm */ None,
            /* pad_mode */ PadMode::Constant,
            vb,
        )?;
        let xs = Tensor::randn(0f32, 1., (1, 2, step_size * len), dev)?;
        let ys = conv1d.forward(&xs)?;
        let mut conv1d = conv1d;
        let mut ys_steps = vec![];
        for idx in 0..len {
            let xs = xs.i((.., .., step_size * idx..step_size * (idx + 1)))?;
            let ys = conv1d.step(&xs.into())?;
            if let Some(ys) = ys.as_option() {
                ys_steps.push(ys.clone())
            }
        }
        let ys_steps = Tensor::cat(&ys_steps, D::Minus1)?;
        let diff = (&ys - &ys_steps)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        if diff > 1e-5 {
            println!("{xs}");
            println!("{ys}");
            println!("{ys_steps}");
            candle::bail!("larger diff than expected {diff}")
        }
        Ok(())
    }

    fn run_conv_tr1d(
        k_size: usize,
        stride: usize,
        step_size: usize,
        len: usize,
        bias: bool,
    ) -> Result<()> {
        // TODO: We should ensure for the seed to be constant when running these tests.
        let dev = &candle::Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle::DType::F32, dev);
        let conv1d = StreamableConvTranspose1d::new(
            /* in_c */ 2, /* out_c */ 3, /* k_size */ k_size,
            /* stride */ stride, /* groups */ 1, /* bias */ bias,
            /* causal */ true, /* norm */ None, vb,
        )?;
        let xs = Tensor::randn(0f32, 1., (1, 2, step_size * len), dev)?;
        let ys = conv1d.forward(&xs)?;
        let mut conv1d = conv1d;
        let mut ys_steps = vec![];
        for idx in 0..len {
            let xs = xs.i((.., .., step_size * idx..step_size * (idx + 1)))?;
            let ys = conv1d.step(&xs.into())?;
            if let Some(ys) = ys.as_option() {
                ys_steps.push(ys.clone())
            }
        }
        let ys_steps = Tensor::cat(&ys_steps, D::Minus1)?;
        let diff = (&ys - &ys_steps)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        if diff > 1e-5 {
            println!("{xs}");
            println!("{ys}");
            println!("{ys_steps}");
            candle::bail!("larger diff than expected {diff}")
        }
        Ok(())
    }

    #[test]
    fn conv1d() -> Result<()> {
        for step_size in [1, 2, 3] {
            for bias in [false, true] {
                run_conv1d(1, 1, 1, step_size, 5, bias)?;
                run_conv1d(2, 1, 1, step_size, 5, bias)?;
                run_conv1d(2, 2, 1, step_size, 6, bias)?;
                run_conv1d(3, 2, 1, step_size, 8, bias)?;
                run_conv1d(3, 2, 2, step_size, 8, bias)?;
            }
        }
        Ok(())
    }

    #[test]
    fn conv_tr1d() -> Result<()> {
        for step_size in [1, 2, 3] {
            for bias in [false, true] {
                run_conv_tr1d(1, 1, step_size, 5, bias)?;
                run_conv_tr1d(2, 1, step_size, 5, bias)?;
                run_conv_tr1d(3, 1, step_size, 5, bias)?;
                run_conv_tr1d(3, 2, step_size, 5, bias)?;
            }
        }
        Ok(())
    }
}
