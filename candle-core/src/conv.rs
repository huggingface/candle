//! 1D and 2D Convolutions
//!
use crate::{op::BackpropOp, op::Op, Error, Result, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) b_size: usize,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub(crate) cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    pub(crate) b_size: usize,
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.k_size - 1)
            + self.output_padding
            + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnFwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonFused,
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride + self.dilation * (self.k_h - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride + self.dilation * (self.k_w - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

/// Parameters for Deformable Convolution 2D (DCNv2)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsDeformConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) stride_h: usize,
    pub(crate) stride_w: usize,
    pub(crate) padding_h: usize,
    pub(crate) padding_w: usize,
    pub(crate) dilation_h: usize,
    pub(crate) dilation_w: usize,
    pub(crate) groups: usize,
    pub(crate) offset_groups: usize,
}

impl ParamsDeformConv2D {
    pub(crate) fn out_h(&self) -> usize {
        let ker_h = self.dilation_h * (self.k_h - 1) + 1;
        (self.i_h + 2 * self.padding_h - ker_h) / self.stride_h + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        let ker_w = self.dilation_w * (self.k_w - 1) + 1;
        (self.i_w + 2 * self.padding_w - ker_w) / self.stride_w + 1
    }

    #[allow(dead_code)]
    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![
            self.b_size,
            self.c_out * self.groups,
            self.out_h(),
            self.out_w(),
        ]
    }
}

impl Tensor {
    fn conv1d_single_group(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv1d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }

        let params = ParamsConv1D {
            b_size,
            l_in,
            c_out: c_out / groups,
            c_in: c_in / groups,
            k_size,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv_transpose1d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        let storage = self.storage().conv_transpose1d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D transposed convolution over the input tensor.
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_in_k, c_out, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups")
        }
        let params = ParamsConvTranspose1D {
            b_size,
            l_in,
            k_size,
            c_out,
            c_in: c_in / groups,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if groups == 1 {
            self.conv_transpose1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv_transpose1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv2d_single_group(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 2D convolution over the input tensor.
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv2d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    pub fn conv2d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k * groups {
            crate::bail!(
                "in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})"
            )
        }
        let params = ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv2d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv2d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    /// Applies a 2D transposed convolution over the input tensor.
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_in_k, c_out, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = ParamsConvTranspose2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            output_padding,
            stride,
            dilation,
        };
        let storage = self.storage().conv_transpose2d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            &params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Performs Deformable Convolution 2D (DCNv2).
    ///
    /// # Arguments
    /// * `offset` - Offset tensor of shape [batch, 2*offset_groups*kH*kW, out_h, out_w]
    /// * `weight` - Convolution weight of shape [out_channels, in_channels/groups, kH, kW]
    /// * `mask` - Optional modulation mask of shape [batch, offset_groups*kH*kW, out_h, out_w]
    /// * `bias` - Optional bias of shape [out_channels]
    /// * `stride` - Stride (stride_h, stride_w)
    /// * `padding` - Padding (pad_h, pad_w)
    /// * `dilation` - Dilation (dilation_h, dilation_w)
    /// * `groups` - Number of convolution groups
    /// * `offset_groups` - Number of offset groups
    ///
    /// # Returns
    /// Output tensor of shape [batch, out_channels, out_h, out_w]
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv2d(
        &self,
        offset: &Tensor,
        weight: &Tensor,
        mask: Option<&Tensor>,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        offset_groups: usize,
    ) -> Result<Tensor> {
        let (batch, in_channels, in_h, in_w) = self.dims4()?;
        let (out_channels, weight_in_c, k_h, k_w) = weight.dims4()?;
        let (offset_batch, offset_c, offset_h, offset_w) = offset.dims4()?;

        // Build params struct
        let params = ParamsDeformConv2D {
            b_size: batch,
            i_h: in_h,
            i_w: in_w,
            k_h,
            k_w,
            c_out: out_channels / groups,
            c_in: in_channels / groups,
            stride_h: stride.0,
            stride_w: stride.1,
            padding_h: padding.0,
            padding_w: padding.1,
            dilation_h: dilation.0,
            dilation_w: dilation.1,
            groups,
            offset_groups,
        };

        let out_h = params.out_h();
        let out_w = params.out_w();

        // Validate parameters
        if batch != offset_batch {
            crate::bail!(
                "deform_conv2d: batch size mismatch, input: {}, offset: {}",
                batch,
                offset_batch
            );
        }
        if offset_c != offset_groups * 2 * k_h * k_w {
            crate::bail!(
                "deform_conv2d: offset channels mismatch, got {}, expected {}",
                offset_c,
                offset_groups * 2 * k_h * k_w
            );
        }
        if offset_h != out_h || offset_w != out_w {
            crate::bail!(
                "deform_conv2d: offset spatial size mismatch, got ({}, {}), expected ({}, {})",
                offset_h,
                offset_w,
                out_h,
                out_w
            );
        }
        if in_channels % groups != 0 {
            crate::bail!("deform_conv2d: in_channels must be divisible by groups");
        }
        if out_channels % groups != 0 {
            crate::bail!("deform_conv2d: out_channels must be divisible by groups");
        }
        if in_channels % offset_groups != 0 {
            crate::bail!("deform_conv2d: in_channels must be divisible by offset_groups");
        }
        if weight_in_c != in_channels / groups {
            crate::bail!(
                "deform_conv2d: weight in_channels mismatch, got {}, expected {}",
                weight_in_c,
                in_channels / groups
            );
        }

        // Validate mask
        if let Some(m) = mask {
            let (mask_batch, mask_c, mask_h, mask_w) = m.dims4()?;
            if mask_batch != batch {
                crate::bail!("deform_conv2d: mask batch size mismatch");
            }
            if mask_c != offset_groups * k_h * k_w {
                crate::bail!(
                    "deform_conv2d: mask channels mismatch, got {}, expected {}",
                    mask_c,
                    offset_groups * k_h * k_w
                );
            }
            if mask_h != out_h || mask_w != out_w {
                crate::bail!("deform_conv2d: mask spatial size mismatch");
            }
        }

        // Validate bias
        if let Some(b) = bias {
            let bias_len = b.dims1()?;
            if bias_len != out_channels {
                crate::bail!(
                    "deform_conv2d: bias length mismatch, got {}, expected {}",
                    bias_len,
                    out_channels
                );
            }
        }

        // Call storage layer (handles im2col + matmul)
        let mask_storage = mask.map(|m| m.storage());
        let mask_layout = mask.map(|m| m.layout().clone());
        let mask_ref = match (&mask_storage, &mask_layout) {
            (Some(s), Some(l)) => Some((&**s, l)),
            _ => None,
        };

        let storage = self.storage().deform_conv2d(
            self.layout(),
            &offset.storage(),
            offset.layout(),
            &weight.storage(),
            weight.layout(),
            mask_ref,
            &params,
        )?;

        let shape = crate::Shape::from((batch, out_channels, out_h, out_w));
        let op = BackpropOp::none(); // Backward pass not supported yet

        let mut result = crate::tensor::from_storage(storage, shape, op, false);

        // Add bias at Tensor layer
        if let Some(b) = bias {
            result = result.broadcast_add(&b.reshape((1, out_channels, 1, 1))?)?;
        }

        Ok(result)
    }
}
