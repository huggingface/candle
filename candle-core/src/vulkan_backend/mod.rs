//! The Vulkan backend: storage + device trait implementations.
//!
//! Scaffold: `affine` (f32) and `reduce` (Sum, f32) are implemented through
//! the `candle_vulkan_kernels` compute shaders; every other operation returns
//! `Err(Error::Msg("vulkan: <op> not implemented (scaffold)"))`.

use std::sync::atomic::Ordering;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use candle_vulkan_kernels::{
    call_affine_f32, call_copy_f32, call_elem_binary_f32, call_elem_unary_f32, call_gather_f32,
    call_gemm_f32, call_gemv_f32, call_gemv_t_f32, call_reduce_max_f32,
    call_reduce_sum_f32,
    call_rms_norm_f32,
    call_rope_f32, call_softmax_last_dim_f32, KernelName,
};

pub use crate::vulkan_backend::device::{VBuf, VulkanDevice, VulkanStorage};

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

mod device;

/// Errors from the Vulkan backend.
#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("kernel error: {0}")]
    KernelError(#[from] candle_vulkan_kernels::VulkanKernelError),

    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

fn not_impl(op: &str) -> Error {
    Error::Msg(format!("vulkan: {op} not implemented (scaffold)"))
}

fn contig_strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// The effective broadcast shape of a (possibly broadcasted) rhs layout:
/// a dim with a zero stride (or size 1) collapses to 1, the remaining
/// dims must form a row-major contiguous block over their own shape.
fn gemm_rhs_transposed(l: &Layout, k: usize) -> Result<bool> {
    let dims = l.dims();
    let strides = l.stride();
    if l.start_offset() != 0 {
        return Err(not_impl("matmul (rhs offset)"));
    }
    let ndim = dims.len();
    if ndim < 2 {
        return Err(not_impl("matmul (rhs rank)"));
    }
    let nk = contig_strides(dims);
    let kn_swapped = {
        let mut kn = contig_strides(dims);
        kn[ndim - 2] = 1;
        kn[ndim - 1] = k;
        kn
    };
    let matches = |expected: &[usize]| {
        expected
            .iter()
            .zip(strides)
            .zip(dims)
            .all(|((&e, &s), &d)| d == 1 || s == e)
    };
    // Dispatch on the physical strides, not the dims: when k == n the
    // last two dims coincide and only the strides tell a row-major (k, n)
    // storage (transposed = false) from a transposed view of a row-major
    // (n, k) buffer (strides [..., 1, k], transposed = true).
    if matches(&nk) {
        Ok(false)
    } else if matches(&kn_swapped) {
        Ok(true)
    } else {
        Err(not_impl("matmul (rhs layout)"))
    }
}

fn materialize(
    device: &VulkanDevice,
    storage: &VulkanStorage,
    layout: &Layout,
) -> Result<Subbuffer<[f32]>> {
    if layout.is_contiguous() && layout.start_offset() == 0 {
        return Ok(storage.as_f32().expect("materialize (dtype)").clone());
    }
    let dims = layout.dims().to_vec();
    if dims.len() > 4 {
        return Err(not_impl("materialize (rank)"));
    }
    let total: usize = dims.iter().product();
    let buf = device.new_f32_buffer(total)?;
    let dst = buf.clone();
    let src = storage.as_f32().expect("materialize (dtype)").clone();
    let sstr = layout.stride().to_vec();
    let dstr = contig_strides(&dims);
    let soff = layout.start_offset();
    let kernels = device.kernels();
    device.execute(
        move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
            call_copy_f32(
                cbb, kernels.as_ref(), &src, &dst, total, &dims, &sstr, &dstr, soff, 0,
            )
            .map_err(|e| e.to_string())
        },
    )?;
    Ok(buf)
}

/// Fused RMSNorm over the last (contiguous) axis for f32:
/// `x * inversesqrt(mean(x^2) + eps) * weight`. `input` may be
/// non-contiguous (it is materialized first); `weight` must be contiguous
/// and have `cols` elements. The result is a fresh contiguous storage with
/// shape `layout.shape()`.
pub fn rms_norm_f32(
    input: &VulkanStorage,
    layout: &Layout,
    weight: &VulkanStorage,
    eps: f32,
) -> Result<(VulkanStorage, Shape)> {
    let dims = layout.dims().to_vec();
    if dims.is_empty() {
        return Err(not_impl("rms_norm (rank)"));
    }
    let cols = dims[dims.len() - 1];
    let total: usize = dims.iter().product();
    let rows = total / cols;
    let in_buf = materialize(&input.device, input, layout)?;
    let w_buf = weight
        .as_f32()
        .ok_or_else(|| Error::Vulkan("rms_norm: weight is not f32".to_string().into()))?
        .clone();
    let out = input.device.new_f32_buffer(total)?;
    let out_arg = out.clone();
    let kernels = input.device.kernels();
    input.device.execute(
        move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
            call_rms_norm_f32(cbb, kernels.as_ref(), &in_buf, &w_buf, &out_arg, rows, cols, eps)
                .map_err(|e| e.to_string())
        },
    )?;
    Ok((
        VulkanStorage::new(VBuf::F32(out), &input.device, total, DType::F32),
        layout.shape().clone(),
    ))
}

/// Numerically stable softmax over the last (contiguous) axis for f32.
/// `input` may be non-contiguous (it is materialized first). The result is
/// a fresh contiguous storage with shape `layout.shape()`.
/// Non-interleaved rotary embeddings for a contiguous f32 `(b, h, t, d)`
/// tensor; `cos`/`sin` are `(t, d/2)` or `(b, t, d/2)` (any dtype is
/// rejected, the CPU fallback in `CustomOp3::vulkan_fwd` is not reachable
/// for other dtypes because this function requires f32 everywhere). The
/// result is a fresh contiguous storage with the input shape.
pub fn rope_f32(
    input: &VulkanStorage,
    layout: &Layout,
    cos: &VulkanStorage,
    cos_layout: &Layout,
    sin: &VulkanStorage,
    sin_layout: &Layout,
) -> Result<(VulkanStorage, Shape)> {
    let dims = layout.dims().to_vec();
    if dims.len() != 4 {
        return Err(not_impl("rope (rank)"));
    }
    let (b, h, t, d) = (dims[0], dims[1], dims[2], dims[3]);
    if d % 2 != 0 || d == 0 {
        return Err(not_impl("rope (head dim)"));
    }
    let unbatched = cos_layout.dims().len() == 3;
    let in_buf = materialize(&input.device, input, layout)?;
    let cos_buf = materialize(&input.device, cos, cos_layout)?;
    let sin_buf = materialize(&input.device, sin, sin_layout)?;
    let total: usize = b * h * t * d;
    let out = input.device.new_f32_buffer(total)?;
    let out_arg = out.clone();
    let kernels = input.device.kernels();
    input.device.execute(
        move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
            call_rope_f32(
                cbb,
                kernels.as_ref(),
                &in_buf,
                &cos_buf,
                &sin_buf,
                &out_arg,
                b,
                h,
                t,
                d,
                unbatched,
            )
            .map_err(|e| e.to_string())
        },
    )?;
    Ok((
        VulkanStorage::new(VBuf::F32(out), &input.device, total, DType::F32),
        layout.shape().clone(),
    ))
}
pub fn softmax_last_dim_f32(
    input: &VulkanStorage,
    layout: &Layout,
) -> Result<(VulkanStorage, Shape)> {
    let dims = layout.dims().to_vec();
    if dims.is_empty() {
        return Err(not_impl("softmax_last_dim (rank)"));
    }
    let cols = dims[dims.len() - 1];
    let total: usize = dims.iter().product();
    let rows = total / cols;
    let in_buf = materialize(&input.device, input, layout)?;
    let out = input.device.new_f32_buffer(total)?;
    let out_arg = out.clone();
    let kernels = input.device.kernels();
    input.device.execute(
        move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
            call_softmax_last_dim_f32(cbb, kernels.as_ref(), &in_buf, &out_arg, rows, cols)
                .map_err(|e| e.to_string())
        },
    )?;
    Ok((
        VulkanStorage::new(VBuf::F32(out), &input.device, total, DType::F32),
        layout.shape().clone(),
    ))
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(not_impl("const_set (layout)"));
        }
        let dim = layout.dims().iter().product::<usize>();
        match self.dtype {
            DType::F32 => {
                let value = scalar.to_f64() as f32;
                let buffer = self.device.upload_f32(&vec![value; dim])?;
                *self = Self::new(VBuf::F32(buffer), &self.device, dim, DType::F32);
            }
            DType::U32 => {
                let value = scalar.to_f64() as u32;
                let buffer = self.device.upload_u32(&vec![value; dim])?;
                *self = Self::new(VBuf::U32(buffer), &self.device, dim, DType::U32);
            }
            _ => return Err(not_impl("const_set (dtype)")),
        }
        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.device.synchronize()?;
        match (&self.buffer, self.dtype) {
            (VBuf::F32(_), DType::F32) => {
                let data = self.device.download_f32(self.as_f32().unwrap())?;
                Ok(CpuStorage::F32(data))
            }
            (VBuf::U32(_), DType::U32) => {
                let data = self.device.download_u32(self.as_u32().unwrap())?;
                Ok(CpuStorage::U32(data))
            }
            _ => Err(not_impl("to_cpu_storage (dtype)")),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_impl("affine (dtype)"));
        }
        let size = layout.dims().iter().product::<usize>();
        let out = self.device.new_f32_buffer(size)?;
        let out_arg = out.clone();
        let input = self.as_f32().expect("affine input is f32").clone();
        let device = self.device.clone();
        let kernels = device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_affine_f32(cbb, kernels.as_ref(), &input, &out_arg, mul as f32, add as f32)
                    .map_err(|e| e.to_string())
            },
        )?;
        Ok(Self::new(VBuf::F32(out), &self.device, size, DType::F32))
    }

    fn reduce_op(&self, ro: ReduceOp, layout: &Layout, axes: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_impl("reduce_op (dtype)"));
        }
        let is_max = matches!(ro, ReduceOp::Max);
        if !is_max && !matches!(ro, ReduceOp::Sum) {
            return Err(not_impl("reduce_op (op)"));
        }
        let ndim = layout.dims().len();
        let last_axis = ndim - 1;
        if axes.len() != 1 || axes[0] != last_axis {
            return Err(not_impl("reduce_op (axes)"));
        }
        let dims = layout.dims().to_vec();
        if dims.len() > 4 {
            return Err(not_impl("reduce_op (rank)"));
        }
        let cols = dims[last_axis];
        let total: usize = dims.iter().product();
        let rows = total / cols;
        let out = self.device.new_f32_buffer(rows)?;
        let out_arg = out.clone();
        let input = materialize(&self.device, self, layout)?;
        let kernels = self.device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                if is_max {
                    call_reduce_max_f32(cbb, kernels.as_ref(), &input, &out_arg, rows, cols)
                } else {
                    call_reduce_sum_f32(cbb, kernels.as_ref(), &input, &out_arg, rows, cols)
                }
                .map_err(|e| e.to_string())
            },
        )?;
        Ok(Self::new(VBuf::F32(out), &self.device, rows, DType::F32))
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(not_impl("powf"))
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(not_impl("elu"))
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(not_impl("cmp"))
    }

    fn to_dtype(&self, _: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        if (self.dtype, dtype) == (DType::U32, DType::F32) {
            let data = self
                .device
                .download_u32(self.as_u32().expect("u32 source"))?;
            let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let buffer = self.device.upload_f32(&data)?;
            return Ok(Self::new(
                VBuf::F32(buffer),
                &self.device,
                self.data_len,
                DType::F32,
            ));
        }
        Err(not_impl("to_dtype"))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(not_impl("unary_impl (dtype)"));
        }
        let name = match B::NAME {
            "exp" => KernelName::ElemExpF32,
            "sin" => KernelName::ElemSinF32,
            "cos" => KernelName::ElemCosF32,
            "neg" => KernelName::ElemNegF32,
            "sqrt" => KernelName::ElemSqrtF32,
            "silu" => KernelName::ElemSiluF32,
            "sigmoid" => KernelName::ElemSigmoidF32,
            other => return Err(not_impl(&format!("unary_impl ({other})"))),
        };
        let dims = layout.dims().to_vec();
        if dims.len() > 4 {
            return Err(not_impl("unary_impl (rank)"));
        }
        let total: usize = dims.iter().product();
        let out = self.device.new_f32_buffer(total)?;
        let out_arg = out.clone();
        let input = materialize(&self.device, self, layout)?;
        let kernels = self.device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_elem_unary_f32(cbb, kernels.as_ref(), name, &input, &out_arg).map_err(|e| e.to_string())
            },
        )?;
        Ok(Self::new(VBuf::F32(out), &self.device, total, DType::F32))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(not_impl("binary_impl (dtype)"));
        }
        let name = match B::NAME {
            "add" => KernelName::ElemAddF32,
            "sub" => KernelName::ElemSubF32,
            "mul" => KernelName::ElemMulF32,
            "div" => KernelName::ElemDivF32,
            other => return Err(not_impl(&format!("binary_impl ({other})"))),
        };
        let ldims = lhs_l.dims().to_vec();
        let rdims = rhs_l.dims().to_vec();
        if ldims.len() > 4 || rdims.len() > 4 {
            return Err(not_impl("binary_impl (rank)"));
        }
        // The PC dims must describe the buffer that `materialize`
        // returns: a contiguous view shaped `rhs_l.dims()`. Passing the
        // collapsed broadcast shape (e.g. (4,1) for a (4,4) broadcast view)
        // would make the shader index the materialized buffer with the
        // wrong offsets.
        let rhs_dims = rhs_l.dims().to_vec();
        let total: usize = ldims.iter().product();
        let out = self.device.new_f32_buffer(total)?;
        let out_arg = out.clone();
        let input = materialize(&self.device, self, lhs_l)?;
        let rbuf = materialize(&self.device, rhs, rhs_l)?;
        let kernels = self.device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_elem_binary_f32(
                    cbb,
                    kernels.as_ref(),
                    name,
                    &input,
                    &rbuf,
                    &out_arg,
                    &ldims,
                    &rhs_dims,
                )
                .map_err(|e| e.to_string())
            },
        )?;
        Ok(Self::new(VBuf::F32(out), &self.device, total, DType::F32))
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(not_impl("where_cond"))
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(not_impl("conv1d"))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(not_impl("conv_transpose1d"))
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(not_impl("conv2d"))
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(not_impl("conv_transpose2d"))
    }

    fn index_select(
        &self,
        other: &Self,
        layout: &Layout,
        other_layout: &Layout,
        axis: usize,
    ) -> Result<Self> {
        if axis != 0 {
            return Err(not_impl("index_select (axis)"));
        }
        if self.dtype != DType::F32 || other.dtype != DType::U32 {
            return Err(not_impl("index_select (dtype)"));
        }
        let dims = layout.dims();
        if dims.len() != 2 {
            return Err(not_impl("index_select (rank)"));
        }
        let dim = dims[1];
        let rows: usize = other_layout.dims().iter().product();
        if other_layout.start_offset() != 0 {
            return Err(not_impl("index_select (ids layout)"));
        }
        let out = self.device.new_f32_buffer(rows * dim)?;
        let out_arg = out.clone();
        let ids = other.as_u32().expect("index_select ids are u32").clone();
        let emb = self.as_f32().expect("index_select source is f32").clone();
        let kernels = self.device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_gather_f32(cbb, kernels.as_ref(), &ids, &emb, &out_arg, rows, dim)
                    .map_err(|e| e.to_string())
            },
        )?;
        Ok(Self::new(
            VBuf::F32(out),
            &self.device,
            rows * dim,
            DType::F32,
        ))
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(not_impl("gather"))
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(not_impl("scatter_set"))
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(not_impl("scatter_add_set"))
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(not_impl("index_add"))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(not_impl("matmul (dtype)"));
        }
        let (b, m, n, k) = bmnk;
        let transposed = gemm_rhs_transposed(rhs_l, k)?;
        let total = b * m * n;
        let out = self.device.new_f32_buffer(total)?;
        let out_arg = out.clone();
        let input = materialize(&self.device, self, lhs_l)?;
        let rbuf = rhs.as_f32().expect("matmul rhs is f32").clone();
        let kernels = self.device.kernels();
        let use_gemv = b == 1 && m == 1 && !transposed;
        let w_stride = rhs_l.stride()[rhs_l.dims().len() - 1];
        let gemv_t_ok = transposed && k <= 3072;
        let use_gemv_t =
            b == 1 && m == 1 && gemv_t_ok && std::env::var("CANDLE_VULKAN_NO_GEMVT").is_err();
        if std::env::var("CANDLE_VULKAN_TRACE").is_ok() {
            eprintln!("matmul b={b} m={m} n={n} k={k} transposed={transposed} gemv={use_gemv} gemv_t={use_gemv_t}");
        }
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                if use_gemv {
                    call_gemv_f32(cbb, kernels.as_ref(), &input, &rbuf, &out_arg, k, n)
                } else if use_gemv_t {
                    call_gemv_t_f32(cbb, kernels.as_ref(), &input, &rbuf, &out_arg, n, k, w_stride)
                } else {
                    call_gemm_f32(
                        cbb, kernels.as_ref(), &input, &rbuf, &out_arg, b, m, n, k, transposed,
                    )
                }
                .map_err(|e| e.to_string())
            },
        )?;
        Ok(Self::new(VBuf::F32(out), &self.device, total, DType::F32))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, layout: &Layout) -> Result<()> {
        if self.dtype != DType::F32 || dst.dtype != DType::F32 {
            return Err(not_impl("copy_strided_src (dtype)"));
        }
        let dims = layout.dims().to_vec();
        let total: usize = dims.iter().product();
        let src_strides = layout.stride().to_vec();
        let dst_strides = contig_strides(&dims);
        let src_offset = layout.start_offset();
        let src = self.as_f32().expect("copy src is f32").clone();
        let dst_buf = dst.as_f32().expect("copy dst is f32").clone();
        let kernels = self.device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_copy_f32(
                    cbb,
                    kernels.as_ref(),
                    &src,
                    &dst_buf,
                    total,
                    &dims,
                    &src_strides,
                    &dst_strides,
                    src_offset,
                    dst_offset,
                )
                .map_err(|e| e.to_string())
            },
        )?;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if self.dtype != DType::F32 || dst.dtype != DType::F32 {
            return Err(not_impl("copy2d (dtype)"));
        }
        let dims = [d1, d2];
        let src_strides = [src_stride1, 1usize];
        let dst_strides = [dst_stride1, 1usize];
        let total = d1 * d2;
        let src = self.as_f32().expect("copy src is f32").clone();
        let dst_buf = dst.as_f32().expect("copy dst is f32").clone();
        let kernels = self.device.kernels();
        self.device.execute(
            move |cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
                call_copy_f32(
                    cbb,
                    kernels.as_ref(),
                    &src,
                    &dst_buf,
                    total,
                    &dims,
                    &src_strides,
                    &dst_strides,
                    src_offset,
                    dst_offset,
                )
                .map_err(|e| e.to_string())
            },
        )?;
        Ok(())
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(not_impl("avg_pool2d"))
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(not_impl("max_pool2d"))
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(not_impl("upsample_nearest1d"))
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(not_impl("upsample_nearest2d"))
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(not_impl("upsample_bilinear2d"))
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.seed_atomic().store(seed, Ordering::Relaxed);
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(self.seed_atomic().load(Ordering::Relaxed))
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.gpu_id(),
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.gpu_id() == other.gpu_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let dim = shape.elem_count();
        if dtype != DType::F32 {
            return Err(not_impl("zeros_impl (dtype)"));
        }
        let buffer = self.upload_f32(&vec![0f32; dim])?;
        Ok(VulkanStorage::new(VBuf::F32(buffer), self, dim, dtype))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let dim = shape.elem_count();
        let buffer = self.new_f32_buffer(dim)?;
        Ok(VulkanStorage::new(VBuf::F32(buffer), self, dim, dtype))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, slice: &[T]) -> Result<Self::Storage> {
        let dtype = T::DTYPE;
        let dim = slice.len();
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = slice.iter().map(|t| t.to_f64() as f32).collect();
                let buffer = self.upload_f32(&data)?;
                Ok(VulkanStorage::new(VBuf::F32(buffer), self, dim, dtype))
            }
            DType::U32 => {
                let data: Vec<u32> = slice.iter().map(|t| t.to_f64() as u32).collect();
                let buffer = self.upload_u32(&data)?;
                Ok(VulkanStorage::new(VBuf::U32(buffer), self, dim, dtype))
            }
            _ => Err(not_impl("storage_from_slice (dtype)")),
        }
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        match storage {
            CpuStorage::F32(data) => {
                let buffer = self.upload_f32(data)?;
                Ok(VulkanStorage::new(
                    VBuf::F32(buffer),
                    self,
                    data.len(),
                    DType::F32,
                ))
            }
            CpuStorage::U32(data) => {
                let buffer = self.upload_u32(data)?;
                Ok(VulkanStorage::new(
                    VBuf::U32(buffer),
                    self,
                    data.len(),
                    DType::U32,
                ))
            }
            _ => Err(not_impl("storage_from_cpu_storage (dtype)")),
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        low: f64,
        upper: f64,
    ) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            return Err(not_impl("rand_uniform (dtype)"));
        }
        use rand::Rng;
        let dim = shape.elem_count();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(low as f32..upper as f32))
            .collect();
        let buffer = self.upload_f32(&data)?;
        Ok(VulkanStorage::new(VBuf::F32(buffer), self, dim, dtype))
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            return Err(not_impl("rand_normal (dtype)"));
        }
        use rand_distr::{Distribution, Normal};
        let dim = shape.elem_count();
        let mut rng = rand::rng();
        let normal = Normal::new(mean, std).expect("std > 0");
        let data: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng) as f32).collect();
        let buffer = self.upload_f32(&data)?;
        Ok(VulkanStorage::new(VBuf::F32(buffer), self, dim, dtype))
    }

    fn synchronize(&self) -> Result<()> {
        VulkanDevice::synchronize(self)
    }
}
