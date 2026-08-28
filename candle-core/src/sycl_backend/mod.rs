//! Intel GPU backend for candle, implemented with SYCL (oneAPI DPC++).
//!
//! Phases 3–4: the full dense op surface (element-wise, reduce, cmp, ternary,
//! indexing, sort, pool/upsample, conv) plus GGUF dequantization and QMatMul
//! (`quantized/sycl.rs`) run on the Intel GPU; oneMKL provides GEMM. Remaining:
//! `candle-nn` fused ops (`sycl_fwd`, Phase 5), fused MMVQ/MMQ quant kernels,
//! and F8E4M3 / MX dtypes.
//!
//! Positioning (feasibility report): Intel-first, vendor-neutral SYCL as the
//! default codepath, NVIDIA/AMD kept only as "must compile" targets.
#![allow(dead_code)]

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::sync::Arc;

pub use candle_sycl_kernels as k;
use k::{BinaryOp, DeviceBuffer, Queue, SyclDType, UnaryOp};

mod error;
pub mod nn_ops;
pub use error::SyclError;

macro_rules! sycl_bail {
    ($($arg:tt)*) => {
        return Err(crate::Error::Sycl(
            $crate::sycl_backend::SyclError::msg(format!($($arg)*)).into(),
        ))
    };
}

fn wrap<T>(r: k::Result<T>) -> Result<T> {
    r.map_err(|e| crate::Error::Sycl(Box::new(SyclError::msg(e.to_string()))))
}

fn to_sycl_dtype(dt: DType) -> Result<SyclDType> {
    Ok(match dt {
        DType::U8 => SyclDType::U8,
        DType::U32 => SyclDType::U32,
        DType::I64 => SyclDType::I64,
        DType::F16 => SyclDType::F16,
        DType::BF16 => SyclDType::BF16,
        DType::F32 => SyclDType::F32,
        DType::F64 => SyclDType::F64,
        other => {
            sycl_bail!("dtype {other:?} is not supported by the SYCL backend yet")
        }
    })
}

fn ffi_layout(l: &Layout) -> Result<k::Layout> {
    if l.start_offset() == 0 && l.is_contiguous() {
        Ok(k::Layout::dense())
    } else {
        wrap(k::Layout::strided(l.dims(), l.stride(), l.start_offset()))
    }
}

/// Raw bytes and element count of a `CpuStorage`, for the H2D path.
fn cpu_bytes(s: &CpuStorage) -> (&[u8], usize) {
    macro_rules! b {
        ($v:expr) => {
            (
                unsafe {
                    std::slice::from_raw_parts(
                        $v.as_ptr() as *const u8,
                        std::mem::size_of_val(&$v[..]),
                    )
                },
                $v.len(),
            )
        };
    }
    match s {
        CpuStorage::U8(v) => b!(v),
        CpuStorage::U32(v) => b!(v),
        CpuStorage::I16(v) => b!(v),
        CpuStorage::I32(v) => b!(v),
        CpuStorage::I64(v) => b!(v),
        CpuStorage::BF16(v) => b!(v),
        CpuStorage::F16(v) => b!(v),
        CpuStorage::F32(v) => b!(v),
        CpuStorage::F64(v) => b!(v),
        CpuStorage::F8E4M3(v) => b!(v),
        CpuStorage::F6E2M3(v)
        | CpuStorage::F6E3M2(v)
        | CpuStorage::F4(v)
        | CpuStorage::F8E8M0(v) => b!(v),
    }
}

/// Build a `CpuStorage` of `dtype` from `n` elements' worth of device bytes.
fn cpu_from_bytes(dtype: DType, bytes: &[u8], n: usize) -> Result<CpuStorage> {
    macro_rules! v {
        ($ty:ty, $ctor:path) => {{
            let mut out = vec![<$ty>::default(); n];
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    out.as_mut_ptr() as *mut u8,
                    n * std::mem::size_of::<$ty>(),
                )
            };
            dst.copy_from_slice(&bytes[..dst.len()]);
            $ctor(out)
        }};
    }
    use half::{bf16, f16};
    Ok(match dtype {
        DType::U8 => v!(u8, CpuStorage::U8),
        DType::U32 => v!(u32, CpuStorage::U32),
        DType::I64 => v!(i64, CpuStorage::I64),
        DType::F16 => v!(f16, CpuStorage::F16),
        DType::BF16 => v!(bf16, CpuStorage::BF16),
        DType::F32 => v!(f32, CpuStorage::F32),
        DType::F64 => v!(f64, CpuStorage::F64),
        other => sycl_bail!("dtype {other:?} not supported by the SYCL backend yet"),
    })
}

// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SyclDevice {
    ordinal: usize,
    queue: Arc<Queue>,
}

impl SyclDevice {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        // A SYCL device always has its own in-order queue.
        Self::new(ordinal)
    }

    /// `true` for integrated GPUs (Meteor Lake / Lunar Lake iGPU) — device and
    /// host share memory, so the backend can skip staging copies (Phase 3+).
    pub fn is_integrated(&self) -> bool {
        self.queue
            .device_info()
            .map(|i| i.is_integrated)
            .unwrap_or(false)
    }

    fn alloc(&self, dtype: DType, elem_count: usize) -> Result<SyclStorage> {
        let bytes = elem_count.max(1) * dtype.size_in_bytes();
        let buffer = wrap(DeviceBuffer::alloc(&self.queue, bytes))?;
        Ok(SyclStorage {
            buffer,
            dtype,
            elem_count,
            device: self.clone(),
        })
    }

    // --- helpers for `quantized/sycl.rs` ---
    pub(crate) fn q(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Allocate a raw USM device buffer of `bytes` bytes on this device's queue.
    /// Public so out-of-tree fused kernels (e.g. crane's GDN launcher) can
    /// allocate their own outputs; pair with [`SyclStorage::from_buffer`].
    pub fn alloc_bytes(&self, bytes: usize) -> Result<DeviceBuffer> {
        wrap(DeviceBuffer::alloc(&self.queue, bytes.max(1)))
    }

    /// Allocate a dense `SyclStorage` of `elem_count` elements of `dtype`.
    pub fn new_storage(&self, dtype: DType, elem_count: usize) -> Result<SyclStorage> {
        self.alloc(dtype, elem_count)
    }
}

#[derive(Debug)]
pub struct SyclStorage {
    buffer: DeviceBuffer,
    dtype: DType,
    elem_count: usize,
    device: SyclDevice,
}

impl SyclStorage {
    pub fn transfer_to_device(&self, dst: &SyclDevice) -> Result<Self> {
        if self.device.same_device(dst) {
            return self.try_clone(&Layout::contiguous(self.elem_count));
        }
        sycl_bail!("cross-device transfer between distinct SYCL queues is not implemented yet")
    }

    fn sd(&self) -> Result<SyclDType> {
        to_sycl_dtype(self.dtype)
    }

    /// The backing USM device buffer. Public so out-of-tree fused kernels
    /// (e.g. crane's GDN launcher) can read a tensor's raw device pointer via
    /// [`k::DeviceBuffer::as_ptr`].
    pub fn buf(&self) -> &DeviceBuffer {
        &self.buffer
    }

    /// Element count of the backing buffer.
    pub fn elems(&self) -> usize {
        self.elem_count
    }

    /// Wrap a device buffer produced out-of-tree (e.g. by a fused kernel) as a
    /// dense `SyclStorage`, ready for [`crate::Tensor::from_storage`]. The
    /// buffer must hold exactly `elem_count` elements of `dtype`.
    pub fn from_buffer(
        device: &SyclDevice,
        buffer: DeviceBuffer,
        dtype: DType,
        elem_count: usize,
    ) -> Self {
        storage_from_buffer(device, buffer, dtype, elem_count)
    }
    pub(crate) fn matmul_raw(
        &self,
        rhs: &SyclStorage,
        (b, m, n, kk): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<SyclStorage> {
        self.matmul(rhs, (b, m, n, kk), lhs_l, rhs_l)
    }
    pub(crate) fn index_select_raw(
        &self,
        ids: &SyclStorage,
        l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<SyclStorage> {
        self.index_select(ids, l, ids_l, dim)
    }
    pub(crate) fn to_dtype_raw(&self, layout: &Layout, dtype: DType) -> Result<SyclStorage> {
        self.to_dtype(layout, dtype)
    }
}

/// Build a dense `SyclStorage` directly from a device buffer (used by the
/// quantized dequant path).
pub(crate) fn storage_from_buffer(
    device: &SyclDevice,
    buffer: DeviceBuffer,
    dtype: DType,
    elem_count: usize,
) -> SyclStorage {
    SyclStorage {
        buffer,
        dtype,
        elem_count,
        device: device.clone(),
    }
}

impl BackendStorage for SyclStorage {
    type Device = SyclDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        let out = self.device.alloc(self.dtype, self.elem_count)?;
        wrap(
            out.buffer
                .copy_from_device(&self.buffer, self.buffer.len_bytes()),
        )?;
        Ok(out)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let mut bytes = vec![0u8; self.buffer.len_bytes()];
        wrap(self.buffer.copy_to_host(&mut bytes))?;
        cpu_from_bytes(self.dtype, &bytes, self.elem_count)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(self.dtype, numel)?;
        let lin = ffi_layout(layout)?;
        wrap(k::affine(
            &self.device.queue,
            self.sd()?,
            &lin,
            &self.buffer,
            &out.buffer,
            numel,
            mul,
            add,
        ))?;
        Ok(out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(self.dtype, numel)?;
        wrap(k::powf(
            &self.device.queue,
            self.sd()?,
            &ffi_layout(layout)?,
            &self.buffer,
            &out.buffer,
            numel,
            e,
        ))?;
        Ok(out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(self.dtype, numel)?;
        wrap(k::elu(
            &self.device.queue,
            self.sd()?,
            &ffi_layout(layout)?,
            &self.buffer,
            &out.buffer,
            numel,
            alpha,
        ))?;
        Ok(out)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        let src_dims = layout.dims();
        let src_stride = layout.stride();
        let src_el: usize = src_dims.iter().product();
        // Reorder so the reduced dims come last: then `out_i * reduce_el + r`
        // decomposes directly over the reordered shape.
        let mut dims = Vec::with_capacity(src_dims.len());
        let mut strides = Vec::with_capacity(src_dims.len());
        let mut out_el = 1usize;
        for (i, &d) in src_dims.iter().enumerate() {
            if !reduce_dims.contains(&i) {
                out_el *= d;
                dims.push(d);
                strides.push(src_stride[i]);
            }
        }
        for &i in reduce_dims {
            dims.push(src_dims[i]);
            strides.push(src_stride[i]);
        }
        let reduce_el = src_el / out_el.max(1);
        let (kop, index_out) = match op {
            ReduceOp::Sum => (k::ReduceOp::Sum, false),
            ReduceOp::Min => (k::ReduceOp::Min, false),
            ReduceOp::Max => (k::ReduceOp::Max, false),
            ReduceOp::ArgMin => (k::ReduceOp::ArgMin, true),
            ReduceOp::ArgMax => (k::ReduceOp::ArgMax, true),
        };
        if matches!(
            op,
            ReduceOp::Min | ReduceOp::Max | ReduceOp::ArgMin | ReduceOp::ArgMax
        ) && layout.shape().elem_count() == 0
        {
            return Err(crate::Error::EmptyTensor { op: "reduce" }.bt());
        }
        let out_dtype = if index_out { DType::U32 } else { self.dtype };
        let out = self.device.alloc(out_dtype, out_el)?;
        let lin = wrap(k::Layout::strided(&dims, &strides, layout.start_offset()))?;
        wrap(k::reduce(
            &self.device.queue,
            kop,
            self.sd()?,
            &lin,
            &self.buffer,
            &out.buffer,
            out_el,
            reduce_el,
        ))?;
        Ok(out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let kop = match op {
            CmpOp::Eq => k::CmpOp::Eq,
            CmpOp::Ne => k::CmpOp::Ne,
            CmpOp::Lt => k::CmpOp::Lt,
            CmpOp::Le => k::CmpOp::Le,
            CmpOp::Gt => k::CmpOp::Gt,
            CmpOp::Ge => k::CmpOp::Ge,
        };
        let numel = lhs_l.shape().elem_count();
        let out = self.device.alloc(DType::U8, numel)?;
        let ll = ffi_layout(lhs_l)?;
        let rl = ffi_layout(rhs_l)?;
        wrap(k::cmp(
            &self.device.queue,
            kop,
            self.sd()?,
            &ll,
            &self.buffer,
            &rl,
            &rhs.buffer,
            &out.buffer,
            numel,
        ))?;
        Ok(out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(dtype, numel)?;
        let lin = ffi_layout(layout)?;
        wrap(k::cast(
            &self.device.queue,
            self.sd()?,
            to_sycl_dtype(dtype)?,
            &lin,
            &self.buffer,
            &out.buffer,
            numel,
        ))?;
        Ok(out)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let op = match B::NAME {
            "neg" => UnaryOp::Neg,
            "abs" => UnaryOp::Abs,
            "sqr" => UnaryOp::Sqr,
            "sqrt" => UnaryOp::Sqrt,
            "recip" => UnaryOp::Recip,
            "exp" => UnaryOp::Exp,
            "log" => UnaryOp::Log,
            "sin" => UnaryOp::Sin,
            "cos" => UnaryOp::Cos,
            "tanh" => UnaryOp::Tanh,
            "erf" => UnaryOp::Erf,
            "ceil" => UnaryOp::Ceil,
            "floor" => UnaryOp::Floor,
            "round" => UnaryOp::Round,
            "sign" => UnaryOp::Sign,
            "relu" => UnaryOp::Relu,
            "silu" => UnaryOp::Silu,
            "gelu" => UnaryOp::Gelu,
            "gelu_erf" => UnaryOp::GeluErf,
            other => sycl_bail!("unary op {other:?} not implemented for SYCL yet"),
        };
        let numel = layout.shape().elem_count();
        let out = self.device.alloc(self.dtype, numel)?;
        let lin = ffi_layout(layout)?;
        wrap(k::unary(
            &self.device.queue,
            op,
            self.sd()?,
            &lin,
            &self.buffer,
            &out.buffer,
            numel,
        ))?;
        Ok(out)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let op = match B::NAME {
            "add" => BinaryOp::Add,
            "sub" => BinaryOp::Sub,
            "mul" => BinaryOp::Mul,
            "div" => BinaryOp::Div,
            "maximum" => BinaryOp::Maximum,
            "minimum" => BinaryOp::Minimum,
            other => sycl_bail!("binary op {other:?} not implemented for SYCL yet"),
        };
        let numel = lhs_l.shape().elem_count();
        let out = self.device.alloc(self.dtype, numel)?;
        let ll = ffi_layout(lhs_l)?;
        let rl = ffi_layout(rhs_l)?;
        wrap(k::binary(
            &self.device.queue,
            op,
            self.sd()?,
            &ll,
            &self.buffer,
            &rl,
            &rhs.buffer,
            &out.buffer,
            numel,
        ))?;
        Ok(out)
    }

    fn where_cond(
        &self,
        cond_l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let numel = cond_l.shape().elem_count();
        let out = self.device.alloc(t.dtype, numel)?;
        wrap(k::where_cond(
            &self.device.queue,
            self.sd()?,
            t.sd()?,
            &ffi_layout(cond_l)?,
            &self.buffer,
            &ffi_layout(t_l)?,
            &t.buffer,
            &ffi_layout(f_l)?,
            &f.buffer,
            &out.buffer,
            numel,
        ))?;
        Ok(out)
    }

    fn conv1d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let dt = self.sd()?;
        let l_out = p.l_out();
        let kk = p.c_in * p.k_size;
        let rows = p.b_size * l_out;
        let st = inp_l.stride();
        let meta = [
            inp_l.start_offset() as i64,
            st[0] as i64,
            st[1] as i64,
            st[2] as i64,
            p.b_size as i64,
            p.c_in as i64,
            p.l_in as i64,
        ];
        let col = self.device.alloc(self.dtype, rows * kk)?;
        wrap(k::im2col1d(
            &self.device.queue,
            dt,
            &self.buffer,
            &col.buffer,
            &meta,
            p.k_size,
            p.stride,
            p.padding,
            p.dilation,
            l_out,
        ))?;
        let w = contiguous_kernel(kernel, kernel_l, p.c_out, kk)?;
        let w_buf = w.as_ref().map_or(&kernel.buffer, |s| &s.buffer);
        let w_off = if w.is_some() {
            0
        } else {
            kernel_l.start_offset()
        };
        // out_col: (rows, c_out) = col (rows, kk) @ w^T (kk, c_out)
        let out_col = self.device.alloc(self.dtype, rows * p.c_out)?;
        wrap(k::gemm(
            &self.device.queue,
            dt,
            false,
            true,
            rows as i64,
            p.c_out as i64,
            kk as i64,
            1.0,
            0.0,
            &col.buffer,
            w_buf,
            &out_col.buffer,
            1,
            0,
            0,
            0,
            0,
            w_off as i64,
        ))?;
        // (b, l_out, c_out) -> (b, c_out, l_out)
        permute_bxc(
            &self.device,
            dt,
            self.dtype,
            &out_col,
            p.b_size,
            l_out,
            p.c_out,
        )
    }

    fn conv_transpose1d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let dt = self.sd()?;
        let out_l = p.l_out();
        let st = inp_l.stride();
        let im = [
            inp_l.start_offset() as i64,
            st[0] as i64,
            st[1] as i64,
            st[2] as i64,
        ];
        let w = contiguous_kernel(kernel, kernel_l, p.c_in, p.c_out * p.k_size)?;
        let w_buf = w.as_ref().map_or(&kernel.buffer, |s| &s.buffer);
        let out = self.device.alloc(self.dtype, p.b_size * p.c_out * out_l)?;
        wrap(k::conv_transpose1d(
            &self.device.queue,
            dt,
            &self.buffer,
            &im,
            w_buf,
            &out.buffer,
            (p.b_size, p.c_in, p.c_out),
            p.l_in,
            p.k_size,
            out_l,
            (p.stride, p.padding, p.dilation),
        ))?;
        Ok(out)
    }

    fn conv2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let dt = self.sd()?;
        let (out_h, out_w) = (p.out_h(), p.out_w());
        let kk = p.c_in * p.k_h * p.k_w;
        let rows = p.b_size * out_h * out_w;
        let meta = nchw9(inp_l);
        let col = self.device.alloc(self.dtype, rows * kk)?;
        wrap(k::im2col2d(
            &self.device.queue,
            dt,
            &self.buffer,
            &col.buffer,
            &meta,
            (p.k_h, p.k_w),
            p.stride,
            p.padding,
            p.dilation,
            (out_h, out_w),
        ))?;
        let w = contiguous_kernel(kernel, kernel_l, p.c_out, kk)?;
        let w_buf = w.as_ref().map_or(&kernel.buffer, |s| &s.buffer);
        let w_off = if w.is_some() {
            0
        } else {
            kernel_l.start_offset()
        };
        let out_col = self.device.alloc(self.dtype, rows * p.c_out)?;
        wrap(k::gemm(
            &self.device.queue,
            dt,
            false,
            true,
            rows as i64,
            p.c_out as i64,
            kk as i64,
            1.0,
            0.0,
            &col.buffer,
            w_buf,
            &out_col.buffer,
            1,
            0,
            0,
            0,
            0,
            w_off as i64,
        ))?;
        // (b, out_h*out_w, c_out) -> (b, c_out, out_h, out_w)
        permute_bxc(
            &self.device,
            dt,
            self.dtype,
            &out_col,
            p.b_size,
            out_h * out_w,
            p.c_out,
        )
    }

    fn conv_transpose2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let dt = self.sd()?;
        let (out_h, out_w) = (p.out_h(), p.out_w());
        let st = inp_l.stride();
        let im = [
            inp_l.start_offset() as i64,
            st[0] as i64,
            st[1] as i64,
            st[2] as i64,
            st[3] as i64,
        ];
        let w = contiguous_kernel(kernel, kernel_l, p.c_in, p.c_out * p.k_h * p.k_w)?;
        let w_buf = w.as_ref().map_or(&kernel.buffer, |s| &s.buffer);
        let out = self
            .device
            .alloc(self.dtype, p.b_size * p.c_out * out_h * out_w)?;
        wrap(k::conv_transpose2d(
            &self.device.queue,
            dt,
            &self.buffer,
            &im,
            w_buf,
            &out.buffer,
            (p.b_size, p.c_in, p.c_out),
            (p.i_h, p.i_w),
            (p.k_h, p.k_w),
            (out_h, out_w),
            (p.stride, p.padding, p.dilation),
        ))?;
        Ok(out)
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let (b, c, h, w) = l.shape().dims4()?;
        let h_out = (h - k.0) / s.0 + 1;
        let w_out = (w - k.1) / s.1 + 1;
        let out = self.device.alloc(self.dtype, b * c * h_out * w_out)?;
        wrap(k::avg_pool2d(
            &self.device.queue,
            self.sd()?,
            &self.buffer,
            &out.buffer,
            &nchw9(l),
            k,
            s,
            h_out,
            w_out,
        ))?;
        Ok(out)
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let (b, c, h, w) = l.shape().dims4()?;
        let h_out = (h - k.0) / s.0 + 1;
        let w_out = (w - k.1) / s.1 + 1;
        let out = self.device.alloc(self.dtype, b * c * h_out * w_out)?;
        wrap(k::max_pool2d(
            &self.device.queue,
            self.sd()?,
            &self.buffer,
            &out.buffer,
            &nchw9(l),
            k,
            s,
            h_out,
            w_out,
        ))?;
        Ok(out)
    }

    fn upsample_nearest1d(&self, l: &Layout, dst: usize) -> Result<Self> {
        let (b, c, _w) = l.shape().dims3()?;
        let st = l.stride();
        let src = [
            l.start_offset() as i64,
            st[0] as i64,
            st[1] as i64,
            st[2] as i64,
            b as i64,
            c as i64,
            l.dims()[2] as i64,
        ];
        let out = self.device.alloc(self.dtype, b * c * dst)?;
        wrap(k::upsample_nearest1d(
            &self.device.queue,
            self.sd()?,
            &self.buffer,
            &out.buffer,
            &src,
            dst,
        ))?;
        Ok(out)
    }

    fn upsample_nearest2d(&self, l: &Layout, dh: usize, dw: usize) -> Result<Self> {
        let (b, c, _, _) = l.shape().dims4()?;
        let out = self.device.alloc(self.dtype, b * c * dh * dw)?;
        wrap(k::upsample_nearest2d(
            &self.device.queue,
            self.sd()?,
            &self.buffer,
            &out.buffer,
            &nchw9(l),
            dh,
            dw,
        ))?;
        Ok(out)
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        dh: usize,
        dw: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let (b, c, hin, win) = l.shape().dims4()?;
        // PyTorch area_pixel_compute_scale, mirroring cpu_backend.
        let sh = if align_corners {
            if dh > 1 {
                (hin - 1) as f64 / (dh - 1) as f64
            } else {
                0.0
            }
        } else {
            scale_h.map_or(hin as f64 / dh as f64, |f| 1.0 / f)
        };
        let sw = if align_corners {
            if dw > 1 {
                (win - 1) as f64 / (dw - 1) as f64
            } else {
                0.0
            }
        } else {
            scale_w.map_or(win as f64 / dw as f64, |f| 1.0 / f)
        };
        let out = self.device.alloc(self.dtype, b * c * dh * dw)?;
        wrap(k::upsample_bilinear2d(
            &self.device.queue,
            self.sd()?,
            &self.buffer,
            &out.buffer,
            &nchw9(l),
            dh,
            dw,
            align_corners,
            sh,
            sw,
        ))?;
        Ok(out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        require_contig(ids_l, "gather ids")?;
        let sdims = l.dims();
        let left: usize = sdims[..dim].iter().product();
        let right: usize = sdims[dim + 1..].iter().product();
        let src_dim = sdims[dim];
        let ids_dim = ids_l.dims()[dim];
        let out = self.device.alloc(self.dtype, left * ids_dim * right)?;
        wrap(k::gather(
            &self.device.queue,
            self.sd()?,
            to_sycl_dtype(ids.dtype)?,
            &ffi_layout(l)?,
            &self.buffer,
            &ids.buffer,
            &out.buffer,
            left,
            src_dim,
            ids_dim,
            right,
        ))?;
        Ok(out)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_impl(false, l, ids, ids_l, src, src_l, dim)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_impl(true, l, ids, ids_l, src, src_l, dim)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        require_contig(ids_l, "index_select ids")?;
        let sdims = l.dims();
        let left: usize = sdims[..dim].iter().product();
        let right: usize = sdims[dim + 1..].iter().product();
        let src_dim = sdims[dim];
        let ids_dim = ids_l.shape().elem_count();
        let out = self.device.alloc(self.dtype, left * ids_dim * right)?;
        // The source may be strided; `lin` carries that.
        let lin = ffi_layout(l)?;
        wrap(k::index_select(
            &self.device.queue,
            self.sd()?,
            to_sycl_dtype(ids.dtype)?,
            &lin,
            &self.buffer,
            &ids.buffer,
            &out.buffer,
            left,
            src_dim,
            ids_dim,
            right,
        ))?;
        Ok(out)
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        require_contig(l, "index_add self")?;
        require_contig(ids_l, "index_add ids")?;
        require_contig(src_l, "index_add src")?;
        let sdims = l.dims();
        let left: usize = sdims[..dim].iter().product();
        let right: usize = sdims[dim + 1..].iter().product();
        let dst_dim = sdims[dim];
        let ids_dim = ids_l.shape().elem_count();
        // out = self.clone(); then accumulate.
        let out = self.try_clone(l)?;
        wrap(k::index_add(
            &self.device.queue,
            self.sd()?,
            to_sycl_dtype(ids.dtype)?,
            &out.buffer,
            &ids.buffer,
            &src.buffer,
            left,
            ids_dim,
            dst_dim,
            right,
        ))?;
        Ok(out)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, kk): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let dt = self.sd()?;
        // oneMKL's strided `gemm_batch` wants one uniform batch stride and a
        // row-major / plain-transpose 2-D tile. Anything it can't express
        // directly (a broadcast/zero batch stride, a ragged batch nesting, or a
        // 2-D tile that is neither row-major nor a plain transpose — e.g. a
        // doubly-strided narrow view of a KV cache) is first materialised into a
        // dense row-major `b*rows*cols` copy. `mm_operand` returns `None` for
        // the stride in exactly those cases.
        let prep = |buf: &DeviceBuffer,
                    l: &Layout,
                    rows: usize,
                    cols: usize|
         -> Result<(Option<SyclStorage>, bool, usize)> {
            let (trans, stride) = mm_operand(l, rows, cols)?;
            let need_dense = match stride {
                None => true,
                Some(0) => b > 1,
                Some(_) => false,
            };
            if need_dense {
                let dense = self.device.alloc(self.dtype, b * rows * cols)?;
                wrap(k::copy_strided(
                    &self.device.queue,
                    dt,
                    &ffi_layout(l)?,
                    buf,
                    &dense.buffer,
                    0,
                    b * rows * cols,
                ))?;
                Ok((Some(dense), false, rows * cols))
            } else {
                Ok((None, trans, stride.unwrap_or(rows * cols)))
            }
        };
        let (lhs_owned, transa, stride_a) = prep(&self.buffer, lhs_l, m, kk)?;
        let (rhs_owned, transb, stride_b) = prep(&rhs.buffer, rhs_l, kk, n)?;
        let off_a = if lhs_owned.is_some() {
            0
        } else {
            lhs_l.start_offset()
        };
        let off_b = if rhs_owned.is_some() {
            0
        } else {
            rhs_l.start_offset()
        };
        let lhs_buf = lhs_owned.as_ref().map_or(&self.buffer, |s| &s.buffer);
        let rhs_buf = rhs_owned.as_ref().map_or(&rhs.buffer, |s| &s.buffer);
        let out = self.device.alloc(self.dtype, b * m * n)?;
        wrap(k::gemm(
            &self.device.queue,
            dt,
            transa,
            transb,
            m as i64,
            n as i64,
            kk as i64,
            1.0,
            0.0,
            lhs_buf,
            rhs_buf,
            &out.buffer,
            b as i64,
            stride_a as i64,
            stride_b as i64,
            (m * n) as i64,
            off_a as i64,
            off_b as i64,
        ))?;
        Ok(out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let numel = src_l.shape().elem_count();
        let lin = ffi_layout(src_l)?;
        wrap(k::copy_strided(
            &self.device.queue,
            self.sd()?,
            &lin,
            &self.buffer,
            &dst.buffer,
            dst_offset,
            numel,
        ))
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
        wrap(k::copy2d(
            &self.device.queue,
            self.sd()?,
            &self.buffer,
            &dst.buffer,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        ))
    }

    fn const_set(&mut self, value: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let numel = layout.shape().elem_count();
        if layout.start_offset() == 0 && layout.is_contiguous() {
            wrap(k::fill(
                &self.device.queue,
                self.sd()?,
                &self.buffer,
                numel,
                value.to_f64(),
            ))
        } else {
            wrap(k::fill_strided(
                &self.device.queue,
                self.sd()?,
                &ffi_layout(layout)?,
                &self.buffer,
                numel,
                value.to_f64(),
            ))
        }
    }
}

/// Materialise a conv kernel `(c_out, ...)` into a contiguous `(c_out, kk)`
/// buffer if it is a view, else return `None` (use it in place at its offset).
fn contiguous_kernel(
    kernel: &SyclStorage,
    kernel_l: &Layout,
    c_out: usize,
    kk: usize,
) -> Result<Option<SyclStorage>> {
    if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
        return Ok(None);
    }
    let dense = kernel.device.alloc(kernel.dtype, c_out * kk)?;
    wrap(k::copy_strided(
        &kernel.device.queue,
        to_sycl_dtype(kernel.dtype)?,
        &ffi_layout(kernel_l)?,
        &kernel.buffer,
        &dense.buffer,
        0,
        c_out * kk,
    ))?;
    Ok(Some(dense))
}

/// `(b, x, c)` contiguous -> dense `(b, c, x)`, for the conv output permute.
fn permute_bxc(
    dev: &SyclDevice,
    dt: SyclDType,
    dtype: DType,
    src: &SyclStorage,
    b: usize,
    x: usize,
    c: usize,
) -> Result<SyclStorage> {
    let out = dev.alloc(dtype, b * c * x)?;
    let lin = wrap(k::Layout::strided(&[b, c, x], &[x * c, 1, c], 0))?;
    wrap(k::copy_strided(
        &dev.queue,
        dt,
        &lin,
        &src.buffer,
        &out.buffer,
        0,
        b * c * x,
    ))?;
    Ok(out)
}

/// `[offset, stride_b, stride_c, stride_h, stride_w, b, c, h, w]` for an NCHW layout.
fn nchw9(l: &Layout) -> [i64; 9] {
    let d = l.dims();
    let s = l.stride();
    [
        l.start_offset() as i64,
        s[0] as i64,
        s[1] as i64,
        s[2] as i64,
        s[3] as i64,
        d[0] as i64,
        d[1] as i64,
        d[2] as i64,
        d[3] as i64,
    ]
}

fn require_contig(l: &Layout, what: &str) -> Result<()> {
    if l.start_offset() == 0 && l.is_contiguous() {
        Ok(())
    } else {
        Err(crate::Error::Sycl(
            SyclError::msg(format!(
                "{what}: SYCL indexing kernels need a contiguous, zero-offset operand                  (got strides {:?}); call .contiguous() upstream — general strided support                  is a later phase",
                l.stride()
            ))
            .into(),
        ))
    }
}

impl SyclStorage {
    #[allow(clippy::too_many_arguments)]
    fn scatter_impl(
        &mut self,
        add: bool,
        l: &Layout,
        ids: &SyclStorage,
        ids_l: &Layout,
        src: &SyclStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        require_contig(l, "scatter self")?;
        require_contig(ids_l, "scatter ids")?;
        require_contig(src_l, "scatter src")?;
        let sdims = l.dims();
        let left: usize = sdims[..dim].iter().product();
        let right: usize = sdims[dim + 1..].iter().product();
        let dst_dim = sdims[dim];
        let src_dim = src_l.dims()[dim];
        wrap(k::scatter(
            &self.device.queue,
            add,
            self.sd()?,
            to_sycl_dtype(ids.dtype)?,
            &self.buffer,
            &ids.buffer,
            &src.buffer,
            left,
            src_dim,
            dst_dim,
            right,
        ))
    }

    /// Per-row argsort along the last dim. Returns a `U32` storage of indices.
    /// Used by `ArgSort::sycl_fwd` in `sort.rs`.
    pub fn argsort(&self, layout: &Layout, asc: bool, last_dim: usize) -> Result<Self> {
        let numel = layout.shape().elem_count();
        let nrows = numel / last_dim.max(1);
        let out = self.device.alloc(DType::U32, numel)?;
        // The bitonic kernel wants a dense row-major buffer; materialise one if
        // the input is a view.
        let dense;
        let src = if layout.start_offset() == 0 && layout.is_contiguous() {
            &self.buffer
        } else {
            dense = self.device.alloc(self.dtype, numel)?;
            wrap(k::copy_strided(
                &self.device.queue,
                self.sd()?,
                &ffi_layout(layout)?,
                &self.buffer,
                &dense.buffer,
                0,
                numel,
            ))?;
            &dense.buffer
        };
        wrap(k::argsort(
            &self.device.queue,
            self.sd()?,
            asc,
            src,
            &out.buffer,
            nrows,
            last_dim,
        ))?;
        Ok(out)
    }
}

/// Classify a matmul operand: is it stored transposed relative to row-major, and
/// what is its batch stride. `rows`/`cols` are the logical (non-transposed) dims.
/// Describes how a matmul operand's layout maps onto oneMKL `gemm_batch`.
///
/// Returns `(trans, Some(batch_stride))` when the layout is usable directly:
/// `batch_stride == 0` marks a broadcast operand (the caller densifies it when
/// `b > 1`). Returns `(_, None)` when the layout cannot be expressed as a
/// row-major / plain-transpose tile with a uniform batch stride and the caller
/// must materialise a dense row-major copy first.
fn mm_operand(l: &Layout, rows: usize, cols: usize) -> Result<(bool, Option<usize>)> {
    let stride = l.stride();
    let rank = stride.len();
    if rank < 2 {
        sycl_bail!("matmul operand rank {rank} < 2");
    }
    let dims = l.dims();
    let (rs, cs) = (stride[rank - 2], stride[rank - 1]);
    // Same acceptance rule as `cuda_backend::gemm_config`: inner stride 1 (or a
    // unit dim) => not transposed; outer stride 1 (or a unit dim) => transposed.
    let trans = if (cs == 1 || cols == 1) && (rs == cols || rows == 1) {
        false
    } else if (rs == 1 || rows == 1) && (cs == rows || cols == 1) {
        true
    } else {
        // Neither row-major nor a plain transpose (e.g. a doubly-strided narrow
        // view): caller densifies.
        return Ok((false, None));
    };
    let batch_dims = &dims[..rank - 2];
    let batch_strides = &stride[..rank - 2];
    let batch: usize = batch_dims.iter().product();
    let inner = rows * cols;
    if batch <= 1 {
        return Ok((trans, Some(inner)));
    }
    if batch_strides.iter().all(|&s| s == 0) {
        return Ok((trans, Some(0))); // broadcast operand
    }
    let mut expect = inner;
    for i in (0..batch_dims.len()).rev() {
        if batch_dims[i] != 1 && batch_strides[i] != expect {
            // Ragged batch nesting: caller densifies.
            return Ok((false, None));
        }
        expect *= batch_dims[i];
    }
    Ok((trans, Some(inner)))
}

impl BackendDevice for SyclDevice {
    type Storage = SyclStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let queue = wrap(Queue::new(ordinal))?;
        Ok(Self { ordinal, queue })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Sycl {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.queue, &rhs.queue)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let s = self.alloc(dtype, shape.elem_count())?;
        wrap(s.buffer.memset_zero())?;
        Ok(s)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc(dtype, shape.elem_count())
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let s = self.alloc(T::DTYPE, data.len())?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        wrap(s.buffer.copy_from_host(bytes))?;
        Ok(s)
    }

    fn storage_from_cpu_storage(&self, c: &CpuStorage) -> Result<Self::Storage> {
        let (bytes, len) = cpu_bytes(c);
        let s = self.alloc(c.dtype(), len)?;
        wrap(s.buffer.copy_from_host(bytes))?;
        Ok(s)
    }

    fn storage_from_cpu_storage_owned(&self, c: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&c)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        // No on-device RNG yet: generate on the CPU and upload. Correct, just an
        // extra host->device copy (~33 GB/s on Meteor Lake, per Phase 0).
        let cpu = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, up)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        sycl_bail!("set_seed")
    }

    fn get_current_seed(&self) -> Result<u64> {
        sycl_bail!("get_current_seed")
    }

    fn synchronize(&self) -> Result<()> {
        wrap(self.queue.synchronize())
    }
}

// GEMM reduced-precision knobs, kept at the same path shape as `candle::cuda`.
pub fn gemm_reduced_precision_f16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f16(_: bool) {}
pub fn gemm_reduced_precision_bf16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_bf16(_: bool) {}
pub fn gemm_reduced_precision_f32() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f32(_: bool) {}
