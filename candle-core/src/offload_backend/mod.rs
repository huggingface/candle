//! A candle backend built on Rust's `std::offload`.
//!
//! # Why this occupies the CUDA slot
//!
//! candle's `Device` and `Storage` enums are closed, and every match arm across
//! candle-core, candle-nn and candle-transformers enumerates `Cpu | Cuda |
//! Metal`. Adding a fourth variant would touch hundreds of sites. But candle
//! already compiles a *dummy* CUDA backend when the `cuda` feature is off, so
//! the `Cuda` variant is a ready-made slot for "some GPU that is not Metal".
//! This module replaces that dummy, which is why `Device::new_cuda(0)` returns
//! an offload device here and why the diff to candle proper is a handful of
//! lines.
//!
//! # Why the kernels are in another crate
//!
//! `offload!` mangles the kernel symbol from the crate that defines it, so the
//! macro invocation and the `#[offload_kernel]` have to be compiled together --
//! and that crate is also compiled for `amdgcn-amd-amdhsa` as `no_std`.
//! candle-core cannot be, so the kernels live in `candle-offload-kernels` and
//! this module only calls its launchers.
//!
//! # Where the data lives
//!
//! `std::offload` has no device allocation: `offload!` infers transfers from
//! argument types, so a buffer passed as `&[T]` is copied to the device before
//! the launch and back afterwards, every time. That is fatal for a framework
//! backend, where `alloc_uninit` must hand back storage that *stays* on the
//! device across hundreds of ops. So allocation goes directly to libomptarget
//! (`ffi`), and device addresses reach kernels as `usize` scalars, which
//! `offload!` passes by value and therefore does not map.
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::ffi::c_void;
use std::sync::atomic::Ordering;

pub mod ffi;
pub mod kernels;
mod ops;

pub use ffi::report as stats_report;
pub use ops::{rms_norm, rope, softmax_last_dim};


/// Runtime switches, so the GPU paths can be turned off one at a time without
/// rebuilding. `CANDLE_OFFLOAD_CPU_ONLY=1` routes every op through the CPU
/// fallback while still keeping tensors resident on the device -- that isolates
/// the cost of the transfers from the cost of the kernels.
pub(crate) struct Config {
    pub cpu_only: bool,
    pub disabled: Vec<String>,
}

impl Config {
    fn get() -> &'static Config {
        static C: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
        C.get_or_init(|| Config {
            cpu_only: std::env::var("CANDLE_OFFLOAD_CPU_ONLY").is_ok_and(|v| v != "0"),
            disabled: std::env::var("CANDLE_OFFLOAD_DISABLE")
                .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default(),
        })
    }
}

pub(crate) fn gpu_allowed(op: &str) -> bool {
    let c = Config::get();
    !c.cpu_only && !c.disabled.iter().any(|d| d == op)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic::AtomicUsize;
        static C: AtomicUsize = AtomicUsize::new(1);
        Self(C.fetch_add(1, Ordering::Relaxed))
    }
}

/// An owned device allocation. Dropping it calls `omp_target_free`, which is
/// the piece `std::offload` cannot express: its mapped buffers are freed when
/// the launch returns, so nothing can outlive a single kernel.
#[derive(Debug)]
pub struct DevBuf {
    ptr: usize,
    bytes: usize,
    ordinal: i32,
}

impl DevBuf {
    fn alloc(bytes: usize, ordinal: i32) -> Result<Self> {
        // omp_target_alloc(0) is allowed to return null, which is not an error
        // here -- candle makes plenty of zero-element tensors.
        if bytes == 0 {
            return Ok(Self { ptr: 0, bytes: 0, ordinal });
        }
        let p = unsafe { ffi::omp_target_alloc(bytes, ordinal) };
        if p.is_null() {
            crate::bail!("offload: omp_target_alloc({bytes}) failed on device {ordinal}")
        }
        ffi::STATS.allocs.fetch_add(1, Ordering::Relaxed);
        ffi::STATS.alloc_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(Self { ptr: p as usize, bytes, ordinal })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { ffi::omp_target_free(self.ptr as *mut c_void, self.ordinal) };
            ffi::STATS.frees.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone)]
pub struct OffloadDevice {
    ordinal: i32,
    id: DeviceId,
    seed: std::sync::Arc<std::sync::Mutex<u64>>,
}

#[derive(Debug)]
pub struct OffloadStorage {
    pub(crate) buf: std::sync::Arc<DevBuf>,
    pub(crate) dtype: DType,
    pub(crate) device: OffloadDevice,
}

impl OffloadStorage {
    #[inline]
    pub(crate) fn ptr(&self) -> usize {
        self.buf.ptr
    }

    #[inline]
    pub(crate) fn elem_count(&self) -> usize {
        let sz = self.dtype.size_in_bytes();
        if sz == 0 {
            0
        } else {
            self.buf.bytes / sz
        }
    }

    /// Byte address of element `i`.
    #[inline]
    pub(crate) fn at(&self, i: usize) -> usize {
        self.buf.ptr + i * self.dtype.size_in_bytes()
    }
}

// Pull the raw bytes out of a CpuStorage regardless of which variant it is.
// candle has no byte-level accessor, and the backend only ever needs to move
// these across the bus, not interpret them.
macro_rules! cpu_ptr_len {
    ($s:expr, $conv:ident, [$($v:ident),*]) => {
        match $s {
            $( crate::CpuStorage::$v(vec) => {
                let n = std::mem::size_of_val(&vec[..]);
                (vec.$conv() as _, n)
            } ),*
        }
    };
}

pub(crate) fn cpu_bytes(s: &CpuStorage) -> (*const u8, usize) {
    cpu_ptr_len!(s, as_ptr, [U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0])
}

pub(crate) fn cpu_bytes_mut(s: &mut CpuStorage) -> (*mut u8, usize) {
    cpu_ptr_len!(s, as_mut_ptr, [U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0])
}

impl OffloadDevice {
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn new_with_stream(_: usize) -> Result<Self> {
        // G2: there are no streams. `offload!` is synchronous and there is no
        // way to express a queue, so this is the same device as `new`.
        crate::bail!("offload: std::offload exposes no stream/queue abstraction")
    }

    fn host(&self) -> i32 {
        unsafe { ffi::omp_get_initial_device() }
    }

    pub(crate) fn upload(&self, dst: usize, src: *const u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let rc = unsafe {
            ffi::omp_target_memcpy(
                dst as *mut c_void,
                src as *mut c_void,
                bytes,
                0,
                0,
                self.ordinal,
                self.host(),
            )
        };
        if rc != 0 {
            crate::bail!("offload: host->device memcpy of {bytes} bytes failed (rc={rc})")
        }
        ffi::STATS.h2d_calls.fetch_add(1, Ordering::Relaxed);
        ffi::STATS.h2d_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn download(&self, dst: *mut u8, src: usize, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let rc = unsafe {
            ffi::omp_target_memcpy(
                dst as *mut c_void,
                src as *mut c_void,
                bytes,
                0,
                0,
                self.host(),
                self.ordinal,
            )
        };
        if rc != 0 {
            crate::bail!("offload: device->host memcpy of {bytes} bytes failed (rc={rc})")
        }
        ffi::STATS.d2h_calls.fetch_add(1, Ordering::Relaxed);
        ffi::STATS.d2h_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn alloc(&self, bytes: usize, dtype: DType) -> Result<OffloadStorage> {
        Ok(OffloadStorage {
            buf: std::sync::Arc::new(DevBuf::alloc(bytes, self.ordinal)?),
            dtype,
            device: self.clone(),
        })
    }

    pub(crate) fn alloc_elems(&self, elems: usize, dtype: DType) -> Result<OffloadStorage> {
        self.alloc(elems * dtype.size_in_bytes(), dtype)
    }
}

impl OffloadDevice {
    /// Device-to-device copy. Used for `try_clone`, contiguous
    /// `copy_strided_src` and `copy2d`, so those never touch the host bus --
    /// and they work for every dtype, including the ones with no kernels.
    pub(crate) fn device_copy(&self, dst: usize, src: usize, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let rc = unsafe {
            ffi::omp_target_memcpy(
                dst as *mut c_void,
                src as *mut c_void,
                bytes,
                0,
                0,
                self.ordinal,
                self.ordinal,
            )
        };
        if rc != 0 {
            crate::bail!("offload: device->device memcpy of {bytes} bytes failed (rc={rc})")
        }
        Ok(())
    }
}

impl OffloadStorage {
    /// Pull the whole buffer down, not just the logically-addressed elements:
    /// the caller still holds a `Layout` with offsets and strides into it.
    pub(crate) fn cpu(&self) -> Result<CpuStorage> {
        let n = self.elem_count();
        let mut c = unsafe {
            crate::cpu_backend::CpuDevice.alloc_uninit(&Shape::from_dims(&[n]), self.dtype)?
        };
        let (p, len) = cpu_bytes_mut(&mut c);
        self.device.download(p, self.ptr(), len)?;
        Ok(c)
    }

    fn from_cpu(&self, c: &CpuStorage) -> Result<Self> {
        self.device.storage_from_cpu_storage(c)
    }

    fn write_back(&mut self, c: &CpuStorage) -> Result<()> {
        let (p, len) = cpu_bytes(c);
        if len != self.buf.bytes {
            crate::bail!(
                "offload: write-back size mismatch ({len} vs {})",
                self.buf.bytes
            )
        }
        self.device.upload(self.ptr(), p, len)
    }

    /// One-operand CPU fallback: device -> host, run candle's CPU kernel,
    /// host -> device. Every call is two bus crossings.
    fn fb1<F>(&self, op: &'static str, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage) -> Result<CpuStorage>,
    {
        let t0 = ffi::bus_bytes();
        let c = self.cpu()?;
        let r = f(&c)?;
        let out = self.from_cpu(&r);
        ffi::note_fallback(op, ffi::bus_bytes() - t0);
        out
    }

    fn fb2<F>(&self, op: &'static str, rhs: &Self, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &CpuStorage) -> Result<CpuStorage>,
    {
        let t0 = ffi::bus_bytes();
        let a = self.cpu()?;
        let b = rhs.cpu()?;
        let r = f(&a, &b)?;
        let out = self.from_cpu(&r);
        ffi::note_fallback(op, ffi::bus_bytes() - t0);
        out
    }

    fn fb3<F>(&self, op: &'static str, x: &Self, y: &Self, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &CpuStorage, &CpuStorage) -> Result<CpuStorage>,
    {
        let t0 = ffi::bus_bytes();
        let a = self.cpu()?;
        let b = x.cpu()?;
        let c = y.cpu()?;
        let r = f(&a, &b, &c)?;
        let out = self.from_cpu(&r);
        ffi::note_fallback(op, ffi::bus_bytes() - t0);
        out
    }
}

impl BackendStorage for OffloadStorage {
    type Device = OffloadDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        let out = self.device.alloc(self.buf.bytes, self.dtype)?;
        self.device.device_copy(out.ptr(), self.ptr(), self.buf.bytes)?;
        Ok(out)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.cpu()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        if let Some(s) = ops::affine(self, l, mul, add)? {
            return Ok(s);
        }
        self.fb1("affine", |c| c.affine(l, mul, add))
    }

    fn powf(&self, l: &Layout, e: f64) -> Result<Self> {
        self.fb1("powf", |c| c.powf(l, e))
    }

    fn elu(&self, l: &Layout, a: f64) -> Result<Self> {
        self.fb1("elu", |c| c.elu(l, a))
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        self.fb1("reduce_op", |c| c.reduce_op(op, l, dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        self.fb2("cmp", rhs, |a, b| a.cmp(op, b, lhs_l, rhs_l))
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        if let Some(s) = ops::to_dtype(self, l, dtype)? {
            return Ok(s);
        }
        self.fb1("to_dtype", |c| c.to_dtype(l, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        if let Some(s) = ops::unary(self, l, B::NAME)? {
            return Ok(s);
        }
        self.fb1(B::NAME, |c| c.unary_impl::<B>(l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if let Some(s) = ops::binary(self, rhs, lhs_l, rhs_l, B::NAME)? {
            return Ok(s);
        }
        self.fb2(B::NAME, rhs, |a, b| a.binary_impl::<B>(b, lhs_l, rhs_l))
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        tl: &Layout,
        f: &Self,
        fl: &Layout,
    ) -> Result<Self> {
        self.fb3("where_cond", t, f, |a, b, c| a.where_cond(l, b, tl, c, fl))
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kl: &Layout,
        p: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.fb2("conv1d", kernel, |a, b| a.conv1d(l, b, kl, p))
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kl: &Layout,
        p: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.fb2("conv_transpose1d", kernel, |a, b| {
            a.conv_transpose1d(l, b, kl, p)
        })
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kl: &Layout,
        p: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.fb2("conv2d", kernel, |a, b| a.conv2d(l, b, kl, p))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kl: &Layout,
        p: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.fb2("conv_transpose2d", kernel, |a, b| {
            a.conv_transpose2d(l, b, kl, p)
        })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        self.fb1("avg_pool2d", |c| c.avg_pool2d(l, k, s))
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        self.fb1("max_pool2d", |c| c.max_pool2d(l, k, s))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        self.fb1("upsample_nearest1d", |c| c.upsample_nearest1d(l, sz))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        self.fb1("upsample_nearest2d", |c| c.upsample_nearest2d(l, h, w))
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align: bool,
        sh: Option<f64>,
        sw: Option<f64>,
    ) -> Result<Self> {
        self.fb1("upsample_bilinear2d", |c| {
            c.upsample_bilinear2d(l, h, w, align, sh, sw)
        })
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        self.fb2("gather", ids, |a, b| a.gather(l, b, ids_l, dim))
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
        let t0 = ffi::bus_bytes();
        let mut a = self.cpu()?;
        let i = ids.cpu()?;
        let s = src.cpu()?;
        a.scatter_set(l, &i, ids_l, &s, src_l, dim)?;
        let r = self.write_back(&a);
        ffi::note_fallback("scatter_set", ffi::bus_bytes() - t0);
        r
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
        let t0 = ffi::bus_bytes();
        let mut a = self.cpu()?;
        let i = ids.cpu()?;
        let s = src.cpu()?;
        a.scatter_add_set(l, &i, ids_l, &s, src_l, dim)?;
        let r = self.write_back(&a);
        ffi::note_fallback("scatter_add_set", ffi::bus_bytes() - t0);
        r
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if let Some(s) = ops::index_select(self, ids, l, ids_l, dim)? {
            return Ok(s);
        }
        self.fb2("index_select", ids, |a, b| a.index_select(b, l, ids_l, dim))
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
        self.fb3("index_add", ids, src, |a, b, c| {
            a.index_add(l, b, ids_l, c, src_l, dim)
        })
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if let Some(s) = ops::matmul(self, rhs, bmnk, lhs_l, rhs_l)? {
            return Ok(s);
        }
        self.fb2("matmul", rhs, |a, b| a.matmul(b, bmnk, lhs_l, rhs_l))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        // Contiguous source is a straight device-to-device copy: no kernel, no
        // bus traffic, and it works for every dtype.
        if src_l.is_contiguous() {
            let n = src_l.shape().elem_count();
            let sz = self.dtype.size_in_bytes();
            return self.device.device_copy(
                dst.at(dst_offset),
                self.at(src_l.start_offset()),
                n * sz,
            );
        }
        let t0 = ffi::bus_bytes();
        let a = self.cpu()?;
        let mut d = dst.cpu()?;
        a.copy_strided_src(&mut d, dst_offset, src_l)?;
        let r = dst.write_back(&d);
        ffi::note_fallback("copy_strided_src", ffi::bus_bytes() - t0);
        r
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
        let sz = self.dtype.size_in_bytes();
        for i in 0..d1 {
            self.device.device_copy(
                dst.at(dst_offset + i * dst_stride1),
                self.at(src_offset + i * src_stride1),
                d2 * sz,
            )?;
        }
        Ok(())
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        if ops::const_set(self, v, l)? {
            return Ok(());
        }
        let t0 = ffi::bus_bytes();
        let mut c = self.cpu()?;
        c.const_set(v, l)?;
        let r = self.write_back(&c);
        ffi::note_fallback("const_set", ffi::bus_bytes() - t0);
        r
    }
}

impl BackendDevice for OffloadDevice {
    type Storage = OffloadStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let n = unsafe { ffi::omp_get_num_devices() };
        if n <= 0 {
            crate::bail!("offload: libomptarget reports no devices")
        }
        if ordinal >= n as usize {
            crate::bail!("offload: device {ordinal} out of range, {n} device(s) visible")
        }
        Ok(Self {
            ordinal: ordinal as i32,
            id: DeviceId::new(),
            seed: std::sync::Arc::new(std::sync::Mutex::new(299792458)),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.ordinal as usize,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let n = shape.elem_count();
        let mut s = self.alloc_elems(n, dtype)?;
        // `omp_target_alloc` does not zero, and there is only an f32 fill
        // kernel, so other dtypes are zeroed by uploading a host buffer.
        if dtype == DType::F32 && gpu_allowed("const_set") {
            unsafe { kernels::candle_offload_fill_f32(s.ptr(), n, 0.0) };
            ffi::note_launch();
        } else {
            let zeros = vec![0u8; s.buf.bytes];
            self.upload(s.ptr(), zeros.as_ptr(), zeros.len())?;
        }
        let _ = &mut s;
        Ok(s)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc_elems(shape.elem_count(), dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, c: &CpuStorage) -> Result<Self::Storage> {
        let (p, len) = cpu_bytes(c);
        let s = self.alloc(len, c.dtype())?;
        self.upload(s.ptr(), p, len)?;
        Ok(s)
    }

    fn storage_from_cpu_storage_owned(&self, c: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&c)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        // No device RNG: `core::arch::amdgpu` has no random source and there is
        // no counter-based generator in core, so this is generated on the host.
        let c = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, up)?;
        self.storage_from_cpu_storage(&c)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let c = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&c)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed.lock().unwrap() = seed;
        crate::cpu_backend::CpuDevice.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed.lock().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        // G2: `offload!` is synchronous -- the launch does not return until the
        // kernel has completed -- so there is nothing to wait for. There is
        // also no way to express a non-blocking launch, which is why this is
        // correct rather than merely convenient.
        Ok(())
    }
}

/// candle's `Device::Cuda` variant holds this type when the `offload` feature
/// is on; these aliases let `lib.rs` swap backends without touching any of the
/// match arms that name `CudaDevice`/`CudaStorage`.
pub type CudaDevice = OffloadDevice;
pub type CudaStorage = OffloadStorage;

impl OffloadStorage {
    /// Present on the real CUDA backend, and reachable from `Tensor::to_device`.
    pub fn transfer_to_device(&self, dst: &OffloadDevice) -> Result<Self> {
        if dst.same_device(&self.device) {
            return self.try_clone(&Layout::contiguous(Shape::from_dims(&[self.elem_count()])));
        }
        let c = self.cpu()?;
        dst.storage_from_cpu_storage(&c)
    }
}

// ---------------------------------------------------------------------------
// Bridge for candle-nn's CustomOps
// ---------------------------------------------------------------------------
//
// candle-nn implements `cuda_fwd` for `rms_norm`, `softmax_last_dim` and
// friends only under `#[cfg(feature = "cuda")]`, and the trait's default
// returns an error. So a model can have every primitive op working on this
// backend and still fail the moment it calls `rms_norm` -- which is every
// transformer. These run the op's own CPU kernel and move the result back, and
// they are counted as fallbacks so the cost shows up in the report.

pub fn custom_via_cpu1<F>(s: &OffloadStorage, f: F) -> Result<(OffloadStorage, Shape)>
where
    F: FnOnce(&CpuStorage) -> Result<(CpuStorage, Shape)>,
{
    let t0 = ffi::bus_bytes();
    let c = s.cpu()?;
    let (out, shape) = f(&c)?;
    let r = s.device.storage_from_cpu_storage(&out)?;
    ffi::note_fallback("custom-op(1)", ffi::bus_bytes() - t0);
    Ok((r, shape))
}

pub fn custom_via_cpu2<F>(
    s1: &OffloadStorage,
    s2: &OffloadStorage,
    f: F,
) -> Result<(OffloadStorage, Shape)>
where
    F: FnOnce(&CpuStorage, &CpuStorage) -> Result<(CpuStorage, Shape)>,
{
    let t0 = ffi::bus_bytes();
    let a = s1.cpu()?;
    let b = s2.cpu()?;
    let (out, shape) = f(&a, &b)?;
    let r = s1.device.storage_from_cpu_storage(&out)?;
    ffi::note_fallback("custom-op(2)", ffi::bus_bytes() - t0);
    Ok((r, shape))
}

pub fn custom_via_cpu3<F>(
    s1: &OffloadStorage,
    s2: &OffloadStorage,
    s3: &OffloadStorage,
    f: F,
) -> Result<(OffloadStorage, Shape)>
where
    F: FnOnce(&CpuStorage, &CpuStorage, &CpuStorage) -> Result<(CpuStorage, Shape)>,
{
    let t0 = ffi::bus_bytes();
    let a = s1.cpu()?;
    let b = s2.cpu()?;
    let c = s3.cpu()?;
    let (out, shape) = f(&a, &b, &c)?;
    let r = s1.device.storage_from_cpu_storage(&out)?;
    ffi::note_fallback("custom-op(3)", ffi::bus_bytes() - t0);
    Ok((r, shape))
}
