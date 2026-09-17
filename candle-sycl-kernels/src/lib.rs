//! Safe-ish Rust wrappers over the `candle-sycl` C ABI (see `csrc/candle_sycl.h`).
//!
//! This crate owns the entire host-side FFI layer for the SYCL backend — there
//! is no `cudarc` equivalent for SYCL, and the feasibility report (§6e) calls
//! for a small in-tree layer rather than a dependency on an unvetted binding
//! crate. `candle-core` consumes only the types in this module.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_int, c_void};
use std::sync::Arc;

mod ffi;
use ffi::*;

/// Dtype tag shared with the C side. Keep discriminants in sync with
/// `CandleSyclDType` in `csrc/candle_sycl.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyclDType {
    U8 = 0,
    U32 = 1,
    I64 = 2,
    F16 = 3,
    BF16 = 4,
    F32 = 5,
    F64 = 6,
}

impl SyclDType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            SyclDType::U8 => 1,
            SyclDType::F16 | SyclDType::BF16 => 2,
            SyclDType::U32 | SyclDType::F32 => 4,
            SyclDType::I64 | SyclDType::F64 => 8,
        }
    }
}

/// Unary op codes — keep in sync with `UnaryOp` in `csrc/elementwise.cpp`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Copy = 0,
    Neg = 1,
    Abs = 2,
    Sqr = 3,
    Sqrt = 4,
    Recip = 5,
    Exp = 6,
    Log = 7,
    Sin = 8,
    Cos = 9,
    Tanh = 10,
    Erf = 11,
    Ceil = 12,
    Floor = 13,
    Round = 14,
    Sign = 15,
    Relu = 16,
    Silu = 17,
    Gelu = 18,
    GeluErf = 19,
    Sigmoid = 20,
}

/// Binary op codes — keep in sync with `BinaryOp` in `csrc/elementwise.cpp`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Maximum = 4,
    Minimum = 5,
}

/// Reduction op codes — keep in sync with `ReduceOp` in `csrc/reduce.cpp`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum = 0,
    Min = 1,
    Max = 2,
    ArgMin = 3,
    ArgMax = 4,
}

/// GGUF quantized dtype, candle source order. Maps to `csrc/quant.cpp` ids.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q4_0 = 3,
    Q4_1 = 4,
    Q5_0 = 5,
    Q5_1 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q2K = 9,
    Q3K = 10,
    Q4K = 11,
    Q5K = 12,
    Q6K = 13,
    Q8K = 14,
}

/// Comparison op codes — keep in sync with `CmpOp` in `csrc/ternary.cpp`.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CmpOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

#[derive(Debug, Clone)]
pub struct SyclError(pub String);
impl std::fmt::Display for SyclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sycl-kernels: {}", self.0)
    }
}
impl std::error::Error for SyclError {}
pub type Result<T> = std::result::Result<T, SyclError>;

fn check(status: c_int, what: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(SyclError(format!("{what} failed (status {status})")))
    }
}

/// Row/strided layout passed to elementwise kernels. `dense()` is the common
/// contiguous, zero-offset case.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Layout {
    num_dims: u32,
    offset: i64,
    dims: [i64; 8],
    strides: [i64; 8],
}

impl Layout {
    pub fn dense() -> Self {
        Self {
            num_dims: 0,
            offset: 0,
            dims: [0; 8],
            strides: [0; 8],
        }
    }

    pub fn strided(dims: &[usize], strides: &[usize], offset: usize) -> Result<Self> {
        if dims.len() > 8 {
            return Err(SyclError(format!("rank {} > 8 not supported", dims.len())));
        }
        let mut d = [0i64; 8];
        let mut s = [0i64; 8];
        for (i, (&di, &si)) in dims.iter().zip(strides).enumerate() {
            d[i] = di as i64;
            s[i] = si as i64;
        }
        Ok(Self {
            num_dims: dims.len().max(1) as u32,
            offset: offset as i64,
            dims: d,
            strides: s,
        })
    }
}

pub struct DeviceInfo {
    pub name: String,
    pub global_mem_bytes: u64,
    pub max_compute_units: u32,
    pub max_clock_khz: u32,
    pub is_integrated: bool,
    pub supports_fp16: bool,
    pub supports_fp64: bool,
}

/// An owned SYCL in-order queue bound to one device, with a size-bucketed USM
/// allocation cache. `malloc_device` for a large buffer costs ~10ms on the
/// Level Zero backend, and ML re-uses the same shapes every step, so freed
/// buffers are parked here and handed straight back.
pub struct Queue {
    raw: *mut CandleSyclQueue,
    pool: std::sync::Mutex<Pool>,
    /// Buffers the pool declined, waiting to be freed behind a synchronize.
    /// See [`Queue::defer_free`].
    pending_free: std::sync::Mutex<Vec<*mut c_void>>,
}

/// Freed device buffers, keyed by exact byte size, plus a running total.
#[derive(Default)]
struct Pool {
    buckets: std::collections::HashMap<usize, Vec<*mut c_void>>,
    bytes: usize,
}
// The underlying `sycl::queue` is safe to share and submit to from multiple
// threads; candle serialises higher up anyway.
unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

const POOL_PER_BUCKET: usize = 16;

/// Upper bound on the bytes parked in the pool, so that the dequantized
/// weights a prefill allocates cannot crowd out the weights and KV cache.
const POOL_MAX_BYTES: usize = 1 << 30;

/// How many declined buffers to accumulate before paying one synchronize to
/// free them all.
const PENDING_FREE_FLUSH: usize = 64;

impl Queue {
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let raw = unsafe { candle_sycl_queue_new(ordinal as c_int) };
        if raw.is_null() {
            return Err(SyclError(format!(
                "no SYCL GPU device at ordinal {ordinal} (have {})",
                Self::device_count()
            )));
        }
        Ok(Arc::new(Self {
            raw,
            pool: std::sync::Mutex::new(Pool::default()),
            pending_free: std::sync::Mutex::new(Vec::new()),
        }))
    }

    fn pool_take(&self, len: usize) -> Option<*mut c_void> {
        let mut pool = self.pool.lock().unwrap();
        let ptr = pool.buckets.get_mut(&len).and_then(Vec::pop);
        if ptr.is_some() {
            pool.bytes = pool.bytes.saturating_sub(len);
        }
        ptr
    }

    fn pool_give(&self, len: usize, ptr: *mut c_void) -> bool {
        let mut pool = self.pool.lock().unwrap();
        if pool.bytes + len > POOL_MAX_BYTES {
            return false;
        }
        let slot = pool.buckets.entry(len).or_default();
        if slot.len() < POOL_PER_BUCKET {
            slot.push(ptr);
            pool.bytes += len;
            true
        } else {
            false
        }
    }

    /// Park `ptr` until it is safe to free.
    ///
    /// `sycl::free` does not wait for work already submitted to the queue, so
    /// freeing a buffer whose kernels are merely enqueued is a use-after-free.
    /// Batch the frees and pay one synchronize per [`PENDING_FREE_FLUSH`].
    fn defer_free(&self, ptr: *mut c_void) {
        let batch = {
            let mut pending = self.pending_free.lock().unwrap();
            pending.push(ptr);
            if pending.len() < PENDING_FREE_FLUSH {
                return;
            }
            std::mem::take(&mut *pending)
        };
        let _ = self.synchronize();
        for p in batch {
            unsafe { candle_sycl_free(self.raw, p) }
        }
    }

    /// Free every cached allocation (e.g. before a large one-off allocation).
    pub fn drain_pool(&self) {
        // Synchronize first, for the reason in [`Queue::defer_free`]: a pooled
        // pointer's `DeviceBuffer` was dropped, which says nothing about whether
        // the kernels reading it have run.
        let _ = self.synchronize();
        let pending = std::mem::take(&mut *self.pending_free.lock().unwrap());
        let mut pool = self.pool.lock().unwrap();
        for (_, ptrs) in pool.buckets.drain() {
            for p in ptrs {
                unsafe { candle_sycl_free(self.raw, p) }
            }
        }
        pool.bytes = 0;
        drop(pool);
        for p in pending {
            unsafe { candle_sycl_free(self.raw, p) }
        }
    }

    pub fn device_count() -> usize {
        unsafe { candle_sycl_device_count() as usize }
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { candle_sycl_synchronize(self.raw) }, "synchronize")
    }

    /// The underlying `sycl::queue *` as an opaque pointer. For out-of-tree
    /// kernels (linked as their own `.so`, e.g. crane's fused GDN launcher)
    /// that must submit onto candle's in-order queue; cast back to
    /// `sycl::queue *` on the C++ side. Valid while this `Queue` (or an `Arc`
    /// clone of it) is alive.
    pub fn native_ptr(&self) -> *mut c_void {
        unsafe { candle_sycl_queue_native(self.raw) }
    }

    pub fn device_info(&self) -> Result<DeviceInfo> {
        let mut raw = CandleSyclDeviceInfo {
            name: [0; 256],
            global_mem_bytes: 0,
            max_compute_units: 0,
            max_clock_khz: 0,
            is_integrated: 0,
            supports_fp16: 0,
            supports_fp64: 0,
        };
        check(
            unsafe { candle_sycl_device_info(self.raw, &mut raw) },
            "device_info",
        )?;
        let name = {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(raw.name.as_ptr() as *const u8, raw.name.len())
            };
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            String::from_utf8_lossy(&bytes[..end]).into_owned()
        };
        Ok(DeviceInfo {
            name,
            global_mem_bytes: raw.global_mem_bytes,
            max_compute_units: raw.max_compute_units,
            max_clock_khz: raw.max_clock_khz,
            is_integrated: raw.is_integrated != 0,
            supports_fp16: raw.supports_fp16 != 0,
            supports_fp64: raw.supports_fp64 != 0,
        })
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        for p in std::mem::take(self.pending_free.get_mut().unwrap()) {
            unsafe { candle_sycl_free(self.raw, p) }
        }
        for (_, ptrs) in self.pool.get_mut().unwrap().buckets.drain() {
            for p in ptrs {
                unsafe { candle_sycl_free(self.raw, p) }
            }
        }
        unsafe { candle_sycl_queue_free(self.raw) }
    }
}

impl std::fmt::Debug for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sycl::Queue({:p})", self.raw)
    }
}

/// An owned USM device allocation.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    len_bytes: usize,
    queue: Arc<Queue>,
    /// `false` for a [`DeviceBuffer::view`] alias: dropping it must neither
    /// free nor pool the pointer, because another `DeviceBuffer` owns it.
    owned: bool,
}
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn alloc(queue: &Arc<Queue>, len_bytes: usize) -> Result<Self> {
        let ptr = match queue.pool_take(len_bytes) {
            Some(p) => p,
            None => {
                let mut p = unsafe { candle_sycl_malloc(queue.raw, len_bytes) };
                if p.is_null() {
                    // The pool buckets by exact size and never shrinks on its
                    // own, so it can hold buffers nothing will ask for again
                    // while a fresh allocation fails. Hand them back and retry.
                    queue.drain_pool();
                    p = unsafe { candle_sycl_malloc(queue.raw, len_bytes) };
                }
                if p.is_null() {
                    return Err(SyclError(format!(
                        "malloc_device({len_bytes}) returned null (device out of memory, \
                         and the allocation pool was already drained)"
                    )));
                }
                p
            }
        };
        Ok(Self {
            ptr,
            len_bytes,
            queue: queue.clone(),
            owned: true,
        })
    }

    /// A non-owning alias of this allocation, so a buffer can be handed to
    /// something that wants a `DeviceBuffer` without copying it. Dropping the
    /// view is a no-op.
    ///
    /// # Safety
    ///
    /// The caller must keep `self` alive for as long as the view is used, and
    /// must not let the view outlive it.
    pub unsafe fn view(&self) -> Self {
        Self {
            ptr: self.ptr,
            len_bytes: self.len_bytes,
            queue: self.queue.clone(),
            owned: false,
        }
    }

    /// A non-owning alias starting `byte_off` into this allocation, for an
    /// operand that is contiguous but does not start at element 0. Dropping the
    /// view is a no-op.
    ///
    /// # Safety
    ///
    /// As [`DeviceBuffer::view`], and `byte_off` must be within the buffer.
    pub unsafe fn view_at(&self, byte_off: usize) -> Self {
        debug_assert!(byte_off <= self.len_bytes);
        let byte_off = byte_off.min(self.len_bytes);
        Self {
            ptr: unsafe { (self.ptr as *mut u8).add(byte_off) as *mut c_void },
            len_bytes: self.len_bytes - byte_off,
            queue: self.queue.clone(),
            owned: false,
        }
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        assert!(src.len() <= self.len_bytes);
        check(
            unsafe {
                candle_sycl_memcpy_htod(
                    self.queue.raw,
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    src.len(),
                )
            },
            "memcpy_htod",
        )
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        assert!(dst.len() <= self.len_bytes);
        check(
            unsafe {
                candle_sycl_memcpy_dtoh(
                    self.queue.raw,
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    dst.len(),
                )
            },
            "memcpy_dtoh",
        )
    }

    pub fn copy_from_device(&self, src: &DeviceBuffer, bytes: usize) -> Result<()> {
        check(
            unsafe { candle_sycl_memcpy_dtod(self.queue.raw, self.ptr, src.ptr, bytes) },
            "memcpy_dtod",
        )
    }

    pub fn memset_zero(&self) -> Result<()> {
        check(
            unsafe { candle_sycl_memset(self.queue.raw, self.ptr, 0, self.len_bytes) },
            "memset",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        // Park the allocation in the queue's cache; if the cache declines it,
        // hand it to `defer_free` rather than freeing here — see that method.
        if !self.queue.pool_give(self.len_bytes, self.ptr) {
            self.queue.defer_free(self.ptr);
        }
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sycl::DeviceBuffer({} bytes @ {:p})",
            self.len_bytes, self.ptr
        )
    }
}

// ---- kernel entry points --------------------------------------------------

pub fn fill(q: &Queue, dt: SyclDType, dst: &DeviceBuffer, numel: usize, value: f64) -> Result<()> {
    check(
        unsafe { candle_sycl_fill(q.raw, dt as u32, dst.ptr, numel, value) },
        "fill",
    )
}

pub fn fill_strided(
    q: &Queue,
    dt: SyclDType,
    lin: &Layout,
    dst: &DeviceBuffer,
    numel: usize,
    value: f64,
) -> Result<()> {
    check(
        unsafe { candle_sycl_fill_strided(q.raw, dt as u32, lin, dst.ptr, numel, value) },
        "fill_strided",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn affine(
    q: &Queue,
    dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
    mul: f64,
    add: f64,
) -> Result<()> {
    check(
        unsafe { candle_sycl_affine(q.raw, dt as u32, lin, inp.ptr, out.ptr, numel, mul, add) },
        "affine",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn elu(
    q: &Queue,
    dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
    alpha: f64,
) -> Result<()> {
    check(
        unsafe { candle_sycl_elu(q.raw, dt as u32, lin, inp.ptr, out.ptr, numel, alpha) },
        "elu",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn powf(
    q: &Queue,
    dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
    exponent: f64,
) -> Result<()> {
    check(
        unsafe { candle_sycl_powf(q.raw, dt as u32, lin, inp.ptr, out.ptr, numel, exponent) },
        "powf",
    )
}

pub fn unary(
    q: &Queue,
    op: UnaryOp,
    dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
) -> Result<()> {
    check(
        unsafe { candle_sycl_unary(q.raw, op as u32, dt as u32, lin, inp.ptr, out.ptr, numel) },
        "unary",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary(
    q: &Queue,
    op: BinaryOp,
    dt: SyclDType,
    lhs_l: &Layout,
    lhs: &DeviceBuffer,
    rhs_l: &Layout,
    rhs: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_binary(
                q.raw, op as u32, dt as u32, lhs_l, lhs.ptr, rhs_l, rhs.ptr, out.ptr, numel,
            )
        },
        "binary",
    )
}

pub fn cast(
    q: &Queue,
    src_dt: SyclDType,
    dst_dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_cast(
                q.raw,
                src_dt as u32,
                dst_dt as u32,
                lin,
                inp.ptr,
                out.ptr,
                numel,
            )
        },
        "cast",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn copy_strided(
    q: &Queue,
    dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    dst_offset: usize,
    numel: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_copy_strided(q.raw, dt as u32, lin, inp.ptr, out.ptr, dst_offset, numel)
        },
        "copy_strided",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn copy2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    d1: usize,
    d2: usize,
    src_stride1: usize,
    dst_stride1: usize,
    src_offset: usize,
    dst_offset: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_copy2d(
                q.raw,
                dt as u32,
                inp.ptr,
                out.ptr,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            )
        },
        "copy2d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn reduce(
    q: &Queue,
    op: ReduceOp,
    dt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    out_el: usize,
    reduce_el: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_reduce(
                q.raw, op as u32, dt as u32, lin, inp.ptr, out.ptr, out_el, reduce_el,
            )
        },
        "reduce",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn cmp(
    q: &Queue,
    op: CmpOp,
    dt: SyclDType,
    lhs_l: &Layout,
    lhs: &DeviceBuffer,
    rhs_l: &Layout,
    rhs: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_cmp(
                q.raw, op as u32, dt as u32, lhs_l, lhs.ptr, rhs_l, rhs.ptr, out.ptr, numel,
            )
        },
        "cmp",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn where_cond(
    q: &Queue,
    cond_dt: SyclDType,
    val_dt: SyclDType,
    cond_l: &Layout,
    cond: &DeviceBuffer,
    t_l: &Layout,
    t_vals: &DeviceBuffer,
    f_l: &Layout,
    f_vals: &DeviceBuffer,
    out: &DeviceBuffer,
    numel: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_where(
                q.raw,
                cond_dt as u32,
                val_dt as u32,
                cond_l,
                cond.ptr,
                t_l,
                t_vals.ptr,
                f_l,
                f_vals.ptr,
                out.ptr,
                numel,
            )
        },
        "where",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn index_select(
    q: &Queue,
    dt: SyclDType,
    idt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    ids: &DeviceBuffer,
    out: &DeviceBuffer,
    left: usize,
    src_dim: usize,
    ids_dim: usize,
    right: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_index_select(
                q.raw, dt as u32, idt as u32, lin, inp.ptr, ids.ptr, out.ptr, left, src_dim,
                ids_dim, right,
            )
        },
        "index_select",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gather(
    q: &Queue,
    dt: SyclDType,
    idt: SyclDType,
    lin: &Layout,
    inp: &DeviceBuffer,
    ids: &DeviceBuffer,
    out: &DeviceBuffer,
    left: usize,
    src_dim: usize,
    ids_dim: usize,
    right: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_gather(
                q.raw, dt as u32, idt as u32, lin, inp.ptr, ids.ptr, out.ptr, left, src_dim,
                ids_dim, right,
            )
        },
        "gather",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn scatter(
    q: &Queue,
    add: bool,
    dt: SyclDType,
    idt: SyclDType,
    out: &DeviceBuffer,
    ids: &DeviceBuffer,
    src: &DeviceBuffer,
    left: usize,
    src_dim: usize,
    dst_dim: usize,
    right: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_scatter(
                q.raw,
                add as c_int,
                dt as u32,
                idt as u32,
                out.ptr,
                ids.ptr,
                src.ptr,
                left,
                src_dim,
                dst_dim,
                right,
            )
        },
        "scatter",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn index_add(
    q: &Queue,
    dt: SyclDType,
    idt: SyclDType,
    out: &DeviceBuffer,
    ids: &DeviceBuffer,
    src: &DeviceBuffer,
    left: usize,
    ids_dim: usize,
    dst_dim: usize,
    right: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_index_add(
                q.raw, dt as u32, idt as u32, out.ptr, ids.ptr, src.ptr, left, ids_dim, dst_dim,
                right,
            )
        },
        "index_add",
    )
}

pub fn argsort(
    q: &Queue,
    dt: SyclDType,
    ascending: bool,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_argsort(
                q.raw,
                dt as u32,
                ascending as c_int,
                inp.ptr,
                out.ptr as *mut u32,
                nrows,
                ncols,
            )
        },
        "argsort",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    src: &[i64; 9],
    k: (usize, usize),
    stride: (usize, usize),
    h_out: usize,
    w_out: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_avg_pool2d(
                q.raw,
                dt as u32,
                inp.ptr,
                out.ptr,
                src.as_ptr(),
                k.0,
                k.1,
                stride.0,
                stride.1,
                h_out,
                w_out,
            )
        },
        "avg_pool2d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    src: &[i64; 9],
    k: (usize, usize),
    stride: (usize, usize),
    h_out: usize,
    w_out: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_max_pool2d(
                q.raw,
                dt as u32,
                inp.ptr,
                out.ptr,
                src.as_ptr(),
                k.0,
                k.1,
                stride.0,
                stride.1,
                h_out,
                w_out,
            )
        },
        "max_pool2d",
    )
}

pub fn upsample_nearest2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    src: &[i64; 9],
    dst_h: usize,
    dst_w: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_upsample_nearest2d(
                q.raw,
                dt as u32,
                inp.ptr,
                out.ptr,
                src.as_ptr(),
                dst_h,
                dst_w,
            )
        },
        "upsample_nearest2d",
    )
}

pub fn upsample_nearest1d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    src: &[i64; 7],
    dst_w: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_upsample_nearest1d(q.raw, dt as u32, inp.ptr, out.ptr, src.as_ptr(), dst_w)
        },
        "upsample_nearest1d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn upsample_bilinear2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    src: &[i64; 9],
    dst_h: usize,
    dst_w: usize,
    align_corners: bool,
    scale_h: f64,
    scale_w: f64,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_upsample_bilinear2d(
                q.raw,
                dt as u32,
                inp.ptr,
                out.ptr,
                src.as_ptr(),
                dst_h,
                dst_w,
                align_corners as c_int,
                scale_h,
                scale_w,
            )
        },
        "upsample_bilinear2d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn im2col2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    col: &DeviceBuffer,
    meta: &[i64; 9],
    k: (usize, usize),
    stride: usize,
    padding: usize,
    dilation: usize,
    out: (usize, usize),
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_im2col2d(
                q.raw,
                dt as u32,
                inp.ptr,
                col.ptr,
                meta.as_ptr(),
                k.0,
                k.1,
                stride,
                padding,
                dilation,
                out.0,
                out.1,
            )
        },
        "im2col2d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn im2col1d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    col: &DeviceBuffer,
    meta: &[i64; 7],
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_l: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_im2col1d(
                q.raw,
                dt as u32,
                inp.ptr,
                col.ptr,
                meta.as_ptr(),
                k,
                stride,
                padding,
                dilation,
                out_l,
            )
        },
        "im2col1d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    im: &[i64; 5],
    ker: &DeviceBuffer,
    out: &DeviceBuffer,
    dims: (usize, usize, usize),
    ihw: (usize, usize),
    khw: (usize, usize),
    ohw: (usize, usize),
    conv: (usize, usize, usize),
) -> Result<()> {
    let (b, c_in, c_out) = dims;
    check(
        unsafe {
            candle_sycl_conv_transpose2d(
                q.raw,
                dt as u32,
                inp.ptr,
                im.as_ptr(),
                ker.ptr,
                out.ptr,
                b,
                c_in,
                c_out,
                ihw.0,
                ihw.1,
                khw.0,
                khw.1,
                ohw.0,
                ohw.1,
                conv.0,
                conv.1,
                conv.2,
            )
        },
        "conv_transpose2d",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    im: &[i64; 4],
    ker: &DeviceBuffer,
    out: &DeviceBuffer,
    dims: (usize, usize, usize),
    il: usize,
    kl: usize,
    out_l: usize,
    conv: (usize, usize, usize),
) -> Result<()> {
    let (b, c_in, c_out) = dims;
    check(
        unsafe {
            candle_sycl_conv_transpose1d(
                q.raw,
                dt as u32,
                inp.ptr,
                im.as_ptr(),
                ker.ptr,
                out.ptr,
                b,
                c_in,
                c_out,
                il,
                kl,
                out_l,
                conv.0,
                conv.1,
                conv.2,
            )
        },
        "conv_transpose1d",
    )
}

/// Fused quantized mat-vec (QMatMul decode path). `w` is the GGUF weight
/// (`n` rows of `k/block_size` blocks); `act` is a dense f32 `(m, k)` with
/// `m <= 8`; `out` is dense f32 `(m, n)`.
#[allow(clippy::too_many_arguments)]
pub fn mmvq(
    q: &Arc<Queue>,
    dt: GgmlDType,
    w: &DeviceBuffer,
    act: &DeviceBuffer,
    out: &DeviceBuffer,
    n: usize,
    k: usize,
    m: usize,
) -> Result<()> {
    let blk = match dt {
        GgmlDType::Q2K
        | GgmlDType::Q3K
        | GgmlDType::Q4K
        | GgmlDType::Q5K
        | GgmlDType::Q6K
        | GgmlDType::Q8K => 256,
        _ => 32,
    };
    let (tmp, ch) = mmvq_scratch(q, n, m, k / blk)?;
    check(
        unsafe {
            candle_sycl_mmvq(
                q.raw,
                dt as u32,
                w.ptr,
                act.ptr as *const f32,
                out.ptr as *mut f32,
                n,
                k,
                m,
                tmp.ptr as *mut f32,
                ch,
            )
        },
        "mmvq",
    )
}

/// Per-(row, chunk) partial-sum scratch for the mat-vec kernels. One weight
/// block per work-item is fastest, so the chunk size `ch` only grows enough to
/// keep the scratch bounded on very large `n * m * nblk` (a vocab head).
fn mmvq_scratch(q: &Arc<Queue>, n: usize, m: usize, nblk: usize) -> Result<(DeviceBuffer, usize)> {
    const MAX_ITEMS: usize = 1 << 24;
    let ch = (n * m * nblk).div_ceil(MAX_ITEMS).max(1);
    let tmp = DeviceBuffer::alloc(q, n * m * nblk.div_ceil(ch) * 4)?;
    Ok((tmp, ch))
}

/// Integer quantized mat-vec: quantizes `act` to int8 blocks on the device, then
/// integer-dots each weight row (the CPU `vec_dot` numerics). `act` and `out`
/// are dense f32, or f16 when `act_f16` / `out_f16`. Returns `Ok(false)` without
/// launching anything when `dt` has no integer kernel; use [`mmvq`] then.
#[allow(clippy::too_many_arguments)]
pub fn mmvq_q8(
    q: &Arc<Queue>,
    dt: GgmlDType,
    w: &DeviceBuffer,
    act: &DeviceBuffer,
    act_f16: bool,
    out: &DeviceBuffer,
    out_f16: bool,
    n: usize,
    k: usize,
    m: usize,
) -> Result<bool> {
    let blk = match mmvq_q8_block(dt) {
        Some(b) if k.is_multiple_of(b) => b,
        _ => return Ok(false),
    };
    let nblk = m * (k / blk);
    let q8 = DeviceBuffer::alloc(q, m * k)?;
    let d8 = DeviceBuffer::alloc(q, nblk * 4)?;
    let s32 = DeviceBuffer::alloc(q, nblk * 8 * 4)?;
    let (tmp, ch) = mmvq_scratch(q, n, m, k / blk)?;
    check(
        unsafe {
            candle_sycl_mmvq_q8(
                q.raw,
                dt as u32,
                w.ptr,
                act.ptr,
                c_int::from(act_f16),
                out.ptr,
                c_int::from(out_f16),
                n,
                k,
                m,
                q8.ptr as *mut i8,
                d8.ptr as *mut f32,
                s32.ptr as *mut i32,
                tmp.ptr as *mut f32,
                ch,
            )
        },
        "mmvq_q8",
    )?;
    Ok(true)
}

/// Activation block size of the integer mat-vec kernel for `dt`, or `None` if
/// `dt` has no integer kernel.
pub fn mmvq_q8_block(dt: GgmlDType) -> Option<usize> {
    match dt {
        GgmlDType::Q4_0 | GgmlDType::Q8_0 => Some(32),
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => Some(256),
        _ => None,
    }
}

/// Gather `n_ids` rows (each `row_blocks` blocks) of a quantized matrix by the
/// `u32` indices in `ids`, dequantized into `dst` (`n_ids * row_blocks *
/// block_size(dt)` f32). Unquantized `dt`s (F32/F16/BF16) are not supported.
pub fn get_rows(
    q: &Queue,
    dt: GgmlDType,
    src: &DeviceBuffer,
    ids: &DeviceBuffer,
    dst: &DeviceBuffer,
    n_ids: usize,
    row_blocks: usize,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_get_rows(
                q.raw,
                dt as u32,
                src.ptr,
                ids.ptr as *const u32,
                dst.ptr,
                n_ids,
                row_blocks,
            )
        },
        "get_rows",
    )
}

/// Dequantise `n_blocks` GGUF blocks into `dst` (`n_blocks * block_size(dt)` f32).
pub fn dequantize(
    q: &Queue,
    dt: GgmlDType,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    n_blocks: usize,
) -> Result<()> {
    check(
        unsafe { candle_sycl_dequantize(q.raw, dt as u32, src.ptr, dst.ptr, n_blocks) },
        "dequantize",
    )
}

/// As [`dequantize`], but writes f16 directly, saving the f32 staging buffer
/// and the extra pass a cast would need.
pub fn dequantize_f16(
    q: &Queue,
    dt: GgmlDType,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    n_blocks: usize,
) -> Result<()> {
    check(
        unsafe { candle_sycl_dequantize_f16(q.raw, dt as u32, src.ptr, dst.ptr, n_blocks) },
        "dequantize_f16",
    )
}

/// Row-wise softmax over the last dim; `inp` is contiguous `(rows, d)`.
pub fn softmax_lastdim(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    out: &DeviceBuffer,
    rows: usize,
    d: usize,
) -> Result<()> {
    check(
        unsafe { candle_sycl_softmax_lastdim(q.raw, dt as u32, inp.ptr, out.ptr, rows, d) },
        "softmax_lastdim",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm(
    q: &Queue,
    dt: SyclDType,
    inp: &DeviceBuffer,
    alpha: &DeviceBuffer,
    out: &DeviceBuffer,
    rows: usize,
    d: usize,
    eps: f32,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_rms_norm(q.raw, dt as u32, inp.ptr, alpha.ptr, out.ptr, rows, d, eps)
        },
        "rms_norm",
    )
}

/// RoPE. `mode`: 0 interleaved (b,h,t,d), 1 half-split (b,h,t,d), 2 half-split (b,t,h,d).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    q: &Queue,
    mode: u32,
    dt: SyclDType,
    inp: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    out: &DeviceBuffer,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    cos_batched: bool,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_rope(
                q.raw,
                mode,
                dt as u32,
                inp.ptr,
                cos.ptr,
                sin.ptr,
                out.ptr,
                b,
                h,
                t,
                d,
                cos_batched as c_int,
            )
        },
        "rope",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gemm(
    q: &Queue,
    dt: SyclDType,
    transa: bool,
    transb: bool,
    m: i64,
    n: i64,
    k: i64,
    alpha: f64,
    beta: f64,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    batch: i64,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    off_a: i64,
    off_b: i64,
) -> Result<()> {
    check(
        unsafe {
            candle_sycl_gemm(
                q.raw,
                dt as u32,
                transa as c_int,
                transb as c_int,
                m,
                n,
                k,
                alpha,
                beta,
                a.ptr,
                b.ptr,
                c.ptr,
                batch,
                stride_a,
                stride_b,
                stride_c,
                off_a,
                off_b,
            )
        },
        "gemm",
    )
}
