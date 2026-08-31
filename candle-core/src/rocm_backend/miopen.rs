//! Opt-in MIOpen forward convolution (`--features miopen`).
//!
//! This mirrors what `cudnn` is to the CUDA backend: a vendor-library fast path
//! layered over the portable im2col + GEMM one in [`super::ops_conv`], which
//! stays the default and is the fallback whenever MIOpen cannot be used.
//!
//! Two things here are load-bearing.
//!
//! * **1-D convolutions live on the W axis.** They are emulated as a 4-D
//!   convolution with `h = 1`, so `padding`/`stride`/`dilation` belong to the
//!   *width* half of every pair. Passing them as the height half (as this file
//!   used to) leaves `y_desc` describing a different tensor from the one the
//!   convolution produces, and MIOpen then writes nothing at all — every
//!   `conv1d` with padding returned an untouched buffer.
//! * **MIOpen only reads packed tensors.** The descriptors below are built with
//!   `new_4d`, which declares a packed layout, so a strided or offset view has
//!   to be routed to the fallback rather than handed over as if contiguous.
//!
//! [`assert_output_dim`] catches any future recurrence of the first class of bug
//! instead of returning a silently zeroed tensor. It — like every other MIOpen
//! failure — routes the convolution back to [`super::ops_conv`] with a one-time
//! warning rather than erroring: this feature is a *fast path*, and turning a
//! convolution that works without `--features miopen` into a hard error with it
//! is never the right answer.

use std::any::TypeId;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use rocm_rs::miopen::convolution::ConvFwdAlgorithm;
use rocm_rs::miopen::tensor::{DataType, TensorDescriptor};
use rocm_rs::miopen::{ConvolutionDescriptor, Handle};

use super::{
    ops_conv, DeviceId, RocmDevice, RocmStorage, RocmStorageSlice as S, SendSyncDeviceMemory,
};
use crate::backend::BackendStorage;
use crate::{Layout, Result};

/// MIOpen tensor dtype for the Rust type `T`.
///
/// `bf16` has to be probed before `f16` — `half::bf16`'s type name contains
/// both.
pub fn miopen_dtype<T: Copy>() -> Result<DataType> {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        Ok(DataType::MiopenFloat)
    } else if type_name.contains("f64") {
        Ok(DataType::MiopenDouble)
    } else if type_name.contains("bf16") {
        Ok(DataType::MiopenBFloat16)
    } else if type_name.contains("f16") {
        Ok(DataType::MiopenHalf)
    } else {
        Err(super::rocm_error(format!(
            "unsupported dtype for MIOpen: {}",
            type_name
        )))
    }
}

/// A forward convolution problem, in MIOpen's NCHW terms.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Conv2dSpec {
    b: usize,
    c_in: usize,
    c_out: usize,
    i_h: usize,
    i_w: usize,
    k_h: usize,
    k_w: usize,
    out_h: usize,
    out_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
}

/// Algorithm choice and workspace size, cached per (device, dtype, problem).
///
/// `miopenFindConvolutionForwardAlgorithm` benchmarks every candidate solver on
/// the real buffers; running it on every forward pass — as this file used to —
/// costs more than the convolution itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AlgoKey {
    device: DeviceId,
    dtype: TypeId,
    spec: Conv2dSpec,
}

/// The workspace lives in the cache next to the algorithm that needs it.
///
/// Allocating it per call and dropping it right after the *asynchronous*
/// `convolution_forward` enqueue was only safe because `hipFree` implicitly
/// synchronises the whole device — correct by accident, at the price of a full
/// device sync per convolution. Keeping it here also means the alloc happens
/// once per problem shape instead of once per forward pass.
struct CachedAlgo {
    algo: ConvFwdAlgorithm,
    workspace_size: usize,
    workspace: Option<rocm_rs::hip::DeviceMemory<u8>>,
}

// The buffer is only ever reached through the cache mutex, and the HIP
// allocation itself is not thread-affine.
unsafe impl Send for CachedAlgo {}

fn algo_cache() -> &'static Mutex<HashMap<AlgoKey, CachedAlgo>> {
    static CACHE: OnceLock<Mutex<HashMap<AlgoKey, CachedAlgo>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_algo_cache() -> Result<std::sync::MutexGuard<'static, HashMap<AlgoKey, CachedAlgo>>> {
    algo_cache()
        .lock()
        .map_err(|_| super::rocm_error("MIOpen algorithm cache is poisoned".to_string()))
}

fn miopen_err(what: &str, e: impl std::fmt::Display) -> crate::Error {
    super::rocm_error(format!("MIOpen {what} failed: {e}"))
}

/// Rejects a convolution whose declared output disagrees with MIOpen's, which
/// sends it to the `ops_conv` fallback.
fn assert_output_dim(
    conv: &ConvolutionDescriptor,
    x: &TensorDescriptor,
    w: &TensorDescriptor,
    spec: &Conv2dSpec,
) -> Result<()> {
    let got = conv
        .get_forward_output_dim(x, w)
        .map_err(|e| miopen_err("get_forward_output_dim", e))?;
    let want = (
        spec.b as i32,
        spec.c_out as i32,
        spec.out_h as i32,
        spec.out_w as i32,
    );
    if got != want {
        crate::bail!("MIOpen output shape {got:?} does not match {want:?} for {spec:?}")
    }
    Ok(())
}

/// A packed NCHW descriptor for the dtype of `T`.
fn desc4<T: Copy>(n: usize, c: usize, h: usize, w: usize) -> Result<TensorDescriptor> {
    TensorDescriptor::new_4d(miopen_dtype::<T>()?, n as i32, c as i32, h as i32, w as i32)
        .map_err(|e| miopen_err("tensor descriptor creation", e))
}

/// Runs a packed forward convolution into `y_ptr`.
///
/// # Safety
/// The three pointers must address packed buffers matching `spec`.
unsafe fn forward<T: Copy + 'static>(
    dev: &RocmDevice,
    handle: &Handle,
    spec: Conv2dSpec,
    x_ptr: *mut c_void,
    w_ptr: *mut c_void,
    y_ptr: *mut c_void,
) -> Result<()> {
    let x_desc = desc4::<T>(spec.b, spec.c_in, spec.i_h, spec.i_w)?;
    let w_desc = desc4::<T>(spec.c_out, spec.c_in, spec.k_h, spec.k_w)?;
    let y_desc = desc4::<T>(spec.b, spec.c_out, spec.out_h, spec.out_w)?;

    let mut conv_desc =
        ConvolutionDescriptor::new().map_err(|e| miopen_err("conv descriptor creation", e))?;
    conv_desc
        .init_2d(
            0, // miopenConvolution
            spec.pad_h as i32,
            spec.pad_w as i32,
            spec.stride_h as i32,
            spec.stride_w as i32,
            spec.dil_h as i32,
            spec.dil_w as i32,
        )
        .map_err(|e| miopen_err("conv descriptor init_2d", e))?;
    assert_output_dim(&conv_desc, &x_desc, &w_desc, &spec)?;

    let key = AlgoKey {
        device: dev.id(),
        dtype: TypeId::of::<T>(),
        spec,
    };

    // The guard is held across the enqueue so the cached workspace cannot be
    // freed or handed to a second convolution while this one is still queued.
    // `convolution_forward` only enqueues, so this serialises host-side work,
    // not GPU execution.
    let mut cache = lock_algo_cache()?;
    let entry = match cache.entry(key) {
        std::collections::hash_map::Entry::Occupied(hit) => hit.into_mut(),
        std::collections::hash_map::Entry::Vacant(slot) => {
            let workspace_size =
                rocm_rs::miopen::convolution::get_convolution_forward_workspace_size(
                    handle, &w_desc, &x_desc, &conv_desc, &y_desc,
                )
                .map_err(|e| miopen_err("workspace size query", e))?;
            let workspace = alloc_workspace(workspace_size)?;
            let (_, perf) = rocm_rs::miopen::convolution::find_convolution_forward_algorithm(
                handle,
                &x_desc,
                x_ptr,
                &w_desc,
                w_ptr,
                &conv_desc,
                &y_desc,
                y_ptr,
                1,
                workspace_ptr(&workspace),
                workspace_size,
                false,
            )
            .map_err(|e| miopen_err("find_convolution_forward_algorithm", e))?;
            let algo = match perf.first() {
                Some(p) => p.__bindgen_anon_1.fwd_algo,
                None => crate::bail!("MIOpen found no forward algorithm for {spec:?}"),
            };
            slot.insert(CachedAlgo {
                algo,
                workspace_size,
                workspace,
            })
        }
    };

    let alpha: [u8; 4] = 1.0f32.to_le_bytes();
    let beta: [u8; 4] = 0.0f32.to_le_bytes();
    rocm_rs::miopen::convolution::convolution_forward(
        handle,
        &alpha,
        &x_desc,
        x_ptr,
        &w_desc,
        w_ptr,
        &conv_desc,
        entry.algo,
        &beta,
        &y_desc,
        y_ptr,
        workspace_ptr(&entry.workspace),
        entry.workspace_size,
    )
    .map_err(|e| miopen_err("convolution_forward", e))
}

fn alloc_workspace(size: usize) -> Result<Option<rocm_rs::hip::DeviceMemory<u8>>> {
    if size == 0 {
        return Ok(None);
    }
    let mem = rocm_rs::hip::DeviceMemory::<u8>::new(size)
        .map_err(|e| miopen_err("workspace allocation", e))?;
    Ok(Some(mem))
}

fn workspace_ptr(ws: &Option<rocm_rs::hip::DeviceMemory<u8>>) -> *mut c_void {
    ws.as_ref().map_or(std::ptr::null_mut(), |w| w.as_ptr())
}

/// Allocates the output and runs one packed convolution.
///
/// # Safety
/// `x_off`/`w_off` must be in range and both operands packed from there.
unsafe fn run_one<T: Copy + Send + Sync + 'static>(
    dev: &RocmDevice,
    x: &SendSyncDeviceMemory<T>,
    x_off: usize,
    w: &SendSyncDeviceMemory<T>,
    w_off: usize,
    dst_el: usize,
    spec: Conv2dSpec,
) -> Result<SendSyncDeviceMemory<T>> {
    let out = dev.alloc::<T>(dst_el)?;
    forward::<T>(
        dev,
        dev.miopen(),
        spec,
        x.ptr_at(x_off),
        w.ptr_at(w_off),
        out.as_ptr(),
    )?;
    Ok(out)
}

/// Warns once that MIOpen declined a problem and `ops_conv` took over.
fn warn_fallback(e: &crate::Error) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!("candle: MIOpen convolution unavailable, falling back to im2col+GEMM: {e}");
    });
}

/// `None` whenever `ops_conv` has to take over.
///
/// That covers both a dtype pair MIOpen has no solver for — f64 above all, it
/// ships no double-precision convolution — and any problem MIOpen *declines* at
/// runtime. A MIOpen failure is a reason to fall back, not a hard error: the
/// portable path computes the same answer, and `--features miopen` must never
/// turn a working convolution into an error.
fn dispatch(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    dst_el: usize,
    spec: Conv2dSpec,
) -> Option<S> {
    let dev = inp.device();
    let (xo, wo) = (l.start_offset(), kernel_l.start_offset());
    let slice = unsafe {
        match (&inp.slice, &kernel.slice) {
            (S::F32(x), S::F32(w)) => run_one(dev, x, xo, w, wo, dst_el, spec).map(S::F32),
            (S::F16(x), S::F16(w)) => run_one(dev, x, xo, w, wo, dst_el, spec).map(S::F16),
            (S::BF16(x), S::BF16(w)) => run_one(dev, x, xo, w, wo, dst_el, spec).map(S::BF16),
            _ => return None,
        }
    };
    match slice {
        Ok(slice) => Some(slice),
        Err(e) => {
            warn_fallback(&e);
            None
        }
    }
}

/// MIOpen reads packed tensors only, so anything strided goes to the fallback.
fn is_packed(l: &Layout) -> bool {
    l.is_contiguous()
}

pub(super) fn conv1d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv1D,
) -> Result<RocmStorage> {
    if !is_packed(l) || !is_packed(kernel_l) {
        return ops_conv::conv1d(inp, l, kernel, kernel_l, params);
    }
    // h = 1: the length, and therefore padding/stride/dilation, is the W axis.
    let spec = Conv2dSpec {
        b: params.b_size,
        c_in: params.c_in,
        c_out: params.c_out,
        i_h: 1,
        i_w: params.l_in,
        k_h: 1,
        k_w: params.k_size,
        out_h: 1,
        out_w: params.l_out(),
        pad_h: 0,
        pad_w: params.padding,
        stride_h: 1,
        stride_w: params.stride,
        dil_h: 1,
        dil_w: params.dilation,
    };
    let dst_el = params.b_size * params.c_out * params.l_out();
    match dispatch(inp, l, kernel, kernel_l, dst_el, spec) {
        Some(slice) => Ok(RocmStorage {
            slice,
            device: inp.device().clone(),
        }),
        None => ops_conv::conv1d(inp, l, kernel, kernel_l, params),
    }
}

pub(super) fn conv2d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv2D,
) -> Result<RocmStorage> {
    if !is_packed(l) || !is_packed(kernel_l) {
        return ops_conv::conv2d(inp, l, kernel, kernel_l, params);
    }
    let spec = Conv2dSpec {
        b: params.b_size,
        c_in: params.c_in,
        c_out: params.c_out,
        i_h: params.i_h,
        i_w: params.i_w,
        k_h: params.k_h,
        k_w: params.k_w,
        out_h: params.out_h(),
        out_w: params.out_w(),
        pad_h: params.padding,
        pad_w: params.padding,
        stride_h: params.stride,
        stride_w: params.stride,
        dil_h: params.dilation,
        dil_w: params.dilation,
    };
    let dst_el = params.b_size * params.c_out * spec.out_h * spec.out_w;
    match dispatch(inp, l, kernel, kernel_l, dst_el, spec) {
        Some(slice) => Ok(RocmStorage {
            slice,
            device: inp.device().clone(),
        }),
        None => ops_conv::conv2d(inp, l, kernel, kernel_l, params),
    }
}

#[cfg(test)]
#[path = "tests_miopen.rs"]
mod tests;
