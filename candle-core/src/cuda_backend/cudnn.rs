use crate::conv::CudnnFwdAlgo as CandleAlgo;
use crate::WithDType;
use cudarc::cudnn::safe::{ConvForward, Cudnn};
use cudarc::cudnn::sys::{
    cudnnConvolutionFwdAlgo_t as FwdAlgo, cudnnConvolutionMode_t, cudnnDataType_t, cudnnMathType_t,
    cudnnTensorFormat_t,
};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DeviceRepr};
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

/// Upper bound on cached `(shape, config) -> (algo, workspace_size)` entries per device and
/// thread so dynamic-shape workloads cannot grow host memory without bound.
const MAX_CACHED_ALGO_ENTRIES: usize = 4096;

/// Maximum scratch workspace size (in bytes) accepted from automatic algorithm selection before
/// falling back to a lower-memory algorithm. Explicit user-requested algorithms are not capped.
const MAX_AUTO_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

const AUTO_FALLBACK_ALGOS: [FwdAlgo; 2] = [
    FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConvAlgoKey {
    is_conv1d: bool,
    input_dtype: cudnnDataType_t,
    compute_dtype: cudnnDataType_t,
    x_shape: [i32; 4],
    src_strides: Option<[i32; 4]>,
    w_shape: [i32; 4],
    pad: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
    requested_algo: Option<CandleAlgo>,
}

struct CudnnDeviceCache {
    cudnn: Arc<Cudnn>,
    fwd_algo_cache: HashMap<ConvAlgoKey, (FwdAlgo, usize)>,
    workspace: Option<CudaSlice<u8>>,
}

impl CudnnDeviceCache {
    fn insert_algo(&mut self, key: ConvAlgoKey, value: (FwdAlgo, usize)) {
        if self.fwd_algo_cache.len() >= MAX_CACHED_ALGO_ENTRIES
            && !self.fwd_algo_cache.contains_key(&key)
        {
            self.fwd_algo_cache.clear();
        }
        self.fwd_algo_cache.insert(key, value);
    }
}

// The cuDNN handles and per-device caches are stored per thread rather than on `CudaDevice`
// because `Cudnn` handles are neither `Send` nor `Sync`. Keeping the scratch workspace
// thread-local and keyed by `DeviceId` (which uniquely identifies a `CudaDevice` and its
// immutable `CudaStream`) ensures same-stream FIFO ordering without cross-thread contention.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, CudnnDeviceCache>> =
        RefCell::new(HashMap::new());
}

impl From<cudarc::cudnn::CudnnError> for crate::Error {
    fn from(err: cudarc::cudnn::CudnnError) -> Self {
        crate::Error::wrap(err)
    }
}

impl From<cudarc::driver::DriverError> for crate::Error {
    fn from(err: cudarc::driver::DriverError) -> Self {
        crate::Error::wrap(err)
    }
}

fn candle_algo_to_cudnn(algo: CandleAlgo) -> FwdAlgo {
    match algo {
        CandleAlgo::ImplicitGemm => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CandleAlgo::ImplicitPrecompGemm => {
            FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        CandleAlgo::Gemm => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CandleAlgo::Direct => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CandleAlgo::Fft => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CandleAlgo::FftTiling => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CandleAlgo::Winograd => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CandleAlgo::WinogradNonFused => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        CandleAlgo::Count => FwdAlgo::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    }
}

fn is_stream_capturing(dev: &crate::cuda_backend::CudaDevice) -> bool {
    matches!(
        dev.cuda_stream().capture_status(),
        Ok(status)
            if status
                != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    )
}

fn select_fwd_algo<T: cudarc::cudnn::CudnnDataType, Y: cudarc::cudnn::CudnnDataType>(
    conv_fwd: &ConvForward<'_, T, Y, T>,
    requested_algo: Option<CandleAlgo>,
) -> crate::Result<(FwdAlgo, usize)> {
    match requested_algo {
        Some(algo) => {
            let alg = candle_algo_to_cudnn(algo);
            let workspace_size = conv_fwd.get_workspace_size(alg)?;
            Ok((alg, workspace_size))
        }
        None => {
            let mut alg = conv_fwd.pick_algorithm()?;
            let mut workspace_size = conv_fwd.get_workspace_size(alg)?;
            if workspace_size > MAX_AUTO_WORKSPACE_BYTES {
                for fallback in AUTO_FALLBACK_ALGOS {
                    if workspace_size <= MAX_AUTO_WORKSPACE_BYTES {
                        break;
                    }
                    if alg == fallback {
                        continue;
                    }
                    if let Ok(fallback_ws) = conv_fwd.get_workspace_size(fallback) {
                        if fallback_ws < workspace_size {
                            alg = fallback;
                            workspace_size = fallback_ws;
                        }
                    }
                }
            }
            Ok((alg, workspace_size))
        }
    }
}

/// Allocates an uninitialized scratch workspace of `workspace_size` bytes on `dev` (cuDNN
/// overwrites workspace memory before reading it). On allocation failure for automatically
/// selected algorithms, retries with lower-memory fallback algorithms.
fn alloc_workspace_with_fallback<
    T: cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    dev: &crate::cuda_backend::CudaDevice,
    conv_fwd: &ConvForward<'_, T, Y, T>,
    requested_algo: Option<CandleAlgo>,
    alg: FwdAlgo,
    workspace_size: usize,
) -> crate::Result<(FwdAlgo, usize, Option<CudaSlice<u8>>)> {
    if workspace_size == 0 {
        return Ok((alg, 0, None));
    }
    match unsafe { dev.alloc::<u8>(workspace_size) } {
        Ok(buf) => Ok((alg, workspace_size, Some(buf))),
        Err(alloc_err) => {
            if requested_algo.is_some() {
                return Err(alloc_err);
            }
            for fallback in AUTO_FALLBACK_ALGOS {
                if alg == fallback {
                    continue;
                }
                if let Ok(fallback_ws) = conv_fwd.get_workspace_size(fallback) {
                    if fallback_ws == 0 {
                        return Ok((fallback, 0, None));
                    }
                    if fallback_ws < workspace_size {
                        if let Ok(buf) = unsafe { dev.alloc::<u8>(fallback_ws) } {
                            return Ok((fallback, fallback_ws, Some(buf)));
                        }
                    }
                }
            }
            Err(alloc_err)
        }
    }
}

fn launch_conv_forward<
    T: DeviceRepr + WithDType + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    key: ConvAlgoKey,
    y_shape: [i32; 4],
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let device_id = dev.id();
    let (cudnn, cached_algo) = CUDNN.with(|caches| -> crate::Result<_> {
        let mut caches = caches.borrow_mut();
        let cache = match caches.entry(device_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let cudnn = Cudnn::new(dev.cuda_stream())?;
                entry.insert(CudnnDeviceCache {
                    cudnn,
                    fwd_algo_cache: HashMap::new(),
                    workspace: None,
                })
            }
        };
        Ok((cache.cudnn.clone(), cache.fwd_algo_cache.get(&key).copied()))
    })?;

    let mut conv = cudnn.create_conv2d::<Y>(
        key.pad,
        key.stride,
        key.dilation,
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if matches!(
        T::DATA_TYPE,
        cudnnDataType_t::CUDNN_DATA_HALF | cudnnDataType_t::CUDNN_DATA_BFLOAT16
    ) {
        // Enable Tensor Core math for half-precision convolutions; ignore errors on
        // architectures or cuDNN builds where it is unsupported.
        let _ = conv.set_math_type(cudnnMathType_t::CUDNN_TENSOR_OP_MATH);
    }

    // Note that `src` already starts at the proper offset.
    let x = match key.src_strides {
        None => cudnn.create_4d_tensor::<T>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, key.x_shape)?,
        Some(strides) => cudnn.create_4d_tensor_ex::<T>(key.x_shape, strides)?,
    };
    let w = cudnn.create_4d_filter::<T>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, key.w_shape)?;
    let y = cudnn.create_4d_tensor::<T>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, y_shape)?;
    let conv_fwd = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };

    let (alg, workspace_size) = match cached_algo {
        Some(cached) => cached,
        None => {
            let selected = select_fwd_algo(&conv_fwd, key.requested_algo)?;
            CUDNN.with(|caches| {
                if let Some(cache) = caches.borrow_mut().get_mut(&device_id) {
                    cache.insert_algo(key, selected);
                }
            });
            selected
        }
    };

    if workspace_size == 0 {
        unsafe {
            conv_fwd.launch::<CudaSlice<u8>, _, _, _>(
                alg,
                None,
                (T::one(), T::zero()),
                src,
                filter,
                dst,
            )?;
        }
        return Ok(());
    }

    // During CUDA graph capture, avoid binding or mutating the persistent thread-local workspace:
    // a later non-captured convolution could grow and free that buffer while the captured graph
    // still holds its raw device pointer. Instead, use a temporary stream-ordered allocation that
    // CUDA captures as graph memory nodes.
    if is_stream_capturing(dev) {
        let (alg, workspace_size, mut workspace) =
            alloc_workspace_with_fallback(dev, &conv_fwd, key.requested_algo, alg, workspace_size)?;
        let mut ws_view: Option<CudaViewMut<'_, u8>> = workspace
            .as_mut()
            .map(|buf| buf.slice_mut(0..workspace_size));
        unsafe {
            conv_fwd.launch(
                alg,
                ws_view.as_mut(),
                (T::one(), T::zero()),
                src,
                filter,
                dst,
            )?;
        }
        return Ok(());
    }

    CUDNN.with(|caches| -> crate::Result<()> {
        let mut caches = caches.borrow_mut();
        let cache = caches
            .get_mut(&device_id)
            .expect("cuDNN device cache was initialized above");
        let mut alg = alg;
        let mut workspace_size = workspace_size;

        let needs_alloc = cache
            .workspace
            .as_ref()
            .is_none_or(|ws| ws.len() < workspace_size);
        if needs_alloc {
            // Release the old workspace before allocating the larger buffer so both are not
            // resident simultaneously on the stream.
            cache.workspace = None;
            let (resolved_alg, resolved_ws, buf) = alloc_workspace_with_fallback(
                dev,
                &conv_fwd,
                key.requested_algo,
                alg,
                workspace_size,
            )?;
            if resolved_alg != alg || resolved_ws != workspace_size {
                alg = resolved_alg;
                workspace_size = resolved_ws;
                cache.insert_algo(key, (alg, workspace_size));
            }
            cache.workspace = buf;
        }

        let mut ws_view: Option<CudaViewMut<'_, u8>> = if workspace_size == 0 {
            None
        } else {
            cache
                .workspace
                .as_mut()
                .map(|buf| buf.slice_mut(0..workspace_size))
        };
        unsafe {
            conv_fwd.launch(
                alg,
                ws_view.as_mut(),
                (T::one(), T::zero()),
                src,
                filter,
                dst,
            )?;
        }
        Ok(())
    })
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    let src_strides = if src_l.is_contiguous() {
        None
    } else {
        let s = src_l.stride();
        Some([s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32])
    };
    let w_shape = [
        params.c_out as i32,
        params.c_in as i32,
        params.k_h as i32,
        params.k_w as i32,
    ];
    let y_shape = [
        params.b_size as i32,
        params.c_out as i32,
        params.out_h() as i32,
        params.out_w() as i32,
    ];
    let key = ConvAlgoKey {
        is_conv1d: false,
        input_dtype: T::DATA_TYPE,
        compute_dtype: Y::DATA_TYPE,
        x_shape,
        src_strides,
        w_shape,
        pad: [params.padding as i32, params.padding as i32],
        stride: [params.stride as i32, params.stride as i32],
        dilation: [params.dilation as i32, params.dilation as i32],
        requested_algo: params.cudnn_fwd_algo,
    };
    launch_conv_forward::<T, Y>(src, filter, dst, key, y_shape, dev)
}

pub(crate) fn launch_conv1d<
    T: DeviceRepr + WithDType + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    let src_strides = if src_l.is_contiguous() {
        None
    } else {
        let s = src_l.stride();
        Some([s[0] as i32, s[1] as i32, s[2] as i32, 1])
    };
    let w_shape = [
        params.c_out as i32,
        params.c_in as i32,
        params.k_size as i32,
        1,
    ];
    let y_shape = [
        params.b_size as i32,
        params.c_out as i32,
        params.l_out() as i32,
        1,
    ];
    let key = ConvAlgoKey {
        is_conv1d: true,
        input_dtype: T::DATA_TYPE,
        compute_dtype: Y::DATA_TYPE,
        x_shape,
        src_strides,
        w_shape,
        pad: [params.padding as i32, 0],
        stride: [params.stride as i32, 1],
        dilation: [params.dilation as i32, 1],
        requested_algo: params.cudnn_fwd_algo,
    };
    launch_conv_forward::<T, Y>(src, filter, dst, key, y_shape, dev)
}
