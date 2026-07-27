//! Metal backend for the fused block-wise FP8 dequant+GEMM kernel.
//!
//! Mirrors the CUDA scalar path (`fp8_gemm.cu`) on Apple Silicon: the kernel source
//! (`kernels/fp8_gemm.metal`) is compiled at runtime the first time it runs on a given device and
//! the resulting pipeline is cached. Metal has no native fp8 type, so the E4M3 weight bytes are
//! decoded to f32 inside the shader.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle::backend::BackendStorage;
use candle::metal_backend::DeviceId;
use candle::{DType, Layout, MetalStorage, Result, Shape};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;

use crate::Fp8BlockGemm;

const KERNEL_SRC: &str = include_str!("../kernels/fp8_gemm.metal");
const FUNCTION_NAME: &str = "fp8_block_gemm_f32";
const TILE: usize = 16;

/// Scalar arguments, laid out to match `struct Fp8Params` in `fp8_gemm.metal` (5 × i32).
#[repr(C)]
#[derive(Clone, Copy)]
struct Fp8Params {
    m: i32,
    k: i32,
    n: i32,
    block_size: i32,
    scale_cols: i32,
}

/// One compiled pipeline per Metal device, compiled on first use.
fn pipeline_for(device: &candle::MetalDevice) -> Result<ComputePipeline> {
    static CACHE: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let id = device.id();
    {
        let guard = cache.lock().unwrap();
        if let Some(p) = guard.get(&id) {
            return Ok(p.clone());
        }
    }
    let mdev = device.metal_device();
    let lib = mdev
        .new_library_with_source(KERNEL_SRC, None)
        .map_err(candle::Error::wrap)?;
    let func = lib
        .get_function(FUNCTION_NAME, None)
        .map_err(candle::Error::wrap)?;
    let pipeline = mdev
        .new_compute_pipeline_state_with_function(&func)
        .map_err(candle::Error::wrap)?;
    cache.lock().unwrap().insert(id, pipeline.clone());
    Ok(pipeline)
}

fn offset_bytes(layout: &Layout, dtype: DType) -> usize {
    layout.start_offset() * dtype.size_in_bytes()
}

pub(crate) fn fp8_block_gemm_metal_fwd(
    op: &Fp8BlockGemm,
    x: &MetalStorage,
    x_l: &Layout,
    w: &MetalStorage,
    w_l: &Layout,
    scale: &MetalStorage,
    scale_l: &Layout,
) -> Result<(MetalStorage, Shape)> {
    if x.dtype() != DType::F32 {
        candle::bail!(
            "fp8-block-gemm (metal) only supports f32 activations, got {:?}",
            x.dtype()
        );
    }
    if w.dtype() != DType::F8E4M3 {
        candle::bail!(
            "fp8-block-gemm (metal) expects an F8E4M3 weight, got {:?}",
            w.dtype()
        );
    }
    if scale.dtype() != DType::F32 {
        candle::bail!(
            "fp8-block-gemm (metal) expects an f32 scale, got {:?}",
            scale.dtype()
        );
    }

    let (m, k) = x_l.shape().dims2()?;
    let (n, wk) = w_l.shape().dims2()?;
    if wk != k {
        candle::bail!("fp8-block-gemm (metal): weight cols {wk} != x cols {k}");
    }
    let (scale_rows, scale_cols) = scale_l.shape().dims2()?;
    if scale_rows != n.div_ceil(op.block_size) || scale_cols != k.div_ceil(op.block_size) {
        candle::bail!(
            "fp8-block-gemm (metal): scale shape {:?} does not match weight shape {:?} for block_size {}",
            (scale_rows, scale_cols),
            (n, k),
            op.block_size
        );
    }

    let device = x.device().clone();
    let pipeline = pipeline_for(&device)?;
    let output = device.new_buffer(m * n, DType::F32, "fp8-metal-out")?;

    let params = Fp8Params {
        m: m as i32,
        k: k as i32,
        n: n as i32,
        block_size: op.block_size as i32,
        scale_cols: scale_cols as i32,
    };

    let encoder = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_label("fp8-block-gemm-metal");
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(x.buffer()), offset_bytes(x_l, x.dtype()));
    encoder.set_input_buffer(1, Some(w.buffer()), offset_bytes(w_l, w.dtype()));
    encoder.set_input_buffer(
        2,
        Some(scale.buffer()),
        offset_bytes(scale_l, scale.dtype()),
    );
    encoder.set_output_buffer(3, Some(output.as_ref()), 0);
    encoder.set_bytes(4, &params);

    let groups = MTLSize {
        width: n.div_ceil(TILE),
        height: m.div_ceil(TILE),
        depth: 1,
    };
    let threads = MTLSize {
        width: TILE,
        height: TILE,
        depth: 1,
    };
    encoder.dispatch_thread_groups(groups, threads);

    let out = MetalStorage::new(output, device.clone(), m * n, DType::F32);
    Ok((out, Shape::from((m, n))))
}
