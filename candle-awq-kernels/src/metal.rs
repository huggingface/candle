//! Metal backend for the fused AWQ dequant+GEMM kernel.
//!
//! Mirrors the CUDA scalar path (`awq_gemm.cu`) on Apple Silicon: the kernel source
//! (`kernels/awq_gemm.metal`) is compiled at runtime the first time it runs on a given device and
//! the resulting pipeline is cached, then dispatched as a threadgroup-tiled GEMM that dequantizes
//! each weight (including AWQ's output-axis nibble permutation) on the fly.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle::backend::BackendStorage;
use candle::metal_backend::DeviceId;
use candle::{DType, Layout, MetalStorage, Result, Shape, Storage};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;

use crate::{AwqGemm, AWQ_PACK_FACTOR};

const KERNEL_SRC: &str = include_str!("../kernels/awq_gemm.metal");
const FUNCTION_NAME: &str = "awq_gemm_f32";
const TILE: usize = 16;

/// Scalar arguments, laid out to match `struct AwqParams` in `awq_gemm.metal` (5 × i32).
#[repr(C)]
#[derive(Clone, Copy)]
struct AwqParams {
    m: i32,
    k: i32,
    n: i32,
    group_size: i32,
    n_packed_out: i32,
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn awq_gemm_metal_fwd(
    op: &AwqGemm,
    x: &MetalStorage,
    x_l: &Layout,
    qweight: &MetalStorage,
    qweight_l: &Layout,
    qzeros: &MetalStorage,
    qzeros_l: &Layout,
) -> Result<(MetalStorage, Shape)> {
    if x.dtype() != DType::F32 {
        candle::bail!(
            "awq-gemm (metal) only supports f32 activations, got {:?}",
            x.dtype()
        );
    }
    if qweight.dtype() != DType::I32 || qzeros.dtype() != DType::I32 {
        candle::bail!(
            "awq-gemm (metal): qweight/qzeros must be i32, got {:?}/{:?}",
            qweight.dtype(),
            qzeros.dtype()
        );
    }
    let (m, k) = x_l.shape().dims2()?;
    let (qk, n_packed) = qweight_l.shape().dims2()?;
    if qk != k {
        candle::bail!("awq-gemm (metal): qweight rows {qk} != x cols {k}");
    }
    let n = n_packed * AWQ_PACK_FACTOR;
    let (_n_groups, qz_packed) = qzeros_l.shape().dims2()?;
    if qz_packed != n_packed {
        candle::bail!("awq-gemm (metal): qzeros cols {qz_packed} != qweight cols {n_packed}");
    }
    if op.scales.dtype() != DType::F32 {
        candle::bail!(
            "awq-gemm (metal): scales must be f32, got {:?}",
            op.scales.dtype()
        );
    }

    let device = x.device().clone();

    let (scales_storage, scales_l) = op.scales.storage_and_layout();
    let scales_metal = match &*scales_storage {
        Storage::Metal(s) => s,
        _ => candle::bail!("awq-gemm (metal): scales must be a metal tensor"),
    };

    let pipeline = pipeline_for(&device)?;
    let output = device.new_buffer(m * n, DType::F32, "awq-metal-out")?;

    let params = AwqParams {
        m: m as i32,
        k: k as i32,
        n: n as i32,
        group_size: op.group_size as i32,
        n_packed_out: n_packed as i32,
    };

    let encoder = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_label("awq-gemm-metal");
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(x.buffer()), offset_bytes(x_l, x.dtype()));
    encoder.set_input_buffer(
        1,
        Some(qweight.buffer()),
        offset_bytes(qweight_l, qweight.dtype()),
    );
    encoder.set_input_buffer(
        2,
        Some(qzeros.buffer()),
        offset_bytes(qzeros_l, qzeros.dtype()),
    );
    encoder.set_input_buffer(
        3,
        Some(scales_metal.buffer()),
        offset_bytes(scales_l, scales_metal.dtype()),
    );
    encoder.set_output_buffer(4, Some(output.as_ref()), 0);
    encoder.set_bytes(5, &params);

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
