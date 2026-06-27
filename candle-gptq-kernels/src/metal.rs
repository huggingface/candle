//! Metal backend for the fused GPTQ dequant+GEMM kernel.
//!
//! Mirrors the CUDA scalar path (`gptq_gemm.cu`) on Apple Silicon: the kernel source
//! (`kernels/gptq_gemm.metal`) is compiled at runtime the first time it runs on a given device and
//! the resulting pipeline is cached, then dispatched as a threadgroup-tiled GEMM that dequantizes
//! each weight on the fly. There is no Metal equivalent of the tensor-core / Marlin paths, so
//! `gptq_gemm` is the only entry point routed here.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle::backend::BackendStorage;
use candle::metal_backend::DeviceId;
use candle::{DType, Layout, MetalStorage, Result, Shape, Storage};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;

use crate::GptqGemm;

const KERNEL_SRC: &str = include_str!("../kernels/gptq_gemm.metal");
const FUNCTION_NAME: &str = "gptq_gemm_f32";
const TILE: usize = 16;

/// Scalar arguments, laid out to match `struct GptqParams` in `gptq_gemm.metal` (6 × i32).
#[repr(C)]
#[derive(Clone, Copy)]
struct GptqParams {
    m: i32,
    k: i32,
    n: i32,
    bits: i32,
    pack_factor: i32,
    n_groups_out: i32,
}

/// One compiled pipeline per Metal device, compiled on first use. `ComputePipeline` is `Send`/`Sync`
/// in `candle-metal-kernels`, so a process-wide cache is sound.
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

/// Byte offset of a contiguous tensor's data within its backing buffer.
fn offset_bytes(layout: &Layout, dtype: DType) -> usize {
    layout.start_offset() * dtype.size_in_bytes()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gptq_gemm_metal_fwd(
    op: &GptqGemm,
    x: &MetalStorage,
    x_l: &Layout,
    qweight: &MetalStorage,
    qweight_l: &Layout,
    qzeros: &MetalStorage,
    qzeros_l: &Layout,
) -> Result<(MetalStorage, Shape)> {
    if x.dtype() != DType::F32 {
        candle::bail!(
            "gptq-gemm (metal) only supports f32 activations, got {:?}",
            x.dtype()
        );
    }
    if qweight.dtype() != DType::I32 || qzeros.dtype() != DType::I32 {
        candle::bail!(
            "gptq-gemm (metal): qweight/qzeros must be i32, got {:?}/{:?}",
            qweight.dtype(),
            qzeros.dtype()
        );
    }
    let (m, k) = x_l.shape().dims2()?;
    let (packed_k, n) = qweight_l.shape().dims2()?;
    if packed_k * op.pack_factor != k {
        candle::bail!(
            "gptq-gemm (metal): qweight rows {packed_k} * pack_factor {} != x cols {k}",
            op.pack_factor
        );
    }
    let (_n_groups, n_packed) = qzeros_l.shape().dims2()?;
    if n_packed * op.pack_factor != n {
        candle::bail!(
            "gptq-gemm (metal): qzeros cols {n_packed} * pack_factor {} != out dim {n}",
            op.pack_factor
        );
    }

    let device = x.device().clone();

    // Extra ride-along tensors (scales: f32, g_idx: i32).
    let (scales_storage, scales_l) = op.scales.storage_and_layout();
    let scales_metal = match &*scales_storage {
        Storage::Metal(s) => s,
        _ => candle::bail!("gptq-gemm (metal): scales must be a metal tensor"),
    };
    let (g_idx_storage, g_idx_l) = op.g_idx.storage_and_layout();
    let g_idx_metal = match &*g_idx_storage {
        Storage::Metal(s) => s,
        _ => candle::bail!("gptq-gemm (metal): g_idx must be a metal tensor"),
    };

    let pipeline = pipeline_for(&device)?;
    let output = device.new_buffer(m * n, DType::F32, "gptq-metal-out")?;

    let params = GptqParams {
        m: m as i32,
        k: k as i32,
        n: n as i32,
        bits: op.bits as i32,
        pack_factor: op.pack_factor as i32,
        n_groups_out: n_packed as i32,
    };

    let encoder = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_label("gptq-gemm-metal");
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
    encoder.set_input_buffer(
        4,
        Some(g_idx_metal.buffer()),
        offset_bytes(g_idx_l, g_idx_metal.dtype()),
    );
    encoder.set_output_buffer(5, Some(output.as_ref()), 0);
    encoder.set_bytes(6, &params);

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
