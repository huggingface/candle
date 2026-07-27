//! Metal backend for the single-token decode attention kernel.
//!
//! Mirrors the CUDA reference path (`decode_attention.cu`) on Apple Silicon: the kernel source
//! (`kernels/decode_attention.metal`) is compiled at runtime the first time it runs on a given
//! device and the resulting pipeline is cached per (device, function).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle::backend::BackendStorage;
use candle::metal_backend::DeviceId;
use candle::{DType, Layout, MetalStorage, Result, Shape};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;

use crate::DecodeAttention;

const KERNEL_SRC: &str = include_str!("../kernels/decode_attention.metal");

/// Scalar arguments, laid out to match `struct DecodeParams` in `decode_attention.metal`
/// (13 × i32 then 1 × f32).
#[repr(C)]
#[derive(Clone, Copy)]
struct DecodeParams {
    hkv_group: i32,
    seqlen_k: i32,
    head_dim: i32,
    q_b_stride: i32,
    q_h_stride: i32,
    k_b_stride: i32,
    k_h_stride: i32,
    k_l_stride: i32,
    v_b_stride: i32,
    v_h_stride: i32,
    v_l_stride: i32,
    o_b_stride: i32,
    o_h_stride: i32,
    scale: f32,
}

/// One compiled pipeline per (Metal device, kernel function), compiled on first use.
fn pipeline_for(device: &candle::MetalDevice, function: &'static str) -> Result<ComputePipeline> {
    static CACHE: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), function);
    {
        let guard = cache.lock().unwrap();
        if let Some(p) = guard.get(&key) {
            return Ok(p.clone());
        }
    }
    let mdev = device.metal_device();
    let lib = mdev
        .new_library_with_source(KERNEL_SRC, None)
        .map_err(candle::Error::wrap)?;
    let func = lib
        .get_function(function, None)
        .map_err(candle::Error::wrap)?;
    let pipeline = mdev
        .new_compute_pipeline_state_with_function(&func)
        .map_err(candle::Error::wrap)?;
    cache.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

fn offset_bytes(layout: &Layout, dtype: DType) -> usize {
    layout.start_offset() * dtype.size_in_bytes()
}

pub(crate) fn decode_attention_metal_fwd(
    op: &DecodeAttention,
    q: &MetalStorage,
    q_l: &Layout,
    k: &MetalStorage,
    k_l: &Layout,
    v: &MetalStorage,
    v_l: &Layout,
) -> Result<(MetalStorage, Shape)> {
    let function = match q.dtype() {
        DType::F32 => "decode_attention_f32",
        DType::F16 => "decode_attention_f16",
        dt => {
            candle::bail!("flashinfer-decode-attention (metal) only supports f32/f16, got {dt:?}")
        }
    };
    if k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        candle::bail!(
            "flashinfer-decode-attention (metal): q/k/v must share dtype ({:?}/{:?}/{:?})",
            q.dtype(),
            k.dtype(),
            v.dtype()
        );
    }

    let (b_sz, num_heads, head_dim) = q_l.shape().dims3()?;
    let (b_sz_k, num_heads_k, seqlen_k, head_dim_k) = k_l.shape().dims4()?;
    if k_l.shape() != v_l.shape() {
        candle::bail!(
            "shape mismatch between k {:?} and v {:?}",
            k_l.shape(),
            v_l.shape()
        );
    }
    if b_sz_k != b_sz || head_dim_k != head_dim {
        candle::bail!(
            "shape mismatch between q {:?} and k {:?}",
            q_l.shape(),
            k_l.shape()
        );
    }
    if num_heads % num_heads_k != 0 {
        candle::bail!(
            "number of kv heads {num_heads_k} must divide the number of query heads {num_heads}"
        )
    }
    if head_dim > 1024 {
        candle::bail!("flashinfer-decode-attention (metal): head_dim {head_dim} exceeds 1024");
    }

    let q_s = q_l.stride();
    let k_s = k_l.stride();
    let v_s = v_l.stride();
    if q_s[2] != 1 {
        candle::bail!("the head dimension of q must be contiguous {q_s:?}");
    }
    if k_s[3] != 1 {
        candle::bail!("the head dimension of k must be contiguous {k_s:?}");
    }
    if v_s[3] != 1 {
        candle::bail!("the head dimension of v must be contiguous {v_s:?}");
    }

    let device = q.device().clone();
    let pipeline = pipeline_for(&device, function)?;
    let elem_count = b_sz * num_heads * head_dim;
    let output = device.new_buffer(elem_count, q.dtype(), "flashinfer-decode-out")?;

    let params = DecodeParams {
        hkv_group: (num_heads / num_heads_k) as i32,
        seqlen_k: seqlen_k as i32,
        head_dim: head_dim as i32,
        q_b_stride: q_s[0] as i32,
        q_h_stride: q_s[1] as i32,
        k_b_stride: k_s[0] as i32,
        k_h_stride: k_s[1] as i32,
        k_l_stride: k_s[2] as i32,
        v_b_stride: v_s[0] as i32,
        v_h_stride: v_s[1] as i32,
        v_l_stride: v_s[2] as i32,
        o_b_stride: (num_heads * head_dim) as i32,
        o_h_stride: head_dim as i32,
        scale: op.softmax_scale,
    };

    // One threadgroup per (batch, query head); threads = next_pow2(head_dim), capped at 1024 to
    // match the fixed-size threadgroup reduction buffer in the shader.
    let mut threads = 1usize;
    while threads < head_dim {
        threads <<= 1;
    }
    if threads > 1024 {
        threads = 1024;
    }

    let encoder = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_label("flashinfer-decode-attention");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(q.buffer()), offset_bytes(q_l, q.dtype()));
    encoder.set_input_buffer(1, Some(k.buffer()), offset_bytes(k_l, k.dtype()));
    encoder.set_input_buffer(2, Some(v.buffer()), offset_bytes(v_l, v.dtype()));
    encoder.set_output_buffer(3, Some(output.as_ref()), 0);
    encoder.set_bytes(4, &params);

    let groups = MTLSize {
        width: num_heads,
        height: b_sz,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(groups, threads_per_group);

    let out = MetalStorage::new(output, device.clone(), elem_count, q.dtype());
    Ok((out, q_l.shape().clone()))
}
