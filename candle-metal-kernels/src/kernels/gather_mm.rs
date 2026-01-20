use crate::metal::{Buffer, ComputeCommandEncoder, Device};
use crate::utils::EncoderProvider;
use crate::{set_params, EncoderParam, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

pub use super::mlx_gemm::GemmDType;

/// Kernel variant for gather_mm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherMmKernel {
    /// Simple kernel: one thread per output element
    /// Best for small outputs or debugging
    Simple,
    /// Tiled kernel: uses threadgroup memory for A
    /// Better for medium-sized outputs
    Tiled,
    /// SIMD kernel: uses vectorized loads
    /// Best for large outputs with large K
    Simd,
}

impl Default for GatherMmKernel {
    fn default() -> Self {
        // SIMD kernel is generally the best performer
        Self::Simd
    }
}

/// Call the fused gather + matmul kernel for MoE (Mixture of Experts) acceleration.
///
/// This kernel fuses the index_select (gather) and matmul operations into a single
/// GPU kernel, eliminating intermediate memory allocations and kernel launch overhead.
///
/// # Arguments
/// * `device` - Metal device
/// * `ep` - Encoder provider
/// * `kernels` - Kernel cache
/// * `dtype` - Data type (F16, BF16, or F32)
/// * `(m, n, k, num_experts)` - Matrix dimensions:
///   - m: number of rows (num_tokens * num_experts_per_tok)
///   - n: output features
///   - k: hidden dimension / input features
///   - num_experts: number of expert weight matrices
/// * `a_offset` - Byte offset into A buffer
/// * `a_buffer` - Input activations [m, k] (flattened tokens with expert expansion)
/// * `b_offset` - Byte offset into B buffer
/// * `b_buffer` - Expert weights [num_experts, n, k] (transposed: each expert is [n, k])
/// * `indices_buffer` - Expert indices [m], one per row (u32)
/// * `output` - Output buffer [m, n]
#[allow(clippy::too_many_arguments)]
pub fn call_gather_mm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (m, n, k, num_experts): (usize, usize, usize, usize),
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    indices_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_gather_mm_with_kernel(
        device,
        ep,
        kernels,
        dtype,
        GatherMmKernel::default(),
        (m, n, k, num_experts),
        a_offset,
        a_buffer,
        b_offset,
        b_buffer,
        indices_buffer,
        output,
    )
}

/// Call gather_mm with a specific kernel variant
#[allow(clippy::too_many_arguments)]
pub fn call_gather_mm_with_kernel(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    kernel_variant: GatherMmKernel,
    (m, n, k, num_experts): (usize, usize, usize, usize),
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    indices_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GatherMMParams {
        m: i32,
        n: i32,
        k: i32,
        num_experts: i32,
        expert_stride: i32,
    }

    let params = GatherMMParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        num_experts: num_experts as i32,
        expert_stride: (n * k) as i32, // Each expert has shape [n, k]
    };

    let (kernel_name, grid_size, group_size) = match kernel_variant {
        GatherMmKernel::Simple => {
            // Simple kernel: one thread per output element
            let name = match dtype {
                GemmDType::F32 => "gather_mm_simple_f32",
                GemmDType::F16 => "gather_mm_simple_f16",
                GemmDType::BF16 => "gather_mm_simple_bf16",
            };
            let grid = MTLSize {
                width: n,
                height: m,
                depth: 1,
            };
            let group = MTLSize {
                width: 32.min(n),
                height: 32.min(m),
                depth: 1,
            };
            (name, grid, group)
        }
        GatherMmKernel::Tiled => {
            // Tiled kernel: one threadgroup per row
            // Each thread handles TN=4 columns
            let name = match dtype {
                GemmDType::F32 => "gather_mm_tiled_f32",
                GemmDType::F16 => "gather_mm_tiled_f16",
                GemmDType::BF16 => "gather_mm_tiled_bf16",
            };
            let threads_per_row = (n + 3) / 4; // Each thread handles 4 columns
            let threads_per_group = 256.min(threads_per_row);
            let grid = MTLSize {
                width: 1,
                height: m,
                depth: 1,
            };
            let group = MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            };
            (name, grid, group)
        }
        GatherMmKernel::Simd => {
            // SIMD kernel: 4 simdgroups per threadgroup, 32 threads per simdgroup
            // Each threadgroup handles 128 columns
            let name = match dtype {
                GemmDType::F32 => "gather_mm_simd_f32",
                GemmDType::F16 => "gather_mm_simd_f16",
                GemmDType::BF16 => "gather_mm_simd_bf16",
            };
            let col_tiles = (n + 127) / 128;
            let grid = MTLSize {
                width: col_tiles,
                height: m,
                depth: 1,
            };
            let group = MTLSize {
                width: 32,  // threads per simdgroup
                height: 4,  // simdgroups per threadgroup
                depth: 1,
            };
            (name, grid, group)
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::GatherMm, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    impl EncoderParam for GatherMMParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    set_params!(
        encoder,
        (
            (a_buffer, a_offset),
            (b_buffer, b_offset),
            indices_buffer,
            output,
            params
        )
    );

    encoder.use_resource(a_buffer, MTLResourceUsage::Read);
    encoder.use_resource(b_buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);

    Ok(())
}
