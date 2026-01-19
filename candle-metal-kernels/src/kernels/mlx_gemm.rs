use crate::metal::{Buffer, ComputeCommandEncoder, Device, MetalDeviceType};
use crate::utils::EncoderProvider;
use crate::{set_params, ConstantValues, EncoderParam, Kernels, MetalKernelError, Source, Value};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
}

/// Tile configuration for GEMM kernel.
///
/// These parameters control the block sizes and warp tiling for the Metal GEMM kernel.
/// Different configurations are optimal for different matrix sizes and data types.
///
/// Reference: MLX steel_gemm_fused.metal
#[derive(Copy, Clone, Debug)]
struct TileConfig {
    bm: usize, // Block size M
    bn: usize, // Block size N
    bk: usize, // Block size K
    wm: usize, // Warp tiles M
    wn: usize, // Warp tiles N
}

impl TileConfig {
    const fn new(bm: usize, bn: usize, bk: usize, wm: usize, wn: usize) -> Self {
        Self { bm, bn, bk, wm, wn }
    }
}

// Predefined tile configurations matching MLX's steel_gemm_fused.metal
// Note: TILE_32_32_16_2_2 is kept for backward compatibility and as a fallback.
// It's used by MLX for small devices ('g'/'p') but we default to medium device configs.
#[allow(dead_code)]
const TILE_32_32_16_2_2: TileConfig = TileConfig::new(32, 32, 16, 2, 2);
const TILE_64_64_16_2_2: TileConfig = TileConfig::new(64, 64, 16, 2, 2);
const TILE_64_64_16_1_2: TileConfig = TileConfig::new(64, 64, 16, 1, 2);
const TILE_64_32_32_2_2: TileConfig = TileConfig::new(64, 32, 32, 2, 2);
const TILE_32_64_16_1_2: TileConfig = TileConfig::new(32, 64, 16, 1, 2);

/// Select optimal tile configuration based on matrix dimensions, data type, transpose mode,
/// and device type.
///
/// This implements MLX's GEMM_TPARAM_MACRO tile selection logic.
/// Reference: refs/mlx/mlx/backend/metal/matmul.cpp lines 88-170
///
/// The selection is based on:
/// - Device type (phone/base-pro for small, ultra for large, others for medium)
/// - Total output size (batch_size * M * N)
/// - Data type (F32 vs F16/BF16)
/// - Transpose mode (nn, nt, tn, tt)
/// - K dimension relative to M and N
fn select_tile_config(
    dtype: GemmDType,
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
    a_trans: bool,
    b_trans: bool,
    device_type: MetalDeviceType,
) -> TileConfig {
    // Special case: For very small M (vector-matrix multiply),
    // use the original 32x32 tile to avoid thread waste.
    // When M is very small (< bm), using larger bm values causes significant
    // thread underutilization because most threads in the M dimension have no work.
    // This is critical for benchmarks like [1, 2048] @ [2048, 2048] (m=1).
    //
    // We use m < 16 as the threshold because:
    // - For m=1 to m=15, even 32x32 tile has some waste but it's the smallest available
    // - For m >= 16, the larger tiles can provide better throughput despite some waste
    if m < 16 {
        return TILE_32_32_16_2_2;
    }

    // MLX uses batch_size * M * N >= 1M as the threshold for "large matmul"
    let total_output = batch_size * m * n;
    let is_large_matmul = total_output >= (1 << 20); // 1M elements

    match device_type {
        // Small devices: phone ('p') and base/pro ('g')
        MetalDeviceType::Phone | MetalDeviceType::BasePro => {
            // MLX: if (devc == 'g' || devc == 'p')
            if !a_trans && b_trans {
                // nt mode
                TILE_64_32_32_2_2
            } else if dtype != GemmDType::F32 {
                // half and bfloat
                TILE_64_64_16_1_2
            } else {
                // float32 default
                TILE_64_64_16_2_2
            }
        }
        // Large device: ultra ('d')
        MetalDeviceType::Ultra => {
            // MLX: if (devc == 'd')
            if is_large_matmul {
                // Large matmul
                if dtype != GemmDType::F32 {
                    // half and bfloat
                    if 2 * m.max(n) > k {
                        // Reasonable K
                        TILE_64_64_16_1_2
                    } else if !a_trans && b_trans {
                        // nt with large K
                        TILE_64_32_32_2_2
                    } else {
                        // nn with large K
                        TILE_32_64_16_1_2
                    }
                } else {
                    // float32 takes default
                    TILE_64_64_16_2_2
                }
            } else {
                // Smaller matmul
                if dtype != GemmDType::F32 {
                    // half and bfloat
                    if !a_trans && b_trans {
                        // nt
                        TILE_64_32_32_2_2
                    } else {
                        // nn
                        TILE_64_64_16_1_2
                    }
                } else {
                    // floats
                    if !a_trans && b_trans {
                        // nt
                        TILE_32_64_16_1_2
                    } else {
                        // nn
                        TILE_64_32_32_2_2
                    }
                }
            }
        }
        // Medium devices: max ('s') and unknown
        MetalDeviceType::Max | MetalDeviceType::Medium => {
            // MLX: default medium device config
            // Use the same logic as before but with medium device defaults
            match dtype {
                GemmDType::F32 => {
                    if !is_large_matmul {
                        if !a_trans && b_trans {
                            TILE_32_64_16_1_2
                        } else {
                            TILE_64_32_32_2_2
                        }
                    } else {
                        TILE_64_64_16_2_2
                    }
                }
                GemmDType::F16 | GemmDType::BF16 => {
                    if is_large_matmul {
                        if 2 * m.max(n) > k {
                            TILE_64_64_16_1_2
                        } else if !a_trans && b_trans {
                            TILE_64_32_32_2_2
                        } else {
                            TILE_32_64_16_1_2
                        }
                    } else if !a_trans && b_trans {
                        TILE_64_32_32_2_2
                    } else {
                        TILE_64_64_16_1_2
                    }
                }
            }
        }
    }
}

/// Check if batch can be collapsed into M dimension.
///
/// MLX's batch collapse optimization (from matmul.cpp lines 700-740):
/// When B is broadcasted (2D), we can collapse batch into M dimension:
/// - [batch, M, K] @ [K, N] -> [batch*M, K] @ [K, N]
///
/// Conditions for batch collapse:
/// 1. batch_size > 1
/// 2. !transpose_a (A is not transposed, i.e., row-major for M dimension)
/// 3. A is contiguous in batch dimension (batch_stride_a == M * K)
/// 4. B is broadcasted (batch_stride_b == 0, meaning B is 2D)
///
/// Returns (effective_batch, effective_m, should_collapse)
fn check_batch_collapse(
    b: usize,
    m: usize,
    k: usize,
    a_trans: bool,
    lhs_stride: &[usize],
    rhs_stride: &[usize],
) -> (usize, usize, bool) {
    if b <= 1 {
        return (b, m, false);
    }

    // A must not be transposed for batch collapse
    if a_trans {
        return (b, m, false);
    }

    // Check A's batch stride - must be contiguous (batch_stride_a == M * K)
    let a_batch_stride = if lhs_stride.len() > 2 {
        lhs_stride[lhs_stride.len() - 3]
    } else {
        m * k
    };

    // Check B's batch stride - must be 0 (broadcasted) for collapse
    let b_batch_stride = if rhs_stride.len() > 2 {
        rhs_stride[rhs_stride.len() - 3]
    } else {
        0 // B is 2D, effectively broadcasted
    };

    // For batch collapse:
    // - A must be contiguous: batch_stride_a == M * K
    // - B must be broadcasted: batch_stride_b == 0
    let a_contiguous = a_batch_stride == m * k;
    let b_broadcasted = b_batch_stride == 0;

    if a_contiguous && b_broadcasted {
        // Collapse batch into M: new_m = batch * m, new_batch = 1
        (1, b * m, true)
    } else {
        (b, m, false)
    }
}

/// Check if we can use split-K strategy for better performance.
///
/// MLX uses split-K when:
/// - batch_size == 1
/// - (M/16) * (N/16) <= 32 (small output)
/// - K/16 >= 8 (large K)
///
/// This is useful for tall-skinny matrices where K >> M*N
#[allow(dead_code)]
fn should_use_split_k(b: usize, m: usize, n: usize, k: usize) -> bool {
    if b != 1 {
        return false;
    }
    let tm = m / 16;
    let tn = n / 16;
    let tk = k / 16;
    (tm * tn) <= 32 && tk >= 8
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    // Check for batch collapse optimization (MLX matmul.cpp lines 700-740)
    // When B is broadcasted (2D), collapse batch into M dimension
    let (effective_batch, effective_m, batch_collapsed) =
        check_batch_collapse(b, m, k, a_trans, lhs_stride, rhs_stride);

    // Use effective dimensions after potential batch collapse
    let m = effective_m;
    let b = effective_batch;

    // Dynamic tile selection based on matrix dimensions, dtype, transpose mode, and device type
    // Reference: MLX GEMM_TPARAM_MACRO in matmul.cpp
    let device_type = device.device_type();
    let tile = select_tile_config(dtype, m, n, k, b, a_trans, b_trans, device_type);
    let (bm, bn, bk, wm, wn) = (tile.bm, tile.bn, tile.bk, tile.wm, tile.wn);

    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    // MLX uses batch_shape.size() > 1 for has_batch, not b > 1
    // When batch dimensions are collapsed into a single dimension (batch_ndim = 1),
    // has_batch should be false because we can use simple stride multiplication
    // instead of the more expensive elem_to_loc_broadcast function
    let batch_ndim = 1; // We always collapse batch dimensions into one
    let has_batch = batch_ndim > 1; // This matches MLX's logic

    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(has_batch)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile_swizzle = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile_swizzle;
    let tm = tm.div_ceil(tile_swizzle);

    // Calculate batch strides based on whether batch was collapsed
    let (batch_stride_a, batch_stride_b) = if batch_collapsed {
        // After batch collapse, there's no batch dimension
        (0isize, 0isize)
    } else {
        let a_stride = if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as isize
        } else {
            (m * k) as isize
        };
        let b_stride = if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as isize
        } else {
            (n * k) as isize
        };
        (a_stride, b_stride)
    };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: if batch_collapsed { k as i32 } else { lda }, // After collapse, lda = K
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };

    // Dynamically generate kernel name based on dtype, transpose mode, and tile config
    // Format: gemm_{trans}_{itype}_{otype}_{bm}_{bn}_{bk}_{wm}_{wn}
    let dtype_str = match dtype {
        GemmDType::F32 => "f32",
        GemmDType::F16 => "f16",
        GemmDType::BF16 => "bf16",
    };
    let trans_str = match (a_trans, b_trans) {
        (false, false) => "nn",
        (true, false) => "tn",
        (false, true) => "nt",
        (true, true) => "tt",
    };
    let name = format!(
        "gemm_{}_{}_{}_{}_{}_{}_{}_{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
    );

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Gemm, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    impl EncoderParam for GemmParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    // Set buffer parameters
    // Note: batch_shape and batch_strides are only needed when has_batch = true
    // Since we always collapse batch dimensions into one (batch_ndim = 1),
    // has_batch is always false, so we don't need to set buffers 6 and 7
    set_params!(
        encoder,
        (
            (lhs_buffer, lhs_offset),
            (rhs_buffer, rhs_offset),
            (),
            output,
            gemm_params
        )
    );

    let grid_size = MTLSize {
        width: tn,
        height: tm,
        depth: /* batch_size_out */ b,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    encoder.use_resource(lhs_buffer, MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}
