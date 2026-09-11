//! Unit tests for the argsort launcher in `crate::sort`.
//!
//! `ASORT_OP` is a per-row bitonic sort with an `extern __shared__` scratch
//! buffer, so it depends on three things the other launchers do not: a grid of
//! exactly `nrows` blocks, a dynamic shared-memory size, and an argument list
//! without the usual `(numel, num_dims, dims_and_strides)` prefix. A wrong grid
//! or shared-memory size mis-orders the output instead of failing, hence the
//! comparisons against a CPU reference rather than hard-coded expectations.

use super::tests::rocm_device;
use crate::{DType, Device, Result, Tensor};

/// Sorts the same data on ROCm and on the CPU and checks the two agree.
///
/// The sort is unstable, so ties may land on different indices on either side;
/// only the sorted *values* and the fact that the indices are a permutation of
/// `0..cols` are guaranteed.
fn check_against_cpu(
    dev: &Device,
    data: &[f32],
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<()> {
    assert_eq!(data.len(), rows * cols);
    let build = |d: &Device| Tensor::from_slice(data, (rows, cols), d)?.to_dtype(dtype);
    let gpu = build(dev)?;
    let cpu = build(&Device::Cpu)?;

    for asc in [true, false] {
        let (gpu_sorted, gpu_indexes) = gpu.sort_last_dim(asc)?;
        let (cpu_sorted, _) = cpu.sort_last_dim(asc)?;
        assert_eq!(gpu_indexes.dtype(), DType::U32, "{dtype:?} asc={asc}");
        assert_eq!(gpu_indexes.dims(), [rows, cols], "{dtype:?} asc={asc}");
        assert_eq!(
            gpu_sorted.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            cpu_sorted.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            "{dtype:?} asc={asc} sorted values"
        );
        let identity: Vec<u32> = (0..cols as u32).collect();
        for row in gpu_indexes.to_vec2::<u32>()? {
            let mut sorted_row = row;
            sorted_row.sort_unstable();
            assert_eq!(
                sorted_row, identity,
                "{dtype:?} asc={asc} index permutation"
            );
        }
    }
    Ok(())
}

/// Every dtype `sort.cu` instantiates `ASORT_OP` for has to resolve a kernel,
/// in both directions. The values stay small and integral so they survive the
/// round trip through `u8` and both half-precision dtypes unchanged.
#[test]
fn argsort_covers_every_instantiated_dtype() -> Result<()> {
    let dev = rocm_device!();
    let data = [7f32, 2., 9., 4., 1., 6.];
    for dtype in [
        DType::U8,
        DType::U32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        check_against_cpu(&dev, &data, 2, 3, dtype)?;
    }
    Ok(())
}

/// A column count that is not a power of two makes `ncols_pad` larger than
/// `ncols`: the padding slots must sort to the end and never reach the output.
/// Several rows at once catch a grid sized by elements rather than by rows.
#[test]
fn argsort_handles_non_power_of_two_columns() -> Result<()> {
    let dev = rocm_device!();
    // 5 columns, so ncols_pad = 8.
    let data = [
        3f32, 1., 4., 1.5, 5., //
        2.5, 1., 7., 8., 2., //
        -1., 0., -3., 2., 0.5,
    ];
    check_against_cpu(&dev, &data, 3, 5, DType::F32)?;
    check_against_cpu(&dev, &data, 3, 5, DType::F16)?;

    // 11 columns (ncols_pad = 16) over a single row.
    let data: Vec<f32> = [5f32, 3., 8., 1., 9., 2., 7., 0., 6., 4., 10.].to_vec();
    check_against_cpu(&dev, &data, 1, 11, DType::F32)
}

/// Equal values are the case where an unstable sort can silently drop or
/// duplicate an index, so the permutation check carries the weight here.
#[test]
fn argsort_handles_ties() -> Result<()> {
    let dev = rocm_device!();
    let data = [
        2f32, 2., 2., 2., 2., 2., // all equal
        1., 3., 1., 3., 1., 3., // two values interleaved
        0., 0., 5., 5., 0., 5.,
    ];
    check_against_cpu(&dev, &data, 3, 6, DType::F32)?;
    check_against_cpu(&dev, &data, 3, 6, DType::BF16)
}

/// More columns than the 1024-thread block cap, across several rows: each
/// thread then walks `col += blockDim.x`, and every row needs its own block
/// with its own `ncols_pad * 4` bytes of shared scratch.
#[test]
fn argsort_handles_rows_wider_than_one_block() -> Result<()> {
    let dev = rocm_device!();
    const ROWS: usize = 3;
    // 1500 is neither a power of two nor a multiple of the block size.
    const COLS: usize = 1500;
    // A deterministic, non-monotonic permutation-like sequence with a few ties.
    let data: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i * 7919) % 977) as f32)
        .collect();
    check_against_cpu(&dev, &data, ROWS, COLS, DType::F32)
}

/// `arg_sort_last_dim` sorts the last dimension of a >2D tensor row by row, so
/// `nrows` is the product of the leading dims rather than `dims[0]`.
#[test]
fn argsort_treats_leading_dims_as_rows() -> Result<()> {
    let dev = rocm_device!();
    let data: Vec<f32> = (0..24).map(|i| ((i * 5) % 7) as f32).collect();
    let gpu = Tensor::from_slice(data.as_slice(), (2, 4, 3), &dev)?;
    let cpu = Tensor::from_slice(data.as_slice(), (2, 4, 3), &Device::Cpu)?;
    let (gpu_sorted, gpu_indexes) = gpu.sort_last_dim(true)?;
    let (cpu_sorted, _) = cpu.sort_last_dim(true)?;
    assert_eq!(gpu_indexes.dims(), [2, 4, 3]);
    assert_eq!(gpu_sorted.to_vec3::<f32>()?, cpu_sorted.to_vec3::<f32>()?);
    Ok(())
}

/// The kernel indexes its input as a dense `row * ncols + col` buffer, so a
/// non-contiguous view has to be rejected rather than silently sorted wrong.
#[test]
fn argsort_rejects_non_contiguous_input() -> Result<()> {
    let dev = rocm_device!();
    let t = Tensor::new(&[[3f32, 1., 4.], [1., 5., 9.]], &dev)?.t()?;
    assert!(t.arg_sort_last_dim(true).is_err());
    Ok(())
}
