//! `RocmDevice::compile`, the `ug` micro-kernel path.
//!
//! `UgIOp1` (covered by `tests/custom_op_tests.rs`) only ever runs a single
//! in-place f32 argument through a kernel that `ug` lowers to a serial loop, so
//! it exercises neither the `Special` instructions nor `ReduceLocal`. The block
//! reductions are the part of the HIP code generator that is not a transcription
//! of `ug-cuda`'s — they use HIP's unsuffixed `__shfl_xor` — so they get a real
//! launch here.

use super::{rocm_error, RocmDevice, RocmStorageSlice};
use crate::{Device, Result, Tensor};

fn device() -> Option<RocmDevice> {
    RocmDevice::new(0).ok()
}

/// Device pointer to `t`'s f32 data.
fn f32_ptr(t: &Tensor) -> Result<*mut std::ffi::c_void> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        crate::Storage::Rocm(storage) => match &storage.slice {
            // SAFETY: `start_offset` indexes an element of this allocation.
            RocmStorageSlice::F32(src) => Ok(unsafe { src.ptr_at(layout.start_offset()) }),
            _ => crate::bail!("expected an f32 rocm storage"),
        },
        _ => crate::bail!("expected a rocm storage"),
    }
}

/// Compile `kernel` and run it over `input` as `(src, dst)`, returning `dst`.
///
/// The samples carry a placeholder launch config — `Kernel::from_instrs` always
/// records grid 1 / block 1 — so the geometry is the caller's to supply.
fn run(
    name: &'static str,
    kernel: candle_ug::lang::ssa::Kernel,
    input: &[f32],
    grid: u32,
    block: u32,
) -> Result<Option<Vec<f32>>> {
    let dev = match device() {
        Some(dev) => dev,
        None => return Ok(None),
    };
    let device = Device::Rocm(dev.clone());

    let src = Tensor::from_vec(input.to_vec(), input.len(), &device)?;
    let src_ptr = f32_ptr(&src)?;
    let dst = dev.alloc_zeros::<f32>(input.len())?;
    let dst_ptr = dst.as_ptr();

    let func = dev.compile(name, kernel)?;
    let mut args = [
        &src_ptr as *const _ as *mut std::ffi::c_void,
        &dst_ptr as *const _ as *mut std::ffi::c_void,
    ];
    func.launch(
        rocm_rs::hip::Dim3::from(grid),
        rocm_rs::hip::Dim3::from(block),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| rocm_error(format!("ug kernel {name} launch failed: {e}")))?;
    dev.synchronize()?;

    Ok(Some(dev.clone_dtoh(&dst)?))
}

/// `Special::BlockIdx` / `Special::ThreadIdx`, over a grid the kernel indexes
/// itself.
#[test]
fn a_block_indexed_ug_kernel_runs_on_rocm() -> Result<()> {
    const BLOCK: usize = 32;
    const BLOCKS: usize = 4;
    let n = BLOCK * BLOCKS;

    let input: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let kernel = candle_ug::samples::ssa::exp(BLOCK)?;
    let Some(out) = run("ug_exp", kernel, &input, BLOCKS as u32, BLOCK as u32)? else {
        return Ok(());
    };

    assert_eq!(out.len(), n);
    for (got, want) in out.iter().zip(input.iter().map(|v| v.exp())) {
        assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
    }
    Ok(())
}

/// `ReduceLocal`, i.e. the `block_reduce_*` helpers and the `__shfl_xor` width
/// of 32 they are built on. One block per row, one thread per column.
#[test]
fn a_ug_block_reduction_runs_on_rocm() -> Result<()> {
    const ROWS: usize = 3;
    const COLS: usize = 32;

    let input: Vec<f32> = (0..ROWS * COLS).map(|i| ((i % 7) as f32) - 3.0).collect();
    let kernel = candle_ug::samples::ssa::softmax_reduce(ROWS, COLS)?;
    let Some(out) = run("ug_softmax", kernel, &input, ROWS as u32, COLS as u32)? else {
        return Ok(());
    };

    assert_eq!(out.len(), ROWS * COLS);
    for row in 0..ROWS {
        let got = &out[row * COLS..(row + 1) * COLS];
        let src = &input[row * COLS..(row + 1) * COLS];
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = src.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        // A wrong reduction shows up as a row that does not sum to one long
        // before the individual terms drift.
        let got_sum: f32 = got.iter().sum();
        assert!(
            (got_sum - 1.0).abs() < 1e-4,
            "row {row} sums to {got_sum}, not 1"
        );
        for (got, want) in got.iter().zip(exps.iter().map(|e| e / sum)) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }
    Ok(())
}
