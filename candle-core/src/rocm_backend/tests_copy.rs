//! Regression tests for buffer copies, empty launches and the F8E4M3 arms.
//!
//! `mod.rs` is far over the workspace 400-line cap, so these live here.

use super::tests::rocm_device;
use super::{RocmDevice, RocmStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Device, Layout, Result, Tensor};

/// `None` when the machine has no ROCm GPU, like [`rocm_device`].
fn device() -> Option<RocmDevice> {
    RocmDevice::new(0).ok()
}

macro_rules! rocm_backend_device {
    () => {
        match crate::rocm_backend::tests_copy::device() {
            Some(dev) => dev,
            None => return Ok(()),
        }
    };
}

fn f32s(dev: &RocmDevice, v: &[f32]) -> Result<RocmStorage> {
    dev.storage_from_slice(v)
}

fn read(s: &RocmStorage) -> Result<Vec<f32>> {
    match s.to_cpu_storage()? {
        CpuStorage::F32(v) => Ok(v),
        other => crate::bail!("unexpected dtype {:?}", other.dtype()),
    }
}

/// Callers over-request: `Tensor::cat` and the autograd accumulators size the
/// copy from the *source shape*, which can run past the end of either
/// allocation. cuda_backend clamps for exactly this reason
/// (`slice_src_and_dst`); unclamped, the `hipMemcpy` here read and wrote off the
/// end of a device buffer.
#[test]
fn copy_strided_src_clamps_an_over_request() -> Result<()> {
    let dev = rocm_backend_device!();

    // Source side: four elements available, eight requested. The tail of the
    // destination has to keep its sentinel rather than take garbage.
    let src = f32s(&dev, &[0., 1., 2., 3.])?;
    let mut dst = f32s(&dev, &[-1.; 8])?;
    src.copy_strided_src(&mut dst, 0, &Layout::contiguous(8))?;
    assert_eq!(read(&dst)?, [0., 1., 2., 3., -1., -1., -1., -1.]);

    // Destination side: eight elements offered, four fit. `guard` is allocated
    // after `dst` so an unclamped copy has a good chance of landing in it.
    let src = f32s(&dev, &[0., 1., 2., 3., 4., 5., 6., 7.])?;
    let mut dst = f32s(&dev, &[-1.; 4])?;
    let guard = f32s(&dev, &[-9.; 8])?;
    src.copy_strided_src(&mut dst, 0, &Layout::contiguous(8))?;
    assert_eq!(read(&dst)?, [0., 1., 2., 3.]);
    assert_eq!(read(&guard)?, [-9.; 8]);

    // A destination offset past the end leaves nothing to copy.
    let mut dst = f32s(&dev, &[-1.; 4])?;
    src.copy_strided_src(&mut dst, 8, &Layout::contiguous(4))?;
    assert_eq!(read(&dst)?, [-1.; 4]);
    Ok(())
}

/// The strided `ucopy` path writes one element per thread into a contiguous
/// destination, so it needs the same destination clamp.
#[test]
fn copy_strided_src_clamps_the_strided_path() -> Result<()> {
    let dev = rocm_backend_device!();
    let src = f32s(&dev, &[0., 1., 2., 3., 4., 5., 6., 7.])?;
    let mut dst = f32s(&dev, &[-1.; 4])?;
    let guard = f32s(&dev, &[-9.; 8])?;
    // (2, 4) transposed is (4, 2) and non-contiguous: eight elements in the
    // order 0 4 1 5 2 6 3 7, of which only four fit.
    let l = Layout::contiguous((2, 4)).transpose(0, 1)?;
    src.copy_strided_src(&mut dst, 0, &l)?;
    assert_eq!(read(&dst)?, [0., 4., 1., 5.]);
    assert_eq!(read(&guard)?, [-9.; 8]);
    Ok(())
}

/// A copy that fits must still move every element — the clamp must not truncate
/// a legitimate request.
#[test]
fn copy_strided_src_still_copies_in_full() -> Result<()> {
    let dev = rocm_backend_device!();
    let src = f32s(&dev, &[0., 1., 2., 3., 4., 5.])?;
    let mut dst = f32s(&dev, &[-1.; 8])?;
    src.copy_strided_src(&mut dst, 2, &Layout::contiguous_with_offset(4, 2))?;
    assert_eq!(read(&dst)?, [-1., -1., 2., 3., 4., 5., -1., -1.]);
    Ok(())
}

/// `launch_config(dev, 0)` used to yield `gridDim.x == 0`, which
/// `hipModuleLaunchKernel` rejects — so every elementwise op on an empty tensor
/// errored instead of being a no-op.
#[test]
fn ops_on_an_empty_tensor_are_a_no_op() -> Result<()> {
    let dev = rocm_device!();
    let t = Tensor::zeros((0, 3), DType::F32, &dev)?;
    assert_eq!(t.neg()?.dims(), [0, 3]);
    assert_eq!(t.affine(2., 1.)?.dims(), [0, 3]);
    assert_eq!((&t + &t)?.dims(), [0, 3]);
    assert_eq!(t.to_dtype(DType::F64)?.dims(), [0, 3]);
    assert_eq!(t.copy()?.dims(), [0, 3]);
    // Reducing away the empty axis yields zeros; reducing the other one yields
    // an empty output, where `src_el / dst_el` would divide by zero.
    assert_eq!(t.sum(0)?.to_vec1::<f32>()?, [0., 0., 0.]);
    assert_eq!(t.sum(1)?.dims(), [0]);
    Ok(())
}

fn fp8(v: &[f32]) -> Vec<float8::F8E4M3> {
    v.iter().copied().map(float8::F8E4M3::from_f32).collect()
}

fn fp8_bits(t: &Tensor) -> Result<Vec<u8>> {
    Ok(t.flatten_all()?
        .to_vec1::<float8::F8E4M3>()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// `Tensor::cat`, `narrow(..).contiguous()` and transpose-then-copy all route
/// through `copy_strided_src`, which used to bail outright on F8E4M3 — failing
/// on ROCm while working on CUDA.
#[test]
fn f8e4m3_supports_strided_and_contiguous_copies() -> Result<()> {
    let dev = rocm_device!();
    let cpu = Device::Cpu;
    let vals = fp8(&(0..12).map(|i| i as f32).collect::<Vec<_>>());
    let c = Tensor::from_slice(&vals, (3, 4), &cpu)?;
    let g = Tensor::from_slice(&vals, (3, 4), &dev)?;

    // Transpose-then-copy takes the strided `ucopy` path.
    assert_eq!(
        fp8_bits(&g.t()?.contiguous()?)?,
        fp8_bits(&c.t()?.contiguous()?)?
    );
    // A narrowed view takes it too, with a non-zero start offset.
    assert_eq!(
        fp8_bits(&g.narrow(1, 1, 2)?.contiguous()?)?,
        fp8_bits(&c.narrow(1, 1, 2)?.contiguous()?)?
    );
    // `cat` on dim 0 takes the contiguous fast path.
    assert_eq!(
        fp8_bits(&Tensor::cat(&[&g, &g], 0)?)?,
        fp8_bits(&Tensor::cat(&[&c, &c], 0)?)?
    );
    Ok(())
}
