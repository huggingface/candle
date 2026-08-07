//! The downstream-kernel path: a crate that owns a `.cu` source of its own has
//! to be able to compile and launch it against a candle tensor without patching
//! candle.
//!
//! This test only uses what such a crate can reach — `RocmDevice`,
//! `get_or_load_custom_func`, `Tensor::storage_and_layout`, `RocmStorageSlice`
//! and `launch_config` — so it fails if any of them stops being public.

use super::{launch_config, RocmDevice, RocmStorageSlice};
use crate::{Device, Result, Tensor};

/// Written in the dialect of `candle-kernels/src/*.cu`, headers included: the
/// staged headers and the HIP shim have to be reachable from a source candle
/// does not own, or a real port (Crane's GDN kernel is the motivating one)
/// would have to vendor them.
const SOURCE: &str = r#"
#include "cuda_utils.cuh"

extern "C" __global__ void candle_test_affine(
    const float *x, float *y, const size_t n, const float mul, const float add) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * mul + add;
    }
}
"#;

fn device() -> Option<RocmDevice> {
    RocmDevice::new(0).ok()
}

#[test]
fn a_downstream_kernel_compiles_and_runs_against_a_tensor() -> Result<()> {
    let dev = match device() {
        Some(dev) => dev,
        None => return Ok(()),
    };
    let device = Device::Rocm(dev.clone());

    let n = 1024usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Tensor::from_vec(input.clone(), n, &device)?;

    let dst = dev.alloc_zeros::<f32>(n)?;
    let (storage, layout) = x.storage_and_layout();
    let src_ptr = match &*storage {
        crate::Storage::Rocm(storage) => match &storage.slice {
            RocmStorageSlice::F32(src) => unsafe { src.ptr_at(layout.start_offset()) },
            _ => crate::bail!("expected an f32 rocm storage"),
        },
        _ => crate::bail!("expected a rocm storage"),
    };

    let func = dev.get_or_load_custom_func("candle_test_affine", "candle_test_custom", SOURCE)?;
    let dst_ptr = dst.as_ptr();
    let (mul, add) = (2.0f32, 1.0f32);
    let mut args = vec![
        &src_ptr as *const _ as *mut std::ffi::c_void,
        &dst_ptr as *const _ as *mut std::ffi::c_void,
        &n as *const _ as *mut std::ffi::c_void,
        &mul as *const _ as *mut std::ffi::c_void,
        &add as *const _ as *mut std::ffi::c_void,
    ];
    let (grid, block) = launch_config(&dev, n);
    func.launch(grid, block, 0, Some(dev.stream()), &mut args)
        .map_err(|e| super::rocm_error(format!("custom kernel launch failed: {e}")))?;
    dev.synchronize()?;

    let out = dev.clone_dtoh(&dst)?;
    let expected: Vec<f32> = input.iter().map(|v| v * mul + add).collect();
    assert_eq!(out, expected);
    Ok(())
}
