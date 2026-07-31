//! The `gemm_reduced_precision_*` knobs.
//!
//! The flags are process-wide, so every test here takes [`FLAGS`] first: without
//! it two of these running in parallel would observe each other's writes. No
//! other unit test in this crate issues a matmul, so nothing outside this module
//! can be perturbed by them.

use std::sync::{Mutex, MutexGuard};

use super::{
    gemm_reduced_precision_bf16, gemm_reduced_precision_f16, gemm_reduced_precision_f32,
    set_gemm_reduced_precision_bf16, set_gemm_reduced_precision_f16,
    set_gemm_reduced_precision_f32, RocmDevice,
};
use crate::{DType, Device, Result, Tensor};

static FLAGS: Mutex<()> = Mutex::new(());

fn lock_flags() -> MutexGuard<'static, ()> {
    // A test that asserted its way out mid-toggle poisons this; the flags are
    // reset unconditionally below, so the guard is still usable.
    FLAGS.lock().unwrap_or_else(|e| e.into_inner())
}

macro_rules! device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            // No ROCm device on this machine.
            Err(_) => return Ok(()),
        }
    };
}

/// `ones(32,32) @ ones(32,32)`, whose exact answer is 32 everywhere — a k of 32
/// is small enough that even an f16 accumulator is exact, so a wrong value here
/// means the compute type was rejected rather than merely less accurate.
fn ones_matmul(dev: &Device, dtype: DType) -> Result<f32> {
    let a = Tensor::ones((32, 32), dtype, dev)?;
    let b = Tensor::ones((32, 32), dtype, dev)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    Ok(c.flatten_all()?.to_vec1::<f32>()?[0])
}

/// The defaults must stay off, i.e. f16/bf16 keep accumulating in f32 exactly as
/// `cuda_backend` does out of the box, and each pair must round-trip so portable
/// code can read back what it asked for even where the backend cannot honour it.
#[test]
fn the_knobs_default_to_off_and_round_trip_rocm() {
    let _guard = lock_flags();
    assert!(!gemm_reduced_precision_f16());
    assert!(!gemm_reduced_precision_bf16());
    assert!(!gemm_reduced_precision_f32());

    set_gemm_reduced_precision_f16(true);
    set_gemm_reduced_precision_bf16(true);
    set_gemm_reduced_precision_f32(true);
    let seen = (
        gemm_reduced_precision_f16(),
        gemm_reduced_precision_bf16(),
        gemm_reduced_precision_f32(),
    );
    set_gemm_reduced_precision_f16(false);
    set_gemm_reduced_precision_bf16(false);
    set_gemm_reduced_precision_f32(false);
    assert_eq!(seen, (true, true, true));
}

/// f16 is the one knob that is live: enabling it swaps the rocBLAS compute type
/// (and with it the alpha/beta pointer type) from f32 to f16. rocBLAS rejects an
/// unsupported compute type with a status code rather than silently ignoring it,
/// so a failure here means gfx1101 cannot select f16 compute at all.
#[test]
fn f16_reduced_precision_is_selectable_rocm() -> Result<()> {
    let _guard = lock_flags();
    let dev = device!();
    let full = ones_matmul(&dev, DType::F16)?;

    set_gemm_reduced_precision_f16(true);
    let reduced = ones_matmul(&dev, DType::F16);
    set_gemm_reduced_precision_f16(false);

    assert_eq!(full, 32.0);
    assert_eq!(reduced?, 32.0);
    Ok(())
}

/// bf16 and f32 store their flag but the GEMM path does not consult it: rocBLAS
/// has no bf16 compute type, and xf32 is a CDNA-only math mode. Results must be
/// bit-identical either way.
#[test]
fn the_no_op_knobs_do_not_change_results_rocm() -> Result<()> {
    let _guard = lock_flags();
    let dev = device!();
    for dtype in [DType::BF16, DType::F32] {
        let before = ones_matmul(&dev, dtype)?;
        set_gemm_reduced_precision_bf16(true);
        set_gemm_reduced_precision_f32(true);
        let after = ones_matmul(&dev, dtype);
        set_gemm_reduced_precision_bf16(false);
        set_gemm_reduced_precision_f32(false);
        assert_eq!(before, 32.0);
        assert_eq!(after?, before);
    }
    Ok(())
}

/// The knob must not be a *silent* no-op: enabling f16 compute has to actually
/// change what rocBLAS computes. Over a k of 4096 an f16 accumulator drifts
/// visibly from an f32 one, so the two results must disagree — while the
/// default (f32 accumulate) stays close to a CPU f32 reference.
#[test]
fn f16_reduced_precision_actually_changes_the_result_rocm() -> Result<()> {
    let _guard = lock_flags();
    let dev = device!();
    const K: usize = 4096;
    let a = Tensor::rand(0f32, 1f32, (16, K), &dev)?.to_dtype(DType::F16)?;
    let b = Tensor::rand(0f32, 1f32, (K, 16), &dev)?.to_dtype(DType::F16)?;

    let full = a.matmul(&b)?.to_dtype(DType::F32)?;
    set_gemm_reduced_precision_f16(true);
    let reduced = a.matmul(&b)?.to_dtype(DType::F32);
    set_gemm_reduced_precision_f16(false);
    let reduced = reduced?;

    let drift = (&full - &reduced)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        drift > 1.0,
        "f16 compute type had no effect (drift {drift})"
    );

    // ... and the default really is the accurate one. The f32 reference runs on
    // the CPU so it shares no code with either GEMM above.
    let reference = a
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .matmul(&b.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)?;
    let full_err = (&full.to_device(&Device::Cpu)? - &reference)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let reduced_err = (&reduced.to_device(&Device::Cpu)? - &reference)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        full_err < reduced_err,
        "f32 accumulate ({full_err}) should beat f16 accumulate ({reduced_err})"
    );
    Ok(())
}
