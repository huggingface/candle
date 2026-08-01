//! The device accessors downstream crates reach for when they mirror a CUDA
//! call site: the stream-owning constructor, the event-tracking pair and the
//! rocBLAS handle.
//!
//! These use nothing that is not public, so they fail if any of it stops being
//! reachable from outside `candle-core`.

use rocm_rs::rocblas::ffi::{rocblas_handle, rocblas_int, rocblas_status};

use super::RocmDevice;
use crate::{Device, Result, Tensor};

fn device() -> Option<RocmDevice> {
    RocmDevice::new(0).ok()
}

macro_rules! rocm_dev {
    () => {
        match device() {
            Some(dev) => dev,
            None => return Ok(()),
        }
    };
}

/// `new_with_stream` has to build a usable device, and — the part that would
/// silently break the allocator if it ever stopped holding — the stream it gets
/// must be its own rather than one shared with another device.
#[test]
fn new_with_stream_gives_the_device_a_private_stream() -> Result<()> {
    let a = match RocmDevice::new_with_stream(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let b = RocmDevice::new(0)?;
    assert!(!a.stream().as_raw().is_null());
    assert_ne!(a.stream().as_raw(), b.stream().as_raw());

    let dev = Device::Rocm(a);
    let t = Tensor::new(&[1f32, 2., 3.], &dev)?;
    assert_eq!((&t + &t)?.to_vec1::<f32>()?, [2., 4., 6.]);
    Ok(())
}

/// `Device::new_rocm_with_stream` is the enum-level door to the same thing.
#[test]
fn device_new_rocm_with_stream_matches_new_rocm() -> Result<()> {
    let dev = match Device::new_rocm_with_stream(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    assert!(dev.is_rocm());
    let t = Tensor::new(&[2f32, 4.], &dev)?;
    assert_eq!(t.sum_all()?.to_scalar::<f32>()?, 6.);
    Ok(())
}

/// Event tracking is off and stays off: the backend never records per-buffer
/// events, so the CUDA opt-out has nothing to switch and must not pretend it
/// did.
#[test]
fn event_tracking_is_off_and_disabling_it_is_a_no_op() -> Result<()> {
    let dev = rocm_dev!();
    assert!(!dev.is_event_tracking());
    // SAFETY: documented as a no-op on ROCm; nothing to uphold.
    unsafe { dev.disable_event_tracking() };
    assert!(!dev.is_event_tracking());

    let device = Device::Rocm(dev);
    let t = Tensor::new(&[1f32, 2., 3.], &device)?;
    assert_eq!((&t * 2.0)?.to_vec1::<f32>()?, [2., 4., 6.]);
    Ok(())
}

/// The handle is bound to the device's stream, and cloning the device does not
/// hand out a second handle bound to something else.
#[test]
fn rocblas_handle_is_bound_to_the_device_stream() -> Result<()> {
    let dev = rocm_dev!();
    let blas = dev.rocblas_handle();
    assert!(!blas.as_raw().is_null());
    assert_eq!(blas.stream().as_raw(), dev.stream().as_raw());

    let clone = dev.clone();
    assert_eq!(clone.rocblas_handle().as_raw(), blas.as_raw());
    assert_eq!(
        clone.rocblas_handle().stream().as_raw(),
        dev.stream().as_raw()
    );
    Ok(())
}

extern "C" {
    fn rocblas_saxpy(
        handle: rocblas_handle,
        n: rocblas_int,
        alpha: *const f32,
        x: *const f32,
        incx: rocblas_int,
        y: *mut f32,
        incy: rocblas_int,
    ) -> rocblas_status;
}

/// What the accessor is *for*: a downstream crate declares a rocBLAS entry point
/// candle does not wrap and calls it against candle's own device memory. The
/// result also shows the call landed on the device stream — `clone_dtoh` drains
/// only that stream before reading back.
#[test]
fn a_downstream_rocblas_call_runs_through_the_handle() -> Result<()> {
    let dev = rocm_dev!();
    let n = 512usize;
    let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y_host: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
    let x = dev.clone_htod(&x_host)?;
    let y = dev.clone_htod(&y_host)?;

    let blas = dev.rocblas_handle();
    let alpha = 3.0f32;
    // SAFETY: both buffers hold `n` f32s with unit stride, `alpha` is a host
    // pointer and rocBLAS' default pointer mode is host, and the handle is the
    // device's own and outlives the call.
    let status = unsafe {
        rocblas_saxpy(
            blas.as_raw(),
            n as rocblas_int,
            &alpha,
            x.as_ptr() as *const f32,
            1,
            y.as_ptr() as *mut f32,
            1,
        )
    };
    assert_eq!(
        status,
        rocm_rs::rocblas::ffi::rocblas_status__rocblas_status_success
    );

    let out = dev.clone_dtoh(&y)?;
    let expected: Vec<f32> = x_host
        .iter()
        .zip(y_host.iter())
        .map(|(x, y)| alpha * x + y)
        .collect();
    assert_eq!(out, expected);
    Ok(())
}
