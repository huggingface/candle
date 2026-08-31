//! What a downstream HIP kernel gets from `device_ptr`, exercised the way such
//! a kernel would: raw `hipMemcpy` off the returned address.

use rocm_rs::hip::{bindings, Stream};

use super::*;
use crate::CpuStorage;

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

fn quantized(dev: &RocmDevice) -> Result<QRocmStorage> {
    let xs: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.11).sin()).collect();
    let mut storage = QRocmStorage::zeros(dev, xs.len(), GgmlDType::Q4_0)?;
    storage.quantize_onto(&CpuStorage::F32(xs))?;
    Ok(storage)
}

/// A blocking read straight off the raw pointer, i.e. what a downstream kernel
/// would address.
fn read_back(ptr: *const u8, len: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; len];
    // SAFETY: `ptr` addresses at least `len` bytes of device memory for as long
    // as the storage it came from is alive, and `out` is `len` bytes of host
    // memory.
    let status = unsafe {
        bindings::hipMemcpy(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            ptr as *const std::ffi::c_void,
            len,
            bindings::hipMemcpyKind_hipMemcpyDeviceToHost,
        )
    };
    if status != bindings::hipError_t_hipSuccess {
        crate::bail!("hipMemcpy from the quantized pointer failed with {status}")
    }
    Ok(out)
}

/// An owned stream that is *not* the device's, so the guard has something to
/// order against.
fn foreign_stream() -> Result<Stream> {
    let mut raw = std::ptr::null_mut();
    // SAFETY: `raw` is a valid out-pointer; the returned handle is owned by the
    // `Stream` below, which destroys it on drop.
    let status = unsafe { bindings::hipStreamCreate(&mut raw) };
    if status != bindings::hipError_t_hipSuccess {
        crate::bail!("hipStreamCreate failed with {status}")
    }
    Ok(Stream::from_raw(raw))
}

#[test]
fn device_ptr_addresses_the_payload() -> Result<()> {
    let dev = rocm_device!();
    let storage = quantized(&dev)?;
    let expected = storage.data()?;
    let ptr = storage.device_ptr()?;
    assert!(!ptr.is_null());
    assert_eq!(read_back(ptr, expected.len())?, expected);
    Ok(())
}

/// The device's own stream is already ordered against the buffer, so the guard
/// hands back the same address and drops without touching the driver.
#[test]
fn a_guard_on_the_device_stream_is_the_bare_pointer() -> Result<()> {
    let dev = rocm_device!();
    let storage = quantized(&dev)?;
    let expected = storage.data()?;
    let (ptr, guard) = storage.device_ptr_with_guard(dev.stream())?;
    assert_eq!(ptr, storage.device_ptr()?);
    assert_eq!(read_back(ptr, expected.len())?, expected);
    drop(guard);
    // The storage outlives the guard, so the payload is still there.
    assert_eq!(storage.data()?, expected);
    Ok(())
}

/// The point of the guard: work submitted on another stream through the raw
/// pointer has completed by the time the guard is gone, so the block cannot be
/// recycled underneath it.
#[test]
fn dropping_a_guard_drains_a_foreign_stream() -> Result<()> {
    let dev = rocm_device!();
    let storage = quantized(&dev)?;
    let expected = storage.data()?;
    let stream = foreign_stream()?;
    let dst = dev.alloc_zeros::<u8>(expected.len())?;

    let (ptr, guard) = storage.device_ptr_with_guard(&stream)?;
    // SAFETY: both pointers address at least `expected.len()` bytes, and the
    // copy is queued while the guard is alive.
    let status = unsafe {
        bindings::hipMemcpyAsync(
            dst.as_ptr(),
            ptr as *const std::ffi::c_void,
            expected.len(),
            bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
            stream.as_raw(),
        )
    };
    assert_eq!(status, bindings::hipError_t_hipSuccess);
    drop(guard);

    // `query` reports the stream idle only because the drop drained it.
    stream.query().map_err(|e| {
        crate::Error::Msg(format!("stream still busy after the guard dropped: {e}"))
    })?;
    let mut got = vec![0u8; expected.len()];
    dst.copy_to_host(&mut got)
        .map_err(|e| crate::Error::Msg(format!("readback failed: {e}")))?;
    assert_eq!(got, expected);
    Ok(())
}
