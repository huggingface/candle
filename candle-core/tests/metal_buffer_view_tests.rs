#![cfg(all(target_os = "macos", feature = "metal"))]

use candle_core::{Device, Result, Tensor};
use candle_metal_kernels::metal::Device as MetalGpuDevice;

#[test]
fn metal_buffer_view_stays_valid_while_guard_alive() -> Result<()> {
    if MetalGpuDevice::system_default().is_none() {
        eprintln!("skipping Metal buffer view test: no Metal device detected");
        return Ok(());
    }
    let device = Device::new_metal(0)?;
    let tensor = Tensor::arange(0f32, 16f32, &device)?.reshape((4, 4))?;
    let view = tensor
        .as_metal_buffer_view()
        .expect("metal tensor should expose its buffer");
    assert_eq!(view.byte_len, 16 * std::mem::size_of::<f32>() as u64);

    assert!(view.buffer.length() as u64 >= view.byte_len);

    // While the guard on the tensor storage is alive we can safely read from the Metal buffer.
    let contents = view.buffer.contents() as *const f32;
    assert!(!contents.is_null());
    unsafe {
        assert_eq!(*contents, 0.0);
        assert_eq!(*contents.add(15), 15.0);
    }

    Ok(())
}
