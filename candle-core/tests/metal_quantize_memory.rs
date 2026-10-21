//! Quantizing onto metal allocates the quantized buffer and nothing else.
//!
//! `QTensor::quantize_onto` used to allocate a zero-filled buffer the size
//! of the quantized weights (rounded up to a power of two by the buffer pool) before quantizing
//! on the cpu and uploading the result into a second buffer. The first buffer stayed in the
//! device's pool until the next synchronize, so loading a model held about twice its weights.
//!
//! One test per file, so that no other test allocates on the device while this one measures.

#[cfg(feature = "metal")]
mod metal {
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Result, Tensor};

    fn allocated(dev: &Device) -> Result<usize> {
        match dev {
            Device::Metal(m) => Ok(m.metal_device().current_allocated_size()),
            _ => candle_core::bail!("not a metal device"),
        }
    }

    /// The device's allocation grows by about the quantized size, not by that plus a zeroed
    /// buffer of the same size.
    fn check(name: &str, dev: &Device, quantize: impl Fn() -> Result<QTensor>) -> Result<()> {
        let before = allocated(dev)?;
        let q = quantize()?;
        let grown = allocated(dev)?.saturating_sub(before);
        let size = q.storage_size_in_bytes();
        assert!(
            grown < size + size / 2,
            "{name}: the device allocated {grown} bytes for a {size}-byte quantized tensor"
        );
        Ok(())
    }

    #[test]
    fn quantizing_onto_metal_allocates_only_the_quantized_buffer() -> Result<()> {
        let dev = Device::new_metal(0)?;
        // 4 Mi f32 values: 4.25 MiB as Q8_0, 2.25 MiB as Q4K.
        let src = Tensor::randn(0f32, 1., (1024, 4096), &Device::Cpu)?;
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K] {
            check(&format!("quantize_onto {dtype:?}"), &dev, || {
                QTensor::quantize_onto(&src, dtype, &dev)
            })?;
        }
        Ok(())
    }
}
