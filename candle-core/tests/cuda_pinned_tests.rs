#![cfg(feature = "cuda")]

use candle_core::{utils, Device, Tensor};

type Result<T> = candle_core::Result<T>;

#[test]
fn cuda_pinned_tensor_roundtrip() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let len = 16_usize;

    let mut pinned_in = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    let expected: Vec<f32> = (0..len).map(|v| v as f32).collect();
    pinned_in
        .as_mut_slice()
        .expect("pinned slice")
        .copy_from_slice(&expected);

    let tensor = Tensor::from_pinned_host(&pinned_in, len)?;
    // Move to CUDA for the roundtrip test (copy_to_pinned_host requires CUDA tensor)
    let tensor = tensor.to_device(&device)?;
    drop(pinned_in);

    // Run a simple tensor op to ensure usability within the tensor API.
    let sum = tensor.sum_all()?.to_scalar::<f32>()?;
    let expected_sum: f32 = expected.iter().copied().sum();
    assert!((sum - expected_sum).abs() < 1e-4);

    let mut pinned_out = unsafe { device.cuda_alloc_pinned::<f32>(len)? };
    tensor.copy_to_pinned_host(&mut pinned_out)?;
    let host = pinned_out.as_slice().expect("pinned slice");
    assert_eq!(host, expected.as_slice());

    Ok(())
}
