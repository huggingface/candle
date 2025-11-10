#![cfg(feature = "cuda")]

use candle_core::{utils, Device, Result, Tensor};

#[test]
fn cuda_pinned_to_gpu_roundtrip() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let pinned = Device::new_cuda_pinned(0)?;
    let cuda = Device::new_cuda(0)?;
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();

    let tensor_pinned = Tensor::new(&data, &pinned)?;
    assert!(matches!(tensor_pinned.device(), Device::CudaPinned(_)));

    let tensor_cuda = tensor_pinned.to_device(&cuda)?;
    assert!(tensor_cuda.device().is_cuda());

    let tensor_back = tensor_cuda.to_device(&pinned)?;
    let roundtrip = tensor_back.to_vec1::<f32>()?;
    assert_eq!(roundtrip, data);

    Ok(())
}
