#![cfg(feature = "cuda")]

// GLOBAL capture restricts CUDA calls from other threads in the same process.
// Keep this test in its own integration-test binary so ordinary CUDA tests can
// run in parallel without interfering with capture.
use candle_core::{DType, Device, Result, Tensor};

#[test]
fn cumsum_cuda_low_precision_graph_replay() -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use candle_core::scalar::Scalar;
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    // All graph work and input updates use this one stream. Event tracking
    // would introduce dependencies on events outside the capture.
    unsafe { cuda.disable_event_tracking() };
    for dtype in [DType::F16, DType::BF16] {
        for rows in [1, 3] {
            for len in [4097, 4128, 4129, 8193] {
                let _cache = cuda.enable_cuda_graph_htod_cache();
                let input = Tensor::ones((rows, len + 2), dtype, &device)?;
                let compute = || input.narrow(1, 1, len)?.t()?.contiguous()?.cumsum(0)?.t();
                drop(compute()?);
                device.synchronize()?;
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
                    .map_err(candle_core::Error::wrap)?;
                let output = compute()?;
                let graph = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(candle_core::Error::wrap)?
                .expect("captured graph");
                for iteration in 0..4 {
                    let scalar = if iteration % 2 == 0 {
                        Scalar::one(dtype)
                    } else {
                        Scalar::zero(dtype)
                    };
                    input.const_set(scalar)?;
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    device.synchronize()?;
                    let expected = Tensor::arange(1f32, (len + 1) as f32, &Device::Cpu)?
                        .affine(if iteration % 2 == 0 { 1. } else { 0. }, 0.)?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        output.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                        vec![expected; rows]
                    );
                }
            }
        }
    }
    Ok(())
}
