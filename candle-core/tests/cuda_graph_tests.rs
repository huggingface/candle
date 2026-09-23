#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::sys::{
    CUgraphInstantiate_flags, CUstreamCaptureMode,
};
use candle_core::{Device, Error, Result, Tensor};

#[test]
fn asort_large_cuda_graph() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let data = (0..8193).rev().map(|v| v as f32).collect::<Vec<_>>();
    let tensor = Tensor::from_vec(data, 8193, &device)?;
    tensor.arg_sort_last_dim(true)?;
    device.synchronize()?;

    let stream = device.as_cuda_device()?.cuda_stream();
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(Error::wrap)?;
    let indexes = tensor.arg_sort_last_dim(true);
    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(Error::wrap)?
        .ok_or_else(|| Error::msg("CUDA argsort captured an empty graph"))?;
    let indexes = indexes?;

    for _ in 0..2 {
        graph.launch().map_err(Error::wrap)?;
        device.synchronize()?;
        assert_eq!(
            indexes.to_vec1::<u32>()?,
            (0..8193).rev().map(|v| v as u32).collect::<Vec<_>>()
        );
    }
    Ok(())
}
