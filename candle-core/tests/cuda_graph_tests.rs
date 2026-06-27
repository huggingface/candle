#![cfg(feature = "cuda")]
use candle_core::{CudaGraph, Device, Result, Tensor};

#[test]
fn capture_and_replay_decode_step() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => return Ok(()),
    };
    let Device::Cuda(cuda_device) = &device else {
        unreachable!()
    };

    let lhs = Tensor::arange(0f32, 16f32, &device)?.reshape((4, 4))?;
    let rhs = Tensor::ones((4, 4), candle_core::DType::F32, &device)?;

    // Warm-up run outside of capture: JIT-loads the multiply kernel's module so that
    // capture only ever records already-loaded kernel launches (see CudaGraph::capture).
    let mut out = (&lhs * &rhs)?;
    device.synchronize()?;
    println!("warmup complete, event_tracking={}", cuda_device.is_event_tracking());

    let (graph, ()) = CudaGraph::capture(cuda_device, || {
        println!("closure: before mul");
        let r = &lhs * &rhs;
        println!("closure: mul ok={}", r.is_ok());
        out = r?;
        println!("closure: after assign");
        Ok(())
    })?;
    println!("capture returned ok");

    let before = out.to_vec2::<f32>()?;
    graph.replay()?;
    let after = out.to_vec2::<f32>()?;
    assert_eq!(before, after);
    assert_eq!(after, lhs.to_vec2::<f32>()?);
    Ok(())
}
