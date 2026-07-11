#![cfg(feature = "cuda")]
use candle_core::{CudaGraph, DType, Device, Result, Tensor};

// Captures a small "decode step" (an elementwise multiply written into a
// persistent output buffer) and checks that replaying the graph reproduces it.
//
// The output buffer `out` is allocated *before* capture and written in place
// with `slice_set`, so the captured graph only records kernel launches and a
// device-to-device copy into already-existing memory. Allocating the result
// *inside* capture would instead make it a graph-owned allocation node whose
// memory is only valid while the graph runs, which is not what a decode loop
// wants (and cannot be read back between replays).
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
    let rhs = Tensor::ones((4, 4), DType::F32, &device)?;
    let zeros = Tensor::zeros((4, 4), DType::F32, &device)?;
    let out = Tensor::zeros((4, 4), DType::F32, &device)?;

    // Warm-up run outside of capture: JIT-loads the multiply and copy kernels so
    // that capture only ever records already-loaded launches (see
    // CudaGraph::capture), then reset `out` back to zeros.
    let tmp = (&lhs * &rhs)?;
    out.slice_set(&tmp, 0, 0)?;
    out.slice_set(&zeros, 0, 0)?;
    device.synchronize()?;

    // Before replay the captured work has not run, so `out` is still zeros.
    let before = out.to_vec2::<f32>()?;
    assert_eq!(before, vec![vec![0f32; 4]; 4]);

    let (graph, ()) = CudaGraph::capture(cuda_device, || {
        let tmp = (&lhs * &rhs)?;
        out.slice_set(&tmp, 0, 0)?;
        Ok(())
    })?;

    // Capture records but does not execute, so `out` is unchanged until replay.
    assert_eq!(out.to_vec2::<f32>()?, vec![vec![0f32; 4]; 4]);

    graph.replay()?;
    device.synchronize()?;
    let after = out.to_vec2::<f32>()?;
    assert_eq!(after, lhs.to_vec2::<f32>()?);

    // Replaying again is idempotent for this graph.
    graph.replay()?;
    device.synchronize()?;
    assert_eq!(out.to_vec2::<f32>()?, lhs.to_vec2::<f32>()?);
    Ok(())
}

// Regression test: on real hardware, a second, independent `CudaGraph::capture`
// on the same device broke a later, completely unrelated allocation with
// `CUDA_ERROR_INVALID_VALUE` -- the very first op of the second capture's
// caller, before any of its own capture/replay code ran. Reproducing it needs
// a tensor allocated before either capture and written into by both (standing
// in for a model-level KV cache reused across independent requests), plus a
// plain, unrelated allocation after the first graph is dropped.
#[test]
fn second_independent_capture_after_first_graph_dropped() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => return Ok(()),
    };
    let Device::Cuda(cuda_device) = &device else {
        unreachable!()
    };

    // Stands in for a model-level buffer (e.g. a paged-attention KV cache)
    // allocated once, before either capture, and written into by both.
    let shared = Tensor::zeros((4, 4), DType::F32, &device)?;
    let rhs = Tensor::ones((4, 4), DType::F32, &device)?;

    {
        let lhs = Tensor::arange(0f32, 16f32, &device)?.reshape((4, 4))?;
        let out = Tensor::zeros((4, 4), DType::F32, &device)?;
        let zeros = Tensor::zeros((4, 4), DType::F32, &device)?;

        // Warm-up outside of capture, then reset back to zeros.
        let tmp = (&lhs * &rhs)?;
        shared.slice_set(&tmp, 0, 0)?;
        out.slice_set(&tmp, 0, 0)?;
        shared.slice_set(&zeros, 0, 0)?;
        out.slice_set(&zeros, 0, 0)?;
        device.synchronize()?;

        let (graph, ()) = CudaGraph::capture(cuda_device, || {
            let tmp = (&lhs * &rhs)?;
            shared.slice_set(&tmp, 0, 0)?;
            out.slice_set(&tmp, 0, 0)?;
            Ok(())
        })?;
        graph.replay()?;
        graph.replay()?;
        device.synchronize()?;
        assert_eq!(out.to_vec2::<f32>()?, lhs.to_vec2::<f32>()?);
        assert_eq!(shared.to_vec2::<f32>()?, lhs.to_vec2::<f32>()?);
        // `graph` is dropped at the end of this scope, before the second,
        // independent request below runs -- matching the production sequence
        // (capture, replay, request completes, graph and its per-request
        // buffers are dropped) that preceded the real-hardware failure.
    }

    // A brand new, unrelated request against the same already-loaded model:
    // fresh buffer, no reference to the first request's `out`/`lhs`/graph.
    // This is the operation that failed on real hardware, before any of the
    // second request's own capture/replay code ran.
    let unrelated = Tensor::zeros((4, 4), DType::F32, &device)?;
    device.synchronize()?;
    assert_eq!(unrelated.to_vec2::<f32>()?, vec![vec![0f32; 4]; 4]);

    // The second request then runs its own independent capture/replay cycle
    // against the same device, reusing the shared model-level buffer.
    let lhs2 = Tensor::arange(16f32, 32f32, &device)?.reshape((4, 4))?;
    let out2 = Tensor::zeros((4, 4), DType::F32, &device)?;
    let zeros = Tensor::zeros((4, 4), DType::F32, &device)?;

    let tmp = (&lhs2 * &rhs)?;
    shared.slice_set(&tmp, 0, 0)?;
    out2.slice_set(&tmp, 0, 0)?;
    shared.slice_set(&zeros, 0, 0)?;
    out2.slice_set(&zeros, 0, 0)?;
    device.synchronize()?;

    let (graph2, ()) = CudaGraph::capture(cuda_device, || {
        let tmp = (&lhs2 * &rhs)?;
        shared.slice_set(&tmp, 0, 0)?;
        out2.slice_set(&tmp, 0, 0)?;
        Ok(())
    })?;
    graph2.replay()?;
    device.synchronize()?;
    assert_eq!(out2.to_vec2::<f32>()?, lhs2.to_vec2::<f32>()?);
    assert_eq!(shared.to_vec2::<f32>()?, lhs2.to_vec2::<f32>()?);
    Ok(())
}
