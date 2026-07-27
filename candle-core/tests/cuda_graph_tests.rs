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

const MANY_ALLOCATIONS_LAYERS: usize = 8;

// Runs `x` through several "layers", each allocating and immediately
// consuming a handful of intermediates (an identity transform in value, to
// keep the numbers bounded) instead of the single intermediate the other
// tests in this file use. Called from inside a capture, each intermediate
// becomes its own graph-owned memory-alloc/free node pair, so this exercises
// the driver's per-device graph memory pool across many nodes instead of one.
fn identity_through_many_temporaries(x: &Tensor, ones: &Tensor) -> Result<Tensor> {
    let mut acc = x.affine(1.0, 0.0)?;
    for _ in 0..MANY_ALLOCATIONS_LAYERS {
        let zeros_layer = Tensor::zeros((4, 4), DType::F32, x.device())?;
        let a = (&acc * ones)?;
        let b = (&a + &zeros_layer)?;
        acc = b.affine(1.0, 0.0)?;
    }
    Ok(acc)
}

// Regression test: `second_independent_capture_after_first_graph_dropped`
// passing was not enough to fix Tachyon-Mesh's real production scenario
// (astorise/candle#17, reopened) -- a multi-layer `Llama::forward()` capture,
// where many intermediates per layer are allocated inside the closure. Their
// working hypothesis was that this exercises the driver's per-device graph
// memory pool (reconciling many stream-ordered allocations on graph
// destruction) rather than just the per-tensor event-tracking guard. This
// test reproduces that shape at the candle-core level: many graph-owned
// allocations per capture, and several new (non-captured) allocations in the
// unrelated op between the two captures, standing in for a RoPE cos/sin
// table build.
#[test]
fn second_independent_capture_with_many_graph_owned_allocations() -> Result<()> {
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
    let ones = Tensor::ones((4, 4), DType::F32, &device)?;

    {
        let lhs = Tensor::arange(0f32, 16f32, &device)?.reshape((4, 4))?;
        let out = Tensor::zeros((4, 4), DType::F32, &device)?;
        let zeros = Tensor::zeros((4, 4), DType::F32, &device)?;

        // Warm-up outside of capture, then reset back to zeros.
        let warm = identity_through_many_temporaries(&lhs, &ones)?;
        shared.slice_set(&warm, 0, 0)?;
        out.slice_set(&warm, 0, 0)?;
        shared.slice_set(&zeros, 0, 0)?;
        out.slice_set(&zeros, 0, 0)?;
        device.synchronize()?;

        let (graph, ()) = CudaGraph::capture(cuda_device, || {
            let result = identity_through_many_temporaries(&lhs, &ones)?;
            shared.slice_set(&result, 0, 0)?;
            out.slice_set(&result, 0, 0)?;
            Ok(())
        })?;
        graph.replay()?;
        device.synchronize()?;
        assert_eq!(out.to_vec2::<f32>()?, lhs.to_vec2::<f32>()?);
        assert_eq!(shared.to_vec2::<f32>()?, lhs.to_vec2::<f32>()?);
        // `graph` (and its many graph-owned allocation nodes) is dropped at
        // the end of this scope, before the second, independent request
        // below runs.
    }

    // A brand new, unrelated request against the same already-loaded model,
    // building several new tensors via a short op chain (standing in for a
    // RoPE cos/sin table) instead of a single `Tensor::zeros`: fresh
    // buffers, no reference to the first request's `out`/`lhs`/graph.
    let positions = Tensor::arange(0f32, 4f32, &device)?.reshape((4, 1))?;
    let freqs = Tensor::arange(1f32, 5f32, &device)?.reshape((1, 4))?;
    let angles = positions.broadcast_mul(&freqs)?;
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    device.synchronize()?;
    let expected_angles: Vec<Vec<f32>> = (0..4)
        .map(|p| (0..4).map(|f| (p as f32) * (f as f32 + 1.0)).collect())
        .collect();
    let expected_cos: Vec<Vec<f32>> = expected_angles
        .iter()
        .map(|row| row.iter().map(|a| a.cos()).collect())
        .collect();
    let expected_sin: Vec<Vec<f32>> = expected_angles
        .iter()
        .map(|row| row.iter().map(|a| a.sin()).collect())
        .collect();
    // The GPU's cos/sin intrinsics aren't required to be bit-identical to the
    // host's, so compare with a tolerance rather than exact equality -- this
    // step only needs to prove the allocation chain ran and produced sane
    // values, not to pin down transcendental-function precision.
    let close = |actual: &[Vec<f32>], expected: &[Vec<f32>]| {
        actual
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .all(|(a, e)| (a - e).abs() < 1e-4)
    };
    let cos_actual = cos.to_vec2::<f32>()?;
    let sin_actual = sin.to_vec2::<f32>()?;
    assert!(
        close(&cos_actual, &expected_cos),
        "cos mismatch: {cos_actual:?} vs {expected_cos:?}"
    );
    assert!(
        close(&sin_actual, &expected_sin),
        "sin mismatch: {sin_actual:?} vs {expected_sin:?}"
    );

    // The second request then runs its own independent capture/replay cycle
    // against the same device, reusing the shared model-level buffer, again
    // through the many-graph-owned-allocations closure.
    let lhs2 = Tensor::arange(16f32, 32f32, &device)?.reshape((4, 4))?;
    let out2 = Tensor::zeros((4, 4), DType::F32, &device)?;
    let zeros = Tensor::zeros((4, 4), DType::F32, &device)?;

    let warm = identity_through_many_temporaries(&lhs2, &ones)?;
    shared.slice_set(&warm, 0, 0)?;
    out2.slice_set(&warm, 0, 0)?;
    shared.slice_set(&zeros, 0, 0)?;
    out2.slice_set(&zeros, 0, 0)?;
    device.synchronize()?;

    let (graph2, ()) = CudaGraph::capture(cuda_device, || {
        let result = identity_through_many_temporaries(&lhs2, &ones)?;
        shared.slice_set(&result, 0, 0)?;
        out2.slice_set(&result, 0, 0)?;
        Ok(())
    })?;
    graph2.replay()?;
    device.synchronize()?;
    assert_eq!(out2.to_vec2::<f32>()?, lhs2.to_vec2::<f32>()?);
    assert_eq!(shared.to_vec2::<f32>()?, lhs2.to_vec2::<f32>()?);
    Ok(())
}
