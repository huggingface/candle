//! Hardware correctness tests for [`super::forward`].
//!
//! The reference is the CPU backend: dequantize the expert stack once, then run
//! the routing with plain tensor ops. That keeps the reference independent of
//! every ROCm code path — it does not even touch the GPU — so a shared bug in
//! the launch geometry cannot hide.

use crate::quantized::{GgmlDType, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{Device, Result, Tensor};

/// `RocmDevice::new` fails on machines without a GPU; those runs skip.
macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Every dtype with an `indexed_moe_forward_*_q8_1` kernel.
const MOE_DTYPES: [GgmlDType; 6] = [
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8_0,
];

/// Deterministic, spread over a couple of octaves so quantization has something
/// to lose. Coprime divisors keep the weight and activation patterns from
/// lining up into an artificially easy dot product.
fn ramp(len: usize, div: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32 / div).sin()).collect()
}

/// The op, on the CPU, from the dequantized experts.
///
/// `out[b][j] = w[ids[b][j]] @ x[b][if input_dim1 == 1 { 0 } else { j }]`.
fn reference(
    w: &Tensor, // (num_experts, n, k), dequantized
    x: &Tensor, // (batch, input_dim1, k)
    ids: &[u32],
    topk: usize,
) -> Result<Tensor> {
    let (batch, input_dim1, _) = x.dims3()?;
    let mut rows = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for j in 0..topk {
            let expert = ids[b * topk + j] as usize;
            let we = w.get(expert)?; // (n, k)
            let xr = x.get(b)?.get(if input_dim1 == 1 { 0 } else { j })?; // (k,)
            rows.push(we.matmul(&xr.unsqueeze(1)?)?.squeeze(1)?);
        }
    }
    Tensor::stack(&rows, 0)?.reshape((batch, topk, ()))
}

/// Run one shape on the GPU and check it against [`reference`].
///
/// The tolerance is relative to the largest magnitude in the reference row
/// rather than per element: the kernel requantizes the activations to `q8_1`,
/// seven mantissa bits, and the residual accumulates over `k` terms, so a dot
/// product that lands near zero has an error set by the row's scale and not by
/// its own value. Same reasoning and same 2% figure as the MMVQ tests.
#[allow(clippy::too_many_arguments)]
fn check(
    device: &Device,
    dtype: GgmlDType,
    num_experts: usize,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize,
    input_dim1: usize,
) -> Result<()> {
    let w = Tensor::from_vec(
        ramp(num_experts * n * k, 61.),
        (num_experts, n, k),
        &Device::Cpu,
    )?;
    // The reference dequantizes the *same* rounding the GPU sees, so the only
    // difference left between the two is the kernel's `q8_1` activations.
    let dequantized = QTensor::quantize(&w, dtype)?.dequantize(&Device::Cpu)?;
    let qw = QTensor::quantize(&w.to_device(device)?, dtype)?;

    let x_data = ramp(batch * input_dim1 * k, 43.);
    let x_cpu = Tensor::from_vec(x_data.clone(), (batch, input_dim1, k), &Device::Cpu)?;
    let x = Tensor::from_vec(x_data, (batch, input_dim1, k), device)?;

    // A spread of experts per token, wrapping so every expert gets used. The
    // `+ 3` matters: routing the first pair to expert 0 makes the expert stride
    // a no-op there, which is exactly how the q8_0 kernel's wrong stride hid.
    let ids: Vec<u32> = (0..batch * topk)
        .map(|i| ((i * 5 + 3) % num_experts) as u32)
        .collect();
    let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), device)?;

    let got = qw.indexed_moe_forward(&x, &ids_t)?;
    assert_eq!(got.dims(), [batch, topk, n], "{dtype:?}");
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let want = reference(&dequantized, &x_cpu, &ids, topk)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 0.02 * scale,
            "{dtype:?} experts={num_experts} n={n} k={k} batch={batch} topk={topk} \
             input_dim1={input_dim1}: element {i} is {g}, reference {w}"
        );
    }
    Ok(())
}

/// One shared activation row per token, i.e. the gate/up projection: the same
/// hidden state goes to all `topk` experts.
#[test]
fn indexed_moe_shared_input_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        check(&device, dtype, 4, 96, 256, 3, 2, 1)?;
    }
    Ok(())
}

/// One activation row per routed pair, i.e. the down projection.
#[test]
fn indexed_moe_per_expert_input_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        check(&device, dtype, 4, 96, 256, 3, 2, 2)?;
    }
    Ok(())
}

/// `k` past one `MATRIX_ROW_PADDING` stride, so a wrong padded row stride in
/// the `q8_1` activation buffer shows up rather than cancelling out.
#[test]
fn indexed_moe_unpadded_k_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        check(&device, dtype, 3, 64, 768, 2, 2, 1)?;
    }
    Ok(())
}

/// A single token routed to a single expert — the smallest launch, and the one
/// where a grid of `(n, 1, 1)` would still pass if the batch/topk flattening
/// were wrong. Paired with a wide expert stack so a stride error moves the
/// result well outside tolerance.
#[test]
fn indexed_moe_single_token_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in MOE_DTYPES {
        check(&device, dtype, 8, 128, 256, 1, 1, 1)?;
    }
    Ok(())
}

#[test]
fn indexed_moe_rejects_a_dtype_without_a_kernel_rocm() -> Result<()> {
    let device = rocm_device!();
    let w = Tensor::zeros((2, 32, 256), crate::DType::F32, &device)?;
    let qw = QTensor::quantize(&w, GgmlDType::Q4_0)?;
    let x = Tensor::zeros((1, 1, 256), crate::DType::F32, &device)?;
    let ids = Tensor::from_vec(vec![0u32], (1, 1), &device)?;
    let err = qw.indexed_moe_forward(&x, &ids).unwrap_err().to_string();
    assert!(err.contains("Q4_0"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn indexed_moe_rejects_a_mismatched_batch_rocm() -> Result<()> {
    let device = rocm_device!();
    let w = Tensor::zeros((2, 32, 256), crate::DType::F32, &device)?;
    let qw = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let x = Tensor::zeros((3, 1, 256), crate::DType::F32, &device)?;
    let ids = Tensor::from_vec(vec![0u32, 1], (2, 1), &device)?;
    let err = qw.indexed_moe_forward(&x, &ids).unwrap_err().to_string();
    assert!(err.contains("batch"), "unexpected error: {err}");
    Ok(())
}
