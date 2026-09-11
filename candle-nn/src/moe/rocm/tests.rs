//! Hardware correctness tests for [`super::moe_gemm_gguf`].
//!
//! The reference is [`reference`] below: `moe_gemm_gguf`'s documented contract
//! transcribed literally as a host-side loop over the dequantized experts. It
//! shares no code with the implementation — in particular it consumes
//! `sorted_token_ids` / `experts_ids` directly rather than inverting the
//! permutation — so a mistake in [`super::unsort`] cannot cancel out.

use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, Device, Result, Tensor};

/// `Device::new_rocm` fails on machines without a GPU; those runs skip.
macro_rules! rocm_device {
    () => {
        match Device::new_rocm(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

fn ramp(len: usize, div: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32 / div).sin()).collect()
}

/// `moe_gemm_gguf`'s contract, on the CPU, from dequantized experts:
///
/// ```text
/// t = sorted_token_ids[m]
/// out[t] = W[experts_ids[m]] @ in[topk_weights ? t : t / topk] * scale(t)
/// ```
#[allow(clippy::too_many_arguments)]
fn reference(
    w: &Tensor, // (num_experts, n, k), dequantized, on the CPU
    input: &Tensor,
    topk_weights: &Option<Vec<f32>>,
    sorted_token_ids: &[u32],
    experts_ids: &[u32],
    topk: usize,
    size_m: usize,
    n: usize,
) -> Result<Tensor> {
    let mut out = vec![0f32; size_m * n];
    for m in 0..size_m {
        let t = sorted_token_ids[m] as usize;
        let expert = experts_ids[m] as usize;
        let input_row = match topk_weights {
            Some(_) => t,
            None => t / topk,
        };
        let scale = match topk_weights {
            Some(tw) => tw[t],
            None => 1.0,
        };
        let row = w
            .get(expert)?
            .matmul(&input.get(input_row)?.unsqueeze(1)?)?
            .squeeze(1)?
            .to_vec1::<f32>()?;
        for (dst, v) in out[t * n..(t + 1) * n].iter_mut().zip(row) {
            *dst = v * scale;
        }
    }
    Tensor::from_vec(out, (size_m, n), &Device::Cpu)
}

/// One shape and routing, checked end to end.
///
/// `weighted` picks the two calling conventions apart: the gate/up projections
/// pass no `topk_weights` and one input row per *token*, the down projection
/// passes them and one input row per *routed pair*.
#[allow(clippy::too_many_arguments)]
fn check(
    device: &Device,
    dtype: GgmlDType,
    num_experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    topk: usize,
    weighted: bool,
) -> Result<()> {
    let size_m = tokens * topk;
    let rows = if weighted { size_m } else { tokens };

    let w = Tensor::from_vec(
        ramp(num_experts * n * k, 61.),
        (num_experts, n, k),
        &Device::Cpu,
    )?;
    let dequantized = QTensor::quantize(&w, dtype)?.dequantize(&Device::Cpu)?;
    let qw = QTensor::quantize(&w.to_device(device)?, dtype)?;

    let input_data = ramp(rows * k, 43.);
    let input_cpu = Tensor::from_vec(input_data.clone(), (rows, k), &Device::Cpu)?;
    let input = Tensor::from_vec(input_data, (rows, k), device)?;

    // Exactly the shape the caller produces: a routing table per token, flattened
    // and sorted by expert, `sorted_token_ids` being the permutation that did it.
    let routing: Vec<u32> = (0..size_m)
        .map(|i| ((i * 5 + 3) % num_experts) as u32)
        .collect();
    let routing_t = Tensor::from_vec(routing, (size_m,), device)?;
    let (experts_ids, sorted_token_ids) = routing_t.sort_last_dim(true)?;
    let experts_host = experts_ids.to_vec1::<u32>()?;
    let sorted_host = sorted_token_ids.to_vec1::<u32>()?;

    let topk_weights_host: Option<Vec<f32>> = if weighted {
        Some((0..size_m).map(|i| 0.25 + (i % 4) as f32 * 0.1).collect())
    } else {
        None
    };
    let topk_weights = match &topk_weights_host {
        Some(v) => Some(Tensor::from_vec(v.clone(), (tokens, topk), device)?),
        None => None,
    };

    let got = super::moe_gemm_gguf(
        &input,
        &qw,
        &topk_weights,
        &sorted_token_ids,
        &experts_ids,
        topk,
    )?;
    assert_eq!(got.dims(), [size_m, n], "{dtype:?}");
    let got = got.flatten_all()?.to_vec1::<f32>()?;

    let want = reference(
        &dequantized,
        &input_cpu,
        &topk_weights_host,
        &sorted_host,
        &experts_host,
        topk,
        size_m,
        n,
    )?
    .flatten_all()?
    .to_vec1::<f32>()?;

    // Same reasoning as the core MMVQ tests: the kernel requantizes the
    // activations to `q8_1`, so the residual is set by the row's scale rather
    // than by each element's own magnitude.
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 0.02 * scale,
            "{dtype:?} weighted={weighted} experts={num_experts} n={n} k={k} \
             tokens={tokens} topk={topk}: element {i} is {g}, reference {w}"
        );
    }
    Ok(())
}

const DTYPES: [GgmlDType; 3] = [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0];

/// The gate/up convention: no routing weights, one input row per token.
#[test]
fn moe_gemm_gguf_unweighted_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in DTYPES {
        check(&device, dtype, 4, 96, 256, 5, 2, false)?;
    }
    Ok(())
}

/// The down convention: routing weights applied, one input row per routed pair.
#[test]
fn moe_gemm_gguf_weighted_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in DTYPES {
        check(&device, dtype, 4, 96, 256, 5, 2, true)?;
    }
    Ok(())
}

/// A single decode token, and a `topk` larger than the batch — the case where
/// getting the `(batch, topk)` flattening backwards still produces the right
/// output *shape*.
#[test]
fn moe_gemm_gguf_single_token_matches_the_cpu_rocm() -> Result<()> {
    let device = rocm_device!();
    for dtype in DTYPES {
        check(&device, dtype, 8, 64, 512, 1, 4, false)?;
        check(&device, dtype, 8, 64, 512, 1, 4, true)?;
    }
    Ok(())
}

/// The routing arrives sorted by expert, so the permutation is non-trivial;
/// this pins that `unsort` actually inverts it rather than being a no-op.
#[test]
fn moe_gemm_gguf_inverts_a_nontrivial_permutation_rocm() -> Result<()> {
    let device = rocm_device!();
    let routing = Tensor::from_vec(vec![3u32, 0, 2, 1], (4,), &device)?;
    let (experts, sorted) = routing.sort_last_dim(true)?;
    assert_eq!(sorted.to_vec1::<u32>()?, [1, 3, 2, 0]);
    let back = super::unsort(&sorted, &experts)?;
    assert_eq!(back.to_vec1::<u32>()?, [3, 0, 2, 1]);
    Ok(())
}

#[test]
fn moe_gemm_gguf_rejects_a_k_mismatch_rocm() -> Result<()> {
    let device = rocm_device!();
    let w = Tensor::zeros((2, 32, 256), DType::F32, &device)?;
    let qw = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let input = Tensor::zeros((2, 128), DType::F32, &device)?;
    let ids = Tensor::from_vec(vec![0u32, 1], (2,), &device)?;
    let err = super::moe_gemm_gguf(&input, &qw, &None, &ids, &ids, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("k=128"), "unexpected error: {err}");
    Ok(())
}
