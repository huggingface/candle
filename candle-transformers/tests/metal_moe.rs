#![cfg(feature = "metal")]

use candle::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QTensor};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Linear};
use candle_transformers::fused_moe::FusedMoeGGUF;
use std::sync::Arc;

const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 512;
const ABS_TOLERANCE: f32 = 5e-4;
const REL_TOLERANCE: f32 = 2e-3;

fn values(len: usize, mut seed: u64, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 32) as u32 as f32 / u32::MAX as f32 - 0.5) * scale
        })
        .collect()
}

fn expert_weights(
    shape: (usize, usize, usize),
    format: GgmlDType,
    seed: u64,
    device: &Device,
) -> Result<(Arc<QTensor>, Tensor)> {
    let source = Tensor::from_vec(
        values(
            shape.0 * shape.1 * shape.2,
            seed,
            2.0 / (shape.2 as f32).sqrt(),
        ),
        shape,
        &Device::Cpu,
    )?;
    let quantized = QTensor::quantize(&source, format)?;
    let reference = quantized.dequantize(&Device::Cpu)?;
    let uploaded = qtensor_from_ggml(
        format,
        quantized.data()?.as_ref(),
        quantized.shape().dims().to_vec(),
        device,
    )?;
    Ok((Arc::new(uploaded), reference))
}

fn reference(
    xs: &Tensor,
    router: &Tensor,
    experts: &[Tensor; 3],
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let xs = xs
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten(0, 1)?;
    let logits = xs.matmul(&router.t()?)?.to_vec2::<f32>()?;
    let mut output = vec![vec![0f32; HIDDEN]; xs.dim(0)?];
    let mut routes = Vec::with_capacity(logits.len());
    for row in logits {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities = row.iter().map(|x| (x - maximum).exp()).collect::<Vec<_>>();
        let mut ids = (0..EXPERTS).collect::<Vec<_>>();
        ids.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        ids.truncate(TOP_K);
        let denominator = if normalize {
            ids.iter().map(|&id| probabilities[id]).sum::<f32>()
        } else {
            probabilities.iter().sum::<f32>()
        };
        routes.push(
            ids.into_iter()
                .map(|id| (id, probabilities[id] / denominator))
                .collect::<Vec<_>>(),
        );
    }
    for expert in 0..EXPERTS {
        let gate = xs.matmul(&experts[0].i(expert)?.t()?)?;
        let up = xs.matmul(&experts[1].i(expert)?.t()?)?;
        let down = (candle_nn::ops::silu(&gate)? * up)?
            .matmul(&experts[2].i(expert)?.t()?)?
            .to_vec2::<f32>()?;
        for (token, routes) in routes.iter().enumerate() {
            if let Some((_, weight)) = routes.iter().find(|(id, _)| *id == expert) {
                for (dst, value) in output[token].iter_mut().zip(&down[token]) {
                    *dst += weight * value;
                }
            }
        }
    }
    Ok(output)
}

#[test]
fn mixed_quantized_expert_layer_matches_dense_cpu() -> Result<()> {
    let device = Device::new_metal(0)?;
    let router = Tensor::from_vec(
        values(EXPERTS * HIDDEN, 7, 2.0 / (HIDDEN as f32).sqrt()),
        (EXPERTS, HIDDEN),
        &Device::Cpu,
    )?;
    let (gate_experts, gate_ref) =
        expert_weights((EXPERTS, INTERMEDIATE, HIDDEN), GgmlDType::Q4K, 11, &device)?;
    let (up_experts, up_ref) = expert_weights(
        (EXPERTS, INTERMEDIATE, HIDDEN),
        GgmlDType::Q8_0,
        13,
        &device,
    )?;
    let (down_experts, down_ref) =
        expert_weights((EXPERTS, HIDDEN, INTERMEDIATE), GgmlDType::Q6K, 17, &device)?;
    let experts = [gate_ref, up_ref, down_ref];
    let mut model = FusedMoeGGUF {
        gate: Linear::new(router.to_device(&device)?, None),
        gate_experts,
        up_experts,
        down_experts,
        act: Activation::Silu,
        norm_topk_prob: true,
        num_experts_per_tok: TOP_K,
        dtype: DType::F16,
    };
    for (tokens, dtype, normalize) in [
        (1, DType::F32, false),
        (1, DType::F16, true),
        (33, DType::F32, true),
        (33, DType::F16, false),
    ] {
        model.norm_topk_prob = normalize;
        let input = Tensor::from_vec(
            values(tokens * HIDDEN, 19, 2.0),
            (1, tokens, HIDDEN),
            &device,
        )?
        .to_dtype(dtype)?;
        let expected = reference(&input, &router, &experts, normalize)?;
        let actual = model.forward(&input, tokens > 1)?;
        assert_eq!(actual.dims(), [1, tokens, HIDDEN]);
        assert_eq!(actual.dtype(), dtype);
        let actual = actual.to_dtype(DType::F32)?.squeeze(0)?.to_vec2::<f32>()?;
        for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                let tolerance = ABS_TOLERANCE + REL_TOLERANCE * expected.abs();
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                    "tokens={tokens} normalize={normalize} dtype={dtype:?} \
                     [{row},{column}]: {actual} != {expected} (tolerance {tolerance})"
                );
            }
        }
    }
    Ok(())
}
