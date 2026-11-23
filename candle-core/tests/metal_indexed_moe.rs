#![cfg(feature = "metal")]

use candle_core::{
    quantized::{GgmlDType, QStorage, QTensor},
    DType, Device, Result, Tensor,
};

const ABS_TOLERANCE: f64 = 1e-4;
const WEIGHT_ROUNDING_TOLERANCE: f64 = 1e-3;
const LARGE_TOKEN_COUNT: usize = u16::MAX as usize + 2;

#[derive(Clone, Copy)]
enum Routing {
    Balanced,
    Repeated,
    Invalid,
}

#[derive(Clone, Copy)]
struct Case {
    tokens: usize,
    experts: usize,
    topk: usize,
    input_slots: usize,
    outputs: usize,
    inputs: usize,
    offset_views: bool,
    routing: Routing,
}

fn packed_weight(
    device: &Device,
    shape: impl Into<candle_core::Shape>,
    dtype: GgmlDType,
) -> Result<(QTensor, Vec<f32>)> {
    let shape = shape.into();
    let data: Vec<f32> = (0..shape.elem_count())
        .map(|i| (i as f32 * 0.173).sin() * 0.4 + (i as f32 * 0.037).cos() * 0.2)
        .collect();
    let weight = QTensor::quantize(&Tensor::from_vec(data, shape.clone(), &Device::Cpu)?, dtype)?;
    let reference = weight
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let storage = QStorage::from_data(weight.data()?, device, dtype)?;
    Ok((QTensor::new(storage, shape)?, reference))
}

fn check_case(device: &Device, dtype: GgmlDType, case: Case) -> Result<()> {
    let (weight, reference_weight) =
        packed_weight(device, (case.experts, case.outputs, case.inputs), dtype)?;
    let input_data: Vec<f32> = (0..case.tokens * case.input_slots * case.inputs)
        .map(|i| (i as f32 * 0.127 + 0.3).cos() * 0.6 + (i as f32 * 0.019).sin() * 0.3)
        .collect();
    let ids_data: Vec<u32> = (0..case.tokens * case.topk)
        .map(|i| match case.routing {
            Routing::Balanced => ((i * 3 + 2 * (i / case.topk)) % case.experts) as u32,
            Routing::Repeated => 0,
            Routing::Invalid => match i % 4 {
                0 => 0,
                1 => case.experts as u32,
                2 => u32::MAX,
                _ => (case.experts - 1) as u32,
            },
        })
        .collect();
    let (input, ids) = if case.offset_views {
        let input_row = case.input_slots * case.inputs;
        let mut padded_input = vec![7f32; (case.tokens + 2) * input_row];
        padded_input[input_row..input_row + input_data.len()].copy_from_slice(&input_data);
        let mut padded_ids = vec![u32::MAX; (case.tokens + 2) * case.topk];
        padded_ids[case.topk..case.topk + ids_data.len()].copy_from_slice(&ids_data);
        (
            Tensor::from_vec(
                padded_input,
                (case.tokens + 2, case.input_slots, case.inputs),
                device,
            )?
            .narrow(0, 1, case.tokens)?,
            Tensor::from_vec(padded_ids, (case.tokens + 2, case.topk), device)?.narrow(
                0,
                1,
                case.tokens,
            )?,
        )
    } else {
        (
            Tensor::from_slice(
                &input_data,
                (case.tokens, case.input_slots, case.inputs),
                device,
            )?,
            Tensor::from_slice(&ids_data, (case.tokens, case.topk), device)?,
        )
    };
    let output = weight.indexed_moe_forward(&input, &ids)?;
    assert_eq!(output.dims(), &[case.tokens, case.topk, case.outputs]);
    assert_eq!(output.dtype(), DType::F32);
    let output = output.flatten_all()?.to_vec1::<f32>()?;
    for token in 0..case.tokens {
        for slot in 0..case.topk {
            let expert = ids_data[token * case.topk + slot] as usize;
            for row in 0..case.outputs {
                let actual = output[(token * case.topk + slot) * case.outputs + row];
                if expert >= case.experts {
                    assert_eq!(
                        actual, 0.,
                        "invalid expert {expert}, token {token}, slot {slot}"
                    );
                    continue;
                }
                let input_start =
                    (token * case.input_slots + slot % case.input_slots) * case.inputs;
                let weight_start = (expert * case.outputs + row) * case.inputs;
                let mut expected = 0f64;
                let mut magnitude = 0f64;
                for k in 0..case.inputs {
                    let product = f64::from(input_data[input_start + k])
                        * f64::from(reference_weight[weight_start + k]);
                    expected += product;
                    magnitude += product.abs();
                }
                // Matrix kernels round unpacked weights to f16 before f32 accumulation.
                let tolerance = ABS_TOLERANCE + WEIGHT_ROUNDING_TOLERANCE * magnitude;
                assert!(
                    actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
                    "{dtype:?}: token {token}, slot {slot}, row {row}: {actual} != {expected} (tolerance {tolerance})"
                );
            }
        }
    }
    if matches!(case.routing, Routing::Repeated) {
        let repeated = weight
            .indexed_moe_forward(&input, &ids)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(output, repeated, "routing must be deterministic");
    }
    Ok(())
}

#[test]
fn indexed_moe_formats_and_ragged_outputs_match_cpu() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::F16,
        GgmlDType::F32,
    ] {
        for (tokens, outputs, input_slots, inputs) in
            [(1, 1, 1, 256), (8, 64, 3, 512), (9, 65, 1, 768)]
        {
            check_case(
                &device,
                dtype,
                Case {
                    tokens,
                    experts: 5,
                    topk: 3,
                    input_slots,
                    outputs,
                    inputs,
                    offset_views: false,
                    routing: Routing::Balanced,
                },
            )?;
        }
    }
    Ok(())
}

#[test]
fn indexed_moe_preserves_repeated_routes_offsets_and_invalid_rows() -> Result<()> {
    let device = Device::new_metal(0)?;
    let base = Case {
        tokens: 33,
        experts: 3,
        topk: 4,
        input_slots: 4,
        outputs: 65,
        inputs: 256,
        offset_views: true,
        routing: Routing::Repeated,
    };
    check_case(&device, GgmlDType::Q4K, base)?;
    for outputs in [64, 65] {
        check_case(
            &device,
            GgmlDType::Q6K,
            Case {
                tokens: 1,
                outputs,
                routing: Routing::Invalid,
                ..base
            },
        )?;
    }
    Ok(())
}

#[test]
fn indexed_moe_preserves_token_indices_above_u16() -> Result<()> {
    check_case(
        &Device::new_metal(0)?,
        GgmlDType::Q4_0,
        Case {
            tokens: LARGE_TOKEN_COUNT,
            experts: 2,
            topk: 1,
            input_slots: 1,
            outputs: 1,
            inputs: 32,
            offset_views: false,
            routing: Routing::Balanced,
        },
    )
}

#[test]
fn indexed_moe_empty_routes_return_empty_outputs() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (weight, _) = packed_weight(&device, (3, 5, 256), GgmlDType::Q4K)?;
    for (tokens, topk) in [(0, 2), (3, 0)] {
        let input = Tensor::zeros((tokens, 1, 256), DType::F32, &device)?;
        let ids = Tensor::zeros((tokens, topk), DType::U32, &device)?;
        let output = weight.indexed_moe_forward(&input, &ids)?;
        assert_eq!(output.dims(), &[tokens, topk, 5]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.elem_count(), 0);
    }
    Ok(())
}

#[test]
fn indexed_moe_rejects_unsupported_inputs() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (weight, _) = packed_weight(&device, (3, 5, 256), GgmlDType::Q4K)?;
    let input = Tensor::zeros((2, 1, 256), DType::F32, &device)?;
    let ids = Tensor::zeros((2, 2), DType::U32, &device)?;
    for (bad_input, bad_ids) in [
        (input.to_dtype(DType::F16)?, ids.clone()),
        (input.clone(), ids.to_dtype(DType::I64)?),
        (
            Tensor::zeros((2, 256, 2), DType::F32, &device)?.transpose(1, 2)?,
            ids.clone(),
        ),
        (input.clone(), ids.transpose(0, 1)?),
    ] {
        assert!(weight.indexed_moe_forward(&bad_input, &bad_ids).is_err());
    }
    let (dense_tail, _) = packed_weight(&device, (3, 5, 33), GgmlDType::F32)?;
    let tail_input = Tensor::zeros((2, 1, 33), DType::F32, &device)?;
    assert!(dense_tail.indexed_moe_forward(&tail_input, &ids).is_err());
    let (unsupported, _) = packed_weight(&device, (3, 5, 256), GgmlDType::BF16)?;
    assert!(unsupported.indexed_moe_forward(&input, &ids).is_err());
    let (small_weight, _) = packed_weight(&Device::Cpu, (1, 1, 256), GgmlDType::Q4K)?;
    for (shape, message) in [
        ((3, 5, 256), "weight shape exceeds its storage"),
        (
            (u32::MAX as usize, 5, 256),
            "dimensions exceed the kernel index range",
        ),
    ] {
        let storage = QStorage::from_data(small_weight.data()?, &device, GgmlDType::Q4K)?;
        let malformed = QTensor::new(storage, shape)?;
        let error = malformed.indexed_moe_forward(&input, &ids).unwrap_err();
        assert!(error.to_string().contains(message), "{shape:?}: {error}");
    }
    Ok(())
}
