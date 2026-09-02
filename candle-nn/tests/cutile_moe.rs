#![cfg(all(feature = "cuda", feature = "cutile"))]

use candle::{DType, Device, Result, Tensor};
use candle_nn::moe::cutile::{
    routed_grouped_matmul, warmup_routed_grouped_matmul, MoeInputMode, MoeRouting,
};
use half::bf16;
use std::sync::Mutex;

const MAX_ABS_DIFF: f32 = 0.02;

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
enum TestInputMode {
    TokenRows,
    RoutedRows,
}

struct Case {
    tokens: usize,
    top_k: usize,
    experts: usize,
    n: usize,
    k: usize,
    input_mode: TestInputMode,
    warmup: bool,
}

fn patterned_bf16(len: usize, salt: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let value = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 101) as f32 - 50.0;
            bf16::from_f32(value * scale / 50.0)
        })
        .collect()
}

fn api_input_mode(input_mode: TestInputMode) -> MoeInputMode {
    match input_mode {
        TestInputMode::TokenRows => MoeInputMode::TokenRows,
        TestInputMode::RoutedRows => MoeInputMode::RoutedRows,
    }
}

fn reference(
    x: &[bf16],
    expert_weights: &[bf16],
    topk_ids: &[u32],
    route_weights: Option<&[f32]>,
    case: &Case,
) -> Vec<f32> {
    let num_routes = case.tokens * case.top_k;
    let mut output = vec![0.0; num_routes * case.n];
    for route in 0..num_routes {
        let x_row = match case.input_mode {
            TestInputMode::TokenRows => route / case.top_k,
            TestInputMode::RoutedRows => route,
        };
        let expert = topk_ids[route] as usize;
        if expert >= case.experts {
            continue;
        }
        for column in 0..case.n {
            let mut acc = 0.0f32;
            for inner in 0..case.k {
                let lhs = f32::from(x[x_row * case.k + inner]);
                let rhs = f32::from(expert_weights[(expert * case.n + column) * case.k + inner]);
                acc += lhs * rhs;
            }
            if let Some(route_weights) = route_weights {
                acc *= route_weights[route];
            }
            output[route * case.n + column] = f32::from(bf16::from_f32(acc));
        }
    }
    output
}

fn run_case(
    case: Case,
    topk_ids: Vec<u32>,
    route_weights: Option<Vec<f32>>,
    label: &str,
) -> Result<()> {
    let _lock = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let num_routes = case.tokens * case.top_k;
    assert_eq!(topk_ids.len(), num_routes);
    if let Some(route_weights) = route_weights.as_deref() {
        assert_eq!(route_weights.len(), num_routes);
    }

    let x_rows = match case.input_mode {
        TestInputMode::TokenRows => case.tokens,
        TestInputMode::RoutedRows => num_routes,
    };
    let x_data = patterned_bf16(x_rows * case.k, 1, 0.25);
    let expert_data = patterned_bf16(case.experts * case.n * case.k, 2, 0.08);
    let expected = reference(
        &x_data,
        &expert_data,
        &topk_ids,
        route_weights.as_deref(),
        &case,
    );

    let device = Device::new_cuda(0)?;
    let x = Tensor::from_vec(x_data, (x_rows, case.k), &device)?;
    let expert_weights = Tensor::from_vec(expert_data, (case.experts, case.n, case.k), &device)?;
    let topk_ids = Tensor::from_vec(topk_ids, (case.tokens, case.top_k), &device)?;
    let route_weights = match route_weights {
        Some(route_weights) => Some(Tensor::from_vec(
            route_weights,
            (case.tokens, case.top_k),
            &device,
        )?),
        None => None,
    };
    let routing = MoeRouting::new(&topk_ids, case.experts)?;

    if case.warmup {
        warmup_routed_grouped_matmul(
            &x,
            &expert_weights,
            &routing,
            api_input_mode(case.input_mode),
            route_weights.as_ref(),
        )?;
    }
    let output = routed_grouped_matmul(
        &x,
        &expert_weights,
        &routing,
        api_input_mode(case.input_mode),
        route_weights.as_ref(),
    )?;
    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.dims2()?, (num_routes, case.n));

    let output = output
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_abs_diff = output
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff <= MAX_ABS_DIFF,
        "{label}: max abs diff {max_abs_diff:.6e} exceeds {MAX_ABS_DIFF:.6e}"
    );
    Ok(())
}

#[test]
fn token_rows_with_inactive_routes_matches_reference() -> Result<()> {
    run_case(
        Case {
            tokens: 5,
            top_k: 2,
            experts: 4,
            n: 29,
            k: 37,
            input_mode: TestInputMode::TokenRows,
            warmup: false,
        },
        vec![0, 1, 1, 3, 0, 3, 3, 0, 99, u32::MAX],
        None,
        "token rows with inactive routes",
    )
}

#[test]
fn routed_rows_with_route_weights_matches_reference() -> Result<()> {
    run_case(
        Case {
            tokens: 4,
            top_k: 3,
            experts: 5,
            n: 23,
            k: 33,
            input_mode: TestInputMode::RoutedRows,
            warmup: false,
        },
        vec![4, 0, 1, 2, 4, 3, 1, 0, 4, 3, 2, 0],
        Some(vec![
            0.55, 0.30, 0.15, 0.60, 0.25, 0.15, 0.45, 0.35, 0.20, 0.50, 0.30, 0.20,
        ]),
        "routed rows with route weights",
    )
}

#[test]
fn warmup_then_token_rows_launch_matches_reference() -> Result<()> {
    run_case(
        Case {
            tokens: 3,
            top_k: 2,
            experts: 3,
            n: 17,
            k: 31,
            input_mode: TestInputMode::TokenRows,
            warmup: true,
        },
        vec![0, 2, 1, 0, 2, 1],
        None,
        "warmed token rows",
    )
}

#[test]
fn multi_tile_grouped_launch_matches_reference() -> Result<()> {
    run_case(
        Case {
            tokens: 129,
            top_k: 1,
            experts: 1,
            n: 129,
            k: 65,
            input_mode: TestInputMode::TokenRows,
            warmup: false,
        },
        vec![0; 129],
        None,
        "multi-tile grouped launch",
    )
}

#[test]
fn shared_routing_composes_gate_and_down_projections() -> Result<()> {
    let _lock = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let tokens = 3;
    let top_k = 2;
    let experts = 3;
    let hidden = 7;
    let intermediate = 5;
    let routes = tokens * top_k;
    let topk_ids = vec![0, 2, 1, 0, 2, 1];
    let route_weights = vec![0.7, 0.3, 0.6, 0.4, 0.55, 0.45];
    let input_data = patterned_bf16(tokens * hidden, 3, 0.2);
    let gate_weights = patterned_bf16(experts * 2 * intermediate * hidden, 4, 0.08);
    let down_weights = patterned_bf16(experts * hidden * intermediate, 5, 0.08);

    let gate_case = Case {
        tokens,
        top_k,
        experts,
        n: 2 * intermediate,
        k: hidden,
        input_mode: TestInputMode::TokenRows,
        warmup: false,
    };
    let gate_up = reference(&input_data, &gate_weights, &topk_ids, None, &gate_case);
    let routed_data = (0..routes * intermediate)
        .map(|index| {
            let route = index / intermediate;
            let column = index % intermediate;
            bf16::from_f32(
                gate_up[route * 2 * intermediate + column]
                    * gate_up[route * 2 * intermediate + intermediate + column],
            )
        })
        .collect::<Vec<_>>();
    let down_case = Case {
        tokens,
        top_k,
        experts,
        n: hidden,
        k: intermediate,
        input_mode: TestInputMode::RoutedRows,
        warmup: false,
    };
    let down_routes = reference(
        &routed_data,
        &down_weights,
        &topk_ids,
        Some(&route_weights),
        &down_case,
    );
    let mut expected = vec![0f32; tokens * hidden];
    for route in 0..routes {
        for column in 0..hidden {
            expected[(route / top_k) * hidden + column] += down_routes[route * hidden + column];
        }
    }

    let device = Device::new_cuda(0)?;
    let input = Tensor::from_vec(input_data, (tokens, hidden), &device)?;
    let gate_weights =
        Tensor::from_vec(gate_weights, (experts, 2 * intermediate, hidden), &device)?;
    let down_weights = Tensor::from_vec(down_weights, (experts, hidden, intermediate), &device)?;
    let topk_ids = Tensor::from_vec(topk_ids, (tokens, top_k), &device)?;
    let route_weights = Tensor::from_vec(route_weights, (tokens, top_k), &device)?;
    let routing = MoeRouting::new(&topk_ids, experts)?;

    let gate_up = routed_grouped_matmul(
        &input,
        &gate_weights,
        &routing,
        MoeInputMode::TokenRows,
        None,
    )?;
    let gate = gate_up.narrow(1, 0, intermediate)?;
    let up = gate_up.narrow(1, intermediate, intermediate)?;
    let routed = (&gate * &up)?;
    let down = routed_grouped_matmul(
        &routed,
        &down_weights,
        &routing,
        MoeInputMode::RoutedRows,
        Some(&route_weights),
    )?;
    let output = down
        .to_dtype(DType::F32)?
        .reshape((tokens, top_k, hidden))?
        .sum(1)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_abs_diff = output
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= MAX_ABS_DIFF, "max abs diff {max_abs_diff}");
    Ok(())
}

#[test]
fn zero_matrix_dimensions_are_rejected() -> Result<()> {
    let _lock = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let device = Device::new_cuda(0)?;
    let topk_ids = Tensor::from_vec(vec![0u32], (1, 1), &device)?;
    let routing = MoeRouting::new(&topk_ids, 1)?;

    let input = Tensor::zeros((1, 1), DType::BF16, &device)?;
    let zero_n_weights = Tensor::zeros((1, 0, 1), DType::BF16, &device)?;
    let error = routed_grouped_matmul(
        &input,
        &zero_n_weights,
        &routing,
        MoeInputMode::TokenRows,
        None,
    )
    .expect_err("zero N must be rejected");
    assert!(error.to_string().contains("nonzero N and K"));

    let zero_k_input = Tensor::zeros((1, 0), DType::BF16, &device)?;
    let zero_k_weights = Tensor::zeros((1, 1, 0), DType::BF16, &device)?;
    let error = routed_grouped_matmul(
        &zero_k_input,
        &zero_k_weights,
        &routing,
        MoeInputMode::TokenRows,
        None,
    )
    .expect_err("zero K must be rejected");
    assert!(error.to_string().contains("nonzero N and K"));
    Ok(())
}
