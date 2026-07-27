#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::{to_vec2_round, to_vec3_round};
use candle::{test_device, DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, lora::LoraLinear, Module, VarBuilder, VarMap};

fn make_base(device: &Device) -> Result<candle_nn::Linear> {
    // in_dim = 3, out_dim = 2
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let base = linear_no_bias(3, 2, vb.pp("q_proj"))?;
    varmap.set_one(
        "q_proj.weight",
        Tensor::new(&[[1f32, 0., 0.], [0., 1., 0.]], device)?,
    )?;
    Ok(base)
}

fn lora_forward_matches_manual_computation(device: &Device) -> Result<()> {
    let base = make_base(device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mut lora =
        LoraLinear::from_peft(base, vb.pp("q_proj"), /*rank=*/ 2, /*alpha=*/ 4.0)?;

    // lora_A: [r=2, in=3], lora_B: [out=2, r=2]
    varmap.set_one(
        "q_proj.lora_A.weight",
        Tensor::new(&[[1f32, 0., 0.], [0., 1., 0.]], device)?,
    )?;
    varmap.set_one(
        "q_proj.lora_B.weight",
        Tensor::new(&[[1f32, 0.], [0., 1.]], device)?,
    )?;

    let x = Tensor::new(&[[1f32, 2., 3.]], device)?;
    let y = lora.forward(&x)?;
    // base_out = [1, 2]; delta = scale * B@A@x with scale = alpha/r = 2.0
    // A@x = [1, 2] (selects first two components), B@(A@x) = [1, 2], * scale(2.0) = [2, 4]
    // y = base_out + delta = [3, 6]
    assert_eq!(to_vec2_round(&y, 4)?, vec![vec![3f32, 6.]]);

    lora.merge()?;
    assert!(lora.is_merged());
    let y_merged = lora.forward(&x)?;
    assert_eq!(to_vec2_round(&y_merged, 4)?, vec![vec![3f32, 6.]]);

    lora.unmerge()?;
    assert!(!lora.is_merged());
    let y_unmerged = lora.forward(&x)?;
    assert_eq!(to_vec2_round(&y_unmerged, 4)?, vec![vec![3f32, 6.]]);

    Ok(())
}

fn lora_multi_adapter_switching(device: &Device) -> Result<()> {
    let base = make_base(device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mut lora = LoraLinear::new(base);

    lora.add_adapter("a", vb.pp("lora_a"), 2, 2.0)?;
    varmap.set_one(
        "lora_a.lora_A.weight",
        Tensor::new(&[[1f32, 0., 0.], [0., 0., 0.]], device)?,
    )?;
    varmap.set_one(
        "lora_a.lora_B.weight",
        Tensor::new(&[[1f32, 0.], [0., 0.]], device)?,
    )?;

    lora.add_adapter("b", vb.pp("lora_b"), 2, 2.0)?;
    varmap.set_one(
        "lora_b.lora_A.weight",
        Tensor::new(&[[0f32, 1., 0.], [0., 0., 0.]], device)?,
    )?;
    varmap.set_one(
        "lora_b.lora_B.weight",
        Tensor::new(&[[0f32, 0.], [1., 0.]], device)?,
    )?;

    let x = Tensor::new(&[[1f32, 2., 3.]], device)?;

    assert_eq!(lora.active_adapter(), Some("b"));
    let y_b = lora.forward(&x)?;
    // base_out = [1, 2]; adapter b: A@x = [2, 0], B@(A@x) = [0, 2], scale=1.0 -> delta=[0, 2]
    assert_eq!(to_vec2_round(&y_b, 4)?, vec![vec![1f32, 4.]]);

    lora.set_active_adapter(Some("a"))?;
    let y_a = lora.forward(&x)?;
    // adapter a: A@x = [1, 0], B@(A@x) = [1, 0], scale=1.0 -> delta=[1, 0]
    assert_eq!(to_vec2_round(&y_a, 4)?, vec![vec![2f32, 2.]]);

    lora.merge()?;
    assert!(lora.set_active_adapter(Some("b")).is_err());
    lora.unmerge()?;
    lora.set_active_adapter(Some("b"))?;
    assert_eq!(lora.active_adapter(), Some("b"));

    lora.set_active_adapter(None)?;
    let y_base = lora.forward(&x)?;
    assert_eq!(to_vec2_round(&y_base, 4)?, vec![vec![1f32, 2.]]);

    Ok(())
}

/// A `LoraLinear` with two adapters of different ranks, for the heterogeneous
/// batching tests: "a" (rank 2, alpha 4) and "b" (rank 1, alpha 3).
fn make_multi_lora(device: &Device) -> Result<LoraLinear> {
    let base = make_base(device)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mut lora = LoraLinear::new(base);

    lora.add_adapter("a", vb.pp("lora_a"), 2, 4.0)?;
    varmap.set_one(
        "lora_a.lora_A.weight",
        Tensor::new(&[[1f32, -1., 0.5], [0.5, 1., -1.]], device)?,
    )?;
    varmap.set_one(
        "lora_a.lora_B.weight",
        Tensor::new(&[[1f32, 0.5], [-0.5, 1.]], device)?,
    )?;

    lora.add_adapter("b", vb.pp("lora_b"), 1, 3.0)?;
    varmap.set_one(
        "lora_b.lora_A.weight",
        Tensor::new(&[[0.25f32, -0.5, 1.]], device)?,
    )?;
    varmap.set_one(
        "lora_b.lora_B.weight",
        Tensor::new(&[[2f32], [-1.]], device)?,
    )?;

    Ok(lora)
}

/// Reference implementation: run each batch row on its own with the row's
/// adapter activated, then stitch the rows back together.
fn sequential_forward(
    lora: &LoraLinear,
    x: &Tensor,
    assignments: &[Option<&str>],
) -> Result<Tensor> {
    let mut lora = lora.clone();
    let mut rows = Vec::with_capacity(assignments.len());
    for (i, assignment) in assignments.iter().enumerate() {
        lora.set_active_adapter(*assignment)?;
        rows.push(lora.forward(&x.narrow(0, i, 1)?)?);
    }
    Tensor::cat(&rows, 0)
}

fn lora_batched_forward_matches_sequential(device: &Device) -> Result<()> {
    let lora = make_multi_lora(device)?;

    // A mixed batch: two adapters (of different ranks), a base-only row, and a
    // repeated adapter.
    let x = Tensor::new(
        &[
            [1f32, 2., 3.],
            [-1., 0.5, 2.],
            [0.25, -1., 1.],
            [3., -2., 0.5],
        ],
        device,
    )?;
    let assignments = [Some("a"), Some("b"), None, Some("a")];

    let batched = lora.forward_with_adapters(&x, &assignments)?;
    let sequential = sequential_forward(&lora, &x, &assignments)?;
    assert_eq!(to_vec2_round(&batched, 4)?, to_vec2_round(&sequential, 4)?);

    // Adapter isolation: each row must be unaffected by the other adapters in
    // the batch, i.e. match a batch where it is the only row.
    for (i, assignment) in assignments.iter().enumerate() {
        let row = lora.forward_with_adapters(&x.narrow(0, i, 1)?, &[*assignment])?;
        assert_eq!(
            to_vec2_round(&row, 4)?,
            vec![to_vec2_round(&batched, 4)?[i].clone()]
        );
    }
    Ok(())
}

fn lora_batched_forward_all_base_matches_base(device: &Device) -> Result<()> {
    let lora = make_multi_lora(device)?;
    let x = Tensor::new(&[[1f32, 2., 3.], [-1., 0.5, 2.]], device)?;

    let batched = lora.forward_with_adapters(&x, &[None, None])?;
    let base = lora.base().forward(&x)?;
    assert_eq!(to_vec2_round(&batched, 4)?, to_vec2_round(&base, 4)?);
    Ok(())
}

fn lora_batched_forward_3d_input(device: &Device) -> Result<()> {
    let lora = make_multi_lora(device)?;

    // [batch=3, seq=2, in=3]: the adapter assignment applies to every token of
    // a sequence.
    let x = Tensor::new(
        &[
            [[1f32, 2., 3.], [0.5, -1., 2.]],
            [[-1., 0.5, 2.], [2., 1., -0.5]],
            [[0.25, -1., 1.], [1., 1., 1.]],
        ],
        device,
    )?;
    let assignments = [Some("b"), None, Some("a")];

    let batched = lora.forward_with_adapters(&x, &assignments)?;
    let sequential = sequential_forward(&lora, &x, &assignments)?;
    assert_eq!(batched.dims(), &[3, 2, 2]);
    assert_eq!(to_vec3_round(&batched, 4)?, to_vec3_round(&sequential, 4)?);
    Ok(())
}

fn lora_batched_forward_errors(device: &Device) -> Result<()> {
    let mut lora = make_multi_lora(device)?;
    let x = Tensor::new(&[[1f32, 2., 3.], [-1., 0.5, 2.]], device)?;

    // Assignment count must match the batch size.
    assert!(lora.forward_with_adapters(&x, &[Some("a")]).is_err());
    // Unknown adapter names are rejected.
    assert!(lora
        .forward_with_adapters(&x, &[Some("a"), Some("nope")])
        .is_err());
    // A merged adapter would be double-counted, so this must fail.
    lora.set_active_adapter(Some("a"))?;
    lora.merge()?;
    assert!(lora.forward_with_adapters(&x, &[Some("a"), None]).is_err());
    lora.unmerge()?;
    assert!(lora.forward_with_adapters(&x, &[Some("a"), None]).is_ok());
    Ok(())
}

/// Reloading the adapter that is currently merged into the base weight would
/// make `unmerge` subtract the new delta from a base still holding the old one;
/// it must be rejected instead. Device-independent control-flow, so CPU only.
#[test]
fn lora_reload_while_merged_is_rejected() -> Result<()> {
    let device = Device::Cpu;
    let base = make_base(&device)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut lora = LoraLinear::new(base);
    lora.add_adapter("a", vb.pp("lora_a"), 2, 2.0)?;
    varmap.set_one(
        "lora_a.lora_A.weight",
        Tensor::new(&[[1f32, 0., 0.], [0., 1., 0.]], &device)?,
    )?;
    varmap.set_one(
        "lora_a.lora_B.weight",
        Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
    )?;
    let x = Tensor::new(&[[1f32, 2., 3.]], &device)?;

    lora.set_active_adapter(Some("a"))?;
    let active = to_vec2_round(&lora.forward(&x)?, 4)?;
    lora.merge()?;
    assert_eq!(to_vec2_round(&lora.forward(&x)?, 4)?, active);

    // Overwriting the merged adapter must be rejected, not silently corrupt base.
    assert!(lora.load_adapter("a", vb.pp("lora_a"), 2, 2.0).is_err());
    // The rejected reload left the merged state untouched.
    assert_eq!(to_vec2_round(&lora.forward(&x)?, 4)?, active);

    // Unmerge restores the same output, and the base weight exactly.
    lora.unmerge()?;
    assert_eq!(to_vec2_round(&lora.forward(&x)?, 4)?, active);
    lora.set_active_adapter(None)?;
    assert_eq!(to_vec2_round(&lora.forward(&x)?, 4)?, vec![vec![1f32, 2.]]);
    // Reloading is allowed once nothing is merged.
    assert!(lora.load_adapter("a", vb.pp("lora_a"), 2, 2.0).is_ok());
    Ok(())
}

/// An adapter checkpoint stored in a different dtype than the model must load
/// and run: `add_adapter_tensors` casts it to the base weight's dtype so the
/// plain `forward` does not hit a mixed-dtype matmul. CPU only (uses f64).
#[test]
fn lora_adapter_dtype_cast_to_base() -> Result<()> {
    let device = Device::Cpu;
    let base = make_base(&device)?; // f32, weight [[1, 0, 0], [0, 1, 0]]
    let avm = VarMap::new();
    let avb = VarBuilder::from_varmap(&avm, DType::F64, &device);
    let mut lora = LoraLinear::new(base);
    lora.add_adapter("a", avb.pp("a"), 2, 4.0)?;
    let x = Tensor::new(&[[1f32, 2., 3.]], &device)?;
    let y = lora.forward(&x)?;
    // lora_B is zero-initialized, so the delta is zero and the output equals
    // the base layer; the point is that the f64 adapter runs against f32 x.
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(to_vec2_round(&y, 4)?, vec![vec![1f32, 2.]]);
    Ok(())
}

test_device!(
    lora_forward_matches_manual_computation,
    lora_forward_matches_manual_computation_cpu,
    lora_forward_matches_manual_computation_gpu,
    lora_forward_matches_manual_computation_metal
);
test_device!(
    lora_multi_adapter_switching,
    lora_multi_adapter_switching_cpu,
    lora_multi_adapter_switching_gpu,
    lora_multi_adapter_switching_metal
);
test_device!(
    lora_batched_forward_matches_sequential,
    lora_batched_forward_matches_sequential_cpu,
    lora_batched_forward_matches_sequential_gpu,
    lora_batched_forward_matches_sequential_metal
);
test_device!(
    lora_batched_forward_all_base_matches_base,
    lora_batched_forward_all_base_matches_base_cpu,
    lora_batched_forward_all_base_matches_base_gpu,
    lora_batched_forward_all_base_matches_base_metal
);
test_device!(
    lora_batched_forward_3d_input,
    lora_batched_forward_3d_input_cpu,
    lora_batched_forward_3d_input_gpu,
    lora_batched_forward_3d_input_metal
);
test_device!(
    lora_batched_forward_errors,
    lora_batched_forward_errors_cpu,
    lora_batched_forward_errors_gpu,
    lora_batched_forward_errors_metal
);

/// Multi-step decode-loop parity for the CUDA BGMV kernel.
///
/// Every one of the `test_device!`-generated `_gpu` tests above still runs
/// with a single `seq > 1` or one-shot call, and without the `lora-cuda`
/// feature they never touch `candle_lora_kernels::bgmv_delta` at all (see
/// `batched_lora_delta` in `candle-nn/src/lora.rs`, which only routes to it
/// for `seq == 1` batches on a CUDA device). A generation loop such as
/// `Llama::generate_with_adapters` calls `forward_with_adapters` with a
/// `[batch, 1, in]` input once per decode step, reusing the same adapter
/// stacks and CUDA buffers across many kernel launches in a row, which is a
/// different code path than a single isolated call. This test drives several
/// such steps back to back on a heterogeneous batch (two adapters and a
/// base-only row) and checks the GPU kernel path matches the CPU gather +
/// batched matmul reference at every step, not just the first.
#[cfg(feature = "lora-cuda")]
#[test]
fn lora_batched_forward_multistep_decode_gpu_matches_cpu() -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0)?;
    let lora_cpu = make_multi_lora(&cpu)?;
    let lora_gpu = make_multi_lora(&gpu)?;

    // Two different adapters and a base-only row, as in the other batched
    // tests; a different input per step stands in for the evolving hidden
    // state a real decode loop would feed in at each new token.
    let assignments = [Some("a"), Some("b"), None];
    let steps: &[[[f32; 3]; 3]] = &[
        [[1., 2., 3.], [-1., 0.5, 2.], [0.25, -1., 1.]],
        [[0.1, -0.2, 0.3], [1.5, -1., 0.], [-0.5, 0.5, 0.5]],
        [[2., 1., -1.], [0., 0., 1.], [-1., -1., -1.]],
        [[0.5, 0.5, 0.5], [-2., 1., 0.5], [1., -1., 0.]],
    ];

    for (step, values) in steps.iter().enumerate() {
        // [batch, 1, in]: the shape a decode step feeds through the model.
        let x_cpu = Tensor::new(values, &cpu)?.unsqueeze(1)?;
        let x_gpu = Tensor::new(values, &gpu)?.unsqueeze(1)?;

        let cpu_out = lora_cpu.forward_with_adapters(&x_cpu, &assignments)?;
        let gpu_out = lora_gpu.forward_with_adapters(&x_gpu, &assignments)?;
        assert_eq!(
            to_vec3_round(&gpu_out, 4)?,
            to_vec3_round(&cpu_out, 4)?,
            "decode step {step} diverged between the CUDA BGMV kernel and the CPU reference"
        );
    }
    Ok(())
}
