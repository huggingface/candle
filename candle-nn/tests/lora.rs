#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils::to_vec2_round, DType, Device, Tensor};
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

#[test]
fn lora_forward_matches_manual_computation() -> Result<()> {
    let device = Device::Cpu;
    let base = make_base(&device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut lora =
        LoraLinear::from_peft(base, vb.pp("q_proj"), /*rank=*/ 2, /*alpha=*/ 4.0)?;

    // lora_A: [r=2, in=3], lora_B: [out=2, r=2]
    varmap.set_one(
        "q_proj.lora_A.weight",
        Tensor::new(&[[1f32, 0., 0.], [0., 1., 0.]], &device)?,
    )?;
    varmap.set_one(
        "q_proj.lora_B.weight",
        Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
    )?;

    let x = Tensor::new(&[[1f32, 2., 3.]], &device)?;
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

#[test]
fn lora_multi_adapter_switching() -> Result<()> {
    let device = Device::Cpu;
    let base = make_base(&device)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut lora = LoraLinear::new(base);

    lora.add_adapter("a", vb.pp("lora_a"), 2, 2.0)?;
    varmap.set_one(
        "lora_a.lora_A.weight",
        Tensor::new(&[[1f32, 0., 0.], [0., 0., 0.]], &device)?,
    )?;
    varmap.set_one(
        "lora_a.lora_B.weight",
        Tensor::new(&[[1f32, 0.], [0., 0.]], &device)?,
    )?;

    lora.add_adapter("b", vb.pp("lora_b"), 2, 2.0)?;
    varmap.set_one(
        "lora_b.lora_A.weight",
        Tensor::new(&[[0f32, 1., 0.], [0., 0., 0.]], &device)?,
    )?;
    varmap.set_one(
        "lora_b.lora_B.weight",
        Tensor::new(&[[0f32, 0.], [1., 0.]], &device)?,
    )?;

    let x = Tensor::new(&[[1f32, 2., 3.]], &device)?;

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
