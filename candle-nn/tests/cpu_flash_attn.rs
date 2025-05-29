use candle::{DType, Device, Result, Tensor};
use candle_nn::cpu_flash_attention::run_flash_attn_cpu;

#[test]
fn cpu_flash_attn() -> Result<()> {
    let b = 1;
    let s = 2;
    let h = 1;
    let d = 4;
    let softmax_scale = 1.0f32 / (d as f32).sqrt();

    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;

    // SDPA needs (b,h,s,d)
    let ground_truth = {
        let att = (q.clone() * softmax_scale as f64)?.matmul(&k.clone().t()?)?;
        let att =
            candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        att.matmul(&v.clone())?
    };

    // Flash attn needs (b,s,h,d)
    let out = run_flash_attn_cpu::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        None,
        softmax_scale,
        None,
        None,
    )?;

    let out_arr: Vec<f32> = out.flatten_all()?.to_vec1()?;
    let ground_truth_arr: Vec<f32> = ground_truth.flatten_all()?.to_vec1()?;
    for (a, b) in out_arr.iter().zip(ground_truth_arr.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} {b}");
    }
    Ok(())
}
