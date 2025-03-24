#![allow(unused_imports, unexpected_cfgs)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, test_utils::to_vec3_round, Device, Result, Tensor};
async fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = candle_nn::ops::softmax(&tensor.log()?, 0)?;
    let t1 = candle_nn::ops::softmax(&tensor.log()?, 1)?;
    let t2 = candle_nn::ops::softmax(&tensor.log()?, 2)?;
    assert_eq!(
        to_vec3_round_async(& t0, 4). await ?, & [[[0.6, 0.5, 0.3636], [0.1111, 0.7143,
        0.5294]], [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]]
    );
    assert_eq!(
        to_vec3_round_async(& t1, 4). await ?, & [[[0.75, 0.1667, 0.3077], [0.25, 0.8333,
        0.6923]], [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]]
    );
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[0.375, 0.125, 0.5], [0.0667, 0.3333,
        0.6]], [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]]
    );
    let t2 = candle_nn::ops::softmax_last_dim(&tensor.log()?)?;
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[0.375, 0.125, 0.5], [0.0667, 0.3333,
        0.6]], [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]]
    );
    Ok(())
}
async fn rms_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t, 4). await ?, & [[[1.019, 0.6794, 4.0762], [0.1674,
        1.6744, 4.521]], [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]]
    );
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[1.019, 0.6794, 4.0762], [0.1674,
        1.6744, 4.521]], [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert!(diff < 1e-5);
    Ok(())
}
async fn rms_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0_async::<f32>()
        .await?;
    assert!(diff < 1e-5);
    Ok(())
}
async fn layer_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t, 4). await ?, & [[[0.7673, - 2.6726, 3.0071], [- 0.7247,
        0.0, 3.4742]], [[- 0.008, - 1.778, 3.991], [1.2071, - 2.8284, 1.9213]]]
    );
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[0.7673, - 2.6726, 3.0071], [- 0.7247,
        0.0, 3.4742]], [[- 0.008, - 1.778, 3.991], [1.2071, - 2.8284, 1.9213]]]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert!(diff < 1e-5);
    Ok(())
}
async fn layer_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0_async::<f32>()
        .await?;
    assert!(diff < 1e-5);
    Ok(())
}
#[test]
async fn softmax_numerical_stability() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = Tensor::new(&[1234f32, 0.], dev)?;
    let softmax = candle_nn::ops::softmax(&xs, 0)?;
    assert_eq!(softmax.to_vec1_async::< f32 > (). await ?, & [1f32, 0.]);
    Ok(())
}
async fn ropei(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }
    Ok(())
}
async fn rope(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }
    Ok(())
}
async fn rope_thd(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &cos, &sin)?.transpose(1, 2)?
    };
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }
    Ok(())
}
async fn sigmoid(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let s1 = candle_nn::ops::sigmoid(&tensor)?;
    let s2 = (1. / (1. + tensor.neg()?.exp()?)?)?;
    let diff = (s1 - s2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    Ok(())
}
candle_wasm_tests::test_device!(ropei, ropei_cpu, ropei_gpu, ropei_metal);
candle_wasm_tests::test_device!(rope, rope_cpu, rope_gpu, rope_metal);
candle_wasm_tests::test_device!(rope_thd, rope_thd_cpu, rope_thd_gpu, rope_thd_metal);
candle_wasm_tests::test_device!(
    softmax, softmax_cpu, softmax_gpu, softmax_metal, softmax_wgpu
);
candle_wasm_tests::test_device!(
    rms_norm, rms_norm_cpu, rms_norm_gpu, rms_norm_metal, rms_norm_wgpu
);
candle_wasm_tests::test_device!(
    rms_norml, rms_norml_cpu, rms_norml_gpu, rms_norml_metal, rms_norml_wgpu
);
candle_wasm_tests::test_device!(layer_norm, ln_cpu, ln_gpu, ln_meta, ln_wgpu);
candle_wasm_tests::test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal, lnl_wgpu);
candle_wasm_tests::test_device!(
    sigmoid, sigmoid_cpu, sigmoid_gpu, sigmoid_metal, sigmoid_wgpu
);
