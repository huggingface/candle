#![allow(unused_imports, unexpected_cfgs)]
#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    #[test]
    fn sdpa_full() -> candle::Result<()> {
        wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
        #[cfg(target_arch = "wasm32")]
        use wasm_bindgen_test::wasm_bindgen_test as test;
        #[cfg(not(target_arch = "wasm32"))]
        use tokio::test as test;
        use candle_wasm_tests::{
            to_vec0_round_async, to_vec1_round_async, to_vec2_round_async,
            to_vec3_round_async,
        };
        use candle::{DType, Device, Tensor};
        const BS: usize = 4;
        const R: usize = 4;
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0005, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_vector() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0001, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_full_softcapping() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        use std::ops::{Div, Mul};
        const BS: usize = 4;
        const R: usize = 4;
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(
                    &att.to_dtype(DType::F32)?.div(SOFTCAP)?.tanh()?.mul(SOFTCAP)?,
                )?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            scale as f32,
            SOFTCAP as f32,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0004, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_vector_softcapping() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        use std::ops::{Div, Mul};
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(
                    &att.to_dtype(DType::F32)?.div(SOFTCAP)?.tanh()?.mul(SOFTCAP)?,
                )?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            scale as f32,
            SOFTCAP as f32,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0001, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_vector_cross() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 24;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0013, "{}", error);
        Ok(())
    }
}
