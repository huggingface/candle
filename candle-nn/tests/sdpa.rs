#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    #[test]
    fn sdpa_full() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};

        // Force seqlen = 100
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

        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.002, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1
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

        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0001, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_2pass() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1 but kseqlen is long (long context)
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 2048;
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

        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.002, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_full_masked() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1
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
        let mask = Tensor::randn(0f32, 1f32, (BS, H, R, L), &device)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&(att.to_dtype(DType::F32)? + &mask)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.006, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_softcapping() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        use std::ops::{Div, Mul};

        // Allow vectorized, seqlen = 1
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
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output =
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, SOFTCAP as f32)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0001, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_2pass_softcapping() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        use std::ops::{Div, Mul};

        // Allow vectorized, seqlen = 1 but kseqlen is long (long context)
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 2048;
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
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output =
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, SOFTCAP as f32)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0021, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_cross() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};

        // Allow vectorized, seqlen = 1. Simulat cross attention case where R != L, R = 1
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

        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;

        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;

        assert!(error <= 0.0017, "{}", error);

        Ok(())
    }

    #[test]
    fn sdpa_vector_gqa_2pass_no_mask() -> candle::Result<()> {
        use candle::{DType, Device, Tensor};
        // GQA && Increase seq_len to 1024 in order to cover 2-pass code branch

        /// Repeats a key or value tensor for grouped query attention
        /// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
        fn repeat_kv(xs: Tensor, n_rep: usize) -> candle::Result<Tensor> {
            if n_rep == 1 {
                Ok(xs)
            } else {
                let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
                // Using cat is faster than a broadcast as it avoids going through a potentially
                // strided copy.
                // https://github.com/huggingface/candle/pull/2043
                Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((
                    b_sz,
                    n_kv_head * n_rep,
                    seq_len,
                    head_dim,
                ))
            }
        }

        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1024;
        const DK: usize = 128;
        const HQ: usize = 28;
        const HKV: usize = 4;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let q = Tensor::randn(0f32, 1f32, (BS, HQ, R, DK), &device)?;
        let k = Tensor::randn(0f32, 1f32, (BS, HKV, L, DK), &device)?;
        let v = Tensor::randn(0f32, 1f32, (BS, HKV, L, DK), &device)?;

        let k_aligned = repeat_kv(k.copy().unwrap(), HQ / HKV)?;
        let v_aligned = repeat_kv(v.copy().unwrap(), HQ / HKV)?;

        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k_aligned.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v_aligned.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        println!("{error}");
        assert!(error <= 0.06, "{}", error);
        Ok(())
    }
}
