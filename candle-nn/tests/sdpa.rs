#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    use candle::{DType, Device, Result, Shape, Tensor};
    use rand::SeedableRng;
    use rand_distr::Distribution;
    use std::ops::{Div, Mul};

    fn randn<S: Into<Shape>>(
        rng: &mut rand::rngs::StdRng,
        shape: S,
        dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let vs: Vec<f32> = (0..elem_count).map(|_| normal.sample(rng)).collect();
        Tensor::from_vec(vs, &shape, dev)
    }

    #[test]
    fn sdpa_full() -> Result<()> {
        // Test the full SDPA kernel path (q_seq > 8)
        const BS: usize = 4;
        const R: usize = 16;
        const L: usize = 16;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
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
        // Larger sequences have higher accumulated error
        assert!(error <= 0.02, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector() -> Result<()> {
        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
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
        assert!(error <= 0.000, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_full_softcapping() -> Result<()> {
        // Test softcapping with sdpa_vector kernel (q_seq = 1)
        // NOTE: Vector kernel only supports q_seq = 1 correctly
        // Full kernel does NOT support softcapping
        const BS: usize = 4;
        const R: usize = 1; // Vector kernel requires q_seq = 1
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
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
        // Slightly higher error for cross-attention case (R=1, L=4)
        assert!(error <= 0.002, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector_softcapping() -> Result<()> {
        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
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
    fn sdpa_vector_cross() -> Result<()> {
        // Allow vectorized, seqlen = 1. Simulat cross attention case where R != L, R = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 24;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
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
        assert!(error <= 0.0013, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_full_causal_with_mask() -> Result<()> {
        // Regression test: a non-causal additive mask (padding/bias) combined
        // with `do_causal = true` must apply BOTH maskings. The wrapper used to
        // drop causal masking whenever any mask was present, which let queries
        // attend to future keys and corrupted prefill/training outputs.
        const BS: usize = 2;
        const H: usize = 3;
        const R: usize = 16; // q_seq > 8 -> full kernel
        const L: usize = 16;
        const DK: usize = 64;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;

        // Non-causal additive padding mask: drop the last two key positions for
        // every query (0.0 = keep, -inf = drop). No row is fully masked, since
        // causal masking still leaves key 0 visible to every query.
        let neg_inf = f32::NEG_INFINITY;
        let mut mask_data = vec![0f32; BS * H * R * L];
        for b in 0..BS {
            for h in 0..H {
                for r in 0..R {
                    for c in (L - 2)..L {
                        mask_data[((b * H + h) * R + r) * L + c] = neg_inf;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (BS, H, R, L), &device)?;

        // Reference: apply BOTH causal masking and the additive mask to scores.
        let mut causal = vec![0f32; R * L];
        for i in 0..R {
            for j in 0..L {
                if j > i {
                    causal[i * L + j] = neg_inf;
                }
            }
        }
        let causal =
            Tensor::from_vec(causal, (1, 1, R, L), &device)?.broadcast_as((BS, H, R, L))?;

        let ground_truth = {
            let att = (q.clone() * scale)?
                .matmul(&k.clone().t()?)?
                .to_dtype(DType::F32)?;
            let att = att.broadcast_add(&causal)?.broadcast_add(&mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };

        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), true, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());

        // No NaNs/infs leaked from the combined mask + causal path.
        let flat = sdpa_output.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|x| x.is_finite()),
            "sdpa output has non-finite values"
        );

        let denom = ground_truth.abs()?.affine(1.0, 1e-3)?;
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / denom)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.1, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_full_fully_masked_row_is_finite() -> Result<()> {
        // Regression test for the softmax-normalizer guard: a query row whose
        // keys are all excluded by the mask must yield a finite (zero) row
        // instead of 0/0 = NaN.
        const BS: usize = 1;
        const H: usize = 2;
        const R: usize = 16; // q_seq > 8 -> full kernel
        const L: usize = 16;
        const DK: usize = 64;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;

        // Additive mask: fully mask query row 0 (all keys), keep the rest.
        let neg_inf = f32::NEG_INFINITY;
        let mut mask_data = vec![0f32; BS * H * R * L];
        for b in 0..BS {
            for h in 0..H {
                for c in 0..L {
                    mask_data[((b * H + h) * R) * L + c] = neg_inf;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (BS, H, R, L), &device)?;

        let out = candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.)?;
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|x| x.is_finite()),
            "fully-masked row produced non-finite values"
        );

        // The fully-masked query row (row 0) must be zero.
        for b in 0..BS {
            for h in 0..H {
                let base = ((b * H + h) * R) * DK;
                for d in 0..DK {
                    assert!(
                        flat[base + d].abs() < 1e-5,
                        "masked row should be zero, got {}",
                        flat[base + d]
                    );
                }
            }
        }
        Ok(())
    }
}
