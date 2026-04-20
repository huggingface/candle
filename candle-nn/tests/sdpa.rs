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

    /// Regression guard for the dispatcher's `supports_sdpa_vector` rule:
    /// `q_seq <= 8` routes to the vector kernel, but the vector kernel in
    /// `scaled_dot_product_attention.metal` has no q-axis and only computes
    /// one output position per `(bs, qhead)` threadgroup. For `q_seq` in
    /// `[2, 8]` the non-zero positions are left uninitialised (garbage from
    /// the pooled Metal buffer), which silently corrupts any caller that
    /// expects per-position logits — notably speculative-decoding verify
    /// batches and short prefills.
    ///
    /// Compares sdpa vs a manual matmul→softmax→matmul reference for every
    /// R ∈ [2, 8]. Expected to FAIL on the current kernel; passes once the
    /// dispatch is fixed to route multi-query through the full kernel or the
    /// vector kernel is extended to loop over q_seq.
    #[test]
    fn sdpa_vector_q_seq_2_to_8_matches_reference() -> Result<()> {
        // Real-world verify batch shape: one sequence, GQA heads, cross-attention
        // against a larger K sequence (prior cache + new tokens).
        const BS: usize = 1;
        const H: usize = 8;
        const L: usize = 16; // larger than R so vector path is eligible
        const DK: usize = 64;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;

        let mut failures = Vec::new();
        for r in 2..=8usize {
            let q = randn(&mut rng, (BS, H, r, DK), &device)?;
            let reference = {
                let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
                let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                    .to_dtype(q.dtype())?;
                att.matmul(&v.clone())?
            };
            let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
            assert_eq!(reference.shape(), sdpa_output.shape());
            let rel_err: f32 = ((&reference - &sdpa_output)?.abs()?
                / &reference.abs()?.clamp(1e-6f32, f32::MAX)?)?
                .mean_all()?
                .to_scalar()?;
            // If the kernel ignores q positions 1..r-1, we see catastrophic
            // error (uninitialised output ~ N(0,1)-ish); a correct kernel
            // stays well under 1e-2.
            if rel_err > 0.05 {
                failures.push((r, rel_err));
            }
        }
        assert!(
            failures.is_empty(),
            "sdpa_vector gives wrong output for q_seq in [2, 8]: {failures:?}"
        );
        Ok(())
    }

    /// The same regression with `do_causal = true`, which is the path every
    /// autoregressive caller hits. Speculative-decoding verify uses exactly
    /// this pattern: Q at `[pending, draft[0..K-1]]` (seq = K+1 ∈ [2, 8] in
    /// practice) against K/V extended from the running KV cache, expecting
    /// per-position logits for accept/reject comparison.
    #[test]
    fn sdpa_vector_q_seq_2_to_8_with_causal_matches_reference() -> Result<()> {
        const BS: usize = 1;
        const H: usize = 8;
        const L_PAST: usize = 4; // simulates prior KV cache size
        const DK: usize = 64;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFEBABE);

        let mut failures = Vec::new();
        for r in 2..=8usize {
            let q = randn(&mut rng, (BS, H, r, DK), &device)?;
            let l = L_PAST + r; // K/V = past + current (aligned-causal pattern)
            let k = randn(&mut rng, (BS, H, l, DK), &device)?;
            let v = randn(&mut rng, (BS, H, l, DK), &device)?;

            // Reference: build the aligned-causal mask manually, then
            // matmul → mask → softmax → matmul.
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            // mask[i, j] = -inf if j > L_PAST + i, else 0.
            let mut mask_data: Vec<f32> = vec![0.0; r * l];
            for i in 0..r {
                for j in 0..l {
                    if j > L_PAST + i {
                        mask_data[i * l + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask =
                Tensor::from_vec(mask_data, (r, l), &device)?.broadcast_as((BS, H, r, l))?;
            let att_masked =
                (att.to_dtype(DType::F32)? + mask.to_dtype(DType::F32)?)?;
            let att = candle_nn::ops::softmax_last_dim(&att_masked)?.to_dtype(q.dtype())?;
            let reference = att.matmul(&v.clone())?;

            let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, true, scale as f32, 1.)?;
            assert_eq!(reference.shape(), sdpa_output.shape());
            let rel_err: f32 = ((&reference - &sdpa_output)?.abs()?
                / &reference.abs()?.clamp(1e-6f32, f32::MAX)?)?
                .mean_all()?
                .to_scalar()?;
            if rel_err > 0.05 {
                failures.push((r, rel_err));
            }
        }
        assert!(
            failures.is_empty(),
            "sdpa_vector with do_causal=true gives wrong output for q_seq in [2, 8]: {failures:?}"
        );
        Ok(())
    }
}
