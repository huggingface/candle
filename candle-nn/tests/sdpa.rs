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

    // Expand grouped KV heads to full Q head count so a reference attention can be
    // computed with a dense matmul. Mirrors quantized_llama's repeat_kv.
    fn repeat_kv(x: &Tensor, gqa_factor: usize) -> Result<Tensor> {
        let (bs, n_kv, seq, dk) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((bs, n_kv, gqa_factor, seq, dk))?
            .reshape((bs, n_kv * gqa_factor, seq, dk))
    }

    // Dense (un-fused) grouped-query attention reference.
    fn reference_gqa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let (_, h_q, _, _) = q.dims4()?;
        let (_, h_kv, _, _) = k.dims4()?;
        let k = repeat_kv(&k.contiguous()?, h_q / h_kv)?;
        let v = repeat_kv(&v.contiguous()?, h_q / h_kv)?;
        let att = (q.contiguous()? * scale)?.matmul(&k.t()?)?;
        let att =
            candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        att.matmul(&v)
    }

    #[test]
    fn sdpa_vector_gqa_decode() -> Result<()> {
        // GQA decode (seq_len == 1, n_head != n_kv_head) through the fused Metal
        // vector kernel. Every other test in this file uses equal head counts, so
        // the grouped-query path was otherwise unexercised.
        const BS: usize = 1;
        const R: usize = 1; // decode: single next token
        const L: usize = 24; // accumulated KV cache length
        const DK: usize = 64;
        const H_Q: usize = 8;
        const H_KV: usize = 2; // gqa_factor = 4

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(31337);
        let q = randn(&mut rng, (BS, H_Q, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H_KV, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H_KV, L, DK), &device)?;
        let reference = reference_gqa(&q, &k, &v, scale)?;
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(reference.shape(), sdpa_output.shape());
        let error: f32 = ((&reference - &sdpa_output)?.abs()? / &reference.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.0013, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_non_contiguous() -> Result<()> {
        // The fused Metal SDPA kernels require contiguous q/k/v. Callers build the
        // attention inputs with `transpose(1, 2)`, which is non-contiguous, so they
        // must remember to `.contiguous()` by hand -- otherwise the kernel walks the
        // wrong strides and silently returns garbage. Assert that layout does not
        // change the result, across both the vector (decode) and full (prefill)
        // paths, using TinyLlama's GQA shape (32 q-heads, 4 kv-heads).
        const BS: usize = 1;
        const DK: usize = 64;
        const H_Q: usize = 32;
        const H_KV: usize = 4; // gqa_factor = 8 (TinyLlama)

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        // (q_seq, k_seq): decode routes through the vector kernel, prefill the full one.
        for (q_seq, k_seq) in [(1usize, 24usize), (5, 5)] {
            // Lay out as (bs, seq, heads, dk) then transpose(1, 2) -> non-contiguous,
            // exactly as the transformer models produce their attention inputs.
            let q = randn(&mut rng, (BS, q_seq, H_Q, DK), &device)?.transpose(1, 2)?;
            let k = randn(&mut rng, (BS, k_seq, H_KV, DK), &device)?.transpose(1, 2)?;
            let v = randn(&mut rng, (BS, k_seq, H_KV, DK), &device)?.transpose(1, 2)?;
            assert!(!k.is_contiguous() && !v.is_contiguous());

            let out = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
            let out_contig = candle_nn::ops::sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                None,
                false,
                scale as f32,
                1.,
            )?;
            assert_eq!(out.shape(), out_contig.shape());
            // Same math on the same values: strided and contiguous must agree.
            let error: f32 = (&out - &out_contig)?.abs()?.max_all()?.to_scalar()?;
            assert!(
                error <= 1e-4,
                "q_seq={q_seq} k_seq={k_seq} max_abs_diff={error}"
            );
        }
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
}
