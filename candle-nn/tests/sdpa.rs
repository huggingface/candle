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

    // softmax(q k^T * scale + mask) v in f32 on contiguous copies, with grouped kv
    // heads expanded and an optional bottom-right aligned causal mask.
    fn naive_sdpa(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        scale: f64,
    ) -> Result<Tensor> {
        let (b, h_q, q_seq, _) = q.dims4()?;
        let (_, h_kv, kv_seq, _) = k.dims4()?;
        let expand = |t: &Tensor| -> Result<Tensor> {
            let d = t.dim(3)?;
            t.to_dtype(DType::F32)?
                .contiguous()?
                .unsqueeze(2)?
                .broadcast_as((b, h_kv, h_q / h_kv, kv_seq, d))?
                .reshape((b, h_q, kv_seq, d))
        };
        let q = q.to_dtype(DType::F32)?.contiguous()?;
        let mut att = (q * scale)?.matmul(&expand(k)?.t()?)?;
        if let Some(mask) = mask {
            att = (att + mask.to_dtype(DType::F32)?)?;
        }
        if causal {
            let off = kv_seq - q_seq;
            let m: Vec<f32> = (0..q_seq)
                .flat_map(|i| {
                    (0..kv_seq).map(move |j| if j > i + off { f32::NEG_INFINITY } else { 0. })
                })
                .collect();
            att = att.broadcast_add(&Tensor::from_vec(m, (q_seq, kv_seq), att.device())?)?;
        }
        candle_nn::ops::softmax_last_dim(&att)?.matmul(&expand(v)?)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar()
    }

    // Non-contiguous views of a (b, h, seq, d) tensor holding the same values.
    fn strided_views(x: &Tensor) -> Result<Vec<(&'static str, Tensor)>> {
        let (b, h, seq, d) = x.dims4()?;
        let dev = x.device();
        // (b, seq, h, d).transpose(1, 2), as produced by attention projections.
        let transposed = x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
        // A window into a larger preallocated KV cache, nonzero start offset.
        let cache = Tensor::zeros((b, h, seq + 7, d), x.dtype(), dev)?
            .slice_assign(&[0..b, 0..h, 3..3 + seq, 0..d], x)?
            .narrow(2, 3, seq)?;
        // Both of the above.
        let transposed_cache = Tensor::zeros((b, seq + 7, h, d), x.dtype(), dev)?
            .slice_assign(&[0..b, 3..3 + seq, 0..h, 0..d], &x.transpose(1, 2)?)?
            .narrow(1, 3, seq)?
            .transpose(1, 2)?;
        // Head dim not packed.
        let strided_head_dim = x.transpose(2, 3)?.contiguous()?.transpose(2, 3)?;
        Ok(vec![
            ("transposed", transposed),
            ("cache", cache),
            ("transposed_cache", transposed_cache),
            ("strided_head_dim", strided_head_dim),
        ])
    }

    #[test]
    fn sdpa_non_contiguous_inputs() -> Result<()> {
        // q/k/v views the kernels cannot index directly must still give the same
        // result as contiguous inputs, on the vector (decode), 2-pass vector and
        // full (prefill) paths.
        const BS: usize = 2;
        const H_Q: usize = 4;
        const DK: usize = 64;
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4020);
        let scale = 0.125;
        let cases = [
            // (q_seq, kv_seq, h_kv, causal)
            (1, 24, 2, false),
            (1, 24, 1, false),
            (1, 1100, 2, false),
            (16, 16, 2, false),
            (37, 50, 2, true),
            (37, 50, 1, false),
        ];
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let tol = if dtype == DType::F32 { 1e-4 } else { 5e-2 };
            for (q_seq, kv_seq, h_kv, causal) in cases {
                let q = randn(&mut rng, (BS, H_Q, q_seq, DK), &device)?.to_dtype(dtype)?;
                let k = randn(&mut rng, (BS, h_kv, kv_seq, DK), &device)?.to_dtype(dtype)?;
                let v = randn(&mut rng, (BS, h_kv, kv_seq, DK), &device)?.to_dtype(dtype)?;
                let expected = naive_sdpa(&q, &k, &v, None, causal, scale)?;
                let contiguous = candle_nn::ops::sdpa(&q, &k, &v, None, causal, scale as f32, 1.)?;
                let qs = strided_views(&q)?;
                let ks = strided_views(&k)?;
                let vs = strided_views(&v)?;
                for ((name, q), ((_, k), (_, v))) in qs.iter().zip(ks.iter().zip(vs.iter())) {
                    let out = candle_nn::ops::sdpa(q, k, v, None, causal, scale as f32, 1.)?;
                    let ctx = format!("{dtype:?} {name} q_seq={q_seq} kv_seq={kv_seq} h_kv={h_kv}");
                    let err = max_abs_diff(&out, &contiguous)?;
                    assert!(err <= tol, "{ctx}: vs contiguous {err}");
                    let err = max_abs_diff(&out, &expected)?;
                    assert!(err <= tol, "{ctx}: vs reference {err}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sdpa_full_non_contiguous_mask() -> Result<()> {
        // The mask's storage offset and strides must be honored too.
        const BS: usize = 2;
        const H: usize = 4;
        const DK: usize = 64;
        const Q_SEQ: usize = 37;
        const KV_SEQ: usize = 50;
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(40200);
        let scale = 0.125;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let tol = if dtype == DType::F32 { 1e-4 } else { 5e-2 };
            let q = randn(&mut rng, (BS, H, Q_SEQ, DK), &device)?.to_dtype(dtype)?;
            let k = randn(&mut rng, (BS, H, KV_SEQ, DK), &device)?.to_dtype(dtype)?;
            let v = randn(&mut rng, (BS, H, KV_SEQ, DK), &device)?.to_dtype(dtype)?;
            let mask = randn(&mut rng, (BS, H, Q_SEQ, KV_SEQ), &device)?.to_dtype(dtype)?;
            let expected = naive_sdpa(&q, &k, &v, Some(&mask), false, scale)?;
            let mut masks = strided_views(&mask)?;
            // Contiguous apart from a nonzero start offset.
            let offset = Tensor::cat(&[&mask, &mask], 0)?.narrow(0, BS, BS)?;
            masks.push(("offset", offset));
            for (name, mask) in masks {
                let out = candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.)?;
                let err = max_abs_diff(&out, &expected)?;
                assert!(err <= tol, "{dtype:?} {name} mask: {err}");
            }
        }
        Ok(())
    }
}
