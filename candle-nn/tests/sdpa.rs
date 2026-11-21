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

    /// Causal attention on the CPU in f32, with the mask aligned to the last keys:
    /// query `i` sees keys `0..=i + (kl - ql)`.
    fn causal_reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let ql = q.dim(2)?;
        let kl = k.dim(2)?;
        let offset = kl - ql;
        let mask: Vec<f32> = (0..ql)
            .flat_map(|i| {
                (0..kl).map(move |j| {
                    if j > i + offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (ql, kl), &Device::Cpu)?;
        let att = (q.matmul(&k.t()?)? * scale)?.broadcast_add(&mask)?;
        candle_nn::ops::softmax_last_dim(&att)?.matmul(v)
    }

    /// Runs the fused kernel with `do_causal` and fewer queries than keys, for every
    /// `kl - ql` from 0 to 33, at each head dim that picks a different key block
    /// (BK 32 at 64, 16 at 128, 8 at 512), and returns one line per shape whose
    /// output differs from the reference by more than the dtype's tolerance.
    ///
    /// With `nan_tail`, K and V are views of the first `kl` rows of a longer buffer
    /// whose other rows are NaN, as a preallocated KV cache would be. Any read of a
    /// key past `kl` then shows as NaN in the output.
    fn causal_fewer_queries_failures(nan_tail: bool) -> Result<Vec<String>> {
        const B: usize = 1;
        const H: usize = 2;
        const TAIL: usize = 64;
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(299792458);
        let mut failures = vec![];
        for (dim, dtypes) in [
            (64, &[DType::F32, DType::BF16][..]),
            (128, &[DType::F32, DType::BF16][..]),
            (512, &[DType::BF16][..]),
        ] {
            let scale = f64::from(dim as u32).sqrt().recip();
            for &dtype in dtypes {
                let tolerance = if dtype == DType::F32 { 1e-4 } else { 3e-2 };
                for offset in 0..=33 {
                    // 33 queries leave a partial query block, 64 fill two, and
                    // `96 - offset % 32` makes `kl` a whole number of key blocks.
                    for ql in [33, 64, 96 - offset % 32] {
                        let kl = ql + offset;
                        let q = randn(&mut rng, (B, H, ql, dim), &Device::Cpu)?.to_dtype(dtype)?;
                        let k = randn(&mut rng, (B, H, kl, dim), &Device::Cpu)?.to_dtype(dtype)?;
                        let v = randn(&mut rng, (B, H, kl, dim), &Device::Cpu)?.to_dtype(dtype)?;
                        let expected = causal_reference(
                            &q.to_dtype(DType::F32)?,
                            &k.to_dtype(DType::F32)?,
                            &v.to_dtype(DType::F32)?,
                            scale,
                        )?;
                        let on_device = |t: &Tensor| -> Result<Tensor> {
                            if !nan_tail {
                                return t.to_device(&device);
                            }
                            let tail = Tensor::full(f32::NAN, (B, H, TAIL, dim), &Device::Cpu)?
                                .to_dtype(dtype)?;
                            Tensor::cat(&[t, &tail], 2)?
                                .to_device(&device)?
                                .narrow(2, 0, kl)
                        };
                        let out = candle_nn::ops::sdpa(
                            &q.to_device(&device)?,
                            &on_device(&k)?,
                            &on_device(&v)?,
                            None,
                            true,
                            scale as f32,
                            1.,
                        )?;
                        let out = out.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                        let diffs = (out - &expected)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
                        let max_diff = diffs.iter().fold(0f32, |m, &d| {
                            if d.is_nan() {
                                f32::INFINITY
                            } else {
                                m.max(d)
                            }
                        });
                        if max_diff > tolerance {
                            failures.push(format!(
                                "head dim {dim} {dtype:?} ql={ql} kl={kl}: max diff {max_diff}"
                            ));
                        }
                    }
                }
            }
        }
        Ok(failures)
    }

    #[test]
    fn sdpa_full_causal_fewer_queries_than_keys() -> Result<()> {
        let failures = causal_fewer_queries_failures(false)?;
        assert!(
            failures.is_empty(),
            "{} shapes differ from the reference:\n{}",
            failures.len(),
            failures.join("\n")
        );
        Ok(())
    }

    #[test]
    fn sdpa_full_causal_reads_no_key_past_kv_len() -> Result<()> {
        let failures = causal_fewer_queries_failures(true)?;
        assert!(
            failures.is_empty(),
            "{} shapes differ from the reference:\n{}",
            failures.len(),
            failures.join("\n")
        );
        Ok(())
    }
}
