use candle::{Device, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

#[test]
fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 0);
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_top_p_matches_full_sort() -> Result<()> {
    use rand::{Rng, SeedableRng};

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    for n in [1, 31, 1023, 1024, 1025, 4096] {
        for kind in 0..4 {
            let mut probabilities: Vec<f32> = (0..n)
                .map(|i| match kind {
                    0 => rng.random(),
                    1 => rng.random_range(0..8) as f32,
                    2 => 1.,
                    _ => (-(i as f32) * 0.1).exp(),
                })
                .collect();
            let sum: f32 = probabilities.iter().sum();
            probabilities.iter_mut().for_each(|v| *v /= sum);
            let logits = Tensor::zeros(n, candle::DType::F32, &Device::Cpu)?;
            for p in [f32::MIN_POSITIVE, 0.1, 0.5, 0.9, 0.999999] {
                let mut expected = probabilities.clone();
                let mut indices: Vec<_> = (0..n).collect();
                indices.sort_by(|&i, &j| expected[j].total_cmp(&expected[i]));
                let mut cumsum = 0.;
                for index in indices {
                    if cumsum >= p {
                        expected[index] = 0.;
                    } else {
                        cumsum += expected[index];
                    }
                }
                for seed in 0..8 {
                    let mut reference = LogitsProcessor::new(seed, Some(1.), None);
                    let mut sampler = LogitsProcessor::new(seed, Some(1.), Some(p as f64));
                    for _ in 0..3 {
                        assert_eq!(
                            sampler.sample_f(&logits, |v| v.copy_from_slice(&probabilities))?,
                            reference.sample_f(&logits, |v| v.copy_from_slice(&expected))?,
                            "n={n}, kind={kind}, p={p}, seed={seed}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn sample_top_p_ties() -> Result<()> {
    let logits = Tensor::zeros(4096, candle::DType::F32, &Device::Cpu)?;
    // The stable sort keeps the lowest token indices when the cutoff falls inside a tie.
    for p in [0.125, 0.25, 0.5] {
        let mut sampler = LogitsProcessor::new(42, Some(1.), Some(p));
        for _ in 0..64 {
            assert!(sampler.sample(&logits)? < (p * 4096.) as u32);
        }
    }
    Ok(())
}

#[test]
fn sample_top_p_unnormalized_probabilities() -> Result<()> {
    let probabilities: Vec<f32> = (0..4096)
        .map(|i| if i % 257 == 0 { 0.01 } else { 0.00001 })
        .collect();
    let logits = Tensor::zeros(probabilities.len(), candle::DType::F32, &Device::Cpu)?;
    // The total mass is below top_p, so even the smallest weights must survive.
    let mut reference = LogitsProcessor::new(42, Some(1.), None);
    let mut sampler = LogitsProcessor::new(42, Some(1.), Some(0.9));
    for _ in 0..128 {
        assert_eq!(
            sampler.sample_f(&logits, |v| v.copy_from_slice(&probabilities))?,
            reference.sample_f(&logits, |v| v.copy_from_slice(&probabilities))?
        );
    }
    Ok(())
}

#[test]
fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_gumbel() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    let logits = Tensor::new(&[-1.0, 0.0, 0.2, 1.0], &Device::Cpu)?;
    let sm = candle_nn::ops::softmax(&logits, 0)?.to_vec1::<f64>()?;
    let mut counts = vec![0f64; 4];
    let samples = 100000;
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        if (counts[i] - sm[i]).abs() > 0.05 {
            panic!("pr mismatch {counts:?} {sm:?}");
        }
    }
    Ok(())
}
