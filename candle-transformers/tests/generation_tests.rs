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
    // The sampler must draw from softmax(logits / temperature); check the
    // empirical distribution over many draws.
    let z: Vec<f64> = [0.1, 0.2, 0.3, 0.4].iter().map(|&v| v / 0.9).collect();
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let expected: Vec<f64> = exps.iter().map(|v| v / sum).collect();
    let samples = 100_000;
    let mut counts = vec![0f64; 4];
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        assert!(
            (counts[i] - expected[i]).abs() < 0.01,
            "token {i}: expected {expected:?}, got {counts:?}"
        );
    }
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    // Nucleus {token2, token3} (cumulative 0.5498 >= 0.5): the clamped
    // tokens must never be sampled, and the nucleus mass must be split in
    // proportion to the (unnormalized) probabilities.
    let samples = 100_000;
    let mut counts = vec![0u32; 4];
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1;
    }
    assert_eq!(counts[0], 0, "clamped token 0 was sampled");
    assert_eq!(counts[1], 0, "clamped token 1 was sampled");
    let p2 = counts[2] as f64 / samples as f64;
    let p3 = counts[3] as f64 / samples as f64;
    assert!(
        (p2 - 0.475).abs() < 0.01 && (p3 - 0.525).abs() < 0.01,
        "top-p split wrong: {counts:?}"
    );
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
    // k = 1: the single candidate (the max-probability token) is always sampled.
    for _ in 0..100 {
        let token = logits_process.sample(&logits)?;
        assert_eq!(token, 3);
    }
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    // k = 2: only the two top-probability tokens may be sampled, in
    // proportion to their probabilities.
    let samples = 100_000;
    let mut counts = vec![0u32; 4];
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1;
    }
    assert_eq!(counts[0], 0, "token 0 is outside the top-k");
    assert_eq!(counts[1], 0, "token 1 is outside the top-k");
    let p2 = counts[2] as f64 / samples as f64;
    let p3 = counts[3] as f64 / samples as f64;
    assert!(
        (p2 - 0.475).abs() < 0.01 && (p3 - 0.525).abs() < 0.01,
        "top-k split wrong: {counts:?}"
    );
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
