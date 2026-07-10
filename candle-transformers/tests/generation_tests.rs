use candle::{Device, Result, Tensor};
use candle_transformers::generation::{
    apply_frequency_presence_penalty, LogitsProcessor, Sampling,
};

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
fn sample_with_min_p_dominant_token() -> Result<()> {
    // Token 3 holds ~98% of the mass; with min_p = 0.5 every other token falls below
    // 0.5 * max_prob, so sampling is deterministic whatever the seed.
    let logits = Tensor::new(&[0.0, 0.0, 0.0, 5.0], &Device::Cpu)?;
    for seed in [0, 42, 1337] {
        let mut logits_process = LogitsProcessor::from_sampling(
            seed,
            Sampling::MinP {
                p: 0.5,
                temperature: 1.0,
            },
        );
        assert_eq!(logits_process.sample(&logits)?, 3);
    }
    Ok(())
}

#[test]
fn sample_with_min_p_filters_and_renormalizes() -> Result<()> {
    // Probabilities are [1/7, 2/7, 4/7]. With min_p = 0.4 the threshold is 4/7 * 0.4 = 1.6/7:
    // token 0 is filtered out, tokens 1 and 2 survive with renormalized odds 1:2.
    let logits = Tensor::new(&[0.0, 2f32.ln(), 4f32.ln()], &Device::Cpu)?;
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        Sampling::MinP {
            p: 0.4,
            temperature: 1.0,
        },
    );
    let samples = 10000;
    let mut counts = [0usize; 3];
    for _ in 0..samples {
        counts[logits_process.sample(&logits)? as usize] += 1;
    }
    assert_eq!(counts[0], 0, "token below the min-p threshold was sampled");
    let ratio = counts[2] as f64 / counts[1] as f64;
    assert!(
        (ratio - 2.0).abs() < 0.2,
        "renormalized odds off: {counts:?}"
    );
    Ok(())
}

#[test]
fn sample_with_min_p_zero_keeps_all_tokens() -> Result<()> {
    // p <= 0 keeps the whole distribution; every token must eventually be sampled.
    let logits = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &Device::Cpu)?;
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        Sampling::MinP {
            p: 0.0,
            temperature: 1.0,
        },
    );
    let mut seen = [false; 4];
    for _ in 0..1000 {
        seen[logits_process.sample(&logits)? as usize] = true;
    }
    assert_eq!(seen, [true; 4]);
    Ok(())
}

#[test]
fn frequency_presence_penalty_matches_openai_semantics() -> Result<()> {
    let logits = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu)?;
    // Token 2 appears twice, token 3 once.
    let context = [2u32, 2, 3];
    let penalized = apply_frequency_presence_penalty(&logits, &context, 0.5, 0.25)?;
    let penalized = penalized.to_vec1::<f32>()?;
    // logit -= frequency_penalty * count + presence_penalty
    assert_eq!(
        penalized,
        [1.0, 1.0, 1.0 - (0.5 * 2.0 + 0.25), 1.0 - (0.5 + 0.25)]
    );
    Ok(())
}

#[test]
fn frequency_presence_penalty_noop_and_out_of_range() -> Result<()> {
    let logits = Tensor::new(&[0.5f32, -0.5, 2.0], &Device::Cpu)?;
    // Zero penalties leave the logits untouched.
    let same = apply_frequency_presence_penalty(&logits, &[0, 1, 2], 0.0, 0.0)?;
    assert_eq!(same.to_vec1::<f32>()?, logits.to_vec1::<f32>()?);
    // Context tokens outside the vocab are ignored rather than panicking.
    let same = apply_frequency_presence_penalty(&logits, &[100], 1.0, 1.0)?;
    assert_eq!(same.to_vec1::<f32>()?, logits.to_vec1::<f32>()?);
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
