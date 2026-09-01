use candle::{Device, Result, Tensor};
use candle_transformers::utils::apply_no_repeat_ngram;

const NEG_INF: f32 = f32::NEG_INFINITY;

#[test]
fn no_repeat_ngram_bigram() -> Result<()> {
    let device = &Device::Cpu;
    // vocab size 4, all logits equal.
    let logits = Tensor::new(&[0f32, 0., 0., 0.], device)?;
    // context: [1, 2, 1] -> prefix is [1]; token 2 previously followed 1, so it is banned.
    let out = apply_no_repeat_ngram(&logits, 2, &[1, 2, 1])?;
    assert_eq!(out.to_vec1::<f32>()?, [0., 0., NEG_INF, 0.]);
    Ok(())
}

#[test]
fn no_repeat_ngram_trigram() -> Result<()> {
    let device = &Device::Cpu;
    let logits = Tensor::zeros(8, candle::DType::F32, device)?;
    // context: [5, 6, 7, 5, 6] -> prefix [5, 6] was previously followed by 7 -> ban 7.
    let out = apply_no_repeat_ngram(&logits, 3, &[5, 6, 7, 5, 6])?;
    let v = out.to_vec1::<f32>()?;
    assert_eq!(v[7], NEG_INF);
    // Nothing else is banned.
    assert_eq!(v.iter().filter(|x| x.is_infinite()).count(), 1);
    Ok(())
}

#[test]
fn no_repeat_ngram_unigram_bans_all_seen() -> Result<()> {
    let device = &Device::Cpu;
    let logits = Tensor::zeros(4, candle::DType::F32, device)?;
    // ngram_size == 1 bans every token already present in the context.
    let out = apply_no_repeat_ngram(&logits, 1, &[0, 2])?;
    assert_eq!(out.to_vec1::<f32>()?, [NEG_INF, 0., NEG_INF, 0.]);
    Ok(())
}

#[test]
fn no_repeat_ngram_noop() -> Result<()> {
    let device = &Device::Cpu;
    let logits = Tensor::new(&[1f32, 2., 3.], device)?;
    // ngram_size == 0 and context shorter than ngram_size are both no-ops.
    assert_eq!(
        apply_no_repeat_ngram(&logits, 0, &[1, 2])?.to_vec1::<f32>()?,
        [1., 2., 3.]
    );
    assert_eq!(
        apply_no_repeat_ngram(&logits, 3, &[1])?.to_vec1::<f32>()?,
        [1., 2., 3.]
    );
    Ok(())
}
