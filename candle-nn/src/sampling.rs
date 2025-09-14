use candle::{BackendStorage, Result, Tensor};

/// Sample according to the Gumbel-Softmax distribution.
pub fn gumbel_softmax<B: BackendStorage, D: candle::shape::Dim>(
    logits: &Tensor<B>,
    temperature: f64,
    dim: D,
) -> Result<Tensor<B>> {
    if temperature <= 0.0 {
        logits.argmax(dim)
    } else {
        // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
        let logits = logits.to_dtype(candle::DType::F32)?;
        let minus_g = logits.rand_like(1e-7f32, 0.999f32)?.log()?.neg()?.log()?;
        if temperature == 1.0 {
            let sampled = (logits - minus_g)?.argmax(dim)?;
            Ok(sampled)
        } else {
            let sampled = (logits + minus_g * (-temperature))?.argmax(dim)?;
            Ok(sampled)
        }
    }
}
