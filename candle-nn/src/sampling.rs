use candle::{Result, Tensor};

/// Sample according to the Gumbel-Softmax distribution.
pub fn gumbel_softmax<D: candle::shape::Dim>(
    logits: &Tensor,
    temperature: f64,
    dim: D,
) -> Result<Tensor> {
    if temperature <= 0.0 {
        logits.argmax(dim)
    } else if temperature == 1.0 {
        let minus_g = logits.rand_like(1e-7, 0.999)?.log()?.neg()?.log()?;
        let sampled = (logits - minus_g)?.argmax(dim)?;
        Ok(sampled)
    } else {
        let minus_g = logits.rand_like(1e-7, 0.999)?.log()?.neg()?.log()?;
        let sampled = (logits + minus_g * (-temperature))?.argmax(dim)?;
        Ok(sampled)
    }
}
