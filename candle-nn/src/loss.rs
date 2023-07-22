use candle::{Result, Tensor};

pub fn nll(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let b_sz = target.dim(0)?;
    inp.gather(target, 1)?
        .sum_all()?
        .affine(-1f64 / b_sz as f64, 0.)
}
