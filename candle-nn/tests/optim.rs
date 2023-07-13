#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{Device, Var};
use candle_nn::SGD;

#[test]
fn sgd_optim() -> Result<()> {
    let x = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(0.1);
    sgd.push(&x);
    let xt = x.as_tensor();
    for _step in 0..100 {
        let loss = ((xt - 4.2)? * (xt - 4.2)?)?;
        sgd.backward_step(&loss)?
    }
    assert_eq!(x.to_scalar::<f32>()?, 4.199999);
    Ok(())
}
