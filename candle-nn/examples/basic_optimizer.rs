#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{
    backprop::{Bwd, BwdDevice},
    CpuDevice, CpuStorage, DType, Result, Tensor,
};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

fn gen_data() -> Result<(Tensor<Bwd<CpuStorage>>, Tensor<Bwd<CpuStorage>>)> {
    // Generate some sample linear data.
    let device = BwdDevice::from(&CpuDevice);
    let w_gen: Tensor<Bwd<CpuStorage>> = Tensor::new(&[[3f32, 1.]], &device)?;
    let b_gen = Tensor::new(-2f32, &device)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &device)?;
    let sample_ys = gen.forward(&sample_xs)?;
    Ok((sample_xs, sample_ys))
}

fn main() -> Result<()> {
    let (sample_xs, sample_ys) = gen_data()?;
    let device = BwdDevice::from(&CpuDevice);

    // Use backprop to run a linear regression between samples and get the coefficients back.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = linear(2, 1, vb.pp("linear"))?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    for step in 0..10000 {
        let ys = model.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        println!("{step} {}", loss.to_vec0::<f32>()?);
    }
    Ok(())
}
