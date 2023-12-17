// This example illustrates how to implement custom operations. These operations can provide their
// own forward pass (CPU and GPU versions) as well as their backward pass.
//
// In this example we add the RMS normalization operation and implement it for f32.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[allow(unused)]
mod cuda_kernels;

use clap::Parser;

use candle::{Shape, Tensor};
mod search_sorted;
use search_sorted::SearchSorted;
use std::fmt::Debug;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let ss: Vec<f32> = vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.];
    let ss_shape = Shape::from_dims(&[2, 5]);
    let vals: Vec<f32> = vec![3., 6., 9., 3., 6., 9.];
    let vals_shape = Shape::from_dims(&[2, 3]);

    println!("Search sorted left");
    let t1 = Tensor::from_vec::<_, f32>(ss, &ss_shape, &device).unwrap();
    let t2 = Tensor::from_vec::<_, f32>(vals, &vals_shape, &device).unwrap();
    let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();
    println!("t1: {t1}");
    println!("t2: {t2}");
    println!("t3: {t3}");

    println!("Search sorted right");
    let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
    println!("t1: {t1}");
    println!("t2: {t2}");
    println!("t3: {t3}");

    Ok(())
}
