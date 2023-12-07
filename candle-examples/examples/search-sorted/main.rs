// This example illustrates how to implement custom operations. These operations can provide their
// own forward pass (CPU and GPU versions) as well as their backward pass.
//
// In this example we add the RMS normalization operation and implement it for f32.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;

use candle::Tensor;
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
    let t1 = Tensor::arange(0_u32, 10_u32, &device)?;
    println!("t1: {t1}");
    println!("t1 shape: {:?}", t1.shape());
    // println!("{t}");
    // let t2 = t.index_select(indexes, dim)]?;
    let t2 = Tensor::from_slice(&[2_u32, 4_u32], &[2], &device)?;
    println!("t2: {t2}");
    println!("t2 shape: {:?}", t2.shape());
    let t3 = t1.apply_op2(&t2, SearchSorted { right: false })?;
    println!("t3: {t3}");
    println!("t3 shape: {:?}", t3.shape());
    Ok(())
}
