/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// use candle::quantized::GgmlType;
use candle::{DType, Device, Result, Tensor};
// use clap::{Parser, Subcommand};

// fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
//     let dim = dim.to_index(xs.shape(), "softmax")?;
//     let max = xs.max_keepdim(dim)?;
//     let diff = xs.broadcast_sub(&max)?;
//     let num = diff.exp()?;
//     let den = num.sum_keepdim(dim)?;
//     num.broadcast_div(&den)
// }

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

struct Matmul;
impl Benchmark for Matmul {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::randn((1024, 1024), DType::F32, &Device::Cpu, 1.0, 0.0)?;
        let rhs = Tensor::randn((1024, 1024), DType::F32, &Device::Cpu, 1.0, 0.0)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 100;
}

// struct Softmax;
// impl Benchmark for Softmax {
//     type PreProcessData = Tensor;
//     type RunResult = Tensor;
//     fn preprocess() -> Result<Self::PreProcessData> {
//         // Typical whisper tiny size.
//         let x = Tensor::randn(0f32, 1., (1, 6, 200, 1500), &Device::Cpu)?;
//         Ok(x)
//     }
//
//     fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
//         softmax(d, D::Minus1)
//     }
//
//     const ITERS: usize = 100;
// }

fn run<B: Benchmark>(iters: Option<usize>) -> Result<()> {
    use std::hint::black_box;

    let iters = iters.unwrap_or(B::ITERS);
    let d = B::preprocess()?;
    let start = std::time::Instant::now();
    for _iter in 0..iters {
        let _res = black_box(B::run_one(black_box(&d))?);
    }
    println!("{:?}", start.elapsed() / iters as u32);
    Ok(())
}

fn main() -> Result<()> {
    run::<Matmul>(None)?;
    Ok(())
}
