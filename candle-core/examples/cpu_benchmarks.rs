/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::quantized::GgmlType;
use candle_core::{Device, Result, Tensor, D};
use clap::{Parser, Subcommand};

fn softmax<D: candle_core::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

// Conv1d example as used in whisper.
struct Conv1d;
impl Benchmark for Conv1d {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let inp = Tensor::randn(0f32, 1., (1, 384, 3000), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (384, 384, 3), &Device::Cpu)?;
        Ok((inp, w))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.conv1d(&d.1, 0, 1, 1)
    }

    const ITERS: usize = 5;
}

// Conv2d example as used in stable-diffusion.
struct Conv2d;
impl Benchmark for Conv2d {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;

    fn preprocess() -> Result<Self::PreProcessData> {
        let inp = Tensor::randn(0f32, 1., (2, 320, 96, 96), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (320, 320, 3, 3), &Device::Cpu)?;
        Ok((inp, w))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.conv2d(&d.1, 0, 1, 1)
    }

    const ITERS: usize = 1;
}

struct Matmul;
impl Benchmark for Matmul {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::randn(0f32, 1., (1024, 1024), &Device::Cpu)?;
        let rhs = Tensor::randn(0f32, 1., (1024, 1024), &Device::Cpu)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 100;
}

// This benchmark is similar to:
// https://github.com/ggerganov/llama.cpp/blob/master/examples/benchmark/benchmark-matmult.cpp
struct QMatMul;
impl Benchmark for QMatMul {
    type PreProcessData = (candle_core::quantized::QMatMul, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let zeros = vec![candle_core::quantized::k_quants::BlockQ4_0::zeros(); 4096 * 11008 / 32];
        let mm = candle_core::quantized::QTensor::new(zeros, (4096, 11008))?;
        let mm = candle_core::quantized::QMatMul::from_qtensor(mm);
        let arg = Tensor::randn(0f32, 1., (128, 11008), &Device::Cpu)?;
        Ok((mm, arg))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.forward(&d.1)
    }

    const ITERS: usize = 100;
}

struct Softmax;
impl Benchmark for Softmax {
    type PreProcessData = Tensor;
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        // Typical whisper tiny size.
        let x = Tensor::randn(0f32, 1., (1, 6, 200, 1500), &Device::Cpu)?;
        Ok(x)
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        softmax(d, D::Minus1)
    }

    const ITERS: usize = 100;
}

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

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Conv1d,
    Conv2d,
    Matmul,
    Qmatmul,
    Softmax,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,

    #[arg(long)]
    iters: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Conv1d => run::<Conv1d>(args.iters)?,
        Task::Conv2d => run::<Conv2d>(args.iters)?,
        Task::Matmul => run::<Matmul>(args.iters)?,
        Task::Softmax => run::<Softmax>(args.iters)?,
        Task::Qmatmul => run::<QMatMul>(args.iters)?,
    }
    Ok(())
}
