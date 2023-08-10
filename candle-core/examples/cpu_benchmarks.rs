/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{Device, Result, Tensor};
use clap::{Parser, Subcommand};

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

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
        d.0.conv1d(&d.1, 0, 1)
    }

    const ITERS: usize = 5;
}

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
        d.0.conv2d(&d.1, 0, 1)
    }

    const ITERS: usize = 1;
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
    }
    Ok(())
}
