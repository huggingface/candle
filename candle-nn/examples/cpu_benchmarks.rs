/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::quantized::GgmlType;
use candle::{CpuStorage, Device, Layout, Module, Result, Shape, Tensor, D};
use clap::{Parser, Subcommand};

const CHECK_CONV2D: bool = false;

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

struct Im2Col {
    h_k: usize,
    w_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col {
    fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl candle::CustomOp1 for Im2Col {
    fn name(&self) -> &'static str {
        "im2col"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let &Self {
            h_k,
            w_k,
            stride,
            dilation,
            padding,
        } = self;
        let (b, c, h, w) = layout.shape().dims4()?;
        let (h_out, w_out) = self.hw_out(h, w);
        let slice = storage.as_slice::<f32>()?;
        let src = &slice[layout.start_offset()..];
        let mut dst = vec![0f32; b * h_out * w_out * c * h_k * w_k];
        let (src_s0, src_s1, src_s2, src_s3) = {
            let s = layout.stride();
            (s[0], s[1], s[2], s[3])
        };
        // TODO: provide specialized kernels for the common use cases.
        // - h_k = w_k = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for b_idx in 0..b {
            let src_idx = b_idx * src_s0;
            let dst_idx = b_idx * h_out * w_out * c * h_k * w_k;
            for h_idx in 0..h_out {
                let dst_idx = dst_idx + h_idx * w_out * c * h_k * w_k;
                for w_idx in 0..w_out {
                    let dst_idx = dst_idx + w_idx * c * h_k * w_k;
                    for c_idx in 0..c {
                        let dst_idx = dst_idx + c_idx * h_k * w_k;
                        let src_idx = c_idx * src_s1 + src_idx;
                        for h_k_idx in 0..h_k {
                            let src_h = h_idx * stride + h_k_idx * dilation;
                            if padding != 0 && (src_h < padding || src_h >= h + padding) {
                                continue;
                            }
                            let src_h = src_h - padding;
                            let src_idx = src_idx + src_h * src_s2;
                            let dst_idx = dst_idx + h_k_idx * w_k;
                            for w_k_idx in 0..w_k {
                                let src_w = w_idx * stride + w_k_idx * dilation;
                                if padding != 0 && (src_w < padding || src_w >= w + padding) {
                                    continue;
                                }
                                let src_w = src_w - padding;
                                let src_idx = src_idx + src_w * src_s3;
                                let dst_idx = dst_idx + w_k_idx;
                                dst[dst_idx] = src[src_idx]
                            }
                        }
                    }
                }
            }
        }
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, (b * h_out * w_out, c * h_k * w_k).into()))
    }
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
        d.0.conv1d(&d.1, 0, 1, 1, 1)
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
        d.0.conv2d(&d.1, 0, 1, 1, 1)
    }

    const ITERS: usize = 5;
}

// Conv2d example as used in stable-diffusion, im2col implementation.
struct Conv2dIm2Col;
impl Benchmark for Conv2dIm2Col {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;

    fn preprocess() -> Result<Self::PreProcessData> {
        let inp = Tensor::randn(0f32, 1., (2, 320, 96, 96), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (320, 320, 3, 3), &Device::Cpu)?;
        Ok((inp, w))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        // d.0.conv2d(&d.1, 0, 1, 1, 1)
        let (b, _, h, w) = d.0.dims4()?;
        let (_, _, h_k, w_k) = d.1.dims4()?;
        let op = Im2Col {
            h_k,
            w_k,
            stride: 1,
            dilation: 1,
            padding: 0,
        };
        let (h_out, w_out) = op.hw_out(h, w);
        let col = d.0.apply_op1_no_bwd(&op)?;
        let res = col.matmul(&d.1.flatten_from(1)?.t()?)?;
        let res = res
            .reshape((b, h_out, w_out, ()))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        if CHECK_CONV2D {
            let res2 = d.0.conv2d(&d.1, op.padding, op.stride, op.dilation, 1);
            let diff = (&res - res2)?.sqr()?.mean_all()?;
            println!("{diff}");
        }
        Ok(res)
    }

    const ITERS: usize = 5;
}

struct MatMul;
impl Benchmark for MatMul {
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

struct MatVec;
impl Benchmark for MatVec {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::randn(0f32, 1., (1024 * 4, 1024 * 4), &Device::Cpu)?;
        let rhs = Tensor::randn(0f32, 1., (1024 * 4, 1), &Device::Cpu)?;
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
    type PreProcessData = (candle::quantized::QMatMul, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let zeros = vec![candle::quantized::k_quants::BlockQ4_0::zeros(); 4096 * 11008 / 32];
        let mm = candle::quantized::QTensor::new(
            candle::quantized::QStorage::Cpu(Box::new(zeros)),
            (4096, 11008),
        )?;
        let mm = candle::quantized::QMatMul::from_qtensor(mm)?;
        let arg = Tensor::randn(0f32, 1., (128, 11008), &Device::Cpu)?;
        Ok((mm, arg))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.forward(&d.1)
    }

    const ITERS: usize = 100;
}

struct Cat;
impl Benchmark for Cat {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::randn(0f32, 1., (1, 32, 2000, 128), &Device::Cpu)?;
        let rhs = Tensor::randn(0f32, 1., (1, 32, 1, 128), &Device::Cpu)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        Tensor::cat(&[&d.0, &d.1], 2)
    }

    const ITERS: usize = 1000;
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
        candle_nn::ops::softmax(d, D::Minus1)
    }

    const ITERS: usize = 100;
}

struct SoftmaxLastDim;
impl Benchmark for SoftmaxLastDim {
    type PreProcessData = Tensor;
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        // Typical whisper tiny size.
        let x = Tensor::randn(0f32, 1., (1, 6, 200, 1500), &Device::Cpu)?;
        Ok(x)
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        candle_nn::ops::softmax_last_dim(d)
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
    Conv2dIm2Col,
    Matmul,
    Matvec,
    Qmatmul,
    Softmax,
    SoftmaxLastDim,
    Cat,
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
        Task::Conv2dIm2Col => run::<Conv2dIm2Col>(args.iters)?,
        Task::Matmul => run::<MatMul>(args.iters)?,
        Task::Matvec => run::<MatVec>(args.iters)?,
        Task::Softmax => run::<Softmax>(args.iters)?,
        Task::SoftmaxLastDim => run::<SoftmaxLastDim>(args.iters)?,
        Task::Qmatmul => run::<QMatMul>(args.iters)?,
        Task::Cat => run::<Cat>(args.iters)?,
    }
    Ok(())
}
