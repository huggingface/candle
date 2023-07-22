#![allow(dead_code)]
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;

use candle::backend::BackendStorage;
use candle::cpu_backend;
use candle::{CpuStorage, CustomOp1, DType, Device, Error, Layout, Result, Shape, Tensor};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

struct LayerNorm;

impl CustomOp1 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        todo!()
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _: &candle::CudaStorage,
        _: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let t = Tensor::arange(0f32, 14f32, &device)?.reshape((2, 7))?;
    println!("{t}");
    Ok(())
}
