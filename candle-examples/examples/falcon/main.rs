#![allow(dead_code)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle::{DType, Device};
use clap::Parser;

mod model;
use model::{Config, Falcon, VarBuilder};

const DTYPE: DType = DType::F16;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let vb = VarBuilder::zeros(DTYPE, &device);
    let config = Config::default();
    let _model = Falcon::load(&vb, config)?;
    Ok(())
}
