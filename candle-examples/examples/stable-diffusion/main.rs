#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod attention;
mod clip;
mod ddim;
mod embeddings;
mod resnet;
mod schedulers;
mod stable_diffusion;
mod unet_2d;
mod unet_2d_blocks;
mod utils;
mod vae;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,
}

fn main() -> Result<()> {
    let _args = Args::parse();
    Ok(())
}
