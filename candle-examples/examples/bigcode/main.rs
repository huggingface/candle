#![allow(unused)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use clap::Parser;

mod model;
use model::{Config, GPTBigCode};

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
    let args = Args::parse();
    Ok(())
}
