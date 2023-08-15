use anyhow::Result;
use clap::Parser;
use std::fs::File;

use candle::quantized::ggml_file::Content;
use candle::{DType, Device};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGML file to load.
    #[arg(long)]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut file = File::open(args.model)?;
    let start = std::time::Instant::now();
    let model = Content::read(&mut file, DType::F16, &Device::Cpu)?;

    println!(
        "Loaded {:?} tensors in {:?}",
        model.tensors.len(),
        start.elapsed()
    );
    Ok(())
}
