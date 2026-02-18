#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Input audio file in wav format.
    #[arg(long)]
    input: Option<String>,

    /// Model id to load from Hugging Face.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-0.6B")]
    model_id: String,

    /// Optional forced language (e.g. English, Chinese).
    #[arg(long)]
    language: Option<String>,

    /// Run in streaming mode.
    #[arg(long, default_value_t = false)]
    stream: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    println!("qwen3-asr example scaffold");
    println!("device: {device:?}");
    println!("model_id: {}", args.model_id);
    if let Some(input) = args.input {
        println!("input: {input}");
    } else {
        println!("input: <none>");
    }
    if let Some(language) = args.language {
        println!("language: {language}");
    }
    println!("stream: {}", args.stream);
    println!("status: scaffold complete, model port follows in next commits");

    Ok(())
}
