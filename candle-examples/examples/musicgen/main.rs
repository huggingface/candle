#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod model;
use model::{Config, VarBuilder};

use anyhow::{Error as E, Result};
use candle::{DType, Device};
use clap::Parser;

const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    tokenizer_config: String,

    #[arg(long)]
    weights: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    let mut tokenizer = Tokenizer::from_file(args.tokenizer_config).map_err(E::msg)?;
    let _tokenizer = tokenizer.with_padding(None).with_truncation(None);

    let weights = unsafe { candle::safetensors::MmapedFile::new(args.weights)? };
    let weights = weights.deserialize()?;
    let _vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
    let _config = Config::default();
    Ok(())
}
