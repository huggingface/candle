#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py
// TODO: Add an offline mode.
// TODO: Add a KV cache.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod encodec_model;
mod musicgen_model;
mod nn;
mod t5_model;

use musicgen_model::{GenConfig, MusicgenForConditionalGeneration};
use nn::VarBuilder;

use anyhow::{Error as E, Result};
use candle::DType;
use clap::Parser;

const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: String,

    /// The tokenizer config.
    #[arg(long)]
    tokenizer: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let mut tokenizer = Tokenizer::from_file(args.tokenizer).map_err(E::msg)?;
    let _tokenizer = tokenizer.with_padding(None).with_truncation(None);

    let model = unsafe { candle::safetensors::MmapedFile::new(args.model)? };
    let model = model.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![model], DTYPE, &device);
    let config = GenConfig::small();
    let _model = MusicgenForConditionalGeneration::load(vb, config)?;
    Ok(())
}
