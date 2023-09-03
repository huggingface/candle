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

use anyhow::{Error as E, Result};
use candle::DType;
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// The tokenizer config.
    #[arg(long)]
    tokenizer: Option<String>,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let tokenizer = match args.tokenizer {
        Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        None => Api::new()?
            .model("facebook/musicgen-small".to_string())
            .get("tokenizer.json")?,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let _tokenizer = tokenizer.with_padding(None).with_truncation(None);

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .repo(Repo::with_revision(
                "facebook/musicgen-small".to_string(),
                RepoType::Model,
                "refs/pr/13".to_string(),
            ))
            .get("model.safetensors")?,
    };
    let model = unsafe { candle::safetensors::MmapedFile::new(model)? };
    let model = model.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![model], DTYPE, &device);
    let config = GenConfig::small();
    let _model = MusicgenForConditionalGeneration::load(vb, config)?;
    Ok(())
}
