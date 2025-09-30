#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py
// TODO: Add an offline mode.
// TODO: Add a KV cache.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod musicgen_model;

use musicgen_model::{GenConfig, MusicgenForConditionalGeneration};

use anyhow::{Error as E, Result};
use candle::{BackendDevice, BackendStorage, DType, Tensor};
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

    #[arg(
        long,
        default_value = "90s rock song with loud guitars and heavy drums"
    )]
    prompt: String,
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    if args.cpu {
        run::<candle::CpuStorage>(args)?;
    } else if candle::utils::cuda_is_available() {
        run::<candle::CudaStorage>(args)?;
    } else if candle::utils::metal_is_available() {
        run::<candle::MetalStorage>(args)?;
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        run::<candle::CpuStorage>(args)?;
    }
    Ok(())
}

fn run<B: BackendStorage>(args: Args) -> Result<()> {
    use tokenizers::Tokenizer;

    let tokenizer = match args.tokenizer {
        Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        None => Api::new()?
            .model("facebook/musicgen-small".to_string())
            .get("tokenizer.json")?,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

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

    let device = B::Device::new(0)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DTYPE, &device)? };
    let config = GenConfig::small();
    let mut model: MusicgenForConditionalGeneration<B> =
        MusicgenForConditionalGeneration::load(vb, config)?;

    let tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("tokens: {tokens:?}");
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    println!("{tokens:?}");
    let embeds = model.text_encoder.forward(&tokens)?;
    println!("{embeds}");

    Ok(())
}
