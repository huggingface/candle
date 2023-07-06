#![allow(dead_code)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use clap::Parser;

mod model;
use model::{Config, Falcon, VarBuilder};

#[cfg(feature = "mkl")]
const DTYPE: DType = DType::F32;
#[cfg(not(feature = "mkl"))]
const DTYPE: DType = DType::BF16;

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

    #[arg(long, default_value = "tiiuae/falcon-7b")]
    model_id: String,

    #[arg(long, default_value = "refs/pr/43")]
    revision: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    use candle_hub::{api::Api, Repo, RepoType};
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = Repo::with_revision(args.model_id, RepoType::Model, args.revision);
    let tokenizer_filename = api.get(&repo, "tokenizer.json").await?;
    let mut filenames = vec![];
    for rfilename in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        let filename = api.get(&repo, rfilename).await?;
        filenames.push(filename);
    }
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let weights = filenames
        .iter()
        .map(|f| Ok(unsafe { candle::safetensors::MmapedFile::new(f)? }))
        .collect::<Result<Vec<_>>>()?;
    let weights = weights
        .iter()
        .map(|f| Ok(f.deserialize()?))
        .collect::<Result<Vec<_>>>()?;

    let vb = VarBuilder::from_safetensors(weights, DTYPE, &device);
    let config = Config::falcon7b();
    config.validate()?;
    let mut model = Falcon::load(&vb, config)?;
    println!("loaded the model in {:?}", start.elapsed());

    let tokens = tokenizer
        .encode(args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&tokens)?;
    println!("{}", logits);
    Ok(())
}
