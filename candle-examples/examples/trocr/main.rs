#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::{trocr, vit};

use tokenizers::Tokenizer;
mod image_processor;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    #[value(name = "base")]
    BaseHandwritten,
    #[value(name = "large")]
    LargeHandwritten,
    BasePrinted,
    LargePrinted,
}

impl Which {
    fn repo_and_branch_name(&self) -> (&str, &str) {
        match self {
            Self::BaseHandwritten => ("microsoft/trocr-base-handwritten", "refs/pr/3"),
            Self::LargeHandwritten => ("microsoft/trocr-large-handwritten", "refs/pr/6"),
            Self::BasePrinted => ("microsoft/trocr-base-printed", "refs/pr/7"),
            Self::LargePrinted => ("microsoft/trocr-large-printed", "main"),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Config {
    encoder: vit::Config,
    decoder: trocr::TrOCRConfig,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    /// Choose the variant of the model to run.
    #[arg(long, default_value = "base")]
    which: Which,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The image file to be processed.
    #[arg(long)]
    image: String,

    /// Tokenization config.
    #[arg(long)]
    tokenizer: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let api = hf_hub::api::sync::Api::new()?;

    let mut tokenizer_dec = {
        let tokenizer_file = match args.tokenizer {
            None => api
                .model(String::from("ToluClassics/candle-trocr-tokenizer"))
                .get("tokenizer.json")?,
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;
        TokenOutputStream::new(tokenizer)
    };
    let device = candle_examples::device(args.cpu)?;

    let vb = {
        let model = match args.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let (repo, branch) = args.which.repo_and_branch_name();
                api.repo(hf_hub::Repo::with_revision(
                    repo.to_string(),
                    hf_hub::RepoType::Model,
                    branch.to_string(),
                ))
                .get("model.safetensors")?
            }
        };
        println!("model: {:?}", model);
        unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? }
    };

    let (encoder_config, decoder_config) = {
        let (repo, branch) = args.which.repo_and_branch_name();
        let config_filename = api
            .repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                branch.to_string(),
            ))
            .get("config.json")?;
        let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        (config.encoder, config.decoder)
    };
    let mut model = trocr::TrOCRModel::new(&encoder_config, &decoder_config, vb)?;

    let processor_config = image_processor::ProcessorConfig::default();
    let processor = image_processor::ViTImageProcessor::new(&processor_config);

    let image = vec![args.image.as_str()];
    let image = processor.preprocess(image)?.to_device(&device)?;

    let encoder_xs = model.encoder().forward(&image)?;

    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);

    let mut token_ids: Vec<u32> = vec![decoder_config.decoder_start_token_id];
    for index in 0..1000 {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;

        let logits = model.decode(&input_ids, &encoder_xs, start_pos)?;

        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        token_ids.push(token);

        if let Some(t) = tokenizer_dec.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if token == decoder_config.eos_token_id {
            break;
        }
    }

    if let Some(rest) = tokenizer_dec.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();

    Ok(())
}
