#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::trocr;

use tokenizers::Tokenizer;
mod image_processor;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    Base,
    Large,
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

    /// Text to be translated
    #[arg(long)]
    image: String,
}

pub fn main() -> anyhow::Result<()> {
    use hf_hub::api::sync::Api;
    let args = Args::parse();

    let tokenizer_dec = {
        let tokenizer = Api::new()?
            .model(String::from("ToluClassics/candle-trocr-tokenizer"))
            .get("tokenizer.json")?;

        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };

    let mut tokenizer_dec = TokenOutputStream::new(tokenizer_dec);

    let device = candle_examples::device(args.cpu)?;

    let vb = {
        let model = match args.model {
            Some(model) => std::path::PathBuf::from(model),
            None => match args.which {
                Which::Base => Api::new()?
                    .repo(hf_hub::Repo::with_revision(
                        "microsoft/trocr-base-handwritten".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/3".to_string(),
                    ))
                    .get("model.safetensors")?,
                Which::Large => Api::new()?
                    .repo(hf_hub::Repo::with_revision(
                        "microsoft/trocr-large-handwritten".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/6".to_string(),
                    ))
                    .get("model.safetensors")?,
            },
        };
        println!("model: {:?}", model);
        unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? }
    };

    let encoder_config = match args.which {
        Which::Base => candle_transformers::models::vit::Config::microsoft_trocr_base_handwritten(),
        Which::Large => {
            candle_transformers::models::vit::Config::microsoft_trocr_base_handwritten()
        }
    };

    let decoder_config = trocr::TrOCRConfig::default();
    let mut model = trocr::TrOCRModel::new(&encoder_config, &decoder_config, vb)?;

    let config = image_processor::ProcessorConfig::default();
    let processor = image_processor::ViTImageProcessor::new(&config);

    let image = vec![args.image.as_str()];
    let image = processor.preprocess(image)?;

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
