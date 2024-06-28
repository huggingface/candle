#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};

use anyhow::{Error as E, Result};
use candle::Tensor;
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(BertModel, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "BAAI/bge-base-en-v1.5".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        let model = BertModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    let sentences = if let Some(prompt) = args.prompt {
        vec![prompt]
    } else {
        vec![
            "The cat sits outside".to_owned(),
            "some other text that will use padding".to_owned(),
        ]
    };

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }
    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let attention_masks = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let attention_mask = Tensor::stack(&attention_masks, 0)?;
    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    println!("Loaded and encoded {:?}", start.elapsed());

    for idx in 0..args.n {
        let start = std::time::Instant::now();

        let ys = model.forward_with_attention_mask(&token_ids, &token_type_ids, &attention_mask)?;

        let embeddings = ys.get_on_dim(1, 0)?;
        let embeddings = normalize_l2(&embeddings)?;

        if idx == 0 {
            println!("{embeddings}");
        }
        println!("Took {:?}", start.elapsed());
    }

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
