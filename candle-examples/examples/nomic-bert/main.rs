#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_transformers::models::nomic_bert::{self, Config, NomicBertModel};

use anyhow::{bail, Error as E, Result};
use candle::{DType, Tensor};
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

    /// The model to use.
    #[arg(long, default_value = "nomic-ai/nomic-embed-text-v1.5")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    /// When set, compute the embedding for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Prefix to prepend (e.g. "search_document: " or "search_query: ").
    #[arg(long)]
    prefix: Option<String>,

    /// Load the model in a specific dtype (f32, f16, bf16). Defaults to f32.
    #[arg(long)]
    dtype: Option<String>,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let repo = Repo::with_revision(args.model_id.clone(), RepoType::Model, args.revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") | None => DType::F32,
        Some(other) => bail!("unsupported dtype: {other}"),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? };
    let model = NomicBertModel::load(vb, &config)?;

    let sentences = if let Some(prompt) = &args.prompt {
        vec![prompt.as_str()]
    } else {
        vec![
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ]
    };

    // Apply prefix if specified.
    let texts: Vec<String> = sentences
        .iter()
        .map(|s| match &args.prefix {
            Some(p) => format!("{p}{s}"),
            None => s.to_string(),
        })
        .collect();

    // Configure padding for batch processing.
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let start = std::time::Instant::now();
    let tokens = tokenizer.encode_batch(texts, true).map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|t| {
            let ids = t.get_ids().to_vec();
            Tensor::new(ids.as_slice(), &device)
        })
        .collect::<candle::Result<Vec<_>>>()?;
    let attention_mask = tokens
        .iter()
        .map(|t| {
            let mask = t.get_attention_mask().to_vec();
            Tensor::new(mask.as_slice(), &device)
        })
        .collect::<candle::Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    println!("Tokenized {:?} in {:?}", token_ids.shape(), start.elapsed());

    let start = std::time::Instant::now();
    let hidden_states = model.forward(&token_ids, None, Some(&attention_mask))?;
    let embeddings = nomic_bert::mean_pooling(&hidden_states, &attention_mask)?;
    let embeddings = nomic_bert::l2_normalize(&embeddings)?;
    println!(
        "Computed embeddings {:?} in {:?}",
        embeddings.shape(),
        start.elapsed()
    );

    if args.prompt.is_some() {
        println!("Embedding (first 10 dims):");
        let vals: Vec<f32> = embeddings.get(0)?.to_dtype(DType::F32)?.to_vec1()?;
        for (i, v) in vals.iter().take(10).enumerate() {
            println!("  [{i}] {v:.6}");
        }
    } else {
        let n = sentences.len();
        let mut similarities = vec![];
        for i in 0..n {
            let e_i = embeddings.get(i)?;
            for j in (i + 1)..n {
                let e_j = embeddings.get(j)?;
                let score = (&e_i * &e_j)?
                    .sum_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                similarities.push((score, i, j));
            }
        }
        similarities.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("\nTop cosine similarities:");
        for &(score, i, j) in similarities.iter().take(5) {
            println!("  {score:.4}  '{}' <-> '{}'", sentences[i], sentences[j]);
        }
    }

    Ok(())
}
