#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::Result;
use candle_utils::{
    get_device,
    loader::{load_config, load_model, load_repo, load_tokenizer},
    normalize_l2,
    tokenization::{build_attention_mask, encode_tokens},
};
use clap::Parser;
use tokenizers::PaddingParams;

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

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,
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

    let device = get_device(args.cpu, false)?;

    let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let default_revision = "refs/pr/21".to_string();
    let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let repo = load_repo(&model_id, &revision)?;
    let mut tokenizer = load_tokenizer(&repo, None)?;
    let mut config: Config = load_config(&repo, None)?;

    if args.approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }

    let model: BertModel = load_model(&repo, &device, DTYPE, &config, None)?;

    if let Some(prompt) = args.prompt {
        let _ = tokenizer.with_padding(None).with_truncation(None);
        let (_tokens, token_ids, token_type_ids) = encode_tokens(&[&prompt], &tokenizer, &device)?;

        println!("Loaded and encoded {:?}", start.elapsed());
        for idx in 0..args.n {
            let start = std::time::Instant::now();
            let ys = model.forward(&token_ids, &token_type_ids, None)?;
            if idx == 0 {
                println!("{ys}");
            }
            println!("Took {:?}", start.elapsed());
        }
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let (tokens, token_ids, token_type_ids) = encode_tokens(&sentences, &tokenizer, &device)?;
        let attention_mask = build_attention_mask(&tokens, &device)?;

        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;

        let embeddings = if args.normalize_embeddings {
            normalize_l2(&embeddings)?
        } else {
            embeddings
        };

        println!("pooled embeddings {:?}", embeddings.shape());

        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = embeddings.get(i)?;
            for j in (i + 1)..n_sentences {
                let e_j = embeddings.get(j)?;
                let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
                let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }
    Ok(())
}
