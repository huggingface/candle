#[cfg(feature = "mkl-unlinked")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
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

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

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

// Remember to set env variable before running.
// Use specific commit vs main to reduce chance of URL breaking later from directory layout changes, etc.
// CANDLE_SINGLE_FILE_BINARY_BUILDER_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
// cargo run --example bert_single_file_binary
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

    let device = candle_examples::device(args.cpu)?;
    let (model, mut tokenizer) = build_model_and_tokenizer_from_bytes(&device)?;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;

        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

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

        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

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

pub fn build_model_and_tokenizer_from_bytes(device: &Device) -> Result<(BertModel, Tokenizer)> {
    let config_data = include_bytes!(env!("CANDLE_BUILDTIME_MODEL_CONFIG"));
    let tokenizer_data = include_bytes!(env!("CANDLE_BUILDTIME_MODEL_TOKENIZER"));
    let weights_data = include_bytes!(env!("CANDLE_BUILDTIME_MODEL_WEIGHTS"));

    let config_string = std::str::from_utf8(config_data)?;
    let config: BertConfig = serde_json::from_str(config_string)?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_data).map_err(anyhow::Error::msg)?;
    let var_builder = VarBuilder::from_slice_safetensors(weights_data, DTYPE, device)?;

    init_model_and_tokenizer(tokenizer, &config, var_builder)
}

pub fn init_model_and_tokenizer(
    mut tokenizer: Tokenizer,
    config: &BertConfig,
    var_builder: VarBuilder,
) -> Result<(BertModel, Tokenizer)> {
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let model = BertModel::load(var_builder, config)?;

    Ok((model, tokenizer))
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
