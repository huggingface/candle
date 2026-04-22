#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::qwen2::{Config, Model};

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{
    utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy},
    Tokenizer,
};

// gte-Qwen1.5-7B-instruct use EOS token as padding token
const EOS_TOKEN: &str = "<|endoftext|>";
const EOS_TOKEN_ID: u32 = 151643;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value = "Alibaba-NLP/gte-Qwen1.5-7B-instruct")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    local_repo: Option<String>,
}

#[derive(Debug)]
struct ConfigFiles {
    pub config: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    pub weights: Vec<std::path::PathBuf>,
}

// Loading the model from the HuggingFace Hub. Network access is required.
fn load_from_hub(model_id: &str, revision: &str) -> Result<ConfigFiles> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    Ok(ConfigFiles {
        config: repo.get("config.json")?,
        tokenizer: repo.get("tokenizer.json")?,
        weights: candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    })
}

// Loading the model from a local directory.
fn load_from_local(local_path: &str) -> Result<ConfigFiles> {
    let local_path = std::path::PathBuf::from(local_path);
    let weight_path = local_path.join("model.safetensors.index.json");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(weight_path)?)?;
    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("`weight map` is not a map"),
        None => panic!("`weight map` not found"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        safetensors_files.insert(
            value
                .as_str()
                .expect("Weight files should be parsed as strings"),
        );
    }
    let safetensors_paths = safetensors_files
        .iter()
        .map(|v| local_path.join(v))
        .collect::<Vec<_>>();
    Ok(ConfigFiles {
        config: local_path.join("config.json"),
        tokenizer: local_path.join("tokenizer.json"),
        weights: safetensors_paths,
    })
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

    // Fetch the model. Do this offline if local path provided.
    println!("Fetching model files...");
    let start = std::time::Instant::now();
    let config_files = match args.local_repo {
        Some(local_path) => load_from_local(&local_path)?,
        None => load_from_hub(&args.model_id, &args.revision)?,
    };
    println!("Model file retrieved in {:?}", start.elapsed());

    // Inputs will be padded to the longest sequence in the batch.
    let padding = PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        pad_to_multiple_of: None,
        pad_id: EOS_TOKEN_ID,
        pad_type_id: 0,
        pad_token: String::from(EOS_TOKEN),
    };

    // Tokenizer setup
    let mut tokenizer = Tokenizer::from_file(config_files.tokenizer).map_err(E::msg)?;
    tokenizer.with_padding(Some(padding));

    // Model initialization
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let config: Config = serde_json::from_slice(&std::fs::read(config_files.config)?)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&config_files.weights, dtype, &device)? };
    let mut model = Model::new(&config, vb)?;
    println!("Model loaded in {:?}", start.elapsed());

    // Encode the queries and the targets
    let instruct = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";
    let documents = vec![
        format!("{instruct}how much protein should a female eat{EOS_TOKEN}"),
        format!("{instruct}summit define{EOS_TOKEN}"),
        format!("As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.{EOS_TOKEN}"),
        format!("Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.{EOS_TOKEN}"),
    ];
    let encoded = tokenizer.encode_batch(documents, true).map_err(E::msg)?;
    let tokens: Vec<&[u32]> = encoded.iter().map(|x| x.get_ids()).collect();
    let tokens = Tensor::new(tokens, &device)?;
    let mask: Vec<&[u32]> = encoded.iter().map(|x| x.get_attention_mask()).collect();
    let mask = Tensor::new(mask, &device)?;

    // Inference
    let start_gen = std::time::Instant::now();
    let logits = model.forward(&tokens, 0, Some(&mask))?;

    // Extract the last hidden states as embeddings since inputs are padded left.
    let (_, seq_len, _) = logits.dims3()?;
    let embd = logits
        .narrow(1, seq_len - 1, 1)?
        .squeeze(1)?
        .to_dtype(DType::F32)?;

    // Calculate the relativity scores. Note the embeddings should be normalized.
    let norm = embd.broadcast_div(&embd.sqr()?.sum_keepdim(1)?.sqrt()?)?;
    let scores = norm.narrow(0, 0, 2)?.matmul(&norm.narrow(0, 2, 2)?.t()?)?;

    // Print the results
    println!("Embedding done in {:?}", start_gen.elapsed());
    println!("Scores: {:?}", scores.to_vec2::<f32>()?);

    Ok(())
}
