#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::fmt::Display;
use std::path::PathBuf;

use anyhow::bail;
use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{debertav2::Config as DebertaV2Config, provence::ProvenceModel};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

enum TaskType {
    Reranker(Box<ProvenceModel>),
}

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ArgsTask {
    Reranker,
}

impl Display for ArgsTask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ArgsTask::Reranker => write!(f, "reranker"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model id to use from HuggingFace
    #[arg(
        long,
        default_value = "naver/provence-reranker-debertav3-v1",
        group = "model_source",
        conflicts_with = "model_path"
    )]
    model_id: String,

    /// Local model path
    #[arg(long, group = "model_source", conflicts_with = "model_id")]
    model_path: Option<PathBuf>,

    /// Revision of the model to use (default: "main")
    #[arg(long, default_value = "main")]
    revision: String,

    /// Question
    #[arg(
        short,
        long,
        default_value = "What goes on the bottom of Shepherd's pie?"
    )]
    question: String,

    /// Context (either repeat the flag or provide a comma-separated list)
    #[arg(
        short,
        long,
        num_args = 1..,
        default_values = &[
            "Shepherd’s pie. History. In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top. Variations and similar dishes. Other potato-topped pies include: The modern ”Cumberland pie” is a version with either beef or lamb and a layer of bread- crumbs and cheese on top. In medieval times, and modern-day Cumbria, the pastry crust had a filling of meat with fruits and spices.. In Quebec, a varia- tion on the cottage pie is called ”Paˆte ́ chinois”. It is made with ground beef on the bottom layer, canned corn in the middle, and mashed potato on top.. The ”shepherdess pie” is a vegetarian version made without meat, or a vegan version made without meat and dairy.. In the Netherlands, a very similar dish called ”philosopher’s stew” () often adds ingredients like beans, apples, prunes, or apple sauce.. In Brazil, a dish called in refers to the fact that a manioc puree hides a layer of sun-dried meat.",
        ]
    )]
    context: Vec<String>,

    /// Which task to run
    #[arg(long, default_value_t = ArgsTask::Reranker)]
    task: ArgsTask,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(TaskType, DebertaV2Config, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;

        // Get files from either the HuggingFace API, or from a specified local directory.
        let (config_filename, tokenizer_filename, weights_filename) =
            get_model_files(&self.model_path, &self.model_id, &self.revision)?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: DebertaV2Config = serde_json::from_str(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {e}")))?;

        tokenizer.with_padding(Some(PaddingParams::default()));

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle_transformers::models::debertav2::DTYPE,
                &device,
            )?
        };

        let vb = vb.set_prefix("deberta");

        match self.task {
            ArgsTask::Reranker => Ok((
                TaskType::Reranker(ProvenceModel::load(vb, &config)?.into()),
                config,
                tokenizer,
            )),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_load_time = std::time::Instant::now();
    let (task_type, _model_config, tokenizer) = args.build_model_and_tokenizer()?;

    println!(
        "Loaded model and tokenizers in {:?}",
        model_load_time.elapsed()
    );

    let tokenize_time = std::time::Instant::now();

    println!(
        "Tokenized and loaded inputs in {:?}",
        tokenize_time.elapsed()
    );

    match task_type {
        TaskType::Reranker(model) => {
            let question = &args.question;
            let context = &args.context[0];

            let input_text = format!("{} [SEP] {}", question, context);

            let encoding = tokenizer
                .encode(input_text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let input_ids = Tensor::new(encoding.get_ids(), &model.device)?.unsqueeze(0)?;
            let attention_mask =
                Tensor::new(encoding.get_attention_mask(), &model.device)?.unsqueeze(0)?;

            let output = model.forward(&input_ids, Some(attention_mask.clone()))?;

            println!("=== Provence Model Output ===\n");

            let ranking_scores = output.ranking_scores.to_vec1::<f32>()?;
            println!("Ranking Score: {:.4}", ranking_scores[0]);
            println!("  (Higher = more relevant context for this query)\n");

            let compression_logits = output.compression_logits;
            let compression_logits = compression_logits.squeeze(2)?;
            let compression_probs = candle_nn::ops::sigmoid(&compression_logits)?;
            let keep_probs_vec = compression_probs.to_vec2::<f32>()?[0].clone();

            let threshold = 0.5;

            let tokens = encoding.get_ids();

            let sep_index = tokens
                .iter()
                .position(|id| tokenizer.decode(&[*id], false).unwrap_or_default() == "[SEP]")
                .unwrap_or(0);

            let mut kept_token_ids: Vec<u32> = Vec::with_capacity(tokens.len());
            let mut kept_context_ids: Vec<u32> = Vec::new();
            let mut removed_context_ids: Vec<u32> = Vec::new();

            for (i, token_id) in tokens.iter().enumerate() {
                if i <= sep_index {
                    kept_token_ids.push(*token_id);
                } else if keep_probs_vec[i] > threshold {
                    kept_token_ids.push(*token_id);
                    kept_context_ids.push(*token_id);
                } else {
                    removed_context_ids.push(*token_id);
                }
            }

            // let pruned_text = tokenizer
            //     .decode(&kept_token_ids, true)
            //     .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

            let pruned_context_text = tokenizer
                .decode(&kept_context_ids, true)
                .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

            let removed_text = tokenizer
                .decode(&removed_context_ids, true)
                .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

            println!("Original Context Length (chars): {}", context.len());

            println!(
                "Pruned Context Length (chars): {}",
                pruned_context_text.len()
            );

            let compression_rate = if !context.is_empty() {
                (1.0 - pruned_context_text.len() as f32 / context.len() as f32) * 100.0
            } else {
                0.0
            };

            println!("Compression Rate (context-only): {:.1}%", compression_rate);

            println!(
                "Kept context tokens: {} | Removed context tokens: {}",
                kept_context_ids.len(),
                removed_context_ids.len()
            );

            println!("\nQuestion:\n{}", question);
            println!("\nPruned (pruned context):\n{}\n", pruned_context_text);

            let print_removed_max = 200;

            if !removed_text.is_empty() {
                println!(
                    "Removed Context (first {} chars):\n{}\n",
                    print_removed_max,
                    &removed_text[..removed_text.len().min(print_removed_max)]
                );
            } else {
                println!(
                    "Removed Context (first {} chars): <none>\n",
                    print_removed_max
                );
            }

            println!("=== Token-level Analysis (first 80 tokens) ===");
            for (i, token_id) in tokens.iter().enumerate().take(80) {
                let token_str = tokenizer.decode(&[*token_id], false).unwrap_or_default();
                let prob = keep_probs_vec[i];

                let status = if i <= sep_index {
                    "KEEP (Q/SPECIAL)"
                } else if prob > threshold {
                    "KEEP"
                } else {
                    "DROP"
                };

                println!(
                    "{:3}: {:20} prob={:.3} -> {}",
                    i,
                    format!("'{}'", token_str),
                    prob,
                    status
                );
            }
        }
    }

    Ok(())
}

fn get_model_files(
    model_path: &Option<PathBuf>,
    model_id: &str,
    revision: &str,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let config_filename = "config.json";
    let tokenizer_filename = "tokenizer.json";
    let weights_filename = "model.safetensors";

    let config;
    let tokenizer;
    let weights;

    match model_path {
        Some(base_path) => {
            if !base_path.is_dir() {
                bail!("Model path {} is not a directory.", base_path.display())
            }

            config = base_path.join(config_filename);
            tokenizer = base_path.join(tokenizer_filename);
            weights = base_path.join(weights_filename);
        }
        None => {
            let repo =
                Repo::with_revision(model_id.to_owned(), RepoType::Model, revision.to_owned());

            let api = Api::new()?;
            let api = api.repo(repo);

            config = api.get(config_filename)?;
            tokenizer = api.get(tokenizer_filename)?;
            weights = api.get(weights_filename)?;
        }
    }

    Ok((config, tokenizer, weights))
}

// From xml-roberta

#[derive(Debug)]
pub enum TokenizeInput<'a> {
    Single(&'a [String]),
    Pairs(&'a [(String, String)]),
}

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    input: TokenizeInput,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = get_tokens(tokenizer, input)?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&token_ids, 0)?)
}

pub fn get_attention_mask(
    tokenizer: &Tokenizer,
    input: TokenizeInput,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = get_tokens(tokenizer, input)?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&attention_mask, 0)?)
}

fn get_tokens(tokenizer: &Tokenizer, input: TokenizeInput) -> anyhow::Result<Vec<Encoding>> {
    let tokens = match input {
        TokenizeInput::Single(text_batch) => tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?,
        TokenizeInput::Pairs(pairs) => tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?,
    };

    Ok(tokens)
}
