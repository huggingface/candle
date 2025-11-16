#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::fmt;
use std::path::PathBuf;

use anyhow::{bail, Context, Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{debertav2::Config as DebertaV2Config, provence::ProvenceModel};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

enum TaskType {
    Single(Box<ProvenceModel>),
}

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ArgsTask {
    Single,
}

impl fmt::Display for ArgsTask {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArgsTask::Single => write!(f, "single"),
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

    /// Threshold
    #[arg(short, long, default_value = "0.5")]
    threshold: f32,

    /// Which task to run
    #[arg(long, default_value_t = ArgsTask::Single)]
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
            ArgsTask::Single => Ok((
                TaskType::Single(ProvenceModel::load(vb, &config)?.into()),
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
        TaskType::Single(model) => {
            let question = &args.question;
            let context = args.context.first().context("context can't be empty")?;

            // Forward only
            println!("Running forward pass only");

            let input_text = model.format_input(question, context);

            let encoding = tokenizer
                .encode(input_text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let input_ids = Tensor::new(encoding.get_ids(), &model.device)?.unsqueeze(0)?;
            let attention_mask =
                Tensor::new(encoding.get_attention_mask(), &model.device)?.unsqueeze(0)?;

            let output = model.forward(&input_ids, Some(attention_mask.clone()))?;

            println!("Forward pass output");
            dbg!(&output);

            // Simple usage
            println!("Running process helper function");
            let result =
                model.process_single(&tokenizer, question, context, args.threshold, true, true)?;

            println!("Simple output");
            println!("Pruned: {}", result.pruned_context);
            println!("Score: {:.2}", result.reranking_score);
            println!("Compression: {:.1}%", result.compression_rate);

            // Detailed usage
            println!("Detailed output");
            let max_tokens = 80;

            let token_details = result.token_details.context("token details is none")?;

            println!("Ranking Score: {:.4}", result.reranking_score);
            println!("  (Higher = more relevant context for this query)\n");

            println!("Original Context Length (chars): {}", context.len());
            println!(
                "Pruned Context Length (chars): {}",
                result.pruned_context.len()
            );
            println!(
                "Compression Rate (context-only): {:.1}%",
                result.compression_rate
            );

            println!("\nQuestion:\n{}", question);
            println!("\nPruned Context:\n{}\n", result.pruned_context);

            println!("=== Token-level Analysis (first {} tokens) ===", max_tokens);
            for detail in token_details.iter().take(max_tokens) {
                println!(
                    "{:3}: {:20} prob={:.3} -> {}",
                    detail.index,
                    format!("'{}'", detail.token),
                    detail.probability,
                    detail.status
                );
            }
            println!("\nNOTE: With sentence rounding, entire sentences are kept/dropped together");
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
