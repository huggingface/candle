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
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DebertaV2SeqClassificationModel,
};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

enum TaskType {
    Reranker(Box<DebertaV2SeqClassificationModel>),
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

    /// Query string
    #[arg(short, long, default_value = "what is panda?")]
    query: String,

    /// Documents (either repeat the flag or provide a comma-separated list)
    #[arg(
        short,
        long,
        num_args = 1..,
        default_values = &[
            "South Korea is a country in East Asia.",
            "There are forests in the mountains.",
            "Pandas look like bears.",
            "There are some animals with black and white fur.",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    )]
    documents: Vec<String>,

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

        let id2label = if let Some(id2label) = &config.id2label {
            id2label.clone()
        } else {
            bail!("Id2Label not found in the model configuration nor specified as a parameter")
        };

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
                TaskType::Reranker(
                    DebertaV2SeqClassificationModel::load(vb, &config, Some(id2label.clone()))?
                        .into(),
                ),
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
        TaskType::Reranker(classification_model) => {
            // create pairs of query and documents
            let pairs = args
                .documents
                .iter()
                .map(|doc| (args.query.clone(), doc.clone()))
                .collect::<Vec<_>>();

            dbg!(&args.documents);

            let input_ids = tokenize_batch(
                &tokenizer,
                TokenizeInput::Pairs(&pairs),
                &classification_model.device,
            )?;

            let attention_mask = get_attention_mask(
                &tokenizer,
                TokenizeInput::Pairs(&pairs),
                &classification_model.device,
            )?;

            let token_type_ids = Tensor::zeros(
                input_ids.dims(),
                input_ids.dtype(),
                &classification_model.device,
            )?;

            let output = classification_model.forward(
                &input_ids,
                Some(token_type_ids),
                Some(attention_mask),
            )?;

            let output = candle_nn::ops::sigmoid(&output)?.t().unwrap();

            let ranks = output
                .arg_sort_last_dim(false)?
                .to_vec2::<u32>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            println!("\nRanking Results:");
            println!("{:-<80}", "");

            args.documents.iter().enumerate().for_each(|(idx, doc)| {
                let rank = ranks.iter().position(|&r| r == idx as u32).unwrap();

                let score = output
                    .get_on_dim(1, idx)
                    .unwrap()
                    .to_dtype(candle::DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();

                println!("Rank #{:<2} | Score: {:.4} | {}", rank + 1, score[0], doc);
            });

            println!("{:-<80}", "");
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
