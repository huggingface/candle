use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{
    Config, XLMRobertaForMaskedLM, XLMRobertaForSequenceClassification,
};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    BgeRerankerBase,
    BgeRerankerLarge,
    BgeRerankerBaseV2,
    XLMRobertaBase,
    XLMRobertaLarge,
    XLMRFormalityClassifier,
}

#[derive(Debug, Clone, ValueEnum)]
enum Task {
    FillMask,
    Reranker,
    TextClassification,
}

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

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "bge-reranker-base")]
    model: Model,

    #[arg(long, default_value = "reranker")]
    task: Task,

    // Path to the tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    // Path to the weight files.
    #[arg(long)]
    weight_files: Option<String>,

    // Path to the config file.
    #[arg(long)]
    config_file: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.task {
            Task::FillMask => match args.model {
                Model::XLMRobertaBase => "FacebookAI/xlm-roberta-base".to_string(),
                Model::XLMRobertaLarge => "FacebookAI/xlm-roberta-large".to_string(),
                _ => anyhow::bail!("BGE models are not supported for fill-mask task"),
            },
            Task::Reranker => match args.model {
                Model::BgeRerankerBase => "BAAI/bge-reranker-base".to_string(),
                Model::BgeRerankerLarge => "BAAI/bge-reranker-large".to_string(),
                Model::BgeRerankerBaseV2 => "BAAI/bge-reranker-base-v2-m3".to_string(),
                _ => anyhow::bail!("XLM-RoBERTa models are not supported for reranker task"),
            },
            Task::TextClassification => match args.model {
                Model::XLMRFormalityClassifier => "s-nlp/xlmr_formality_classifier".to_string(),
                _ => anyhow::bail!(
                    "XLM-RoBERTa models are not supported for text classification task"
                ),
            },
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };

    let weights_filename = match args.weight_files {
        Some(files) => PathBuf::from(files),
        None => match repo.get("model.safetensors") {
            Ok(safetensors) => safetensors,
            Err(_) => match repo.get("pytorch_model.bin") {
                Ok(pytorch_model) => pytorch_model,
                Err(e) => {
                    return Err(anyhow::Error::msg(format!("Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {e}")));
                }
            },
        },
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;

    let vb = if weights_filename.ends_with("model.safetensors") {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], candle::DType::F16, &device)
                .unwrap()
        }
    } else {
        println!("Loading weights from pytorch_model.bin");
        VarBuilder::from_pth(&weights_filename, candle::DType::F16, &device).unwrap()
    };
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id,
            ..Default::default()
        }))
        .with_truncation(None)
        .map_err(E::msg)?;

    match args.task {
        Task::FillMask => {
            let prompt = vec![
                "Hello I'm a <mask> model.".to_string(),
                "I'm a <mask> boy.".to_string(),
                "I'm <mask> in berlin.".to_string(),
            ];
            let model = XLMRobertaForMaskedLM::new(&config, vb)?;

            let input_ids = tokenize_batch(&tokenizer, TokenizeInput::Single(&prompt), &device)?;
            let attention_mask =
                get_attention_mask(&tokenizer, TokenizeInput::Single(&prompt), &device)?;

            let token_type_ids = Tensor::zeros(input_ids.dims(), input_ids.dtype(), &device)?;

            let output = model
                .forward(
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                    None,
                    None,
                    None,
                )?
                .to_dtype(candle::DType::F32)?;

            let max_outs = output.argmax(2)?;

            let max_out = max_outs.to_vec2::<u32>()?;
            let max_out_refs: Vec<&[u32]> = max_out.iter().map(|v| v.as_slice()).collect();
            let decoded = tokenizer.decode_batch(&max_out_refs, true).unwrap();
            for (i, sentence) in decoded.iter().enumerate() {
                println!("Sentence: {} : {}", i + 1, sentence);
            }
        }
        Task::Reranker => {
            let query = "what is panda?".to_string();

            let documents = ["South Korea is a country in East Asia.".to_string(),
                "There are forests in the mountains.".to_string(),
                "Pandas look like bears.".to_string(),
                "There are some animals with black and white fur.".to_string(),
                "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.".to_string()];

            // create pairs of query and documents
            let pairs = documents
                .iter()
                .map(|doc| (query.clone(), doc.clone()))
                .collect::<Vec<_>>();
            let input_ids = tokenize_batch(&tokenizer, TokenizeInput::Pairs(&pairs), &device)?;
            let attention_mask =
                get_attention_mask(&tokenizer, TokenizeInput::Pairs(&pairs), &device)?;
            let token_type_ids = Tensor::zeros(input_ids.dims(), input_ids.dtype(), &device)?;

            let model = XLMRobertaForSequenceClassification::new(1, &config, vb)?;

            let output = model.forward(&input_ids, &attention_mask, &token_type_ids)?;
            let output = candle_nn::ops::sigmoid(&output)?.t().unwrap();
            let ranks = output
                .arg_sort_last_dim(false)?
                .to_vec2::<u32>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            println!("\nRanking Results:");
            println!("{:-<80}", "");
            documents.iter().enumerate().for_each(|(idx, doc)| {
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
        Task::TextClassification => {
            let sentences = vec![
                "I like you. I love you".to_string(),
                "Hey, what's up?".to_string(),
                "Siema, co porabiasz?".to_string(),
                "I feel deep regret and sadness about the situation in international politics."
                    .to_string(),
            ];
            let model = XLMRobertaForSequenceClassification::new(2, &config, vb)?;
            let input_ids = tokenize_batch(&tokenizer, TokenizeInput::Single(&sentences), &device)?;

            let attention_mask =
                get_attention_mask(&tokenizer, TokenizeInput::Single(&sentences), &device)?;
            let token_type_ids = Tensor::zeros(input_ids.dims(), input_ids.dtype(), &device)?;

            let logits = model
                .forward(&input_ids, &attention_mask, &token_type_ids)?
                .to_dtype(candle::DType::F32)?;

            let probabilities = softmax(&logits, 1)?;
            let probs_vec = probabilities.to_vec2::<f32>()?;

            println!("Formality Scores:");
            for (i, (text, probs)) in sentences.iter().zip(probs_vec.iter()).enumerate() {
                println!("Text {}: \"{}\"", i + 1, text);
                println!("  formal: {:.4}", probs[0]);
                println!("  informal: {:.4}", probs[1]);
                println!();
            }
        }
    }
    Ok(())
}

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
    let tokens = match input {
        TokenizeInput::Single(text_batch) => tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?,
        TokenizeInput::Pairs(pairs) => tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?,
    };

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
    let tokens = match input {
        TokenizeInput::Single(text_batch) => tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?,
        TokenizeInput::Pairs(pairs) => tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?,
    };

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&attention_mask, 0)?)
}
