#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::fmt::Display;
use std::path::PathBuf;

use anyhow::bail;
use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config as DebertaV2Config, DebertaV2NERModel};
use candle_transformers::models::debertav2::{DebertaV2SeqClassificationModel, Id2Label};
use candle_transformers::models::debertav2::{NERItem, TextClassificationItem};
use clap::{ArgGroup, Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

enum TaskType {
    Ner(Box<DebertaV2NERModel>),
    TextClassification(Box<DebertaV2SeqClassificationModel>),
}

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ArgsTask {
    /// Named Entity Recognition
    Ner,

    /// Text Classification
    TextClassification,
}

impl Display for ArgsTask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ArgsTask::Ner => write!(f, "ner"),
            ArgsTask::TextClassification => write!(f, "text-classification"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(group(ArgGroup::new("model")
    .required(true)
    .args(&["model_id", "model_path"])))]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model id to use from HuggingFace
    #[arg(long, requires_if("model_id", "revision"))]
    model_id: Option<String>,

    /// Revision of the model to use (default: "main")
    #[arg(long, default_value = "main")]
    revision: String,

    /// Specify a sentence to inference. Specify multiple times to inference multiple sentences.
    #[arg(long = "sentence", name="sentences", num_args = 1..)]
    sentences: Vec<String>,

    /// Use the pytorch weights rather than the by-default safetensors
    #[arg(long)]
    use_pth: bool,

    /// Perform a very basic benchmark on inferencing, using N number of iterations
    #[arg(long)]
    benchmark_iters: Option<usize>,

    /// Which task to run
    #[arg(long, default_value_t = ArgsTask::Ner)]
    task: ArgsTask,

    /// Use model from a specific directory instead of HuggingFace local cache.
    /// Using this ignores model_id and revision args.
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Pass in an Id2Label if the model config does not provide it, in JSON format. Example: --id2label='{"0": "True", "1": "False"}'
    #[arg(long)]
    id2label: Option<String>,
}

impl Args {
    fn build_model_and_tokenizer(
        &self,
    ) -> Result<(TaskType, DebertaV2Config, Tokenizer, Id2Label)> {
        let device = candle_examples::device(self.cpu)?;

        // Get files from either the HuggingFace API, or from a specified local directory.
        let (config_filename, tokenizer_filename, weights_filename) = {
            match &self.model_path {
                Some(base_path) => {
                    if !base_path.is_dir() {
                        bail!("Model path {} is not a directory.", base_path.display())
                    }

                    let config = base_path.join("config.json");
                    let tokenizer = base_path.join("tokenizer.json");
                    let weights = if self.use_pth {
                        base_path.join("pytorch_model.bin")
                    } else {
                        base_path.join("model.safetensors")
                    };
                    (config, tokenizer, weights)
                }
                None => {
                    let repo = Repo::with_revision(
                        self.model_id.as_ref().unwrap().clone(),
                        RepoType::Model,
                        self.revision.clone(),
                    );
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
                }
            }
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: DebertaV2Config = serde_json::from_str(&config)?;

        // Command-line id2label takes precedence. Otherwise, use model config's id2label.
        // If neither is specified, then we can't proceed.
        let id2label = if let Some(id2labelstr) = &self.id2label {
            serde_json::from_str(id2labelstr.as_str())?
        } else if let Some(id2label) = &config.id2label {
            id2label.clone()
        } else {
            bail!("Id2Label not found in the model configuration nor specified as a parameter")
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let vb = if self.use_pth {
            VarBuilder::from_pth(
                &weights_filename,
                candle_transformers::models::debertav2::DTYPE,
                &device,
            )?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    candle_transformers::models::debertav2::DTYPE,
                    &device,
                )?
            }
        };

        let vb = vb.set_prefix("deberta");

        match self.task {
            ArgsTask::Ner => Ok((
                TaskType::Ner(DebertaV2NERModel::load(vb, &config, Some(id2label.clone()))?.into()),
                config,
                tokenizer,
                id2label,
            )),
            ArgsTask::TextClassification => Ok((
                TaskType::TextClassification(
                    DebertaV2SeqClassificationModel::load(vb, &config, Some(id2label.clone()))?
                        .into(),
                ),
                config,
                tokenizer,
                id2label,
            )),
        }
    }
}

fn get_device(model_type: &TaskType) -> &Device {
    match model_type {
        TaskType::Ner(ner_model) => &ner_model.device,
        TaskType::TextClassification(classification_model) => &classification_model.device,
    }
}

struct ModelInput {
    encoding: Vec<Encoding>,
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
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

    let model_load_time = std::time::Instant::now();
    let (task_type, _model_config, tokenizer, id2label) = args.build_model_and_tokenizer()?;

    println!(
        "Loaded model and tokenizers in {:?}",
        model_load_time.elapsed()
    );

    let device = get_device(&task_type);

    let tokenize_time = std::time::Instant::now();

    let model_input: ModelInput = {
        let tokenizer_encodings = tokenizer
            .encode_batch(args.sentences, true)
            .map_err(E::msg)?;

        let mut encoding_stack: Vec<Tensor> = Vec::default();
        let mut attention_mask_stack: Vec<Tensor> = Vec::default();
        let mut token_type_id_stack: Vec<Tensor> = Vec::default();

        for encoding in &tokenizer_encodings {
            encoding_stack.push(Tensor::new(encoding.get_ids(), device)?);
            attention_mask_stack.push(Tensor::new(encoding.get_attention_mask(), device)?);
            token_type_id_stack.push(Tensor::new(encoding.get_type_ids(), device)?);
        }

        ModelInput {
            encoding: tokenizer_encodings,
            input_ids: Tensor::stack(&encoding_stack[..], 0)?,
            attention_mask: Tensor::stack(&attention_mask_stack[..], 0)?,
            token_type_ids: Tensor::stack(&token_type_id_stack[..], 0)?,
        }
    };

    println!(
        "Tokenized and loaded inputs in {:?}",
        tokenize_time.elapsed()
    );

    match task_type {
        TaskType::Ner(ner_model) => {
            if let Some(num_iters) = args.benchmark_iters {
                create_benchmark(num_iters, model_input)(
                    |input_ids, token_type_ids, attention_mask| {
                        ner_model.forward(input_ids, Some(token_type_ids), Some(attention_mask))?;
                        Ok(())
                    },
                )?;

                std::process::exit(0);
            }

            let inference_time = std::time::Instant::now();
            let logits = ner_model.forward(
                &model_input.input_ids,
                Some(model_input.token_type_ids),
                Some(model_input.attention_mask),
            )?;

            println!("Inferenced inputs in {:?}", inference_time.elapsed());

            let max_scores_vec = softmax(&logits, 2)?.max(2)?.to_vec2::<f32>()?;
            let max_indices_vec: Vec<Vec<u32>> = logits.argmax(2)?.to_vec2()?;
            let input_ids = model_input.input_ids.to_vec2::<u32>()?;
            let mut results: Vec<Vec<NERItem>> = Default::default();

            for (input_row_idx, input_id_row) in input_ids.iter().enumerate() {
                let mut current_row_result: Vec<NERItem> = Default::default();
                let current_row_encoding = model_input.encoding.get(input_row_idx).unwrap();
                let current_row_tokens = current_row_encoding.get_tokens();
                let current_row_max_scores = max_scores_vec.get(input_row_idx).unwrap();

                for (input_id_idx, _input_id) in input_id_row.iter().enumerate() {
                    // Do not include special characters in output
                    if current_row_encoding.get_special_tokens_mask()[input_id_idx] == 1 {
                        continue;
                    }

                    let max_label_idx = max_indices_vec
                        .get(input_row_idx)
                        .unwrap()
                        .get(input_id_idx)
                        .unwrap();

                    let label = id2label.get(max_label_idx).unwrap().clone();

                    // Do not include those labeled as "O" ("Other")
                    if label == "O" {
                        continue;
                    }

                    current_row_result.push(NERItem {
                        entity: label,
                        word: current_row_tokens[input_id_idx].clone(),
                        score: current_row_max_scores[input_id_idx],
                        start: current_row_encoding.get_offsets()[input_id_idx].0,
                        end: current_row_encoding.get_offsets()[input_id_idx].1,
                        index: input_id_idx,
                    });
                }

                results.push(current_row_result);
            }

            println!("\n{results:?}");
        }

        TaskType::TextClassification(classification_model) => {
            let inference_time = std::time::Instant::now();
            let logits = classification_model.forward(
                &model_input.input_ids,
                Some(model_input.token_type_ids),
                Some(model_input.attention_mask),
            )?;

            println!("Inferenced inputs in {:?}", inference_time.elapsed());

            let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
            let scores = softmax(&logits, 1)?.max(1)?.to_vec1::<f32>()?;
            let mut results = Vec::<TextClassificationItem>::default();

            for (idx, prediction) in predictions.iter().enumerate() {
                results.push(TextClassificationItem {
                    label: id2label[prediction].clone(),
                    score: scores[idx],
                });
            }

            println!("\n{results:?}");
        }
    }
    Ok(())
}

fn create_benchmark<F>(
    num_iters: usize,
    model_input: ModelInput,
) -> impl Fn(F) -> Result<(), candle::Error>
where
    F: Fn(&Tensor, Tensor, Tensor) -> Result<(), candle::Error>,
{
    move |code: F| -> Result<(), candle::Error> {
        println!("Running {num_iters} iterations...");
        let mut durations = Vec::with_capacity(num_iters);
        for _ in 0..num_iters {
            let token_type_ids = model_input.token_type_ids.clone();
            let attention_mask = model_input.attention_mask.clone();
            let start = std::time::Instant::now();
            code(&model_input.input_ids, token_type_ids, attention_mask)?;
            let duration = start.elapsed();
            durations.push(duration.as_nanos());
        }

        let min_time = *durations.iter().min().unwrap();
        let max_time = *durations.iter().max().unwrap();
        let avg_time = durations.iter().sum::<u128>() as f64 / num_iters as f64;

        println!("Min time: {:.3} ms", min_time as f64 / 1_000_000.0);
        println!("Avg time: {:.3} ms", avg_time / 1_000_000.0);
        println!("Max time: {:.3} ms", max_time as f64 / 1_000_000.0);
        Ok(())
    }
}
