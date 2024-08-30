#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

enum Model {
    MixFormer(MixFormer),
    Phi(Phi),
    Phi3(Phi3),
    Quantized(QMixFormer),
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::MixFormer(m) => m.forward(&input)?,
                Model::Phi(m) => m.forward(&input)?,
                Model::Quantized(m) => m.forward(&input)?,
                Model::Phi3(m) => m.forward(&input, pos)?.i((.., 0, ..))?,
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest()? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "1")]
    V1,
    #[value(name = "1.5")]
    V1_5,
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
    #[value(name = "3-medium")]
    V3Medium,
    #[value(name = "2-old")]
    V2Old,
    PuffinPhiV2,
    PhiHermes,
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

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    mmlu_dir: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "2")]
    model: WhichModel,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The dtype to be used for running the model, e.g. f32, bf16, or f16.
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
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => {
            if args.quantized {
                "lmz/candle-quantized-phi".to_string()
            } else {
                match args.model {
                    WhichModel::V1 => "microsoft/phi-1".to_string(),
                    WhichModel::V1_5 => "microsoft/phi-1_5".to_string(),
                    WhichModel::V2 | WhichModel::V2Old => "microsoft/phi-2".to_string(),
                    WhichModel::V3 => "microsoft/Phi-3-mini-4k-instruct".to_string(),
                    WhichModel::V3Medium => "microsoft/Phi-3-medium-4k-instruct".to_string(),
                    WhichModel::PuffinPhiV2 | WhichModel::PhiHermes => {
                        "lmz/candle-quantized-phi".to_string()
                    }
                }
            }
        }
    };
    let revision = match args.revision {
        Some(rev) => rev.to_string(),
        None => {
            if args.quantized {
                "main".to_string()
            } else {
                match args.model {
                    WhichModel::V1 => "refs/pr/8".to_string(),
                    WhichModel::V1_5 => "refs/pr/73".to_string(),
                    WhichModel::V2Old => "834565c23f9b28b96ccbeabe614dd906b6db551a".to_string(),
                    WhichModel::V2
                    | WhichModel::V3
                    | WhichModel::V3Medium
                    | WhichModel::PuffinPhiV2
                    | WhichModel::PhiHermes => "main".to_string(),
                }
            }
        }
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => match args.model {
            WhichModel::V1
            | WhichModel::V1_5
            | WhichModel::V2
            | WhichModel::V2Old
            | WhichModel::V3
            | WhichModel::V3Medium => repo.get("tokenizer.json")?,
            WhichModel::PuffinPhiV2 | WhichModel::PhiHermes => {
                repo.get("tokenizer-puffin-phi-v2.json")?
            }
        },
    };
    let filenames = match args.weight_file {
        Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
        None => {
            if args.quantized {
                match args.model {
                    WhichModel::V1 => vec![repo.get("model-v1-q4k.gguf")?],
                    WhichModel::V1_5 => vec![repo.get("model-q4k.gguf")?],
                    WhichModel::V2 | WhichModel::V2Old => vec![repo.get("model-v2-q4k.gguf")?],
                    WhichModel::PuffinPhiV2 => vec![repo.get("model-puffin-phi-v2-q4k.gguf")?],
                    WhichModel::PhiHermes => vec![repo.get("model-phi-hermes-1_3B-q4k.gguf")?],
                    WhichModel::V3 | WhichModel::V3Medium => anyhow::bail!(
                        "use the quantized or quantized-phi examples for quantized phi-v3"
                    ),
                }
            } else {
                match args.model {
                    WhichModel::V1 | WhichModel::V1_5 => vec![repo.get("model.safetensors")?],
                    WhichModel::V2 | WhichModel::V2Old | WhichModel::V3 | WhichModel::V3Medium => {
                        candle_examples::hub_load_safetensors(
                            &repo,
                            "model.safetensors.index.json",
                        )?
                    }
                    WhichModel::PuffinPhiV2 => vec![repo.get("model-puffin-phi-v2.safetensors")?],
                    WhichModel::PhiHermes => vec![repo.get("model-phi-hermes-1_3B.safetensors")?],
                }
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = || match args.model {
        WhichModel::V1 => Config::v1(),
        WhichModel::V1_5 => Config::v1_5(),
        WhichModel::V2 | WhichModel::V2Old => Config::v2(),
        WhichModel::PuffinPhiV2 => Config::puffin_phi_v2(),
        WhichModel::PhiHermes => Config::phi_hermes_1_3b(),
        WhichModel::V3 | WhichModel::V3Medium => {
            panic!("use the quantized or quantized-phi examples for quantized phi-v3")
        }
    };
    let device = candle_examples::device(args.cpu)?;
    let model = if args.quantized {
        let config = config();
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &filenames[0],
            &device,
        )?;
        let model = match args.model {
            WhichModel::V2 | WhichModel::V2Old => QMixFormer::new_v2(&config, vb)?,
            _ => QMixFormer::new(&config, vb)?,
        };
        Model::Quantized(model)
    } else {
        let dtype = match args.dtype {
            Some(dtype) => std::str::FromStr::from_str(&dtype)?,
            None => {
                if args.model == WhichModel::V3 || args.model == WhichModel::V3Medium {
                    device.bf16_default_to_f32()
                } else {
                    DType::F32
                }
            }
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        match args.model {
            WhichModel::V1 | WhichModel::V1_5 | WhichModel::V2 => {
                let config_filename = repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: PhiConfig = serde_json::from_str(&config)?;
                let phi = Phi::new(&config, vb)?;
                Model::Phi(phi)
            }
            WhichModel::V3 | WhichModel::V3Medium => {
                let config_filename = repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: Phi3Config = serde_json::from_str(&config)?;
                let phi3 = Phi3::new(&config, vb)?;
                Model::Phi3(phi3)
            }
            WhichModel::V2Old => {
                let config = config();
                Model::MixFormer(MixFormer::new_v2(&config, vb)?)
            }
            WhichModel::PhiHermes | WhichModel::PuffinPhiV2 => {
                let config = config();
                Model::MixFormer(MixFormer::new(&config, vb)?)
            }
        }
    };
    println!("loaded the model in {:?}", start.elapsed());

    match (args.prompt, args.mmlu_dir) {
        (None, None) | (Some(_), Some(_)) => {
            anyhow::bail!("exactly one of --prompt and --mmlu-dir must be specified")
        }
        (Some(prompt), None) => {
            let mut pipeline = TextGeneration::new(
                model,
                tokenizer,
                args.seed,
                args.temperature,
                args.top_p,
                args.repeat_penalty,
                args.repeat_last_n,
                args.verbose_prompt,
                &device,
            );
            pipeline.run(&prompt, args.sample_len)?;
        }
        (None, Some(mmlu_dir)) => mmlu(model, tokenizer, &device, mmlu_dir)?,
    }
    Ok(())
}

fn mmlu<P: AsRef<std::path::Path>>(
    mut model: Model,
    tokenizer: Tokenizer,
    device: &Device,
    mmlu_dir: P,
) -> anyhow::Result<()> {
    for dir_entry in mmlu_dir.as_ref().read_dir()?.flatten() {
        let dir_entry = dir_entry.path();
        let theme = match dir_entry.file_stem().and_then(|v| v.to_str()) {
            None => "".to_string(),
            Some(v) => match v.strip_suffix("_test") {
                None => v.replace('_', " "),
                Some(v) => v.replace('_', " "),
            },
        };
        if dir_entry.extension().as_ref().and_then(|v| v.to_str()) != Some("csv") {
            continue;
        }
        println!("reading {dir_entry:?}");
        let dir_entry = std::fs::File::open(dir_entry)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(dir_entry);
        let token_a = tokenizer.token_to_id("A").unwrap();
        let token_b = tokenizer.token_to_id("B").unwrap();
        let token_c = tokenizer.token_to_id("C").unwrap();
        let token_d = tokenizer.token_to_id("D").unwrap();
        for row in reader.records() {
            let row = match row {
                Err(_) => continue,
                Ok(row) => row,
            };
            if row.len() < 5 {
                continue;
            }
            let question = row.get(0).unwrap();
            let answer_a = row.get(1).unwrap();
            let answer_b = row.get(2).unwrap();
            let answer_c = row.get(3).unwrap();
            let answer_d = row.get(4).unwrap();
            let answer = row.get(5).unwrap();
            let prompt = format!(
                    "{} {theme}.\n{question}\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}\nAnswer:\n",
                    "The following are multiple choice questions (with answers) about"
                );
            let tokens = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
            let tokens = tokens.get_ids().to_vec();
            let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
            let logits = match &mut model {
                Model::MixFormer(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
                Model::Phi(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
                Model::Phi3(m) => {
                    m.clear_kv_cache();
                    m.forward(&input, 0)?
                }
                Model::Quantized(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
            let pr_a = logits_v[token_a as usize];
            let pr_b = logits_v[token_b as usize];
            let pr_c = logits_v[token_c as usize];
            let pr_d = logits_v[token_d as usize];
            let model_answer = if pr_a > pr_b && pr_a > pr_c && pr_a > pr_d {
                "A"
            } else if pr_b > pr_c && pr_b > pr_d {
                "B"
            } else if pr_c > pr_d {
                "C"
            } else {
                "D"
            };

            println!("{prompt}\n -> {model_answer} vs {answer}");
        }
    }
    Ok(())
}
