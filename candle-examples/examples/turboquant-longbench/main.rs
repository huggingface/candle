use anyhow::{bail, Error as E, Result};
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;

mod llama;
mod mistral;

use candle_nn::quant_kv::{turbo_quant::TurboQuantConfig, QuantAlgorithm};

const EOS_TOKEN: &str = "</s>";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    Llama31_8bInstruct,
    Ministral7bInstruct,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value = "llama31-8b-instruct")]
    which: Which,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    let is_cuda = device.is_cuda();
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => {
            if is_cuda {
                DType::BF16
            } else {
                DType::F32
            }
        }
    };

    println!("Initializing API and fetching the model...");
    let api = Api::new()?;
    let model_id = match args.which {
        Which::Llama31_8bInstruct => "meta-llama/Llama-3.1-8B-Instruct",
        Which::Ministral7bInstruct => "mistralai/Mistral-7B-Instruct-v0.3",
    };

    let revision = "main".to_string();
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision,
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")
        .unwrap_or_else(|_| vec![repo.get("model.safetensors").unwrap()]);

    let tokenizer = Tokenizer::from_file(tokenizer_filename.clone()).map_err(E::msg)?;
    let mut tokenizer_stream =
        candle_examples::token_output_stream::TokenOutputStream::new(tokenizer.clone());

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    // Fake Longbench QA Scenario
    let document = "Extensive context spanning general knowledge regarding Paris: Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region, also known as the région parisienne, Paris Region. The City of Paris is the centre and seat of government of the region and province of Île-de-France. Paris is especially known for its museums and architectural landmarks: the Louvre was the most visited art museum in the world in 2020. Important landmarks include the Eiffel Tower and the Arc de Triomphe.";
    let question = "Based on the text above, what river is Paris situated on?";

    let prompt = format!("Document Context:\n{document}\n{document}\n{document}\n{document}\nQuestion: {question}\nAnswer:");
    let mut tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    println!(
        "Starting inference map for ({}) with context length: {} tokens...",
        model_id,
        tokens.len()
    );
    let mut token_generated = 0;
    let mut generated_text = String::new();
    let start_gen = std::time::Instant::now();

    match args.which {
        Which::Llama31_8bInstruct => {
            let config: llama::LlamaConfig =
                serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let config = config.into_config(false);

            let algo = QuantAlgorithm::TurboQuant(TurboQuantConfig::new_3p5bit(
                config.hidden_size / config.num_attention_heads,
                args.seed,
            ));
            let mut cache = llama::Cache::new(true, dtype, &config, &device, algo)?;
            let llama = llama::Llama::load(vb, &config)?;

            let eos_token_id = config.eos_token_id.or_else(|| {
                tokenizer
                    .token_to_id(EOS_TOKEN)
                    .map(llama::LlamaEosToks::Single)
            });
            let mut index_pos = 0;

            for index in 0..args.sample_len {
                let (context_size, context_index) = if index > 0 {
                    (1, index_pos)
                } else {
                    (tokens.len(), 0)
                };
                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;

                let logits = llama
                    .forward(&input, context_index, &mut cache)?
                    .squeeze(0)?;
                let next_token = logits_processor.sample(&logits)?;

                token_generated += 1;
                tokens.push(next_token);
                index_pos += context_size;

                match eos_token_id {
                    Some(llama::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                        break
                    }
                    Some(llama::LlamaEosToks::Multiple(ref eos_ids))
                        if eos_ids.contains(&next_token) =>
                    {
                        break
                    }
                    _ => (),
                }

                if let Some(t) = tokenizer_stream.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                    generated_text.push_str(&t);
                }
            }
        }
        Which::Ministral7bInstruct => {
            let config: mistral::Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let head_dim = config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads);

            let algo =
                QuantAlgorithm::TurboQuant(TurboQuantConfig::new_3p5bit(head_dim, args.seed));
            let mut mistral = mistral::Model::new(&config, vb, algo)?;
            let eos_token = tokenizer.token_to_id(EOS_TOKEN).unwrap_or(2);

            for index in 0..args.sample_len {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let start_pos = tokens.len().saturating_sub(context_size);
                let ctxt = &tokens[start_pos..];

                let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
                let logits = mistral
                    .forward(&input, start_pos)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?;

                let next_token = logits_processor.sample(&logits)?;
                tokens.push(next_token);
                token_generated += 1;

                if next_token == eos_token {
                    break;
                }

                if let Some(t) = tokenizer_stream.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                    generated_text.push_str(&t);
                }
            }
        }
    }

    if let Some(rest) = tokenizer_stream.decode_rest().map_err(E::msg)? {
        print!("{rest}");
        generated_text.push_str(&rest);
    }

    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({:.2} token/s)",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64()
    );

    let success = generated_text.to_lowercase().contains("seine");
    println!(
        "LongBench Evaluation Result: {}",
        if success {
            "PASS - Found Correct Answer"
        } else {
            "FAIL - Extraction Missing"
        }
    );

    Ok(())
}
