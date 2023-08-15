// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;

mod model;
use model::{Config, Llama};

const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_PROMPT: &str = r"
EDWARD:
I wonder how our princely father 'scaped,
Or whether he be 'scaped away or no
From Clifford's and Northumberland's pursuit:
Had he been ta'en, we should have heard the news;
Had he been slain, we should have heard the news;
Or had he 'scaped, methinks we should have heard
The happy tidings of his good escape.
How fares my brother? why is he so sad?

RICHARD:
I cannot joy, until I be resolved
Where our right valiant father is become.
I saw him in the battle range about;
And watch'd him how he singled Clifford forth.
Methought he bore him in the thickest troop
As doth a lion in a herd of neat;
Or as a bear, encompass'd round with dogs,
Who having pinch'd a few and made them cry,
The rest stand all aloof, and bark at him.
So fared our father with his enemies;
So fled his enemies my warlike father:
Methinks, 'tis prize enough to be his son.
See how the morning opes her golden gates,
And takes her farewell of the glorious sun!
How well resembles it the prime of youth,
Trimm'd like a younker prancing to his love!

EDWARD:
Dazzle mine eyes, or do I see three suns?

RICHARD:
Three glorious suns, each one a perfect sun;
Not separated with the racking clouds,
But sever'd in a pale clear-shining sky.
See, see! they join, embrace, and seem to kiss,
As if they vow'd some league inviolable:
Now are they but one lamp, one light, one sun.
In this the heaven figures some event.

EDWARD:
'Tis wondrous strange, the like yet never heard of.
I think it cites us, brother, to the field,
That we, the sons of brave Plantagenet,
Each one already blazing by our meeds,
Should notwithstanding join our lights together
And over-shine the earth as this the world.
Whate'er it bodes, henceforward will I bear
Upon my target three fair-shining suns.
";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use npy instead of safetensors
    #[arg(long)]
    npy: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use f32 computations rather than f16.
    #[arg(long)]
    use_f32: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    v1: bool,

    #[arg(long)]
    use_flash_attn: bool,

    /// The folder name that contains safetensor weights and json files 
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
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

    let device = candle_examples::device(args.cpu)?;
    let config = if args.v1 {
        Config::config_7b_v1(args.use_flash_attn)
    } else {
        Config::config_7b_v2(args.use_flash_attn)
    };
    let dtype = if args.use_f32 { DType::F32 } else { DType::F16 };
    let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;
    let (llama, tokenizer_filename) = match args.npy {
        Some(filename) => {
            let vb = VarBuilder::from_npz(filename, dtype, &device)?;
            let tokenizer = std::path::PathBuf::from("llama-tokenizer.json");
            (Llama::load(vb, &cache, &config)?, tokenizer)
        }
        None => {
            let api = Api::new()?;
            let model_id = args.model_id.unwrap_or_else(|| {
                if args.v1 {
                    "Narsil/amall-7b".to_string()
                } else {
                    "meta-llama/Llama-2-7b-hf".to_string()
                }
            });
            println!("loading the model weights from {model_id}");
            let api = api.model(model_id);

            let tokenizer_filename = match &args.local_weights {
                Some(path) => {
                    (path.to_owned() + "tokenizer.json").into()
                }
                _=> {
                    api.get("tokenizer.json")?
                }
            };

            let mut filenames = vec![];
            for rfilename in [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ] {
                match &args.local_weights {
                    Some(path) => {
                        filenames.push((path.to_owned() + rfilename).into());
                    }
                    _=> {
                        let filename = api.get(rfilename)?;
                        filenames.push(filename);
                    }
                };
            }

            println!("building the model");
            let handles = filenames
                .iter()
                .map(|f| Ok(unsafe { candle::safetensors::MmapedFile::new(f.as_path())? }))
                .collect::<Result<Vec<_>>>()?;
            let tensors: Vec<_> = handles
                .iter()
                .map(|h| Ok(h.deserialize()?))
                .collect::<Result<Vec<_>>>()?;

            let vb = VarBuilder::from_safetensors(tensors, dtype, &device);
            (Llama::load(vb, &cache, &config)?, tokenizer_filename)
        }
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature);
    let mut new_tokens = vec![];
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);
        new_tokens.push(next_token);

        let tk = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        if [",", ".", ":", "?", "'", "\""].contains(&tk.as_str()) || index == args.sample_len-1 || next_token==2 { //2 for end token
            print!(
                "{} ",
                tokenizer.decode(&new_tokens, true).map_err(E::msg)?
            );
            new_tokens.clear();
        }

        if next_token == 2 {
            break;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
