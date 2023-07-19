// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json
//
// In order to convert the llama weights to a .npz file, run:
// python examples/llama/convert_checkpoint.py ..../LLaMA/7B/consolidated.00.pth

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};

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

fn precompute_freqs_cis(config: &Config, device: &Device) -> Result<Tensor> {
    let n_elem = config.n_embd / config.n_head;
    let theta: Vec<_> = (0..n_elem)
        .step_by(2)
        .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let shape = [1, MAX_SEQ_LEN, n_elem / 2, 1];
    let idx_theta_cos = idx_theta.cos()?.reshape(&shape)?;
    let idx_theta_sin = idx_theta.sin()?.reshape(&shape)?;
    Ok(Tensor::cat(&[&idx_theta_cos, &idx_theta_sin], D::Minus1)?)
}

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
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    let config = Config::config_7b();
    let cache = model::Cache::new(!args.no_kv_cache, &config, &device);
    let dtype = if args.use_f32 { DType::F32 } else { DType::F16 };
    let (llama, tokenizer_filename) = match args.npy {
        Some(filename) => {
            let vb = VarBuilder::from_npz(filename, dtype, &device)?;
            let tokenizer = std::path::PathBuf::from("llama-tokenizer.json");
            (Llama::load(vb, &cache, &config)?, tokenizer)
        }
        None => {
            let api = Api::new()?;
            let repo = Repo::new("Narsil/amall-7b".to_string(), RepoType::Model);
            println!("loading the model weights");
            let tokenizer_filename = api.get(&repo, "tokenizer.json")?;
            let mut filenames = vec![];
            for rfilename in [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ] {
                let filename = api.get(&repo, rfilename)?;
                filenames.push(filename);
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

    println!("pre-computing the positional embeddings");
    let freqs_cis = precompute_freqs_cis(&config, &device)?;
    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature);
    let mut new_tokens = vec![];
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    for index in 0..args.sample_len {
        let start_gen = std::time::Instant::now();
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let freqs_cis = if cache.use_kv_cache {
            freqs_cis.narrow(1, index_pos, ctxt.len())?
        } else {
            freqs_cis.clone()
        };
        let logits = llama.forward(&input, &freqs_cis)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        new_tokens.push(next_token);
        println!("> {:?}", start_gen.elapsed());
        println!(
            "{} token: {} '{}'",
            index + 1,
            next_token,
            tokenizer.decode(vec![next_token], true).map_err(E::msg)?
        );
    }
    let dt = start_gen.elapsed();
    println!(
        "{} tokens generated ({} token/s)\n----\n{}\n----",
        args.sample_len,
        args.sample_len as f64 / dt.as_secs_f64(),
        tokenizer.decode(new_tokens, true).map_err(E::msg)?
    );
    Ok(())
}
