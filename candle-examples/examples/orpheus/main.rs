#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use candle_transformers::models::snac::{Config as SnacConfig, Model as SnacModel};
use tokenizers::Tokenizer;

// https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/realtime_streaming_example/main.py#L43
const STOP_TOKEN_ID: u32 = 128258;

#[derive(Parser)]
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

    #[arg(long, default_value = "Hey, how are you doing today?")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    /// The output wav file.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    #[arg(long, default_value = "3b-0.1-ft")]
    which: Which,

    #[arg(long, default_value = "tara")]
    voice: Voice,

    #[arg(long)]
    use_flash_attn: bool,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Voice {
    #[value(name = "tara")]
    Tara,
    #[value(name = "leah")]
    Leah,
    #[value(name = "jess")]
    Jess,
    #[value(name = "leo")]
    Leo,
    #[value(name = "dan")]
    Dan,
    #[value(name = "mia")]
    Mia,
    #[value(name = "zac")]
    Zac,
    #[value(name = "zoe")]
    Zoe,
}

impl Voice {
    fn as_str(&self) -> &'static str {
        match self {
            Voice::Tara => "tara",
            Voice::Leah => "leah",
            Voice::Jess => "jess",
            Voice::Leo => "leo",
            Voice::Dan => "dan",
            Voice::Mia => "mia",
            Voice::Zac => "zac",
            Voice::Zoe => "zoe",
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "3b-0.1-ft")]
    ThreeB0_1Ft,
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
    let prompt = args.prompt.clone();
    let mut model = Model::load(args)?;
    model.run(&prompt)?;
    Ok(())
}

struct Model {
    model: Llama,
    tokenizer: Tokenizer,
    logits_processor: candle_transformers::generation::LogitsProcessor,
    cache: Cache,
    device: Device,
    verbose_prompt: bool,
    snac: SnacModel,
    out_file: String,
    voice: Voice,
}

fn load_snac(device: &Device) -> Result<SnacModel> {
    let api = hf_hub::api::sync::Api::new()?;
    let m = api.model("hubertsiuzdak/snac_24khz".to_string());
    let config = m.get("config.json")?;
    let config: SnacConfig = serde_json::from_reader(std::fs::File::open(config)?)?;
    let m = api.model("lmz/candle-snac".to_string());
    let model = m.get("snac_24khz.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, device)? };
    let model = SnacModel::new(&config, vb)?;
    Ok(model)
}

impl Model {
    fn load(args: Args) -> Result<Self> {
        let start = std::time::Instant::now();
        let api = hf_hub::api::sync::Api::new()?;
        let model_id = match args.model_id {
            Some(model_id) => model_id.to_string(),
            None => match args.which {
                Which::ThreeB0_1Ft => "canopylabs/orpheus-3b-0.1-ft".to_string(),
            },
        };
        let revision = match args.revision {
            Some(r) => r,
            None => "main".to_string(),
        };
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id,
            hf_hub::RepoType::Model,
            revision,
        ));
        let model_files = match args.model_file {
            Some(m) => vec![m.into()],
            None => match args.which {
                Which::ThreeB0_1Ft => {
                    candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
                }
            },
        };
        let config = match args.config_file {
            Some(m) => m.into(),
            None => repo.get("config.json")?,
        };
        let tokenizer = match args.tokenizer_file {
            Some(m) => m.into(),
            None => repo.get("tokenizer.json")?,
        };
        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

        let start = std::time::Instant::now();
        let device = candle_examples::device(args.cpu)?;
        let dtype = device.bf16_default_to_f32();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
        let config: LlamaConfig = serde_json::from_reader(std::fs::File::open(config)?)?;
        let config = config.into_config(args.use_flash_attn);
        let model = Llama::load(vb, &config)?;
        let logits_processor = {
            use candle_transformers::generation::{LogitsProcessor, Sampling};
            let temperature = args.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (args.top_k.as_ref(), args.top_p.as_ref()) {
                    (None, None) => Sampling::All { temperature },
                    (Some(&k), None) => Sampling::TopK { k, temperature },
                    (None, Some(&p)) => Sampling::TopP { p, temperature },
                    (Some(&k), Some(&p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(args.seed, sampling)
        };

        println!("loaded the model in {:?}", start.elapsed());
        let cache = Cache::new(true, dtype, &config, &device)?;
        let snac = load_snac(&device)?;
        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            cache,
            device,
            verbose_prompt: args.verbose_prompt,
            snac,
            voice: args.voice,
            out_file: args.out_file,
        })
    }

    fn run(&mut self, prompt: &str) -> Result<()> {
        println!("running the model on '{}'", prompt);
        let device = &self.device;
        let prompt = format!("{voice}: {prompt}", voice = self.voice.as_str());
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        // https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/orpheus_tts_pypi/orpheus_tts/engine_class.py#L82
        let mut tokens = [
            &[128259],
            tokens.get_ids(),
            &[128009, 128260, 128261, 128257],
        ]
        .concat();
        if self.verbose_prompt {
            println!("{:?}", tokens);
        }
        let mut cache = self.cache.clone();

        println!("starting the inference loop");
        let mut index_pos = 0;
        let mut audio_tokens = vec![];
        for index in 0..2000 {
            let (context_size, context_index) = if index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            index_pos += ctxt.len();

            let next_token = self.logits_processor.sample(&logits)?;
            if let Some(tok) = self.tokenizer.id_to_token(next_token) {
                match tok.strip_prefix("<custom_token_") {
                    Some(tok) => match tok.strip_suffix('>') {
                        Some(tok) => {
                            let tok = tok.parse::<u32>()?;
                            // https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/orpheus_tts_pypi/orpheus_tts/decoder.py#L86C35-L86C63
                            let tok = tok - 10 - ((audio_tokens.len() as u32 % 7) * 4096);
                            audio_tokens.push(tok);
                        }
                        None => {
                            println!("{index}: unexpected custom token {next_token} {tok}");
                        }
                    },
                    None => {
                        println!("{index}: unexpected token {next_token} {tok}");
                    }
                }
            }
            if next_token == STOP_TOKEN_ID {
                println!("reached stop token");
                break;
            }
            tokens.push(next_token);
        }
        println!("generated {} audio tokens", audio_tokens.len());
        let mut codes0 = vec![];
        let mut codes1 = vec![];
        let mut codes2 = vec![];
        for audio_tokens in audio_tokens.chunks_exact(7) {
            codes0.push(audio_tokens[0]);
            for i in [1, 4] {
                codes1.push(audio_tokens[i]);
            }
            for i in [2, 3, 5, 6] {
                codes2.push(audio_tokens[i]);
            }
        }
        let codes0 = Tensor::new(codes0, device)?.unsqueeze(0)?;
        let codes1 = Tensor::new(codes1, device)?.unsqueeze(0)?;
        let codes2 = Tensor::new(codes2, device)?.unsqueeze(0)?;
        let pcm = self.snac.decode(&[&codes0, &codes1, &codes2])?;
        println!("decoded to pcm {pcm:?}");
        let mut output = std::fs::File::create(&self.out_file)?;
        let pcm = pcm.i(0)?.i(0)?.to_vec1::<f32>()?;
        candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24000)?;
        Ok(())
    }
}
