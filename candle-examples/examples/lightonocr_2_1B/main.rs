use std::path::PathBuf;

use candle::{DType, Device, Tensor, bail};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::lightonocr_2_1b::model;
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use candle::Result;

mod preprocessor;
use preprocessor::{PreprocessedImage, preprocess};

const PATCH_SIZE: u32 = 14;
const MERGE_SIZE: u32 = 2;
const DEFAULT_MODEL: &str = "lightonai/LightOnOCR-2-1B";
const DEFAULT_MAX_EDGE: u32 = 768;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;

pub struct TextGeneration {
    model: model::Model,
    device: Device,
    tokenizer: TokenOutputStream,
    max_new_tokens: usize,
    image_end: u32,
}

impl TextGeneration {
    fn new(model: model::Model, tokenizer: Tokenizer, device: &Device, max_new_tokens: usize) -> Self {
        Self {
            model,
            device: device.clone(),
            tokenizer: TokenOutputStream::new(tokenizer),
            max_new_tokens,
            image_end: 151645,
        }
    }

    fn run(&mut self, image: String, max_edge: u32, dtype: DType) -> Result<()> {
        println!("Running..");
        self.model.language_model.clear_kv_cache();
        let img = image::open(image).map_err(|e| candle::Error::wrap(e.to_string()))?;
        let preprocessed = preprocess(&img, max_edge, &self.device, dtype)
            .map_err(|e| candle::Error::wrap(e.to_string()))?;

        let merged_ph = preprocessed.ph / MERGE_SIZE as usize;
        let merged_pw = preprocessed.pw / MERGE_SIZE as usize;
        let num_image_tokens = merged_ph * merged_pw;

        self.encode_image_tokens(num_image_tokens, preprocessed)
    }

    fn encode_image_tokens(&mut self, num_image_tokens: usize, preprocessed: PreprocessedImage) -> Result<()> {
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(self.tokenizer.tokenizer()
                .encode(s, false)
                .map_err(candle::Error::wrap)?
                .get_ids()
                .to_vec())
        };

        let system_tokens    = encode("system")?;
        let user_tokens      = encode("user\n")?;
        let newline_tokens   = encode("\n")?;
        let assistant_tokens = encode("assistant\n")?;

        let image_pad = 151655;
        let image_start = 151644u32;
        let image_tokens: Vec<u32> = vec![image_pad as u32; num_image_tokens];

        let mut input_ids: Vec<u32> = Vec::new();

        input_ids.push(image_start);
        input_ids.extend_from_slice(&system_tokens);
        input_ids.push(self.image_end);
        input_ids.extend_from_slice(&newline_tokens);
        input_ids.push(image_start);
        input_ids.extend_from_slice(&user_tokens);
        input_ids.extend_from_slice(&image_tokens);
        input_ids.push(self.image_end);
        input_ids.extend_from_slice(&newline_tokens);
        input_ids.push(image_start);
        input_ids.extend_from_slice(&assistant_tokens);

        let seq_len = input_ids.len();

        let device = &self.device;
        let input_tensor = Tensor::from_vec(input_ids, (1, seq_len), device)?;

        let logits = self.model.forward(&input_tensor, &preprocessed.pixel_values, 0)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut offset = seq_len;

        let first = TextGeneration::greedy(logits)?;
        generated.push(first);

        for _ in 1..self.max_new_tokens {
            let last = *generated.last().unwrap();

            if last == self.image_end {
                break;
            }

            let input = Tensor::from_vec(vec![last], (1, 1), device)?;
            let logits = self.model.language_model.forward(&input, offset)?;
            let token = TextGeneration::greedy(logits)?;
            generated.push(token);
            offset += 1;
        }

        let decode_ids: Vec<u32> = generated.iter()
            .copied()
            .filter(|&t| t != self.image_end)
            .collect();

        let output = self.tokenizer.tokenizer()
            .decode(&decode_ids, true)
            .map_err(|e| candle::Error::wrap(e.to_string()))?;

        println!("{output}");

        Ok(())
    }

    fn greedy(logits: Tensor) -> Result<u32> {
        let logits = logits.squeeze(0)?;
        let seq = logits.dim(0)?;
        let last = logits.narrow(0, seq - 1, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits_vec = last.to_vec1::<f32>()?;

        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (idx, value) in logits_vec.iter().enumerate() {
            if *value > max_val {
                max_val = *value;
                max_idx = idx;
            }
        }

        Ok(max_idx as u32)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value = DEFAULT_MODEL)]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, help = "Path to model config.json (downloaded from HF if not provided)")]
    config: Option<String>,

    #[arg(long, help = "Path to tokenizer.json (downloaded from HF if not provided)")]
    tokenizer: Option<String>,

    #[arg(long, help = "Path to model weights .safetensors (downloaded from HF if not provided)")]
    weights: Option<String>,

    #[arg(long, default_value = "candle-examples/examples/lightonocr_2_1B/assets/730501a_sundbyberg_stockholm.pdf-01.png")]
    image_location: String,

    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_new_tokens: usize,

    #[arg(long, help = "Maximum edge length for image resize (must be >= 14, dimensions padded to multiples of 28)")]
    max_edge: Option<u32>,
}

fn download_hf_files(model_id: &str, revision: &str) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let api = Api::new().map_err(|e| candle::Error::wrap(e.to_string()))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let config_path = repo.get("config.json")
        .map_err(|e| candle::Error::wrap(e.to_string()))?;
    let tokenizer_path = repo.get("tokenizer.json")
        .map_err(|e| candle::Error::wrap(e.to_string()))?;

    let weights_paths = match repo.get("model.safetensors") {
        Ok(f) => vec![f],
        Err(_) => {
            let f = repo.get("pytorch_model.bin")
                .map_err(|_| candle::Error::Msg(
                    "Could not find model.safetensors or pytorch_model.bin on HuggingFace. Use --weights to specify a local file.".to_string()
                ))?;
            vec![f]
        }
    };

    Ok((config_path, tokenizer_path, weights_paths))
}

fn read_config(path: &PathBuf) -> Result<model::Config> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content)
        .map_err(|e| candle::Error::wrap(format!("Failed to parse config: {e}")))
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None if device.is_cuda() => DType::BF16,
        None => DType::F32,
    };

    let max_edge = args.max_edge.unwrap_or(DEFAULT_MAX_EDGE);
    if max_edge < PATCH_SIZE {
        bail!("--max-edge must be at least {PATCH_SIZE}, got {max_edge}");
    }

    let (config_path, tokenizer_path, weights_paths) = if args.config.is_some()
        || args.tokenizer.is_some()
        || args.weights.is_some()
    {
        let config = args.config.as_deref().expect("--config required when using local files");
        let tokenizer = args.tokenizer.as_deref().expect("--tokenizer required when using local files");
        let weights = args.weights.as_deref().expect("--weights required when using local files");
        (PathBuf::from(config), PathBuf::from(tokenizer), vec![PathBuf::from(weights)])
    } else {
        println!("Downloading model files from {}/{}...", args.model_id, args.revision);
        download_hf_files(&args.model_id, &args.revision)?
    };

    let cfg = read_config(&config_path)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, &device)?
    };

    let model = model::Model::new(cfg, vb)?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| candle::Error::wrap(format!("Tokenizer error: {e}")))?;

    let mut text_generation = TextGeneration::new(model, tokenizer, &device, args.max_new_tokens);
    text_generation.run(args.image_location, max_edge, dtype)
}
