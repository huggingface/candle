use std::path::PathBuf;

use candle::{bail, DType, Device, Result, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::lightonocr_2_1b::model;
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

mod preprocessor;
use preprocessor::preprocess;

const PATCH_SIZE: u32 = 14;
const DEFAULT_MODEL: &str = "lightonai/LightOnOCR-2-1B";
const DEFAULT_MAX_EDGE: u32 = 768;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;

pub struct TextGeneration {
    model: model::Model,
    device: Device,
    tokenizer: TokenOutputStream,
    max_new_tokens: usize,
    image_token_id: u32,
    eos_token_id: u32,
    spatial_merge_size: usize,
}

impl TextGeneration {
    fn new(
        model: model::Model,
        tokenizer: Tokenizer,
        device: &Device,
        max_new_tokens: usize,
    ) -> Self {
        let image_token_id = model.model_config.image_token_id as u32;
        let eos_token_id = model.model_config.eos_token_id as u32;
        let spatial_merge_size = model.model_config.spatial_merge_size;
        Self {
            model,
            device: device.clone(),
            tokenizer: TokenOutputStream::new(tokenizer),
            max_new_tokens,
            image_token_id,
            eos_token_id,
            spatial_merge_size,
        }
    }

    fn run(&mut self, image: String, max_edge: u32, dtype: DType) -> Result<()> {
        self.model.language_model.clear_kv_cache();
        let img = image::open(image).map_err(|e| candle::Error::Msg(e.to_string()))?;
        let preprocessed = preprocess(&img, max_edge, &self.device, dtype)
            .map_err(|e| candle::Error::Msg(e.to_string()))?;

        let merged_ph = preprocessed.ph / self.spatial_merge_size;
        let merged_pw = preprocessed.pw / self.spatial_merge_size;
        let num_image_tokens = merged_ph * merged_pw;

        self.generate(num_image_tokens, preprocessed)
    }

    fn generate(
        &mut self,
        num_image_tokens: usize,
        preprocessed: preprocessor::PreprocessedImage,
    ) -> Result<()> {
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(self
                .tokenizer
                .tokenizer()
                .encode(s, false)
                .map_err(candle::Error::wrap)?
                .get_ids()
                .to_vec())
        };

        // Build prompt: <|vision_start|>system<|vision_end|>\n<|vision_start|>user\n<image><|vision_end|>\n<|vision_start|>assistant\n
        let vision_start: u32 = 151644;
        let newline_tokens = encode("\n")?;
        let mut input_ids: Vec<u32> = Vec::new();

        input_ids.push(vision_start);
        input_ids.extend_from_slice(&encode("system")?);
        input_ids.push(self.eos_token_id);
        input_ids.extend_from_slice(&newline_tokens);
        input_ids.push(vision_start);
        input_ids.extend_from_slice(&encode("user\n")?);
        input_ids.extend(std::iter::repeat(self.image_token_id).take(num_image_tokens));
        input_ids.push(self.eos_token_id);
        input_ids.extend_from_slice(&newline_tokens);
        input_ids.push(vision_start);
        input_ids.extend_from_slice(&encode("assistant\n")?);

        let seq_len = input_ids.len();
        let input_tensor = Tensor::from_vec(input_ids, (1, seq_len), &self.device)?;

        let logits = self
            .model
            .forward(&input_tensor, &preprocessed.pixel_values, 0)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut offset = seq_len;

        generated.push(Self::greedy(logits)?);

        for _ in 1..self.max_new_tokens {
            let last = *generated.last().unwrap();
            if last == self.eos_token_id {
                break;
            }
            let input = Tensor::from_vec(vec![last], (1, 1), &self.device)?;
            let logits = self.model.language_model.forward(&input, offset)?;
            generated.push(Self::greedy(logits)?);
            offset += 1;
        }

        let decode_ids: Vec<u32> = generated
            .iter()
            .copied()
            .filter(|&t| t != self.eos_token_id)
            .collect();

        let output = self
            .tokenizer
            .tokenizer()
            .decode(&decode_ids, true)
            .map_err(|e| candle::Error::Msg(format!("Failed to decode: {e}")))?;

        println!("{output}");
        Ok(())
    }

    fn greedy(logits: Tensor) -> Result<u32> {
        let logits = logits.squeeze(0)?;
        let seq = logits.dim(0)?;
        let last = logits
            .narrow(0, seq - 1, 1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let logits_vec = last.to_vec1::<f32>()?;

        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);

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

    #[arg(
        long,
        help = "Path to model config.json (downloaded from HF if not provided)"
    )]
    config: Option<String>,

    #[arg(
        long,
        help = "Path to tokenizer.json (downloaded from HF if not provided)"
    )]
    tokenizer: Option<String>,

    #[arg(
        long,
        help = "Path to model weights .safetensors (downloaded from HF if not provided)"
    )]
    weights: Option<String>,

    #[arg(long)]
    image_location: String,

    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_new_tokens: usize,

    #[arg(
        long,
        help = "Maximum edge length for image resize (must be >= 14, dimensions padded to multiples of 28)"
    )]
    max_edge: Option<u32>,
}

fn download_hf_files(model_id: &str, revision: &str) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let api = Api::new().map_err(|e| candle::Error::Msg(e.to_string()))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let config_path = repo
        .get("config.json")
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(e.to_string()))?;

    let weights_paths = match repo.get("model.safetensors") {
        Ok(f) => vec![f],
        Err(_) => {
            let f = repo.get("pytorch_model.bin").map_err(|_| {
                candle::Error::Msg(
                    "Could not find model.safetensors or pytorch_model.bin. Use --weights to specify a local file.".to_string(),
                )
            })?;
            vec![f]
        }
    };

    Ok((config_path, tokenizer_path, weights_paths))
}

fn load_config(path: &PathBuf) -> Result<model::Config> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content)
        .map_err(|e| candle::Error::Msg(format!("Failed to parse config: {e}")))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(dtype) => bail!("unsupported dtype: {dtype}"),
        None if device.is_cuda() => DType::BF16,
        None => DType::F32,
    };

    let max_edge = args.max_edge.unwrap_or(DEFAULT_MAX_EDGE);
    if max_edge < PATCH_SIZE {
        bail!("--max-edge must be at least {PATCH_SIZE}, got {max_edge}");
    }

    let (config_path, tokenizer_path, weights_paths) =
        if args.config.is_some() || args.tokenizer.is_some() || args.weights.is_some() {
            let config = args
                .config
                .as_deref()
                .expect("--config is required when using local files");
            let tokenizer = args
                .tokenizer
                .as_deref()
                .expect("--tokenizer is required when using local files");
            let weights = args
                .weights
                .as_deref()
                .expect("--weights is required when using local files");
            (
                PathBuf::from(config),
                PathBuf::from(tokenizer),
                vec![PathBuf::from(weights)],
            )
        } else {
            println!("downloading model from {}/{}", args.model_id, args.revision);
            download_hf_files(&args.model_id, &args.revision)?
        };

    let cfg = load_config(&config_path)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, &device)? };

    let model = model::Model::new(cfg, vb)?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {e}")))?;

    let mut text_generation = TextGeneration::new(model, tokenizer, &device, args.max_new_tokens);
    text_generation.run(args.image_location, max_edge, dtype)
}
