#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use candle::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::mistral3::{Mistral3Config, Mistral3ForConditionalGeneration};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::path::PathBuf;

// CLIP normalization constants (used by Pixtral/Mistral3)
const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    /// Mistral-Small-3.1-24B-Instruct-2503
    #[value(name = "small-3.1-24b")]
    MistralSmall31_24B,
    /// Mistral-Small-3.2-24B-Instruct-2506
    #[value(name = "small-3.2-24b")]
    MistralSmall32_24B,
}

impl WhichModel {
    fn model_id(&self) -> &'static str {
        match self {
            Self::MistralSmall31_24B => "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            Self::MistralSmall32_24B => "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum DTypeArg {
    #[value(name = "f32")]
    F32,
    #[value(name = "f16")]
    F16,
    #[value(name = "bf16")]
    BF16,
    /// Auto-detect based on device capabilities
    #[value(name = "auto")]
    Auto,
}

impl DTypeArg {
    fn as_dtype(self, device: &Device) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::Auto => {
                if device.is_cpu() {
                    DType::F32
                } else if device.supports_bf16() {
                    DType::BF16
                } else {
                    DType::F16
                }
            }
        }
    }
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

    /// The prompt to use for generation.
    #[arg(long, default_value = "Describe this image in detail.")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 1000)]
    sample_len: usize,

    /// The model variant to use.
    #[arg(long, default_value = "small-3.2-24b")]
    which: WhichModel,

    /// HuggingFace model ID (overrides --which).
    #[arg(long)]
    model_id: Option<String>,

    /// Local model directory path (overrides --model-id and --which).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// HuggingFace revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Path to tokenizer file (tekken.json or tokenizer.json).
    #[arg(long)]
    tokenizer_file: Option<PathBuf>,

    /// Path to config.json file.
    #[arg(long)]
    config_file: Option<PathBuf>,

    /// Comma-separated list of weight files.
    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Path to input image file.
    #[arg(long)]
    image: Option<String>,

    /// Data type for model weights.
    #[arg(long, default_value = "auto")]
    dtype: DTypeArg,

    /// Only run vision encoder (for testing).
    #[arg(long)]
    vision_only: bool,
}

/// Tokenizer wrapper trait for unified interface
trait TokenizerWrapper: Send {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn get_token(&self, token: &str) -> Option<u32>;
}

/// Wrapper for tekken tokenizer
#[cfg(feature = "tekken")]
struct TekkenTokenizer(tekken::Tekkenizer);

#[cfg(feature = "tekken")]
impl TokenizerWrapper for TekkenTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.0
            .encode(text, add_special_tokens, false)
            .map_err(|e| anyhow::anyhow!("Tekken encode error: {}", e))
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        self.0
            .decode(ids, tekken::SpecialTokenPolicy::Ignore)
            .map_err(|e| anyhow::anyhow!("Tekken decode error: {}", e))
    }

    fn get_token(&self, token: &str) -> Option<u32> {
        self.0.get_control_token(token).ok()
    }
}

/// Wrapper for HuggingFace tokenizers
struct HfTokenizer(tokenizers::Tokenizer);

impl TokenizerWrapper for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.0
            .encode(text, add_special_tokens)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("HF tokenizer encode error: {}", e))
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("HF tokenizer decode error: {}", e))
    }

    fn get_token(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }
}

/// Load tokenizer from file (supports both tekken and HF formats)
fn load_tokenizer(path: &std::path::Path) -> Result<Box<dyn TokenizerWrapper>> {
    #[allow(unused_variables)]
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

    #[cfg(feature = "tekken")]
    if filename == "tekken.json" || path.to_string_lossy().contains("tekken") {
        match tekken::Tekkenizer::from_file(path) {
            Ok(t) => {
                println!("Loaded Tekken tokenizer from {:?}", path);
                return Ok(Box::new(TekkenTokenizer(t)));
            }
            Err(e) => {
                println!(
                    "Failed to load as Tekken tokenizer: {}, trying HF format",
                    e
                );
            }
        }
    }

    // Try HuggingFace tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", path, e))?;
    println!("Loaded HuggingFace tokenizer from {:?}", path);
    Ok(Box::new(HfTokenizer(tokenizer)))
}

/// Normalize image tensor with CLIP mean and std
fn normalize_image(image: &Tensor) -> Result<Tensor> {
    let device = image.device();
    let mean = Tensor::from_vec(IMAGE_MEAN.to_vec(), (3, 1, 1), device)?;
    let std = Tensor::from_vec(IMAGE_STD.to_vec(), (3, 1, 1), device)?;

    let (_, c, h, w) = image.dims4()?;
    let mean = mean.broadcast_as((1, c, h, w))?;
    let std = std.broadcast_as((1, c, h, w))?;

    Ok(image.sub(&mean)?.div(&std)?)
}

/// Load and preprocess image
fn load_image(
    path: &str,
    patch_size: usize,
    spatial_merge_size: usize,
    device: &Device,
) -> Result<(Tensor, Vec<(usize, usize)>)> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;

    let (width, height) = (img.width() as usize, img.height() as usize);

    // Ensure dimensions are multiples of (patch_size * spatial_merge_size)
    // This is required by the multi-modal projector's reshape operation
    let align_size = patch_size * spatial_merge_size;
    let new_height = (height / align_size) * align_size;
    let new_width = (width / align_size) * align_size;

    let img = if new_height != height || new_width != width {
        println!(
            "Resizing image from {}x{} to {}x{} (align_size={})",
            width, height, new_width, new_height, align_size
        );
        img.resize_exact(
            new_width as u32,
            new_height as u32,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        img
    };

    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);

    // Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    let data: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
        .collect();

    let tensor = Tensor::from_vec(data, (height, width, 3), device)?
        .permute((2, 0, 1))? // (H, W, C) -> (C, H, W)
        .unsqueeze(0)?; // (C, H, W) -> (1, C, H, W)

    // Normalize with CLIP constants
    let tensor = normalize_image(&tensor)?;

    let image_sizes = vec![(height, width)];

    Ok((tensor, image_sizes))
}

struct TextGeneration {
    model: Mistral3ForConditionalGeneration,
    tokenizer: Box<dyn TokenizerWrapper>,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    device: Device,
    image_token_index: usize,
    eos_token_id: u32,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Mistral3ForConditionalGeneration,
        tokenizer: Box<dyn TokenizerWrapper>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        image_token_index: usize,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        // Get EOS token ID (default to 2 for Mistral)
        let eos_token_id = tokenizer.get_token("</s>").unwrap_or(2);

        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            image_token_index,
            eos_token_id,
        }
    }

    fn run(
        &mut self,
        prompt: &str,
        pixel_values: Option<&Tensor>,
        image_sizes: Option<&[(usize, usize)]>,
        sample_len: usize,
    ) -> Result<()> {
        self.model.clear_kv_cache();

        // Build input with image tokens if image is provided
        let (input_ids, num_image_tokens) = if let Some(sizes) = image_sizes {
            // Calculate number of image tokens
            let patch_size = 14; // Pixtral default
            let spatial_merge = 2; // Mistral3 default
            let (h, w) = sizes[0];
            let num_image_tokens =
                (h / patch_size / spatial_merge) * (w / patch_size / spatial_merge);

            // Build prompt with image tokens
            // Format: [INST] <image tokens> prompt [/INST]
            let inst_start = self.tokenizer.get_token("[INST]").unwrap_or(3);
            let inst_end = self.tokenizer.get_token("[/INST]").unwrap_or(4);

            let prompt_tokens = self.tokenizer.encode(prompt, false)?;

            let mut input_ids = vec![inst_start];
            input_ids.extend(std::iter::repeat_n(
                self.image_token_index as u32,
                num_image_tokens,
            ));
            input_ids.extend(prompt_tokens);
            input_ids.push(inst_end);

            (input_ids, num_image_tokens)
        } else {
            // Text-only mode
            let tokens = self.tokenizer.encode(prompt, true)?;
            (tokens, 0)
        };

        println!(
            "Input tokens: {} (including {} image tokens)",
            input_ids.len(),
            num_image_tokens
        );

        let mut tokens = input_ids.clone();
        let input_tensor = Tensor::new(
            input_ids
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.device,
        )?
        .unsqueeze(0)?;

        // First forward pass with image
        let start_gen = std::time::Instant::now();
        let logits = self
            .model
            .forward(&input_tensor, pixel_values, image_sizes, 0)?;

        let logits = logits.squeeze(0)?.i(logits.dim(1)? - 1..)?.squeeze(0)?;
        let logits = logits.to_dtype(DType::F32)?;

        let next_token = self.logits_processor.sample(&logits)?;
        tokens.push(next_token);

        // Print first token
        if let Ok(text) = self.tokenizer.decode(&[next_token], true) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }

        let mut generated_tokens = 1usize;

        // Continue generation
        for _ in 1..sample_len {
            let input =
                Tensor::new(&[tokens[tokens.len() - 1] as i64], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, None, None, tokens.len() - 1)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

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

            if next_token == self.eos_token_id {
                break;
            }

            if let Ok(text) = self.tokenizer.decode(&[next_token], true) {
                print!("{}", text);
                std::io::stdout().flush()?;
            }
        }

        let dt = start_gen.elapsed();
        println!();
        println!(
            "\n{} tokens generated ({:.2} token/s)",
            generated_tokens,
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(())
    }
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

    let device = candle_examples::device(args.cpu)?;
    let dtype = args.dtype.as_dtype(&device);
    println!("Device: {:?}, dtype: {:?}", device, dtype);

    let start = std::time::Instant::now();

    // Determine file paths
    let (config_path, tokenizer_path, weight_files) = if let Some(ref model_dir) = args.model_dir {
        // Local model directory
        println!("Loading from local directory: {:?}", model_dir);

        let config_path = args
            .config_file
            .unwrap_or_else(|| model_dir.join("config.json"));

        let tokenizer_path = args.tokenizer_file.unwrap_or_else(|| {
            let tekken = model_dir.join("tekken.json");
            if tekken.exists() {
                tekken
            } else {
                model_dir.join("tokenizer.json")
            }
        });

        let weight_files = if let Some(ref files) = args.weight_files {
            files.split(',').map(PathBuf::from).collect()
        } else {
            // Find all safetensors files
            std::fs::read_dir(model_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .map(|e| e.path())
                .collect()
        };

        (config_path, tokenizer_path, weight_files)
    } else {
        // HuggingFace Hub
        let api = Api::new()?;
        let model_id = args
            .model_id
            .clone()
            .unwrap_or_else(|| args.which.model_id().to_string());
        println!("Loading from HuggingFace: {}", model_id);

        let repo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            args.revision.clone(),
        ));

        let config_path = args
            .config_file
            .unwrap_or_else(|| repo.get("config.json").expect("config.json not found"));

        let tokenizer_path = args.tokenizer_file.unwrap_or_else(|| {
            repo.get("tekken.json")
                .or_else(|_| repo.get("tokenizer.json"))
                .expect("tokenizer not found")
        });

        let weight_files = if let Some(ref files) = args.weight_files {
            files.split(',').map(PathBuf::from).collect()
        } else {
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        (config_path, tokenizer_path, weight_files)
    };

    println!("Retrieved files in {:?}", start.elapsed());

    // Load config
    let config: Mistral3Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    println!("Model config:");
    println!(
        "  Vision: hidden_size={}, layers={}",
        config.vision_config.hidden_size, config.vision_config.num_hidden_layers
    );
    println!(
        "  Text: hidden_size={}, vocab_size={}, layers={}",
        config.text_config.hidden_size,
        config.text_config.vocab_size,
        config.text_config.num_hidden_layers
    );
    println!("  Image token index: {}", config.image_token_index);
    println!("  Spatial merge size: {}", config.spatial_merge_size);

    // Load image if provided
    // Image will be converted to model dtype (e.g., BF16) to match vision_tower weights
    let (pixel_values, image_sizes) = if let Some(ref image_path) = args.image {
        println!("\nLoading image: {}", image_path);
        let (pixels, sizes) = load_image(
            image_path,
            config.vision_config.patch_size,
            config.spatial_merge_size,
            &device,
        )?;
        // Convert pixels to model dtype (BF16/F16/F32) to match vision_tower weights
        let pixels = pixels.to_dtype(dtype)?;
        println!("  Image shape: {:?}", pixels.dims());
        println!("  Image dtype: {:?}", pixels.dtype());
        println!("  Image sizes: {:?}", sizes);
        (Some(pixels), Some(sizes))
    } else {
        println!("\nNo image provided, running in text-only mode");
        (None, None)
    };

    // Load model
    println!("\nLoading model weights...");
    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };

    if args.vision_only {
        // Vision-only mode for testing
        use candle_transformers::models::pixtral::vision_model::Model as PixtralVisionModel;

        let vision_model = PixtralVisionModel::new(&config.vision_config, vb.pp("vision_tower"))?;
        println!("Loaded vision model in {:?}", start.elapsed());

        if let Some(ref pixels) = pixel_values {
            // pixels already converted to model dtype above
            let embs = vision_model.forward(pixels)?;
            println!("Vision embeddings shape: {:?}", embs.dims());
            println!(
                "Vision embeddings (first 5 values): {:?}",
                embs.i((0, 0, ..5))?
            );
        } else {
            println!("No image provided for vision-only mode");
        }
    } else {
        let model = Mistral3ForConditionalGeneration::new(&config, vb)?;
        println!("Loaded model in {:?}", start.elapsed());

        // Load tokenizer
        let tokenizer = load_tokenizer(&tokenizer_path)?;

        let mut pipeline = TextGeneration::new(
            model,
            tokenizer,
            args.seed,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repeat_penalty,
            args.repeat_last_n,
            &device,
            config.image_token_index,
        );

        println!("\nGenerating...\n");
        pipeline.run(
            &args.prompt,
            pixel_values.as_ref(),
            image_sizes.as_deref(),
            args.sample_len,
        )?;
    }

    Ok(())
}
