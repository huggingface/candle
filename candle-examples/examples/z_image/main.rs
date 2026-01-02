//! Z-Image Text-to-Image Generation Example
//!
//! Z-Image is a text-to-image generation model from Alibaba using Flow Matching.
//!
//! # Running the example
//!
//! ```bash
//! # With Metal (Apple Silicon) - auto-download from HuggingFace
//! cargo run --features metal --example z_image --release -- \
//!     --model turbo \
//!     --prompt "A beautiful landscape with mountains" \
//!     --height 1024 --width 1024 --num-steps 9
//!
//! # With CUDA
//! cargo run --features cuda --example z_image --release -- \
//!     --model turbo \
//!     --prompt "A beautiful landscape" --height 1024 --width 1024
//!
//! # With local weights
//! cargo run --features metal --example z_image --release -- \
//!     --model turbo --model-path weights/Z-Image-Turbo \
//!     --prompt "A cat" --height 512 --width 512
//!
//! # On CPU (slow)
//! cargo run --example z_image --release -- --cpu \
//!     --model turbo \
//!     --prompt "A cat" --height 512 --width 512
//! ```
//!
//! # Model Files
//!
//! Models are automatically downloaded from HuggingFace, or you can download manually:
//! <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>

use anyhow::{Error as E, Result};
use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    calculate_shift, get_noise, postprocess_image, AutoEncoderKL, Config,
    FlowMatchEulerDiscreteScheduler, SchedulerConfig, TextEncoderConfig, VaeConfig,
    ZImageTextEncoder, ZImageTransformer2DModel,
};
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// Z-Image scheduler constants
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    /// Z-Image-Turbo: optimized for fast inference (8-9 steps)
    Turbo,
}

impl Model {
    fn repo(&self) -> &'static str {
        match self {
            Self::Turbo => "Tongyi-MAI/Z-Image-Turbo",
        }
    }

    fn default_steps(&self) -> usize {
        match self {
            Self::Turbo => 9,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A beautiful landscape with mountains and a lake"
    )]
    prompt: String,

    /// The negative prompt (for CFG).
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Number of inference steps.
    #[arg(long)]
    num_steps: Option<usize>,

    /// Guidance scale for CFG.
    #[arg(long, default_value_t = 5.0)]
    guidance_scale: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value = "turbo")]
    model: Model,

    /// Override path to the model weights directory (uses HuggingFace by default).
    #[arg(long)]
    model_path: Option<String>,

    /// Output image filename.
    #[arg(long, default_value = "z_image_output.png")]
    output: String,
}

/// Format user prompt for Qwen3 chat template
/// Corresponds to add_generation_prompt=True, enable_thinking=True
///
/// Format:
/// <|im_start|>user
/// {prompt}<|im_end|>
/// <|im_start|>assistant
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

fn run(args: Args) -> Result<()> {
    let num_steps = args.num_steps.unwrap_or_else(|| args.model.default_steps());

    println!("Z-Image Text-to-Image Generation");
    println!("================================");
    println!("Model: {:?}", args.model);
    println!("Prompt: {}", args.prompt);
    println!("Size: {}x{}", args.width, args.height);
    println!("Steps: {}", num_steps);
    println!("Guidance scale: {}", args.guidance_scale);

    let device = candle_examples::device(args.cpu)?;
    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
        println!("Seed: {}", seed);
    }
    let dtype = device.bf16_default_to_f32();

    // Resolve model: use provided path or download from HuggingFace
    let api = Api::new()?;
    let repo = api.model(args.model.repo().to_string());
    let use_local = args.model_path.is_some();
    let model_path = args.model_path.map(std::path::PathBuf::from);

    if use_local {
        println!(
            "\nLoading models from local path: {}",
            model_path.as_ref().unwrap().display()
        );
    } else {
        println!(
            "\nDownloading model from HuggingFace: {}",
            args.model.repo()
        );
    }

    // ==================== Load Tokenizer ====================
    println!("Loading tokenizer...");
    let tokenizer_path = if use_local {
        model_path
            .as_ref()
            .unwrap()
            .join("tokenizer")
            .join("tokenizer.json")
    } else {
        repo.get("tokenizer/tokenizer.json")?
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // ==================== Load Text Encoder ====================
    println!("Loading text encoder...");
    let text_encoder_config_path = if use_local {
        model_path
            .as_ref()
            .unwrap()
            .join("text_encoder")
            .join("config.json")
    } else {
        repo.get("text_encoder/config.json")?
    };
    let text_encoder_cfg: TextEncoderConfig = if text_encoder_config_path.exists() {
        serde_json::from_reader(std::fs::File::open(&text_encoder_config_path)?)?
    } else {
        TextEncoderConfig::z_image()
    };

    let text_encoder_weights = {
        let files: Vec<std::path::PathBuf> = if use_local {
            (1..=3)
                .map(|i| {
                    model_path
                        .as_ref()
                        .unwrap()
                        .join("text_encoder")
                        .join(format!("model-{:05}-of-00003.safetensors", i))
                })
                .filter(|p| p.exists())
                .collect()
        } else {
            (1..=3)
                .map(|i| repo.get(&format!("text_encoder/model-{:05}-of-00003.safetensors", i)))
                .filter_map(|r| r.ok())
                .collect()
        };

        if files.is_empty() {
            anyhow::bail!("Text encoder weights not found");
        }

        let files: Vec<&str> = files.iter().map(|p| p.to_str().unwrap()).collect();
        unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? }
    };

    let text_encoder = ZImageTextEncoder::new(&text_encoder_cfg, text_encoder_weights)?;

    // ==================== Load Transformer ====================
    println!("Loading transformer...");
    let transformer_config_path = if use_local {
        model_path
            .as_ref()
            .unwrap()
            .join("transformer")
            .join("config.json")
    } else {
        repo.get("transformer/config.json")?
    };
    let transformer_cfg: Config = if transformer_config_path.exists() {
        serde_json::from_reader(std::fs::File::open(&transformer_config_path)?)?
    } else {
        Config::z_image_turbo()
    };

    let transformer_weights = {
        let files: Vec<std::path::PathBuf> = if use_local {
            (1..=3)
                .map(|i| {
                    model_path
                        .as_ref()
                        .unwrap()
                        .join("transformer")
                        .join(format!(
                            "diffusion_pytorch_model-{:05}-of-00003.safetensors",
                            i
                        ))
                })
                .filter(|p| p.exists())
                .collect()
        } else {
            (1..=3)
                .map(|i| {
                    repo.get(&format!(
                        "transformer/diffusion_pytorch_model-{:05}-of-00003.safetensors",
                        i
                    ))
                })
                .filter_map(|r| r.ok())
                .collect()
        };

        if files.is_empty() {
            anyhow::bail!("Transformer weights not found");
        }

        let files: Vec<&str> = files.iter().map(|p| p.to_str().unwrap()).collect();
        unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? }
    };

    let transformer = ZImageTransformer2DModel::new(&transformer_cfg, transformer_weights)?;

    // ==================== Load VAE ====================
    println!("Loading VAE...");
    let vae_config_path = if use_local {
        model_path.as_ref().unwrap().join("vae").join("config.json")
    } else {
        repo.get("vae/config.json")?
    };
    let vae_cfg: VaeConfig = if vae_config_path.exists() {
        serde_json::from_reader(std::fs::File::open(&vae_config_path)?)?
    } else {
        VaeConfig::z_image()
    };

    let vae_path = if use_local {
        let path = model_path
            .as_ref()
            .unwrap()
            .join("vae")
            .join("diffusion_pytorch_model.safetensors");
        if !path.exists() {
            anyhow::bail!("VAE weights not found at {:?}", path);
        }
        path
    } else {
        repo.get("vae/diffusion_pytorch_model.safetensors")?
    };

    let vae_weights = unsafe {
        VarBuilder::from_mmaped_safetensors(&[vae_path.to_str().unwrap()], dtype, &device)?
    };
    let vae = AutoEncoderKL::new(&vae_cfg, vae_weights)?;

    // ==================== Initialize Scheduler ====================
    let scheduler_cfg = SchedulerConfig::z_image_turbo();
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);

    // ==================== Prepare Inputs ====================
    println!("\nTokenizing prompt...");
    let formatted_prompt = format_prompt_for_qwen3(&args.prompt);
    let tokens = tokenizer
        .encode(formatted_prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("Token count: {}", tokens.len());

    // Create input tensor
    let input_ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &device)?;

    // Get text embeddings (from second-to-last layer)
    println!("Encoding text...");
    let cap_feats = text_encoder.forward(&input_ids)?;
    let cap_mask = Tensor::ones((1, tokens.len()), DType::U8, &device)?;

    // Process negative prompt for CFG
    let (neg_cap_feats, neg_cap_mask) = if !args.negative_prompt.is_empty()
        && args.guidance_scale > 1.0
    {
        let formatted_neg = format_prompt_for_qwen3(&args.negative_prompt);
        let neg_tokens = tokenizer
            .encode(formatted_neg.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let neg_input_ids = Tensor::from_vec(neg_tokens.clone(), (1, neg_tokens.len()), &device)?;
        let neg_feats = text_encoder.forward(&neg_input_ids)?;
        let neg_mask = Tensor::ones((1, neg_tokens.len()), DType::U8, &device)?;
        (Some(neg_feats), Some(neg_mask))
    } else {
        (None, None)
    };

    // ==================== Calculate Latent Dimensions ====================
    // Formula from Python pipeline: latent = 2 * (image_size // 16)
    // This ensures: latent is divisible by patch_size=2, and VAE decode (8x) gives correct size
    let patch_size = transformer_cfg.all_patch_size[0];
    let vae_align = 16; // vae_scale_factor * 2 = 8 * 2 = 16

    // Validate input dimensions
    if !args.height.is_multiple_of(vae_align) || !args.width.is_multiple_of(vae_align) {
        anyhow::bail!(
            "Image dimensions must be divisible by {}. Got {}x{}. \
             Try {}x{} or {}x{} instead.",
            vae_align,
            args.width,
            args.height,
            (args.width / vae_align) * vae_align,
            (args.height / vae_align) * vae_align,
            ((args.width / vae_align) + 1) * vae_align,
            ((args.height / vae_align) + 1) * vae_align
        );
    }

    // Correct latent size formula: 2 * (image_size // 16)
    let latent_h = 2 * (args.height / vae_align);
    let latent_w = 2 * (args.width / vae_align);
    println!("Latent size: {}x{}", latent_w, latent_h);

    // Calculate image sequence length for shift
    let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
    let mu = calculate_shift(
        image_seq_len,
        BASE_IMAGE_SEQ_LEN,
        MAX_IMAGE_SEQ_LEN,
        BASE_SHIFT,
        MAX_SHIFT,
    );
    println!("Image sequence length: {}, mu: {:.4}", image_seq_len, mu);

    // Set timesteps
    scheduler.set_timesteps(num_steps, Some(mu));

    // ==================== Generate Initial Noise ====================
    println!("\nGenerating initial noise...");
    let mut latents = get_noise(1, 16, latent_h, latent_w, &device)?.to_dtype(dtype)?;

    // Add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
    latents = latents.unsqueeze(2)?;

    // ==================== Denoising Loop ====================
    println!("\nStarting denoising loop ({} steps)...", num_steps);

    for step in 0..num_steps {
        let t = scheduler.current_timestep_normalized();
        let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;

        // Model prediction
        let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;

        // Apply CFG if guidance_scale > 1.0
        let noise_pred = if args.guidance_scale > 1.0 {
            if let (Some(ref neg_feats), Some(ref neg_mask)) = (&neg_cap_feats, &neg_cap_mask) {
                let neg_pred = transformer.forward(&latents, &t_tensor, neg_feats, neg_mask)?;
                // CFG: pred = neg + scale * (pos - neg)
                let diff = (&noise_pred - &neg_pred)?;
                (&neg_pred + (diff * args.guidance_scale)?)?
            } else {
                // No negative prompt, use unconditional with zeros
                noise_pred
            }
        } else {
            noise_pred
        };

        // Negate the prediction (Z-Image specific)
        let noise_pred = noise_pred.neg()?;

        // Remove frame dimension for scheduler: (B, C, 1, H, W) -> (B, C, H, W)
        let noise_pred_4d = noise_pred.squeeze(2)?;
        let latents_4d = latents.squeeze(2)?;

        // Scheduler step
        let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;

        // Add back frame dimension
        latents = prev_latents.unsqueeze(2)?;

        println!(
            "Step {}/{}: t = {:.4}, sigma = {:.4}",
            step + 1,
            num_steps,
            t,
            scheduler.current_sigma()
        );
    }

    // ==================== VAE Decode ====================
    println!("\nDecoding latents with VAE...");
    // Remove frame dimension: (B, C, 1, H, W) -> (B, C, H, W)
    let latents = latents.squeeze(2)?;
    let image = vae.decode(&latents)?;

    // Post-process: [-1, 1] -> [0, 255]
    let image = postprocess_image(&image)?;

    // ==================== Save Image ====================
    println!("Saving image to {}...", args.output);
    let image = image.i(0)?; // Remove batch dimension
    candle_examples::save_image(&image, &args.output)?;

    println!("\nDone! Image saved to {}", args.output);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
