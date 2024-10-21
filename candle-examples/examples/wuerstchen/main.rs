#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::stable_diffusion;
use candle_transformers::models::wuerstchen;

use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use tokenizers::Tokenizer;

const PRIOR_GUIDANCE_SCALE: f64 = 4.0;
const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The decoder weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    decoder_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The CLIP weight file used by the prior model, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    prior_clip_weights: Option<String>,

    /// The prior weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    prior_weights: Option<String>,

    /// The VQGAN weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    vqgan_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for prior tokenization.
    prior_tokenizer: Option<String>,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    PriorTokenizer,
    Clip,
    PriorClip,
    Decoder,
    VqGan,
    Prior,
}

impl ModelFile {
    fn get(&self, filename: Option<String>) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo_main = "warp-ai/wuerstchen";
                let repo_prior = "warp-ai/wuerstchen-prior";
                let (repo, path) = match self {
                    Self::Tokenizer => (repo_main, "tokenizer/tokenizer.json"),
                    Self::PriorTokenizer => (repo_prior, "tokenizer/tokenizer.json"),
                    Self::Clip => (repo_main, "text_encoder/model.safetensors"),
                    Self::PriorClip => (repo_prior, "text_encoder/model.safetensors"),
                    Self::Decoder => (repo_main, "decoder/diffusion_pytorch_model.safetensors"),
                    Self::VqGan => (repo_main, "vqgan/diffusion_pytorch_model.safetensors"),
                    Self::Prior => (repo_prior, "prior/diffusion_pytorch_model.safetensors"),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn output_filename(
    basename: &str,
    sample_idx: i64,
    num_samples: i64,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

fn encode_prompt(
    prompt: &str,
    uncond_prompt: Option<&str>,
    tokenizer: std::path::PathBuf,
    clip_weights: std::path::PathBuf,
    clip_config: stable_diffusion::clip::Config,
    device: &Device,
) -> Result<Tensor> {
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let tokens_len = tokens.len();
    while tokens.len() < clip_config.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the clip transformer.");
    let text_model =
        stable_diffusion::build_clip_transformer(&clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward_with_mask(&tokens, tokens_len - 1)?;
    match uncond_prompt {
        None => Ok(text_embeddings),
        Some(uncond_prompt) => {
            let mut uncond_tokens = tokenizer
                .encode(uncond_prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let uncond_tokens_len = uncond_tokens.len();
            while uncond_tokens.len() < clip_config.max_position_embeddings {
                uncond_tokens.push(pad_id)
            }
            let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;

            let uncond_embeddings =
                text_model.forward_with_mask(&uncond_tokens, uncond_tokens_len - 1)?;
            let text_embeddings = Tensor::cat(&[text_embeddings, uncond_embeddings], 0)?;
            Ok(text_embeddings)
        }
    }
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        uncond_prompt,
        cpu,
        height,
        width,
        tokenizer,
        final_image,
        num_samples,
        clip_weights,
        prior_weights,
        vqgan_weights,
        decoder_weights,
        tracing,
        ..
    } = args;

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(cpu)?;
    let height = height.unwrap_or(1024);
    let width = width.unwrap_or(1024);

    let prior_text_embeddings = {
        let tokenizer = ModelFile::PriorTokenizer.get(args.prior_tokenizer)?;
        let weights = ModelFile::PriorClip.get(args.prior_clip_weights)?;
        encode_prompt(
            &prompt,
            Some(&uncond_prompt),
            tokenizer.clone(),
            weights,
            stable_diffusion::clip::Config::wuerstchen_prior(),
            &device,
        )?
    };
    println!("generated prior text embeddings {prior_text_embeddings:?}");

    let text_embeddings = {
        let tokenizer = ModelFile::Tokenizer.get(tokenizer)?;
        let weights = ModelFile::Clip.get(clip_weights)?;
        encode_prompt(
            &prompt,
            None,
            tokenizer.clone(),
            weights,
            stable_diffusion::clip::Config::wuerstchen(),
            &device,
        )?
    };
    println!("generated text embeddings {text_embeddings:?}");

    println!("Building the prior.");
    let b_size = 1;
    let image_embeddings = {
        // https://huggingface.co/warp-ai/wuerstchen-prior/blob/main/prior/config.json
        let latent_height = (height as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
        let latent_width = (width as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (b_size, PRIOR_CIN, latent_height, latent_width),
            &device,
        )?;

        let prior = {
            let file = ModelFile::Prior.get(prior_weights)?;
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device)?
            };
            wuerstchen::prior::WPrior::new(
                /* c_in */ PRIOR_CIN,
                /* c */ 1536,
                /* c_cond */ 1280,
                /* c_r */ 64,
                /* depth */ 32,
                /* nhead */ 24,
                args.use_flash_attn,
                vb,
            )?
        };
        let prior_scheduler = wuerstchen::ddpm::DDPMWScheduler::new(60, Default::default())?;
        let timesteps = prior_scheduler.timesteps();
        let timesteps = &timesteps[..timesteps.len() - 1];
        println!("prior denoising");
        for (index, &t) in timesteps.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
            let ratio = (Tensor::ones(2, DType::F32, &device)? * t)?;
            let noise_pred = prior.forward(&latent_model_input, &ratio, &prior_text_embeddings)?;
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_text, noise_pred_uncond) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred = (noise_pred_uncond
                + ((noise_pred_text - noise_pred_uncond)? * PRIOR_GUIDANCE_SCALE)?)?;
            latents = prior_scheduler.step(&noise_pred, t, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{} done, {:.2}s", index + 1, timesteps.len(), dt);
        }
        ((latents * 42.)? - 1.)?
    };

    println!("Building the vqgan.");
    let vqgan = {
        let file = ModelFile::VqGan.get(vqgan_weights)?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device)?
        };
        wuerstchen::paella_vq::PaellaVQ::new(vb)?
    };

    println!("Building the decoder.");

    // https://huggingface.co/warp-ai/wuerstchen/blob/main/decoder/config.json
    let decoder = {
        let file = ModelFile::Decoder.get(decoder_weights)?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device)?
        };
        wuerstchen::diffnext::WDiffNeXt::new(
            /* c_in */ DECODER_CIN,
            /* c_out */ DECODER_CIN,
            /* c_r */ 64,
            /* c_cond */ 1024,
            /* clip_embd */ 1024,
            /* patch_size */ 2,
            args.use_flash_attn,
            vb,
        )?
    };

    for idx in 0..num_samples {
        // https://huggingface.co/warp-ai/wuerstchen/blob/main/model_index.json
        let latent_height = (image_embeddings.dim(2)? as f64 * LATENT_DIM_SCALE) as usize;
        let latent_width = (image_embeddings.dim(3)? as f64 * LATENT_DIM_SCALE) as usize;

        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (b_size, DECODER_CIN, latent_height, latent_width),
            &device,
        )?;

        println!("diffusion process with prior {image_embeddings:?}");
        let scheduler = wuerstchen::ddpm::DDPMWScheduler::new(12, Default::default())?;
        let timesteps = scheduler.timesteps();
        let timesteps = &timesteps[..timesteps.len() - 1];
        for (index, &t) in timesteps.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let ratio = (Tensor::ones(1, DType::F32, &device)? * t)?;
            let noise_pred =
                decoder.forward(&latents, &ratio, &image_embeddings, Some(&text_embeddings))?;
            latents = scheduler.step(&noise_pred, t, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{} done, {:.2}s", index + 1, timesteps.len(), dt);
        }
        println!(
            "Generating the final image for sample {}/{}.",
            idx + 1,
            num_samples
        );
        let image = vqgan.decode(&(&latents * 0.3764)?)?;
        let image = (image.clamp(0f32, 1f32)? * 255.)?
            .to_dtype(DType::U8)?
            .i(0)?;
        let image_filename = output_filename(&final_image, idx + 1, num_samples, None);
        candle_examples::save_image(&image, image_filename)?
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
