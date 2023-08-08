#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod attention;
mod clip;
mod ddim;
mod embeddings;
mod resnet;
mod schedulers;
mod stable_diffusion;
mod unet_2d;
mod unet_2d_blocks;
mod utils;
mod vae;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use clap::Parser;
use tokenizers::Tokenizer;

const GUIDANCE_SCALE: f64 = 7.5;

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

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The UNet weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: String,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[arg(long)]
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
}

impl Args {
    fn clip_weights(&self) -> String {
        match &self.clip_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/pytorch_model.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/clip_v2.1.safetensors".to_string(),
            },
        }
    }

    fn vae_weights(&self) -> String {
        match &self.vae_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/vae.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/vae_v2.1.safetensors".to_string(),
            },
        }
    }

    fn unet_weights(&self) -> String {
        match &self.unet_weights {
            Some(w) => w.clone(),
            None => match self.sd_version {
                StableDiffusionVersion::V1_5 => "data/unet.safetensors".to_string(),
                StableDiffusionVersion::V2_1 => "data/unet_v2.1.safetensors".to_string(),
            },
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

fn run(args: Args) -> Result<()> {
    let clip_weights = args.clip_weights();
    let vae_weights = args.vae_weights();
    let unet_weights = args.unet_weights();
    let Args {
        prompt,
        uncond_prompt,
        cpu,
        height,
        width,
        n_steps,
        tokenizer,
        final_image,
        sliced_attention_size,
        num_samples,
        sd_version,
        ..
    } = args;
    let sd_config = match sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
    };

    let scheduler = sd_config.build_scheduler(n_steps)?;
    let device = candle_examples::device(cpu)?;

    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match tokenizer.get_padding() {
        Some(padding) => padding.pad_id,
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

    let mut uncond_tokens = tokenizer
        .encode(uncond_prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
        uncond_tokens.push(pad_id)
    }
    let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let text_model = sd_config.build_clip_transformer(&clip_weights, &device)?;
    let text_embeddings = text_model.forward(&tokens)?;
    let uncond_embeddings = text_model.forward(&uncond_tokens)?;
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?;

    println!("text-embeddings: {text_embeddings:?}");
    println!("Building the autoencoder.");
    let vae = sd_config.build_vae(&vae_weights, &device)?;
    println!("Building the unet.");
    let unet = sd_config.build_unet(&unet_weights, &device, 4)?;

    let bsize = 1;
    for idx in 0..num_samples {
        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (bsize, 4, sd_config.height / 8, sd_config.width / 8),
            &device,
        )?;

        // scale the initial noise by the standard deviation required by the scheduler
        latents = (latents * scheduler.init_noise_sigma())?;

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
            println!("Timestep {timestep_index}/{n_steps}");
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred =
                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * GUIDANCE_SCALE)?)?;
            latents = scheduler.step(&noise_pred, timestep, &latents)?;

            if args.intermediary_images {
                let image = vae.decode(&(&latents / 0.18215)?)?;
                let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
                let _image = (image * 255.)?.to_dtype(DType::U8);
                let _image_filename =
                    output_filename(&final_image, idx + 1, num_samples, Some(timestep_index + 1));
                // TODO: save igame
            }
        }

        println!(
            "Generating the final image for sample {}/{}.",
            idx + 1,
            num_samples
        );
        let image = vae.decode(&(&latents / 0.18215)?)?;
        // TODO: Add the clamping between 0 and 1.
        let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let _image = (image * 255.)?.to_dtype(DType::U8);
        let _image_filename = output_filename(&final_image, idx + 1, num_samples, None);
        // TODO: save image.
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
