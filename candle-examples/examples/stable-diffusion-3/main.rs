mod clip;
mod sampling;
mod vae;

use candle::{DType, IndexOp, Tensor};
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};

use crate::clip::StableDiffusion3TripleClipWithTokenizer;
use crate::vae::{build_sd3_vae_autoencoder, sd3_vae_vb_rename};

use anyhow::{Ok, Result};
use clap::Parser;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "3-medium")]
    V3Medium,
    #[value(name = "3.5-large")]
    V3_5Large,
    #[value(name = "3.5-large-turbo")]
    V3_5LargeTurbo,
}

impl Which {
    fn is_3_5(&self) -> bool {
        match self {
            Self::V3Medium => false,
            Self::V3_5Large | Self::V3_5LargeTurbo => true,
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A cute rusty robot holding a candle torch in its hand, \
        with glowing neon text \"LETS GO RUSTY\" displayed on its chest, \
        bright background, high quality, 4k"
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

    /// Use flash_attn to accelerate attention operation in the MMDiT.
    #[arg(long)]
    use_flash_attn: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// The model to use.
    #[arg(long, default_value = "3-medium")]
    which: Which,

    /// The seed to use when generating random samples.
    #[arg(long)]
    num_inference_steps: Option<usize>,

    // CFG scale.
    #[arg(long)]
    cfg_scale: Option<f64>,

    // Time shift factor (alpha).
    #[arg(long, default_value_t = 3.0)]
    time_shift: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        uncond_prompt,
        cpu,
        tracing,
        use_flash_attn,
        height,
        width,
        num_inference_steps,
        cfg_scale,
        time_shift,
        seed,
        which,
    } = Args::parse();

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(cpu)?;
    let default_inference_steps = match which {
        Which::V3_5Large => 28,
        Which::V3_5LargeTurbo => 4,
        Which::V3Medium => 28,
    };
    let num_inference_steps = num_inference_steps.unwrap_or(default_inference_steps);
    let default_cfg_scale = match which {
        Which::V3_5Large => 4.0,
        Which::V3_5LargeTurbo => 1.0,
        Which::V3Medium => 4.0,
    };
    let cfg_scale = cfg_scale.unwrap_or(default_cfg_scale);

    let api = hf_hub::api::sync::Api::new()?;
    let img = if which.is_3_5() {
        let sai_repo = {
            let name = match which {
                Which::V3_5Large => "stabilityai/stable-diffusion-3.5-large",
                Which::V3_5LargeTurbo => "stabilityai/stable-diffusion-3.5-large-turbo",
                Which::V3Medium => unreachable!(),
            };
            api.repo(hf_hub::Repo::model(name.to_string()))
        };
        let clip_g_file = sai_repo.get("text_encoders/clip_g.safetensors")?;
        let clip_l_file = sai_repo.get("text_encoders/clip_l.safetensors")?;
        let t5xxl_file = sai_repo.get("text_encoders/t5xxl_fp16.safetensors")?;
        let model_file = {
            let model_file = match which {
                Which::V3_5Large => "sd3.5_large.safetensors",
                Which::V3_5LargeTurbo => "sd3.5_large_turbo.safetensors",
                Which::V3Medium => unreachable!(),
            };
            sai_repo.get(model_file)?
        };
        let (context, y) = {
            let mut triple = StableDiffusion3TripleClipWithTokenizer::new_split(
                &clip_g_file,
                &clip_l_file,
                &t5xxl_file,
                &device,
            )?;
            let (context, y) = triple.encode_text_to_embedding(prompt.as_str(), &device)?;
            let (context_uncond, y_uncond) =
                triple.encode_text_to_embedding(uncond_prompt.as_str(), &device)?;
            (
                Tensor::cat(&[context, context_uncond], 0)?,
                Tensor::cat(&[y, y_uncond], 0)?,
            )
        };

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F16, &device)?
        };
        let x = {
            let mmdit = MMDiT::new(
                &MMDiTConfig::sd3_5_large(),
                use_flash_attn,
                vb.pp("model.diffusion_model"),
            )?;

            if let Some(seed) = seed {
                device.set_seed(seed)?;
            }
            let start_time = std::time::Instant::now();
            let x = sampling::euler_sample(
                &mmdit,
                &y,
                &context,
                num_inference_steps,
                cfg_scale,
                time_shift,
                height,
                width,
            )?;
            let dt = start_time.elapsed().as_secs_f32();
            println!(
                "Sampling done. {num_inference_steps} steps. {:.2}s. Average rate: {:.2} iter/s",
                dt,
                num_inference_steps as f32 / dt
            );
            x
        };

        {
            let vb_vae = vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
            let autoencoder = build_sd3_vae_autoencoder(vb_vae)?;

            // Apply TAESD3 scale factor. Seems to be significantly improving the quality of the image.
            // https://github.com/comfyanonymous/ComfyUI/blob/3c60ecd7a83da43d694e26a77ca6b93106891251/nodes.py#L721-L723
            autoencoder.decode(&((x / 1.5305)? + 0.0609)?)?
        }
    } else {
        let sai_repo = {
            let name = "stabilityai/stable-diffusion-3-medium";
            api.repo(hf_hub::Repo::model(name.to_string()))
        };
        let model_file = sai_repo.get("sd3_medium_incl_clips_t5xxlfp16.safetensors")?;
        let vb_fp16 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F16, &device)?
        };

        let (context, y) = {
            let vb_fp32 = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)?
            };
            let mut triple = StableDiffusion3TripleClipWithTokenizer::new(
                vb_fp16.pp("text_encoders"),
                vb_fp32.pp("text_encoders"),
            )?;
            let (context, y) = triple.encode_text_to_embedding(prompt.as_str(), &device)?;
            let (context_uncond, y_uncond) =
                triple.encode_text_to_embedding(uncond_prompt.as_str(), &device)?;
            (
                Tensor::cat(&[context, context_uncond], 0)?,
                Tensor::cat(&[y, y_uncond], 0)?,
            )
        };

        let x = {
            let mmdit = MMDiT::new(
                &MMDiTConfig::sd3_medium(),
                use_flash_attn,
                vb_fp16.pp("model.diffusion_model"),
            )?;

            if let Some(seed) = seed {
                device.set_seed(seed)?;
            }
            let start_time = std::time::Instant::now();
            let x = sampling::euler_sample(
                &mmdit,
                &y,
                &context,
                num_inference_steps,
                cfg_scale,
                time_shift,
                height,
                width,
            )?;
            let dt = start_time.elapsed().as_secs_f32();
            println!(
                "Sampling done. {num_inference_steps} steps. {:.2}s. Average rate: {:.2} iter/s",
                dt,
                num_inference_steps as f32 / dt
            );
            x
        };

        {
            let vb_vae = vb_fp16.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
            let autoencoder = build_sd3_vae_autoencoder(vb_vae)?;

            // Apply TAESD3 scale factor. Seems to be significantly improving the quality of the image.
            // https://github.com/comfyanonymous/ComfyUI/blob/3c60ecd7a83da43d694e26a77ca6b93106891251/nodes.py#L721-L723
            autoencoder.decode(&((x / 1.5305)? + 0.0609)?)?
        }
    };
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(candle::DType::U8)?;
    candle_examples::save_image(&img.i(0)?, "out.jpg")?;
    Ok(())
}
