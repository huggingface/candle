mod clip;
mod sampling;
mod vae;

use candle::{DType, IndexOp, Tensor};
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};

use crate::clip::StableDiffusion3TripleClipWithTokenizer;
use crate::vae::{build_sd3_vae_autoencoder, sd3_vae_vb_rename};

use anyhow::{Ok, Result};
use clap::Parser;

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

    /// The CUDA device ID to use.
    #[arg(long, default_value = "0")]
    cuda_device_id: usize,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Use flash_attn to accelerate attention operation in the MMDiT.
    #[arg(long)]
    use_flash_attn: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value = "1024")]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value = "1024")]
    width: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value = "28")]
    num_inference_steps: usize,

    // CFG scale.
    #[arg(long, default_value = "4.0")]
    cfg_scale: f64,

    // Time shift factor (alpha).
    #[arg(long, default_value = "3.0")]
    time_shift: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Your main code here
    run(args)
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        uncond_prompt,
        cpu,
        cuda_device_id,
        tracing,
        use_flash_attn,
        height,
        width,
        num_inference_steps,
        cfg_scale,
        time_shift,
        seed,
    } = args;

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // TODO: Support and test on Metal.
    let device = if cpu {
        candle::Device::Cpu
    } else {
        candle::Device::cuda_if_available(cuda_device_id)?
    };

    let api = hf_hub::api::sync::Api::new()?;
    let sai_repo = {
        let name = "stabilityai/stable-diffusion-3-medium";
        api.repo(hf_hub::Repo::model(name.to_string()))
    };
    let model_file = sai_repo.get("sd3_medium_incl_clips_t5xxlfp16.safetensors")?;
    let vb_fp16 = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F16, &device)?
    };

    let (context, y) = {
        let vb_fp32 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_file.clone()],
                DType::F32,
                &device,
            )?
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

    let img = {
        let vb_vae = vb_fp16
            .clone()
            .rename_f(sd3_vae_vb_rename)
            .pp("first_stage_model");
        let autoencoder = build_sd3_vae_autoencoder(vb_vae)?;

        // Apply TAESD3 scale factor. Seems to be significantly improving the quality of the image.
        // https://github.com/comfyanonymous/ComfyUI/blob/3c60ecd7a83da43d694e26a77ca6b93106891251/nodes.py#L721-L723
        autoencoder.decode(&((x.clone() / 1.5305)? + 0.0609)?)?
    };
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(candle::DType::U8)?;
    candle_examples::save_image(&img.i(0)?, "out.jpg")?;
    Ok(())
}
