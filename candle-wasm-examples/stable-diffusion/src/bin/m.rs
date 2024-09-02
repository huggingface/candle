use std::io::Cursor;

use candle_transformers::models::stable_diffusion::{self, clip, unet_2d, vae};

use anyhow::Error as E;
use candle::{DType, Device, IndexOp, Tensor};

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use wasm_bindgen::prelude::*;
use wasm_helper::{
    generic_error::{GenericError, GenericResult},
    hfhub::api::Api,
    opfs::read_file,
    safetensor_var_builder::var_builder_from_opfs_safetensors,
};
use web_time::Instant;

#[wasm_bindgen]
pub struct Model {
    device: Device,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeviceConfig {
    #[serde(default = "default_use_gpu")]
    use_gpu: bool,
    #[serde(default = "default_meta_buffer_size")]
    meta_buffer_size: u32,
    #[serde(default = "default_max_workload_size")]
    max_workload_size: u64,
    #[serde(default = "default_buffer_cached_max_allowed_size")]
    buffer_cached_max_allowed_size: u64,
    #[serde(default = "default_use_cache")]
    use_cache: bool,
    #[serde(default = "default_queue_delay_miliseconds")]
    queue_delay_miliseconds: u32,
    #[serde(default = "default_flush_gpu_before_buffer_init")]
    flush_gpu_before_buffer_init: bool,
    #[serde(default = "default_queue_delay_factor")]
    queue_delay_factor: f32,
    #[serde(default = "default_buffer_mapping_size")]
    buffer_mapping_size: u32,
}

fn default_buffer_mapping_size() -> u32 {
    1
}

fn default_queue_delay_factor() -> f32 {
    0.0
}

fn default_queue_delay_miliseconds() -> u32 {
    0
}

fn default_flush_gpu_before_buffer_init() -> bool {
    false
}

fn default_max_workload_size() -> u64 {
    1024u64 * 1024 * 1024 * 2 //2gb,
}

fn default_meta_buffer_size() -> u32 {
    10 * 1024 * 1024 //10mb
}

fn default_buffer_cached_max_allowed_size() -> u64 {
    1024 * 1024 * 1024 * 8 //8gb
}

fn default_use_cache() -> bool {
    true //8gb
}

fn default_use_gpu() -> bool {
    true //8gb
}

use candle::{Module, D};
use stable_diffusion::vae::AutoEncoderKL;

#[derive(Deserialize)]
struct Args {
    /// The prompt to be used for image generation.
    #[serde(default = "default_prompt")]
    prompt: String,

    #[serde(default)]
    uncond_prompt: String,

    /// The height in pixels of the generated image.
    #[serde(default)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[serde(default)]
    width: Option<usize>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[serde(default)]
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[serde(default)]
    n_steps: Option<usize>,

    /// The number of samples to generate iteratively.
    #[serde(default = "default_num_samples")]
    num_samples: usize,

    /// The numbers of samples to generate simultaneously.
    #[serde(default = "default_num_batch")]
    bsize: usize,

    /// The name of the final image to generate.
    #[serde(default = "default_sd_version")]
    sd_version: StableDiffusionVersion,

    #[serde(default)]
    use_flash_attn: bool,

    #[serde(default)]
    use_f16: bool,

    #[serde(default)]
    guidance_scale: Option<f64>,

    #[serde(default)]
    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    #[serde(default = "default_im2im_strength")]
    img2img_strength: f64,

    /// The seed to use when generating random samples.
    #[serde(default)]
    seed: Option<u64>,
}

fn default_prompt() -> String {
    "A very realistic photo of a rusty robot walking on a sandy beach".to_string()
}

fn default_num_samples() -> usize {
    1
}

fn default_num_batch() -> usize {
    1
}

fn default_sd_version() -> StableDiffusionVersion {
    StableDiffusionVersion::V1_5
}

fn default_im2im_strength() -> f64 {
    0.8
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

impl ModelFile {
    async fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> GenericResult<std::path::PathBuf> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path).await?;
                log::info!("returned file: {:?}", filename);
                Ok(filename)
            }
        }
    }
}

// Saves an image to disk using the image crate, this expects an input with shape
// (c, height, width).
pub fn save_image2(img: &Tensor) -> GenericResult<Vec<u8>> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        return Err(GenericError::from(
            "save_image expects an input of shape (3, height, width)",
        ));
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => return Err(GenericError::from("error saving image")),
        };
    let mut bytes: Vec<u8> = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| GenericError::Anyhow(e.into()))?;
    //image.save(p).map_err(candle::Error::wrap)?;
    Ok(bytes)
}

#[allow(clippy::too_many_arguments)]
async fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
) -> GenericResult<Vec<Vec<u8>>> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?
        .to_device_async(&Device::Cpu)
        .await?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;

    let mut result = vec![];
    for batch in 0..bsize {
        let image = images.i(batch)?;
        result.push(save_image2(&image)?);
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
async fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> GenericResult<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };

    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16).await?;
    let data = read_file(tokenizer).await?;
    let tokenizer = Tokenizer::from_bytes(data).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    log::info!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        return Err(GenericError::from(format!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        ))
        .into());
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    log::info!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file
        .get(clip_weights, sd_version, false)
        .await?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };

    let vs = var_builder_from_opfs_safetensors(&clip_weights, DType::F32, device).await?;

    let text_model = clip::ClipTextTransformer::new(vs, &clip_config)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            return Err(GenericError::from(format!(
                "the negative prompt is too long, {} > max-tokens ({})",
                uncond_tokens.len(),
                sd_config.clip.max_position_embeddings
            ))
            .into());
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub async fn load(config: String) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::new(log::Level::Info).message_on_new_line());

        let args: DeviceConfig = serde_json::from_str(&config)?;
        let DeviceConfig {
            use_gpu,
            buffer_cached_max_allowed_size,
            max_workload_size,
            use_cache,
            meta_buffer_size,
            queue_delay_miliseconds,
            flush_gpu_before_buffer_init,
            buffer_mapping_size,
            ..
        } = args;

        let device = match !use_gpu {
            true => Device::Cpu,
            false => {
                let config = candle::WgpuDeviceConfig {
                    buffer_cached_max_allowed_size,
                    max_workload_size,
                    meta_buffer_size,
                    use_cache,
                    queue_delay_miliseconds,
                    flush_gpu_before_buffer_init,
                    buffer_mapping_size,
                    ..Default::default()
                };
                Device::new_webgpu_config(0, config).await?
            }
        };

        return Ok(Model { device });
    }

    pub async fn run(&self, config: String) -> Result<JsValue, JsError> {
        let args: Args = serde_json::from_str(&config)?;
        let Args {
            prompt,
            uncond_prompt,
            height,
            width,
            n_steps,
            sliced_attention_size,
            num_samples,
            bsize,
            sd_version,
            use_f16,
            guidance_scale,
            use_flash_attn,
            img2img,
            img2img_strength,
            seed,
            ..
        } = args;

        if !(0. ..=1.).contains(&img2img_strength) {
            return Err(GenericError::from(format!(
                "img2img-strength should be between 0 and 1, got {img2img_strength}"
            ))
            .into());
        }

        // let _guard = if tracing {
        //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        //     tracing_subscriber::registry().with(chrome_layer).init();
        //     Some(guard)
        // } else {
        //     None
        // };

        let guidance_scale = match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };
        let n_steps = match n_steps {
            Some(n_steps) => n_steps,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 30,
                StableDiffusionVersion::Turbo => 1,
            },
        };
        let dtype = if use_f16 { DType::F16 } else { DType::F32 };
        let sd_config = match sd_version {
            StableDiffusionVersion::V1_5 => {
                stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::V2_1 => {
                stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Xl => {
                stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
                sliced_attention_size,
                height,
                width,
            ),
        };

        let scheduler = sd_config.build_scheduler(n_steps)?;
        let device = &self.device;

        if let Some(seed) = seed {
            device.set_seed(seed)?;
        }
        let use_guide_scale = guidance_scale > 1.0;

        let which = match sd_version {
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
            _ => vec![true],
        };
        let mut text_embedding = vec![];

        for first in which {
            text_embedding.push(
                text_embeddings(
                    &prompt,
                    &uncond_prompt,
                    None,
                    None,
                    sd_version,
                    &sd_config,
                    use_f16,
                    &device,
                    dtype,
                    use_guide_scale,
                    first,
                )
                .await?,
            );
        }

        let text_embeddings = Tensor::cat(&text_embedding, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;
        log::info!("{text_embeddings:?}");

        log::info!("Building the autoencoder.");
        let vae_weights = ModelFile::Vae.get(None, sd_version, use_f16).await?;

        let vs_ae = var_builder_from_opfs_safetensors(&vae_weights, DType::F32, device).await?;
        let vae = vae::AutoEncoderKL::new(vs_ae, 3, 3, sd_config.autoencoder.clone())?;

        let init_latent_dist: Option<stable_diffusion::vae::DiagonalGaussianDistribution> =
            match &img2img {
                None => None,
                Some(_) => {
                    todo!()
                }
            };
        log::info!("Building the unet.");
        let unet_weights = ModelFile::Unet.get(None, sd_version, use_f16).await?;

        let vs_unet = var_builder_from_opfs_safetensors(&unet_weights, DType::F32, device).await?;
        let unet = unet_2d::UNet2DConditionModel::new(
            vs_unet,
            4,
            4,
            use_flash_attn,
            sd_config.unet.clone(),
        )?;

        let t_start = if img2img.is_some() {
            n_steps - (n_steps as f64 * img2img_strength) as usize
        } else {
            0
        };

        let vae_scale = match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };

        for idx in 0..num_samples {
            let timesteps = scheduler.timesteps();
            let latents = match &init_latent_dist {
                Some(init_latent_dist) => {
                    let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        scheduler.add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }
                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        (bsize, 4, sd_config.height / 8, sd_config.width / 8),
                        &device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(dtype)?;

            log::info!("starting sampling");
            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let start_time = Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                let latent_model_input =
                    scheduler.scale_model_input(latent_model_input, timestep)?;
                let noise_pred =
                    unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond
                        + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                device.synchronize_async().await?;
                let dt = start_time.elapsed().as_secs_f32();
                log::info!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);
            }

            log::info!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );

            device.synchronize_async().await?;
            let result = save_image(&vae, &latents, vae_scale, bsize).await?;

            match &device {
                candle::Device::WebGpu(gpu) => {
                    gpu.print_bindgroup_reuseinfo2();
                }
                _ => {}
            };

            if let Some(val) = result.first() {
                log::info!("Image saved");
                return Ok(js_sys::Uint8Array::from(&val[..]).into());
            }
        }
        Ok(JsValue::null())
    }
}

fn main() {}
