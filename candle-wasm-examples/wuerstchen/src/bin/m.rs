
use std::io::Cursor;

use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip;
use candle_transformers::models::wuerstchen;

use candle::{DType, Device, IndexOp, Tensor};
use anyhow::Error as E;

use tokenizers::Tokenizer;
use serde::{Serialize, Deserialize};



use wasm_bindgen::prelude::*;
use wasm_helper::{generic_error::{GenericError, GenericResult}, hfhub::api::Api, opfs::read_file, safetensor_var_builder::var_builder_from_opfs_safetensors};

const PRIOR_GUIDANCE_SCALE: f64 = 4.0;
const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

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
    async fn get(&self, filename: Option<String>) -> GenericResult<std::path::PathBuf> {
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
                let filename = Api::new()?.model(repo.to_string()).get(path).await?;
                log::info!("returned file: {:?}", filename);
                Ok(filename)
            }
        }
    }
}

// fn output_filename(
//     basename: &str,
//     sample_idx: i64,
//     num_samples: i64,
//     timestep_idx: Option<usize>,
// ) -> String {
//     let filename = if num_samples > 1 {
//         match basename.rsplit_once('.') {
//             None => format!("{basename}.{sample_idx}.png"),
//             Some((filename_no_extension, extension)) => {
//                 format!("{filename_no_extension}.{sample_idx}.{extension}")
//             }
//         }
//     } else {
//         basename.to_string()
//     };
//     match timestep_idx {
//         None => filename,
//         Some(timestep_idx) => match filename.rsplit_once('.') {
//             None => format!("{filename}-{timestep_idx}.png"),
//             Some((filename_no_extension, extension)) => {
//                 format!("{filename_no_extension}-{timestep_idx}.{extension}")
//             }
//         },
//     }
// }

async fn encode_prompt(
    prompt: &str,
    uncond_prompt: Option<&str>,
    tokenizer: std::path::PathBuf,
    clip_weights: std::path::PathBuf,
    clip_config: stable_diffusion::clip::Config,
    device: &Device,
) -> GenericResult<Tensor> {
    log::info!("Encode Prompt");
    let data = read_file(tokenizer).await?;
    //log::info!("data : {:?}", data);
    let tokenizer = Tokenizer::from_bytes(&data).map_err(E::msg)?;
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    log::info!("Running with prompt \"{prompt}\".");
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

    log::info!("Building the clip transformer.");
    let vs = var_builder_from_opfs_safetensors(&clip_weights, DType::F32, device).await?;
    let text_model =   clip::ClipTextTransformer::new(vs, &clip_config)?;

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







#[wasm_bindgen]
pub struct Model {
    device : Device

}

#[derive(Debug, Serialize, Deserialize)]
struct DeviceConfig{
    #[serde(default = "default_use_gpu")]
    use_gpu : bool,
    #[serde(default = "default_meta_buffer_size")]
    meta_buffer_size : u32, 
    #[serde(default = "default_max_workload_size")]
    max_workload_size : u64, 
    #[serde(default = "default_buffer_cached_max_allowed_size")]
    buffer_cached_max_allowed_size : u64,
    #[serde(default = "default_use_cache")]
    use_cache : bool,
}


fn default_max_workload_size() -> u64 {
    1024u64*1024*1024*2 //2gb,
}


fn default_meta_buffer_size() -> u32 {
    10*1024*1024//10mb
}


fn default_buffer_cached_max_allowed_size() -> u64 {
    1024*1024*1024*8 //8gb
}

fn default_use_cache() -> bool {
    true //8gb
}

fn default_use_gpu() -> bool {
    true //8gb
}


#[derive(Debug, Serialize, Deserialize)]
struct Args {
    /// The prompt to be used for image generation.
    #[serde(default = "default_prompt")]
    prompt: String,

    #[serde(default)]
    uncond_prompt: String,

    /// Run on CPU rather than on GPU.
    #[serde(default)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[serde(default)]
    tracing: bool,

    #[serde(rename = "use_flash_attn")]
    #[serde(default)]
    use_flash_attn: bool,

    /// The height in pixels of the generated image.
    height: Option<usize>,

    /// The width in pixels of the generated image.
    width: Option<usize>,

    /// The decoder weight file, in .safetensors format.
    decoder_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    clip_weights: Option<String>,

    /// The CLIP weight file used by the prior model, in .safetensors format.
    prior_clip_weights: Option<String>,

    /// The prior weight file, in .safetensors format.
    prior_weights: Option<String>,

    /// The VQGAN weight file, in .safetensors format.
    vqgan_weights: Option<String>,

    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The file specifying the tokenizer to used for prior tokenization.
    prior_tokenizer: Option<String>,

    /// The number of samples to generate.
    #[serde(default = "default_num_samples")]
    num_samples: i64,

    /// The name of the final image to generate.
    #[serde(default = "default_final_image")]
    final_image: String,

    #[serde(default = "default_prior_steps")]
    prior_steps : u64,

    #[serde(default = "default_vgan_steps")]
    vgan_steps : u64
}


fn default_prompt() -> String {
    "A very realistic photo of a rusty robot walking on a sandy beach".to_string()
}

fn default_num_samples() -> i64 {
    1
}

fn default_final_image() -> String {
    "sd_final.png".to_string()
}

fn default_prior_steps() -> u64 {
    2
}

fn default_vgan_steps() -> u64 {
    2
}

#[wasm_bindgen]
impl Model {

    #[wasm_bindgen(constructor)]
    pub async fn load(config: String) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::new(log::Level::Info).message_on_new_line());
        
        let args : DeviceConfig = serde_json::from_str(&config)?;
        let DeviceConfig {
            use_gpu,
            buffer_cached_max_allowed_size,
            max_workload_size,
            use_cache,
            meta_buffer_size
        } = args;

        let device = match !use_gpu{
            true => Device::Cpu,
            false =>  
            {
                let config = candle::wgpu_backend::DeviceConfig{buffer_cached_max_allowed_size, max_workload_size, meta_buffer_size, use_cache };
                Device::new_webgpu_config(0, config).await?
            }
        };
        
        

        return Ok(Model{device});
    }

    pub async fn run(&self, config: String) -> Result<JsValue, JsError> {
        log::info!("Start run, config: {config}");

        //clear_all(true).await?;

        //clear_directory(open_dir("/.cache/huggingface/models--warp-ai--wuerstchen").await?, true).await?;

        let args : Args = serde_json::from_str(&config)?;
        log::info!("loaded args");
        let Args {
            prompt,
            uncond_prompt,
            height,
            width,
            tokenizer,
            num_samples,
            clip_weights,
            prior_weights,
            vqgan_weights,
            decoder_weights,
            prior_steps,
            vgan_steps,
            ..
        } = args;
        
        // let _guard = if tracing {
        //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        //     tracing_subscriber::registry().with(chrome_layer).init();
        //     Some(guard)
        // } else {
        //     None
        // };
    
        let device = &self.device;
        log::info!("loaded device");

        let height = height.unwrap_or(1024);
        let width = width.unwrap_or(1024);
    
        log::info!("loading Models:");
        let prior_text_embeddings = {
            let tokenizer = ModelFile::PriorTokenizer.get(args.prior_tokenizer).await?;

            log::info!("tokenizer loaded");

            let weights = ModelFile::PriorClip.get(args.prior_clip_weights).await?;

            log::info!("weights loaded");

            encode_prompt(
                &prompt,
                Some(&uncond_prompt),
                tokenizer.clone(),
                weights,
                stable_diffusion::clip::Config::wuerstchen_prior(),
                device,
            ).await.map_err(|f| JsError::new(&f.to_string()))?
        };

        log::info!("loaded Models:");

        log::info!("generated prior text embeddings {prior_text_embeddings:?}");
    
        let text_embeddings = {
            let tokenizer = ModelFile::Tokenizer.get(tokenizer).await?;
            let weights = ModelFile::Clip.get(clip_weights).await?;
            encode_prompt(
                &prompt,
                None,
                tokenizer.clone(),
                weights,
                stable_diffusion::clip::Config::wuerstchen(),
                device,
            ).await.map_err(|f| JsError::new(&f.to_string()))?
        };
         log::info!("generated text embeddings {text_embeddings:?}");
    
         log::info!("Building the prior.");
        let b_size = 1;
        let image_embeddings = {
            // https://huggingface.co/warp-ai/wuerstchen-prior/blob/main/prior/config.json
            let latent_height = (height as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
            let latent_width = (width as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
            let mut latents = Tensor::randn(
                0f32,
                1f32,
                (b_size, PRIOR_CIN, latent_height, latent_width),
                device,
            )?;
    
            let prior = {
                let file = ModelFile::Prior.get(prior_weights).await?;
                let vb = var_builder_from_opfs_safetensors(file, DType::F32, device).await?;
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
            let prior_scheduler = wuerstchen::ddpm::DDPMWScheduler::new(prior_steps as usize, Default::default())?;
            let timesteps = prior_scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            log::info!("prior denoising");
            for (index, &t) in timesteps.iter().enumerate() {
                //let start_time = std::time::Instant::now();
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
                let ratio = (Tensor::ones(2, DType::F32, device)? * t)?;
                let noise_pred = prior.forward(&latent_model_input, &ratio, &prior_text_embeddings)?;
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_text, noise_pred_uncond) = (&noise_pred[0], &noise_pred[1]);
                let noise_pred = (noise_pred_uncond
                    + ((noise_pred_text - noise_pred_uncond)? * PRIOR_GUIDANCE_SCALE)?)?;
                latents = prior_scheduler.step(&noise_pred, t, &latents)?;
                
                //let dt = start_time.elapsed().as_secs_f32();
                
                //log::info!("step {}/{} done, {:.2}s", index + 1, timesteps.len(), dt);
                device.synchronize_async().await?;
                log::info!("step {}/{} done", index + 1, timesteps.len());
            }
            ((latents * 42.)? - 1.)?
        };
    
        log::info!("Building the vqgan.");
        let vqgan = {
            let file = ModelFile::VqGan.get(vqgan_weights).await?;
            let vb = var_builder_from_opfs_safetensors(file, DType::F32, device).await?;
            wuerstchen::paella_vq::PaellaVQ::new(vb)?
        };
    
        log::info!("Building the decoder.");
    
        // https://huggingface.co/warp-ai/wuerstchen/blob/main/decoder/config.json
        let decoder = {
            let file = ModelFile::Decoder.get(decoder_weights).await?;
            let vb = var_builder_from_opfs_safetensors(file, DType::F32, device).await?;
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
                device,
            )?;
    
             log::info!("diffusion process with prior {image_embeddings:?}");
            let scheduler = wuerstchen::ddpm::DDPMWScheduler::new(vgan_steps as usize, Default::default())?;
            let timesteps = scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            for (index, &t) in timesteps.iter().enumerate() {
                //let start_time = std::time::Instant::now();
                let ratio = (Tensor::ones(1, DType::F32, device)? * t)?;
                let noise_pred =
                    decoder.forward(&latents, &ratio, &image_embeddings, Some(&text_embeddings))?;
                latents = scheduler.step(&noise_pred, t, &latents)?;
                //let dt = start_time.elapsed().as_secs_f32();
                //log::info!("step {}/{} done, {:.2}s", index + 1, timesteps.len(), dt);
                device.synchronize_async().await?;
                log::info!("step {}/{} done", index + 1, timesteps.len());
            }
             log::info!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );

            log::info!("decoding image:");


            let image = vqgan.decode(&(&latents * 0.3764)?)?;

            log::info!("Image decoded");


            let image = (image.clamp(0f32, 1f32)? * 255.)?.to_cpu_device().await?
                .to_dtype(DType::U8)?
                .i(0)?;

            //let image_filename = output_filename(&final_image, idx + 1, num_samples, None);

            log::info!( "Image created");

            let image_png = save_image(&image)?;

            log::info!( "Image saved");
            // match device{
            //     Device::WebGpu(wgpu_device) => {wgpu_device.clear_cache()},
            //     _ => {}
            // }
            return  Ok(js_sys::Uint8Array::from(&image_png[..]).into());
           
        }
        // match device{
        //     Device::WebGpu(wgpu_device) => {wgpu_device.clear_cache()},
        //     _ => {}
        // }
        Ok(JsValue::null())
        //Ok("Test Result".to_owned())
    }
}



// Saves an image to disk using the image crate, this expects an input with shape
// (c, height, width).
pub fn save_image(img: &Tensor) -> GenericResult<Vec<u8>> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        return Err(GenericError::from("save_image expects an input of shape (3, height, width)"))
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => return Err(GenericError::from("error saving image")),
        };
    let mut bytes: Vec<u8> = Vec::new();
    image.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png).map_err(|e| GenericError::Anyhow(e.into()))?;
    //image.save(p).map_err(candle::Error::wrap)?;
    Ok(bytes)
}


fn main()  {
}
