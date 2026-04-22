#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use candle::{DType, Device, Result, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use candle_transformers::models::quantized_blip;

use tokenizers::Tokenizer;

enum Model {
    M(blip::BlipForConditionalGeneration),
    Q(quantized_blip::BlipForConditionalGeneration),
}

impl Model {
    fn text_decoder_forward(&mut self, xs: &Tensor, img_xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::M(m) => m.text_decoder().forward(xs, img_xs),
            Self::Q(m) => m.text_decoder().forward(xs, img_xs),
        }
    }
}

// TODO: Maybe add support for the conditional prompt.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the quantized version of the model.
    #[arg(long)]
    quantized: bool,
}

const SEP_TOKEN_ID: u32 = 102;

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 384, 384). OpenAI normalization is applied.
pub fn load_image<P: AsRef<std::path::Path>>(p: P) -> Result<Tensor> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean =
        Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], &Device::Cpu)?
        .reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            if args.quantized {
                let api = api.model("lmz/candle-blip".to_string());
                api.get("blip-image-captioning-large-q4k.gguf")?
            } else {
                let api = api.repo(hf_hub::Repo::with_revision(
                    "Salesforce/blip-image-captioning-large".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/18".to_string(),
                ));
                api.get("model.safetensors")?
            }
        }
        Some(model) => model.into(),
    };
    let tokenizer = match args.tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("Salesforce/blip-image-captioning-large".to_string());
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);

    let config = blip::Config::image_captioning_large();

    let device = candle_examples::device(args.cpu)?;
    let (image_embeds, device, mut model) = if args.quantized {
        let device = Device::Cpu;
        let image = load_image(args.image)?.to_device(&device)?;
        println!("loaded image {image:?}");

        let vb = quantized_blip::VarBuilder::from_gguf(model_file, &device)?;
        let model = quantized_blip::BlipForConditionalGeneration::new(&config, vb)?;
        let image_embeds = image.unsqueeze(0)?.apply(model.vision_model())?;
        (image_embeds, device, Model::Q(model))
    } else {
        let image = load_image(args.image)?.to_device(&device)?;
        println!("loaded image {image:?}");

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
        let image_embeds = image.unsqueeze(0)?.apply(model.vision_model())?;
        (image_embeds, device, Model::M(model))
    };

    let mut token_ids = vec![30522u32];
    for index in 0..1000 {
        let context_size = if index > 0 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.text_decoder_forward(&input_ids, &image_embeds)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        if token == SEP_TOKEN_ID {
            break;
        }
        token_ids.push(token);
        if let Some(t) = tokenizer.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();
    Ok(())
}
