#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use candle::quantized::QuantizedBackend;
use candle::{BackendDevice, BackendStorage, Module};
use candle_transformers::quantized_nn::Linear;
use clap::Parser;

use candle::{DType, Result, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use candle_transformers::models::quantized_blip;

use tokenizers::Tokenizer;

enum Model<QB: QuantizedBackend> {
    M(blip::BlipForConditionalGeneration<QB::Storage>),
    Q(quantized_blip::BlipForConditionalGeneration<QB>),
}

impl<QB: QuantizedBackend> Model<QB> {
    fn text_decoder_forward(
        &mut self,
        xs: &Tensor<QB::Storage>,
        img_xs: &Tensor<QB::Storage>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
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
pub fn load_image<P: AsRef<std::path::Path>, B: BackendStorage>(
    p: P,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], device)?.reshape((3, 1, 1))?;
    let std =
        Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], device)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.cpu {
        run::<candle::QCpuStorage>(args)?;
    } else if candle::utils::cuda_is_available() {
        run::<candle::QCudaStorage>(args)?;
    } else if candle::utils::metal_is_available() {
        run::<candle::QMetalStorage>(args)?;
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        run::<candle::QCpuStorage>(args)?;
    }
    Ok(())
}

fn run<QB: QuantizedBackend>(args: Args) -> anyhow::Result<()>
where
    Linear<QB>: Module<QB::Storage>,
{
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

    let device = QB::Device::new(0)?;
    let (image_embeds, device, mut model): (_, _, Model<QB>) = if args.quantized {
        let image: Tensor<QB::Storage> = load_image(args.image, &device)?;
        println!("loaded image {image:?}");

        let vb: quantized_blip::VarBuilder<QB> =
            quantized_blip::VarBuilder::from_gguf(model_file, &device)?;
        let model = quantized_blip::BlipForConditionalGeneration::new(&config, vb)?;
        let image_embeds = image.unsqueeze(0)?.apply(model.vision_model())?;
        (image_embeds, device, Model::Q(model))
    } else {
        let image = load_image(args.image, &device)?;
        println!("loaded image {image:?}");

        let vb: VarBuilder<QB::Storage> =
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
