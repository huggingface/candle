use candle::{BackendDevice, BackendStorage, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segformer::{
    Config, ImageClassificationModel, SemanticSegmentationModel,
};
use clap::{Args, Parser, Subcommand};
use imageproc::image::Rgb;
use imageproc::integral_image::ArrayData;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(about, version, long_about = None)]
struct CliArgs {
    #[arg(long, help = "use cpu")]
    cpu: bool,
    #[command(subcommand)]
    command: Commands,
}
#[derive(Args, Debug)]
struct SegmentationArgs {
    #[arg(
        long,
        help = "name of the huggingface hub model",
        default_value = "nvidia/segformer-b0-finetuned-ade-512-512"
    )]
    model_name: String,
    #[arg(
        long,
        help = "path to the label file in json format",
        default_value = "candle-examples/examples/segformer/assets/labels.json"
    )]
    label_path: PathBuf,
    #[arg(long, help = "path to for the output mask image")]
    output_path: PathBuf,
    #[arg(help = "path to image as input")]
    image: PathBuf,
}

#[derive(Args, Debug)]
struct ClassificationArgs {
    #[arg(
        long,
        help = "name of the huggingface hub model",
        default_value = "paolinox/segformer-finetuned-food101"
    )]
    model_name: String,
    #[arg(help = "path to image as input")]
    image: PathBuf,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Segment(SegmentationArgs),
    Classify(ClassificationArgs),
}

fn get_vb_and_config<'a, B: BackendStorage + 'a>(
    model_name: String,
    device: &B::Device,
) -> anyhow::Result<(VarBuilder<'a, B>, Config)> {
    println!("loading model {model_name} via huggingface hub");
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(model_name.clone());
    let model_file = api.get("model.safetensors")?;
    println!("model {model_name} downloaded and loaded");
    let vb: VarBuilder<'a, B> =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], candle::DType::F32, device)? };
    let config = std::fs::read_to_string(api.get("config.json")?)?;
    let config: Config = serde_json::from_str(&config)?;
    println!("{config:?}");
    Ok((vb, config))
}

#[derive(Debug, serde::Deserialize)]
struct LabelItem {
    index: u32,
    color: String,
}

fn segmentation_task<B: BackendStorage>(
    args: SegmentationArgs,
    device: &B::Device,
) -> anyhow::Result<()> {
    let label_file = std::fs::read_to_string(&args.label_path)?;
    let label_items: Vec<LabelItem> = serde_json::from_str(&label_file)?;
    let label_colors: HashMap<u32, Rgb<u8>> = label_items
        .iter()
        .map(|x| {
            (x.index - 1, {
                let color = x.color.trim_start_matches('#');
                let r = u8::from_str_radix(&color[0..2], 16).unwrap();
                let g = u8::from_str_radix(&color[2..4], 16).unwrap();
                let b = u8::from_str_radix(&color[4..6], 16).unwrap();
                Rgb([r, g, b])
            })
        })
        .collect();

    let image: Tensor<B> =
        candle_examples::imagenet::load_image224(args.image, device)?.unsqueeze(0)?;
    let (vb, config) = get_vb_and_config(args.model_name, device)?;
    let num_labels = label_items.len();

    let model = SemanticSegmentationModel::new(&config, num_labels, vb)?;
    let segmentations = model.forward(&image)?;

    // generate a mask image
    let mask = &segmentations.squeeze(0)?.argmax(0)?;
    let (h, w) = mask.dims2()?;
    let mask = mask.flatten_all()?.to_vec1::<u32>()?;
    let mask = mask
        .iter()
        .flat_map(|x| label_colors[x].data())
        .collect::<Vec<u8>>();
    let mask: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, mask).unwrap();
    // resize
    let mask = image::DynamicImage::from(mask);
    let mask = mask.resize_to_fill(
        w as u32 * 4,
        h as u32 * 4,
        image::imageops::FilterType::CatmullRom,
    );
    mask.save(args.output_path.clone())?;
    println!("mask image saved to {:?}", args.output_path);
    Ok(())
}

fn classification_task<B: BackendStorage>(
    args: ClassificationArgs,
    device: &B::Device,
) -> anyhow::Result<()> {
    let image: Tensor<B> =
        candle_examples::imagenet::load_image224(args.image, device)?.unsqueeze(0)?;
    let (vb, config) = get_vb_and_config(args.model_name, device)?;
    let num_labels = 7;
    let model = ImageClassificationModel::new(&config, num_labels, vb)?;
    let classification = model.forward(&image)?;
    let classification = candle_nn::ops::softmax_last_dim(&classification)?;
    let classification = classification.squeeze(0)?;
    println!(
        "classification logits {:?}",
        classification.to_vec1::<f32>()?
    );
    let label_id = classification.argmax(0)?.to_scalar::<u32>()?;
    let label_id = format!("{label_id}");
    println!("label: {}", config.id2label[&label_id]);
    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    if args.cpu {
        run::<candle::CpuStorage>(args)?;
    } else {
        #[cfg(feature = "cuda")]
        run::<candle::CudaStorage>(args)?;

        #[cfg(feature = "metal")]
        run::<candle::MetalStorage>(args)?;
    }
    Ok(())
}

fn run<B: BackendStorage>(args: CliArgs) -> anyhow::Result<()> {
    let device = B::Device::new(0)?;
    if let Commands::Segment(args) = args.command {
        segmentation_task::<B>(args, &device)?
    } else if let Commands::Classify(args) = args.command {
        classification_task::<B>(args, &device)?
    }
    Ok(())
}
