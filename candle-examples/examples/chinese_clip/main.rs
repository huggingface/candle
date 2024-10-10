#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Tensor};
use candle_nn as nn;
use candle_transformers::models::chinese_clip::{ChineseClipConfig, ChineseClipModel};
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let device = candle_examples::device(args.cpu)?;
    let var = load_weights(args.model, &device)?;
    let clip_model = ChineseClipModel::new(var, &ChineseClipConfig::clip_vit_base_patch16())?;
    tracing::info!("Transformer loaded. ");

    let (pixel_values, vec_imgs) = load_images(args.images, &device)?;
    tracing::info!("Images loaded. ");

    let tokenizer = load_tokenizer()?;
    let (input_ids, type_ids, attention_mask, text_sequences) =
        tokenize_sequences(args.sequences, &tokenizer, &device)?;

    tracing::info!("Computing ... ");
    let (_logits_per_text, logits_per_image) = clip_model.forward(
        &pixel_values,
        &input_ids,
        Some(&type_ids),
        Some(&attention_mask),
    )?;
    let softmax_image = nn::ops::softmax(&logits_per_image, 1)?;

    let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;

    let probability_vec = softmax_image_vec
        .iter()
        .map(|v| v * 100.0)
        .collect::<Vec<f32>>();

    let probability_per_image = probability_vec.len() / vec_imgs.len();

    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        tracing::info!("\n\nResults for image: {}\n", img);

        for (i, p) in prob.iter().enumerate() {
            tracing::info!("Probability: {:.4}% Text: {} ", p, text_sequences[i]);
        }
    }

    Ok(())
}

pub fn load_weights(model: Option<String>, device: &Device) -> anyhow::Result<nn::VarBuilder> {
    let model_file = match model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = hf_hub::Repo::with_revision(
                "OFA-Sys/chinese-clip-vit-base-patch16".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/3".to_string(),
            );
            let api = api.repo(repo);
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    Ok(unsafe { nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, device)? })
}

pub fn load_tokenizer() -> anyhow::Result<Tokenizer> {
    let tokenizer_file = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_hub::Repo::with_revision(
            "OFA-Sys/chinese-clip-vit-base-patch16".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/3".to_string(),
        );
        let api = api.repo(repo);
        api.get("tokenizer.json")?
    };

    Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)
}

pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Vec<String>)> {
    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "自行车比赛".to_string(),
            "两只猫咪".to_string(),
            "拿着蜡烛的机器人".to_string(),
        ],
    };

    let mut input_ids = vec![];
    let mut type_ids = vec![];
    let mut attention_mask = vec![];
    let mut max_len = 0;

    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(anyhow::Error::msg)?;
        input_ids.push(encoding.get_ids().to_vec());
        type_ids.push(encoding.get_type_ids().to_vec());
        attention_mask.push(encoding.get_attention_mask().to_vec());
        if encoding.get_ids().len() > max_len {
            max_len = encoding.get_ids().len();
        }
    }

    let pad_id = *tokenizer
        .get_vocab(true)
        .get("[PAD]")
        .ok_or(anyhow::Error::msg("No pad token"))?;

    let input_ids: Vec<Vec<u32>> = input_ids
        .iter_mut()
        .map(|item| {
            item.extend(vec![pad_id; max_len - item.len()]);
            item.to_vec()
        })
        .collect();

    let type_ids: Vec<Vec<u32>> = type_ids
        .iter_mut()
        .map(|item| {
            item.extend(vec![0; max_len - item.len()]);
            item.to_vec()
        })
        .collect();

    let attention_mask: Vec<Vec<u32>> = attention_mask
        .iter_mut()
        .map(|item| {
            item.extend(vec![0; max_len - item.len()]);
            item.to_vec()
        })
        .collect();

    let input_ids = Tensor::new(input_ids, device)?;
    let type_ids = Tensor::new(type_ids, device)?;
    let attention_mask = Tensor::new(attention_mask, device)?;

    Ok((input_ids, type_ids, attention_mask, vec_seq))
}

pub fn load_images(
    images: Option<Vec<String>>,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let vec_imgs = match images {
        Some(imgs) => imgs,
        None => vec![
            "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "candle-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };

    let mut images = vec![];

    for path in vec_imgs.iter() {
        let tensor = load_image(path, 224, device)?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?.to_device(device)?;
    Ok((images, vec_imgs))
}

fn load_image<T: AsRef<std::path::Path>>(
    path: T,
    image_size: usize,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8().into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], device)?.reshape((3, 1, 1))?;
    let std =
        Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], device)?.reshape((3, 1, 1))?;
    let img = (img.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?;

    Ok(img)
}
