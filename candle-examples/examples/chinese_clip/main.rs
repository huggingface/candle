#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn as nn;
use candle_transformers::models::chinese_clip::{ChineseClipConfig, ChineseClipModel};

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
    tracing::info!("Image loaded. ");
    tracing::info!("pixel_values: {:?}", pixel_values.shape());

    let features = clip_model.get_image_features(&pixel_values)?;
    tracing::warn!("features: {}", features.to_string());

    let tokenizer = load_tokenizer()?;
    let (input_ids, _type_ids, text_sequences) =
        tokenize_sequences(args.sequences, &tokenizer, &device)?;
    tracing::info!("\n{}", input_ids.to_string());
    let features = clip_model.get_text_features(&input_ids, None, None)?;
    tracing::warn!("features: {}", features.to_string());

    let (_logits_per_text, logits_per_image) = clip_model.forward(&pixel_values, &input_ids)?;
    tracing::info!(
        "====> {:?}, {:?}",
        _logits_per_text.shape(),
        logits_per_image.shape()
    );
    let softmax_image = nn::ops::softmax(&logits_per_image, 1)?;
    let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;
    tracing::info!("softmax_image_vec: {:?}", softmax_image_vec);

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
                "36e679e".to_string(),
            );
            let api = api.repo(repo);
            api.get("pytorch_model.bin")?
        }
        Some(model) => model.into(),
    };
    Ok(nn::VarBuilder::from_pth(model_file, DType::F32, device)?)
}

pub fn load_tokenizer() -> anyhow::Result<Tokenizer> {
    // let model_file = match model {
    //     None => {
    //         let api = hf_hub::api::sync::Api::new()?;
    //         let repo = hf_hub::Repo::with_revision(
    //             "OFA-Sys/chinese-clip-vit-base-patch16".to_string(),
    //             hf_hub::RepoType::Model,
    //             "36e679e".to_string(),
    //         );
    //         let api = api.repo(repo);
    //         api.get("pytorch_model.bin")?
    //     }
    //     Some(model) => model.into(),
    // };
    let tokenizer_file = "/home/shawn/workspace/rum-backend-python/tmp/tokenizer.json";
    Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)
}

pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("[PAD]")
        .ok_or(anyhow::Error::msg("No pad token"))?;
    tracing::debug!("pad_id: {}", pad_id);

    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "自行车比赛".to_string(),
            "两只猫".to_string(),
            "拿着蜡烛的机器人".to_string(),
            // "a cycling race".to_string(),
            // "a photo of two cats".to_string(),
            // "a robot holding a candle".to_string(),
        ],
    };

    let mut tokens = vec![];

    // let mut type_ids = vec![];
    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(anyhow::Error::msg)?;
        tracing::info!("encoding: {:?}", encoding);
        tokens.push(encoding);
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
    



    // Pad the sequences to have the same length
    // for token_vec in tokens.iter_mut() {
    //     let len_diff = max_len - token_vec.len();
    //     if len_diff > 0 {
    //         token_vec.extend(vec![pad_id; len_diff]);
    //     }
    // }

    // let input_ids = Tensor::new(tokens, device)?;
    // let type_ids = Tensor::new(type_ids, device)?;

    // Ok((input_ids, type_ids, vec_seq))
    todo!()
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
        let tensor = load_image(path, 224)?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?.to_device(device)?;
    Ok((images, vec_imgs))
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();

    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    // .unsqueeze(0)?;
    Ok(img)
}
