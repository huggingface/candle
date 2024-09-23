use std::any;

use anyhow::Ok;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::chinese_clip::{
    text_model::{self, ChineseClipTextTransformer},
    vision_model,
};

use rayon::vec;
use tokenizers::{tokenizer, Tokenizer};

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
    let (vision_transformer, text_transformer) = load_transformer(args.model, &device)?;
    tracing::info!("Transformer loaded. ");

    let images = load_images(args.images, &device)?;
    tracing::info!("Image loaded. ");
    // let images_feature = vision_transformer.forward(&images)?;
    // tracing::info!("Images feature: {:?}", images_feature);

    let tokenizer = load_tokenizer()?;
    let (token, text_sequences) = tokenize_sequences(args.sequences, &tokenizer, &device)?;
    // let token = tokenizer
    //     .encode("hello", true)
    //     .map_err(anyhow::Error::msg)?;
    tracing::info!("token: {}", token.to_string());
    let text_feature =text_transformer.forward(&token.i(0)?, None, None)?;
    tracing::info!("Text feature: {:?}", text_feature);

    Ok(())
}

pub fn load_weights(model: Option<String>, device: &Device) -> anyhow::Result<VarBuilder> {
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
    Ok(VarBuilder::from_pth(model_file, DType::F32, device)?)
}

pub fn load_transformer(
    model: Option<String>,
    device: &Device,
) -> anyhow::Result<(
    vision_model::ChineseClipVisionTransformer,
    text_model::ChineseClipTextTransformer,
)> {
    let var = load_weights(model, device)?;
    let vision_transformer = vision_model::ChineseClipVisionTransformer::new(
        var.pp("vision_model"),
        &vision_model::ChineseClipVisionConfig::clip_vit_base_patch16(),
    )?;
    let text_transformer = text_model::ChineseClipTextTransformer::new(
        var.pp("text_model"),
        &text_model::ChineseClipTextConfig::clip_vit_base_patch16(),
    )?;
    Ok((vision_transformer, text_transformer))
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
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("[PAD]")
        .ok_or(anyhow::Error::msg("No pad token"))?;
    tracing::info!("pad_id: {}", pad_id);

    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "一场自行车比赛".to_string(),
            "一张包含两只猫的照片".to_string(),
            "一个拿着蜡烛的机器人".to_string(),
        ],
    };

    let mut tokens = vec![];

    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(anyhow::Error::msg)?;
        tracing::info!("encoding: {:?}", encoding);
        tokens.push(encoding.get_ids().to_vec());
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }

    let input_ids = Tensor::new(tokens, device)?;

    Ok((input_ids, vec_seq))
}

pub fn load_images(images: Option<Vec<String>>, device: &Device) -> anyhow::Result<Tensor> {
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
    Ok(images)
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
