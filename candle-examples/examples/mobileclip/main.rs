#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle::{DType, Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::mobileclip;

use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    S1,
    S2,
}

impl Which {
    fn model_name(&self) -> String {
        let name = match self {
            Self::S1 => "S1",
            Self::S2 => "S2",
        };
        format!("apple/MobileCLIP-{}-OpenCLIP", name)
    }

    fn config(&self) -> mobileclip::MobileClipConfig {
        match self {
            Self::S1 => mobileclip::MobileClipConfig::s1(),
            Self::S2 => mobileclip::MobileClipConfig::s2(),
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,

    #[arg(value_enum, long, default_value_t=Which::S1)]
    which: Which,
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];
    for path in paths {
        let tensor = candle_examples::imagenet::load_image_with_std_mean(
            path,
            image_size,
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
        )?;
        images.push(tensor);
    }
    let images = Tensor::stack(&images, 0)?;
    Ok(images)
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model_name = args.which.model_name();
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(model_name);
    let model_file = if args.use_pth {
        api.get("open_clip_pytorch_model.bin")?
    } else {
        api.get("open_clip_model.safetensors")?
    };
    let tokenizer = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let config = &args.which.config();
    let device = candle_examples::device(args.cpu)?;
    let vec_imgs = match args.images {
        Some(imgs) => imgs,
        None => vec![
            "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "candle-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };
    let images = load_images(&vec_imgs, config.image_size)?.to_device(&device)?;
    let vb = if args.use_pth {
        VarBuilder::from_pth(&model_file, DType::F32, &device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)? }
    };

    let model = mobileclip::MobileClipModel::new(vb, config)?;
    let (input_ids, vec_seq) = tokenize_sequences(args.sequences, &tokenizer, &device)?;
    let (_logits_per_text, logits_per_image) = model.forward(&images, &input_ids)?;
    let softmax_image = softmax(&logits_per_image, 1)?;
    let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;
    println!("softmax_image_vec: {:?}", softmax_image_vec);
    let probability_vec = softmax_image_vec
        .iter()
        .map(|v| v * 100.0)
        .collect::<Vec<f32>>();
    let probability_per_image = probability_vec.len() / vec_imgs.len();

    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        println!("\n\nResults for image: {}\n", img);

        for (i, p) in prob.iter().enumerate() {
            println!("Probability: {:.4}% Text: {}", p, vec_seq[i]);
        }
    }

    Ok(())
}

pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    // let pad_id = *tokenizer
    // .get_vocab(true)
    // .get("<|endoftext|>")
    // .ok_or(E::msg("No pad token"))?;

    // The model does not work well if the text is padded using the <|endoftext|> token, using 0
    // as the original OpenCLIP code.
    let pad_id = 0;

    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a candle".to_string(),
        ],
    };

    let mut tokens = vec![];
    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
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
