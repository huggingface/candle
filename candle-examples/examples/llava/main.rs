pub mod constants;
pub mod conversation;
pub mod image_processor;

use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::Cache;

use anyhow::{bail, Error as E, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llava::config::{
    HFGenerationConfig, HFLLaVAConfig, HFPreProcessorConfig,
};
use candle_transformers::models::llava::{config::LLaVAConfig, LLaVA};
use clap::Parser;
use constants::*;
use conversation::Conversation;
use hf_hub::api::sync::Api;
use image_processor::{process_image, ImageProcessor};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about,long_about=None)]
struct Args {
    #[arg(long, default_value = "llava-hf/llava-v1.6-vicuna-7b-hf")]
    model_path: String,
    #[arg(long, default_value = "tokenizer/tokenizer.json")]
    tokenizer_path: String,
    #[arg(long)]
    model_base: Option<String>,
    #[arg(long)]
    image_file: String, // Required
    #[arg(long)]
    conv_mode: Option<String>,
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, action)]
    hf: bool,
    #[arg(long, action)]
    cpu: bool,
    #[arg(long, action)]
    no_kv_cache: bool,
    #[arg(long)]
    prompt: String,
    /// The seed to use when generating random samples. Copy from candle llama. Not exist in python llava.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

//from https://github.com/huggingface/candle/blob/main/candle-examples/examples/clip/main.rs
fn load_image<T: AsRef<std::path::Path>>(
    path: T,
    processor: &ImageProcessor,
    llava_config: &LLaVAConfig,
    dtype: DType,
) -> Result<((u32, u32), Tensor)> {
    let img = image::ImageReader::open(path)?.decode()?;
    let img_tensor = process_image(&img, processor, llava_config)?;
    Ok(((img.width(), img.height()), img_tensor.to_dtype(dtype)?))
}

fn get_model_name_from_path(model_path: &str) -> String {
    let model_paths: Vec<String> = model_path
        .trim_matches('/')
        .split('/')
        .map(|s| s.to_string())
        .collect();
    if model_paths.last().unwrap().starts_with("checkpoint-") {
        format!(
            "{}_{}",
            model_paths[model_paths.len() - 2],
            model_paths.last().unwrap()
        )
    } else {
        model_paths.last().unwrap().to_string()
    }
}

fn duplicate_vec<T>(vec: &[T], n: usize) -> Vec<T>
where
    T: Clone,
{
    let mut res = Vec::new();
    for _ in 0..n {
        res.extend(vec.to_owned());
    }
    res
}

fn insert_separator<T>(x: Vec<Vec<T>>, sep: Vec<T>) -> Vec<Vec<T>>
where
    T: Clone,
{
    let sep = vec![sep];
    let sep = duplicate_vec(&sep, x.len());
    let mut res = x
        .iter()
        .zip(sep.iter())
        .flat_map(|(x, y)| vec![x.clone(), y.clone()])
        .collect::<Vec<Vec<T>>>();
    res.pop();
    res
}

fn tokenizer_image_token(
    prompt: &str,
    tokenizer: &Tokenizer,
    image_token_index: i64,
    llava_config: &LLaVAConfig,
) -> Result<Tensor> {
    let prompt_chunks = prompt
        .split("<image>")
        .map(|s| {
            tokenizer
                .encode(s, true)
                .unwrap()
                .get_ids()
                .to_vec()
                .iter()
                .map(|x| *x as i64)
                .collect()
        })
        .collect::<Vec<Vec<i64>>>();
    let mut input_ids = Vec::new();
    let mut offset = 0;
    if !prompt_chunks.is_empty()
        && !prompt_chunks[0].is_empty()
        && prompt_chunks[0][0] == llava_config.bos_token_id as i64
    {
        offset = 1;
        input_ids.push(prompt_chunks[0][0]);
    }

    for x in insert_separator(
        prompt_chunks,
        duplicate_vec(&[image_token_index], offset + 1),
    )
    .iter()
    {
        input_ids.extend(x[1..].to_vec())
    }
    let input_len = input_ids.len();
    Tensor::from_vec(input_ids, (1, input_len), &Device::Cpu).map_err(E::msg)
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    println!("Start loading model");
    let api = Api::new()?;
    let api = api.model(args.model_path.clone());
    let (llava_config, tokenizer, clip_vision_config, image_processor) = if args.hf {
        let config_filename = api.get("config.json")?;
        let hf_llava_config: HFLLaVAConfig =
            serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let generation_config_filename = api.get("generation_config.json")?;
        let generation_config: HFGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_filename)?)?;
        let preprocessor_config_filename = api.get("preprocessor_config.json")?;
        let preprocessor_config: HFPreProcessorConfig =
            serde_json::from_slice(&std::fs::read(preprocessor_config_filename)?)?;
        let llava_config =
            hf_llava_config.to_llava_config(&generation_config, &preprocessor_config);
        let tokenizer_filename = api.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let clip_vision_config = hf_llava_config.to_clip_vision_config();
        (
            llava_config,
            tokenizer,
            Some(clip_vision_config),
            ImageProcessor::from_hf_preprocessor_config(&preprocessor_config),
        )
    } else {
        let config_filename = api.get("config.json")?;
        let llava_config: LLaVAConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(&args.tokenizer_path)
            .map_err(|e| E::msg(format!("Error loading {}: {}", &args.tokenizer_path, e)))?;
        (
            llava_config.clone(),
            tokenizer,
            None,
            ImageProcessor::from_pretrained(&llava_config.mm_vision_tower.unwrap())?,
        )
    };

    let llama_config = llava_config.to_llama_config();
    let dtype: DType = match llava_config.torch_dtype.as_str() {
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        _ => bail!("unsupported dtype"),
    };

    let eos_token_id = llava_config.eos_token_id;

    println!("setting kv cache");
    let mut cache = Cache::new(!args.no_kv_cache, dtype, &llama_config, &device)?;

    println!("loading model weights");

    let weight_filenames =
        candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, dtype, &device)? };
    let llava: LLaVA = LLaVA::load(vb, &llava_config, clip_vision_config)?;

    println!("generating conv template");
    let image_token_se = format!(
        "{}{}{}",
        DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
    );
    let qs = if args.prompt.contains(IMAGE_PLACEHOLDER) {
        if llava_config.mm_use_im_start_end {
            args.prompt.replace(IMAGE_PLACEHOLDER, &image_token_se)
        } else {
            args.prompt.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        }
    } else if llava_config.mm_use_im_start_end {
        format!("{}\n{}", image_token_se, args.prompt)
    } else {
        format!("{}\n{}", DEFAULT_IMAGE_TOKEN, args.prompt)
    };

    let model_name = get_model_name_from_path(&args.model_path).to_lowercase();
    let conv_mode = if model_name.contains("llama-2") {
        "llava_llama_2"
    } else if model_name.contains("mistral") {
        "mistral_instruct"
    } else if model_name.contains("v1.6-34b") {
        "chatml_direct"
    } else if model_name.contains("v1") {
        "llava_v1"
    } else if model_name.contains("mpt") {
        "mpt"
    } else {
        "llava_v0"
    };
    if args.conv_mode.is_some() && args.conv_mode.as_deref() != Some(conv_mode) {
        println!(
            "Warning: the model is trained with {}, but you are using {}",
            conv_mode,
            args.conv_mode.as_deref().unwrap()
        );
    } else {
        args.conv_mode = Some(conv_mode.to_string());
    }

    let mut conv = match args.conv_mode {
        Some(conv_mode) => match conv_mode.as_str() {
            "chatml_direct" => Conversation::conv_chatml_direct(),
            "llava_v1" => Conversation::conv_llava_v1(),
            _ => todo!("not implement yet"),
        },
        None => bail!("conv_mode is required"),
    };
    conv.append_user_message(Some(&qs));
    conv.append_assistant_message(None);
    let prompt = conv.get_prompt();
    println!("loading image");
    let (image_size, image_tensor) =
        load_image(&args.image_file, &image_processor, &llava_config, dtype)
            .map_err(|e| E::msg(format!("Error loading {}: {}", &args.image_file, e)))?;
    let image_tensor = image_tensor.to_device(&device)?;

    let mut logits_processor = {
        let temperature = f64::from(args.temperature);
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    // get input tokens
    let tokens = tokenizer_image_token(
        &prompt,
        &tokenizer,
        llava_config.image_token_index as i64,
        &llava_config,
    )?;
    let mut input_embeds =
        llava.prepare_inputs_labels_for_multimodal(&tokens, &[image_tensor], &[image_size])?;
    //inference loop, based on https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    let mut index_pos = 0;
    for index in 0..args.max_new_tokens {
        let (_, input_embeds_len, _) = input_embeds.dims3()?;
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (input_embeds_len, 0)
        };
        let input = input_embeds.i((.., input_embeds_len.saturating_sub(context_size).., ..))?;
        let logits = llava.forward(&input, context_index, &mut cache)?; //[1,32000]
        let logits = logits.squeeze(0)?;
        let (_, input_len, _) = input.dims3()?;
        index_pos += input_len;
        let next_token = logits_processor.sample(&logits)?;
        let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &device)?;
        let next_embeds = llava.llama.embed(&next_token_tensor)?.unsqueeze(0)?;
        input_embeds = Tensor::cat(&[input_embeds, next_embeds], 1)?;
        if next_token == eos_token_id as u32 {
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    Ok(())
}
