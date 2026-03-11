use std::path::Path;

use candle::{DType, Device, Tensor, bail};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::lightonocr_2_1b::{self, model};
use clap::Parser;
use serde_json::from_str;
use tokenizers::Tokenizer;
use candle::Result;

pub struct TextGeneration{
    model: model::Model,
    device: Device,
    tokenizer: TokenOutputStream, 
}

impl TextGeneration {
    fn new(
        model: model::Model,
        tokenizer: Tokenizer, 
        device: &Device,
    ) -> Self {
        Self { 
            model, device: 
            device.clone(), 
            tokenizer: TokenOutputStream::new(tokenizer) ,
        }
    }
    pub fn load_and_resize_image(
        &self,
        image: String, 
        max_edge: u32, 
        patch_size: u32,
        dtype: DType,
        device: &Device
    )-> Result<Tensor>{
        let mut img = match image::open(image){
            Ok(i) => {
                i
            }
            Err(_) => {
                candle::bail!("Unable to open the provided image.")
            }
        };
        let (w, h) = (img.width(), img.height());
        let longest = w.max(h);

        if longest > max_edge{
            let scale = max_edge as f32 / longest as f32;
            let new_w = ((w as f32 * scale) / patch_size as f32).round() as u32 * patch_size;
            let new_h = ((h as f32 * scale) / patch_size as f32).round() as u32 * patch_size;

            img = img.resize_exact(
            new_w, 
            new_h, 
            image::imageops::FilterType::Lanczos3);
        }
        
        let img = img.to_rgb8();
        let data: Vec<f32> = img.pixels()
            .flat_map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
            .collect();
        let tensor = Tensor::from_vec(data, (h as usize, w as usize, 3), device)?
            .permute((2, 0, 1))?  // HWC → CHW
            .to_dtype(dtype)?;

        Ok(tensor)
    }

    fn run(&mut self, 
        image: String, 
        dtype: DType,
        ) -> Result<()>{
        self.model.clear_kv_cache();
        let preprocessor = &self.model.preprocessor;
        let patch_size = preprocessor.patch_size;
        let max_edge = preprocessor.patch_size;
        let device = &self.device;

        let image_tensor = self.load_and_resize_image(image, max_edge, patch_size, dtype, &device)?;

        let preprocessed = self.model.preprocessor.preprocess(&image_tensor)?;
        let (_, h, w) = preprocessed.dims3()?;
        let num_image_tokens = (h / patch_size as usize) * (w / patch_size as usize);

        let input_tensor = self.encode_image_tokens(num_image_tokens)?;

        Ok(())
    }

    fn encode_image_tokens(&mut self, num_image_tokens: usize) -> Result<Tensor>{
        let encode = |s: &str| -> Result<Vec<u32>> {
            let encoded= match self.tokenizer.tokenizer()
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("{}", e)){
                    Ok(e) => e,
                    Err(_) => candle::bail!("Unable to encode prompt")
                }.get_ids()
                .to_vec();

            Ok(encoded)
        };

        let user_tokens      = encode("user\n")?;
        let newline_tokens   = encode("\n")?;

        let image_pad = self.model.model_config.pad_token_id;
        let image_start = 151644u32;
        let image_end = 151645u32;
        let image_tokens: Vec<u32> = vec![image_pad as u32; num_image_tokens];

        let mut input_ids: Vec<u32> = Vec::new();
        
        /* 
        let chat_template = match ChatTemplate::from_tokenizer_config(template){
            Ok(c) => c,
            Err(e) => candle::bail!("Failed to load tokenizer config")
        };*/
        input_ids.push(image_start);
        input_ids.extend_from_slice(&user_tokens);
        input_ids.extend_from_slice(&image_tokens);
        input_ids.push(image_end);
        input_ids.extend_from_slice(&newline_tokens);

        let seq_len = input_ids.len();
        let device = &self.device;
        let input_tensor = Tensor::from_vec(
            input_ids,
            (1, seq_len),
            device,
        )?;

        let mut generated: Vec<u32> = Vec::new();
        let mut offset = seq_len;
        let count = 0;
        loop {
            let last = *generated.last().unwrap();

            if last == image_end{
                break;
            }

            // Upper limit on number of tokens generated to avoid infinite generation loop
            if count > 1024 {
                break;
            }

            let input = Tensor::from_vec(
                vec![last], (1, 1), device)?;

            let logits = self.model.language_model
            .forward(&input, offset)?;
            let token = self.greedy(logits)?;
            generated.push(token);
            offset += 1;

        }

        let decode_ids: Vec<u32> = generated.iter()
            .copied()
            .filter(|&t| t != image_end)
            .collect();

        let output = match self.tokenizer.tokenizer()
            .decode(&decode_ids, true)
            .map_err(|e| anyhow::anyhow!("{}", e)){
                Ok(out) => out,
                Err(_) => candle::bail!("Failed to decode tokens")
            };

        println!("{}", output);

        Ok(input_tensor)
    }
    
    fn greedy(&self, logits: Tensor) -> Result<u32>{
        let logits = logits.unsqueeze(0)?;
        let seq = logits.dim(0)?;
        let last = logits
        .narrow(0,seq -1, 1)?
        .squeeze(0)?
        .to_dtype(DType::F32)?;

        let logits_vec = last.to_vec1::<f32>()?;

        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (idx, value) in logits_vec.iter().enumerate() {
            if *value > max_val {
                max_val = *value;
                max_idx = idx;
            }
        }

        Ok(max_idx as u32)
    }
}


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {

    #[arg(long)]
    cpu: bool,

    // Location of the config file for the model
    #[arg(long)]
    model_config: Option<String>,

    // Location of the config file for the preprocesor
    #[arg(long)]
    processor_config: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    //Location of the weights for the model
    #[arg(long)]
    model_weights: Option<String>,

    //Location of the Tokenizer config
    #[arg(long)]
    tokenizer_config: Option<String>,

    //Location of the image to transcribe
    #[arg(long)]
    image_location: String,
}

pub fn main() -> Result<()>{
    let args = Args::parse(); 
    let device = candle_examples::device(args.cpu)?;

    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F32,
    };  
    // TODO all of this can be moved into the new function 
    // also  if config not found need to use defaults.
    //build preprocessor:
    let preprocessor_cfg = load_config(
        args.processor_config.as_deref(), 
        "Preprocessor config");
    let processor_cfg: lightonocr_2_1b::preprocessor::Config = from_str(&preprocessor_cfg)
    .expect("Failed ");
    let prepreprocessor = processor_cfg.image_processor;

    //Load model weights:
    let weights_path = args.model_weights.as_deref()
    .expect("Empty model weights field");

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
    }?;

    let config_str = load_config(
        args.model_config.as_deref(), 
        "Model Config");

    let cfg: model::Config = serde_json::from_str(&config_str)
    .expect("Failed to deserialize config");
    let model = model::Model::new(cfg, prepreprocessor, vb)?;

    let tokenizer_path = args.tokenizer_config.as_deref().expect("No tokenizer config found");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))
        .expect("Failed to create tokenizer");
    
    let mut text_generation = TextGeneration::new(model, tokenizer, &device);

    text_generation.run(args.image_location, dtype)
    
}

pub fn load_config(file: Option<&str>, which_config: &str) -> String {
 
    let path = Path::new(
        file.expect(
            format!("Please provide a file for {which_config}")
            .as_str()
        )
    );

    let config_str = std::fs::read_to_string(path)
    .expect(
        format!("Failed to read path for {which_config}"
        ).as_str()
    );

    config_str
}   
