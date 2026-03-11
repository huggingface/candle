use candle::{DType, Device, Tensor};
use candle_examples::{chat_template::{self, ChatTemplate, Conversation}, device, token_output_stream::TokenOutputStream};
use candle_transformers::models::lightonocr_2_1b::{self, model::Model};
use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;
use candle::Result;

pub struct TextGeneration{
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream, 
    config: lightonocr_2_1b::model::Config  
}

impl TextGeneration {
    fn new(
        model: Model,
        tokenizer: Tokenizer, 
        device: &Device,
        config: lightonocr_2_1b::model::Config,
    ) -> Self {
        Self { 
            model, device: 
            device.clone(), 
            tokenizer: TokenOutputStream::new(tokenizer) ,
            config: config
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
        tokenizer_config: String,
        template: String,
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

        let logits = self.model.forward(
            &input_tensor, 
            &preprocessed, 0
        )?;



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

    //Location of the weights for the model
    #[arg(long)]
    model_weights: Option<String>
}

pub fn main(){

}