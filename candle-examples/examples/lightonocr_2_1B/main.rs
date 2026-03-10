use candle::{DType, Device, Tensor};
use candle_examples::{chat_template::{self, ChatTemplate}, device, token_output_stream::TokenOutputStream};
use candle_transformers::models::lightonocr_2_1b::{self, model::Model};
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
        tokenizer_config: String
        ) -> Result<()>{
        self.model.clear_kv_cache();
        let preprocessor = &self.model.preprocessor;
        let patch_size = preprocessor.patch_size;
        let max_edge = preprocessor.patch_size;
        let device = &self.device;

        let image_tensor = self.load_and_resize_image(image, max_edge, patch_size, dtype, &device)?;

        let preprocessed = self.model.preprocessor.preprocess(&image_tensor);

        let chat_template = match ChatTemplate::from_tokenizer_config(tokenizer_config){
            Ok(tmpl) => tmpl,
            Err(_) => candle::bail!("Unable to load tokenizer config"),
        };

        


        Ok(())
    }

    
}


pub fn main(){

}