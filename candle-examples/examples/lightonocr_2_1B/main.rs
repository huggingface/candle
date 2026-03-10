use candle::{Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::lightonocr_2_1b::model::Model;
use tokenizers::Tokenizer;
use candle::Result;

pub struct TextGeneration{
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,   
}

impl TextGeneration {
    fn new(
        model: Model,
        tokenizer: Tokenizer, 
        device: &Device,
    ) -> Self {
        Self { 
            model, device: 
            device.clone(), 
            tokenizer: TokenOutputStream::new(tokenizer) 
        }
    }
}


pub fn main(){
    
}