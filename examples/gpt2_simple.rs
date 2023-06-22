use anyhow::Result;
use candle::nn::models::gpt2::Gpt2;
use candle::{Device, Tensor};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = Device::Cpu;

    //wget https://huggingface.co/gpt2/raw/main/tokenizer.json
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let encoded = tokenizer.encode("What is deep learning ?", true).unwrap();

    let ids = encoded.get_ids();
    assert_eq!(ids, &[2061, 318, 2769, 4673, 5633]);
    let input_ids = Tensor::from_slice(ids, (1, ids.len()), &device)?;

    let position_ids: Vec<_> = (0..ids.len() as u32).collect();
    let position_ids = Tensor::from_slice(&position_ids, (1, position_ids.len()), &device)?;

    //wget https://huggingface.co/gpt2/raw/main/model.safetensors
    let gpt2 = Gpt2::load("model.safetensors")?;
    let logits = gpt2.forward(&input_ids, &position_ids)?;
    // let id = logits.argmax(-1)?;
    todo!("Argmax");
    // let token = tokenizer.decode(&[id], true).unwrap();
    // assert_eq!(token, "");
    Ok(())
}
