use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{moondream, quantized_moondream},
};
use candle_wasm_example_moondream::console_log;
use js_sys::Date;
use serde::Deserialize;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

enum SelectedModel {
    Moondream(moondream::Model),
    Quantized(quantized_moondream::Model),
}

#[wasm_bindgen]
pub struct Model {
    model: SelectedModel,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    bos_token: u32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    image_embeddings: Option<Tensor>,
    index: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]

pub struct ModelName {
    pub _name_or_path: String,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(weights: Vec<u8>, tokenizer: Vec<u8>, quantized: bool) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let device = Device::Cpu;
        let config = moondream::Config::v2();

        console_log!("config loaded in {:?}", Date::now());
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let start = Date::now();
        console_log!("weights len: {:?}", weights.len());
        let model = if quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
                &weights, &device,
            )?;
            console_log!("weights loaded");
            let model = quantized_moondream::Model::new(&config, vb)?;
            SelectedModel::Quantized(model)
        } else {
            let device = &Device::Cpu;
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
            let model = moondream::Model::new(&config, vb)?;
            SelectedModel::Moondream(model)
        };
        console_log!("model loaded in {:?}s", (Date::now() - start) / 1000.);
        let logits_processor = LogitsProcessor::new(299792458, None, None);
        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            bos_token: 0,
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            image_embeddings: None,
            index: 0,
        })
    }

    pub fn set_image_embeddings(&mut self, image: Vec<u8>) -> Result<(), JsError> {
        let device = Device::Cpu;

        console_log!("loading image as tensor");
        let start = Date::now();
        let image: Tensor = self.load_image(image)?.to_device(&device)?;
        console_log!("image loaded in {:?}s", (Date::now() - start) / 1000.);
        let start = Date::now();
        let image_embeds = image.unsqueeze(0)?;
        let image_embeds = match self.model {
            SelectedModel::Moondream(ref m) => image_embeds.apply(m.vision_encoder())?,
            SelectedModel::Quantized(ref m) => image_embeds.apply(m.vision_encoder())?,
        };
        console_log!(
            "loaded and encoded the image {image:?} in {:?}",
            (Date::now() - start) / 1000.
        );
        self.image_embeddings = Some(image_embeds);
        Ok(())
    }

    #[wasm_bindgen]
    pub fn init_with_image_prompt(
        &mut self,
        prompt: String,
        seed: u64,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
    ) -> Result<String, JsError> {
        if self.image_embeddings.is_none() {
            return Err(JsError::new("Image embeddings are not set."));
        }

        let prompt = format!("\n\nQuestion: {0}\n\nAnswer:", prompt);
        console_log!("Processing prompt: {:?}", prompt);

        match &mut self.model {
            SelectedModel::Moondream(m) => m.text_model.clear_kv_cache(),
            SelectedModel::Quantized(m) => m.text_model.clear_kv_cache(),
        };

        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();
        self.index = 0;

        // Moondream tokenizer bos_token is "<|endoftext|>"
        // https://huggingface.co/vikhyatk/moondream2/blob/main/special_tokens_map.json
        self.bos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => return Err(JsError::new("BOS token not found in the tokenizer.")),
        };

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?;

        if tokens.is_empty() {
            return Err(JsError::new(
                "Empty prompts are not supported in the Moondream model.",
            ));
        }

        if verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let tokens = tokens.get_ids().to_vec();
        let text = match self.process(&tokens) {
            Ok(token) => token,
            Err(_e) => {
                console_log!("error decoding token");
                "".to_string()
            }
        };
        Ok(text)
    }
    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let last_token = *self.tokens.last().unwrap();
        let text = match self.process(&[last_token]) {
            Ok(token) => token,
            Err(_e) => {
                console_log!("error decoding last_token");
                "".to_string()
            }
        };
        Ok(text)
    }
}
impl Model {
    fn load_image(&self, image: Vec<u8>) -> Result<Tensor, JsError> {
        let img = image::io::Reader::new(std::io::Cursor::new(image))
            .with_guessed_format()?
            .decode()
            .map_err(|e| JsError::new(&e.to_string()))?
            .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (378, 378, 3), &Device::Cpu)?.permute((2, 0, 1))?;
        let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?.reshape((3, 1, 1))?;
        (data.to_dtype(candle::DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

impl Model {
    fn process(&mut self, tokens: &[u32]) -> Result<String, JsError> {
        let image_embeddings = match &self.image_embeddings {
            Some(embeddings) => embeddings,
            None => return Err(JsError::new("Image embeddings are not set.")),
        };

        let device = Device::Cpu;
        let context_size = if self.index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = if self.index > 0 {
            match self.model {
                SelectedModel::Moondream(ref mut model) => model.text_model.forward(&input)?,
                SelectedModel::Quantized(ref mut model) => model.text_model.forward(&input)?,
            }
        } else {
            let bos_token = Tensor::new(&[self.bos_token], &device)?.unsqueeze(0)?;
            match self.model {
                SelectedModel::Moondream(ref mut model) => {
                    model
                        .text_model
                        .forward_with_img(&bos_token, &input, image_embeddings)?
                }
                SelectedModel::Quantized(ref mut model) => {
                    model
                        .text_model
                        .forward_with_img(&bos_token, &input, image_embeddings)?
                }
            }
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);
        let token = match self.tokenizer.decode(&[next_token], true) {
            Ok(token) => token,
            Err(e) => {
                console_log!("error decoding token: {:?}", e);
                "".to_string()
            }
        };
        self.index += 1;
        console_log!("token: {:?}: {:?}", token, next_token);
        Ok(token)
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
