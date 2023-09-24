use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
pub use candle_transformers::models::t5::{Config, T5EncoderModel, T5ForConditionalGeneration};
use candle_wasm_example_t5::console_log;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub struct ModelEncoder {
    model: T5EncoderModel,
    tokenizer: Tokenizer,
}
#[wasm_bindgen]

pub struct ModelConditionalGeneration {
    model: T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    config: Config,
}

#[wasm_bindgen]
impl ModelConditionalGeneration {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
    ) -> Result<ModelConditionalGeneration, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let device = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
        let mut config: Config = serde_json::from_slice(&config)?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let model = T5ForConditionalGeneration::load(vb, &config)?;
        config.use_cache = false;
        Ok(Self {
            model,
            tokenizer,
            config,
        })
    }
    pub fn decode(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let input: ConditionalGenerationParams =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        let device = &Device::Cpu;
        self.model.clear_kv_cache();
        let mut output_token_ids = [self.config.pad_token_id as u32].to_vec();
        let prompt = input.prompt;
        let repeat_penalty = input.repeat_penalty;
        let repeat_last_n = input.repeat_last_n;
        let seed = input.seed;
        let max_length = usize::clamp(input.max_length.unwrap_or(512), 0, 512);
        let temperature = if input.temperature <= 0. {
            None
        } else {
            Some(input.temperature)
        };
        let top_p = if input.top_p <= 0. || input.top_p >= 1. {
            None
        } else {
            Some(input.top_p)
        };
        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?
            .get_ids()
            .to_vec();

        let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let encoder_output = self.model.encode(&input_token_ids)?;
        let mut decoded = String::new();
        for index in 0.. {
            if output_token_ids.len() > max_length {
                break;
            }
            let decoder_token_ids = if index == 0 {
                Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
            } else {
                let last_token = *output_token_ids.last().unwrap();
                Tensor::new(&[last_token], device)?.unsqueeze(0)?
            };
            let logits = self
                .model
                .decode(&decoder_token_ids, &encoder_output)?
                .squeeze(0)?;
            let logits = if repeat_penalty == 1. {
                logits
            } else {
                let start_at = output_token_ids.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &output_token_ids[start_at..],
                )?
            };

            let next_token_id = logits_processor.sample(&logits)?;
            if next_token_id as usize == self.config.eos_token_id {
                break;
            }
            output_token_ids.push(next_token_id);
            if let Some(text) = self.tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                decoded += &text;
            }
        }
        Ok(serde_wasm_bindgen::to_value(
            &ConditionalGenerationOutput {
                generation: decoded,
            },
        )?)
    }
}

#[wasm_bindgen]
impl ModelEncoder {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
    ) -> Result<ModelEncoder, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let device = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
        let mut config: Config = serde_json::from_slice(&config)?;
        config.use_cache = false;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let model = T5EncoderModel::load(vb, &config)?;
        Ok(Self { model, tokenizer })
    }

    pub fn decode(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let device = &Device::Cpu;
        let input: DecoderParams =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;

        self.model.clear_kv_cache();
        let sentences = input.sentences;
        let normalize_embeddings = input.normalize_embeddings;
        let n_sentences = sentences.len();
        let mut all_embeddings = Vec::with_capacity(n_sentences);
        for sentence in sentences {
            let tokens = self
                .tokenizer
                .encode(sentence, true)
                .map_err(|m| JsError::new(&m.to_string()))?
                .get_ids()
                .to_vec();
            let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
            let embeddings = self.model.forward(&token_ids)?;
            console_log!("generated embeddings {:?}", embeddings.shape());
            // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
            let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
            let embeddings = if normalize_embeddings {
                embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?
            } else {
                embeddings
            };
            console_log!("{:?}", embeddings.shape());
            all_embeddings.push(embeddings.squeeze(0)?.to_vec1::<f32>()?);
        }

        Ok(serde_wasm_bindgen::to_value(&DecoderOutput {
            embeddings: all_embeddings,
        })?)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ConditionalGenerationOutput {
    generation: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DecoderOutput {
    embeddings: Vec<Vec<f32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DecoderParams {
    sentences: Vec<String>,
    normalize_embeddings: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConditionalGenerationParams {
    prompt: String,
    temperature: f64,
    seed: u64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    max_length: Option<usize>,
}
fn main() {
    console_error_panic_hook::set_once();
}
