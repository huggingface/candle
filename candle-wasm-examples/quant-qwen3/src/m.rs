use candle::quantized::gguf_file;
use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use js_sys::Date;
use std::io::Cursor;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

use crate::console_log;
use crate::profiler::ProfileGuard;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;

#[wasm_bindgen]
pub struct Model {
    model: QuantizedQwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        _config: Vec<u8>, // Not used for GGUF, but keep for compatibility
    ) -> Result<Model, JsError> {
        let _prof = ProfileGuard::new("total_load");
        console_error_panic_hook::set_once();

        let device = Device::Cpu;

        // Tokenizer loading
        {
            let _prof = ProfileGuard::new("load_tokenizer");
            console_log!("Loading tokenizer...");
            let tokenizer =
                Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;

            // Get EOS token
            let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
                Some(&token) => token,
                None => match tokenizer.get_vocab(true).get("<|im_end|>") {
                    Some(&token) => token,
                    None => {
                        console_log!("Warning: no EOS token found, using 0");
                        0
                    }
                },
            };

            let start = Date::now();
            console_log!(
                "Weights size: {} bytes ({:.2} MB)",
                weights.len(),
                weights.len() as f64 / 1_048_576.0
            );

            // Load GGUF quantized model with SIMD optimizations
            let model = {
                let _prof = ProfileGuard::new("parse_gguf");

                let mut cursor = Cursor::new(weights);
                let content = gguf_file::Content::read(&mut cursor)
                    .map_err(|e| JsError::new(&format!("Failed to read GGUF: {}", e)))?;

                console_log!("GGUF file parsed, loading model weights...");

                // Use the new integrated API with optimizations
                QuantizedQwen3::from_gguf(content, &mut cursor, &device)?
            };

            let load_time = (Date::now() - start) / 1000.0;
            console_log!("Quantized model loaded in {:.2}s", load_time);

            let logits_processor = LogitsProcessor::new(299792458, None, None);

            Ok(Self {
                model,
                tokenizer,
                tokens: vec![],
                logits_processor,
                repeat_penalty: 1.,
                repeat_last_n: 64,
                eos_token,
            })
        }
    }

    #[wasm_bindgen]
    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: f64,
    ) -> Result<String, JsError> {
        let _prof = ProfileGuard::new("init_with_prompt");

        // Clear KV cache
        {
            let _prof = ProfileGuard::new("clear_kv_cache");
            self.model.clear_kv_cache();
        }

        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };

        let seed = seed as u64;
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();

        let tokens = {
            let _prof = ProfileGuard::new("tokenize_prompt");
            self.tokenizer
                .encode(prompt, true)
                .map_err(|m| JsError::new(&m.to_string()))?
                .get_ids()
                .to_vec()
        };

        console_log!("Prompt encoded to {} tokens", tokens.len());

        let text = self
            .process(&tokens)
            .map_err(|m| JsError::new(&m.to_string()))?;

        Ok(text)
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let _prof = ProfileGuard::new("next_token");

        let last_token = *self.tokens.last().unwrap();
        let text = self
            .process(&[last_token])
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }

    #[wasm_bindgen]
    pub fn is_eos(&self) -> bool {
        self.tokens.last().map_or(false, |&t| t == self.eos_token)
    }

    #[wasm_bindgen]
    pub fn get_token_count(&self) -> usize {
        self.tokens.len()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        let _prof = ProfileGuard::new("reset_model");
        self.tokens.clear();
        self.model.clear_kv_cache();
    }

    #[wasm_bindgen]
    pub fn generate_tokens(&mut self, count: usize) -> Result<String, JsError> {
        let _prof = ProfileGuard::new("generate_tokens_batch");

        let mut result = String::new();

        for _ in 0..count {
            if self.is_eos() {
                break;
            }

            let last_token = *self.tokens.last().unwrap();
            let text = self
                .process(&[last_token])
                .map_err(|m| JsError::new(&m.to_string()))?;
            result.push_str(&text);
        }

        Ok(result)
    }
}

impl Model {
    fn process(&mut self, tokens: &[u32]) -> candle::Result<String> {
        let _prof = ProfileGuard::new("process_token");

        let dev = Device::Cpu;

        let input = {
            let _prof = ProfileGuard::new("create_input_tensor");
            Tensor::new(tokens, &dev)?.unsqueeze(0)?
        };

        // Calculate offset (position in sequence)
        let offset = self.tokens.len();

        // Forward pass - this is where most time is spent
        let logits = {
            let _prof = ProfileGuard::new("model_forward");
            self.model.forward(&input, offset)?
        };

        let logits = {
            let _prof = ProfileGuard::new("logits_post_process");
            logits.squeeze(0)?.to_dtype(DType::F32)?
        };

        // Apply repeat penalty if enabled
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let _prof = ProfileGuard::new("apply_repeat_penalty");
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            let context = &self.tokens[start_at..];
            candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, context)?
        };

        let next_token = {
            let _prof = ProfileGuard::new("sample_token");
            self.logits_processor.sample(&logits)?
        };

        self.tokens.push(next_token);

        let token = {
            let _prof = ProfileGuard::new("decode_token");
            match self.tokenizer.decode(&[next_token], false) {
                Ok(token) => token,
                Err(e) => {
                    console_log!("Error decoding token: {:?}", e);
                    "".to_string()
                }
            }
        };

        Ok(token)
    }
}
