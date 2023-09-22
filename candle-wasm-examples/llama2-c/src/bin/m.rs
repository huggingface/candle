use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_wasm_example_llama2::worker::{Model as M, ModelData};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: M,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
}

impl Model {
    fn process(&mut self, tokens: &[u32]) -> candle::Result<String> {
        const REPEAT_LAST_N: usize = 64;
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;
        let logits = self.inner.llama.forward(&input, tokens.len())?;
        let logits = logits.squeeze(0)?;
        let logits = if self.repeat_penalty == 1. || tokens.is_empty() {
            logits
        } else {
            let start_at = self.tokens.len().saturating_sub(REPEAT_LAST_N);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);
        let text = match self.inner.tokenizer.id_to_token(next_token) {
            Some(text) => text.replace('▁', " ").replace("<0x0A>", "\n"),
            None => "".to_string(),
        };
        Ok(text)
    }
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<u8>, tokenizer: Vec<u8>) -> Result<Model, JsError> {
        let model = M::load(ModelData {
            tokenizer,
            model: weights,
        });
        let logits_processor = LogitsProcessor::new(299792458, None, None);
        match model {
            Ok(inner) => Ok(Self {
                inner,
                logits_processor,
                tokens: vec![],
                repeat_penalty: 1.,
            }),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    #[wasm_bindgen]
    pub fn get_seq_len(&mut self) -> usize {
        self.inner.config.seq_len
    }

    #[wasm_bindgen]
    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        seed: u64,
    ) -> Result<String, JsError> {
        // First reset the cache.
        {
            let mut cache = self.inner.cache.kvs.lock().unwrap();
            for elem in cache.iter_mut() {
                *elem = None
            }
        }
        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.tokens.clear();
        let tokens = self
            .inner
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?
            .get_ids()
            .to_vec();
        let text = self
            .process(&tokens)
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let last_token = *self.tokens.last().unwrap();
        let text = self
            .process(&[last_token])
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }
}

fn main() {}
