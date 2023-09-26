use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use candle_nn::VarBuilder;
use candle_wasm_example_phi::console_log;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use js_sys::Date;

enum SelectedModel {
    MixFormer(MixFormer),
    Quantized(QMixFormer),
}

#[wasm_bindgen]
pub struct Model {
    model: SelectedModel,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        quantized: bool,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let config: Config = Config::v1_5();
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let start = Date::now();
        let model = if quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(&weights)?;
            let model = QMixFormer::new(&config, vb)?;
            SelectedModel::Quantized(model)
        } else {
            let device = &Device::Cpu;
            let vb =
                 VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
            let model = MixFormer::new(&config, vb)?;
            SelectedModel::MixFormer(model)
        };
        console_log!("model loaded in {:?}s", (Date::now() - start)/1000.);
        let logits_processor = LogitsProcessor::new(299792458, None, None);
        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
        })
    }
    #[wasm_bindgen]
    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> Result<String, JsError> {
        // // First reset the cache.
        // {
        //     let mut cache = self.model.cache.kvs.lock().unwrap();
        //     for elem in cache.iter_mut() {
        //         *elem = None
        //     }
        // }
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
        let tokens = self
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

impl Model {
    fn process(&mut self, tokens: &[u32]) -> candle::Result<String> {
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;
        let logits = match &mut self.model {
            SelectedModel::MixFormer(m) => m.forward(&input)?,
            SelectedModel::Quantized(m) => m.forward(&input)?,
        };        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
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
        let token = match self.tokenizer.decode(&[next_token], false){
            Ok(token) => token,
            Err(e) => {
                console_log!("error decoding token: {:?}", e);
                "".to_string()
            }
        };
        Ok(token)
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
