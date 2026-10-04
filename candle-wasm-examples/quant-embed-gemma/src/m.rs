// Quantized embedding Gemma inference implementation,
// modified from 'https://github.com/huggingface/text-embeddings-inference/blob/main/backends/candle/src/models/gemma3.rs'
// added quant support, wasm support to this base implementation. Also Embedgemma has 2 FP dense layers after the base quant model, hence 
// added these as well.

use candle::quantized::gguf_file;
use candle::{Device, DType, Tensor};
use candle_transformers::models::quantized_gemma3::ModelWeights as BaseGemma3Model;
use candle_nn::{linear_b as linear, Activation, Linear, Module, VarBuilder};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma3Config {
    pub hidden_size: usize,
}

// reference - text-embeddings-inference/blob/main/backends/core/src/lib.rs
#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub max_length: u32,
    pub pooled_indices: Vec<u32>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.cumulative_seq_lengths.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Gemma3Model {
    base_model: BaseGemma3Model,
    dense1: Linear,
    dense2: Linear,
}

impl Gemma3Model {
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        vb_dense1: VarBuilder,
        vb_dense2: VarBuilder,
        config: &Gemma3Config,
    ) -> candle::Result<Self> {

        let device = Device::Cpu;

        console_log!("Loading base model...");

        let base_model = BaseGemma3Model::from_gguf(ct, reader, &device)?;

        console_log!("Loading dense1...");

        let dense1 = 
            candle_nn::linear_no_bias(config.hidden_size, 4 * config.hidden_size, vb_dense1.pp("linear"))?;

        console_log!("Loading dense2...");

        let dense2 = 
            candle_nn::linear_no_bias(4 * config.hidden_size, config.hidden_size, vb_dense2.pp("linear"))?;

        Ok(Self {
            base_model,
            dense1,
            dense2,
        })
    }

    pub fn forward(&mut self, batch: Batch) -> candle::Result<Option<Tensor>> {

        let seq_len = batch.input_ids.len();

        let input_ids = Tensor::from_vec(
            batch.input_ids.clone(),
            (1, seq_len),
            &Device::Cpu,
        )?;

        let pooled_embeddings = self.base_model.forward(&input_ids, 0)?;

        let dense1_hidden = self.dense1.forward(&pooled_embeddings)?;
        let output_before_norm = self.dense2.forward(&dense1_hidden)?;

        // L2 normalize the tensor (before converting to Vec)
        let norm = output_before_norm.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = output_before_norm.broadcast_div(&norm)?;

        Ok(Some(normalized))
    
    }
}

#[wasm_bindgen]
pub struct Gemma3Embedder {
    model: Gemma3Model,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl Gemma3Embedder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights: Vec<u8>,
        weights_dense1: Vec<u8>,
        weights_dense2: Vec<u8>,
        tokenizer: Vec<u8>, 
        config: Vec<u8>
    ) -> Result<Gemma3Embedder, JsError> {
        console_error_panic_hook::set_once();

        let device = Device::Cpu;
        console_log!("Loading tokenizer...");
        let tokenizer = 
            Tokenizer::from_bytes(&tokenizer).map_err(|e| JsError::new(&e.to_string()))?;

        console_log!("Loading config...");
        let config: Gemma3Config = 
            serde_json::from_slice(&config).map_err(|e| JsError::new(&e.to_string()))?;

        let mut cursor = std::io::Cursor::new(weights);

        console_log!("Reading gguf...");
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| JsError::new(&format!("GGUF parse error: {e}")))?;

        console_log!("Reading dense 1...");
        let vb_dense1 =
            VarBuilder::from_buffered_safetensors(weights_dense1, DType::F32, &device)
                .map_err(|e| JsError::new(&e.to_string()))?;

        console_log!("Reading dense 2...");
        let vb_dense2 =
            VarBuilder::from_buffered_safetensors(weights_dense2, DType::F32, &device)
                .map_err(|e| JsError::new(&e.to_string()))?;

        console_log!("Reading combined model...");
        let model = 
            Gemma3Model::load(content, &mut cursor, vb_dense1, vb_dense2, &config)
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(Self { model, tokenizer })
    }

    /// Embed a single sentence
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        let batch = self.encode_batch(&[text.to_string()])?;
        let normed_output = self.model.forward(batch).map_err(|e| JsError::new(&e.to_string()))?;
        let normed_output =
            normed_output.ok_or_else(|| JsError::new("Embedding generation failed!"))?;
        let normed_output =
            normed_output.to_vec2::<f32>().map_err(|e| JsError::new(&e.to_string()))?;
        let emb = normed_output[0].clone();

        Ok(emb)
    }

    /// Internal helper: convert texts -> Batch
    fn encode_batch(&self, texts: &[String]) -> Result<Batch, JsError> {
        let mut all_ids = Vec::new();
        let mut all_positions = Vec::new();
        let mut cumulative = vec![0];
        let mut max_len = 0;

        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| JsError::new(&format!("Tokenization failed: {e}")))?;

            let ids = encoding.get_ids().to_vec();
            let len = ids.len();
            max_len = max_len.max(len);
            
            all_ids.extend(ids.iter().cloned());
            all_positions.extend((0..len as u32).collect::<Vec<u32>>());
            cumulative.push(cumulative.last().unwrap() + len as u32);
        }

        let token_len = all_ids.len();
        Ok(Batch {
            input_ids: all_ids,
            token_type_ids: vec![0; token_len],
            position_ids: all_positions,
            cumulative_seq_lengths: cumulative,
            max_length: max_len as u32,
            pooled_indices: (0..texts.len() as u32).collect(),
        })
    }


}