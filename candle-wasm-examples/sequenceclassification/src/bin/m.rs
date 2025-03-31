use candle as candle_core;
use candle_core::{Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::bert::{BertForSequenceClassification, Config, DTYPE};
use candle_transformers::models::modernbert::{
    Config as ModernBertConfig, ModernBertForSequenceClassification,
};
use serde::Deserialize;
use std::{collections::HashMap, error::Error};
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::Model;
use tokenizers::{
    models::wordpiece::WordPiece, Encoding, PaddingParams, Tokenizer, TruncationParams,
};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BertTokenizerConfig {
    model_max_length: Option<usize>,
    do_lower_case: Option<bool>,
    tokenize_chinese_chars: Option<bool>,
    strip_accents: Option<bool>,
}

impl BertTokenizerConfig {
    fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::error::Error> {
        let tokenizer_config: BertTokenizerConfig = serde_json::from_slice(bytes)?;
        Ok(tokenizer_config)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BertSpecialTokens {
    unk_token: String,
    sep_token: String,
    pad_token: String,
    cls_token: String,
    mask_token: String,
}

impl BertSpecialTokens {
    fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::error::Error> {
        let special_tokens_map: BertSpecialTokens = serde_json::from_slice(bytes)?;
        Ok(special_tokens_map)
    }
}
pub struct BertTokenizer {
    tokenizer: Tokenizer,
}

impl BertTokenizer {
    /// # Arguments
    /// * vocab - The vocab file bytes. Generally, the file is named vocab.txt in Bert models
    /// * special_tokens_map - Special tokens map file. Generally, the file is named vocab.txt
    pub fn from_bytes(
        vocab: &[u8],
        tokenizer_config: &[u8],
        special_tokens_map: &[u8],
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let special_tokens = BertSpecialTokens::from_bytes(special_tokens_map)?;
        let tokenizer_config = BertTokenizerConfig::from_bytes(tokenizer_config)?;
        let wp_builder = WordPiece::from_bytes(vocab)?;
        let wp = wp_builder.unk_token(special_tokens.unk_token).build()?;

        let mut tokenizer = Tokenizer::new(wp.clone());
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        let sep = (
            special_tokens.sep_token.clone(),
            wp.token_to_id(special_tokens.sep_token.as_str())
                .ok_or("Failed to retrieve sep token id")?,
        );
        let cls = (
            special_tokens.cls_token.clone(),
            wp.token_to_id(special_tokens.cls_token.as_str())
                .ok_or("Failed to retrieve cls token id")?,
        );
        let post_processor = BertProcessing::new(sep, cls);
        tokenizer.with_post_processor(Some(post_processor));
        tokenizer.with_normalizer(Some(BertNormalizer::new(
            false,
            tokenizer_config.tokenize_chinese_chars.unwrap_or(false),
            tokenizer_config.strip_accents,
            tokenizer_config.do_lower_case.unwrap_or(false),
        )));
        let pad_token = special_tokens.pad_token;

        tokenizer
            .with_padding(Some(PaddingParams {
                pad_token: pad_token.clone(),
                pad_id: wp
                    .token_to_id(pad_token.as_str())
                    .ok_or("Failed to retrieve pad id")?,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: tokenizer_config
                    .model_max_length
                    .unwrap_or(Config::default().max_position_embeddings),
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..Default::default()
            }))?;

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Encoding, Box<dyn Error + Send + Sync>> {
        self.tokenizer.encode(text, true)
    }

    pub fn batch_encode(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Encoding>, Box<dyn Error + Send + Sync>> {
        self.tokenizer.encode_batch(texts, true)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ModernBertTokenizerConfig {
    // NOTE: this field is arbitrarily large in ModernBert classifier models tokenizer configuration and only
    // 128 bits would accommodate it
    // It is max_len that is used instead
    // model_max_length: Option<usize>,
    max_len: Option<usize>,
}

impl ModernBertTokenizerConfig {
    fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::error::Error> {
        let tokenizer_config: ModernBertTokenizerConfig = serde_json::from_slice(bytes)?;
        Ok(tokenizer_config)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ModernBertSpecialToken {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ModernBertSpecialTokens {
    pad_token: ModernBertSpecialToken,
    sep_token: ModernBertSpecialToken,
    unk_token: ModernBertSpecialToken,
    mask_token: ModernBertSpecialToken,
    cls_token: ModernBertSpecialToken,
}
impl ModernBertSpecialTokens {
    fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::error::Error> {
        let special_tokens_map: ModernBertSpecialTokens = serde_json::from_slice(bytes)?;
        Ok(special_tokens_map)
    }
}

#[wasm_bindgen]
pub struct ModernBertTokenizer {
    tokenizer: Tokenizer,
}

impl ModernBertTokenizer {
    pub fn from_bytes(
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        special_tokens_map: &[u8],
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut tokenizer = Tokenizer::from_bytes(tokenizer)?;
        let special_tokens = ModernBertSpecialTokens::from_bytes(special_tokens_map)?;
        let tokenizer_config = ModernBertTokenizerConfig::from_bytes(tokenizer_config)?;
        console_log!("Success");

        let pad_token = special_tokens.pad_token.content;
        let pad_id = tokenizer
            .token_to_id(pad_token.clone().as_str())
            .ok_or("Failed to retrieve pad token id")?;

        tokenizer
            .with_padding(Some(PaddingParams {
                pad_token,
                pad_id,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: tokenizer_config.max_len.unwrap_or(8192),
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..Default::default()
            }))?;

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Encoding, Box<dyn Error + Send + Sync>> {
        self.tokenizer.encode(text, true)
    }

    pub fn batch_encode(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Encoding>, Box<dyn Error + Send + Sync>> {
        self.tokenizer.encode_batch(texts, true)
    }
}

#[wasm_bindgen]
pub struct ClassificationOutput {
    id2label: Option<HashMap<u32, String>>,
    argmax: Vec<u32>,
    probs: Vec<Vec<f32>>,
}

#[wasm_bindgen]
impl ClassificationOutput {
    fn new(argmax: Vec<u32>, probs: Vec<Vec<f32>>, id2label: Option<HashMap<u32, String>>) -> Self {
        assert_eq!(
            argmax.len(),
            probs.len(),
            "argmax and probs must have the same length"
        );
        if !probs.is_empty() {
            let expected_len = probs[0].len();
            assert!(
                probs.iter().all(|p| p.len() == expected_len),
                "All probability vectors must have the same length"
            );
        }
        ClassificationOutput {
            id2label,
            argmax,
            probs,
        }
    }

    /// Returns the class indices for each prediction
    #[wasm_bindgen]
    pub fn get_classes(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.argmax).unwrap_or(JsValue::NULL)
    }

    /// Returns the labels for each prediction
    #[wasm_bindgen]
    pub fn get_labels(&self) -> JsValue {
        let labels: Vec<String> = match &self.id2label {
            Some(mapping) => self
                .argmax
                .iter()
                .map(|&id| mapping.get(&id).cloned().unwrap_or(id.to_string()))
                .collect(),
            None => self.argmax.iter().map(|id| id.to_string()).collect(),
        };
        serde_wasm_bindgen::to_value(&labels).unwrap_or(JsValue::NULL)
    }

    /// Returns the probabilities for each prediction
    #[wasm_bindgen]
    pub fn get_probs(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.probs).unwrap_or(JsValue::NULL)
    }

    /// Returns the number of predictions.
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.argmax.len()
    }

    /// Returns true if there are no predictions.
    #[wasm_bindgen]
    pub fn is_empty(&self) -> bool {
        self.argmax.is_empty()
    }

    /// Returns the probability for a specific class index in a specific prediction, or `None` if out of bounds.
    #[wasm_bindgen]
    pub fn get_prob(&self, prediction_idx: usize, class_idx: u32) -> Option<f32> {
        self.probs.get(prediction_idx).and_then(|probs| {
            let class_idx = class_idx as usize;
            if class_idx < probs.len() {
                probs.get(class_idx).copied()
            } else {
                None
            }
        })
    }

    #[wasm_bindgen]
    pub fn to_json(&self) -> JsValue {
        // Create a struct to match the desired JSON structure
        let json_struct = serde_json::json!({
            "classes": self.argmax,
            "probs": self.probs,
            "labels": match &self.id2label {
                Some(mapping) => self.argmax.iter().map(|&id| mapping.get(&id).cloned().unwrap_or(id.to_string())).collect::<Vec<String>>(),
                None => self.argmax.iter().map(|id| id.to_string()).collect::<Vec<String>>(),
            },
            "id2label": &self.id2label,
        });

        serde_wasm_bindgen::to_value(&json_struct).unwrap_or(JsValue::NULL)
    }
}

#[wasm_bindgen]
pub struct BertPredictor {
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: Device,
    config: Config,
}

#[wasm_bindgen]
impl BertPredictor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: Vec<u8>,
        model: Vec<u8>,
        vocab: Vec<u8>,
        tokenizer_config: Vec<u8>,
        special_tokens_map: Vec<u8>,
        num_labels: Option<usize>,
    ) -> Result<Self, JsError> {
        let device = Device::Cpu;

        let config: Config = serde_json::from_reader(config.as_slice())?;

        let vb = VarBuilder::from_buffered_safetensors(model, DTYPE, &device)?;
        let model = BertForSequenceClassification::load(vb, &config, num_labels)?;

        let tokenizer = BertTokenizer::from_bytes(
            vocab.as_slice(),
            tokenizer_config.as_slice(),
            special_tokens_map.as_slice(),
        )
        .map_err(|m| JsError::new(&m.to_string()))?;

        Ok(BertPredictor {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn predict(&self, texts: Vec<String>) -> Result<ClassificationOutput, JsError> {
        console_log!("Tokenizing");
        let encoded = self
            .tokenizer
            .batch_encode(texts.iter().map(|v| v.to_lowercase()).collect())
            .map_err(|m| JsError::new(&m.to_string()))?;

        let input_ids = Tensor::stack(
            &encoded
                .iter()
                .map(|enc| Tensor::new(enc.get_ids(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;
        let attention_mask = Tensor::stack(
            &encoded
                .iter()
                .map(|enc| Tensor::new(enc.get_attention_mask(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;
        let token_type_ids = Tensor::stack(
            &encoded
                .iter()
                .map(|enc| Tensor::new(enc.get_type_ids(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;

        console_log!("Inferring");
        let logits = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        let probs = softmax(&logits, 1)?;
        let argmax = logits.argmax(1)?;
        let id2label = self.config.classifier_config.as_ref().map(|config| {
            config
                .id2label
                .iter()
                .filter_map(|(key, value)| {
                    key.parse::<u32>()
                        .ok()
                        .map(|parsed_key| (parsed_key, value.clone()))
                })
                .collect::<HashMap<u32, String>>()
        });

        Ok(ClassificationOutput::new(
            argmax
                .to_vec1::<u32>()
                .map_err(|m| JsError::new(&m.to_string()))?,
            probs
                .to_vec2::<f32>()
                .map_err(|m| JsError::new(&m.to_string()))?,
            id2label,
        ))
    }
}

#[wasm_bindgen]
pub struct ModernBertPredictor {
    model: ModernBertForSequenceClassification,
    tokenizer: ModernBertTokenizer,
    device: Device,
    #[allow(dead_code)]
    config: ModernBertConfig,
    id2label: Option<HashMap<u32, String>>,
}

#[wasm_bindgen]
impl ModernBertPredictor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: Vec<u8>,
        model: Vec<u8>,
        tokenizer: Vec<u8>,
        tokenizer_config: Vec<u8>,
        special_tokens_map: Vec<u8>,
    ) -> Result<Self, JsError> {
        let device = Device::Cpu;

        let config: ModernBertConfig = serde_json::from_reader(config.as_slice())?;
        let vb = VarBuilder::from_buffered_safetensors(model, candle_core::DType::F32, &device)?;
        console_log!("Loading safe tensors");
        let model = ModernBertForSequenceClassification::load(vb, &config)?;

        console_log!("Loading Tokenizer");
        let tokenizer = ModernBertTokenizer::from_bytes(
            tokenizer.as_slice(),
            tokenizer_config.as_slice(),
            special_tokens_map.as_slice(),
        )
        .map_err(|m| JsError::new(&m.to_string()))?;

        let id2label = config.classifier_config.as_ref().map(|config| {
            config
                .id2label
                .iter()
                .filter_map(|(key, value)| {
                    key.parse::<u32>()
                        .ok()
                        .map(|parsed_key| (parsed_key, value.clone()))
                })
                .collect::<HashMap<u32, String>>()
        });

        Ok(ModernBertPredictor {
            model,
            tokenizer,
            device,
            config,
            id2label,
        })
    }

    pub fn predict(&self, texts: Vec<String>) -> Result<ClassificationOutput, JsError> {
        console_log!("Tokenizing");
        let encoded = self
            .tokenizer
            .batch_encode(texts.iter().map(|v| v.to_lowercase()).collect())
            .map_err(|m| JsError::new(&m.to_string()))?;

        let input_ids = Tensor::stack(
            &encoded
                .iter()
                .map(|enc| Tensor::new(enc.get_ids(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;
        let attention_mask = Tensor::stack(
            &encoded
                .iter()
                .map(|enc| Tensor::new(enc.get_attention_mask(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;

        console_log!("Inferring");
        let logits = self.model.forward(&input_ids, &attention_mask)?;
        console_log!("Inference finished");
        let probs = softmax(&logits, 1)?;
        let argmax = logits.argmax(1)?;

        Ok(ClassificationOutput {
            probs: probs
                .to_vec2::<f32>()
                .map_err(|m| JsError::new(&m.to_string()))?,
            argmax: argmax
                .to_vec1::<u32>()
                .map_err(|m| JsError::new(&m.to_string()))?,
            id2label: self.id2label.clone(),
        })
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
