use candle::{self as candle_core, D};
use candle_core::{Device, Error as CandleError, Result as CandleResult, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::modernbert::{
    Config as ModernBertConfig, ModernBertForSequenceClassification,
};
use serde::Deserialize;
use std::error::Error;
use tokenizers::{Encoding, PaddingParams, Tokenizer, TruncationParams};

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
struct ModernBertTokenizerConfig {
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
    special_tokens_map: ModernBertSpecialTokens,
}
type EncodingError = Box<dyn Error + Send + Sync>;
#[derive(Debug)]
pub enum ModernBertError {
    EncodingError(EncodingError),
    CandleError(CandleError),
    OtherError(String),
    DeserializationError(serde_json::Error),
}
use std::fmt;

impl fmt::Display for ModernBertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModernBertError::EncodingError(err) => write!(f, "IO error: {}", err),
            ModernBertError::CandleError(err) => write!(f, "Candle error: {}", err),
            ModernBertError::OtherError(err) => write!(f, "Other error {}", err),
            ModernBertError::DeserializationError(err) => {
                write!(f, "DeserializationError error {}", err)
            }
        }
    }
}

impl From<EncodingError> for ModernBertError {
    fn from(err: EncodingError) -> Self {
        ModernBertError::EncodingError(err)
    }
}
impl From<CandleError> for ModernBertError {
    fn from(err: CandleError) -> Self {
        ModernBertError::CandleError(err)
    }
}
impl From<String> for ModernBertError {
    fn from(err: String) -> Self {
        ModernBertError::OtherError(err)
    }
}
impl From<serde_json::Error> for ModernBertError {
    fn from(err: serde_json::Error) -> Self {
        ModernBertError::DeserializationError(err)
    }
}

impl ModernBertTokenizer {
    pub fn from_bytes(
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        special_tokens_map: &[u8],
    ) -> Result<Self, ModernBertError> {
        let mut tokenizer = Tokenizer::from_bytes(tokenizer)?;
        let special_tokens = ModernBertSpecialTokens::from_bytes(special_tokens_map)?;
        let tokenizer_config = ModernBertTokenizerConfig::from_bytes(tokenizer_config)?;
        console_log!("Success");

        let pad_token = special_tokens.pad_token.content.clone();
        let pad_id = tokenizer
            .token_to_id(pad_token.clone().as_str())
            .ok_or(String::from("Failed to retrieve pad token id"))?;

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

        Ok(Self {
            tokenizer,
            special_tokens_map: special_tokens,
        })
    }
    fn tokenize_hypothesis(&self, x: String, sep_token: &str) -> Vec<String> {
        vec![sep_token.into(), x]
    }
    /// Encode a text with hypothesis
    ///
    /// ```rust
    /// tokenizer.encode(
    ///     "Financial markets gained 5 %.".to_string(),
    ///     "The sentence is positive.".to_string(),
    /// )
    /// ```
    pub fn encode(
        &self,
        text: String,
        hypothesis: String,
        sep_token: &str,
    ) -> Result<Encoding, Box<dyn Error + Send + Sync>> {
        let encoded_hypothesis = self.tokenize_hypothesis(hypothesis, sep_token);
        let mut input = vec![text];
        input.extend(encoded_hypothesis);
        self.tokenizer.encode(input, true)
    }

    pub fn encode_batch(
        &self,
        texts: Vec<String>,
        hypothesis: String,
        sep_token: &str,
    ) -> Result<Vec<Encoding>, Box<dyn Error + Send + Sync>> {
        // TODO: avoid unnecessary copies
        let encoded_hypothesis = self.tokenize_hypothesis(hypothesis, sep_token);
        let mut inputs: Vec<Vec<String>> = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            inputs.push(vec![text.to_string()]);
            inputs[i].extend(encoded_hypothesis.clone());
        }
        self.tokenizer.encode_batch(inputs, true)
    }
}

#[wasm_bindgen]
pub struct ClassificationOutput {
    labels: Vec<String>,
    probs: Vec<Vec<f32>>,
}

#[wasm_bindgen]
impl ClassificationOutput {
    /// Returns the labels for each prediction
    #[wasm_bindgen]
    pub fn get_labels(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.labels).unwrap_or(JsValue::NULL)
    }

    /// Returns the probabilities for each prediction
    #[wasm_bindgen]
    pub fn get_probs(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.probs).unwrap_or(JsValue::NULL)
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
        let json_struct = serde_json::json!({
            "probs": self.probs,
            "labels": self.labels
        });

        serde_wasm_bindgen::to_value(&json_struct).unwrap_or(JsValue::NULL)
    }
}

#[wasm_bindgen]
pub struct ModernBertPredictor {
    model: ModernBertForSequenceClassification,
    tokenizer: ModernBertTokenizer,
    device: Device,
    #[allow(dead_code)]
    config: ModernBertConfig,
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

        Ok(ModernBertPredictor {
            model,
            tokenizer,
            device,
            config,
        })
    }

    fn get_entail_contradict_id(&self) -> Result<(u32, u32), String> {
        let label2id = self
            .config
            .classifier_config
            .clone()
            .ok_or("Classifier configuration is missing")?
            .label2id;

        let mut entail_id: Option<u32> = None;
        let mut contradict_id: Option<u32> = None;

        match label2id.len() {
            2..=3 => (),
            n => return Err(format!("Invalid number of labels ({}): must be 2 or 3", n)),
        }

        for (label, id) in &label2id {
            match label.as_str() {
                l if l.starts_with("entail") => {
                    entail_id = Some(*id);
                    if label2id.len() == 2 {
                        for (other_label, other_id) in &label2id {
                            if other_label != label {
                                contradict_id = Some(*other_id);
                                break;
                            }
                        }
                        break;
                    }
                }
                l if l.starts_with("contradict") || l.starts_with("not_entail") => {
                    contradict_id = Some(*id)
                }
                _ => continue,
            }
        }

        let entail_id = entail_id.ok_or("No 'entail' label found in label2id")?;
        let contradict_id = contradict_id.ok_or("No 'contradict' label found in label2id")?;

        Ok((entail_id, contradict_id))
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, ModernBertError> {
        let logits = self.model.forward(input_ids, attention_mask)?;
        let (entail_id, _) = self.get_entail_contradict_id()?;
        let column_indices = Tensor::new(&[entail_id], &self.device)?;
        let selected_logits = logits.index_select(&column_indices, D::Minus1)?;
        Ok(selected_logits)
    }

    fn forward_multilabel(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, ModernBertError> {
        console_log!("Inferring");
        let logits = self.model.forward(input_ids, attention_mask)?;
        console_log!("Inference finished");

        let (entail_id, contradict_id) = self.get_entail_contradict_id()?;

        let mut col_indices = vec![contradict_id, entail_id];
        col_indices.sort();
        let column_indices = Tensor::from_vec(col_indices.clone(), (2,), &self.device)?;
        let selected_logits = logits.index_select(&column_indices, D::Minus1)?;
        let probs = softmax(&selected_logits, 1)?;

        let entailment_id = Tensor::from_vec(vec![entail_id], (1,), &self.device)?;
        let entailment = probs.index_select(&entailment_id, D::Minus1)?;
        Ok(entailment)
    }

    pub fn predict(
        &self,
        text: String,
        hypotheses: Vec<String>,
        multi_label: bool,
    ) -> Result<ClassificationOutput, JsError> {
        let outputs = self
            .process_batch(vec![text], hypotheses.clone(), multi_label)
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(ClassificationOutput {
            labels: hypotheses,
            probs: outputs,
        })
    }

    fn process_batch(
        &self,
        texts: Vec<String>,
        hypotheses: Vec<String>,
        multi_label: bool,
    ) -> Result<Vec<Vec<f32>>, ModernBertError> {
        let mut outputs = Vec::with_capacity(hypotheses.len());
        let sep_token = &self.tokenizer.special_tokens_map.sep_token;

        for hypothesis in hypotheses {
            let encodings = self.tokenizer.encode_batch(
                texts.clone(),
                hypothesis,
                sep_token.content.as_str(),
            )?;
            let (input_ids, attention_mask) = self.prepare_tensors(&encodings)?;

            let logits = if multi_label {
                self.forward_multilabel(&input_ids, &attention_mask)?
            } else {
                self.forward(&input_ids, &attention_mask)?
            };
            outputs.push(logits);
        }

        let outputs = Tensor::cat(&outputs, 1)?;
        let probs = if multi_label {
            outputs.to_vec2::<f32>()?
        } else {
            softmax(&outputs, 1)?.to_vec2::<f32>()?
        };

        Ok(probs)
    }

    fn prepare_tensors(&self, encodings: &[Encoding]) -> Result<(Tensor, Tensor), ModernBertError> {
        let input_ids = Tensor::stack(
            &encodings
                .iter()
                .map(|enc| Tensor::new(enc.get_ids(), &self.device))
                .collect::<CandleResult<Vec<_>>>()?,
            0,
        )?;
        let attention_mask = Tensor::stack(
            &encodings
                .iter()
                .map(|enc| Tensor::new(enc.get_attention_mask(), &self.device))
                .collect::<CandleResult<Vec<_>>>()?,
            0,
        )?;
        Ok((input_ids, attention_mask))
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
