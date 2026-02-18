mod utils;

const MODEL_ID: &str = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0";
const REVISION_ID: &str = "main";

use candle::{self as candle_core, IndexOp, D};
use candle_core::{Device, Error as CandleError, Result as CandleResult, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::modernbert::{
    Config as ModernBertConfig, ModernBertForSequenceClassification,
};
use hf_hub::api::sync::ApiError;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use std::{fmt, fs};
use tokenizers::{Encoding, PaddingParams, Tokenizer, TruncationParams};
use utils::{
    download_config, download_safetensors, download_special_map_config, download_tokenizer,
    download_tokenizer_config,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ModernBertTokenizerConfig {
    max_len: Option<usize>,
}

impl ModernBertTokenizerConfig {
    fn from_file(path: &str) -> Result<Self, DeserializationError> {
        let reader = File::open(path)?;
        let tokenizer_config: ModernBertTokenizerConfig = serde_json::from_reader(reader)?;
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

#[derive(Debug)]
pub enum DeserializationError {
    IOError(std::io::Error),
    SerdeError(serde_json::Error),
}

impl fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeserializationError::IOError(err) => write!(f, "{}", err),
            DeserializationError::SerdeError(err) => write!(f, "{}", err),
        }
    }
}

impl From<std::io::Error> for DeserializationError {
    fn from(err: std::io::Error) -> Self {
        DeserializationError::IOError(err)
    }
}
impl From<serde_json::Error> for DeserializationError {
    fn from(err: serde_json::Error) -> Self {
        DeserializationError::SerdeError(err)
    }
}

impl ModernBertSpecialTokens {
    fn from_file(path: &str) -> Result<Self, DeserializationError> {
        let reader = File::open(path)?;
        let special_tokens_map: ModernBertSpecialTokens = serde_json::from_reader(reader)?;
        Ok(special_tokens_map)
    }
}

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
    DeserializationError(DeserializationError),
    PathBufToStrError(PathBufToStrError),
}

impl fmt::Display for ModernBertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModernBertError::EncodingError(err) => write!(f, "IO error: {}", err),
            ModernBertError::CandleError(err) => write!(f, "Candle error: {}", err),
            ModernBertError::OtherError(err) => write!(f, "Other error {}", err),
            ModernBertError::DeserializationError(err) => {
                write!(f, "DeserializationError error {}", err)
            }
            ModernBertError::PathBufToStrError(err) => {
                write!(f, "{}", err)
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
impl From<DeserializationError> for ModernBertError {
    fn from(err: DeserializationError) -> Self {
        ModernBertError::DeserializationError(err)
    }
}
impl From<PathBufToStrError> for ModernBertError {
    fn from(err: PathBufToStrError) -> Self {
        ModernBertError::PathBufToStrError(err)
    }
}

#[derive(Debug)]
pub struct PathBufToStrError();

impl fmt::Display for PathBufToStrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed converting PathBuf to str")
    }
}

impl ModernBertTokenizer {
    pub fn from_file(
        tokenizer: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
    ) -> Result<Self, ModernBertError> {
        let special_tokens = ModernBertSpecialTokens::from_file(
            special_tokens_map.to_str().ok_or(PathBufToStrError())?,
        )?;
        let mut tokenizer = Tokenizer::from_file(tokenizer.to_str().ok_or(PathBufToStrError())?)?;
        let tokenizer_config = ModernBertTokenizerConfig::from_file(
            tokenizer_config.to_str().ok_or(PathBufToStrError())?,
        )?;

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
    ///     "The sentence is positive".to_string(),
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
        let encoded_hypothesis = self.tokenize_hypothesis(hypothesis, sep_token);
        let mut inputs: Vec<Vec<String>> = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            inputs.push(vec![text.to_string()]);
            inputs[i].extend(encoded_hypothesis.clone());
        }
        self.tokenizer.encode_batch(inputs, true)
    }
}

#[derive(Debug)]
pub struct ClassificationOutput {
    labels: Vec<String>,
    probs: Tensor,
}

pub struct ModernBertPredictor {
    model: ModernBertForSequenceClassification,
    tokenizer: ModernBertTokenizer,
    device: Device,
    config: ModernBertConfig,
}

impl ModernBertPredictor {
    pub fn from_file(
        config: PathBuf,
        model: PathBuf,
        tokenizer: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
    ) -> Result<Self, ModernBertError> {
        let device = Device::Cpu;
        let reader = File::open(config).map_err(DeserializationError::IOError)?;
        let config: ModernBertConfig =
            serde_json::from_reader(reader).map_err(DeserializationError::SerdeError)?;

        let buffer = fs::read(model).map_err(DeserializationError::IOError)?;
        let vb = VarBuilder::from_buffered_safetensors(buffer, candle_core::DType::F32, &device)?;
        let model = ModernBertForSequenceClassification::load(vb, &config)?;

        let tokenizer =
            ModernBertTokenizer::from_file(tokenizer, tokenizer_config, special_tokens_map)?;

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
        let logits = self.model.forward(input_ids, attention_mask)?;

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
    ) -> Result<ClassificationOutput, ModernBertError> {
        let outputs = self.process_batch(vec![text], hypotheses.clone(), multi_label)?;
        Ok(ClassificationOutput {
            labels: hypotheses,
            probs: outputs,
        })
    }

    pub fn predict_batch(
        &self,
        texts: Vec<String>,
        hypotheses: Vec<String>,
        multi_label: bool,
    ) -> Result<ClassificationOutput, ModernBertError> {
        let outputs = self.process_batch(texts, hypotheses.clone(), multi_label)?;
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
    ) -> Result<Tensor, ModernBertError> {
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
            outputs
        } else {
            softmax(&outputs, 1)?
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

#[derive(Debug)]
pub enum ZeroShotError {
    ModernBertError(ModernBertError),
    ApiError(ApiError),
}

impl From<ModernBertError> for ZeroShotError {
    fn from(err: ModernBertError) -> Self {
        ZeroShotError::ModernBertError(err)
    }
}

impl From<ApiError> for ZeroShotError {
    fn from(err: ApiError) -> Self {
        ZeroShotError::ApiError(err)
    }
}

impl fmt::Display for ZeroShotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeroShotError::ApiError(err) => write!(f, "{}", err),
            ZeroShotError::ModernBertError(err) => write!(f, "{}", err),
        }
    }
}

fn main() -> Result<(), ZeroShotError> {
    let examples = vec![
        "Financial markets gained 5 %.".to_string(),
        "Financial markets went down by 5%.".to_string(),
    ];

    let hypotheses = vec![
        "This sentence is positive.".to_string(),
        "This sentence is negative.".to_string(),
    ];
    let model = ModernBertPredictor::from_file(
        download_config(MODEL_ID, REVISION_ID)?,
        download_safetensors(MODEL_ID, REVISION_ID)?,
        download_tokenizer(MODEL_ID, REVISION_ID)?,
        download_tokenizer_config(MODEL_ID, REVISION_ID)?,
        download_special_map_config(MODEL_ID, REVISION_ID)?,
    )?;
    let output = model.predict_batch(examples.clone(), hypotheses, true)?;
    for (i, text) in examples.iter().enumerate().take(2) {
        let hyp1 = &output.labels[0];
        let hyp2 = &output.labels[1];
        let prob_hypothesis_1 = output
            .probs
            .i((i, 0))
            .map_err(ModernBertError::CandleError)?;
        let prob_hypothesis_2 = output
            .probs
            .i((i, 1))
            .map_err(ModernBertError::CandleError)?;
        println!("Text: {}", text);
        println!(
            "\tProbability of entailment of hypothesis 1: `{}` = {:?}",
            hyp1, prob_hypothesis_1
        );
        println!(
            "\tProbability of entailment of hypothesis 2: `{}` = {:?}\n",
            hyp2, prob_hypothesis_2
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: Device = Device::Cpu;

    fn setup_model_and_data(
    ) -> Result<(ModernBertPredictor, Vec<String>, Vec<String>), ZeroShotError> {
        let examples = vec![
            "Financial markets gained 5 %.".to_string(),
            "Financial markets went down by 5%.".to_string(),
        ];
        let hypotheses = vec![
            "This sentence is positive.".to_string(),
            "This sentence is negative.".to_string(),
        ];
        let model = ModernBertPredictor::from_file(
            download_config(MODEL_ID, REVISION_ID)?,
            download_safetensors(MODEL_ID, REVISION_ID)?,
            download_tokenizer(MODEL_ID, REVISION_ID)?,
            download_tokenizer_config(MODEL_ID, REVISION_ID)?,
            download_special_map_config(MODEL_ID, REVISION_ID)?,
        )?;
        Ok((model, examples, hypotheses))
    }

    // Helper function to compare tensors
    fn assert_tensor_eq(
        actual: &Tensor,
        expected: &Tensor,
        tolerance: f32,
    ) -> Result<(), ZeroShotError> {
        let diff = (actual - expected)
            .map_err(ModernBertError::CandleError)?
            .abs()
            .map_err(ModernBertError::CandleError)?;
        let max_diff = diff
            .max_all()
            .map_err(ModernBertError::CandleError)?
            .to_scalar::<f32>()
            .map_err(ModernBertError::CandleError)?;
        assert!(
            max_diff < tolerance,
            "Logits differ too much from expected values. Max difference: {}. Expected: {:?}, Got: {:?}",
            max_diff,
            expected.to_vec2::<f32>().map_err(ModernBertError::CandleError)?,
            actual.to_vec2::<f32>().map_err(ModernBertError::CandleError)?
        );
        Ok(())
    }

    /// Testing that the behaviour is equivalent to Python
    ///
    /// ```python
    /// from transformers import pipeline
    ///
    /// model_id = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
    ///
    /// examples = ["Financial markets gained 5 %.", "Financial markets went down by 5%."]
    /// hypotheses = ["This sentence is positive.", "This sentence is negative."]
    /// print("Inference with multi_label = True")
    /// pipe = pipeline("zero-shot-classification", model=model_id)
    /// output = pipe(
    ///     examples,
    ///     candidate_labels=hypotheses,
    ///     hypothesis_template="{}",
    ///     multi_label=True,
    /// )
    /// print(output)
    /// # [
    /// #     {
    /// #         "sequence": "Financial markets gained 5 %.",
    /// #         "labels": ["This sentence is positive.", "This sentence is negative."],
    /// #         "scores": [0.9642423391342163, 0.0035960127133876085],
    /// #     },
    /// #     {
    /// #         "sequence": "Financial markets went down by 5%.",
    /// #         "labels": ["This sentence is negative.", "This sentence is positive."],
    /// #         "scores": [0.995209813117981, 0.00039684693911112845],
    /// #     },
    /// # ]
    ///
    /// print("Inference with multi_label=False")
    /// output = pipe(
    ///     examples,
    ///     candidate_labels=hypotheses,
    ///     hypothesis_template="{}",
    ///     multi_label=False,
    /// )
    /// print(output)
    /// # [
    /// #     {
    /// #         "sequence": "Financial markets gained 5 %.",
    /// #         "labels": ["This sentence is positive.", "This sentence is negative."],
    /// #         "scores": [0.9893742203712463, 0.010625768452882767],
    /// #     },
    /// #     {
    /// #         "sequence": "Financial markets went down by 5%.",
    /// #         "labels": ["This sentence is negative.", "This sentence is positive."],
    /// #         "scores": [0.9986220002174377, 0.001377981505356729],
    /// #     },
    /// # ]
    /// ```

    #[test]
    fn test_inference() -> Result<(), ZeroShotError> {
        let (model, examples, hypotheses) = setup_model_and_data()?;

        // Test multi_label = true
        {
            let output = model.predict_batch(examples.clone(), hypotheses.clone(), true)?;
            let probs = output.probs;
            let expected_probs = Tensor::new(
                vec![
                    vec![0.9642423391342163f32, 0.0035960127133876085f32],
                    vec![0.00039684693911112845f32, 0.995209813117981f32],
                ],
                &DEVICE,
            )
            .map_err(ModernBertError::CandleError)?;
            assert_tensor_eq(&probs, &expected_probs, 1e-4)?;
        }

        // Test multi_label = false
        {
            let output = model.predict_batch(examples.clone(), hypotheses.clone(), false)?;
            let probs = output.probs;
            let expected_probs = Tensor::new(
                vec![
                    vec![0.9893742203712463f32, 0.010625768452882767f32],
                    vec![0.001377981505356729f32, 0.9986220002174377f32],
                ],
                &DEVICE,
            )
            .map_err(ModernBertError::CandleError)?;
            assert_tensor_eq(&probs, &expected_probs, 1e-4)?;
        }

        Ok(())
    }
}
