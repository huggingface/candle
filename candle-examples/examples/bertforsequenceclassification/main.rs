use candle::{Device, Tensor};
use candle_transformers::models::bert::{BertForSequenceClassification, Config, DTYPE};
use hf_hub::{
    api::sync::{Api, ApiError},
    Repo,
};
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::Model;
use tokenizers::{
    models::wordpiece::WordPiece, Encoding, PaddingParams, Tokenizer, TruncationParams,
};

use candle_nn::{ops::softmax, VarBuilder};
use core::fmt;
use serde::Deserialize;
use std::{
    collections::HashMap,
    error::Error,
    fs::{self, File},
    path::PathBuf,
};

pub type TokenizerError = Box<dyn std::error::Error + Send + Sync>;

const MODEL_ID: &str = "textattack/bert-base-uncased-yelp-polarity";
const REVISION_ID: &str = "refs/pr/1";

fn download_safetensors(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("model.safetensors")?;
    Ok(weights)
}

fn download_tokenizer_config(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("tokenizer_config.json")?;
    Ok(weights)
}

fn download_special_map_config(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("special_tokens_map.json")?;
    Ok(weights)
}

fn download_vocab(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("vocab.txt")?;
    Ok(weights)
}

fn download_config(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));

    let config = repo.get("config.json")?;
    Ok(config)
}

fn default_binary_labels() -> HashMap<String, String> {
    let mut labels = HashMap::new();
    labels.insert("0".into(), "Negative".to_string());
    labels.insert("1".into(), "Positive".to_string());
    labels
}

#[derive(Debug)]
pub struct PathBufToStrError();

impl fmt::Display for PathBufToStrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed converting PathBuf to str")
    }
}

impl Error for PathBufToStrError {}

#[derive(Debug)]
pub enum DeserializeError {
    Serde(serde_json::error::Error),
    Io(std::io::Error),
}

// Implement From traits for error conversion
impl From<serde_json::error::Error> for DeserializeError {
    fn from(err: serde_json::error::Error) -> Self {
        DeserializeError::Serde(err)
    }
}

impl From<std::io::Error> for DeserializeError {
    fn from(err: std::io::Error) -> Self {
        DeserializeError::Io(err)
    }
}

#[derive(Debug)]
pub enum TokenizerInitializationError {
    DeserializeError(DeserializeError),
    PathBufToStrError(PathBufToStrError),
    MissingID(String),
    Other(TokenizerError),
}

impl From<TokenizerError> for TokenizerInitializationError {
    fn from(err: TokenizerError) -> Self {
        TokenizerInitializationError::Other(err)
    }
}
impl From<DeserializeError> for TokenizerInitializationError {
    fn from(err: DeserializeError) -> Self {
        TokenizerInitializationError::DeserializeError(err)
    }
}

impl From<PathBufToStrError> for TokenizerInitializationError {
    fn from(err: PathBufToStrError) -> Self {
        TokenizerInitializationError::PathBufToStrError(err)
    }
}

impl From<&str> for TokenizerInitializationError {
    fn from(err: &str) -> Self {
        TokenizerInitializationError::MissingID(err.into())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BertTokenizerConfig {
    model_max_length: Option<usize>,
    do_lower_case: Option<bool>,
    tokenize_chinese_chars: Option<bool>,
    strip_accents: Option<bool>,
}

impl BertTokenizerConfig {
    fn from_file(path: &str) -> Result<Self, DeserializeError> {
        let reader = File::open(path)?;
        let tokenizer_config: BertTokenizerConfig = serde_json::from_reader(reader)?;
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
    fn from_file(path: &str) -> Result<Self, DeserializeError> {
        let reader = File::open(path)?;
        let special_tokens_map: BertSpecialTokens = serde_json::from_reader(reader)?;
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
    pub fn from_file(
        vocab: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
    ) -> Result<Self, TokenizerInitializationError> {
        let special_tokens =
            BertSpecialTokens::from_file(special_tokens_map.to_str().ok_or(PathBufToStrError())?)?;
        let tokenizer_config =
            BertTokenizerConfig::from_file(tokenizer_config.to_str().ok_or(PathBufToStrError())?)?;
        let wp_builder = WordPiece::from_file(vocab.to_str().ok_or(PathBufToStrError())?);
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

    pub fn encode(&self, text: &str) -> Result<Encoding, TokenizerError> {
        self.tokenizer.encode(text, true)
    }

    pub fn batch_encode(&self, texts: Vec<String>) -> Result<Vec<Encoding>, TokenizerError> {
        self.tokenizer.encode_batch(texts, true)
    }
}

#[derive(Debug)]
pub enum PredictionError {
    TokenizerInitializationError(TokenizerInitializationError),
    TokenizerError(TokenizerError),
    ModelError(candle::Error),
    ApiError(ApiError),
    IoError(std::io::Error),
    DeserializeError(serde_json::error::Error),
}

impl From<TokenizerInitializationError> for PredictionError {
    fn from(value: TokenizerInitializationError) -> Self {
        PredictionError::TokenizerInitializationError(value)
    }
}

impl From<ApiError> for PredictionError {
    fn from(value: ApiError) -> Self {
        PredictionError::ApiError(value)
    }
}

impl From<std::io::Error> for PredictionError {
    fn from(value: std::io::Error) -> Self {
        PredictionError::IoError(value)
    }
}

impl From<candle::Error> for PredictionError {
    fn from(value: candle::Error) -> Self {
        PredictionError::ModelError(value)
    }
}

impl From<serde_json::error::Error> for PredictionError {
    fn from(value: serde_json::error::Error) -> Self {
        PredictionError::DeserializeError(value)
    }
}

impl From<TokenizerError> for PredictionError {
    fn from(value: TokenizerError) -> Self {
        PredictionError::TokenizerError(value)
    }
}

fn main() -> Result<(), PredictionError> {
    let device = Device::Cpu;

    let tokenizer = BertTokenizer::from_file(
        download_vocab(MODEL_ID, REVISION_ID)?,
        download_tokenizer_config(MODEL_ID, REVISION_ID)?,
        download_special_map_config(MODEL_ID, REVISION_ID)?,
    )?;
    let filepath = download_safetensors(MODEL_ID, REVISION_ID)?;
    let buffer = fs::read(filepath)?;
    let config_filepath = download_config(MODEL_ID, REVISION_ID)?;
    let config: Config = serde_json::from_reader(File::open(config_filepath)?)?;

    let vb = VarBuilder::from_buffered_safetensors(buffer, DTYPE, &device)?;
    let bert = BertForSequenceClassification::load(vb, &config, Some(2))?;

    let texts = vec![
        "This product is bad".to_string(),
        "This product is good".to_string(),
    ];
    let encodings = tokenizer.batch_encode(texts.clone())?;
    let input_ids = Tensor::new(
        encodings
            .iter()
            .map(|e| e.get_ids().to_vec())
            .collect::<Vec<_>>(),
        &device,
    )?;
    let attention_mask = Tensor::new(
        encodings
            .iter()
            .map(|e| e.get_attention_mask().to_vec())
            .collect::<Vec<_>>(),
        &device,
    )?;
    let token_type_ids = Tensor::new(
        encodings
            .iter()
            .map(|e| e.get_type_ids().to_vec())
            .collect::<Vec<_>>(),
        &device,
    )?;

    let logits = bert.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
    let probs = softmax(&logits, 1)?;

    let argmax = logits.argmax(1)?;
    let argmax_vec = argmax.to_vec1::<u32>()?;
    let probs_vec = probs.to_vec2::<f32>()?;

    let labels = match config.classifier_config {
        Some(cc) => cc.id2label,
        None => {
            println!("Warning: No classifier config found, using default binary labels");
            default_binary_labels()
        }
    };

    for (i, &pred) in argmax_vec.iter().enumerate() {
        let label = labels.get(&pred.to_string()).unwrap();
        let confidence = probs_vec[i][pred as usize];
        let sequence = &texts[i];
        println!(
            "Sequence: '{}': Class = {}, Confidence = {:.4}",
            sequence, label, confidence
        );
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: Device = Device::Cpu;
    /// Testing that the behaviour is equivalent to Python
    ///
    /// ```python
    /// from transformers import AutoTokenizer, AutoModelForSequenceClassification
    /// import torch
    ///
    /// MODEL_ID = "textattack/bert-base-uncased-yelp-polarity"
    /// tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    /// model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    /// # device = torch.device("cpu")
    ///
    ///
    /// # Example text inputs
    /// texts = [
    ///     "This product is bad",
    ///     "This product is good",
    /// ]
    ///
    /// # Tokenize input texts
    /// inputs = tokenizer(
    ///     texts,
    ///     padding=True,
    ///     truncation=True,
    ///     max_length=512,  # Match Rust's max_position_embeddings
    ///     return_tensors="pt"
    /// )
    ///
    /// print("Tokenized inputs:", inputs)
    ///
    /// # Perform inference
    /// with torch.no_grad():
    ///     outputs = model(**inputs)
    ///     logits = outputs.logits
    ///     print("Logits:", logits)
    ///     # tensor(
    ///     #     [[ 4.5233, -4.2527],
    ///     #     [-2.7210,  3.1902]]
    ///     # )
    /// ```
    #[test]
    fn test_tokenizer_output() -> Result<(), PredictionError> {
        let tokenizer = BertTokenizer::from_file(
            download_vocab(MODEL_ID, REVISION_ID)?,
            download_tokenizer_config(MODEL_ID, REVISION_ID)?,
            download_special_map_config(MODEL_ID, REVISION_ID)?,
        )?;

        let texts = vec![
            "This product is bad".to_string(),
            "This product is good".to_string(),
        ];
        let encodings = tokenizer.batch_encode(texts)?;

        let expected_input_ids = Tensor::new(
            vec![
                vec![101u32, 2023u32, 4031u32, 2003u32, 2919u32, 102u32],
                vec![101u32, 2023u32, 4031u32, 2003u32, 2204u32, 102u32],
            ],
            &DEVICE,
        )?;
        let expected_attention_mask = Tensor::new(
            vec![
                vec![1u32, 1u32, 1u32, 1u32, 1u32, 1u32],
                vec![1u32, 1u32, 1u32, 1u32, 1u32, 1u32],
            ],
            &DEVICE,
        )?;
        let expected_token_type_ids = Tensor::new(
            vec![
                vec![0u32, 0u32, 0u32, 0u32, 0u32, 0u32],
                vec![0u32, 0u32, 0u32, 0u32, 0u32, 0u32],
            ],
            &DEVICE,
        )?;

        let actual_input_ids = Tensor::new(
            encodings
                .iter()
                .map(|e| e.get_ids().to_vec())
                .collect::<Vec<_>>(),
            &DEVICE,
        )?;
        let actual_attention_mask = Tensor::new(
            encodings
                .iter()
                .map(|e| e.get_attention_mask().to_vec())
                .collect::<Vec<_>>(),
            &DEVICE,
        )?;
        let actual_token_type_ids = Tensor::new(
            encodings
                .iter()
                .map(|e| e.get_type_ids().to_vec())
                .collect::<Vec<_>>(),
            &DEVICE,
        )?;

        assert_eq!(
            actual_input_ids.to_vec2::<u32>()?,
            expected_input_ids.to_vec2::<u32>()?,
            "Input IDs do not match expected values"
        );
        assert_eq!(
            actual_attention_mask.to_vec2::<u32>()?,
            expected_attention_mask.to_vec2::<u32>()?,
            "Attention mask does not match expected values"
        );
        assert_eq!(
            actual_token_type_ids.to_vec2::<u32>()?,
            expected_token_type_ids.to_vec2::<u32>()?,
            "Token type IDs do not match expected values"
        );

        Ok(())
    }
    #[test]
    fn test_inference_logits() -> Result<(), PredictionError> {
        let tokenizer = BertTokenizer::from_file(
            download_vocab(MODEL_ID, REVISION_ID)?,
            download_tokenizer_config(MODEL_ID, REVISION_ID)?,
            download_special_map_config(MODEL_ID, REVISION_ID)?,
        )?;
        let filepath = download_safetensors(MODEL_ID, REVISION_ID)?;
        let buffer = fs::read(filepath)?;
        let config_filepath = download_config(MODEL_ID, REVISION_ID)?;
        let config: Config = serde_json::from_reader(File::open(config_filepath)?)?;

        let vb = VarBuilder::from_buffered_safetensors(buffer, DTYPE, &DEVICE)?;
        let model = BertForSequenceClassification::load(vb, &config, Some(2))?;

        // Input texts
        let texts = vec![
            "This product is bad".to_string(),
            "This product is good".to_string(),
        ];
        let encodings = tokenizer.batch_encode(texts)?;

        let input_ids = Tensor::new(
            encodings
                .iter()
                .map(|e| e.get_ids().to_vec())
                .collect::<Vec<_>>(),
            &DEVICE,
        )?;
        let attention_mask = Tensor::new(
            encodings
                .iter()
                .map(|e| e.get_attention_mask().to_vec())
                .collect::<Vec<_>>(),
            &DEVICE,
        )?;
        let token_type_ids = Tensor::new(
            encodings
                .iter()
                .map(|e| e.get_type_ids().to_vec())
                .collect::<Vec<_>>(),
            &DEVICE,
        )?;

        let logits = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Expected logits from Python example
        let expected_logits = Tensor::new(
            vec![vec![4.5233f32, -4.2527], vec![-2.7210, 3.1902]],
            &DEVICE,
        )?;

        let diff = (&logits - &expected_logits)?.abs()?;
        let tolerance: f32 = 1e-4;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        assert!(
            max_diff < tolerance,
            "Logits differ too much from expected values. Max difference: {:?}. Expected: {:?}, Got: {:?}",
            max_diff,
            expected_logits.to_vec2::<f32>()?,
            logits.to_vec2::<f32>()?
        );

        Ok(())
    }
}
