//! TranslateGemma - Translation models based on Gemma 3 architecture.
//!
//! TranslateGemma is a family of open translation models from Google, fine-tuned from
//! Gemma 3 checkpoints. Available in 4B, 12B, and 27B parameter sizes, supporting
//! translation across 55 languages.
//!
//! The architecture is identical to Gemma 3 (`Gemma3ForConditionalGeneration`), with
//! a specialized chat template for translation tasks.
//!
//! # Model Variants
//! - `google/translategemma-4b-it` - Optimized for mobile/edge deployment
//! - `google/translategemma-12b-it` - Consumer laptop deployment
//! - `google/translategemma-27b-it` - Maximum fidelity, single H100/TPU
//!
//! # Chat Template Format
//! TranslateGemma uses a specific template with ISO 639-1 language codes:
//! ```text
//! <bos><start_of_turn>user
//! <translate source_lang={src} target_lang={tgt}>
//! {text}
//! </translate><end_of_turn>
//! <start_of_turn>model
//! ```
//!
//! # Example
//! ```ignore
//! use candle_transformers::models::gemma::translate::{TranslateGemma, TranslateConfig, LanguageCode};
//!
//! let translator = TranslateGemma::new(&config, translate_config, tokenizer, false, vb)?;
//! let result = translator.translate("Hello!", LanguageCode::English, LanguageCode::French)?;
//! println!("{}", result.text);
//! ```

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::generation::LogitsProcessor;

// Re-use Gemma 3 model from sibling module
use super::gemma3::{Config as Gemma3Config, Model as Gemma3Model};

/// ISO 639-1 language codes supported by TranslateGemma.
///
/// TranslateGemma supports 55 core languages with additional experimental pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageCode {
    Arabic,
    Bulgarian,
    Bengali,
    Catalan,
    Czech,
    Danish,
    German,
    Greek,
    English,
    Spanish,
    Estonian,
    Persian,
    Finnish,
    French,
    Gujarati,
    Hebrew,
    Hindi,
    Croatian,
    Hungarian,
    Indonesian,
    Italian,
    Japanese,
    Kannada,
    Korean,
    Lithuanian,
    Latvian,
    Macedonian,
    Malayalam,
    Marathi,
    Malay,
    Burmese,
    Dutch,
    Norwegian,
    Polish,
    Portuguese,
    Romanian,
    Russian,
    Slovak,
    Slovenian,
    Albanian,
    Serbian,
    Swedish,
    Swahili,
    Tamil,
    Telugu,
    Thai,
    Tagalog,
    Turkish,
    Ukrainian,
    Urdu,
    Vietnamese,
    Chinese,
    /// Custom language code (ISO 639-1 or regionalized like "en-US")
    Custom(&'static str),
}

impl LanguageCode {
    /// Returns the ISO 639-1 language code string.
    pub fn as_str(&self) -> &str {
        match self {
            LanguageCode::Arabic => "ar",
            LanguageCode::Bulgarian => "bg",
            LanguageCode::Bengali => "bn",
            LanguageCode::Catalan => "ca",
            LanguageCode::Czech => "cs",
            LanguageCode::Danish => "da",
            LanguageCode::German => "de",
            LanguageCode::Greek => "el",
            LanguageCode::English => "en",
            LanguageCode::Spanish => "es",
            LanguageCode::Estonian => "et",
            LanguageCode::Persian => "fa",
            LanguageCode::Finnish => "fi",
            LanguageCode::French => "fr",
            LanguageCode::Gujarati => "gu",
            LanguageCode::Hebrew => "he",
            LanguageCode::Hindi => "hi",
            LanguageCode::Croatian => "hr",
            LanguageCode::Hungarian => "hu",
            LanguageCode::Indonesian => "id",
            LanguageCode::Italian => "it",
            LanguageCode::Japanese => "ja",
            LanguageCode::Kannada => "kn",
            LanguageCode::Korean => "ko",
            LanguageCode::Lithuanian => "lt",
            LanguageCode::Latvian => "lv",
            LanguageCode::Macedonian => "mk",
            LanguageCode::Malayalam => "ml",
            LanguageCode::Marathi => "mr",
            LanguageCode::Malay => "ms",
            LanguageCode::Burmese => "my",
            LanguageCode::Dutch => "nl",
            LanguageCode::Norwegian => "no",
            LanguageCode::Polish => "pl",
            LanguageCode::Portuguese => "pt",
            LanguageCode::Romanian => "ro",
            LanguageCode::Russian => "ru",
            LanguageCode::Slovak => "sk",
            LanguageCode::Slovenian => "sl",
            LanguageCode::Albanian => "sq",
            LanguageCode::Serbian => "sr",
            LanguageCode::Swedish => "sv",
            LanguageCode::Swahili => "sw",
            LanguageCode::Tamil => "ta",
            LanguageCode::Telugu => "te",
            LanguageCode::Thai => "th",
            LanguageCode::Tagalog => "tl",
            LanguageCode::Turkish => "tr",
            LanguageCode::Ukrainian => "uk",
            LanguageCode::Urdu => "ur",
            LanguageCode::Vietnamese => "vi",
            LanguageCode::Chinese => "zh",
            LanguageCode::Custom(code) => code,
        }
    }

    /// Parse a language code from string.
    pub fn from_str(s: &str) -> Option<Self> {
        let base = s.split(&['-', '_'][..]).next().unwrap_or(s).to_lowercase();
        match base.as_str() {
            "ar" => Some(LanguageCode::Arabic),
            "bg" => Some(LanguageCode::Bulgarian),
            "bn" => Some(LanguageCode::Bengali),
            "ca" => Some(LanguageCode::Catalan),
            "cs" => Some(LanguageCode::Czech),
            "da" => Some(LanguageCode::Danish),
            "de" => Some(LanguageCode::German),
            "el" => Some(LanguageCode::Greek),
            "en" => Some(LanguageCode::English),
            "es" => Some(LanguageCode::Spanish),
            "et" => Some(LanguageCode::Estonian),
            "fa" => Some(LanguageCode::Persian),
            "fi" => Some(LanguageCode::Finnish),
            "fr" => Some(LanguageCode::French),
            "gu" => Some(LanguageCode::Gujarati),
            "he" => Some(LanguageCode::Hebrew),
            "hi" => Some(LanguageCode::Hindi),
            "hr" => Some(LanguageCode::Croatian),
            "hu" => Some(LanguageCode::Hungarian),
            "id" => Some(LanguageCode::Indonesian),
            "it" => Some(LanguageCode::Italian),
            "ja" => Some(LanguageCode::Japanese),
            "kn" => Some(LanguageCode::Kannada),
            "ko" => Some(LanguageCode::Korean),
            "lt" => Some(LanguageCode::Lithuanian),
            "lv" => Some(LanguageCode::Latvian),
            "mk" => Some(LanguageCode::Macedonian),
            "ml" => Some(LanguageCode::Malayalam),
            "mr" => Some(LanguageCode::Marathi),
            "ms" => Some(LanguageCode::Malay),
            "my" => Some(LanguageCode::Burmese),
            "nl" => Some(LanguageCode::Dutch),
            "no" => Some(LanguageCode::Norwegian),
            "pl" => Some(LanguageCode::Polish),
            "pt" => Some(LanguageCode::Portuguese),
            "ro" => Some(LanguageCode::Romanian),
            "ru" => Some(LanguageCode::Russian),
            "sk" => Some(LanguageCode::Slovak),
            "sl" => Some(LanguageCode::Slovenian),
            "sq" => Some(LanguageCode::Albanian),
            "sr" => Some(LanguageCode::Serbian),
            "sv" => Some(LanguageCode::Swedish),
            "sw" => Some(LanguageCode::Swahili),
            "ta" => Some(LanguageCode::Tamil),
            "te" => Some(LanguageCode::Telugu),
            "th" => Some(LanguageCode::Thai),
            "tl" => Some(LanguageCode::Tagalog),
            "tr" => Some(LanguageCode::Turkish),
            "uk" => Some(LanguageCode::Ukrainian),
            "ur" => Some(LanguageCode::Urdu),
            "vi" => Some(LanguageCode::Vietnamese),
            "zh" => Some(LanguageCode::Chinese),
            _ => None,
        }
    }

    /// Get full language name for display.
    pub fn name(&self) -> &str {
        match self {
            LanguageCode::Arabic => "Arabic",
            LanguageCode::Bulgarian => "Bulgarian",
            LanguageCode::Bengali => "Bengali",
            LanguageCode::Catalan => "Catalan",
            LanguageCode::Czech => "Czech",
            LanguageCode::Danish => "Danish",
            LanguageCode::German => "German",
            LanguageCode::Greek => "Greek",
            LanguageCode::English => "English",
            LanguageCode::Spanish => "Spanish",
            LanguageCode::Estonian => "Estonian",
            LanguageCode::Persian => "Persian",
            LanguageCode::Finnish => "Finnish",
            LanguageCode::French => "French",
            LanguageCode::Gujarati => "Gujarati",
            LanguageCode::Hebrew => "Hebrew",
            LanguageCode::Hindi => "Hindi",
            LanguageCode::Croatian => "Croatian",
            LanguageCode::Hungarian => "Hungarian",
            LanguageCode::Indonesian => "Indonesian",
            LanguageCode::Italian => "Italian",
            LanguageCode::Japanese => "Japanese",
            LanguageCode::Kannada => "Kannada",
            LanguageCode::Korean => "Korean",
            LanguageCode::Lithuanian => "Lithuanian",
            LanguageCode::Latvian => "Latvian",
            LanguageCode::Macedonian => "Macedonian",
            LanguageCode::Malayalam => "Malayalam",
            LanguageCode::Marathi => "Marathi",
            LanguageCode::Malay => "Malay",
            LanguageCode::Burmese => "Burmese",
            LanguageCode::Dutch => "Dutch",
            LanguageCode::Norwegian => "Norwegian",
            LanguageCode::Polish => "Polish",
            LanguageCode::Portuguese => "Portuguese",
            LanguageCode::Romanian => "Romanian",
            LanguageCode::Russian => "Russian",
            LanguageCode::Slovak => "Slovak",
            LanguageCode::Slovenian => "Slovenian",
            LanguageCode::Albanian => "Albanian",
            LanguageCode::Serbian => "Serbian",
            LanguageCode::Swedish => "Swedish",
            LanguageCode::Swahili => "Swahili",
            LanguageCode::Tamil => "Tamil",
            LanguageCode::Telugu => "Telugu",
            LanguageCode::Thai => "Thai",
            LanguageCode::Tagalog => "Tagalog",
            LanguageCode::Turkish => "Turkish",
            LanguageCode::Ukrainian => "Ukrainian",
            LanguageCode::Urdu => "Urdu",
            LanguageCode::Vietnamese => "Vietnamese",
            LanguageCode::Chinese => "Chinese",
            LanguageCode::Custom(code) => code,
        }
    }
}

/// Configuration for TranslateGemma inference.
#[derive(Debug, Clone)]
pub struct TranslateConfig {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (None = greedy).
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling threshold.
    pub top_p: Option<f64>,
    /// Repetition penalty (1.0 = no penalty).
    pub repeat_penalty: f32,
    /// Context window for repetition penalty.
    pub repeat_last_n: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: None, // Greedy decoding for consistent translations
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}

/// Result of a translation operation.
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Translated text.
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Time taken in seconds.
    pub elapsed_secs: f64,
}

impl TranslationResult {
    /// Tokens per second throughput.
    pub fn tokens_per_second(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.tokens_generated as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

/// TranslateGemma prompt formatter.
pub struct PromptFormatter {
    bos_token: String,
    start_of_turn: String,
    end_of_turn: String,
}

impl Default for PromptFormatter {
    fn default() -> Self {
        Self {
            bos_token: "<bos>".to_string(),
            start_of_turn: "<start_of_turn>".to_string(),
            end_of_turn: "<end_of_turn>".to_string(),
        }
    }
}

impl PromptFormatter {
    /// Format a translation prompt.
    pub fn format(&self, text: &str, source_lang: &str, target_lang: &str) -> String {
        format!(
            "{bos}{start}user\n<translate source_lang={src} target_lang={tgt}>\n{text}\n</translate>{end}\n{start}model\n",
            bos = self.bos_token,
            start = self.start_of_turn,
            end = self.end_of_turn,
            src = source_lang,
            tgt = target_lang,
            text = text,
        )
    }
}

/// TranslateGemma model wrapper.
///
/// Wraps a Gemma 3 model with translation-specific functionality.
pub struct TranslateGemma {
    model: Gemma3Model,
    tokenizer: Tokenizer,
    config: TranslateConfig,
    device: Device,
    formatter: PromptFormatter,
    eos_token_id: u32,
    eot_token_id: u32,
}

impl TranslateGemma {
    /// Create a new TranslateGemma instance.
    pub fn new(
        model_config: &Gemma3Config,
        translate_config: TranslateConfig,
        tokenizer: Tokenizer,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let model = Gemma3Model::new(use_flash_attn, model_config, vb)?;

        let eos_token_id = tokenizer.token_to_id("<eos>").unwrap_or(1);
        let eot_token_id = tokenizer
            .token_to_id("<end_of_turn>")
            .unwrap_or(eos_token_id);

        Ok(Self {
            model,
            tokenizer,
            config: translate_config,
            device,
            formatter: PromptFormatter::default(),
            eos_token_id,
            eot_token_id,
        })
    }

    /// Translate text from source to target language.
    pub fn translate(
        &mut self,
        text: &str,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> Result<TranslationResult> {
        self.translate_with_codes(text, source_lang.as_str(), target_lang.as_str())
    }

    /// Translate using raw language code strings.
    pub fn translate_with_codes(
        &mut self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<TranslationResult> {
        self.model.clear_kv_cache();

        let prompt = self.formatter.format(text, source_lang, target_lang);

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| candle::Error::Msg(format!("Tokenization error: {}", e)))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let mut output_tokens = Vec::new();

        let mut logits_processor =
            LogitsProcessor::new(self.config.seed, self.config.temperature, self.config.top_p);

        let start_time = std::time::Instant::now();

        for index in 0..self.config.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let logits = if self.config.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                crate::utils::apply_repeat_penalty(
                    &logits,
                    self.config.repeat_penalty,
                    &tokens[start_at..],
                )?
            } else {
                logits
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if next_token == self.eos_token_id || next_token == self.eot_token_id {
                break;
            }

            output_tokens.push(next_token);
        }

        let elapsed = start_time.elapsed().as_secs_f64();

        let text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| candle::Error::Msg(format!("Decoding error: {}", e)))?;

        let text = text
            .trim()
            .trim_end_matches("<end_of_turn>")
            .trim_end_matches("<eos>")
            .trim()
            .to_string();

        Ok(TranslationResult {
            text,
            tokens_generated: output_tokens.len(),
            elapsed_secs: elapsed,
        })
    }

    /// Batch translate multiple texts.
    pub fn translate_batch(
        &mut self,
        texts: &[&str],
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> Result<Vec<TranslationResult>> {
        texts
            .iter()
            .map(|text| self.translate(text, source_lang, target_lang))
            .collect()
    }

    /// Clear KV cache.
    pub fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: TranslateConfig) {
        self.config = config;
    }

    /// Get reference to tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Model variant enumeration for TranslateGemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslateGemmaVariant {
    /// 4B parameter model
    T4B,
    /// 12B parameter model
    T12B,
    /// 27B parameter model
    T27B,
}

impl TranslateGemmaVariant {
    /// HuggingFace model ID.
    pub fn model_id(&self) -> &str {
        match self {
            TranslateGemmaVariant::T4B => "google/translategemma-4b-it",
            TranslateGemmaVariant::T12B => "google/translategemma-12b-it",
            TranslateGemmaVariant::T27B => "google/translategemma-27b-it",
        }
    }

    /// Approximate parameter count in billions.
    pub fn params_billions(&self) -> f32 {
        match self {
            TranslateGemmaVariant::T4B => 4.0,
            TranslateGemmaVariant::T12B => 12.0,
            TranslateGemmaVariant::T27B => 27.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_code_parsing() {
        assert_eq!(LanguageCode::from_str("en"), Some(LanguageCode::English));
        assert_eq!(LanguageCode::from_str("EN"), Some(LanguageCode::English));
        assert_eq!(LanguageCode::from_str("en-US"), Some(LanguageCode::English));
        assert_eq!(
            LanguageCode::from_str("pt-BR"),
            Some(LanguageCode::Portuguese)
        );
        assert_eq!(LanguageCode::from_str("xx"), None);
    }

    #[test]
    fn test_prompt_formatter() {
        let formatter = PromptFormatter::default();
        let prompt = formatter.format("Hello", "en", "fr");
        assert!(prompt.contains("<bos>"));
        assert!(prompt.contains("source_lang=en"));
        assert!(prompt.contains("target_lang=fr"));
        assert!(prompt.contains("Hello"));
    }
}
