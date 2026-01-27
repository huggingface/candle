//! TranslateGemma - Translation models based on Gemma 3 architecture.
//!
//! TranslateGemma is a family of open translation models from Google, fine-tuned from
//! Gemma 3 checkpoints. Available in 4B, 12B, and 27B parameter sizes, supporting
//! translation across 55 languages.
//!
//! The architecture is identical to Gemma 3, with a specialized prompt format
//! for translation tasks.
//!
//! # Model Variants
//! - `google/translategemma-4b-it` - Optimized for mobile/edge deployment
//! - `google/translategemma-12b-it` - Consumer laptop deployment  
//! - `google/translategemma-27b-it` - Maximum fidelity, single H100/TPU
//!
//! # Prompt Format
//! TranslateGemma uses a specific format with ISO 639-1 language codes:
//! ```text
//! <bos><start_of_turn>user
//! <translate type=text source_lang_code={src} target_lang_code={tgt}>
//! {text}
//! </translate><end_of_turn>
//! <start_of_turn>model
//! ```
//!
//! # Example
//! ```ignore
//! use candle_transformers::models::gemma::translate_gemma::{LanguageCode, format_translate_prompt};
//!
//! let prompt = format_translate_prompt("Hello!", "en", "fr");
//! // Use with gemma3::Model for inference
//! ```

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
}

impl LanguageCode {
    /// Returns the ISO 639-1 language code string.
    pub fn as_str(&self) -> &'static str {
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
        }
    }

    /// Parse a language code from string.
    pub fn parse(s: &str) -> Option<Self> {
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
    pub fn name(&self) -> &'static str {
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
        }
    }

    /// Returns all supported language codes.
    pub fn all() -> &'static [LanguageCode] {
        &[
            LanguageCode::Arabic,
            LanguageCode::Bulgarian,
            LanguageCode::Bengali,
            LanguageCode::Catalan,
            LanguageCode::Czech,
            LanguageCode::Danish,
            LanguageCode::German,
            LanguageCode::Greek,
            LanguageCode::English,
            LanguageCode::Spanish,
            LanguageCode::Estonian,
            LanguageCode::Persian,
            LanguageCode::Finnish,
            LanguageCode::French,
            LanguageCode::Gujarati,
            LanguageCode::Hebrew,
            LanguageCode::Hindi,
            LanguageCode::Croatian,
            LanguageCode::Hungarian,
            LanguageCode::Indonesian,
            LanguageCode::Italian,
            LanguageCode::Japanese,
            LanguageCode::Kannada,
            LanguageCode::Korean,
            LanguageCode::Lithuanian,
            LanguageCode::Latvian,
            LanguageCode::Macedonian,
            LanguageCode::Malayalam,
            LanguageCode::Marathi,
            LanguageCode::Malay,
            LanguageCode::Burmese,
            LanguageCode::Dutch,
            LanguageCode::Norwegian,
            LanguageCode::Polish,
            LanguageCode::Portuguese,
            LanguageCode::Romanian,
            LanguageCode::Russian,
            LanguageCode::Slovak,
            LanguageCode::Slovenian,
            LanguageCode::Albanian,
            LanguageCode::Serbian,
            LanguageCode::Swedish,
            LanguageCode::Swahili,
            LanguageCode::Tamil,
            LanguageCode::Telugu,
            LanguageCode::Thai,
            LanguageCode::Tagalog,
            LanguageCode::Turkish,
            LanguageCode::Ukrainian,
            LanguageCode::Urdu,
            LanguageCode::Vietnamese,
            LanguageCode::Chinese,
        ]
    }
}

/// Format the user message content for TranslateGemma.
///
/// This creates the inner content that should be used as the user message
/// in a Gemma chat template:
///
/// ```ignore
/// let content = format_translate_content("Hello!", "en", "fr");
/// // Returns: "<translate source_lang=en target_lang=fr>\nHello!\n</translate>"
/// // Use with ChatTemplate::gemma() as the user message content
/// ```
pub fn format_translate_content(text: &str, source_lang: &str, target_lang: &str) -> String {
    format!(
        "<translate source_lang={} target_lang={}>\n{}\n</translate>",
        source_lang, target_lang, text
    )
}

/// Format the translation prompt for TranslateGemma.
///
/// This creates the prompt that matches HuggingFace's apply_chat_template output.
/// Note: BOS token is added separately by the tokenizer.
pub fn format_translate_prompt(text: &str, source_lang: &str, target_lang: &str) -> String {
    let source_name = LanguageCode::parse(source_lang)
        .map(|l| l.name())
        .unwrap_or(source_lang);
    let target_name = LanguageCode::parse(target_lang)
        .map(|l| l.name())
        .unwrap_or(target_lang);

    format!(
        "<start_of_turn>user\n\
         You are a professional {source_name} ({source_lang}) to {target_name} ({target_lang}) translator. \
         Your goal is to accurately convey the meaning and nuances of the original {source_name} text \
         while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n\
         Produce only the {target_name} translation, without any additional explanations or commentary. \
         Please translate the following {source_name} text into {target_name}:\n\
         {text}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}

/// Model variant enumeration for TranslateGemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslateGemmaVariant {
    /// 4B parameter model - optimized for edge/mobile
    T4B,
    /// 12B parameter model - consumer laptop deployment
    T12B,
    /// 27B parameter model - maximum quality
    T27B,
}

impl TranslateGemmaVariant {
    /// HuggingFace model ID.
    pub fn model_id(&self) -> &'static str {
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
        assert_eq!(LanguageCode::parse("en"), Some(LanguageCode::English));
        assert_eq!(LanguageCode::parse("EN"), Some(LanguageCode::English));
        assert_eq!(LanguageCode::parse("en-US"), Some(LanguageCode::English));
        assert_eq!(LanguageCode::parse("pt-BR"), Some(LanguageCode::Portuguese));
        assert_eq!(LanguageCode::parse("xx"), None);
    }

    #[test]
    fn test_format_translate_content() {
        let content = format_translate_content("Hello", "en", "fr");
        assert!(content.contains("source_lang=en"));
        assert!(content.contains("target_lang=fr"));
        assert!(content.contains("Hello"));
        assert!(content.contains("<translate"));
        assert!(content.contains("</translate>"));
    }

    #[test]
    fn test_format_translate_prompt() {
        let prompt = format_translate_prompt("Hello", "en", "fr");
        assert!(prompt.starts_with("<start_of_turn>user"));
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("<start_of_turn>model"));
        assert!(prompt.contains("source_lang=en"));
    }

    #[test]
    fn test_variant_model_ids() {
        assert_eq!(
            TranslateGemmaVariant::T4B.model_id(),
            "google/translategemma-4b-it"
        );
    }
}
