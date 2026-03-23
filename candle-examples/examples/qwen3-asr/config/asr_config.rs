//! Top-level ASR config (mirrors `Qwen3ASRConfig`).
//!
//! The official HuggingFace model card stores most of the actual architecture
//! parameters under `thinker_config`, with only a small top-level wrapper.

use serde::{Deserialize, Serialize};

use crate::config::{AudioEncoderConfig, TextConfig};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AsrConfig {
    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    #[serde(default)]
    pub model_type: Option<String>,

    /// Forced aligner only: special token id used to select timestamp prediction positions.
    #[serde(default)]
    pub timestamp_token_id: Option<u32>,

    /// Forced aligner only: length in milliseconds of one predicted timestamp segment.
    #[serde(default)]
    pub timestamp_segment_time: Option<f64>,

    #[serde(default)]
    pub support_languages: Vec<String>,

    #[serde(default)]
    pub thinker_config: ThinkerConfig,

    #[serde(default)]
    pub transformers_version: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkerConfig {
    #[serde(default)]
    pub model_type: Option<String>,

    /// Forced aligner only: number of timestamp classes predicted by the head.
    #[serde(default)]
    pub classify_num: Option<usize>,

    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    #[serde(default)]
    pub pad_token_id: Option<u32>,

    #[serde(default)]
    pub audio_token_id: Option<u32>,

    #[serde(default)]
    pub audio_start_token_id: Option<u32>,

    #[serde(default)]
    pub audio_end_token_id: Option<u32>,

    #[serde(default)]
    pub dtype: Option<String>,

    #[serde(default)]
    pub initializer_range: Option<f64>,

    #[serde(default)]
    pub audio_config: AudioEncoderConfig,

    #[serde(default)]
    pub text_config: TextConfig,
}

#[cfg(test)]
mod tests {
    use super::AsrConfig;

    #[test]
    fn test_parse_minimal_asr_config() -> anyhow::Result<()> {
        let json = r#"
        {
          "model_type": "qwen3_asr",
          "support_languages": ["English"],
          "thinker_config": {
            "audio_token_id": 123,
            "audio_start_token_id": 124,
            "audio_end_token_id": 125,
            "audio_config": { "d_model": 896 },
            "text_config": { "hidden_size": 1024 }
          }
        }
        "#;

        let cfg: AsrConfig = serde_json::from_str(json)?;
        if cfg.model_type.as_deref() != Some("qwen3_asr") {
            anyhow::bail!("unexpected model_type: {:?}", cfg.model_type);
        }
        if cfg.support_languages != vec!["English".to_string()] {
            anyhow::bail!("unexpected support_languages: {:?}", cfg.support_languages);
        }
        if cfg.thinker_config.audio_token_id != Some(123) {
            anyhow::bail!(
                "unexpected audio_token_id: {:?}",
                cfg.thinker_config.audio_token_id
            );
        }
        if cfg.thinker_config.audio_config.d_model != 896 {
            anyhow::bail!(
                "unexpected audio_config.d_model: {}",
                cfg.thinker_config.audio_config.d_model
            );
        }
        if cfg.thinker_config.text_config.hidden_size != 1024 {
            anyhow::bail!(
                "unexpected text_config.hidden_size: {}",
                cfg.thinker_config.text_config.hidden_size
            );
        }

        Ok(())
    }

    #[test]
    fn test_parse_forced_aligner_fields() -> anyhow::Result<()> {
        let json = r#"
        {
          "model_type": "qwen3_asr",
          "timestamp_token_id": 151705,
          "timestamp_segment_time": 80,
          "thinker_config": {
            "model_type": "qwen3_forced_aligner",
            "classify_num": 5000,
            "audio_token_id": 123,
            "audio_start_token_id": 124,
            "audio_end_token_id": 125,
            "audio_config": { "d_model": 896 },
            "text_config": { "hidden_size": 1024 }
          }
        }
        "#;

        let cfg: AsrConfig = serde_json::from_str(json)?;
        if cfg.timestamp_token_id != Some(151705) {
            anyhow::bail!(
                "unexpected timestamp_token_id: {:?}",
                cfg.timestamp_token_id
            );
        }
        if cfg.timestamp_segment_time != Some(80.0) {
            anyhow::bail!(
                "unexpected timestamp_segment_time: {:?}",
                cfg.timestamp_segment_time
            );
        }
        if cfg.thinker_config.model_type.as_deref() != Some("qwen3_forced_aligner") {
            anyhow::bail!(
                "unexpected thinker_config.model_type: {:?}",
                cfg.thinker_config.model_type
            );
        }
        if cfg.thinker_config.classify_num != Some(5000) {
            anyhow::bail!(
                "unexpected thinker_config.classify_num: {:?}",
                cfg.thinker_config.classify_num
            );
        }
        Ok(())
    }
}
