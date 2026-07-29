//! Qwen3-TTS model configuration

use serde::{Deserialize, Serialize};

/// Top-level config corresponding to the model's `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TTSConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim_override: Option<usize>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_num_codebook_groups")]
    pub num_codebook_groups: usize,
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,
    #[serde(default = "default_speaker_embed_dim")]
    pub speaker_embed_dim: usize,
}

impl Qwen3TTSConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl Default for Qwen3TTSConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: None,
            head_dim_override: None,
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            sliding_window: None,
            num_codebook_groups: default_num_codebook_groups(),
            codebook_size: default_codebook_size(),
            speaker_embed_dim: default_speaker_embed_dim(),
        }
    }
}

fn default_model_type() -> String {
    "qwen3_tts".to_string()
}
fn default_vocab_size() -> usize {
    151936
}
fn default_hidden_size() -> usize {
    896
}
fn default_intermediate_size() -> usize {
    4864
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_num_attention_heads() -> usize {
    14
}
fn default_max_position_embeddings() -> usize {
    32768
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_num_codebook_groups() -> usize {
    16
}
fn default_codebook_size() -> usize {
    2048
}
fn default_speaker_embed_dim() -> usize {
    1024
}

// ── ParsedModelConfig ────────────────────────────────────────────────────────

/// Variant enum inferred from `config.json → tts_model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Base,
    CustomVoice,
    VoiceDesign,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base => write!(f, "base"),
            Self::CustomVoice => write!(f, "custom_voice"),
            Self::VoiceDesign => write!(f, "voice_design"),
        }
    }
}

/// All configuration dimensions needed to construct every sub-model.
/// Parsed from the HuggingFace `config.json`.
#[derive(Debug, Clone)]
pub struct ParsedModelConfig {
    pub model_type: ModelType,
    pub model_size: String,
    // Talker
    pub talker_hidden_size: usize,
    pub talker_intermediate_size: usize,
    pub talker_num_hidden_layers: usize,
    pub talker_num_attention_heads: usize,
    pub talker_num_key_value_heads: usize,
    pub talker_head_dim: usize,
    pub talker_vocab_size: usize,
    pub talker_text_vocab_size: usize,
    pub talker_text_hidden_size: usize,
    pub talker_rms_norm_eps: f64,
    pub talker_rope_theta: f64,
    pub talker_max_position_embeddings: usize,
    pub mrope_section: Option<[usize; 3]>,
    // CodePredictor
    pub cp_hidden_size: usize,
    pub cp_intermediate_size: usize,
    pub cp_num_hidden_layers: usize,
    pub cp_num_attention_heads: usize,
    pub cp_num_key_value_heads: usize,
    pub cp_head_dim: usize,
    pub cp_vocab_size: usize,
    pub cp_num_code_groups: usize,
    pub cp_rms_norm_eps: f64,
    pub cp_rope_theta: f64,
}

impl ParsedModelConfig {
    /// Parse from a JSON string (`config.json` contents).
    pub fn from_json(content: &str) -> candle::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| candle::Error::Msg(format!("failed to parse config.json: {e}")))?;

        let model_type = match v["tts_model_type"].as_str().unwrap_or("base") {
            "custom_voice" => ModelType::CustomVoice,
            "voice_design" => ModelType::VoiceDesign,
            _ => ModelType::Base,
        };
        let model_size = v["tts_model_size"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        let t = &v["talker_config"];
        let cp = &t["code_predictor_config"];

        let mrope_section = t["rope_scaling"]["mrope_section"]
            .as_array()
            .and_then(|arr| {
                if arr.len() == 3 {
                    Some([
                        arr[0].as_u64()? as usize,
                        arr[1].as_u64()? as usize,
                        arr[2].as_u64()? as usize,
                    ])
                } else {
                    None
                }
            });

        Ok(Self {
            model_type,
            model_size,
            talker_hidden_size: t["hidden_size"].as_u64().unwrap_or(1024) as usize,
            talker_intermediate_size: t["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            talker_num_hidden_layers: t["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
            talker_num_attention_heads: t["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            talker_num_key_value_heads: t["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            talker_head_dim: t["head_dim"].as_u64().unwrap_or(128) as usize,
            talker_vocab_size: t["vocab_size"].as_u64().unwrap_or(3072) as usize,
            talker_text_vocab_size: t["text_vocab_size"].as_u64().unwrap_or(151936) as usize,
            talker_text_hidden_size: t["text_hidden_size"].as_u64().unwrap_or(2048) as usize,
            talker_rms_norm_eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            talker_rope_theta: t["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            talker_max_position_embeddings: t["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            mrope_section,
            cp_hidden_size: cp["hidden_size"].as_u64().unwrap_or(1024) as usize,
            cp_intermediate_size: cp["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            cp_num_hidden_layers: cp["num_hidden_layers"].as_u64().unwrap_or(5) as usize,
            cp_num_attention_heads: cp["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            cp_num_key_value_heads: cp["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            cp_head_dim: cp["head_dim"].as_u64().unwrap_or(128) as usize,
            cp_vocab_size: cp["vocab_size"].as_u64().unwrap_or(2048) as usize,
            cp_num_code_groups: cp["num_code_groups"].as_u64().unwrap_or(16) as usize,
            cp_rms_norm_eps: cp["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            cp_rope_theta: cp["rope_theta"].as_f64().unwrap_or(1_000_000.0),
        })
    }

    /// Human-readable label, e.g. `"1.7B CustomVoice"`.
    pub fn label(&self) -> String {
        let size = match self.model_size.as_str() {
            "0b6" => "0.6B",
            "1b7" => "1.7B",
            other => other,
        };
        let variant = match self.model_type {
            ModelType::Base => "Base",
            ModelType::CustomVoice => "CustomVoice",
            ModelType::VoiceDesign => "VoiceDesign",
        };
        format!("{} {}", size, variant)
    }
}
