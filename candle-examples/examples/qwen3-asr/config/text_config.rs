//! Text model config (subset of `Qwen3ASRTextConfig`).

use serde::{Deserialize, Deserializer, Serialize};

/// RoPE scaling configuration.
///
/// This is intentionally compatible with HuggingFace configs which may contain both
/// `"type"` and `"rope_type"` fields. If both are present, `rope_type` wins.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RopeScaling {
    pub rope_type: Option<String>,
    pub factor: Option<f64>,
    pub original_max_position_embeddings: Option<usize>,
    pub attention_factor: Option<f64>,
    pub beta_fast: Option<f64>,
    pub beta_slow: Option<f64>,
    pub short_factor: Option<Vec<f64>>,
    pub long_factor: Option<Vec<f64>>,
    pub low_freq_factor: Option<f64>,
    pub high_freq_factor: Option<f64>,

    // ASR uses mRoPE with an interleaving section (e.g. [24, 20, 20]).
    pub mrope_section: Vec<usize>,

    /// Whether to use interleaved multimodal RoPE layout.
    pub interleaved: bool,

    /// Some configs use a more explicit flag name for mRoPE interleaving.
    pub mrope_interleaved: bool,
}

#[derive(Deserialize)]
struct RopeScalingHelper {
    #[serde(rename = "type")]
    type_field: Option<String>,

    rope_type: Option<String>,
    factor: Option<f64>,
    original_max_position_embeddings: Option<usize>,
    attention_factor: Option<f64>,
    beta_fast: Option<f64>,
    beta_slow: Option<f64>,
    short_factor: Option<Vec<f64>>,
    long_factor: Option<Vec<f64>>,
    low_freq_factor: Option<f64>,
    high_freq_factor: Option<f64>,

    #[serde(default)]
    mrope_section: Vec<usize>,

    #[serde(default)]
    interleaved: bool,

    #[serde(default)]
    mrope_interleaved: bool,
}

impl<'de> Deserialize<'de> for RopeScaling {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = RopeScalingHelper::deserialize(deserializer)?;
        Ok(Self {
            rope_type: helper.rope_type.or(helper.type_field),
            factor: helper.factor,
            original_max_position_embeddings: helper.original_max_position_embeddings,
            attention_factor: helper.attention_factor,
            beta_fast: helper.beta_fast,
            beta_slow: helper.beta_slow,
            short_factor: helper.short_factor,
            long_factor: helper.long_factor,
            low_freq_factor: helper.low_freq_factor,
            high_freq_factor: helper.high_freq_factor,
            mrope_section: helper.mrope_section,
            interleaved: helper.interleaved,
            mrope_interleaved: helper.mrope_interleaved,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
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

    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,

    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,

    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    #[serde(default = "default_use_cache")]
    pub use_cache: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

fn default_vocab_size() -> usize {
    151_936
}

fn default_hidden_size() -> usize {
    4096
}

fn default_intermediate_size() -> usize {
    22_016
}

fn default_num_hidden_layers() -> usize {
    32
}

fn default_num_attention_heads() -> usize {
    32
}

fn default_num_key_value_heads() -> usize {
    32
}

fn default_head_dim() -> usize {
    128
}

fn default_attention_dropout() -> f64 {
    0.0
}

fn default_attention_bias() -> bool {
    false
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_max_position_embeddings() -> usize {
    128_000
}

fn default_initializer_range() -> f64 {
    0.02
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_use_cache() -> bool {
    true
}

fn default_rope_theta() -> f64 {
    5_000_000.0
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            head_dim: default_head_dim(),
            attention_dropout: default_attention_dropout(),
            attention_bias: default_attention_bias(),
            hidden_act: default_hidden_act(),
            max_position_embeddings: default_max_position_embeddings(),
            initializer_range: default_initializer_range(),
            rms_norm_eps: default_rms_norm_eps(),
            use_cache: default_use_cache(),
            tie_word_embeddings: false,
            rope_theta: default_rope_theta(),
            rope_scaling: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RopeScaling, TextConfig};

    #[test]
    fn test_rope_scaling_deserialize_rope_type_wins() -> anyhow::Result<()> {
        let json = r#"{"type":"old","rope_type":"linear","factor":2.0}"#;
        let cfg: RopeScaling = serde_json::from_str(json)?;
        if cfg.rope_type.as_deref() != Some("linear") {
            anyhow::bail!("expected rope_type=linear, got {:?}", cfg.rope_type);
        }
        if cfg.factor != Some(2.0) {
            anyhow::bail!("expected factor=2.0, got {:?}", cfg.factor);
        }
        Ok(())
    }

    #[test]
    fn test_rope_scaling_deserialize_interleaved_flags() -> anyhow::Result<()> {
        let json = r#"{"mrope_section":[24,20,20],"interleaved":true,"mrope_interleaved":true}"#;
        let cfg: RopeScaling = serde_json::from_str(json)?;
        if cfg.mrope_section != vec![24, 20, 20] {
            anyhow::bail!("unexpected mrope_section: {:?}", cfg.mrope_section);
        }
        if !cfg.interleaved {
            anyhow::bail!("expected interleaved=true");
        }
        if !cfg.mrope_interleaved {
            anyhow::bail!("expected mrope_interleaved=true");
        }
        Ok(())
    }

    #[test]
    fn test_text_config_deserialize_attention_fields() -> anyhow::Result<()> {
        let json = r#"{"attention_dropout":0.25,"attention_bias":true}"#;
        let cfg: TextConfig = serde_json::from_str(json)?;
        if (cfg.attention_dropout - 0.25).abs() > f64::EPSILON {
            anyhow::bail!("unexpected attention_dropout: {}", cfg.attention_dropout);
        }
        if !cfg.attention_bias {
            anyhow::bail!("expected attention_bias=true");
        }
        Ok(())
    }
}
