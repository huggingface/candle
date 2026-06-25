use candle::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_add_final_layer_norm() -> bool {
    true
}

fn default_scale_embedding() -> bool {
    true
}

fn default_decoder_start_token_id() -> u32 {
    2 // MBart convention
}

/// Configuration for BART encoder-decoder models.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct BartConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    /// If None, defaults to d_model (encoder output matches decoder hidden size).
    #[serde(alias = "encoder_hidden_size")]
    pub cross_attention_hidden_size: Option<usize>,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub activation_function: candle_nn::Activation,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub activation_dropout: f64,
    /// Defaults to 2 (MBart convention) if null/missing.
    pub decoder_start_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default = "default_decoder_start_token_id")]
    pub eos_token_id: u32,
    #[serde(default)]
    pub forced_eos_token_id: Option<u32>,
    #[serde(default = "default_scale_embedding")]
    pub scale_embedding: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_add_final_layer_norm")]
    pub add_final_layer_norm: bool,

    // Encoder-specific fields (optional for backward compatibility with decoder-only configs)
    /// Number of encoder layers. Defaults to decoder_layers if not specified.
    #[serde(default)]
    pub encoder_layers: Option<usize>,
    /// Number of encoder attention heads. Defaults to decoder_attention_heads if not specified.
    #[serde(default)]
    pub encoder_attention_heads: Option<usize>,
    /// Encoder FFN intermediate dimension. Defaults to decoder_ffn_dim if not specified.
    #[serde(default)]
    pub encoder_ffn_dim: Option<usize>,
    /// Forced BOS token ID for multilingual models (e.g., target language token for mBART).
    #[serde(default)]
    pub forced_bos_token_id: Option<u32>,
    /// Model type identifier (e.g., "bart", "mbart").
    #[serde(default)]
    pub model_type: Option<String>,
    /// Whether this is an encoder-decoder model.
    #[serde(default)]
    pub is_encoder_decoder: Option<bool>,
    /// Controls whether to apply layernorm_embedding. Defaults to true for BART.
    #[serde(default)]
    pub normalize_embedding: Option<bool>,
    /// Force pre-layer norm architecture. If None, auto-detect from model_type.
    /// - mbart, donut: PRE-LAYERNORM (norm before attention)
    /// - bart: POST-LAYERNORM (norm after residual)
    #[serde(default)]
    pub use_pre_layer_norm: Option<bool>,
}

impl Default for BartConfig {
    fn default() -> Self {
        // Default values matching naver-clova-ix/donut-base
        Self {
            vocab_size: 57525,
            d_model: 1024,
            cross_attention_hidden_size: Some(1024),
            decoder_layers: 4,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: candle_nn::Activation::Gelu,
            max_position_embeddings: 1536,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: Some(2),
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            forced_eos_token_id: Some(2),
            scale_embedding: true,
            tie_word_embeddings: true,
            add_final_layer_norm: true,
            // Encoder fields default to None (use decoder values)
            encoder_layers: None,
            encoder_attention_heads: None,
            encoder_ffn_dim: None,
            forced_bos_token_id: None,
            model_type: None,
            is_encoder_decoder: None,
            normalize_embedding: None,
            use_pre_layer_norm: None,
        }
    }
}

/// Layer norm order for BART-like models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerNormOrder {
    /// PRE-LAYERNORM: norm BEFORE attention (MBart, Donut)
    Pre,
    /// POST-LAYERNORM: norm AFTER residual add (BART)
    Post,
}

impl BartConfig {
    /// Returns the decoder start token ID, defaulting to bos_token_id if not specified.
    pub fn get_decoder_start_token_id(&self) -> u32 {
        self.decoder_start_token_id.unwrap_or(self.bos_token_id)
    }

    /// Returns the forced BOS token ID for multilingual models.
    pub fn get_forced_bos_token_id(&self) -> Option<u32> {
        self.forced_bos_token_id
    }

    /// Returns the number of encoder layers, defaulting to decoder_layers.
    pub fn encoder_layers(&self) -> usize {
        self.encoder_layers.unwrap_or(self.decoder_layers)
    }

    /// Returns the number of encoder attention heads, defaulting to decoder_attention_heads.
    pub fn encoder_attention_heads(&self) -> usize {
        self.encoder_attention_heads
            .unwrap_or(self.decoder_attention_heads)
    }

    /// Returns the encoder FFN dimension, defaulting to decoder_ffn_dim.
    pub fn encoder_ffn_dim(&self) -> usize {
        self.encoder_ffn_dim.unwrap_or(self.decoder_ffn_dim)
    }

    /// Prepare initial decoder token sequence for generation.
    ///
    /// Returns [decoder_start_token_id]. For mBART, the target language token
    /// (forced_bos_token_id) should be forced as the first generated token
    /// after the decoder start token.
    pub fn initial_decoder_tokens(&self) -> Vec<u32> {
        vec![self.get_decoder_start_token_id()]
    }

    /// Determine layer norm order based on config or model_type.
    /// - Explicit `use_pre_layer_norm` takes precedence
    /// - Otherwise auto-detect: mbart/donut → Pre, bart → Post
    pub fn layer_norm_order(&self) -> LayerNormOrder {
        if let Some(use_pre) = self.use_pre_layer_norm {
            return if use_pre {
                LayerNormOrder::Pre
            } else {
                LayerNormOrder::Post
            };
        }
        // Auto-detect from model_type
        match self.model_type.as_deref() {
            Some("bart") => LayerNormOrder::Post,
            Some("mbart") | Some("donut") => LayerNormOrder::Pre,
            // Default to Pre for backward compatibility with MBart-based decoders
            _ => LayerNormOrder::Pre,
        }
    }

    /// Preset configuration for facebook/bart-base.
    pub fn bart_base() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 768,
            cross_attention_hidden_size: None,
            decoder_layers: 6,
            decoder_attention_heads: 12,
            decoder_ffn_dim: 3072,
            activation_function: candle_nn::Activation::Gelu,
            max_position_embeddings: 1024,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: Some(2),
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            forced_eos_token_id: Some(2),
            scale_embedding: true,
            tie_word_embeddings: true,
            add_final_layer_norm: false,
            encoder_layers: Some(6),
            encoder_attention_heads: Some(12),
            encoder_ffn_dim: Some(3072),
            forced_bos_token_id: None,
            model_type: Some("bart".to_string()),
            is_encoder_decoder: Some(true),
            normalize_embedding: Some(true),
            use_pre_layer_norm: Some(false), // BART uses POST-LAYERNORM
        }
    }

    /// Preset configuration for facebook/mbart-large-50-many-to-many-mmt.
    pub fn mbart_large_50() -> Self {
        Self {
            vocab_size: 250054,
            d_model: 1024,
            cross_attention_hidden_size: None,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: candle_nn::Activation::Gelu,
            max_position_embeddings: 1024,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: Some(2),
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            forced_eos_token_id: Some(2),
            scale_embedding: true,
            tie_word_embeddings: true,
            add_final_layer_norm: true,
            encoder_layers: Some(12),
            encoder_attention_heads: Some(16),
            encoder_ffn_dim: Some(4096),
            forced_bos_token_id: None, // Set dynamically based on target language
            model_type: Some("mbart".to_string()),
            is_encoder_decoder: Some(true),
            normalize_embedding: Some(true),
            use_pre_layer_norm: Some(true), // MBart uses PRE-LAYERNORM
        }
    }
}

/// Weight prefix variants for different BART checkpoint formats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BartWeightPrefix {
    /// VisionEncoderDecoder format: decoder.model.decoder.*
    VisionEncoderDecoder,
    /// Full BART/mBART format: model.decoder.*
    TextModel,
    /// Direct decoder format: decoder.*
    DirectDecoder,
}

impl BartWeightPrefix {
    /// Detect weight prefix by checking tensor existence.
    /// IMPORTANT: Check most specific (nested) path first!
    pub fn detect(vb: &VarBuilder) -> Result<Self> {
        // VisionEncoderDecoder is most nested - check first
        if vb
            .pp("decoder")
            .pp("model")
            .pp("decoder")
            .contains_tensor("embed_tokens.weight")
        {
            Ok(Self::VisionEncoderDecoder)
        } else if vb
            .pp("model")
            .pp("decoder")
            .contains_tensor("embed_tokens.weight")
        {
            Ok(Self::TextModel)
        } else if vb.pp("decoder").contains_tensor("embed_tokens.weight") {
            Ok(Self::DirectDecoder)
        } else {
            Err(candle::Error::Msg(
                "Cannot detect BART weight prefix: no embed_tokens found".into(),
            ))
        }
    }

    /// Returns the decoder path prefix for this weight format.
    pub fn decoder_prefix(&self) -> &'static str {
        match self {
            Self::VisionEncoderDecoder => "decoder.model.decoder",
            Self::TextModel => "model.decoder",
            Self::DirectDecoder => "decoder",
        }
    }

    /// Returns the LM head path prefix for this weight format.
    pub fn lm_head_prefix(&self) -> &'static str {
        match self {
            Self::VisionEncoderDecoder => "decoder.lm_head",
            Self::TextModel => "lm_head",
            Self::DirectDecoder => "lm_head",
        }
    }
}

/// Learned positional embedding with offset (MBart convention).
#[derive(Debug, Clone)]
pub struct BartLearnedPositionalEmbedding {
    offset: usize,
    weights: Embedding,
}

impl BartLearnedPositionalEmbedding {
    pub fn load(vb: VarBuilder, cfg: &BartConfig) -> Result<Self> {
        // MBart uses offset=2, reserving positions 0 and 1
        let offset: usize = 2;
        let num_embeddings = cfg.max_position_embeddings + offset;
        let embedding_dim = cfg.d_model;
        let weights = embedding(num_embeddings, embedding_dim, vb)?;
        Ok(Self { offset, weights })
    }

    pub fn forward(&self, input_ids: &Tensor, past_key_values_length: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let positions = Tensor::arange(
            past_key_values_length as u32,
            (seq_len + past_key_values_length) as u32,
            input_ids.device(),
        )?
        .expand((b_sz, seq_len))?;

        let positions =
            positions.broadcast_add(&Tensor::new(self.offset as u32, input_ids.device())?)?;
        self.weights.forward(&positions)
    }
}
