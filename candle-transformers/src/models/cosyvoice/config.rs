//! CosyVoice3 Configuration
//!
//! Configuration structures for all CosyVoice3 components including
//! LLM, Flow (DiT), and HiFT Vocoder.

use serde::Deserialize;

/// CosyVoice3 sample rate and basic parameters
pub const SAMPLE_RATE: usize = 24000;
pub const MEL_HOP_SIZE: usize = 480;
pub const MEL_N_FFT: usize = 1920;
pub const MEL_N_MELS: usize = 80;

/// Token related constants
pub const SPEECH_TOKEN_SIZE: usize = 6561;
pub const TOKEN_FRAME_RATE: usize = 25;
pub const TOKEN_MEL_RATIO: usize = 2;

/// LLM constants
pub const LLM_DIM: usize = 896;
pub const LLM_VOCAB_SIZE: usize = 6761; // 6561 + 200 special

/// DiT constants
pub const DIT_DIM: usize = 1024;
pub const DIT_DEPTH: usize = 22;
pub const DIT_HEADS: usize = 16;
pub const DIT_HEAD_DIM: usize = 64;

/// HiFT constants
pub const HIFT_BASE_CHANNELS: usize = 512;
pub const ISTFT_N_FFT: usize = 16;
pub const ISTFT_HOP_LEN: usize = 4;

/// Speaker embedding dimension
pub const SPK_EMBED_DIM: usize = 192;

/// CosyVoice3 complete configuration
#[derive(Debug, Clone, Deserialize)]
pub struct CosyVoice3Config {
    pub sample_rate: usize,
    pub llm: CosyVoice3LMConfig,
    pub flow: FlowConfig,
    pub hift: HiFTConfig,
}

impl Default for CosyVoice3Config {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            llm: CosyVoice3LMConfig::default(),
            flow: FlowConfig::default(),
            hift: HiFTConfig::default(),
        }
    }
}

/// CosyVoice3 LLM configuration (based on Qwen2)
#[derive(Debug, Clone, Deserialize)]
pub struct CosyVoice3LMConfig {
    pub llm_input_size: usize,
    pub llm_output_size: usize,
    pub speech_token_size: usize,
    /// text:speech token mix ratio
    pub mix_ratio: (usize, usize),
    /// Embedded Qwen2 config
    pub qwen2: Qwen2Config,
}

impl Default for CosyVoice3LMConfig {
    fn default() -> Self {
        Self {
            llm_input_size: LLM_DIM,
            llm_output_size: LLM_DIM,
            speech_token_size: SPEECH_TOKEN_SIZE,
            mix_ratio: (5, 15),
            qwen2: Qwen2Config::default(),
        }
    }
}

/// Qwen2 configuration (simplified version used by CosyVoice-BlankEN)
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Self {
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4864,
            max_position_embeddings: 32768,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 151936,
            tie_word_embeddings: true,
        }
    }
}

/// Flow Decoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub vocab_size: usize,
    pub token_mel_ratio: usize,
    pub pre_lookahead_len: usize,
    pub dit: DiTConfig,
    pub cfm: CFMConfig,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            input_size: MEL_N_MELS,
            output_size: MEL_N_MELS,
            vocab_size: SPEECH_TOKEN_SIZE,
            token_mel_ratio: TOKEN_MEL_RATIO,
            pre_lookahead_len: 3,
            dit: DiTConfig::default(),
            cfm: CFMConfig::default(),
        }
    }
}

/// DiT (Diffusion Transformer) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DiTConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub ff_mult: usize,
    pub mel_dim: usize,
    pub spk_dim: usize,
    pub static_chunk_size: usize,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            dim: DIT_DIM,
            depth: DIT_DEPTH,
            heads: DIT_HEADS,
            dim_head: DIT_HEAD_DIM,
            ff_mult: 2,
            mel_dim: MEL_N_MELS,
            spk_dim: MEL_N_MELS,
            static_chunk_size: 50,
        }
    }
}

/// CFM (Conditional Flow Matching) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct CFMConfig {
    pub sigma_min: f64,
    pub t_scheduler: String,
    pub inference_cfg_rate: f64,
    pub n_timesteps: usize,
}

impl Default for CFMConfig {
    fn default() -> Self {
        Self {
            sigma_min: 1e-6,
            t_scheduler: "cosine".to_string(),
            inference_cfg_rate: 0.0, // Temporarily disabled for debugging
            n_timesteps: 10,
        }
    }
}

/// HiFT Generator configuration
#[derive(Debug, Clone, Deserialize)]
pub struct HiFTConfig {
    pub in_channels: usize,
    pub base_channels: usize,
    pub nb_harmonics: usize,
    pub sampling_rate: usize,
    pub nsf_alpha: f64,
    pub nsf_sigma: f64,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub istft_n_fft: usize,
    pub istft_hop_len: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// Source ResBlock kernel sizes (for source signal processing)
    pub source_resblock_kernel_sizes: Vec<usize>,
    /// Source ResBlock dilation sizes
    pub source_resblock_dilation_sizes: Vec<Vec<usize>>,
    /// conv_pre lookahead (right padding) for causal conv
    pub conv_pre_look_right: usize,
}

impl Default for HiFTConfig {
    fn default() -> Self {
        Self {
            in_channels: MEL_N_MELS,
            base_channels: HIFT_BASE_CHANNELS,
            nb_harmonics: 8,
            sampling_rate: SAMPLE_RATE,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            upsample_rates: vec![8, 5, 3],
            upsample_kernel_sizes: vec![16, 11, 7],
            istft_n_fft: ISTFT_N_FFT,
            istft_hop_len: ISTFT_HOP_LEN,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            // Source ResBlocks: 2 blocks matching upsample stages except the last
            source_resblock_kernel_sizes: vec![7, 11],
            source_resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5]],
            conv_pre_look_right: 4,
        }
    }
}

/// F0 Predictor configuration
#[derive(Debug, Clone, Deserialize)]
pub struct F0PredictorConfig {
    pub in_channels: usize,
    pub cond_channels: usize,
    pub kernel_size: usize,
    pub num_layers: usize,
}

impl Default for F0PredictorConfig {
    fn default() -> Self {
        Self {
            in_channels: MEL_N_MELS,
            cond_channels: 512,
            kernel_size: 3,
            num_layers: 5,
        }
    }
}

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            top_k: 25,
            top_p: 0.8,
            temperature: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Whether to enable streaming output
    pub stream: bool,
    /// Speech rate (0.5-2.0)
    pub speed: f32,
    /// Sampling configuration
    pub sampling: SamplingConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            stream: false,
            speed: 1.0,
            sampling: SamplingConfig::default(),
        }
    }
}

