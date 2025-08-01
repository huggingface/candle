use super::voxtral_llama::{VoxtralLlama, VoxtralLlamaCache, VoxtralLlamaConfig};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_no_bias, Conv1d, Dropout, LayerNorm, Linear, VarBuilder,
};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct VoxtralEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub scale_embedding: bool,
    pub activation_function: String,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub initializer_range: f64,
    pub attention_dropout: f64,
    // These are set to 0.0 for compatibility with Whisper modular architecture
    pub dropout: f64,
    pub layerdrop: f64,
    pub activation_dropout: f64,
}

#[derive(Debug, Clone)]
pub struct VoxtralConfig {
    pub audio_config: VoxtralEncoderConfig,
    pub text_config: VoxtralLlamaConfig,
    pub audio_token_id: usize,
    pub projector_hidden_act: String,
}

impl Default for VoxtralConfig {
    fn default() -> Self {
        Self {
            audio_config: VoxtralEncoderConfig::default(),
            text_config: VoxtralLlamaConfig::voxtral_3b(),
            audio_token_id: 24,
            projector_hidden_act: "gelu".to_string(),
        }
    }
}

impl Default for VoxtralEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 51866,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            num_key_value_heads: 20,
            head_dim: 64,
            scale_embedding: false,
            activation_function: "gelu".to_string(),
            num_mel_bins: 128,
            max_source_positions: 1500,
            initializer_range: 0.02,
            attention_dropout: 0.0,
            // Set for Whisper compatibility
            dropout: 0.0,
            layerdrop: 0.0,
            activation_dropout: 0.0,
        }
    }
}

impl VoxtralEncoderConfig {
    /// Ensures dropout values are properly set for Whisper compatibility
    pub fn with_whisper_compatibility(mut self) -> Self {
        self.dropout = 0.0;
        self.layerdrop = 0.0;
        self.activation_dropout = 0.0;
        self
    }
}

/// Custom cache for multimodal inputs
#[derive(Debug, Clone)]
pub struct VoxtralCache {
    cache: VoxtralLlamaCache,
    audio_processed: bool,
    cached_audio_embeds: Option<Tensor>,
    cached_audio_positions: Option<Vec<(usize, usize)>>,
}

#[derive(Debug, Clone)]
pub struct VoxtralGenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub device: Device,
    /// If cache is None, the model will create a new cache.
    pub cache: Option<VoxtralCache>,
}

impl VoxtralGenerationConfig {
    pub fn new(device: Device) -> Self {
        Self {
            max_new_tokens: 500,
            temperature: 0.0,
            top_p: None,
            device,
            cache: None,
        }
    }
}

impl VoxtralCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &VoxtralLlamaConfig,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            cache: VoxtralLlamaCache::new(use_kv_cache, dtype, config, device)?,
            audio_processed: false,
            cached_audio_embeds: None,
            cached_audio_positions: None,
        })
    }

    pub fn reset(&mut self) {
        // Reset the audio cache state
        self.audio_processed = false;
        self.cached_audio_embeds = None;
        self.cached_audio_positions = None;
        // Note: LlamaCache reset needs to be handled at a higher level
        // as it requires device access
    }
}

/// Safely clamp tensor values for different dtypes
fn safe_clamp(x: &Tensor) -> Result<Tensor> {
    match x.dtype() {
        DType::F16 => {
            // Match PyTorch exactly: torch.finfo(torch.float16).max - 1000 = 64504.0
            let max_val = 64504.0;
            x.clamp(-max_val, max_val)
        }
        DType::BF16 => {
            // BF16 has larger range, typically doesn't need clamping
            Ok(x.clone())
        }
        _ => Ok(x.clone()),
    }
}

/// Replace audio tokens in embeddings with projected audio features
pub fn replace_audio_tokens(
    inputs_embeds: &Tensor,
    audio_embeds: &Tensor,
    audio_positions: &[(usize, usize)],
    device: &Device,
) -> Result<Tensor> {
    if audio_positions.is_empty() {
        return Ok(inputs_embeds.clone());
    }

    let (batch_size, seq_len, hidden_size) = inputs_embeds.dims3()?;
    let num_audio_tokens = audio_positions.len();

    // HF-style: audio_embeds shape is (total_audio_seq_len, hidden_size)
    let audio_embeds_dims = audio_embeds.dims2()?;
    let total_audio_embeds = audio_embeds_dims.0;

    // HF-style: Use audio embeddings one-to-one with audio tokens
    // We should now have the right number of audio tokens in the input sequence
    let audio_embeds = if total_audio_embeds >= num_audio_tokens {
        // Take the first num_audio_tokens embeddings to match the audio tokens
        if num_audio_tokens == total_audio_embeds {
            audio_embeds.clone()
        } else {
            audio_embeds.i(0..num_audio_tokens)?
        }
    } else {
        candle::bail!(
            "Not enough audio embeddings: need {}, got {}. Input sequence should have {} audio tokens.",
            num_audio_tokens,
            total_audio_embeds,
            total_audio_embeds
        );
    };

    // Create result tensor starting with text embeddings
    let mut result = inputs_embeds.clone();

    // Replace audio tokens with audio embeddings
    // Since we don't have scatter operations, we'll do this manually
    for (idx, &(batch_idx, seq_idx)) in audio_positions.iter().enumerate() {
        if batch_idx >= batch_size || seq_idx >= seq_len {
            candle::bail!(
                "Invalid audio position: ({}, {}) for tensor shape ({}, {}, {})",
                batch_idx,
                seq_idx,
                batch_size,
                seq_len,
                hidden_size
            );
        }

        // Get the audio embedding for this position
        let audio_embed = audio_embeds.i(idx)?;

        // Create a mask for this specific position
        let mut position_mask = vec![0f32; batch_size * seq_len];
        position_mask[batch_idx * seq_len + seq_idx] = 1.0;
        let position_mask = Tensor::new(position_mask.as_slice(), device)?
            .reshape((batch_size, seq_len, 1))?
            .to_dtype(inputs_embeds.dtype())?;

        // Broadcast audio embedding to full tensor shape
        let audio_embed_broadcast = audio_embed.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((
            batch_size,
            seq_len,
            hidden_size,
        ))?;

        // Update result: keep original where mask is 0, use audio where mask is 1
        let inverse_mask = (1.0 - &position_mask)?;
        result = (result.broadcast_mul(&inverse_mask)?
            + audio_embed_broadcast.broadcast_mul(&position_mask)?)?;
    }

    Ok(result)
}

/// Find positions of audio tokens in input sequences
pub fn find_audio_token_positions(
    input_ids: &Tensor,
    audio_token_id: usize,
) -> Result<Vec<(usize, usize)>> {
    // Handle both i64 and u32 token types by converting to i64 first if needed
    let input_ids = if input_ids.dtype() == candle::DType::U32 {
        input_ids.to_dtype(candle::DType::I64)?
    } else {
        input_ids.clone()
    };

    let input_ids = input_ids.to_vec2::<i64>()?;
    let mut positions = Vec::new();

    for (batch_idx, sequence) in input_ids.iter().enumerate() {
        for (seq_idx, &token_id) in sequence.iter().enumerate() {
            if token_id as usize == audio_token_id {
                positions.push((batch_idx, seq_idx));
            }
        }
    }

    Ok(positions)
}

#[derive(Debug, Clone)]
struct VoxtralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    attention_dropout: Dropout,
}

impl VoxtralAttention {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;

        if head_dim * num_heads != embed_dim {
            candle::bail!(
                "embed_dim must be divisible by num_heads ({} % {} != 0)",
                embed_dim,
                num_heads
            );
        }

        let scaling = (head_dim as f64).powf(-0.5);

        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        let attention_dropout = Dropout::new(cfg.attention_dropout as f32);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling,
            attention_dropout,
        })
    }

    fn reshape_for_scores(&self, x: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        x.reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }
}

impl Module for VoxtralAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;

        // Project queries, keys, and values - apply scaling to queries to match PyTorch SDPA
        let q = (self.q_proj.forward(x)? * self.scaling)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        let q = self.reshape_for_scores(&q, seq_len, bsz)?;
        let k = self.reshape_for_scores(&k, seq_len, bsz)?;
        let v = self.reshape_for_scores(&v, seq_len, bsz)?;

        // Manual SDPA-like implementation to match Python's numerical behavior exactly
        // Use F16 precision throughout to match PyTorch's F16 model
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        // Apply softmax in same precision as input (F16) to match Python
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Apply attention dropout (disabled during inference)
        let attn_weights = self.attention_dropout.forward(&attn_weights, false)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, embed_dim)
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            bsz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
struct VoxtralEncoderLayer {
    self_attn: VoxtralAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: candle_nn::Activation,
    dropout: Dropout,
    activation_dropout: Dropout,
}

impl VoxtralEncoderLayer {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;

        let self_attn = VoxtralAttention::new(cfg, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(embed_dim, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;

        let activation = match cfg.activation_function.as_str() {
            "gelu" => candle_nn::Activation::Gelu,
            "relu" => candle_nn::Activation::Relu,
            _ => candle::bail!(
                "Unsupported activation function: {}",
                cfg.activation_function
            ),
        };

        let dropout = Dropout::new(cfg.dropout as f32);
        let activation_dropout = Dropout::new(cfg.activation_dropout as f32);

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation,
            dropout,
            activation_dropout,
        })
    }

    pub fn get_fc1_out_dim(&self) -> usize {
        // Return the intermediate size from the config
        // Since Linear doesn't expose out_dim
        self.fc1.weight().dims()[0]
    }

    fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        // Self-attention with residual connection
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = self.dropout.forward(&x, training)?;
        let x = (x + residual)?;

        // Feed-forward network with residual connection
        let residual = &x;
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.apply(&self.activation)?;
        let x = self.activation_dropout.forward(&x, training)?;
        let x = self.fc2.forward(&x)?;
        let x = self.dropout.forward(&x, training)?;
        let x = (x + residual)?;

        // Safe clamping for numerical stability
        safe_clamp(&x)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Tensor,
    layers: Vec<VoxtralEncoderLayer>,
    layer_norm: LayerNorm,
    dropout: Dropout,
    layerdrop: f64,
}

impl VoxtralEncoder {
    pub fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Ensure Whisper compatibility
        let cfg = cfg.clone().with_whisper_compatibility();

        let embed_dim = cfg.hidden_size;

        // Convolutional layers for processing mel features
        let conv1 = candle_nn::conv1d(
            cfg.num_mel_bins,
            embed_dim,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let conv2 = candle_nn::conv1d(
            embed_dim,
            embed_dim,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        // Position embeddings
        let embed_positions = vb.get(
            (cfg.max_source_positions, embed_dim),
            "embed_positions.weight",
        )?;

        // Transformer layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VoxtralEncoderLayer::new(
                &cfg,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(cfg.dropout as f32);

        Ok(Self {
            conv1,
            conv2,
            embed_positions,
            layers,
            layer_norm,
            dropout,
            layerdrop: cfg.layerdrop,
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        self.forward_with_training(input_features, false)
    }

    pub fn forward_with_training(&self, input_features: &Tensor, training: bool) -> Result<Tensor> {
        // Keep conv layers in F16 to avoid shape issues
        let expected_dtype = self.conv1.weight().dtype();
        let input_features = if input_features.dtype() != expected_dtype {
            input_features.to_dtype(expected_dtype)?
        } else {
            input_features.clone()
        };

        // Apply convolutional layers with GELU activation
        let x = if false {
            // Keep conv layers in F16
            // Convert conv1 weights to F32 for computation
            let conv1_weight_f32 = self.conv1.weight().to_dtype(DType::F32)?;
            let conv1_bias_f32 = if let Some(bias) = self.conv1.bias() {
                Some(bias.to_dtype(DType::F32)?)
            } else {
                None
            };

            // Manual conv1d operation with F32 precision - conv1 has stride=1, padding=1
            let mut conv_result = input_features.conv1d(&conv1_weight_f32, 1, 1, 1, 1)?;
            if let Some(bias) = conv1_bias_f32 {
                conv_result = conv_result.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)?;
            }
            conv_result
        } else {
            self.conv1.forward(&input_features)?
        };

        // Apply GELU activation after conv1 (matches Python: conv1 -> GELU)
        let x = x.gelu()?;

        // Apply conv2 (matches Python: conv2)
        let x = if false {
            // Keep conv layers in F16
            // Convert conv2 weights to F32 for computation
            let conv2_weight_f32 = self.conv2.weight().to_dtype(DType::F32)?;
            let conv2_bias_f32 = if let Some(bias) = self.conv2.bias() {
                Some(bias.to_dtype(DType::F32)?)
            } else {
                None
            };

            // Manual conv1d operation with F32 precision - conv2 has stride=2, padding=1
            let mut conv_result = x.conv1d(&conv2_weight_f32, 2, 1, 1, 1)?;
            if let Some(bias) = conv2_bias_f32 {
                conv_result = conv_result.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)?;
            }
            conv_result
        } else {
            self.conv2.forward(&x)?
        };

        // Apply GELU activation after conv2 (FIX: matches Python: conv2 -> GELU)
        let x = x.gelu()?;

        // Reshape: (batch, embed_dim, seq_len) -> (batch, seq_len, embed_dim)
        let x = x.transpose(1, 2)?;

        // Add position embeddings - handle F32 position embeddings + F16 hidden states like PyTorch
        let seq_len = x.dim(1)?;
        let positions = self.embed_positions.i(..seq_len)?;

        // PyTorch automatically promotes F16 + F32 -> F32, then converts back to original dtype
        // We need to match this behavior exactly
        let x = if false {
            // Keep position embeddings in mixed precision
            // Force F32 computation for position embeddings
            let x_f32 = x.to_dtype(candle::DType::F32)?;
            let positions_f32 = positions.to_dtype(candle::DType::F32)?;
            x_f32.broadcast_add(&positions_f32)? // Keep result in F32
        } else if x.dtype() != positions.dtype() {
            // Convert hidden states to F32 for addition (positions are already F32)
            let x_f32 = x.to_dtype(candle::DType::F32)?;
            let result_f32 = x_f32.broadcast_add(&positions)?;
            // Convert back to original hidden states dtype (F16)
            result_f32.to_dtype(x.dtype())?
        } else {
            x.broadcast_add(&positions)?
        };

        // Apply dropout
        let mut x = self.dropout.forward(&x, training)?;

        for (idx, layer) in self.layers.iter().enumerate() {
            // Keep all computation in F16
            x = self.forward_layer_with_dropout(&x, layer, idx, training)?;
        }

        // Apply final layer normalization (critical for proper output values!)
        let x = self.layer_norm.forward(&x)?;

        Ok(x)
    }

    /// Forward a single layer with stochastic depth (layer dropout)
    fn forward_layer_with_dropout(
        &self,
        x: &Tensor,
        layer: &VoxtralEncoderLayer,
        _layer_idx: usize,
        training: bool,
    ) -> Result<Tensor> {
        if training && self.layerdrop > 0.0 {
            // Apply stochastic depth with proper randomization
            let mut rng = rand::rng();
            let keep_prob = 1.0 - self.layerdrop;
            let keep: bool = rng.random::<f64>() < keep_prob;

            if !keep {
                // Skip layer entirely (identity mapping)
                return Ok(x.clone());
            }
        }

        layer.forward(x, training)
    }

    /// Get the output dimension of the first FC layer (needed for projector)
    pub fn get_intermediate_size(&self) -> usize {
        if !self.layers.is_empty() {
            self.layers[0].get_fc1_out_dim()
        } else {
            // Fallback to config value
            5120 // Default intermediate size
        }
    }

    /// Process long audio sequences in chunks to save memory
    pub fn process_long_audio(
        &self,
        input_features: &Tensor,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let (_batch_size, _num_mel, seq_len) = input_features.dims3()?;

        if seq_len <= chunk_size {
            return self.forward(input_features);
        }

        let mut outputs = Vec::new();
        let step = chunk_size - overlap;

        for start in (0..seq_len).step_by(step) {
            let end = (start + chunk_size).min(seq_len);
            let chunk = input_features.i((.., .., start..end))?;

            // Process chunk
            let output = self.forward(&chunk)?;

            // Handle overlap by averaging
            if !outputs.is_empty() && overlap > 0 {
                let overlap_frames = overlap / 2; // Account for conv2 stride
                let last_output: &mut Tensor = outputs.last_mut().unwrap();
                let last_len = last_output.dim(1)?;

                // Average overlapping regions
                let overlap_start = last_len.saturating_sub(overlap_frames);
                let overlap_new = output.i((.., ..overlap_frames, ..))?;
                let overlap_old = last_output.i((.., overlap_start.., ..))?;
                let averaged = ((overlap_old + overlap_new)? * 0.5)?;

                // Update last output
                *last_output =
                    Tensor::cat(&[&last_output.i((.., ..overlap_start, ..))?, &averaged], 1)?;

                // Add non-overlapping part of current chunk
                outputs.push(output.i((.., overlap_frames.., ..))?);
            } else {
                outputs.push(output);
            }
        }

        // Concatenate all outputs
        let outputs_ref: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&outputs_ref, 1)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralMultiModalProjector {
    linear_1: Linear,
    linear_2: Linear,
    activation: candle_nn::Activation,
}

impl VoxtralMultiModalProjector {
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let linear_1 = linear_no_bias(
            cfg.audio_config.intermediate_size,
            cfg.text_config.hidden_size,
            vb.pp("linear_1"),
        )?;

        let linear_2 = linear_no_bias(
            cfg.text_config.hidden_size,
            cfg.text_config.hidden_size,
            vb.pp("linear_2"),
        )?;

        let activation = match cfg.projector_hidden_act.as_str() {
            "gelu" => candle_nn::Activation::Gelu,
            "relu" => candle_nn::Activation::Relu,
            _ => candle::bail!(
                "Unsupported projector activation: {}",
                cfg.projector_hidden_act
            ),
        };

        Ok(Self {
            linear_1,
            linear_2,
            activation,
        })
    }

    pub fn forward(&self, audio_features: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(audio_features)?;
        let x = x.apply(&self.activation)?;
        self.linear_2.forward(&x)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralForConditionalGeneration {
    audio_tower: VoxtralEncoder,
    language_model: VoxtralLlama,
    multi_modal_projector: VoxtralMultiModalProjector,
    audio_token_id: usize,
    audio_config: VoxtralEncoderConfig,
    text_config: VoxtralLlamaConfig,
}

impl VoxtralForConditionalGeneration {
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let audio_tower = VoxtralEncoder::new(&cfg.audio_config, vb.pp("audio_tower"))?;
        let language_model = VoxtralLlama::load(vb.pp("language_model"), &cfg.text_config)?;
        let multi_modal_projector =
            VoxtralMultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;

        Ok(Self {
            audio_tower,
            language_model,
            multi_modal_projector,
            audio_token_id: cfg.audio_token_id,
            audio_config: cfg.audio_config.clone(),
            text_config: cfg.text_config.clone(),
        })
    }

    /// Get the audio token ID used for this model
    pub fn audio_token_id(&self) -> usize {
        self.audio_token_id
    }

    /// Get the text model configuration
    pub fn text_config(&self) -> &VoxtralLlamaConfig {
        &self.text_config
    }

    /// Get the audio encoder configuration
    pub fn audio_config(&self) -> &VoxtralEncoderConfig {
        &self.audio_config
    }

    /// Process audio features through encoder and projector
    pub fn get_audio_embeds(&self, input_features: &Tensor) -> Result<Tensor> {
        let audio_outputs = self.audio_tower.forward(input_features)?;

        // Following HF implementation: reshape to (-1, config.intermediate_size) before projection
        // Python: audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
        // This transforms [1, 1500, 1280] -> [375, 5120] using intermediate_size from config
        let (batch_size, seq_len, hidden_size) = audio_outputs.dims3()?;

        // The key insight: Python reshapes from [1, 1500, 1280] to [375, 5120]
        // This means 1500 * 1280 = 375 * 5120 (1920000 elements)
        // So we need: new_batch_size = (batch_size * seq_len * hidden_size) / intermediate_size
        let total_elements = batch_size * seq_len * hidden_size;
        let new_batch_size = total_elements / self.audio_config.intermediate_size;

        // Verify the division is exact
        if total_elements % self.audio_config.intermediate_size != 0 {
            return Err(candle::Error::DimOutOfRange {
                shape: candle::Shape::from_dims(&[batch_size, seq_len, hidden_size]),
                dim: 0,
                op: "reshape",
            });
        }

        let audio_hidden =
            audio_outputs.reshape((new_batch_size, self.audio_config.intermediate_size))?;

        // Project to text space - this gives us embeddings for each audio position
        let projected = self.multi_modal_projector.forward(&audio_hidden)?;

        // Return shape: (batch_size * seq_len, text_hidden_size)
        // This matches HF implementation - no pooling, keep all audio token embeddings
        Ok(projected)
    }

    /// Process long audio sequences efficiently
    pub fn get_audio_embeds_chunked(
        &self,
        input_features: &Tensor,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let audio_outputs =
            self.audio_tower
                .process_long_audio(input_features, chunk_size, overlap)?;

        // Reshape and project (now outputs hidden_size, needs reshape to intermediate_size)
        let (batch_size, seq_len, hidden_size) = audio_outputs.dims3()?;
        // Apply same reshape logic as get_audio_embeds
        let total_elements = batch_size * seq_len * hidden_size;
        let new_batch_size = total_elements / self.audio_config.intermediate_size;
        let audio_hidden =
            audio_outputs.reshape((new_batch_size, self.audio_config.intermediate_size))?;

        let projected = self.multi_modal_projector.forward(&audio_hidden)?;

        // Reshape back to (batch_size, seq_len, text_hidden_size) for pooling
        let text_hidden_size = self.text_config.hidden_size;
        let projected = projected.reshape((batch_size, seq_len, text_hidden_size))?;

        // Apply mean pooling to reduce to single audio embedding per batch
        let pooled = projected.mean(1)?; // Mean across sequence dimension

        // Return shape: (batch_size, text_hidden_size)
        Ok(pooled)
    }

    /// Forward pass with audio features and text input
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_features: Option<&Tensor>,
        cache: &mut VoxtralCache,
        index_pos: usize,
    ) -> Result<Tensor> {
        // Get text embeddings
        let mut inputs_embeds = self.language_model.embed(input_ids)?;

        // If audio features are provided and not yet processed
        if let Some(features) = input_features {
            if !cache.audio_processed {
                let audio_embeds = self.get_audio_embeds(features)?;

                let audio_positions = find_audio_token_positions(input_ids, self.audio_token_id)?;

                // Cache for future use
                cache.cached_audio_embeds = Some(audio_embeds.clone());
                cache.cached_audio_positions = Some(audio_positions.clone());
                cache.audio_processed = true;

                inputs_embeds = replace_audio_tokens(
                    &inputs_embeds,
                    &audio_embeds,
                    &audio_positions,
                    input_ids.device(),
                )?;
            }
        }

        // Forward through language model using forward_input_embed
        self.language_model
            .forward_input_embed(&inputs_embeds, index_pos, &mut cache.cache)
    }

    /// Generate text given audio input
    pub fn generate(
        &self,
        input_ids: &Tensor,
        input_features: Option<&Tensor>,
        config: VoxtralGenerationConfig,
    ) -> Result<Vec<u32>> {
        // Validate inputs
        if config.max_new_tokens == 0 {
            return input_ids.i(0)?.to_vec1::<u32>(); // Get first batch
        }

        if config.temperature < 0.0 {
            candle::bail!(
                "Temperature must be non-negative, got {}",
                config.temperature
            );
        }

        if let Some(p) = config.top_p {
            if !(0.0..=1.0).contains(&p) {
                candle::bail!("top_p must be between 0 and 1, got {}", p);
            }
        }

        let mut final_cache = if let Some(cache) = config.cache {
            cache
        } else {
            // Get the dtype from the language model by creating a small embedding
            let dummy_token = Tensor::new(&[1u32], &config.device)?;
            let dummy_embed = self.language_model.embed(&dummy_token)?;
            let model_dtype = dummy_embed.dtype();
            VoxtralCache::new(true, model_dtype, &self.text_config, &config.device)?
        };
        let mut tokens = input_ids.i(0)?.to_vec1::<u32>()?; // Get first batch
        let initial_len = tokens.len();

        for idx in 0..config.max_new_tokens {
            let (input, index_pos) = if idx == 0 {
                (input_ids.clone(), 0)
            } else {
                // For subsequent generation steps, use only the last token
                let last_token = tokens[tokens.len() - 1];
                let calculated_pos = initial_len + idx - 1;
                (
                    Tensor::new(&[last_token], &config.device)?.unsqueeze(0)?,
                    calculated_pos,
                )
            };

            let logits = if idx == 0 {
                // First pass - include audio features
                match self.forward(&input, input_features, &mut final_cache, index_pos) {
                    Ok(logits) => logits,
                    Err(e) => {
                        return Err(candle::Error::Msg(format!(
                            "Failed to generate tokens: {}",
                            e
                        )));
                    }
                }
            } else {
                // Subsequent passes - text only
                match self.forward(&input, None, &mut final_cache, index_pos) {
                    Ok(logits) => logits,
                    Err(e) => {
                        return Err(candle::Error::Msg(format!(
                            "Failed to generate tokens: {}",
                            e
                        )));
                    }
                }
            };

            // Handle both 2D [batch, vocab] and 3D [batch, seq_len, vocab] logits
            let logits = if logits.dims().len() == 3 {
                // 3D case: [batch, seq_len, vocab] -> get last token
                logits.i((.., logits.dim(1)? - 1, ..))?
            } else {
                // 2D case: [batch, vocab] -> already the right shape
                logits
            };

            let next_token = if config.temperature > 0.0 {
                // Sample with temperature
                let prs = (logits / config.temperature)?;
                let prs = candle_nn::ops::softmax_last_dim(&prs)?;

                if let Some(top_p_val) = config.top_p {
                    // Apply top-p sampling
                    sample_top_p(&prs.squeeze(0)?, top_p_val, &config.device)?
                } else {
                    // Sample from full distribution
                    let probs_vec = prs.squeeze(0)?.to_vec1::<f32>()?;
                    let mut rng = rand::rng();
                    let mut cumsum = 0.0;
                    let rand_val: f32 = rng.random();
                    let mut sampled = 0u32;

                    for (idx, &prob) in probs_vec.iter().enumerate() {
                        cumsum += prob;
                        if cumsum > rand_val {
                            sampled = idx as u32;
                            break;
                        }
                    }
                    sampled
                }
            } else {
                // Greedy decoding - find the token with highest probability
                let argmax_result = match logits.argmax(D::Minus1) {
                    Ok(result) => result,
                    Err(e) => {
                        return Err(candle::Error::Msg(format!("Argmax failed: {}", e)));
                    }
                };

                // Handle the case where argmax returns [1] instead of scalar

                if argmax_result.dims().is_empty() {
                    // Already a scalar
                    match argmax_result.to_scalar::<u32>() {
                        Ok(token) => token,
                        Err(e) => {
                            return Err(candle::Error::Msg(format!("to_scalar failed: {}", e)));
                        }
                    }
                } else if argmax_result.dims() == [1] {
                    // Shape [1] - extract the single element
                    match argmax_result.i(0) {
                        Ok(scalar_tensor) => match scalar_tensor.to_scalar::<u32>() {
                            Ok(token) => token,
                            Err(e) => {
                                return Err(candle::Error::Msg(format!(
                                    "to_scalar on extracted element failed: {}",
                                    e
                                )));
                            }
                        },
                        Err(e) => {
                            return Err(candle::Error::Msg(format!(
                                "indexing argmax result failed: {}",
                                e
                            )));
                        }
                    }
                } else {
                    return Err(candle::Error::Msg(format!(
                        "Unexpected argmax result shape: {:?}",
                        argmax_result.shape()
                    )));
                }
            };

            tokens.push(next_token);

            // Check for EOS tokens - Voxtral uses different EOS tokens than hardcoded 2
            // Based on the Mistral/Voxtral tokenizer, common EOS tokens are:
            // 2 = </s>, 0 = <pad>, 128001, 128009 from various chat formats
            let eos_tokens = [2u32, 128001, 128009, 128256]; // Don't include 0 as it might be valid generation

            // Check for EOS tokens only if not ignoring them
            if eos_tokens.contains(&next_token) {
                break;
            }

            // Also break if we get repeated pad tokens (might indicate the model is stuck)
            if next_token == 0 && tokens.len() > 5 {
                let last_5_tokens = &tokens[tokens.len() - 5..];
                if last_5_tokens.iter().all(|&t| t == 0) {
                    break;
                }
            }
        }

        Ok(tokens)
    }
}

/// Sample from top-p probability distribution
fn sample_top_p(probs: &Tensor, top_p: f64, _device: &Device) -> Result<u32> {
    let (sorted_probs, sorted_indices) = probs.sort_last_dim(false)?;
    let cumsum = sorted_probs.cumsum(D::Minus1)?;
    let mask = cumsum.le(top_p)?;

    // Apply mask and renormalize
    let filtered_probs = sorted_probs.where_cond(&mask, &Tensor::zeros_like(&sorted_probs)?)?;
    let filtered_probs = (&filtered_probs / filtered_probs.sum_keepdim(D::Minus1)?)?;

    // Sample from filtered distribution
    // Since multinomial is not available, we'll use a simple sampling approach
    let probs_vec = filtered_probs.to_vec1::<f32>()?;
    let mut cumsum = 0.0;
    let mut rng = rand::rng();
    let rand_val: f32 = rng.random();
    let mut sample_idx = 0;

    for (idx, &prob) in probs_vec.iter().enumerate() {
        cumsum += prob;
        if cumsum > rand_val {
            sample_idx = idx;
            break;
        }
    }

    sorted_indices.i(sample_idx)?.to_scalar::<u32>()
}
