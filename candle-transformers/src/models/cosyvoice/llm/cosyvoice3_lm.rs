//! CosyVoice3 Language Model
//!
//! Implements autoregressive speech token generation based on Qwen2 architecture.
//! Supports Grouped Query Attention (GQA), RoPE, and KV Cache.

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use std::sync::Arc;

use crate::models::cosyvoice::config::{CosyVoice3LMConfig, SamplingConfig};

/// Special token IDs for speech tokens
pub const SOS_TOKEN: u32 = 6561;
pub const EOS_TOKEN: u32 = 6562;
pub const TASK_ID_TOKEN: u32 = 6563;
pub const FILL_TOKEN: u32 = 6564;

/// Rotary Position Embedding for LLM
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// CosyVoice3 Language Model
///
/// Based on Qwen2 architecture with GQA (Grouped Query Attention),
/// RoPE (Rotary Position Embedding), and KV Cache support.
#[derive(Debug)]
pub struct CosyVoice3LM {
    /// Text token embedding
    text_embedding: Embedding,
    /// Speech token embedding
    speech_embedding: Embedding,
    /// LLM decoder for speech token logits
    llm_decoder: Linear,
    /// Transformer layers
    transformer_layers: Vec<TransformerLayer>,
    /// Final norm
    norm: candle_nn::RmsNorm,
    /// Rotary embedding (kept for debugging and future extensions)
    #[allow(dead_code)]
    rotary_emb: Arc<RotaryEmbedding>,
    /// Speech token vocabulary size
    speech_token_size: usize,
    /// Hidden size (kept for debugging and future extensions)
    #[allow(dead_code)]
    hidden_size: usize,
    /// Device
    device: Device,
    /// Dtype
    dtype: DType,
}

/// Transformer layer with GQA support
#[derive(Debug)]
struct TransformerLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl TransformerLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_size: usize,
        eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            hidden_size / num_heads,
            rotary_emb,
            vb.pp("self_attn"),
        )?;
        let mlp = Mlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(hidden_size, eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(hidden_size, eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, seqlen_offset)?;
        let x = (x + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Grouped Query Attention with RoPE and KV Cache
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        // CosyVoice3 uses bias for q_proj, k_proj, v_proj but no bias for o_proj
        let q_proj =
            candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj =
            candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let query_states = self.q_proj.forward(x)?;
        let key_states = self.k_proj.forward(x)?;
        let value_states = self.v_proj.forward(x)?;

        // Reshape for attention: [B, T, H, D] -> [B, H, T, D]
        let query_states = query_states
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // KV Cache
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // GQA: repeat KV heads to match query heads
        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        // Attention computation
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match mask {
            None => attn_weights,
            Some(m) => attn_weights.broadcast_add(m)?,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        // Reshape back: [B, H, T, D] -> [B, T, H*D]
        attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// MLP with SiLU activation (SwiGLU)
#[derive(Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let x = (gate * up)?;
        self.down_proj.forward(&x)
    }
}

impl CosyVoice3LM {
    pub fn new(config: &CosyVoice3LMConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let hidden_size = config.llm_input_size;
        let head_dim = hidden_size / config.qwen2.num_attention_heads;

        // Create rotary embedding
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            config.qwen2.max_position_embeddings,
            config.qwen2.rope_theta,
            &device,
        )?);

        // Text embedding
        let text_embedding = candle_nn::embedding(
            config.qwen2.vocab_size,
            config.llm_input_size,
            vb.pp("llm.embed_tokens"),
        )?;

        // Speech embedding
        let speech_vocab_size = config.speech_token_size + 200;
        let speech_embedding = candle_nn::embedding(
            speech_vocab_size,
            config.llm_input_size,
            vb.pp("speech_embedding"),
        )?;

        // Transformer layers
        let mut transformer_layers = Vec::new();
        let vb_layers = vb.pp("llm.layers");
        for i in 0..config.qwen2.num_hidden_layers {
            let layer = TransformerLayer::new(
                config.llm_input_size,
                config.qwen2.num_attention_heads,
                config.qwen2.num_key_value_heads,
                config.qwen2.intermediate_size,
                config.qwen2.rms_norm_eps,
                rotary_emb.clone(),
                vb_layers.pp(i),
            )?;
            transformer_layers.push(layer);
        }

        // Final norm
        let norm = candle_nn::rms_norm(
            config.llm_input_size,
            config.qwen2.rms_norm_eps,
            vb.pp("llm.norm"),
        )?;

        // LLM decoder (no bias as per official implementation)
        let llm_decoder = candle_nn::linear_no_bias(
            config.llm_output_size,
            speech_vocab_size,
            vb.pp("llm_decoder"),
        )?;

        Ok(Self {
            text_embedding,
            speech_embedding,
            llm_decoder,
            transformer_layers,
            norm,
            rotary_emb,
            speech_token_size: config.speech_token_size,
            hidden_size,
            device,
            dtype,
        })
    }

    /// Forward through transformer with embeddings
    fn forward_embeds(&mut self, embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_batch, seq_len, _) = embeds.dims3()?;

        // Causal mask
        let mask = if seq_len > 1 {
            let mask: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if j > i {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Some(
                Tensor::from_vec(mask, (1, 1, seq_len, seq_len), &self.device)?
                    .to_dtype(self.dtype)?,
            )
        } else {
            None
        };

        let mut x = embeds.clone();
        for layer in self.transformer_layers.iter_mut() {
            x = layer.forward(&x, mask.as_ref(), seqlen_offset)?;
        }

        self.norm.forward(&x)
    }

    /// Autoregressive speech token generation
    pub fn inference(
        &mut self,
        text_tokens: &Tensor,
        prompt_text_tokens: &Tensor,
        prompt_speech_tokens: &Tensor,
        sampling_config: &SamplingConfig,
    ) -> Result<Vec<u32>> {
        // 1. Concatenate text tokens
        let text = Tensor::cat(&[prompt_text_tokens, text_tokens], 1)?;
        let text_len = text.dim(1)?;
        let prompt_text_len = prompt_text_tokens.dim(1)?;

        // 2. Embedding
        let text_emb = self.text_embedding.forward(&text)?;

        let sos_id = Tensor::from_slice(&[SOS_TOKEN], (1, 1), &self.device)?;
        let task_id = Tensor::from_slice(&[TASK_ID_TOKEN], (1, 1), &self.device)?;

        let sos_emb = self.speech_embedding.forward(&sos_id)?;
        let task_id_emb = self.speech_embedding.forward(&task_id)?;
        let prompt_speech_emb = self.speech_embedding.forward(prompt_speech_tokens)?;

        // 3. Build input sequence
        let lm_input = Tensor::cat(&[&sos_emb, &text_emb, &task_id_emb, &prompt_speech_emb], 1)?;

        // 4. Calculate min/max length (matching Python implementation)
        let min_len = (text_len - prompt_text_len) * 2;
        let max_len = (text_len - prompt_text_len) * 20;
        let mut out_tokens = Vec::new();

        self.clear_kv_cache();

        // First forward pass with full context
        let hidden = self.forward_embeds(&lm_input, 0)?;
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.narrow(1, seq_len - 1, 1)?;
        let logits = self.llm_decoder.forward(&last_hidden)?.squeeze(1)?;
        
        // ignore_eos = true for first token (step 0 < min_len)
        let next_token = self.sampling_ids(&logits, sampling_config, &out_tokens, true)?;

        if next_token >= self.speech_token_size as u32 {
            return Ok(out_tokens);
        }
        out_tokens.push(next_token);

        let mut seqlen_offset = lm_input.dim(1)?;

        // Subsequent tokens
        for step in 1..max_len {
            let next_id = Tensor::from_slice(&[*out_tokens.last().unwrap()], (1, 1), &self.device)?;
            let current_input = self.speech_embedding.forward(&next_id)?;

            let hidden = self.forward_embeds(&current_input, seqlen_offset)?;
            let logits = self.llm_decoder.forward(&hidden)?.squeeze(1)?;

            // ignore_eos = true if we haven't reached min_len
            let ignore_eos = step < min_len;
            let next_token = self.sampling_ids(&logits, sampling_config, &out_tokens, ignore_eos)?;

            if next_token >= self.speech_token_size as u32 {
                break;
            }

            out_tokens.push(next_token);
            seqlen_offset += 1;
        }

        Ok(out_tokens)
    }

    /// Debug version of inference with detailed logging
    pub fn inference_debug(
        &mut self,
        text_tokens: &Tensor,
        prompt_text_tokens: &Tensor,
        prompt_speech_tokens: &Tensor,
        sampling_config: &SamplingConfig,
    ) -> Result<Vec<u32>> {
        // 1. Concatenate text tokens
        let text = Tensor::cat(&[prompt_text_tokens, text_tokens], 1)?;
        let text_len = text.dim(1)?;
        let prompt_text_len = prompt_text_tokens.dim(1)?;

        // 2. Embedding
        let text_emb = self.text_embedding.forward(&text)?;

        let sos_id = Tensor::from_slice(&[SOS_TOKEN], (1, 1), &self.device)?;
        let task_id = Tensor::from_slice(&[TASK_ID_TOKEN], (1, 1), &self.device)?;

        let sos_emb = self.speech_embedding.forward(&sos_id)?;
        let task_id_emb = self.speech_embedding.forward(&task_id)?;
        let prompt_speech_emb = self.speech_embedding.forward(prompt_speech_tokens)?;

        // 3. Build input sequence
        let lm_input = Tensor::cat(&[&sos_emb, &text_emb, &task_id_emb, &prompt_speech_emb], 1)?;

        // 4. Calculate min/max length (matching Python implementation)
        let min_len = (text_len - prompt_text_len) * 2;
        let max_len = (text_len - prompt_text_len) * 20;
        let mut out_tokens = Vec::new();

        println!("min_len: {}, max_len: {}", min_len, max_len);

        self.clear_kv_cache();

        // First forward pass with full context
        let hidden = self.forward_embeds(&lm_input, 0)?;
        
        // Debug: print hidden stats
        {
            let h_f32 = hidden.to_dtype(candle::DType::F32)?;
            let flat = h_f32.flatten_all()?;
            let mean = flat.mean_all()?.to_scalar::<f32>()?;
            let var = flat.broadcast_sub(&flat.mean_all()?)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let std = var.sqrt();
            println!("hidden_states: shape={:?}, mean={:.6}, std={:.6}", hidden.shape().dims(), mean, std);
        }
        
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.narrow(1, seq_len - 1, 1)?;
        
        // Debug: print last_hidden stats
        {
            let h_f32 = last_hidden.to_dtype(candle::DType::F32)?;
            let flat = h_f32.flatten_all()?;
            let mean = flat.mean_all()?.to_scalar::<f32>()?;
            let var = flat.broadcast_sub(&flat.mean_all()?)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let std = var.sqrt();
            println!("last_hidden: shape={:?}, mean={:.6}, std={:.6}", last_hidden.shape().dims(), mean, std);
        }
        
        let logits = self.llm_decoder.forward(&last_hidden)?.squeeze(1)?;
        
        // Debug: print logits stats
        {
            let l_f32 = logits.to_dtype(candle::DType::F32)?;
            let flat = l_f32.flatten_all()?;
            let mean = flat.mean_all()?.to_scalar::<f32>()?;
            let var = flat.broadcast_sub(&flat.mean_all()?)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let std = var.sqrt();
            println!("logits: shape={:?}, mean={:.6}, std={:.6}", logits.shape().dims(), mean, std);
            
            // Print top 10 tokens
            let probs = candle_nn::ops::softmax_last_dim(&l_f32)?;
            let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = probs_vec.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("Top 10 tokens:");
            for (i, (token_id, prob)) in indexed.iter().take(10).enumerate() {
                println!("  {}. token={}, prob={:.6}", i+1, token_id, prob);
            }
        }
        
        // ignore_eos = true for first token (step 0 < min_len)
        let next_token = self.sampling_ids_debug(&logits, sampling_config, &out_tokens, true, 0)?;

        if next_token >= self.speech_token_size as u32 {
            return Ok(out_tokens);
        }
        out_tokens.push(next_token);

        let mut seqlen_offset = lm_input.dim(1)?;

        // Subsequent tokens (only first 30 for debug)
        for step in 1..30.min(max_len) {
            let next_id = Tensor::from_slice(&[*out_tokens.last().unwrap()], (1, 1), &self.device)?;
            let current_input = self.speech_embedding.forward(&next_id)?;

            let hidden = self.forward_embeds(&current_input, seqlen_offset)?;
            let logits = self.llm_decoder.forward(&hidden)?.squeeze(1)?;

            // ignore_eos = true if we haven't reached min_len
            let ignore_eos = step < min_len;
            let next_token = self.sampling_ids_debug(&logits, sampling_config, &out_tokens, ignore_eos, step)?;

            if next_token >= self.speech_token_size as u32 {
                break;
            }

            out_tokens.push(next_token);
            seqlen_offset += 1;
        }

        Ok(out_tokens)
    }

    /// Debug version of sampling_ids
    #[allow(dead_code)]
    fn sampling_ids_debug(
        &self,
        logits: &Tensor,
        config: &SamplingConfig,
        decoded_tokens: &[u32],
        ignore_eos: bool,
        step: usize,
    ) -> Result<u32> {
        let logits = if config.temperature != 1.0 {
            (logits / config.temperature as f64)?
        } else {
            logits.clone()
        };

        let logits = if config.repetition_penalty != 1.0 && !decoded_tokens.is_empty() {
            self.apply_repetition_penalty(&logits, decoded_tokens, config.repetition_penalty)?
        } else {
            logits
        };

        // Use RAS (Repetition Aware Sampling) like Python
        let top_ids = self.ras_sampling_debug(&logits, decoded_tokens, config.top_k, config.top_p, step)?;
        
        println!("  Step {}: token {} (ignore_eos={})", step, top_ids, ignore_eos);
        
        Ok(top_ids)
    }

    /// Debug version of RAS sampling
    #[allow(dead_code)]
    fn ras_sampling_debug(&self, logits: &Tensor, decoded_tokens: &[u32], top_k: usize, top_p: f32, step: usize) -> Result<u32> {
        let win_size = 10;
        let tau_r = 0.1;
        
        // First try nucleus sampling
        let top_ids = self.nucleus_sampling(logits, top_k, top_p)?;
        
        // Check repetition in recent window (matching Python: decoded_tokens[-win_size:])
        let recent_tokens = if decoded_tokens.len() > win_size {
            &decoded_tokens[decoded_tokens.len() - win_size..]
        } else {
            decoded_tokens
        };
        
        let rep_num = recent_tokens.iter().filter(|&&t| t == top_ids).count();
        let threshold = (win_size as f32 * tau_r) as usize;
        
        if step < 15 {
            println!("    RAS: nucleus_token={}, recent_tokens={:?}, rep_num={}, threshold={}", 
                     top_ids, recent_tokens, rep_num, threshold);
        }
        
        // If repetition exceeds threshold, use random sampling
        if rep_num >= threshold {
            let random_token = self.random_sampling(logits)?;
            if step < 15 {
                println!("    RAS: TRIGGERED random sampling -> {}", random_token);
            }
            return Ok(random_token);
        }
        
        Ok(top_ids)
    }

    /// Sampling with ignore_eos support (matching Python's sampling_ids)
    fn sampling_ids(
        &self,
        logits: &Tensor,
        config: &SamplingConfig,
        decoded_tokens: &[u32],
        ignore_eos: bool,
    ) -> Result<u32> {
        let logits = if config.temperature != 1.0 {
            (logits / config.temperature as f64)?
        } else {
            logits.clone()
        };

        let logits = if config.repetition_penalty != 1.0 && !decoded_tokens.is_empty() {
            self.apply_repetition_penalty(&logits, decoded_tokens, config.repetition_penalty)?
        } else {
            logits
        };

        // Retry sampling until we get a valid token (matching Python behavior)
        let max_trials = 100;
        for _ in 0..max_trials {
            // Use RAS (Repetition Aware Sampling) like Python
            let top_ids = self.ras_sampling(&logits, decoded_tokens, config.top_k, config.top_p)?;
            
            // If ignore_eos is false, or the token is a valid speech token
            if !ignore_eos || top_ids < self.speech_token_size as u32 {
                return Ok(top_ids);
            }
            // Otherwise retry
        }

        // Fallback: if we exceed max_trials, just return any token
        self.ras_sampling(&logits, decoded_tokens, config.top_k, config.top_p)
    }

    /// Repetition Aware Sampling (RAS) from VALL-E 2
    /// If a token appears too frequently in recent history, use random sampling instead
    fn ras_sampling(&self, logits: &Tensor, decoded_tokens: &[u32], top_k: usize, top_p: f32) -> Result<u32> {
        let win_size = 10;
        let tau_r = 0.1;
        
        // First try nucleus sampling
        let top_ids = self.nucleus_sampling(logits, top_k, top_p)?;
        
        // Check repetition in recent window (matching Python: decoded_tokens[-win_size:])
        let recent_tokens = if decoded_tokens.len() > win_size {
            &decoded_tokens[decoded_tokens.len() - win_size..]
        } else {
            decoded_tokens
        };
        
        let rep_num = recent_tokens.iter().filter(|&&t| t == top_ids).count();
        
        // If repetition exceeds threshold, use random sampling
        // Python: if rep_num >= win_size * tau_r (i.e., >= 1.0 when win_size=10, tau_r=0.1)
        if rep_num >= (win_size as f32 * tau_r) as usize {
            return self.random_sampling(logits);
        }
        
        Ok(top_ids)
    }

    /// Random sampling from softmax distribution
    fn random_sampling(&self, logits: &Tensor) -> Result<u32> {
        let probs = candle_nn::ops::softmax_last_dim(logits)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
        
        let random_val: f32 = rand::random();
        let mut cumulative = 0.0f32;
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumulative += prob;
            if random_val < cumulative {
                return Ok(idx as u32);
            }
        }
        
        // Fallback to last token
        Ok((probs_vec.len() - 1) as u32)
    }

    /// Nucleus sampling implementation matching Python's cosyvoice.utils.common.nucleus_sampling
    fn nucleus_sampling(&self, logits: &Tensor, top_k: usize, top_p: f32) -> Result<u32> {
        let probs = candle_nn::ops::softmax_last_dim(logits)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Collect candidates until we reach top_p or top_k
        let mut cum_prob = 0.0f32;
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        for (idx, prob) in indexed.iter() {
            if cum_prob < top_p && candidates.len() < top_k {
                cum_prob += prob;
                candidates.push((*idx, *prob));
            } else {
                break;
            }
        }

        if candidates.is_empty() {
            // Fallback to argmax if no candidates
            return Ok(indexed.first().map(|(i, _)| *i as u32).unwrap_or(0));
        }

        // Normalize probabilities for sampling
        let total_prob: f32 = candidates.iter().map(|(_, p)| p).sum();
        let normalized: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(i, p)| (*i, p / total_prob))
            .collect();

        // Multinomial sampling
        let random_val: f32 = rand::random();
        let mut cumulative = 0.0f32;
        for (idx, prob) in normalized.iter() {
            cumulative += prob;
            if random_val < cumulative {
                return Ok(*idx as u32);
            }
        }

        // Fallback to last candidate
        Ok(candidates.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }

    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        decoded_tokens: &[u32],
        penalty: f32,
    ) -> Result<Tensor> {
        let mut logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        for &token in decoded_tokens {
            let idx = token as usize;
            if idx < logits_vec.len() {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }
        Tensor::from_vec(logits_vec, logits.shape(), logits.device())
    }

    pub fn speech_token_size(&self) -> usize {
        self.speech_token_size
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.transformer_layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens() {
        assert_eq!(SOS_TOKEN, 6561);
        assert_eq!(EOS_TOKEN, 6562);
        assert_eq!(TASK_ID_TOKEN, 6563);
        assert_eq!(FILL_TOKEN, 6564);
    }
}
