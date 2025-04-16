//! Gemma 3 model implementation with quantization support.
//!
//! Gemma 3 is a family of multimodal language models developed by Google.
//! This implementation provides quantization for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group-Query Attention (GQA) with specialized key-value heads
//! - RMSNorm for layer normalization
//! - Specialized attention patterns with separate normalization for Q/K/V
//! - Feed-forward network with SwiGLU activation
//! - Support for 2/3/4/8-bit quantization
//!
//! References:
//! - [Gemma 3 Models](https://blog.google/technology/developers/gemma-3/)
//!

use std::collections::HashMap;

use crate::quantized_nn::RmsNorm;
use candle::quantized::QTensor;
use candle::quantized::{ggml_file, gguf_file};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 131072; // Gemma 3 supports 128K context window

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul, // ffn_gate in GGUF
    feed_forward_up: QMatMul,   // ffn_up in GGUF
    feed_forward_down: QMatMul, // ffn_down in GGUF
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    // Attention components
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    // Specialized normalization for Q and K
    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,
    
    // Layer normalization
    attention_norm: RmsNorm,        // Applied before attention
    post_attention_norm: RmsNorm,   // Applied after attention
    ffn_norm: RmsNorm,              // Applied before feedforward
    post_ffn_norm: RmsNorm,         // Applied after feedforward
    
    // Feed-forward network
    mlp: Mlp,
    
    // Attention parameters
    n_head: usize,       // Number of query heads
    n_kv_head: usize,    // Number of key-value heads
    head_dim: usize,     // Dimension of each head
    q_dim: usize,        // Total dimension for queries
    kv_dim: usize,       // Total dimension for keys/values
    
    // Rotary embedding
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    
    // Cache
    kv_cache: Option<(Tensor, Tensor)>,
    
    // Tracing
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        
        // Apply normalization for query and compute query projection
        let norm_x = self.attention_norm.forward(x)?;
        let q = self.attention_wq.forward(&norm_x)?;
        
        // Apply normalization for key/value and compute key/value projections
        let k = self.attention_wk.forward(&norm_x)?;
        let v = self.attention_wv.forward(&norm_x)?;
        
        // For Gemma3, q has shape [b_sz, seq_len, 2048]
        // Reshape into multiple heads, each with head_dim=256
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        
        // For Gemma3, k has shape [b_sz, seq_len, 1024]
        // Reshape into kv_head heads, each with head_dim=256
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        
        // For Gemma3, v has shape [b_sz, seq_len, 1024]
        // Reshape into kv_head heads, each with head_dim=256
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
            
        // Apply per-head normalization to q and k
        // This step requires reshaping to apply the norm to each head separately
        let q_reshaped = q.reshape((b_sz * self.n_head * seq_len, self.head_dim))?;
        let q_normed = self.attention_q_norm.forward(&q_reshaped)?;
        let q = q_normed.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        
        let k_reshaped = k.reshape((b_sz * self.n_kv_head * seq_len, self.head_dim))?;
        let k_normed = self.attention_k_norm.forward(&k_reshaped)?;
        let k = k_normed.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

        // Apply rotary positional embeddings
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // Handle KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Support for GQA (Grouped Query Attention)
        let k = crate::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = crate::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Compute attention scores
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        
        let y = att.matmul(&v.contiguous()?)?;
        
        let y = y.transpose(1, 2)?; // Now [b_sz, seq_len, n_head, head_dim]
        
        let y = y.reshape(&[b_sz, seq_len, self.q_dim])?;
        
        // Apply output projection
        let y = self.attention_wo.forward(&y)?;
        
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata
        let head_count = md_get("gemma3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("gemma3.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("gemma3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("gemma3.embedding_length")?.to_u32()? as usize;
        let key_length = md_get("gemma3.attention.key_length")?.to_u32()? as usize;
        let value_length = md_get("gemma3.attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("gemma3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        
        let rope_freq_base = md_get("gemma3.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(1000000f32);
        
        // Compute the dimensions for queries, keys, and values
        // These are the total dimensions when projected across all heads
        let q_dim = head_count * key_length;
        let kv_dim = head_count_kv * key_length;

        // Precompute rotary embeddings
        let (cos, sin) = precomput_freqs_cis(key_length, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings and output projection
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?, // Use tied weights if output.weight doesn't exist
        };

        // Build the layers
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            
            // Load attention weights
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            
            // Load normalization weights
            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            
            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            
            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            
            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_attention_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            
            // Load feedforward network weights
            let feed_forward_gate = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            
            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };
            
            // Tracing spans
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm: attention_q_norm,
                attention_k_norm: attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                kv_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;

        println!("{} {}", &x, &index_pos);
        
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        
        // Embedding
        let mut layer_in = self.tok_embeddings.forward(x)?;
        
        // Process through each layer
        for (i, layer) in self.layers.iter_mut().enumerate() {
            
            // Attention block
            let residual = &layer_in;
            let attn = layer.forward_attn(&layer_in, mask.as_ref(), index_pos)?;
            
            let x = (attn + residual)?;
            
            // Post attention normalization
            let x = layer.post_attention_norm.forward(&x)?;
            
            // Feed-forward block
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            
            // Post feed-forward normalization and residual connection
            let x = (x + residual)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            
            layer_in = x;
        }

        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        
        let _enter = self.span_output.enter();
        let output = self.output.forward(&x)?;

        Ok(output)
    }
}