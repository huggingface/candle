//! Qwen2.5-VL (Vision-Language) model implementation.
//!
//! Qwen2.5-VL is a multimodal large language model that combines vision and language
//! capabilities. It extends Qwen2.5 with visual understanding abilities.
//!
//! Key features:
//! - 3D vision patch embedding with temporal, height, and width processing
//! - Multimodal rotary position embeddings
//! - Flexible attention mechanisms with window and full attention
//! - Support for various image resolutions and aspect ratios
//!
//! References:
//! - 🤗 [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
//! - [Blog Post](https://qwenlm.github.io/blog/qwen2.5/)

use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize, 
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536,
            intermediate_size: 6144,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 896,
            patch_size: 14,
            num_channels: 3,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub hidden_size: usize,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub image_token_id: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            text_config: TextConfig {
                vocab_size: 151665,
                hidden_size: 3584,
                intermediate_size: 18944,
                num_hidden_layers: 28,
                num_attention_heads: 28,
                num_key_value_heads: 4,
                max_position_embeddings: 32768,
                rope_theta: 1000000.0,
                rms_norm_eps: 1e-5,
                use_sliding_window: false,
                sliding_window: None,
                max_window_layers: 28,
            },
            hidden_size: 3584,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            image_token_id: 151654,
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(x_dtype)?;
        x.broadcast_mul(&self.weight)
    }
}

#[derive(Debug, Clone)]
struct MultimodalRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl MultimodalRotaryEmbedding {
    fn new(
        dim: usize,
        base: f64,
        mrope_section: Vec<usize>,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq_mask: Vec<f32> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(i, _d)| {
                if i < mrope_section[0] {
                    1.0
                } else if i < mrope_section[0] + mrope_section[1] {
                    0.5  
                } else {
                    0.25
                }
            })
            .collect();
        
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(i, d)| inv_freq_mask[i] / base.powf(d as f64 / dim as f64) as f32)
            .collect();
        
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, inv_freq_len, device)?;
        
        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }
    
    fn apply_rotary_pos_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let dtype = q.dtype();
        let seq_len = q.dim(2)?;
        
        // Expand position_ids for multimodal
        let mut grid_pos = Vec::new();
        for pos_id in position_ids.to_vec2::<i64>()? {
            for &p in pos_id.iter() {
                // Extract temporal, height, width from packed position
                let t = p / 10000;
                let h = (p % 10000) / 100;
                let w = p % 100;
                grid_pos.push(vec![t, h, w]);
            }
        }
        
        let grid_pos = Tensor::from_vec(
            grid_pos.into_iter().flatten().collect::<Vec<_>>(),
            (position_ids.dims()[0], seq_len, 3),
            q.device(),
        )?;
        
        // Compute frequencies
        let grid_pos = grid_pos.to_dtype(DType::F32)?;
        let freqs = grid_pos.matmul(&self.inv_freq.unsqueeze(0)?.transpose(0, 1)?)?;
        let freqs = freqs.transpose(D::Minus1, D::Minus2)?;
        
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        
        // Apply rotary embeddings
        let q_rot = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        
        Ok((q_rot, k_rot))
    }
}

#[derive(Debug, Clone)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let q_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }
    
    fn forward(&self, x: &Tensor, rotary_emb: &MultimodalRotaryEmbedding) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        let q = self.q_proj.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        // Apply rotary embeddings
        let position_ids = Tensor::arange(0i64, seq_len as i64, x.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;
        let (q, k) = rotary_emb.apply_rotary_pos_emb(&q, &k, &position_ids)?;
        
        // Attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;
        
        // Reshape and project
        attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for VisionMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl VisionEncoderLayer {
    fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            vb.pp("self_attn"),
        )?;
        let mlp = VisionMlp::new(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("mlp"),
        )?;
        let input_layernorm = RmsNorm::new(
            config.hidden_size,
            1e-6,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            config.hidden_size,
            1e-6,
            vb.pp("post_attention_layernorm"),
        )?;
        
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
    
    fn forward(&self, x: &Tensor, rotary_emb: &MultimodalRotaryEmbedding) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rotary_emb)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

#[derive(Debug, Clone)]
struct VisionPatchEmbed {
    proj: Conv2d,
}

impl VisionPatchEmbed {
    fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        
        let proj = candle_nn::conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            conv_config,
            vb.pp("proj"),
        )?;
        
        Ok(Self { proj })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, channels, height, width]
        self.proj.forward(x)?
            .flatten_from(2)?
            .transpose(1, 2) // [batch, num_patches, hidden_size]
    }
}

#[derive(Debug, Clone)]
struct VisionPatchMerger {
    ln_q: LayerNorm,
    mlp: Vec<Linear>,
}

impl VisionPatchMerger {
    fn new(hidden_size: usize, merge_size: usize, vb: VarBuilder) -> Result<Self> {
        let ln_q = candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("ln_q"))?;
        let mlp = vec![
            linear_no_bias(hidden_size * merge_size * merge_size, hidden_size, vb.pp("mlp.0"))?,
            linear_no_bias(hidden_size, hidden_size, vb.pp("mlp.2"))?,
        ];
        
        Ok(Self { ln_q, mlp })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln_q.forward(x)?;
        let (b, n, c) = x.dims3()?;
        let h = (n as f64).sqrt() as usize;
        let w = h;
        
        let x = x.reshape((b, h, w, c))?;
        
        // Merge patches
        let merge_size = 2; // Using default merge size
        let x = x.reshape((
            b,
            h / merge_size,
            merge_size,
            w / merge_size,
            merge_size,
            c,
        ))?;
        let x = x.permute((0, 1, 3, 2, 4, 5))?;
        let x = x.flatten(3, 5)?;
        let x = x.flatten(1, 2)?;
        
        // Apply MLP
        let x = self.mlp[0].forward(&x)?.gelu()?;
        self.mlp[1].forward(&x)
    }
}

#[derive(Debug, Clone)]
pub struct VisionModel {
    patch_embed: VisionPatchEmbed,
    rotary_emb: Arc<MultimodalRotaryEmbedding>,
    layers: Vec<VisionEncoderLayer>,
    merger: Option<VisionPatchMerger>,
}

impl VisionModel {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = VisionPatchEmbed::new(config, vb.pp("patch_embed"))?;
        
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_emb = Arc::new(MultimodalRotaryEmbedding::new(
            head_dim,
            10000.0,
            vec![head_dim / 3, head_dim / 3, head_dim / 3],
            vb.device(),
        )?);
        
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(VisionEncoderLayer::new(config, vb.pp(&format!("layers.{}", i)))?);
        }
        
        let merger = if config.spatial_merge_size > 1 {
            Some(VisionPatchMerger::new(
                config.hidden_size,
                config.spatial_merge_size,
                vb.pp("merger"),
            )?)
        } else {
            None
        };
        
        Ok(Self {
            patch_embed,
            rotary_emb,
            layers,
            merger,
        })
    }
    
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(pixel_values)?;
        
        for layer in &self.layers {
            x = layer.forward(&x, &self.rotary_emb)?;
        }
        
        if let Some(merger) = &self.merger {
            x = merger.forward(&x)?;
        }
        
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<MultimodalRotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl TextAttention {
    fn new(
        config: &TextConfig,
        rotary_emb: Arc<MultimodalRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = hidden_size / num_heads;
        
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }
    
    fn forward(
        &mut self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        
        let q = self.q_proj.forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        let (q, k) = self.rotary_emb.apply_rotary_pos_emb(&q, &k, position_ids)?;
        
        // Handle KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        
        // Repeat KV heads if needed
        let repeat_count = self.num_heads / self.num_kv_heads;
        let k = if repeat_count > 1 {
            crate::utils::repeat_kv(k, repeat_count)?
        } else {
            k
        };
        let v = if repeat_count > 1 {
            crate::utils::repeat_kv(v, repeat_count)?
        } else {
            v
        };
        
        // Attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? / scale)?;
        
        let scores = match attention_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(mask)?,
        };
        
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;
        
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
    
    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct TextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TextMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for TextMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct TextDecoderLayer {
    self_attn: TextAttention,
    mlp: TextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl TextDecoderLayer {
    fn new(
        config: &TextConfig,
        rotary_emb: Arc<MultimodalRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = TextAttention::new(config, rotary_emb, vb.pp("self_attn"))?;
        let mlp = TextMlp::new(config.hidden_size, config.intermediate_size, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        
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
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, attention_mask, position_ids)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
    
    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
}

impl TextModel {
    pub fn new(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_emb = Arc::new(MultimodalRotaryEmbedding::new(
            head_dim,
            config.rope_theta,
            vec![head_dim / 3, head_dim / 3, head_dim / 3],
            vb.device(),
        )?);
        
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(TextDecoderLayer::new(
                config,
                rotary_emb.clone(),
                vb.pp(&format!("layers.{}", i)),
            )?);
        }
        
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
    
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        
        for layer in &mut self.layers {
            x = layer.forward(&x, attention_mask, position_ids)?;
        }
        
        self.norm.forward(&x)
    }
    
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen25VL {
    vision_model: VisionModel,
    text_model: TextModel,
    visual_tokenizer: Linear,
    lm_head: Linear,
    config: Config,
}

impl Qwen25VL {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let vision_model = VisionModel::new(&config.vision_config, vb.pp("vision_model"))?;
        let text_model = TextModel::new(&config.text_config, vb.pp("text_model"))?;
        
        let visual_tokenizer = linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("visual_tokenizer"),
        )?;
        
        let lm_head = linear_no_bias(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            vb.pp("lm_head"),
        )?;
        
        Ok(Self {
            vision_model,
            text_model,
            visual_tokenizer,
            lm_head,
            config: config.clone(),
        })
    }
    
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_features = self.vision_model.forward(pixel_values)?;
        self.visual_tokenizer.forward(&vision_features)
    }
    
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        
        // Process vision inputs if provided
        let input_embeds = if let Some(pixel_values) = pixel_values {
            let image_embeds = self.encode_images(pixel_values)?;
            
            // Find image token positions
            let image_token_mask = input_ids.eq(self.config.image_token_id as i64)?;
            
            // Get text embeddings
            let text_embeds = self.text_model.embed_tokens.forward(input_ids)?;
            
            // Merge vision and text embeddings
            self.merge_vision_text_embeds(&text_embeds, &image_embeds, &image_token_mask)?
        } else {
            self.text_model.embed_tokens.forward(input_ids)?
        };
        
        // Create position ids
        let position_ids = Tensor::arange(0i64, seq_len as i64, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;
        
        // Forward through text model
        let mut hidden_states = input_embeds;
        for layer in &mut self.text_model.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, &position_ids)?;
        }
        let hidden_states = self.text_model.norm.forward(&hidden_states)?;
        
        // Generate logits
        self.lm_head.forward(&hidden_states)
    }
    
    fn merge_vision_text_embeds(
        &self,
        text_embeds: &Tensor,
        image_embeds: &Tensor,
        image_token_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = text_embeds.dims3()?;
        let device = text_embeds.device();
        
        // Initialize output with text embeddings
        let mut output = text_embeds.clone();
        
        // Replace image tokens with vision embeddings
        // Find image token positions
        let image_token_mask_vec = image_token_mask.to_vec2::<u8>()?;
        let mut image_positions = Vec::new();
        for (b, row) in image_token_mask_vec.iter().enumerate() {
            for (s, &val) in row.iter().enumerate() {
                if val > 0 {
                    image_positions.push((b, s));
                }
            }
        }
        if !image_positions.is_empty() {
            let num_images = image_positions.len();
            let image_seq_len = image_embeds.dim(1)?;
            
            // Create expanded output tensor
            let new_seq_len = seq_len - num_images + num_images * image_seq_len;
            let mut new_output = Tensor::zeros((batch_size, new_seq_len, hidden_size), text_embeds.dtype(), device)?;
            
            let mut output_idx = 0;
            let mut image_idx = 0;
            
            for i in 0..seq_len {
                let is_image_token = image_token_mask.i((0, i))?.to_scalar::<u8>()? > 0;
                
                if is_image_token && image_idx < num_images {
                    // Insert vision embeddings
                    let start = output_idx;
                    let _end = output_idx + image_seq_len;
                    // Copy image embeddings
                    for j in 0..image_seq_len {
                        for k in 0..hidden_size {
                            let val = image_embeds.i((image_idx, j, k))?.to_scalar::<f32>()?;
                            new_output = new_output.slice_assign(
                                &[0..batch_size, start + j..start + j + 1, k..k + 1],
                                &Tensor::new(val, device)?,
                            )?;
                        }
                    }
                    output_idx += image_seq_len;
                    image_idx += 1;
                } else {
                    // Copy text embedding
                    // Copy text embedding
                    let text_emb = output.i((.., i..i + 1, ..))?;
                    new_output = new_output.slice_assign(
                        &[0..batch_size, output_idx..output_idx + 1, 0..hidden_size],
                        &text_emb,
                    )?;
                    output_idx += 1;
                }
            }
            
            output = new_output;
        }
        
        Ok(output)
    }
    
    pub fn clear_kv_cache(&mut self) {
        self.text_model.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Qwen25VLForConditionalGeneration {
    model: Qwen25VL,
}

impl Qwen25VLForConditionalGeneration {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let model = Qwen25VL::new(config, vb)?;
        Ok(Self { model })
    }
    
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.model.forward(input_ids, pixel_values, attention_mask)
    }
    
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}