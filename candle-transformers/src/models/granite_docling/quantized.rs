//! Quantized Granite-Docling model loading from GGUF weights.
//!
//! Loads two GGUF files:
//! - Text decoder GGUF (llama architecture, `blk.*` tensor names)
//! - Vision mmproj GGUF (`v.blk.*` + `mm.model.fc.*` tensor names)

use super::config::{QuantizedVisionConfig, TextConfig};
use crate::models::quantized_siglip;
use crate::quantized_nn::{self, Embedding, Linear, RmsNorm};
use crate::quantized_var_builder::VarBuilder;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};

// ---------------------------------------------------------------------------
// Pixel Shuffle Connector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Connector {
    modality_projection: Linear,
    scale_factor: usize,
}

impl Connector {
    fn new(cfg: &QuantizedVisionConfig, vb: VarBuilder) -> Result<Self> {
        let input_dim = cfg.connector_input_dim();
        let output_dim = cfg.projection_dim;
        let modality_projection =
            quantized_nn::linear_no_bias(input_dim, output_dim, vb.pp("model.fc"))?;
        Ok(Self {
            modality_projection,
            scale_factor: cfg.scale_factor,
        })
    }

    fn pixel_shuffle(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, dim) = xs.dims3()?;
        let s = self.scale_factor;
        let h = (seq_len as f64).sqrt() as usize;
        let w = h;

        let xs = xs.reshape((b, h, w, dim))?;
        let xs = xs.reshape((b, h, w / s, dim * s))?;
        let xs = xs.permute((0, 2, 1, 3))?;
        let xs = xs.reshape((b, h / s, w / s, dim * s * s))?;
        xs.reshape((b, (h / s) * (w / s), dim * s * s))
    }
}

impl Module for Connector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.pixel_shuffle(xs)?;
        xs.apply(&self.modality_projection)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let max_seq = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;
        let positions =
            Tensor::arange(0u32, max_seq as u32, dev)?.to_dtype(DType::F32)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        // GGUF weights are permuted for llama.cpp's interleaved RoPE convention
        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (self.k.as_ref(), self.v.as_ref()) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, k], 2)?;
                let v = Tensor::cat(&[prev_v, v], 2)?;
                (k, v)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    fn current_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dim(2).unwrap_or(0))
    }

    fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ---------------------------------------------------------------------------
// Quantized Attention (GQA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: KvCache,
}

impl Attention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        let q_proj = quantized_nn::linear_no_bias(h, num_heads * head_dim, vb.pp("attn_q"))?;
        let k_proj = quantized_nn::linear_no_bias(h, num_kv_heads * head_dim, vb.pp("attn_k"))?;
        let v_proj = quantized_nn::linear_no_bias(h, num_kv_heads * head_dim, vb.pp("attn_v"))?;
        let o_proj =
            quantized_nn::linear_no_bias(num_heads * head_dim, h, vb.pp("attn_output"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: KvCache::new(),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let offset = self.kv_cache.current_len();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA: repeat KV heads
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.t()?)? / scale;
        let attn = match attention_mask {
            Some(mask) => attn?.broadcast_add(mask)?,
            None => attn?,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        out.transpose(1, 2)?
            .reshape((b, seq_len, ()))?
            .apply(&self.o_proj)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))
    }
}

// ---------------------------------------------------------------------------
// Quantized MLP (SiLU-gated)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = quantized_nn::linear_no_bias(h, i, vb.pp("ffn_gate"))?;
        let up_proj = quantized_nn::linear_no_bias(h, i, vb.pp("ffn_up"))?;
        let down_proj = quantized_nn::linear_no_bias(i, h, vb.pp("ffn_down"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(xs)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.clone())?;
        let mlp = Mlp::new(cfg, vb.clone())?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rotary, attention_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ---------------------------------------------------------------------------
// Quantized Text Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    dtype: DType,
}

impl TextModel {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let dtype = DType::F32; // quantized models compute in f32
        let embed_tokens = Embedding::new(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embd"))?;

        let vb_layers = vb.pp("blk");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("output_norm"))?;
        let rotary = RotaryEmbedding::new(cfg, dtype, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            dtype,
        })
    }

    fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    fn causal_mask(&self, seq_len: usize, past_kv_len: usize, device: &Device) -> Result<Tensor> {
        let total_len = past_kv_len + seq_len;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j > past_kv_len + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?.to_dtype(self.dtype)
    }

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        self.forward_embeds(&xs)
    }

    fn forward_embeds(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let past_kv_len = self.layers[0].self_attn.kv_cache.current_len();

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, past_kv_len, xs.device())?)
        };

        let mut hidden = xs.clone();
        for layer in self.layers.iter_mut() {
            hidden = layer.forward(&hidden, &self.rotary, mask.as_ref())?;
        }

        let hidden = self.norm.forward(&hidden)?;

        // Tied embeddings
        let w = self.embed_tokens.embeddings();
        hidden.broadcast_matmul(&w.t()?)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.self_attn.kv_cache.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level Quantized Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Model {
    vision_model: quantized_siglip::VisionModel,
    connector: Connector,
    text_model: TextModel,
    image_token_id: u32,
}

impl Model {
    /// Load from two GGUF files: vision mmproj + text decoder.
    pub fn new(
        vision_vb: VarBuilder,
        vision_cfg: &QuantizedVisionConfig,
        text_vb: VarBuilder,
        text_cfg: &TextConfig,
        image_token_id: u32,
    ) -> Result<Self> {
        let vision_model = quantized_siglip::VisionModel::new(
            vision_cfg.hidden_size,
            vision_cfg.intermediate_size,
            vision_cfg.num_hidden_layers,
            vision_cfg.num_attention_heads,
            vision_cfg.image_size,
            vision_cfg.patch_size,
            vision_cfg.layer_norm_eps,
            vision_vb.pp("v"),
        )?;

        let connector = Connector::new(vision_cfg, vision_vb.pp("mm"))?;
        let text_model = TextModel::new(text_cfg, text_vb)?;

        Ok(Self {
            vision_model,
            connector,
            text_model,
            image_token_id,
        })
    }

    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_out = self.vision_model.forward(pixel_values)?;
        let connected = self.connector.forward(&vision_out)?;
        let (n, seq, hidden) = connected.dims3()?;
        connected.reshape((1, n * seq, hidden))
    }

    pub fn setup(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.clear_kv_cache();

        let image_features = self.encode_image(pixel_values)?;
        let text_embeds = self.text_model.embed_tokens().forward(input_ids)?;

        let input_embeds =
            self.merge_image_tokens(&text_embeds, &image_features, input_ids)?;

        self.text_model.forward_embeds(&input_embeds)
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.forward(input_ids)
    }

    fn merge_image_tokens(
        &self,
        text_embeds: &Tensor,
        image_features: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden) = text_embeds.dims3()?;
        let image_seq_len = image_features.dim(1)?;
        let input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let text_embeds = text_embeds.to_dtype(image_features.dtype())?;

        let mut batch_results = Vec::with_capacity(b);
        for batch_idx in 0..b {
            let mut img_idx = 0;
            let mut tokens = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                if input_ids_vec[batch_idx * seq_len + pos] == self.image_token_id
                    && img_idx < image_seq_len
                {
                    tokens.push(image_features.i((batch_idx, img_idx))?.unsqueeze(0)?);
                    img_idx += 1;
                } else {
                    tokens.push(text_embeds.i((batch_idx, pos))?.unsqueeze(0)?);
                }
            }
            let batch_embeds = Tensor::cat(&tokens, 0)?.unsqueeze(0)?;
            batch_results.push(batch_embeds);
        }
        Tensor::cat(&batch_results, 0)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text_model.clear_kv_cache();
    }
}
