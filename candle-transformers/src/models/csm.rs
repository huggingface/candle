#![allow(unused)]
//! Implementation of the Conversational Speech Model (CSM) from Sesame
//!
//! See: [CSM](Conversational Speech Model)
//!
/// CSM (Conversational Speech Model) is a speech generation model from Sesame that generates RVQ
/// audio codes from text and audio inputs. The model architecture employs a Llama backbone and a
/// smaller audio decoder that produces Mimi audio codes.
///
use crate::generation::LogitsProcessor;
use crate::models::encodec;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_b, Embedding, Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

#[derive(serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    #[serde(rename = "llama-1B")]
    Llama1B,
    #[serde(rename = "llama-100M")]
    Llama100M,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub audio_num_codebooks: usize,
    pub audio_vocab_size: usize,
    pub backbone_flavor: Flavor,
    pub decoder_flavor: Flavor,
    pub text_vocab_size: usize,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    vocab_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    embed_dim: usize,
    max_seq_len: usize,
    intermediate_dim: usize,
    norm_eps: f64,
    rope_base: f64,
    scale_factor: usize,
}

impl LlamaConfig {
    pub fn from_flavor(flavor: Flavor) -> Self {
        match flavor {
            Flavor::Llama1B => Self {
                vocab_size: 128256,
                num_layers: 16,
                num_heads: 32,
                num_kv_heads: 8,
                embed_dim: 2048,
                max_seq_len: 2048,
                intermediate_dim: 8192,
                norm_eps: 1e-5,
                rope_base: 500_000.,
                scale_factor: 32,
            },
            Flavor::Llama100M => Self {
                vocab_size: 128256,
                num_layers: 4,
                num_heads: 8,
                num_kv_heads: 2,
                embed_dim: 1024,
                max_seq_len: 2048,
                intermediate_dim: 8192,
                norm_eps: 1e-5,
                rope_base: 500_000.,
                scale_factor: 32,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &LlamaConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.embed_dim / cfg.num_heads;
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_base.powf(i as f64 / dim as f64) as f32)
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
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}
fn rms_norm(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get((hidden_size,), "scale")?;
    Ok(RmsNorm::new(weight, eps))
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    num_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
}

impl Attention {
    fn new(cfg: &LlamaConfig, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.embed_dim / cfg.num_heads;
        let kv_dim = cfg.num_kv_heads * head_dim;

        let q_proj = linear_b(cfg.embed_dim, cfg.embed_dim, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(cfg.embed_dim, kv_dim, false, vb.pp("k_proj"))?;
        let v_proj = linear_b(cfg.embed_dim, kv_dim, false, vb.pp("v_proj"))?;
        let o_proj = linear_b(cfg.embed_dim, cfg.embed_dim, false, vb.pp("output_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            kv_cache: None,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            num_kv_groups: cfg.num_heads / cfg.num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = crate::utils::repeat_kv(value_states, self.num_kv_groups)?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_b(cfg.embed_dim, cfg.intermediate_dim, false, vb.pp("w1"))?;
        let w2 = linear_b(cfg.intermediate_dim, cfg.embed_dim, false, vb.pp("w2"))?;
        let w3 = linear_b(cfg.embed_dim, cfg.intermediate_dim, false, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.w1)?.silu()?;
        let rhs = xs.apply(&self.w3)?;
        (lhs * rhs)?.apply(&self.w2)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    mlp_norm: RmsNorm,
    sa_norm: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl Layer {
    fn new(cfg: &LlamaConfig, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let mlp_norm = rms_norm(cfg.embed_dim, cfg.norm_eps, vb.pp("mlp_norm"))?;
        let sa_norm = rms_norm(cfg.embed_dim, cfg.norm_eps, vb.pp("sa_norm"))?;
        let attn = Attention::new(cfg, rotary_emb, vb.pp("attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            mlp_norm,
            sa_norm,
            attn,
            mlp,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.sa_norm.forward(xs)?;
        let xs = self.attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct LlamaModel {
    layers: Vec<Layer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl LlamaModel {
    pub fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_layers {
            let layer = Layer::new(cfg, rotary_emb.clone(), vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = rms_norm(cfg.embed_dim, cfg.norm_eps, vb.pp("norm"))?;
        Ok(Self {
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    fn prepare_decoder_attention_mask(
        &self,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((1, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len, _embed_dim) = xs.dims3()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        let ys = xs.narrow(1, seq_len - 1, 1)?.apply(&self.norm)?;
        Ok(ys)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    backbone: LlamaModel,
    decoder: LlamaModel,
    codebook0_head: Linear,
    audio_embeddings: Embedding,
    text_embeddings: Embedding,
    projection: Linear,
    audio_head: Tensor,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let backbone_cfg = LlamaConfig::from_flavor(cfg.backbone_flavor);
        let backbone = LlamaModel::new(&backbone_cfg, vb.pp("backbone"))?;
        let decoder_cfg = LlamaConfig::from_flavor(cfg.decoder_flavor);
        let decoder = LlamaModel::new(&decoder_cfg, vb.pp("decoder"))?;
        let backbone_dim = backbone_cfg.embed_dim;
        let decoder_dim = decoder_cfg.embed_dim;
        let audio_embeddings = embedding(
            cfg.audio_vocab_size * cfg.audio_num_codebooks,
            backbone_dim,
            vb.pp("audio_embeddings"),
        )?;
        let text_embeddings =
            embedding(cfg.text_vocab_size, backbone_dim, vb.pp("text_embeddings"))?;
        let projection = linear_b(backbone_dim, decoder_dim, false, vb.pp("projection"))?;
        let codebook0_head = linear_b(
            backbone_dim,
            cfg.audio_vocab_size,
            false,
            vb.pp("codebook0_head"),
        )?;
        let audio_head = vb.get(
            (
                cfg.audio_num_codebooks - 1,
                decoder_dim,
                cfg.audio_vocab_size,
            ),
            "audio_head",
        )?;
        Ok(Self {
            backbone,
            decoder,
            codebook0_head,
            audio_embeddings,
            text_embeddings,
            projection,
            audio_head,
            config: cfg.clone(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.backbone.clear_kv_cache();
        self.decoder.clear_kv_cache();
    }

    pub fn generate_frame(
        &mut self,
        tokens: &Tensor,
        tokens_mask: &Tensor,
        input_pos: usize,
        lp: &mut LogitsProcessor,
    ) -> Result<Vec<u32>> {
        let h = tokens.clone(); // TODO
        let h = self.backbone.forward(&h, input_pos)?;
        let c0_logits = h.apply(&self.codebook0_head)?;
        let c0_sample = lp.sample(&c0_logits)?;
        let mut all_samples = vec![c0_sample];
        let c0_sample = Tensor::from_slice(&[c0_sample], (1, 1), &self.decoder.device)?;
        let c0_embed = self.audio_embeddings.forward(&c0_sample)?;
        let mut curr_h = Tensor::cat(&[h, c0_embed], 1)?;

        self.decoder.clear_kv_cache();
        for i in 0..(self.config.audio_num_codebooks - 1) {
            let proj_h = curr_h.apply(&self.projection)?;
            let decoder_h = self.decoder.forward(&proj_h, i)?;
            let ci_logits = decoder_h.matmul(&self.audio_head.get(i)?)?;
            let ci_sample = lp.sample(&ci_logits)?;
            all_samples.push(ci_sample);
            let ci_sample = Tensor::from_slice(&[ci_sample], (1, 1), &self.decoder.device)?;
            let ci_embed = self.audio_embeddings.forward(&ci_sample)?;
            curr_h = ci_embed
        }
        Ok(all_samples)
    }
}
