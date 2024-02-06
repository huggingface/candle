#![allow(unused)]
use super::with_tracing::{layer_norm, linear_no_bias as linear, LayerNorm, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

// https://huggingface.co/RWKV/HF_v5-Eagle-7B/blob/main/configuration_rwkv5.py
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub attention_hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_size: usize,
    pub intermediate_size: Option<usize>,
    pub layer_norm_epsilon: f64,
}

#[derive(Debug, Clone)]
struct SelfAttention {
    key: Linear,
    receptance: Linear,
    value: Linear,
    gate: Linear,
    output: Linear,
    ln_x: candle_nn::GroupNorm,
}

impl SelfAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let attn_hidden_size = cfg.attention_hidden_size;
        let key = linear(hidden_size, attn_hidden_size, vb.pp("key"))?;
        let receptance = linear(hidden_size, attn_hidden_size, vb.pp("receptance"))?;
        let value = linear(hidden_size, attn_hidden_size, vb.pp("value"))?;
        let gate = linear(hidden_size, attn_hidden_size, vb.pp("gate"))?;
        let output = linear(attn_hidden_size, hidden_size, vb.pp("output"))?;
        let ln_x = candle_nn::group_norm(
            hidden_size / cfg.head_size,
            hidden_size,
            1e-5,
            vb.pp("ln_x"),
        )?;
        Ok(Self {
            key,
            value,
            receptance,
            gate,
            output,
            ln_x,
        })
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    key: Linear,
    receptance: Linear,
    value: Linear,
}

impl FeedForward {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let int_size = cfg
            .intermediate_size
            .unwrap_or(((cfg.hidden_size as f64 * 3.5) as usize) / 32 * 32);
        let key = linear(cfg.hidden_size, int_size, vb.pp("key"))?;
        let receptance = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("receptance"))?;
        let value = linear(int_size, cfg.hidden_size, vb.pp("value"))?;
        Ok(Self {
            key,
            receptance,
            value,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    pre_ln: Option<LayerNorm>,
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl Block {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln1 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln2"))?;
        let pre_ln = if layer_index == 0 {
            let ln = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("pre_ln"))?;
            Some(ln)
        } else {
            None
        };
        let attention = SelfAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embeddings"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_m.pp("blocks");
        for block_index in 0..cfg.num_hidden_layers {
            let block = Block::new(block_index, cfg, vb_b.pp(block_index))?;
            blocks.push(block)
        }
        let ln_out = layer_norm(cfg.hidden_size, 1e-5, vb_m.pp("ln_out"))?;
        let head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            blocks,
            ln_out,
            head,
        })
    }
}
