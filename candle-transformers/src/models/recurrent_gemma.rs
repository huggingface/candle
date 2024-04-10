#![allow(unused)]
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone, Copy)]
pub enum TemporalBlockType {
    Attention,
    Recurrent,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub width: usize,
    pub num_heads: usize,
    pub mlp_expanded_width: usize,
    pub lru_width: Option<usize>,
    pub embeddings_scale_by_sqrt_dim: bool,
    pub attention_window_size: usize,
    pub logits_soft_cap: Option<f32>,
    pub block_types: Vec<TemporalBlockType>,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(dim, "scale")?;
        Ok(Self { scale, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.scale + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct Embedder {
    input_embedding: Tensor,
}

impl Embedder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_embedding = vb.get((cfg.vocab_size, cfg.width), "input_embedding")?;
        Ok(Self { input_embedding })
    }
}

#[derive(Debug, Clone)]
struct MlpBlock {
    ffw_up: Linear,
    ffw_down: Linear,
}

impl MlpBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let width = cfg.width;
        let expanded_width = cfg.mlp_expanded_width;
        let ffw_up = linear(width, expanded_width, true, vb.pp("ffw_up"))?;
        let ffw_down = linear(expanded_width, width, true, vb.pp("ffw_down"))?;
        Ok(Self { ffw_up, ffw_down })
    }
}

#[derive(Debug, Clone)]
struct Conv1D {
    w: Tensor,
    b: Tensor,
}

impl Conv1D {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((4, cfg.width), "w")?;
        let b = vb.get((cfg.width,), "b")?;
        Ok(Self { w, b })
    }
}

#[derive(Debug, Clone)]
struct BlockDiagonalLinear {
    w: Tensor,
    b: Tensor,
    block_width: usize,
}

impl BlockDiagonalLinear {
    fn new(width: usize, num_blocks: usize, vb: VarBuilder) -> Result<Self> {
        let block_width = width / num_blocks;
        let w = vb.get((num_blocks, block_width, block_width), "w")?;
        let b = vb.get((num_blocks, block_width), "b")?;
        Ok(Self { w, b, block_width })
    }
}

// Real-Gated Linear Recurrent Unit
#[derive(Debug, Clone)]
struct Rglru {
    a_param: Tensor,
    input_gate: BlockDiagonalLinear,
    a_gate: BlockDiagonalLinear,
}

impl Rglru {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let width = cfg.width;
        let num_heads = cfg.num_heads;
        let input_gate = BlockDiagonalLinear::new(width, num_heads, vb.pp("input_gate"))?;
        let a_gate = BlockDiagonalLinear::new(width, num_heads, vb.pp("a_gate"))?;
        let a_param = vb.get((width,), "a_param")?;
        Ok(Self {
            input_gate,
            a_param,
            a_gate,
        })
    }
}

#[derive(Debug, Clone)]
struct RecurrentBlock {
    linear_y: Linear,
    linear_x: Linear,
    linear_out: Linear,
    conv_1d: Conv1D,
    rg_lru: Rglru,
}

impl RecurrentBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let lru_width = cfg.lru_width.unwrap_or(cfg.width);
        let linear_y = linear(cfg.width, lru_width, true, vb.pp("linear_y"))?;
        let linear_x = linear(cfg.width, lru_width, true, vb.pp("linear_x"))?;
        let linear_out = linear(lru_width, cfg.width, true, vb.pp("linear_out"))?;
        let conv_1d = Conv1D::new(cfg, vb.pp("conv_1d"))?;
        let rg_lru = Rglru::new(cfg, vb.pp("rg_lru"))?;
        Ok(Self {
            linear_y,
            linear_x,
            linear_out,
            conv_1d,
            rg_lru,
        })
    }
}

#[derive(Debug, Clone)]
struct LocalAttentionBlock {
    proj_q: Linear,
    proj_k: Linear,
    proj_v: Linear,
    proj_final: Linear,
}

impl LocalAttentionBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let w = cfg.width;
        let proj_q = linear(w, w, true, vb.pp("proj_q"))?;
        let proj_k = linear(w, w, true, vb.pp("proj_k"))?;
        let proj_v = linear(w, w, true, vb.pp("proj_v"))?;
        let proj_final = linear(w, w, false, vb.pp("proj_final"))?;
        Ok(Self {
            proj_q,
            proj_k,
            proj_v,
            proj_final,
        })
    }
}

#[derive(Debug, Clone)]
enum Inner {
    Recurrent(RecurrentBlock),
    Attention(LocalAttentionBlock),
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    temporal_pre_norm: RmsNorm,
    channel_pre_norm: RmsNorm,
    inner: Inner,
    mlp_block: MlpBlock,
}

impl ResidualBlock {
    fn new(block_type: TemporalBlockType, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let temporal_pre_norm = RmsNorm::new(cfg.width, 1e-6, vb.pp("temporal_pre_norm"))?;
        let channel_pre_norm = RmsNorm::new(cfg.width, 1e-6, vb.pp("channel_pre_norm"))?;
        let inner = match block_type {
            TemporalBlockType::Recurrent => {
                let block = RecurrentBlock::new(cfg, vb.pp("recurrent_block"))?;
                Inner::Recurrent(block)
            }
            TemporalBlockType::Attention => {
                let block = LocalAttentionBlock::new(cfg, vb.pp("attention_block"))?;
                Inner::Attention(block)
            }
        };
        let mlp_block = MlpBlock::new(cfg, vb.pp("mlp_block"))?;
        Ok(Self {
            temporal_pre_norm,
            channel_pre_norm,
            inner,
            mlp_block,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embedder: Embedder,
    blocks: Vec<ResidualBlock>,
    final_norm: RmsNorm,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embedder = Embedder::new(cfg, vb.pp("embedder"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = vec![];
        for (block_id, &block_type) in cfg.block_types.iter().enumerate() {
            let block = ResidualBlock::new(block_type, cfg, vb_b.pp(block_id))?;
            blocks.push(block)
        }
        let final_norm = RmsNorm::new(cfg.width, 1e-6, vb.pp("final_norm"))?;
        Ok(Self {
            embedder,
            blocks,
            final_norm,
        })
    }
}
