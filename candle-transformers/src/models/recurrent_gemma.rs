#![allow(unused)]
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum TemporalBlockType {
    Attention,
    Recurrent,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub lru_width: Option<usize>,
    pub attention_window_size: usize,
    pub conv1d_width: usize,
    pub logits_soft_cap: f64,
    pub hidden_activation: candle_nn::Activation,
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(alias = "_block_types")]
    pub block_types: Vec<TemporalBlockType>,
    pub attention_bias: bool,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
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
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size / 2;
        let gate_proj = linear(h, intermediate_size, true, vb.pp("gate_proj"))?;
        let up_proj = linear(h, intermediate_size, true, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, h, true, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        (gate * xs.apply(&self.up_proj))?.apply(&self.down_proj)
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
    recurrent_param: Tensor,
    input_gate_weight: Tensor,
    input_gate_bias: Tensor,
    recurrent_gate_weight: Tensor,
    recurrent_gate_bias: Tensor,
}

impl Rglru {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let lru_width = cfg.lru_width.unwrap_or(h);
        let n_heads = cfg.num_attention_heads;
        let block_width = lru_width / n_heads;
        let recurrent_param = vb.get((lru_width,), "recurrent_param")?;
        let input_gate_weight = vb.get((n_heads, block_width, block_width), "input_gate_weight")?;
        let input_gate_bias = vb.get((n_heads, block_width), "input_gate_bias")?;
        let recurrent_gate_weight =
            vb.get((n_heads, block_width, block_width), "recurrent_gate_weight")?;
        let recurrent_gate_bias = vb.get((n_heads, block_width), "recurrent_gate_bias")?;
        Ok(Self {
            recurrent_param,
            input_gate_bias,
            input_gate_weight,
            recurrent_gate_bias,
            recurrent_gate_weight,
        })
    }
}

#[derive(Debug, Clone)]
struct RecurrentBlock {
    linear_y: Linear,
    linear_x: Linear,
    linear_out: Linear,
    conv_1d: candle_nn::Conv1d,
    rg_lru: Rglru,
    act_fn: candle_nn::Activation,
}

impl RecurrentBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let lru_width = cfg.lru_width.unwrap_or(h);
        let linear_y = linear(h, lru_width, true, vb.pp("linear_y"))?;
        let linear_x = linear(h, lru_width, true, vb.pp("linear_x"))?;
        let linear_out = linear(lru_width, h, true, vb.pp("linear_out"))?;
        let conv_1d = candle_nn::conv1d(
            lru_width,
            lru_width,
            cfg.conv1d_width,
            candle_nn::Conv1dConfig {
                groups: lru_width,
                padding: cfg.conv1d_width - 1,
                ..Default::default()
            },
            vb.pp("conv_1d"),
        )?;
        let rg_lru = Rglru::new(cfg, vb.pp("rg_lru"))?;
        Ok(Self {
            linear_y,
            linear_x,
            linear_out,
            conv_1d,
            rg_lru,
            act_fn: cfg.hidden_activation,
        })
    }

    pub fn forward(&self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct SdpaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl SdpaAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_proj = linear(h, n_heads * hd, cfg.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear(h, n_kv_heads * hd, cfg.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear(h, n_kv_heads * hd, cfg.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear(n_heads * hd, h, true, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim: hd,
            hidden_size: h,
        })
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    pub fn forward(&self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let (bsz, q_len, _) = xs.dims3()?;

        let query_states = xs.apply(&self.q_proj)?;
        let key_states = xs.apply(&self.k_proj)?;
        let value_states = xs.apply(&self.v_proj)?;

        let query_states = query_states
            .reshape((bsz, q_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((bsz, q_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((bsz, q_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        // TODO: rope
        // TODO: kv-cache
        let key_states = self.repeat_kv(key_states)?;
        let value_states = self.repeat_kv(value_states)?;
        let xs = {
            let att = (query_states.matmul(&key_states.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if q_len == 1 {
                att
            } else {
                // TODO: attn mask
                todo!()
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&value_states.contiguous()?)?
        };

        let xs = xs
            .transpose(1, 2)?
            .reshape((bsz, q_len, self.hidden_size))?;
        self.o_proj.forward(&xs)
    }
}

#[derive(Debug, Clone)]
enum TemporalBlock {
    Recurrent(RecurrentBlock),
    Attention(SdpaAttention),
}

impl TemporalBlock {
    pub fn forward(&self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        match self {
            Self::Recurrent(b) => b.forward(xs, pos),
            Self::Attention(b) => b.forward(xs, pos),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    temporal_pre_norm: RmsNorm,
    channel_pre_norm: RmsNorm,
    temporal_block: TemporalBlock,
    mlp_block: Mlp,
}

impl DecoderLayer {
    fn new(block_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let temporal_pre_norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("temporal_pre_norm"))?;
        let channel_pre_norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("channel_pre_norm"))?;
        let temporal_block = match cfg.block_types[block_idx % cfg.block_types.len()] {
            TemporalBlockType::Recurrent => {
                let block = RecurrentBlock::new(cfg, vb.pp("temporal_block"))?;
                TemporalBlock::Recurrent(block)
            }
            TemporalBlockType::Attention => {
                let block = SdpaAttention::new(cfg, vb.pp("temporal_block"))?;
                TemporalBlock::Attention(block)
            }
        };
        let mlp_block = Mlp::new(cfg, vb.pp("mlp_block"))?;
        Ok(Self {
            temporal_pre_norm,
            channel_pre_norm,
            temporal_block,
            mlp_block,
        })
    }

    pub fn forward(&self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.temporal_pre_norm)?;
        let xs = self.temporal_block.forward(&xs, pos)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.channel_pre_norm)?.apply(&self.mlp_block)?;
        xs + residual
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let vb_b = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(idx, cfg, vb_b.pp(idx))?;
            layers.push(layer)
        }
        let final_norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("final_norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let mut xs = xs.apply(&self.embed_tokens)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, pos)?
        }
        xs.apply(&self.final_norm)
    }
}
