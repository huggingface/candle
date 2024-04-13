// This implementation is based on the python version from huggingface/transformers.
// https://github.com/huggingface/transformers/blob/b109257f4fb8b1166e7c53cc5418632014ed53a5/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L2
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, Linear, VarBuilder};
use std::sync::Arc;

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
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
}

fn default_max_seq_len() -> usize {
    8192
}

#[derive(Debug, Clone)]
pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub(crate) fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub(crate) fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
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
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

impl RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        if cfg.partial_rotary_factor != 0.5 {
            candle::bail!("partial-rotary-factor {} <> 0.5", cfg.partial_rotary_factor)
        }
        let dim = cfg.head_dim / 2;
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub(crate) fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
        let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin))?;
        let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin))?;
        Ok((q_embed, k_embed))
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

// Real-Gated Linear Recurrent Unit
#[derive(Debug, Clone)]
pub(crate) struct Rglru {
    pub(crate) recurrent_param: Tensor,
    pub(crate) input_gate_weight: Tensor,
    pub(crate) input_gate_bias: Tensor,
    pub(crate) recurrent_gate_weight: Tensor,
    pub(crate) recurrent_gate_bias: Tensor,
    pub(crate) block_width: usize,
    pub(crate) n_heads: usize,
    pub(crate) recurrent_states: Option<Tensor>,
}

fn baddbmm(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.broadcast_add(&b.matmul(c)?)
}

fn softplus(xs: &Tensor) -> Result<Tensor> {
    (xs.exp()? + 1.0)?.log()
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
            block_width,
            n_heads,
            recurrent_states: None,
        })
    }

    // https://github.com/huggingface/transformers/blob/0bd58f1ce0573c0e3269de4215a17d318add49b9/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L303
    pub(crate) fn forward(&mut self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, lru_width) = xs.dims3()?;
        let pos = Tensor::arange(pos as u32, (pos + seq_len) as u32, xs.device())?;
        let reset = pos.eq(0u32)?.unsqueeze(1)?.unsqueeze(0)?;
        let reshape_act = xs
            .reshape((b_sz * seq_len, self.n_heads, self.block_width))?
            .permute((1, 0, 2))?
            .contiguous()?;

        let res = baddbmm(
            &self.input_gate_bias.unsqueeze(1)?,
            &reshape_act,
            &self.input_gate_weight,
        )?;
        let input_gate = res.transpose(0, 1)?.reshape((b_sz, seq_len, lru_width))?;
        let input_gate = candle_nn::ops::sigmoid(&input_gate)?;
        let res = baddbmm(
            &self.recurrent_gate_bias.unsqueeze(1)?,
            &reshape_act,
            &self.recurrent_gate_weight,
        )?;
        let recurrent_gate = res.transpose(0, 1)?.reshape((b_sz, seq_len, lru_width))?;
        let recurrent_gate = candle_nn::ops::sigmoid(&recurrent_gate)?;

        let log_recurrent_gate =
            (recurrent_gate * (-8.0))?.broadcast_mul(&softplus(&self.recurrent_param)?)?;
        let recurrent_gate = log_recurrent_gate.exp()?;
        let a_square = (log_recurrent_gate * 2.)?.exp()?;

        // Gate the input.
        let gated_inputs = (xs * input_gate)?;

        let reset = reset.to_dtype(a_square.dtype())?;
        let multiplier =
            reset.broadcast_add(&((1.0 - &reset)?.broadcast_mul(&(1.0 - a_square)?.sqrt()?))?)?;
        let normalized_x = (gated_inputs * multiplier.to_dtype(xs.dtype()))?;

        let (hidden_states, recurrent_states) = rnn_scan(
            &normalized_x,
            &recurrent_gate,
            &reset,
            self.recurrent_states.as_ref(),
        )?;
        self.recurrent_states = Some(recurrent_states);
        Ok(hidden_states)
    }
}

fn rnn_scan(
    hidden_states: &Tensor,
    recurrent_gate: &Tensor,
    reset: &Tensor,
    recurrent_states: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let acc_dtype = DType::F32;
    let dev = hidden_states.device();
    let in_dtype = hidden_states.dtype();
    let inv_reset = (1.0 - reset)?.to_dtype(recurrent_gate.dtype())?;
    let recurrent_gate = recurrent_gate.broadcast_mul(&inv_reset)?;
    let (c, r) = if hidden_states.dim(1)? == 1 {
        match recurrent_states {
            None => {
                let next_state = hidden_states.i((.., 0))?.to_dtype(acc_dtype)?;
                (hidden_states.clone(), next_state)
            }
            Some(recurrent_states) => {
                let contextualized_states =
                    recurrent_gate.to_dtype(acc_dtype)? * recurrent_states.unsqueeze(1)?;
                let contextualized_states =
                    (contextualized_states + hidden_states.to_dtype(acc_dtype)?)?;
                let c = contextualized_states.to_dtype(in_dtype)?;
                let l = contextualized_states.dim(1)?;
                let r = contextualized_states.i((.., l - 1))?;
                (c, r)
            }
        }
    } else {
        let mut recurrent_states = match recurrent_states {
            None => Tensor::zeros(hidden_states.i((.., 0))?.shape(), acc_dtype, dev)?,
            Some(r) => r.clone(),
        };
        let mut contextualized_states = vec![];
        for t in 0..hidden_states.dim(1)? {
            recurrent_states =
                (recurrent_gate.i((.., t))?.to_dtype(acc_dtype)? * recurrent_states)?;
            recurrent_states =
                (recurrent_states + hidden_states.i((.., t))?.to_dtype(acc_dtype)?)?;
            contextualized_states.push(recurrent_states.to_dtype(in_dtype)?)
        }
        let contextualized_states = Tensor::stack(&contextualized_states, 1)?;
        (contextualized_states, recurrent_states)
    };
    Ok((c, r))
}

#[derive(Debug, Clone)]
struct RecurrentBlock {
    linear_y: Linear,
    linear_x: Linear,
    linear_out: Linear,
    conv_1d: candle_nn::Conv1d,
    conv1d_state: Option<Tensor>,
    conv1d_width: usize,
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
            conv1d_state: None,
            conv1d_width: cfg.conv1d_width,
            rg_lru,
            act_fn: cfg.hidden_activation,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len, _) = xs.dims3()?;

        let y_branch = xs.apply(&self.linear_y)?.apply(&self.act_fn)?;
        let x_branch = xs.apply(&self.linear_x)?.transpose(1, 2)?;
        let x_branch = if pos == 0 {
            let x_len = x_branch.dim(D::Minus1)?;
            let pad = self.conv1d_width as i64 - x_len as i64 - 1;
            let padded = match pad.cmp(&0) {
                std::cmp::Ordering::Equal => x_branch.clone(),
                std::cmp::Ordering::Less => {
                    let rev_pad = (-pad) as usize;
                    x_branch.narrow(D::Minus1, rev_pad, x_len - rev_pad)?
                }
                std::cmp::Ordering::Greater => {
                    x_branch.pad_with_zeros(D::Minus1, pad as usize, 0)?
                }
            };
            self.conv1d_state = Some(padded);
            x_branch
                .apply(&self.conv_1d)?
                .narrow(D::Minus1, 0, seq_len)?
        } else {
            let conv_state = match self.conv1d_state.as_ref() {
                None => candle::bail!("empty cache despite pos > 0"),
                Some(s) => Tensor::cat(&[s, &x_branch], D::Minus1)?,
            };
            let w = self.conv_1d.weight().i((.., 0, ..))?;
            let x_branch = conv_state.broadcast_mul(&w)?.sum(D::Minus1)?;
            let x_branch = match self.conv_1d.bias() {
                None => x_branch,
                Some(b) => x_branch.broadcast_add(b)?,
            };
            let x_branch = x_branch.unsqueeze(D::Minus1)?;
            self.conv1d_state = Some(conv_state.i((.., .., 1..))?);
            x_branch
        };
        let x_branch = x_branch.transpose(1, 2)?;
        let x_branch = self.rg_lru.forward(&x_branch, pos)?;
        (x_branch * y_branch)?.apply(&self.linear_out)
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
    kv_cache: Option<(Tensor, Tensor)>,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl SdpaAttention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
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
            kv_cache: None,
            rotary_emb,
        })
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_heads / self.n_kv_heads;
        crate::utils::repeat_kv(x, n_rep)
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        pos: usize,
    ) -> Result<Tensor> {
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
        let query_states = query_states.chunk(2, D::Minus1)?;
        let key_states = key_states.chunk(2, D::Minus1)?;
        let (query_rot, key_rot) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states[0], &key_states[0], pos)?;
        let query_states = Tensor::cat(&[&query_rot, &query_states[1]], D::Minus1)?.contiguous()?;
        let key_states = Tensor::cat(&[&key_rot, &key_states[1]], D::Minus1)?.contiguous()?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = self.repeat_kv(key_states)?;
        let value_states = self.repeat_kv(value_states)?;
        let xs = {
            let att = (query_states.matmul(&key_states.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if q_len == 1 {
                att
            } else {
                match attention_mask {
                    None => att,
                    Some(mask) => att.broadcast_add(mask)?,
                }
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
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        pos: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Recurrent(b) => b.forward(xs, pos),
            Self::Attention(b) => b.forward(xs, attention_mask, pos),
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
    fn new(
        block_idx: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let temporal_pre_norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("temporal_pre_norm"))?;
        let channel_pre_norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("channel_pre_norm"))?;
        let temporal_block = match cfg.block_types[block_idx % cfg.block_types.len()] {
            TemporalBlockType::Recurrent => {
                let block = RecurrentBlock::new(cfg, vb.pp("temporal_block"))?;
                TemporalBlock::Recurrent(block)
            }
            TemporalBlockType::Attention => {
                let block = SdpaAttention::new(rotary_emb, cfg, vb.pp("temporal_block"))?;
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

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        pos: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.temporal_pre_norm)?;
        let xs = self.temporal_block.forward(&xs, attention_mask, pos)?;
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
    lm_head: Linear,
    hidden_size: usize,
    logits_soft_cap: f64,
    dtype: DType,
    device: Device,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let vb_b = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(idx, rotary_emb.clone(), cfg, vb_b.pp(idx))?;
            layers.push(layer)
        }
        let final_norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("final_norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            logits_soft_cap: cfg.logits_soft_cap,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
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
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let (b_size, seq_len) = xs.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, pos)?;
            Some(mask)
        };
        let xs = xs.apply(&self.embed_tokens)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), pos)?;
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.final_norm)?
            .apply(&self.lm_head)?;
        let logits = ((logits / self.logits_soft_cap)?.tanh()? * self.logits_soft_cap)?;
        Ok(logits)
    }
}
