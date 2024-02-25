use super::with_tracing::{layer_norm, linear_no_bias as linear, LayerNorm, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::collections::{HashMap, HashSet};

fn default_num_attention_heads() -> usize {
    64
}

// https://huggingface.co/RWKV/HF_v5-Eagle-7B/blob/main/configuration_rwkv5.py
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub attention_hidden_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    pub head_size: usize,
    pub intermediate_size: Option<usize>,
    pub layer_norm_epsilon: f64,
    pub rescale_every: usize,
}

pub struct StatePerLayer {
    pub extract_key_value: Tensor,
    pub linear_attention: Tensor,
    pub feed_forward: Tensor,
}

pub struct State {
    pub per_layer: Vec<StatePerLayer>,
    pub pos: usize,
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, dev: &Device) -> Result<Self> {
        let mut per_layer = Vec::with_capacity(cfg.num_hidden_layers);
        // Certainly a weird convention but taken from modeling_rwkv5.py
        let num_attention_heads = cfg.hidden_size / cfg.num_attention_heads;
        for _layer_idx in 0..cfg.num_hidden_layers {
            let extract_key_value = Tensor::zeros((batch_size, cfg.hidden_size), DType::F32, dev)?;
            let linear_attention = Tensor::zeros(
                (
                    batch_size,
                    num_attention_heads,
                    cfg.hidden_size / num_attention_heads,
                    cfg.hidden_size / num_attention_heads,
                ),
                DType::F32,
                dev,
            )?;
            let feed_forward = Tensor::zeros((batch_size, cfg.hidden_size), DType::F32, dev)?;
            per_layer.push(StatePerLayer {
                extract_key_value,
                linear_attention,
                feed_forward,
            });
        }
        Ok(Self { per_layer, pos: 0 })
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    key: Linear,
    receptance: Linear,
    value: Linear,
    gate: Linear,
    output: Linear,
    ln_x: candle_nn::GroupNorm,
    time_mix_key: Tensor,
    time_mix_value: Tensor,
    time_mix_receptance: Tensor,
    time_decay: Tensor,
    time_faaaa: Tensor,
    time_mix_gate: Tensor,
    layer_id: usize,
    n_attn_heads: usize,
}

impl SelfAttention {
    pub fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
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
        let time_mix_key = vb.get((1, 1, cfg.hidden_size), "time_mix_key")?;
        let time_mix_value = vb.get((1, 1, cfg.hidden_size), "time_mix_value")?;
        let time_mix_receptance = vb.get((1, 1, cfg.hidden_size), "time_mix_receptance")?;
        let n_attn_heads = cfg.hidden_size / cfg.head_size;
        let time_decay = vb.get((n_attn_heads, cfg.head_size), "time_decay")?;
        let time_faaaa = vb.get((n_attn_heads, cfg.head_size), "time_faaaa")?;
        let time_mix_gate = vb.get((1, 1, cfg.hidden_size), "time_mix_gate")?;
        Ok(Self {
            key,
            value,
            receptance,
            gate,
            output,
            ln_x,
            time_mix_key,
            time_mix_value,
            time_mix_receptance,
            time_decay,
            time_faaaa,
            time_mix_gate,
            layer_id,
            n_attn_heads,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let h = self.time_decay.dim(0)?;
        let (b, t, s) = xs.dims3()?;
        let s = s / h;
        let (receptance, key, value, gate) = {
            // extract key-value
            let shifted = state.per_layer[self.layer_id].extract_key_value.clone();
            let shifted = if shifted.rank() == 2 {
                shifted.unsqueeze(1)?
            } else {
                shifted
            };
            let key = ((xs * &self.time_mix_key)? + &shifted * (1.0 - &self.time_mix_key)?)?;
            let value = ((xs * &self.time_mix_value)? + &shifted * (1.0 - &self.time_mix_value)?)?;
            let receptance = ((xs * &self.time_mix_receptance)?
                + &shifted * (1.0 - &self.time_mix_receptance)?)?;
            let gate = ((xs * &self.time_mix_gate)? + &shifted * (1.0 - &self.time_mix_gate)?)?;

            let key = self.key.forward(&key)?;
            let value = self.value.forward(&value)?;
            let receptance = self.receptance.forward(&receptance)?;
            let gate = candle_nn::ops::silu(&self.gate.forward(&gate)?)?;
            state.per_layer[self.layer_id].extract_key_value = xs.i((.., t - 1))?;
            (receptance, key, value, gate)
        };
        // linear attention
        let mut state_ = state.per_layer[self.layer_id].linear_attention.clone();
        let key = key.reshape((b, t, h, s))?.permute((0, 2, 3, 1))?;
        let value = value.reshape((b, t, h, s))?.transpose(1, 2)?;
        let receptance = receptance.reshape((b, t, h, s))?.transpose(1, 2)?;

        let time_decay = self
            .time_decay
            .exp()?
            .neg()?
            .exp()?
            .reshape(((), 1, 1))?
            .reshape((self.n_attn_heads, (), 1))?;
        let time_faaaa =
            self.time_faaaa
                .reshape(((), 1, 1))?
                .reshape((self.n_attn_heads, (), 1))?;

        let mut out: Vec<Tensor> = Vec::with_capacity(t);
        for t_ in 0..t {
            let rt = receptance.i((.., .., t_..t_ + 1))?.contiguous()?;
            let kt = key.i((.., .., .., t_..t_ + 1))?.contiguous()?;
            let vt = value.i((.., .., t_..t_ + 1))?.contiguous()?;
            let at = kt.matmul(&vt)?;
            let rhs = (time_faaaa.broadcast_mul(&at)? + &state_)?;
            let out_ = rt.matmul(&rhs)?.squeeze(2)?;
            state_ = (&at + time_decay.broadcast_mul(&state_))?;
            out.push(out_)
        }
        let out = Tensor::cat(&out, 1)?.reshape((b * t, h * s, 1))?;
        let out = out.apply(&self.ln_x)?.reshape((b, t, h * s))?;
        let out = (out * gate)?.apply(&self.output)?;
        state.per_layer[self.layer_id].linear_attention = state_;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    time_mix_key: Tensor,
    time_mix_receptance: Tensor,
    key: Linear,
    receptance: Linear,
    value: Linear,
    layer_id: usize,
}

impl FeedForward {
    pub fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let int_size = cfg
            .intermediate_size
            .unwrap_or(((cfg.hidden_size as f64 * 3.5) as usize) / 32 * 32);
        let key = linear(cfg.hidden_size, int_size, vb.pp("key"))?;
        let receptance = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("receptance"))?;
        let value = linear(int_size, cfg.hidden_size, vb.pp("value"))?;
        let time_mix_key = vb.get((1, 1, cfg.hidden_size), "time_mix_key")?;
        let time_mix_receptance = vb.get((1, 1, cfg.hidden_size), "time_mix_receptance")?;
        Ok(Self {
            key,
            receptance,
            value,
            time_mix_key,
            time_mix_receptance,
            layer_id,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let shifted = &state.per_layer[self.layer_id].feed_forward;
        let key = (xs.broadcast_mul(&self.time_mix_key)?
            + shifted.broadcast_mul(&(1.0 - &self.time_mix_key)?)?)?;
        let receptance = (xs.broadcast_mul(&self.time_mix_receptance)?
            + shifted.broadcast_mul(&(1.0 - &self.time_mix_receptance)?)?)?;
        let key = key.apply(&self.key)?.relu()?.sqr()?;
        let value = key.apply(&self.value)?;
        let receptance = candle_nn::ops::sigmoid(&receptance.apply(&self.receptance)?)?;
        state.per_layer[self.layer_id].feed_forward = xs.i((.., xs.dim(1)? - 1))?;
        let xs = (receptance * value)?;
        Ok(xs)
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
    pub fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln1 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln2"))?;
        let pre_ln = if layer_id == 0 {
            let ln = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("pre_ln"))?;
            Some(ln)
        } else {
            None
        };
        let attention = SelfAttention::new(layer_id, cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(layer_id, cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let xs = match self.pre_ln.as_ref() {
            None => xs.clone(),
            Some(pre_ln) => xs.apply(pre_ln)?,
        };
        let attention = self.attention.forward(&xs.apply(&self.ln1)?, state)?;
        let xs = (xs + attention)?;
        let feed_forward = self.feed_forward.forward(&xs.apply(&self.ln2)?, state)?;
        let xs = (xs + feed_forward)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head: Linear,
    rescale_every: usize,
    layers_are_rescaled: bool,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("rwkv");
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
            rescale_every: cfg.rescale_every,
            layers_are_rescaled: false, // This seem to only happen for the f16/bf16 dtypes.
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (_b_size, _seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embeddings)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs, state)?;
            if self.layers_are_rescaled && (block_idx + 1) % self.rescale_every == 0 {
                xs = (xs / 2.)?
            }
        }
        let xs = xs.apply(&self.ln_out)?.apply(&self.head)?;
        state.pos += 1;
        Ok(xs)
    }
}

type Bytes = Vec<u8>;

// https://github.com/BlinkDL/ChatRWKV/blob/095e812aef15a1f74107f6c39d13578a2412dc46/RWKV_v5_demo.py#L14
pub struct Tokenizer {
    table: Vec<Vec<Vec<Bytes>>>,
    good: Vec<HashSet<u8>>,
    idx2token: HashMap<u32, Vec<u8>>,
    token2idx: HashMap<Vec<u8>, u32>,
}

impl Tokenizer {
    pub fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let file = std::fs::File::open(p)?;
        let token2idx: HashMap<String, u32> =
            serde_json::from_reader(file).map_err(candle::Error::wrap)?;
        let token2idx = token2idx
            .into_iter()
            .map(|(key, value)| (key.into_bytes(), value))
            .collect::<HashMap<_, _>>();
        let idx2token = token2idx
            .iter()
            .map(|(key, value)| (*value, key.to_vec()))
            .collect::<HashMap<_, _>>();

        let max_idx = token2idx.values().copied().max().unwrap_or(0);

        let mut table = vec![vec![vec![]; 256]; 256];
        let mut good = vec![HashSet::new(); 256];
        for idx in (0..(1 + max_idx)).rev() {
            let s = match idx2token.get(&idx) {
                None => continue,
                Some(s) => s,
            };
            if s.len() >= 2 {
                let (s0, s1) = (s[0], s[1]);
                table[s0 as usize][s1 as usize].push(s.to_vec());
                good[s0 as usize].insert(s1);
            }
        }
        Ok(Self {
            table,
            good,
            idx2token,
            token2idx,
        })
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        for token_id in tokens.iter() {
            if let Some(token) = self.idx2token.get(token_id) {
                v.extend_from_slice(token.as_slice())
            }
        }
        v
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(tokens);
        String::from_utf8(bytes).map_err(candle::Error::wrap)
    }

    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let mut s = vec![bytes[i]];
            if i + 1 < bytes.len() && self.good[bytes[i] as usize].contains(&bytes[i + 1]) {
                let table = &self.table[bytes[i] as usize][bytes[i + 1] as usize];
                for table_elem in table.iter() {
                    if bytes[i..].starts_with(table_elem) {
                        s = table_elem.to_vec();
                        break;
                    }
                }
            }
            i += s.len();
            let token = match self.token2idx.get(&s) {
                None => candle::bail!("unexpected token '{}' {s:?}", String::from_utf8_lossy(&s)),
                Some(token) => *token,
            };
            tokens.push(token)
        }
        Ok(tokens)
    }

    pub fn encode(&self, str: &str) -> Result<Vec<u32>> {
        self.encode_bytes(str.as_bytes())
    }
}
