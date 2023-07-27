use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};

fn linear(size1: usize, size2: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = if bias {
        Some(vb.get(size2, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err.into());
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug)]
pub struct Config {
    vocab_size: usize,
    max_position_embeddings: usize,
    num_hidden_layers: usize,
    hidden_size: usize,
    layer_norm_epsilon: f64,
    n_inner: Option<usize>,
    num_attention_heads: usize,
    multi_query: bool,
}

struct Attention {
    c_attn: Linear,
    c_proj: Linear,
    embed_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    multi_query: bool,
}

impl Attention {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let head_dim = hidden_size / cfg.num_attention_heads;
        let kv_heads = if cfg.multi_query {
            1
        } else {
            cfg.num_attention_heads
        };
        let kv_dim = kv_heads * head_dim;
        let c_attn = linear(hidden_size, hidden_size + 2 * kv_dim, true, vb.pp("c_attn"))?;
        let c_proj = linear(hidden_size, hidden_size, true, vb.pp("c_proj"))?;
        Ok(Self {
            c_proj,
            c_attn,
            embed_dim: hidden_size,
            kv_dim,
            head_dim,
            multi_query: cfg.multi_query,
        })
    }

    fn attn(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        todo!()
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let qkv = self.c_attn.forward(hidden_states)?;
        let (query, key_value) = if self.multi_query {
            let query = qkv.i((.., .., ..self.embed_dim))?;
            let key_value = qkv.i((.., .., self.embed_dim..))?;
            (query, key_value)
        } else {
            let mut dims = qkv.dims().to_vec();
            dims.pop();
            dims.push(self.embed_dim);
            dims.push(self.head_dim * 3);
            let qkv = qkv.reshape(dims)?.transpose(1, 2)?;
            let query = qkv.i((.., .., .., ..self.head_dim))?;
            let key_value = qkv.i((.., .., .., self.head_dim..))?;
            (query, key_value)
        };
        // TODO: layer past
        let key = key_value.narrow(D::Minus1, 0, self.head_dim)?;
        let value = key_value.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let (attn_output, attn_weights) = self.attn(&query, &key.t()?, &value)?; // TODO: masks
        todo!()
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(inner_dim: usize, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let c_fc = linear(cfg.hidden_size, inner_dim, true, vb.pp("c_fc"))?;
        let c_proj = linear(inner_dim, cfg.hidden_size, true, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(hidden_states)?.gelu()?;
        let hidden_states = self.c_proj.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

// TODO: Add cross-attention?
struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let inner_dim = cfg.n_inner.unwrap_or(4 * hidden_size);
        let ln_1 = layer_norm(hidden_size, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = Attention::load(vb.pp("attn"), cfg)?;
        let ln_2 = layer_norm(hidden_size, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = Mlp::load(inner_dim, vb.pp("mlp"), cfg)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.ln_1.forward(hidden_states)?;
        let attn_outputs = self.attn.forward(&hidden_states)?;
        let hidden_states = (&attn_outputs + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.ln_2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (&hidden_states + residual)?;
        Ok(hidden_states)
    }
}

pub struct GPTBigCode {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: Config,
}

impl GPTBigCode {
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn load(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let wte = embedding(cfg.vocab_size, hidden_size, vb.pp("wte"))?;
        let wpe = embedding(cfg.max_position_embeddings, hidden_size, vb.pp("wpe"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(&format!("h.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(hidden_size, cfg.layer_norm_epsilon, vb.pp("ln_f"))?;
        let lm_head = linear(hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            lm_head,
            ln_f,
            config: cfg,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let input_embeds = self.wte.forward(input_ids)?;
        let position_embeds = self.wpe.forward(position_ids)?;
        let mut hidden_states = (&input_embeds + &position_embeds)?;
        for block in self.blocks.iter_mut() {
            hidden_states = block.forward(&hidden_states)?;
        }
        let hidden_states = self.ln_f.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, seq_len))?;
        let logits = self.lm_head.forward(&hidden_states)?.squeeze(1)?;
        Ok(logits)
    }
}
