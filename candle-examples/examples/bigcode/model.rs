use candle::{DType, Device, IndexOp, Result, Tensor, D};
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
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug)]
pub struct Config {
    pub vocab_size: usize,
    // max_position_embeddings aka n_positions
    pub max_position_embeddings: usize,
    // num_hidden_layers aka n_layer
    pub num_hidden_layers: usize,
    // hidden_size aka n_embd
    pub hidden_size: usize,
    pub layer_norm_epsilon: f64,
    pub n_inner: Option<usize>,
    // num_attention_heads aka n_head
    pub num_attention_heads: usize,
    pub multi_query: bool,
    pub use_cache: bool,
}

impl Config {
    pub fn starcoder() -> Self {
        Self {
            vocab_size: 49152,
            max_position_embeddings: 8192,
            num_hidden_layers: 40,
            hidden_size: 6144,
            layer_norm_epsilon: 1e-5,
            n_inner: Some(24576),
            num_attention_heads: 48,
            multi_query: true,
            use_cache: true,
        }
    }
}

struct Attention {
    c_attn: Linear,
    c_proj: Linear,
    kv_cache: Option<Tensor>,
    use_cache: bool,
    embed_dim: usize,
    kv_dim: usize,
    num_heads: usize,
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
            kv_cache: None,
            use_cache: cfg.use_cache,
            kv_dim,
            head_dim,
            num_heads: cfg.num_attention_heads,
            multi_query: cfg.multi_query,
        })
    }

    fn attn(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        // TODO: check if we need scaling/upcasting.
        let scale_factor = 1f64 / (self.head_dim as f64).sqrt();
        let initial_query_shape = query.shape();
        let key_len = key.dim(D::Minus1)?;
        let (query, key, attn_shape) = if self.multi_query {
            let (b_sz, query_len, _) = query.dims3()?;
            let query = query.reshape((b_sz, query_len * self.num_heads, key_len))?;
            let attn_shape = (b_sz, query_len, self.num_heads, key_len);
            (query, key.clone(), attn_shape)
        } else {
            let (b_sz, _num_heads, query_len, _head_dim) = query.dims4()?;
            let query = query.reshape((b_sz, query_len * self.num_heads, key_len))?;
            let key = key.reshape((b_sz * self.num_heads, self.head_dim, key_len))?;
            let attn_shape = (b_sz, self.num_heads, query_len, key_len);
            (query, key, attn_shape)
        };

        let attn_weights = (query.matmul(&key)? * scale_factor)?.reshape(attn_shape)?;
        let attn_weights = attn_weights.softmax(D::Minus1)?;
        let attn_output = if self.multi_query {
            attn_weights
                .reshape(query.shape())?
                .matmul(value)?
                .reshape(initial_query_shape)?
        } else {
            attn_weights.matmul(value)?
        };
        Ok(attn_output)
    }

    fn forward(&mut self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
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
        let mut key_value = key_value;
        if self.use_cache {
            if let Some(kv_cache) = &self.kv_cache {
                // TODO: we could trim the tensors to MAX_SEQ_LEN so that this would work for
                // arbitrarily large sizes.
                key_value = Tensor::cat(&[kv_cache, &key_value], D::Minus2)?.contiguous()?;
            }
            self.kv_cache = Some(key_value.clone())
        }

        let key = key_value.narrow(D::Minus1, 0, self.head_dim)?;
        let value = key_value.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let attn_output = self.attn(&query, &key.t()?, &value, attention_mask)?;
        let attn_output = if self.multi_query {
            attn_output
        } else {
            attn_output
                .transpose(1, 2)?
                .reshape(hidden_states.shape())?
        };
        let attn_output = self.c_proj.forward(&attn_output)?;
        Ok(attn_output)
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

    fn forward(&mut self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.ln_1.forward(hidden_states)?;
        let attn_outputs = self.attn.forward(&hidden_states, attention_mask)?;
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

    pub fn forward(&mut self, input_ids: &Tensor, past_len: usize) -> Result<Tensor> {
        let dev = input_ids.device();
        let (b_sz, seq_len) = input_ids.dims2()?;
        let attention_mask = Tensor::zeros(1, DType::F32, input_ids.device())?; // TODO
        let position_ids = Tensor::arange(past_len as u32, (past_len + seq_len) as u32, dev)?;
        let position_ids = position_ids.unsqueeze(0)?.broadcast_as((b_sz, seq_len))?;
        let input_embeds = self.wte.forward(input_ids)?;
        let position_embeds = self.wpe.forward(&position_ids)?;
        let mut hidden_states = (&input_embeds + &position_embeds)?;
        for block in self.blocks.iter_mut() {
            hidden_states = block.forward(&hidden_states, &attention_mask)?;
        }
        let hidden_states = self.ln_f.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, seq_len))?;
        let logits = self.lm_head.forward(&hidden_states)?.squeeze(1)?;
        Ok(logits)
    }
}
