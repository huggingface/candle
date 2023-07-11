use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::MAX_SEQ_LEN;

pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

impl Config {
    pub fn config_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            n_layer: 32,
            n_head: 32,
            n_embd: 4096,
        }
    }
}

#[derive(Clone)]
pub struct Cache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, config: &Config, device: &Device) -> Self {
        Self {
            masks: Arc::new(Mutex::new(HashMap::new())),
            use_kv_cache,
            kvs: Arc::new(Mutex::new(vec![None; config.n_layer])),
            device: device.clone(),
        }
    }

    fn mask(&self, t: usize) -> Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            // TODO: If we support bool or u8 tensors, this would be better.
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u32::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    Ok(Linear::new(weight, None))
}

fn linear_multi(sz1: usize, sz2: usize, prefixes: &[&str], vb: &VarBuilder) -> Result<Linear> {
    let weights = prefixes
        .iter()
        .map(|p| vb.pp(p).get((sz1, sz2), "weight")?.t())
        .collect::<Result<Vec<_>>>()?;
    let weight = Tensor::cat(&weights, 0)?;
    Ok(Linear::new(weight, None))
}

struct Embedding {
    embeddings: Tensor,
}

impl Embedding {
    fn new(embeddings: Tensor) -> Self {
        Self { embeddings }
    }

    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        Ok(Self::new(embeddings))
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        Tensor::embedding(indexes, &self.embeddings)
    }
}

struct RmsNorm {
    scale: Tensor,
}

impl RmsNorm {
    fn load(size: usize, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(size, "weight")?;
        Ok(Self::new(scale))
    }

    fn new(scale: Tensor) -> Self {
        Self { scale }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        // This is a no-op if x's dtype is already f32.
        let x = x.to_dtype(DType::F32)?;
        let (seq_len, hidden_size) = x.shape().r2()?;
        let norm_x = ((&x * &x)?.sum(&[1])? / hidden_size as f64)?;
        let norm_x = norm_x.broadcast_as((seq_len, hidden_size))?;
        let x_normed = (x / (norm_x + 1e-5)?.sqrt()?)?;
        let size = self.scale.shape().r1()?;
        let scale = self
            .scale
            .to_dtype(DType::F32)?
            .broadcast_as((seq_len, size))?;
        let x = (scale * x_normed)?;
        let x = x.to_dtype(in_dtype)?;
        Ok(x)
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    cache: Cache,
}

impl CausalSelfAttention {
    fn new(c_attn: Linear, c_proj: Linear, n_head: usize, cache: &Cache) -> Self {
        Self {
            c_attn,
            c_proj,
            n_head,
            cache: cache.clone(),
        }
    }

    fn apply_rotary_emb(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        let mut dims = x.dims().to_vec();
        let fcis_dims = freqs_cis.dims();
        let freqs_cis = if dims[1] < fcis_dims[1] {
            freqs_cis.narrow(1, 0, dims[1])?
        } else {
            freqs_cis.clone()
        };
        let v = dims.pop().unwrap();
        dims.push(v / 2);
        dims.push(2);
        let x = x.reshape(dims)?;
        let re_x = x.narrow(D::Minus1, 0, 1)?;
        let im_x = x.narrow(D::Minus1, 1, 1)?;
        let re_f = freqs_cis
            .narrow(D::Minus1, 0, 1)?
            .broadcast_as(re_x.shape())?;
        let im_f = freqs_cis
            .narrow(D::Minus1, 1, 1)?
            .broadcast_as(im_x.shape())?;
        let re = ((&re_x * &re_f)? - (&im_x * &im_f)?)?;
        let im = ((&re_x * &im_f)? + (&im_x * &re_f)?)?;
        let rope = Tensor::cat(&[&re, &im], D::Minus1)?;
        let rope = rope.flatten_from(D::Minus2)?;
        Ok(rope)
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor, block_idx: usize) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let (t, c) = x.shape().r2()?;
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.to_dtype(DType::F32)?;
        let n_embd = c;
        let q = qkv.narrow(1, 0, n_embd)?;
        let k = qkv.narrow(1, n_embd, n_embd)?;
        let v = qkv.narrow(1, 2 * n_embd, n_embd)?;
        let target_dim = [t, self.n_head, c / self.n_head];
        let k = k.reshape(target_dim.as_slice())?.transpose(0, 1)?;
        let q = q.reshape(target_dim.as_slice())?.transpose(0, 1)?;
        let mut v = v.reshape(target_dim.as_slice())?.transpose(0, 1)?;
        let q = self.apply_rotary_emb(&q, freqs_cis)?;
        let mut k = self.apply_rotary_emb(&k, freqs_cis)?;

        if self.cache.use_kv_cache {
            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 1)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 1)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            cache[block_idx] = Some((k.clone(), v.clone()))
        }

        let att = (q.matmul(&k.t()?)? / (k.dim(D::Minus1)? as f64).sqrt())?;
        let mask = self.cache.mask(t)?.broadcast_as(att.shape())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = att.softmax(D::Minus1)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(0, 1)?.reshape(&[t, c])?;
        let y = y.to_dtype(x_dtype)?;
        let y = self.c_proj.forward(&y)?;
        Ok(y)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size = (cfg.hidden_size / cfg.n_head) * cfg.n_head;
        let c_attn = linear_multi(size_in, size, &["q_proj", "k_proj", "v_proj"], &vb)?;
        let o_proj = linear(size, size_in, vb.pp("o_proj"))?;
        Ok(Self::new(c_attn, o_proj, cfg.n_head, cache))
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, &on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(c_fc1: Linear, c_fc2: Linear, c_proj: Linear) -> Self {
        Self {
            c_fc1,
            c_fc2,
            c_proj,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(rms_1: RmsNorm, attn: CausalSelfAttention, rms_2: RmsNorm, mlp: Mlp) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor, block_idx: usize) -> Result<Tensor> {
        let x = (self
            .attn
            .forward(&self.rms_1.forward(x)?, freqs_cis, block_idx)?
            + x)?;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + x)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let input_layernorm = RmsNorm::load(cfg.hidden_size, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::load(cfg.hidden_size, vb.pp("post_attention_layernorm"))?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    fn new(wte: Embedding, blocks: Vec<Block>, ln_f: RmsNorm, lm_head: Linear) -> Self {
        Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        }
    }

    pub fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
        // TODO: Support for mini-batches? (i.e. r2)
        let t = x.shape().r1()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, freqs_cis, block_idx)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.narrow(0, t - 1, 1)?;
        let logits = self.lm_head.forward(&x)?;
        let logits = logits.to_dtype(DType::F32)?;
        let (b, vocab_size) = logits.shape().r2()?;
        assert_eq!(b, 1);
        logits.reshape(vocab_size)
    }

    pub fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let wte = Embedding::load(cfg, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let norm = RmsNorm::load(cfg.hidden_size, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.n_layer)
            .map(|i| Block::load(vb.pp(&format!("model.layers.{i}")), cache, cfg).unwrap())
            .collect();

        Ok(Self::new(wte, blocks, norm, lm_head))
    }
}
