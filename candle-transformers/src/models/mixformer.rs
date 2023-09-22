#![allow(unused)]
/// MixFormer model.
/// https://huggingface.co/microsoft/phi-1_5
/// https://arxiv.org/abs/2309.05463
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};

const MAX_SEQ_LEN: usize = 4096;

// https://huggingface.co/microsoft/phi-1_5/blob/main/configuration_mixformer_sequential.py
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    n_inner: Option<usize>,
    n_head: usize,
    rotary_dim: usize,
    activation_function: Activation,
    layer_norm_epsilon: f64,
    tie_word_embeddings: bool,
    pad_vocab_size_multiple: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50304,
            n_positions: 2048,
            n_embd: 1024,
            n_layer: 20,
            n_inner: None,
            n_head: 16,
            rotary_dim: usize::min(32, 1024 / 16),
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
            tie_word_embeddings: false,
            pad_vocab_size_multiple: 64,
        }
    }
}

#[derive(Debug)]
struct Embedding {
    wte: candle_nn::Embedding,
}

impl Embedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        Ok(Self { wte })
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.wte.forward(xs)
    }
}

#[derive(Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        qkv: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_b_size, seqlen, three, _, _headdim) = qkv.dims5()?;
        if three != 3 {
            candle::bail!("unexpected shape for qkv {:?}", qkv.shape())
        }
        let (_rotary_seqlen, rotary_dim) = self.cos.dims2()?;
        let q_rot = qkv.i((.., .., 0, .., ..rotary_dim))?;
        let q_pass = qkv.i((.., .., 0, .., rotary_dim..))?;
        let k_rot = qkv.i((.., .., 1, .., ..rotary_dim))?;
        let k_pass = qkv.i((.., .., 1, .., rotary_dim..))?;
        let q12 = q_rot.chunk(2, D::Minus1)?;
        let k12 = k_rot.chunk(2, D::Minus1)?;
        let (q1, q2) = (&q12[0], &q12[1]);
        let (k1, k2) = (&k12[0], &k12[1]);
        let c = self.cos.narrow(0, seqlen_offset, seqlen)?;
        let s = self.sin.narrow(0, seqlen_offset, seqlen)?;
        let q_rot = Tensor::cat(
            &[((q1 * &c)? - (q2 * &s)?)?, ((q1 * &s)? + (q2 * &c)?)?],
            D::Minus1,
        )?;
        let k_rot = Tensor::cat(
            &[((k1 * &c)? - (k2 * &s)?)?, ((k1 * &s)? + (k2 * &c)?)?],
            D::Minus1,
        )?;
        let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?;
        let k = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?;
        let v = qkv.i((.., .., 2..3))?;
        Ok((q, k, v))
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    act: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_inner = cfg.n_inner.unwrap_or(4 * cfg.n_embd);
        let fc1 = candle_nn::linear(cfg.n_embd, n_inner, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(n_inner, cfg.n_embd, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: cfg.activation_function,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

#[derive(Debug)]
struct CausalLMHead {
    ln: candle_nn::LayerNorm,
    linear: candle_nn::Linear,
}

impl CausalLMHead {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln"))?;
        let linear = candle_nn::linear(cfg.n_embd, cfg.vocab_size, vb.pp("linear"))?;
        Ok(Self { ln, linear })
    }
}

impl Module for CausalLMHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.ln)?
            .apply(&self.linear)?
            .to_dtype(DType::F32)
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MHA {
    wqkv: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    rotary_emb: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
    head_dim: usize,
    n_head: usize,
    softmax_scale: f64,
}

impl MHA {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.n_embd / cfg.n_head;
        let op_size = cfg.n_embd;
        let wqkv = candle_nn::linear(cfg.n_embd, 3 * op_size, vb.pp("Wqkv"))?;
        let out_proj = candle_nn::linear(op_size, cfg.n_embd, vb.pp("out_proj"))?;
        let rotary_emb = RotaryEmbedding::new(cfg.rotary_dim, MAX_SEQ_LEN, vb.device())?;
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Ok(Self {
            wqkv,
            out_proj,
            head_dim,
            n_head: cfg.n_head,
            kv_cache: None,
            rotary_emb,
            softmax_scale,
        })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, n_embd) = xs.dims3()?;
        let qkv = self
            .wqkv
            .forward(xs)?
            .reshape((b_size, seq_len, 3, (), self.head_dim))?;
        let seqlen_offset = match &self.kv_cache {
            None => 0,
            Some((prev_k, _)) => prev_k.dim(1)?,
        };
        // In the python implementation, a single tensor is returned with the third axis of size 3.
        let (q, k, v) = self.rotary_emb.apply_rotary_emb_qkv(&qkv, seqlen_offset)?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 1)?;
                let v = Tensor::cat(&[prev_v, &v], 1)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        // scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        let q = q.reshape((b_size, (), self.head_dim * self.n_head))?; // b, t, h*d
        let k = k.reshape((b_size, (), self.head_dim * self.n_head))?; // b, s, h*d
        let attn_weights = (q.matmul(&k.t()?)? * self.softmax_scale)?; // b, t, s

        // causal_mask = torch.triu(torch.full((seqlen_q, seqlen_k), -10000.0, device=scores.device), 1)
        // scores = scores + causal_mask.to(dtype=scores.dtype)
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output.flatten_from(D::Minus2)?.apply(&self.out_proj)
    }
}

#[derive(Debug)]
struct ParallelBlock {
    ln: candle_nn::LayerNorm,
    mixer: MHA,
    mlp: MLP,
}

impl ParallelBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln"))?;
        let mixer = MHA::new(cfg, vb.pp("mixer"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self { ln, mixer, mlp })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.ln)?;
        let attn_outputs = self.mixer.forward(&xs)?;
        let feed_forward_hidden_states = self.mlp.forward(&xs)?;
        attn_outputs + feed_forward_hidden_states + residual
    }
}

#[derive(Debug)]
pub struct MixFormerSequentialForCausalLM {
    embedding: Embedding,
    blocks: Vec<ParallelBlock>,
    head: CausalLMHead,
}

impl MixFormerSequentialForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("layers");
        let embedding = Embedding::new(cfg, vb.pp(0))?;
        let mut blocks = Vec::new();
        for i in 0..cfg.n_layer {
            let block = ParallelBlock::new(cfg, vb.pp(i + 1))?;
            blocks.push(block)
        }
        let head = CausalLMHead::new(cfg, vb.pp(cfg.n_layer + 1))?;
        Ok(Self {
            embedding,
            blocks,
            head,
        })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.embedding)?;
        for block in self.blocks.iter_mut() {
            xs = block.forward(&xs)?
        }
        xs.apply(&self.head)
    }
}
