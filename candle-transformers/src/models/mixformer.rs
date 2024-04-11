use crate::models::with_tracing::{linear, Embedding as E, Linear};
/// MixFormer model.
/// https://huggingface.co/microsoft/phi-1_5
/// https://arxiv.org/abs/2309.05463
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;

const MAX_SEQ_LEN: usize = 4096;

// https://huggingface.co/microsoft/phi-1_5/blob/d38e6f954ec29b96fe2cf033937dad64e279b5d9/configuration_mixformer_sequential.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) n_positions: usize,
    pub(crate) n_embd: usize,
    pub(crate) n_layer: usize,
    pub(crate) n_inner: Option<usize>,
    pub(crate) n_head: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) activation_function: Activation,
    pub(crate) layer_norm_epsilon: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) pad_vocab_size_multiple: usize,
}

impl Config {
    pub fn v1() -> Self {
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

    pub fn v1_5() -> Self {
        Self {
            vocab_size: 51200,
            n_positions: 2048,
            n_embd: 2048,
            n_layer: 24,
            n_inner: None,
            n_head: 32,
            rotary_dim: usize::min(32, 2048 / 32),
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
            tie_word_embeddings: false,
            pad_vocab_size_multiple: 64,
        }
    }

    pub fn v2() -> Self {
        Self {
            vocab_size: 51200,
            n_positions: 2048,
            n_embd: 2560,
            n_layer: 32,
            n_inner: None,
            n_head: 32,
            rotary_dim: usize::min(32, 2560 / 32),
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
            tie_word_embeddings: false,
            pad_vocab_size_multiple: 64,
        }
    }

    // https://huggingface.co/teknium/Puffin-Phi-v2/blob/main/config.json
    pub fn puffin_phi_v2() -> Self {
        Self {
            vocab_size: 50304,
            n_positions: 2048,
            n_embd: 2048,
            n_layer: 24,
            n_inner: None,
            n_head: 32,
            rotary_dim: usize::min(32, 2048 / 32),
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
            tie_word_embeddings: false,
            pad_vocab_size_multiple: 64,
        }
    }

    // https://huggingface.co/teknium/Phi-Hermes-1.3B/blob/main/config.json
    pub fn phi_hermes_1_3b() -> Self {
        Self {
            vocab_size: 50304,
            n_positions: 2048,
            n_embd: 2048,
            n_layer: 24,
            n_inner: None,
            n_head: 32,
            rotary_dim: usize::min(32, 2048 / 32),
            activation_function: Activation::NewGelu,
            layer_norm_epsilon: 1e-5,
            tie_word_embeddings: false,
            pad_vocab_size_multiple: 64,
        }
    }
}

#[derive(Debug, Clone)]
struct Embedding {
    wte: E,
}

impl Embedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = E::new(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        Ok(Self { wte })
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.wte.forward(xs)
    }
}

fn get_mask(size: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, dtype: DType, dev: &Device) -> Result<Self> {
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
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
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
        let rotary_dim = rotary_dim * 2;
        let q_rot = qkv.i((.., .., 0, .., ..rotary_dim))?.contiguous()?;
        let q_pass = qkv.i((.., .., 0, .., rotary_dim..))?;
        let k_rot = qkv.i((.., .., 1, .., ..rotary_dim))?.contiguous()?;
        let k_pass = qkv.i((.., .., 1, .., rotary_dim..))?;
        let c = self.cos.narrow(0, seqlen_offset, seqlen)?;
        let s = self.sin.narrow(0, seqlen_offset, seqlen)?;
        let q_rot = candle_nn::rotary_emb::rope_thd(&q_rot, &c, &s)?;
        let k_rot = candle_nn::rotary_emb::rope_thd(&k_rot, &c, &s)?;
        let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?;
        let k = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?;
        let v = qkv.i((.., .., 2))?;
        Ok((q, k, v))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
    span: tracing::Span,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_inner = cfg.n_inner.unwrap_or(4 * cfg.n_embd);
        let fc1 = linear(cfg.n_embd, n_inner, vb.pp("fc1"))?;
        let fc2 = linear(n_inner, cfg.n_embd, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: cfg.activation_function,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct CausalLMHead {
    ln: candle_nn::LayerNorm,
    linear: Linear,
}

impl CausalLMHead {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln"))?;
        let linear = linear(cfg.n_embd, cfg.vocab_size, vb.pp("linear"))?;
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

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MHA {
    wqkv: Linear,
    out_proj: Linear,
    rotary_emb: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
    head_dim: usize,
    softmax_scale: f64,
    span: tracing::Span,
    span_rope: tracing::Span,
    span_mask: tracing::Span,
    span_softmax: tracing::Span,
}

impl MHA {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.n_embd / cfg.n_head;
        let op_size = cfg.n_embd;
        let wqkv = linear(cfg.n_embd, 3 * op_size, vb.pp("Wqkv"))?;
        let out_proj = linear(op_size, cfg.n_embd, vb.pp("out_proj"))?;
        let rotary_emb =
            RotaryEmbedding::new(cfg.rotary_dim, MAX_SEQ_LEN, vb.dtype(), vb.device())?;
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Ok(Self {
            wqkv,
            out_proj,
            head_dim,
            kv_cache: None,
            rotary_emb,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "mha"),
            span_rope: tracing::span!(tracing::Level::TRACE, "rope"),
            span_mask: tracing::span!(tracing::Level::TRACE, "mask"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let qkv = self
            .wqkv
            .forward(xs)?
            .reshape((b_size, seq_len, 3, (), self.head_dim))?;
        let seqlen_offset = match &self.kv_cache {
            None => 0,
            Some((prev_k, _)) => prev_k.dim(1)?,
        };
        // In the python implementation, a single tensor is returned with the third axis of size 3.
        let (q, k, v) = {
            let _enter = self.span_rope.enter();
            self.rotary_emb.apply_rotary_emb_qkv(&qkv, seqlen_offset)?
        };
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
        let q = q.transpose(1, 2)?.flatten_to(1)?; // b*h, t, d
        let k = k.transpose(1, 2)?.flatten_to(1)?; // b*h, s, d
        let v = v.transpose(1, 2)?.flatten_to(1)?; // b*h, s, d
        let attn_weights = (q.matmul(&k.t()?)? * self.softmax_scale)?; // b*h, t, s

        // causal_mask = torch.triu(torch.full((seqlen_q, seqlen_k), -10000.0, device=scores.device), 1)
        // scores = scores + causal_mask.to(dtype=scores.dtype)
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => {
                let _enter = self.span_mask.enter();
                attn_weights.broadcast_add(mask)?
            }
        };
        let attn_weights = {
            let _enter = self.span_softmax.enter();
            candle_nn::ops::softmax_last_dim(&attn_weights)?
        };

        // output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        // attn_weights: b*h,t,s, v: b*h,s,d
        let attn_output = attn_weights.matmul(&v)?;
        // b*h,t,d
        let attn_output = attn_output
            .reshape((b_size, (), seq_len, self.head_dim))?
            .transpose(1, 2)?
            .flatten_from(D::Minus2)?;
        attn_output.apply(&self.out_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct ParallelBlock {
    ln: candle_nn::LayerNorm,
    mixer: MHA,
    mlp: MLP,
    span: tracing::Span,
}

impl ParallelBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln"))?;
        let mixer = MHA::new(cfg, vb.pp("mixer"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            ln,
            mixer,
            mlp,
            span: tracing::span!(tracing::Level::TRACE, "block"),
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = xs;
        let xs = xs.apply(&self.ln)?;
        let attn_outputs = self.mixer.forward(&xs, mask)?;
        let feed_forward_hidden_states = self.mlp.forward(&xs)?;
        attn_outputs + feed_forward_hidden_states + residual
    }

    fn clear_kv_cache(&mut self) {
        self.mixer.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct MixFormerSequentialForCausalLM {
    embedding: Embedding,
    blocks: Vec<ParallelBlock>,
    head: CausalLMHead,
    span: tracing::Span,
}

impl MixFormerSequentialForCausalLM {
    pub fn new_v2(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_head = vb.pp("lm_head");
        let vb = vb.pp("transformer");
        let embedding = Embedding::new(cfg, vb.pp("embd"))?;
        let mut blocks = Vec::new();
        for i in 0..cfg.n_layer {
            let block = ParallelBlock::new(cfg, vb.pp("h").pp(i))?;
            blocks.push(block)
        }
        let head = CausalLMHead::new(cfg, vb_head)?;
        Ok(Self {
            embedding,
            blocks,
            head,
            span: tracing::span!(tracing::Level::TRACE, "mixformer"),
        })
    }

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
            span: tracing::span!(tracing::Level::TRACE, "mixformer"),
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embedding)?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len, xs.dtype(), xs.device())?)
        };
        for block in self.blocks.iter_mut() {
            xs = block.forward(&xs, mask.as_ref())?
        }
        xs.narrow(1, seq_len - 1, 1)?.apply(&self.head)?.squeeze(1)
    }

    pub fn forward_with_img(
        &mut self,
        bos_token: &Tensor,
        xs: &Tensor,
        img_embeds: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = xs.apply(&self.embedding)?;
        let bos_token = bos_token.apply(&self.embedding)?;
        // Python implementation sequence order is <bos token embedding><img embedding><rest of text embedding>
        // https://github.com/vikhyat/moondream/blob/a9d788a20d1543fb1479edc54106e88cff7759d3/moondream/moondream.py#L43-L56
        let mut xs = Tensor::cat(&[bos_token, img_embeds.clone(), xs], 1)?;
        let (_b_size, seq_len, _embds) = xs.dims3()?;
        let mask = Some(get_mask(seq_len, xs.dtype(), xs.device())?);
        for block in self.blocks.iter_mut() {
            xs = block.forward(&xs, mask.as_ref())?
        }
        let xs = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.head)?
            .squeeze(1)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.blocks.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}
