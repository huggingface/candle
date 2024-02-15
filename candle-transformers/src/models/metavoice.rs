use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct SpeakerEncoderConfig {
    mel_window_step: usize,
    mel_n_channels: usize,
    sampling_rate: usize,
    partial_n_frames: usize,
    model_hidden_size: usize,
    model_embedding_size: usize,
    model_num_layers: usize,
}

pub struct SpeakerEncoder {
    lstms: Vec<candle_nn::LSTM>,
    linear: Linear,
}

impl SpeakerEncoder {
    pub fn new(cfg: &SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut lstms = Vec::with_capacity(cfg.model_num_layers);
        let vb_l = vb.pp("lstm");
        for layer_idx in 0..cfg.model_num_layers {
            let c = candle_nn::LSTMConfig {
                layer_idx,
                ..Default::default()
            };
            let lstm = candle_nn::lstm(
                cfg.mel_n_channels,
                cfg.model_hidden_size,
                c,
                vb_l.pp(layer_idx),
            )?;
            lstms.push(lstm)
        }
        let linear = linear(
            cfg.model_hidden_size,
            cfg.model_embedding_size,
            true,
            vb.pp("linear"),
        )?;
        Ok(Self { lstms, linear })
    }

    fn compute_partial_slices(
        _n_samples: usize,
        _rate: f64,
        _min_coverage: f64,
    ) -> Result<(Tensor, Tensor)> {
        todo!()
    }

    pub fn embed_utterance(&self, wav: &[f32], rate: f64, min_coverage: f64) -> Result<Tensor> {
        let (_wav_slices, _mel_slices) =
            Self::compute_partial_slices(wav.len(), rate, min_coverage)?;
        todo!()
    }
}

impl Module for SpeakerEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        use candle_nn::RNN;
        let mut xs = xs.clone();
        for lstm in self.lstms.iter() {
            let res = lstm.seq(&xs)?;
            let res: Vec<_> = res.into_iter().map(|s| s.h().clone()).collect();
            xs = Tensor::stack(&res, 1)?;
        }
        let embeds_raw = xs.apply(&self.linear)?.relu()?;
        // TODO: normalize.
        Ok(embeds_raw)
    }
}

pub struct BPETokenizer;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum AttnKernelType {
    Fa2,
    TorchAttn,
    Hand,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum NonLinearityType {
    Gelu,
    Swiglu,
}

enum Norm {
    RMSNorm(candle_nn::RmsNorm),
    LayerNorm(candle_nn::LayerNorm),
}

fn linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

// https://github.com/metavoiceio/metavoice-src/blob/11550bb4e8a1ad032cc1556cc924f7a4e767cbfa/fam/llm/model.py#L27
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub block_size: usize,
    pub vocab_sizes: Vec<usize>,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub bias: bool,
    pub causal: bool,
    pub spk_emb_on_text: bool,
    pub norm_type: NormType,
    pub rmsnorm_eps: f64,
    pub nonlinearity_type: NonLinearityType,
    pub swiglu_multiple_of: Option<usize>,
    pub attn_kernel_type: AttnKernelType,
    pub kv_cache_enabled: bool,
}

impl GPTConfig {
    // {'n_layer': 24, 'n_head': 16, 'n_embd': 2048, 'block_size': 2048, 'bias': False,
    // 'vocab_sizes': [2562], 'dropout': 0.0, 'causal': True, 'norm_type': 'rmsnorm',
    // 'rmsnorm_eps': 1e-05, 'nonlinearity_type': 'swiglu', 'spk_emb_on_text': True,
    // 'attn_kernel_type': 'torch_attn', 'swiglu_multiple_of': 256, 'spkemb_dropout': 0.1}
    pub fn cfg() -> Self {
        Self {
            block_size: 2048,
            vocab_sizes: vec![2562],
            swiglu_multiple_of: Some(256),
            attn_kernel_type: AttnKernelType::TorchAttn,
            spk_emb_on_text: true,
            nonlinearity_type: NonLinearityType::Swiglu,
            rmsnorm_eps: 1e-5,
            causal: true,
            bias: false,
            n_embd: 2048,
            n_head: 16,
            n_layer: 24,
            norm_type: NormType::RMSNorm,
            kv_cache_enabled: false,
        }
    }
}

impl Norm {
    fn new(cfg: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        match cfg.norm_type {
            NormType::RMSNorm => {
                let rms_norm = candle_nn::rms_norm(cfg.n_embd, cfg.rmsnorm_eps, vb)?;
                Ok(Self::RMSNorm(rms_norm))
            }
            NormType::LayerNorm => {
                let ln_cfg = candle_nn::LayerNormConfig {
                    affine: cfg.bias,
                    ..Default::default()
                };
                let layer_norm = candle_nn::layer_norm(cfg.n_embd, ln_cfg, vb)?;
                Ok(Self::LayerNorm(layer_norm))
            }
        }
    }
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::RMSNorm(m) => m.forward(xs),
            Self::LayerNorm(m) => m.forward(xs),
        }
    }
}

// https://github.com/metavoiceio/metavoice-src/blob/11550bb4e8a1ad032cc1556cc924f7a4e767cbfa/fam/llm/layers/attn.py#L18
struct SelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
}

impl SelfAttention {
    fn new(cfg: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        // The different attention variants are likely to be identical but still we only accept
        // TorchAttn for now.
        if cfg.attn_kernel_type != AttnKernelType::TorchAttn {
            candle::bail!("only TorchAttn is supported")
        }
        let c_attn = linear(cfg.n_embd, cfg.n_embd * 3, cfg.bias, vb.pp("c_attn"))?;
        let c_proj = linear(cfg.n_embd, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
        })
    }
}

impl Module for SelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, c) = xs.dims3()?;
        let c_x = xs
            .apply(&self.c_attn)?
            .reshape((b, t, 3, self.n_head, c / self.n_head))?;
        let q = c_x.i((.., .., 0))?;
        let k = c_x.i((.., .., 1))?;
        let v = c_x.i((.., .., 2))?;
        // TODO: KV cache
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let att = (q.matmul(&k.t()?)? / (k.dim(D::Minus1)? as f64).sqrt())?;
        // TODO: causal mask
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let att = att.matmul(&v)?.transpose(1, 2)?;
        att.reshape((b, t, c))?.apply(&self.c_proj)
    }
}

// https://github.com/metavoiceio/metavoice-src/blob/11550bb4e8a1ad032cc1556cc924f7a4e767cbfa/fam/llm/layers/layers.py#L43
#[allow(clippy::upper_case_acronyms)]
enum MLP {
    Gelu {
        c_fc: Linear,
        c_proj: Linear,
    },
    Swiglu {
        w1: Linear,
        w3: Linear,
        c_proj: Linear,
    },
}

impl MLP {
    fn new(cfg: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = 4 * cfg.n_embd;
        let slf = match cfg.nonlinearity_type {
            NonLinearityType::Gelu => {
                let c_fc = linear(cfg.n_embd, hidden_dim, cfg.bias, vb.pp("c_fc"))?;
                let c_proj = linear(hidden_dim, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
                Self::Gelu { c_fc, c_proj }
            }
            NonLinearityType::Swiglu => {
                let hidden_dim = (2 * hidden_dim) / 3;
                let swiglu_multiple_of = match cfg.swiglu_multiple_of {
                    None => candle::bail!("swiglu-multiple-of has to be set"),
                    Some(smo) => smo,
                };
                let hidden_dim =
                    swiglu_multiple_of * (hidden_dim + swiglu_multiple_of - 1) / swiglu_multiple_of;
                let w1 = linear(cfg.n_embd, hidden_dim, cfg.bias, vb.pp("w1"))?;
                let w3 = linear(cfg.n_embd, hidden_dim, cfg.bias, vb.pp("w3"))?;
                let c_proj = linear(hidden_dim, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
                Self::Swiglu { w1, w3, c_proj }
            }
        };
        Ok(slf)
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu { c_fc, c_proj } => xs.apply(c_fc)?.gelu()?.apply(c_proj),
            Self::Swiglu { w1, w3, c_proj } => {
                let w1 = xs.apply(w1)?;
                let w3 = xs.apply(w3)?;
                (w1.silu()? * w3)?.apply(c_proj)
            }
        }
    }
}

// https://github.com/metavoiceio/metavoice-src/blob/11550bb4e8a1ad032cc1556cc924f7a4e767cbfa/fam/llm/layers/combined.py#L7
struct Block {
    ln_1: Norm,
    ln_2: Norm,
    attn: SelfAttention,
    mlp: MLP,
}

impl Block {
    fn new(cfg: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let ln_1 = Norm::new(cfg, vb.pp("ln_1"))?;
        let ln_2 = Norm::new(cfg, vb.pp("ln_2"))?;
        let attn = SelfAttention::new(cfg, vb.pp("attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Block {
            ln_1,
            ln_2,
            attn,
            mlp,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + xs.apply(&self.ln_1)?.apply(&self.attn))?;
        let xs = (&xs + xs.apply(&self.ln_2)?.apply(&self.mlp))?;
        Ok(xs)
    }
}

// https://github.com/metavoiceio/metavoice-src/blob/11550bb4e8a1ad032cc1556cc924f7a4e767cbfa/fam/llm/model.py#L79
#[allow(unused)]
#[allow(clippy::upper_case_acronyms)]
pub struct GPT {
    wtes: Vec<candle_nn::Embedding>,
    wpe: candle_nn::Embedding,
    h: Vec<Block>,
    ln_f: Norm,
    lm_heads: Vec<Linear>,
    cfg: GPTConfig,
}

impl GPT {
    pub fn new(cfg: GPTConfig, vb: VarBuilder) -> Result<Self> {
        let ln_f = Norm::new(&cfg, vb.pp("ln_f"))?;
        let mut wtes = Vec::with_capacity(cfg.vocab_sizes.len());
        let vb_w = vb.pp("wtes");
        for (idx, vocab_size) in cfg.vocab_sizes.iter().enumerate() {
            let wte = candle_nn::embedding(*vocab_size, cfg.n_embd, vb_w.pp(idx))?;
            wtes.push(wte)
        }
        let wpe = candle_nn::embedding(cfg.block_size, cfg.n_embd, vb.pp("wpe"))?;

        let mut h = Vec::with_capacity(cfg.n_layer);
        let vb_h = vb.pp("h");
        for idx in 0..cfg.n_layer {
            let block = Block::new(&cfg, vb_h.pp(idx))?;
            h.push(block)
        }

        let mut lm_heads = Vec::with_capacity(cfg.vocab_sizes.len());
        let vb_l = vb.pp("lm_heads");
        for (idx, vocab_size) in cfg.vocab_sizes.iter().enumerate() {
            let head = linear_no_bias(cfg.n_embd, *vocab_size, vb_l.pp(idx))?;
            lm_heads.push(head)
        }
        Ok(Self {
            wtes,
            wpe,
            h,
            ln_f,
            lm_heads,
            cfg,
        })
    }

    pub fn config(&self) -> &GPTConfig {
        &self.cfg
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Vec<Tensor>> {
        let device = idx.device();
        let (b, _num_hierarchies, t) = idx.dims3()?;
        let pos = Tensor::arange(0u32, t as u32, device)?;
        let pos_emb = pos.apply(&self.wpe)?;
        let mut tok_emb = Tensor::zeros((b, t, self.cfg.n_embd), DType::F32, device)?;
        for (wte_idx, wte) in self.wtes.iter().enumerate() {
            let emb = idx.i((.., wte_idx, ..))?.apply(wte)?;
            tok_emb = (tok_emb + emb)?;
        }
        // TODO: speaker embs.
        let spk_emb = 0f64;
        let mut xs = ((pos_emb + tok_emb)? + spk_emb)?;
        for block in self.h.iter() {
            xs = xs.apply(block)?
        }
        let xs = xs.apply(&self.ln_f)?;
        let mut logits = Vec::with_capacity(self.lm_heads.len());
        for lm_head in self.lm_heads.iter() {
            // causal mode only.
            let ys = xs.narrow(1, t - 1, 1)?.apply(lm_head)?;
            logits.push(ys)
        }
        Ok(logits)
    }
}
