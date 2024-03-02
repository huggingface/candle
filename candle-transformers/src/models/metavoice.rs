use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_b, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

// Equivalent to torch.repeat_interleave
fn repeat_interleave(img: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let img = img.unsqueeze(dim + 1)?;
    let mut dims = img.dims().to_vec();
    dims[dim + 1] = repeats;
    img.broadcast_as(dims)?.flatten(dim, dim + 1)
}
pub mod speaker_encoder {
    use super::*;

    #[derive(Debug, Clone, serde::Deserialize)]
    pub struct Config {
        pub mel_window_step: usize,
        pub mel_n_channels: usize,
        pub sampling_rate: usize,
        pub partial_n_frames: usize,
        pub model_hidden_size: usize,
        pub model_embedding_size: usize,
        pub model_num_layers: usize,
    }

    pub struct Model {
        lstms: Vec<candle_nn::LSTM>,
        linear: Linear,
    }

    impl Model {
        pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
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
            let linear = linear_b(
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

    impl Module for Model {
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
}

pub mod gpt {
    use super::*;

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

    // https://github.com/metavoiceio/metavoice-src/blob/11550bb4e8a1ad032cc1556cc924f7a4e767cbfa/fam/llm/model.py#L27
    #[derive(Debug, Clone)]
    pub struct Config {
        pub block_size: usize,
        pub vocab_sizes: Vec<usize>,
        pub target_vocab_sizes: Vec<usize>,
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

    impl Config {
        pub fn cfg1b_v0_1() -> Self {
            Self {
                n_layer: 6,
                n_head: 6,
                n_embd: 384,
                block_size: 1024,
                bias: false,
                vocab_sizes: vec![1538, 1025],
                causal: false,
                target_vocab_sizes: vec![1025, 1025, 1025, 1025, 1025, 1025],
                swiglu_multiple_of: Some(256),
                norm_type: NormType::RMSNorm,
                kv_cache_enabled: false,
                attn_kernel_type: AttnKernelType::TorchAttn,
                spk_emb_on_text: true,
                nonlinearity_type: NonLinearityType::Gelu,
                rmsnorm_eps: 1e-5,
            }
        }
    }

    impl Norm {
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
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
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
            // The different attention variants are likely to be identical but still we only accept
            // TorchAttn for now.
            if cfg.attn_kernel_type != AttnKernelType::TorchAttn {
                candle::bail!("only TorchAttn is supported")
            }
            if cfg.kv_cache_enabled {
                candle::bail!("kv_cache_enabled=true is not supported")
            }
            let c_attn = linear_b(cfg.n_embd, cfg.n_embd * 3, cfg.bias, vb.pp("c_attn"))?;
            let c_proj = linear_b(cfg.n_embd, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
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
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
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
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
            let hidden_dim = 4 * cfg.n_embd;
            let slf = match cfg.nonlinearity_type {
                NonLinearityType::Gelu => {
                    let c_fc = linear_b(cfg.n_embd, hidden_dim, cfg.bias, vb.pp("c_fc"))?;
                    let c_proj = linear_b(hidden_dim, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
                    Self::Gelu { c_fc, c_proj }
                }
                NonLinearityType::Swiglu => {
                    let hidden_dim = (2 * hidden_dim) / 3;
                    let swiglu_multiple_of = match cfg.swiglu_multiple_of {
                        None => candle::bail!("swiglu-multiple-of has to be set"),
                        Some(smo) => smo,
                    };
                    let hidden_dim = swiglu_multiple_of * (hidden_dim + swiglu_multiple_of - 1)
                        / swiglu_multiple_of;
                    let w1 = linear_b(cfg.n_embd, hidden_dim, cfg.bias, vb.pp("w1"))?;
                    let w3 = linear_b(cfg.n_embd, hidden_dim, cfg.bias, vb.pp("w3"))?;
                    let c_proj = linear_b(hidden_dim, cfg.n_embd, cfg.bias, vb.pp("c_proj"))?;
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
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
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
    #[allow(clippy::upper_case_acronyms)]
    pub struct Model {
        wtes: Vec<candle_nn::Embedding>,
        wpe: candle_nn::Embedding,
        h: Vec<Block>,
        ln_f: Norm,
        lm_heads: Vec<Linear>,
        cfg: Config,
    }

    impl Model {
        pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
            let vb_t = vb.pp("transformer");
            let ln_f = Norm::new(&cfg, vb_t.pp("ln_f"))?;
            let mut wtes = Vec::with_capacity(cfg.vocab_sizes.len());
            let vb_w = vb_t.pp("wtes");
            for (idx, vocab_size) in cfg.vocab_sizes.iter().enumerate() {
                let wte = candle_nn::embedding(*vocab_size, cfg.n_embd, vb_w.pp(idx))?;
                wtes.push(wte)
            }
            let wpe = candle_nn::embedding(cfg.block_size, cfg.n_embd, vb_t.pp("wpe"))?;

            let mut h = Vec::with_capacity(cfg.n_layer);
            let vb_h = vb_t.pp("h");
            for idx in 0..cfg.n_layer {
                let block = Block::new(&cfg, vb_h.pp(idx))?;
                h.push(block)
            }

            let mut lm_heads = Vec::with_capacity(cfg.target_vocab_sizes.len());
            let vb_l = vb.pp("lm_heads");
            for (idx, vocab_size) in cfg.target_vocab_sizes.iter().enumerate() {
                let head = linear_b(cfg.n_embd, *vocab_size, false, vb_l.pp(idx))?;
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

        pub fn config(&self) -> &Config {
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
            let mut xs = (pos_emb.broadcast_add(&tok_emb)? + spk_emb)?;
            for block in self.h.iter() {
                xs = xs.apply(block)?
            }
            let xs = xs.apply(&self.ln_f)?;
            let mut logits = Vec::with_capacity(self.lm_heads.len());
            for lm_head in self.lm_heads.iter() {
                // non-causal mode only.
                let ys = xs.apply(lm_head)?;
                logits.push(ys)
            }
            Ok(logits)
        }
    }
}

pub mod transformer {
    use super::*;

    #[derive(Debug, Clone, serde::Deserialize)]
    pub struct Config {
        pub block_size: usize,
        pub vocab_size: usize,
        pub n_layer: usize,
        pub n_head: usize,
        pub dim: usize,
        pub speaker_emb_dim: usize,
        pub intermediate_size: Option<usize>,
        pub n_local_heads: Option<usize>,
        pub norm_eps: f64,
    }

    impl Config {
        pub fn cfg1b_v0_1() -> Self {
            Self {
                n_layer: 24,
                n_head: 16,
                dim: 2048,
                vocab_size: 2562,
                speaker_emb_dim: 256,
                block_size: 2048,
                intermediate_size: None,
                n_local_heads: None,
                norm_eps: 1e-5,
            }
        }

        fn n_local_heads(&self) -> usize {
            self.n_local_heads.unwrap_or(self.n_head)
        }

        fn head_dim(&self) -> usize {
            self.dim / self.n_head
        }

        fn intermediate_size(&self) -> usize {
            match self.intermediate_size {
                Some(intermediate_size) => intermediate_size,
                None => {
                    let hidden_dim = self.dim * 4;
                    let n_hidden = ((2 * hidden_dim) as f64 / 3.) as usize;
                    (n_hidden + 255) / 256 * 256
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    struct FeedForward {
        w1: Linear,
        w2: Linear,
        w3: Linear,
    }

    impl FeedForward {
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
            let i_size = cfg.intermediate_size();
            let w1 = linear_b(cfg.dim, i_size, false, vb.pp("swiglu.w1"))?;
            let w2 = linear_b(i_size, cfg.dim, false, vb.pp("w2"))?;
            let w3 = linear_b(cfg.dim, i_size, false, vb.pp("swiglu.w3"))?;
            Ok(Self { w1, w2, w3 })
        }
    }

    impl Module for FeedForward {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let swiglu = (candle_nn::ops::silu(&xs.apply(&self.w1)?)? * xs.apply(&self.w3))?;
            swiglu.apply(&self.w2)
        }
    }

    #[derive(Debug, Clone)]
    struct Attention {
        wqkv: Linear,
        wo: Linear,
        dim: usize,
        kv_size: usize,
        n_local_heads: usize,
        head_dim: usize,
        n_head: usize,
        kv_cache: Option<(Tensor, Tensor)>,
    }

    impl Attention {
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
            let n_local_heads = cfg.n_local_heads();
            let head_dim = cfg.head_dim();
            let total_head_dim = (cfg.n_head + 2 * n_local_heads) * head_dim;
            let wqkv = linear_b(cfg.dim, total_head_dim, false, vb.pp("wqkv"))?;
            let wo = linear_b(cfg.dim, cfg.dim, false, vb.pp("wo"))?;
            Ok(Self {
                wqkv,
                wo,
                dim: cfg.dim,
                kv_size: n_local_heads * head_dim,
                n_local_heads,
                head_dim,
                n_head: cfg.n_head,
                kv_cache: None,
            })
        }

        fn forward(&mut self, xs: &Tensor, _pos: usize, mask: &Tensor) -> Result<Tensor> {
            let (b_sz, seqlen, _) = xs.dims3()?;

            let qkv = xs.apply(&self.wqkv)?;
            let q = qkv.narrow(D::Minus1, 0, self.dim)?;
            let k = qkv.narrow(D::Minus1, self.dim, self.kv_size)?;
            let v = qkv.narrow(D::Minus1, self.dim + self.kv_size, self.kv_size)?;
            let q = q
                .reshape((b_sz, seqlen, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b_sz, seqlen, self.n_local_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seqlen, self.n_local_heads, self.head_dim))?
                .transpose(1, 2)?;

            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k, &k], 2)?;
                    let v = Tensor::cat(&[prev_v, &v], 2)?;
                    (k, v)
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));

            let k = repeat_interleave(&k, self.n_head / self.n_local_heads, 1)?;
            let v = repeat_interleave(&v, self.n_head / self.n_local_heads, 1)?;

            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = attn_weights.broadcast_add(mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, seqlen, self.dim))?
                .apply(&self.wo)
        }

        fn clear_kv_cache(&mut self) {
            self.kv_cache = None
        }
    }

    #[derive(Debug, Clone)]
    struct Block {
        attention: Attention,
        feed_forward: FeedForward,
        ffn_norm: RmsNorm,
        attention_norm: RmsNorm,
    }

    impl Block {
        fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
            let attention = Attention::new(cfg, vb.pp("attention"))?;
            let feed_forward = FeedForward::new(cfg, vb.pp("feed_forward"))?;
            let ffn_norm = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("ffn_norm"))?;
            let attention_norm = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("attention_norm"))?;
            Ok(Self {
                attention,
                feed_forward,
                ffn_norm,
                attention_norm,
            })
        }

        fn forward(&mut self, xs: &Tensor, pos: usize, mask: &Tensor) -> Result<Tensor> {
            let hs = xs.apply(&self.attention_norm)?;
            let hs = (xs + self.attention.forward(&hs, pos, mask))?;
            &hs + hs.apply(&self.ffn_norm)?.apply(&self.feed_forward)
        }

        fn clear_kv_cache(&mut self) {
            self.attention.clear_kv_cache()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Model {
        tok_embeddings: Embedding,
        pos_embeddings: Embedding,
        speaker_cond_pos: Linear,
        layers: Vec<Block>,
        norm: RmsNorm,
        output: Linear,
        spk_cond_mask: Tensor,
    }

    impl Model {
        pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
            let tok_embeddings = embedding(cfg.vocab_size, cfg.dim, vb.pp("tok_embeddings"))?;
            let pos_embeddings = embedding(cfg.block_size, cfg.dim, vb.pp("pos_embeddings"))?;
            let speaker_cond_pos = linear_b(
                cfg.speaker_emb_dim,
                cfg.dim,
                false,
                vb.pp("speaker_cond_pos"),
            )?;
            let mut layers = Vec::with_capacity(cfg.n_layer);
            let vb_l = vb.pp("layers");
            for layer_idx in 0..cfg.n_layer {
                let layer = Block::new(cfg, vb_l.pp(layer_idx))?;
                layers.push(layer)
            }
            let norm = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("norm"))?;
            let output = linear_b(cfg.dim, cfg.vocab_size, false, vb.pp("output"))?;
            let spk_cond_mask = Tensor::cat(
                &[
                    Tensor::ones((1, 1, cfg.dim), DType::F32, vb.device())?,
                    Tensor::zeros((1, 1, cfg.dim), DType::F32, vb.device())?,
                ],
                0,
            )?;
            Ok(Self {
                tok_embeddings,
                pos_embeddings,
                speaker_cond_pos,
                layers,
                norm,
                output,
                spk_cond_mask,
            })
        }

        pub fn clear_kv_cache(&mut self) {
            for layer in self.layers.iter_mut() {
                layer.clear_kv_cache()
            }
        }

        pub fn forward(&mut self, xs: &Tensor, spk_emb: &Tensor, pos: usize) -> Result<Tensor> {
            let (_b_sz, seqlen) = xs.dims2()?;
            let mask: Vec<_> = (0..seqlen)
                .flat_map(|i| (0..seqlen).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
                .collect();
            let mask = Tensor::from_slice(&mask, (1, 1, seqlen, seqlen), xs.device())?;
            let input_pos = Tensor::arange(pos as u32, (pos + seqlen) as u32, xs.device())?;
            let tok_embeddings = xs.apply(&self.tok_embeddings)?;
            let pos_embeddings = input_pos.apply(&self.pos_embeddings)?;
            let mut xs = tok_embeddings
                .broadcast_add(&pos_embeddings)?
                .broadcast_add(
                    &spk_emb
                        .apply(&self.speaker_cond_pos)?
                        .broadcast_mul(&self.spk_cond_mask)?,
                )?;
            for layer in self.layers.iter_mut() {
                xs = layer.forward(&xs, pos, &mask)?
            }
            xs.narrow(1, seqlen - 1, 1)?
                .apply(&self.norm)?
                .apply(&self.output)
        }
    }
}

pub mod adapters {
    // https://github.com/metavoiceio/metavoice-src/blob/9078234c496d76adbec06df789b6b04b1875f129/fam/llm/adapters/tilted_encodec.py
    pub struct TiltedEncodec {
        end_of_audio_token: u32,
    }

    impl TiltedEncodec {
        pub fn new(end_of_audio_token: u32) -> Self {
            Self { end_of_audio_token }
        }

        pub fn decode(&self, tokens: &[Vec<u32>]) -> (Vec<u32>, Vec<Vec<u32>>) {
            let mut text_ids = vec![];
            let mut extracted_audio_ids = vec![];
            let mut min_audio_ids_len = usize::MAX;
            for (book_id, tokens) in tokens.iter().enumerate() {
                let mut audio_ids = vec![];
                for &t in tokens.iter() {
                    #[allow(clippy::comparison_chain)]
                    if t > self.end_of_audio_token {
                        if book_id == 0 {
                            text_ids.push(t)
                        }
                    } else if t < self.end_of_audio_token {
                        audio_ids.push(t)
                    }
                }
                min_audio_ids_len = usize::min(min_audio_ids_len, audio_ids.len());
                extracted_audio_ids.push(audio_ids)
            }
            for audio_ids in extracted_audio_ids.iter_mut() {
                audio_ids.truncate(min_audio_ids_len)
            }
            (text_ids, extracted_audio_ids)
        }
    }

    // https://github.com/metavoiceio/metavoice-src/blob/9078234c496d76adbec06df789b6b04b1875f129/fam/llm/adapters/flattened_encodec.py#L4
    pub struct FlattenedInterleavedEncodec2Codebook {
        end_of_audio_token: u32,
    }

    impl FlattenedInterleavedEncodec2Codebook {
        pub fn new(end_of_audio_token: u32) -> Self {
            Self { end_of_audio_token }
        }

        pub fn decode(&self, tokens: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            let mut text_ids = vec![];
            let mut audio_ids1 = vec![];
            let mut audio_ids2 = vec![];
            for &t in tokens.iter() {
                #[allow(clippy::comparison_chain)]
                if t < self.end_of_audio_token {
                    audio_ids1.push(t)
                } else if t < 2 * self.end_of_audio_token {
                    audio_ids2.push(t - self.end_of_audio_token)
                } else {
                    text_ids.push(t)
                }
            }
            (text_ids, audio_ids1, audio_ids2)
        }
    }
}
