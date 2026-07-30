//! CodePredictor: generates 15 acoustic codec tokens per semantic frame.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Linear, RmsNorm, VarBuilder};
use candle_nn::{linear_no_bias as lnb, rms_norm as rnorm};

use super::config::{ParsedModelConfig, Qwen3TTSConfig};
use super::kv_cache::{AnyKVCache, KVCache, PreAllocKVCache};
use super::create_causal_mask;

// ── CodePredictorConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub num_code_groups: usize,
    /// When Some, a projection from this dim → hidden_size is used.
    pub codec_embed_dim: Option<usize>,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab_size: 2048,
            num_code_groups: 16,
            codec_embed_dim: None,
        }
    }
}

impl CodePredictorConfig {
    pub fn from_parsed(p: &ParsedModelConfig) -> Self {
        let codec_embed_dim = if p.talker_hidden_size != p.cp_hidden_size {
            Some(p.talker_hidden_size)
        } else {
            None
        };
        Self {
            hidden_size: p.cp_hidden_size,
            intermediate_size: p.cp_intermediate_size,
            num_hidden_layers: p.cp_num_hidden_layers,
            num_attention_heads: p.cp_num_attention_heads,
            num_key_value_heads: p.cp_num_key_value_heads,
            head_dim: p.cp_head_dim,
            rms_norm_eps: p.cp_rms_norm_eps,
            rope_theta: p.cp_rope_theta,
            vocab_size: p.cp_vocab_size,
            num_code_groups: p.cp_num_code_groups,
            codec_embed_dim,
        }
    }

    pub fn codec_embed_dim(&self) -> usize {
        self.codec_embed_dim.unwrap_or(self.hidden_size)
    }

    fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            vocab_size: self.vocab_size,
            ..Default::default()
        }
    }
}

// ── Attention + MLP + Layer (local, lighter copy) ─────────────────────────────

fn apply_rope_local(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle::D::Minus1)?;
    let x1 = x.narrow(candle::D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(candle::D::Minus1, d / 2, d / 2)?;
    let cos = cos
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;
    let sin = sin
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;
    let r = Tensor::cat(
        &[
            &(x1.mul(&cos)? - x2.mul(&sin)?)?,
            &(x2.mul(&cos)? + x1.mul(&sin)?)?,
        ],
        candle::D::Minus1,
    )?;
    Ok(r)
}

struct CPAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f64,
}

impl CPAttention {
    fn new(cfg: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_kv_heads();
        let d = cfg.head_dim();
        Ok(Self {
            q_proj: lnb(h, nq * d, vb.pp("q_proj"))?,
            k_proj: lnb(h, nkv * d, vb.pp("k_proj"))?,
            v_proj: lnb(h, nkv * d, vb.pp("v_proj"))?,
            o_proj: lnb(nq * d, h, vb.pp("o_proj"))?,
            q_norm: rnorm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rnorm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nq,
            num_kv: nkv,
            head_dim: d,
            scale: 1.0 / (d as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.num_kv, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_kv, self.head_dim))?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let cos_s = rope_cos.i(offset..offset + s)?;
        let sin_s = rope_sin.i(offset..offset + s)?;
        let q = apply_rope_local(&q, &cos_s, &sin_s)?;
        let k = apply_rope_local(&k, &cos_s, &sin_s)?;

        let (k, v, mask) = if let Some(c) = kv_cache {
            let res = c.update(&k, &v)?;
            let mask = if res.window_start > 0 {
                let win_len = res.k.dim(2)?;
                let m = super::kv_cache::create_window_causal_mask(
                    offset, s, res.window_start, win_len, res.k.device())?
                    .to_dtype(res.k.dtype())?;
                Some(m)
            } else {
                mask.cloned()
            };
            (res.k, res.v, mask)
        } else {
            (k, v, mask.cloned())
        };

        let n_rep = self.num_heads / self.num_kv;
        let k = if n_rep > 1 {
            let (b2, h, sq, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b2, h, n_rep, sq, d))?
                .reshape((b2, h * n_rep, sq, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b2, h, sq, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b2, h, n_rep, sq, d))?
                .reshape((b2, h * n_rep, sq, d))?
        } else {
            v
        };

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn =
            (q.matmul(&k.transpose(candle::D::Minus2, candle::D::Minus1)?)? * self.scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(&m.to_dtype(attn.dtype())?)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

struct CPMLP {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl CPMLP {
    fn new(cfg: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: lnb(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: lnb(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: lnb(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?)
    }
}

struct CPLayer {
    attn: CPAttention,
    mlp: CPMLP,
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl CPLayer {
    fn new(cfg: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: CPAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: CPMLP::new(cfg, vb.pp("mlp"))?,
            pre_norm: rnorm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_norm: rnorm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self
            .attn
            .forward(&self.pre_norm.forward(x)?, cos, sin, mask, kv_cache, offset)?;
        let h = (h + residual)?;
        let residual = &h;
        let h = self.mlp.forward(&self.post_norm.forward(&h)?)?;
        Ok((h + residual)?)
    }
}

// ── CodePredictor ─────────────────────────────────────────────────────────────

/// Generates acoustic tokens (RVQ groups 2–16) for each semantic frame.
pub struct CodePredictor {
    codec_embeddings: Vec<Embedding>,
    projection: Option<Linear>,
    layers: Vec<CPLayer>,
    norm: RmsNorm,
    lm_heads: Vec<Linear>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    config: CodePredictorConfig,
    prefill_mask: Tensor,
    device: Device,
    dtype: DType,
}

impl CodePredictor {
    pub fn new(config: CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let layer_cfg = config.to_layer_config();
        let n_acoustic = config.num_code_groups - 1;
        let embed_dim = config.codec_embed_dim();

        let mut codec_embeddings = Vec::with_capacity(n_acoustic);
        for i in 0..n_acoustic {
            codec_embeddings.push(embedding(
                config.vocab_size,
                embed_dim,
                vb.pp(format!("model.codec_embedding.{i}")),
            )?);
        }

        let projection = if embed_dim != config.hidden_size {
            Some(candle_nn::linear(
                embed_dim,
                config.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };

        let layers = (0..config.num_hidden_layers)
            .map(|i| CPLayer::new(&layer_cfg, vb.pp(format!("model.layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let norm = rnorm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        let mut lm_heads = Vec::with_capacity(n_acoustic);
        for i in 0..n_acoustic {
            lm_heads.push(lnb(config.hidden_size, config.vocab_size, vb.pp(format!("lm_head.{i}")))?);
        }

        // Build RoPE tables (max 1024 positions is plenty for CP)
        let max_seq = 1024_usize;
        let inv_freq: Vec<f32> = (0..config.head_dim)
            .step_by(2)
            .map(|i| 1.0 / (config.rope_theta as f32).powf(i as f32 / config.head_dim as f32))
            .collect();
        let inv_freq_t = Tensor::new(inv_freq.as_slice(), vb.device())?;
        let positions: Vec<f32> = (0..max_seq).map(|i| i as f32).collect();
        let pos_t = Tensor::new(positions.as_slice(), vb.device())?.unsqueeze(1)?;
        let freqs = pos_t.matmul(&inv_freq_t.unsqueeze(0)?)?;
        let rope_cos = freqs.cos()?;
        let rope_sin = freqs.sin()?;

        let prefill_mask = create_causal_mask(2, 0, vb.device())?;
        let device = vb.device().clone();
        let dtype = vb.dtype();

        Ok(Self {
            codec_embeddings,
            projection,
            layers,
            norm,
            lm_heads,
            rope_cos,
            rope_sin,
            config,
            prefill_mask,
            device,
            dtype,
        })
    }

    /// Generate all 15 acoustic tokens for one semantic frame.
    ///
    /// Returns a `[num_acoustic]` U32 tensor on-device.
    pub fn generate_acoustic_codes(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cp_kv_caches: &mut [AnyKVCache],
    ) -> Result<Tensor> {
        // Reset caches from previous frame
        for c in cp_kv_caches.iter_mut() {
            c.reset();
        }

        let n = self.config.num_code_groups - 1; // 15
        let device = talker_hidden.device();

        // Prefill: [talker_hidden, semantic_embed]
        let input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;
        let input = self.maybe_project(&input)?;
        let s = input.dim(1)?;
        let dynamic_mask;
        let mask = if s == 2 {
            &self.prefill_mask
        } else {
            dynamic_mask = create_causal_mask(s, 0, device)?;
            &dynamic_mask
        };
        let mut h = input;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(
                &h,
                &self.rope_cos,
                &self.rope_sin,
                Some(mask),
                Some(&mut cp_kv_caches[i]),
                0,
            )?;
        }
        h = self.norm.forward(&h)?;

        // Predict first acoustic code
        let last = h.i((.., s - 1..s, ..))?;
        let first_code = self.lm_heads[0].forward(&last)?.argmax(D::Minus1)?.flatten_all()?;

        let mut all_codes = Tensor::zeros(n, DType::U32, device)?;
        all_codes = all_codes.slice_assign(&[0..1], &first_code)?;
        let mut prev_code = first_code;

        // Autoregressive decode of remaining 14 codes
        let mut offset = s;
        for g in 1..n {
            let embed = self.codec_embeddings[g - 1].forward(&prev_code)?.unsqueeze(0)?;
            let embed = self.maybe_project(&embed)?;
            let mut hh = embed;
            for (i, layer) in self.layers.iter().enumerate() {
                hh = layer.forward(
                    &hh,
                    &self.rope_cos,
                    &self.rope_sin,
                    None,
                    Some(&mut cp_kv_caches[i]),
                    offset,
                )?;
            }
            hh = self.norm.forward(&hh)?;
            let code = self.lm_heads[g].forward(&hh)?.argmax(D::Minus1)?.flatten_all()?;
            all_codes = all_codes.slice_assign(&[g..g + 1], &code)?;
            prev_code = code;
            offset += 1;
        }

        Ok(all_codes)
    }

    pub fn embed_codes_for_group(&self, group_idx: usize, codes: &Tensor) -> Result<Tensor> {
        if group_idx >= self.codec_embeddings.len() {
            candle::bail!(
                "Invalid group_idx {} (max {})",
                group_idx,
                self.codec_embeddings.len() - 1
            );
        }
        Ok(self.codec_embeddings[group_idx].forward(codes)?.unsqueeze(0)?)
    }

    /// Sum acoustic embeddings from a GPU `[num_acoustic]` tensor.
    pub fn get_acoustic_embeddings_sum_from_tensor(&self, codes: &Tensor) -> Result<Tensor> {
        let n = codes.dim(0)?;
        if n != self.codec_embeddings.len() {
            candle::bail!("Expected {} acoustic codes, got {}", self.codec_embeddings.len(), n);
        }
        let first = self.codec_embeddings[0]
            .forward(&codes.narrow(0, 0, 1)?)?
            .unsqueeze(0)?;
        (1..n).try_fold(first, |acc, i| {
            let e = self.codec_embeddings[i]
                .forward(&codes.narrow(0, i, 1)?)?
                .unsqueeze(0)?;
            acc.add(&e)
        })
    }

    pub fn new_kv_caches(&self) -> Vec<AnyKVCache> {
        const CP_MAX: usize = 17;
        (0..self.config.num_hidden_layers)
            .map(|_| {
                if self.device.is_cuda() || self.device.is_metal() {
                    PreAllocKVCache::new(
                        1,
                        self.config.num_key_value_heads,
                        CP_MAX,
                        self.config.head_dim,
                        self.dtype,
                        &self.device,
                    )
                    .map(AnyKVCache::PreAlloc)
                    .unwrap_or_else(|_| AnyKVCache::Concat(KVCache::new()))
                } else {
                    AnyKVCache::Concat(KVCache::new())
                }
            })
            .collect()
    }

    fn maybe_project(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(p) = &self.projection {
            p.forward(x)
        } else {
            Ok(x.clone())
        }
    }
}
