use super::quantized_qwen3::{Gguf, RotaryEmbedding};
use super::with_tracing::QMatMul;
use crate::fused_moe::{FusedMoeGGUF, MoeCfg};
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::gguf_file;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::Linear;
use candle_nn::{Embedding, Module};
use std::sync::Arc;
#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoeGGUF),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
        }
    }
}

pub struct QuantizedAttention {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    attention_wo: QMatMul,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    num_kv_groups: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    dtype: DType,
    kv_cache: ConcatKvCache,
}

impl QuantizedAttention {
    pub fn new<R: std::io::Seek + std::io::Read>(
        gg: &mut Gguf<R>,
        prefix: &str,
        dtype: DType,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        device: &Device,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let attention_wq = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let attention_wk = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let attention_wv = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;

        let attention_bq = gg.tensor(&format!("{prefix}.attn_q.bias"));
        let attention_bk = gg.tensor(&format!("{prefix}.attn_k.bias"));
        let attention_bv = gg.tensor(&format!("{prefix}.attn_v.bias"));

        let attention_bq = if attention_bq.is_ok() {
            Some(
                attention_bq
                    .unwrap()
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let attention_bk = if attention_bk.is_ok() {
            Some(
                attention_bk
                    .unwrap()
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let attention_bv = if attention_bv.is_ok() {
            Some(
                attention_bv
                    .unwrap()
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let attention_wo = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
        let q_norm = Some(gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?);
        let k_norm = Some(gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?);
        let kv_cache = ConcatKvCache::new(2);
        Ok(QuantizedAttention {
            attention_wq,
            attention_wk,
            attention_wv,
            attention_bq,
            attention_bk,
            attention_bv,
            attention_wo,
            q_norm,
            k_norm,
            n_head: num_heads,
            n_kv_head: num_kv_heads,
            head_dim,
            num_kv_groups,
            rotary_emb: rotary_emb.clone(),
            dtype,
            kv_cache,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        input_pos: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let in_dtype = x.dtype();
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = if self.attention_bq.is_some() {
            q.broadcast_add(self.attention_bq.as_ref().unwrap())?
        } else {
            q
        };

        let k = if self.attention_bk.is_some() {
            k.broadcast_add(self.attention_bk.as_ref().unwrap())?
        } else {
            k
        };

        let v = if self.attention_bv.is_some() {
            v.broadcast_add(self.attention_bv.as_ref().unwrap())?
        } else {
            v
        };

        let q = q
            .reshape((1, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // Per‑head RMSNorm in qwen3
            let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
            let k_flat = k.flatten(0, 2)?;

            // q_norm and k_norm weights stored in f32 format in qwen3 gguf
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;

            let q = q_flat.reshape((1, self.n_head, seq_len, self.head_dim))?;
            let k = k_flat.reshape((1, self.n_kv_head, seq_len, self.head_dim))?;

            (q, k)
        } else {
            (q, k)
        };

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let (q, k) = self.rotary_emb.apply(&q, &k, input_pos)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)
        let reshaped_ctx =
            ctx.transpose(1, 2)?
                .reshape((b, seq_len, self.n_head * self.head_dim))?;

        self.attention_wo.forward(&reshaped_ctx.to_dtype(in_dtype)?)
    }
}

struct LayerWeights {
    self_attn: QuantizedAttention,
    attention_norm: RmsNorm,
    mlp: MoeOrMlp,
    ffn_norm: RmsNorm,
}

impl LayerWeights {
    fn forward_attn(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        self.self_attn.forward(x, mask, offset)
    }
}

pub struct GGUFQWenMoE {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    dtype: DType,
    device: Device,
}

impl GGUFQWenMoE {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let arch = md_get("general.architecture")?.to_string()?;

        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;

        let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let head_dim = if head_dim.is_ok() {
            head_dim.unwrap().to_u32()? as usize
        } else {
            embedding_length / head_count
        };
        let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let expert_shared_feed_forward_length =
            md_get(format!("{arch}.expert_shared_feed_forward_length").as_str());
        let shared_expert_intermediate_size = match expert_shared_feed_forward_length {
            Ok(length) => {
                if length.to_u32()? > 0 {
                    Some(length.to_u32()? as usize)
                } else {
                    None
                }
            }
            _ => None,
        };

        let moe_cfg = MoeCfg {
            moe_intermediate_size: md_get(format!("{arch}.expert_feed_forward_length").as_str())?
                .to_u32()? as usize,
            num_experts: md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize,
            norm_topk_prob: shared_expert_intermediate_size.is_none(),
            num_experts_per_tok: md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()?
                as usize,
            hidden_size: head_dim,
            act: candle_nn::Activation::Silu,
            decoder_sparse_step: None,
        };

        let tok_embeddings = gg.tensor("token_embd.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let output = match gg.qmatmul("output.weight") {
            Ok(v) => v,
            _ => {
                // use tie_word_embeddings
                gg.qmatmul("token_embd.weight")?
            }
        };

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            context_length,
            rope_freq_base as f64,
            device,
        )?);
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let mlp = if moe_cfg.num_experts > 0
                && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0
            {
                let gate_ws = gg
                    .tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
                    .dequantize(&device)?
                    .to_dtype(DType::F32)?;
                let gate = Linear::new(gate_ws, None);
                let gate_experts = Arc::new(gg.tensor(&format!("{prefix}.ffn_gate_exps.weight"))?);
                let up_experts = Arc::new(gg.tensor(&format!("{prefix}.ffn_up_exps.weight"))?);
                let down_experts = Arc::new(gg.tensor(&format!("{prefix}.ffn_down_exps.weight"))?);
                let moe = FusedMoeGGUF {
                    gate,
                    gate_experts,
                    up_experts,
                    down_experts,
                    act: candle_nn::Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                    dtype,
                };

                MoeOrMlp::FusedMoe(moe)
            } else {
                let mlp = {
                    let feed_forward_w1 = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
                    let feed_forward_w2 = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
                    let feed_forward_w3 = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
                    Mlp {
                        feed_forward_w1,
                        feed_forward_w2,
                        feed_forward_w3,
                    }
                };
                MoeOrMlp::Mlp(mlp)
            };

            let attention_norm =
                gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let ffn_norm = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

            let self_attn = QuantizedAttention::new(
                &mut gg,
                &prefix,
                dtype,
                head_count,
                head_count_kv,
                head_dim,
                rms_norm_eps,
                device,
                rotary_emb.clone(),
            )?;
            layers.push(LayerWeights {
                self_attn,
                attention_norm,
                mlp,
                ffn_norm,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            dtype,
            device: device.clone(),
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let mut xs = self.tok_embeddings.forward(x)?;
        let (b, l) = x.dims2()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in self.layers.iter_mut() {
            let x = xs;
            let residual = &x;

            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, causal_mask.as_ref(), offset)?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x, causal_mask.is_some())?;
            let x = (x + residual)?;
            xs = x
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }
}
