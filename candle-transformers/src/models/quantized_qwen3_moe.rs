//! Quantized Qwen3 MoE implementation.
//!
//!
//! References:
//! - [Qwen3 MoE Models](https://huggingface.co/docs/transformers/model_doc/qwen3_moe) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::models::quantized_qwen3::{AttentionWeights, Gguf, MlpWeights, RotaryEmbedding};
use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

// Sparse MoE block stores the concatenated weights of all experts, no split!
#[derive(Debug, Clone)]
struct SparseMoeBlockWeights {
    gate: QMatMul,
    experts_gate: QMatMul,
    experts_up: QMatMul,
    experts_down: QMatMul,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    span: tracing::Span,
}

impl SparseMoeBlockWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        act: Activation,
        norm_topk_prob: bool,
        num_experts_per_tok: usize,
    ) -> Result<Self> {
        let gate = gg.qmatmul(&format!("{prefix}.ffn_gate_inp.weight"))?;
        let experts_gate = gg.qmatmul(&format!("{prefix}.ffn_gate_exps.weight"))?;
        let experts_up = gg.qmatmul(&format!("{prefix}.ffn_up_exps.weight"))?;
        let experts_down = gg.qmatmul(&format!("{prefix}.ffn_down_exps.weight"))?;
        let span = tracing::span!(tracing::Level::TRACE, "MoEBlock");
        Ok(Self {
            gate,
            experts_gate,
            experts_up,
            experts_down,
            act,
            norm_topk_prob,
            num_experts_per_tok,
            span,
        })
    }
}

impl Module for SparseMoeBlockWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        //[b_size * seq_len, hidden_dim]
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Extract topk experts per token
        let experts_per_tok = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let mut routing_weights = routing_weights.gather(&experts_per_tok, D::Minus1)?;

        if self.norm_topk_prob {
            routing_weights =
                routing_weights.broadcast_div(&routing_weights.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self
                .experts_gate
                .indexed_moe_forward(&xs, &experts_per_tok)?;
            let up = self.experts_up.indexed_moe_forward(&xs, &experts_per_tok)?;
            self.experts_down
                .indexed_moe_forward(&(up * gate.apply(&self.act)?)?, &experts_per_tok)?
        };

        ys.broadcast_mul(&routing_weights.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((b_size, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

#[derive(Debug, Clone)]
enum MoeOrMlpWeights {
    Moe(SparseMoeBlockWeights),
    Mlp(MlpWeights),
}

impl Module for MoeOrMlpWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Moe(m) => m.forward(xs),
            Self::Mlp(m) => m.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: AttentionWeights,
    feed_forward: MoeOrMlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        num_experts: usize,
        decoder_sparse_step: usize,
        norm_topk_prob: bool,
        num_experts_per_tok: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let feed_forward = if num_experts > 0 && (layer_idx + 1).is_multiple_of(decoder_sparse_step)
        {
            MoeOrMlpWeights::Moe(SparseMoeBlockWeights::new(
                gg,
                &prefix,
                candle_nn::Activation::Silu,
                norm_topk_prob,
                num_experts_per_tok,
            )?)
        } else {
            MoeOrMlpWeights::Mlp(MlpWeights::new(gg, &prefix)?)
        };
        Ok(Self {
            self_attn,
            feed_forward,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.feed_forward)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        num_experts_per_tok: Option<usize>,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3moe.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3moe.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3moe.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3moe.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3moe.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3moe.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3moe.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3moe.rope.freq_base")?.to_f32()? as f64;
        let decoder_sparse_step = 1;
        let moe_intermediate_size =
            md_get("qwen3moe.expert_feed_forward_length")?.to_u32()? as usize;
        let num_experts_per_tok = if let Some(n) = num_experts_per_tok {
            n
        } else {
            md_get("qwen3moe.expert_used_count")?.to_u32()? as usize
        };
        let num_experts = md_get("qwen3moe.expert_count")?.to_u32()? as usize;
        let norm_topk_prob = false;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(DecoderLayer::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
                num_experts,
                decoder_sparse_step,
                norm_topk_prob,
                num_experts_per_tok,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
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

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
