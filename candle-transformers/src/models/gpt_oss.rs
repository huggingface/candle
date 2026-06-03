//! GPT-OSS (OpenAI open-weights MoE family: gpt-oss-20b / gpt-oss-120b).
//!
//! Sparse MoE (top-4 routing), attention sinks (per-head learnable logits added
//! to the pre-softmax scores), alternating sliding-window / full attention, YaRN
//! RoPE for long context, and clamped-SwiGLU experts.
//!
//! Expert weights ship fused + MXFP4 (gate_up_proj / down_proj as `_blocks` +
//! `_scales`, one e8m0 exponent per 32-value block); they are dequantized to the
//! model dtype at load (see `dequant_mxfp4`). gate/up are interleaved on the
//! output dim. Everything else is bf16 (config `modules_to_not_convert`).
//!
//! Open items before numerical parity vs HF `modeling_gpt_oss`:
//!  - YaRN rope scaling (currently plain RoPE).
//!  - offset-aware KV cache for generation.
use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub sliding_window: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f64,
}

fn default_rope_theta() -> f64 {
    150000.0
}
fn default_eps() -> f64 {
    1e-5
}
fn default_max_pos() -> usize {
    131072
}
fn default_true() -> bool {
    true
}
fn default_swiglu_limit() -> f64 {
    7.0
}

impl Config {
    /// gpt-oss-20b reference shape (24 layers / 32 experts).
    pub fn gpt_oss_20b() -> Self {
        Self {
            vocab_size: 201088,
            hidden_size: 2880,
            intermediate_size: 2880,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 64,
            num_local_experts: 32,
            num_experts_per_tok: 4,
            sliding_window: 128,
            rope_theta: 150000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 131072,
            attention_bias: true,
            swiglu_limit: 7.0,
        }
    }

    /// gpt-oss alternates sliding-window and full attention, starting sliding.
    fn is_sliding(&self, layer_idx: usize) -> bool {
        layer_idx % 2 == 0
    }
}

// --- RoPE -------------------------------------------------------------------
// TODO: YaRN scaling for the 131k context window. Plain RoPE for now.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_pos = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, dev)?;
        let t = Tensor::arange(0u32, max_pos as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq.reshape((1, dim / 2))?)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

// --- Attention (GQA + sinks + sliding window) -------------------------------
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sinks: Tensor, // [num_heads] learnable per-head sink logits
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, rotary: &RotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let _ = rotary;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let lin = |i, o, vb| {
            if cfg.attention_bias {
                linear(i, o, vb)
            } else {
                linear_no_bias(i, o, vb)
            }
        };
        Ok(Self {
            q_proj: lin(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: lin(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: lin(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: lin(nh * hd, h, vb.pp("o_proj"))?,
            sinks: vb.get(nh, "sinks")?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, offset)?;
        // TODO: KV cache append goes here (offset-aware) once wired.
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        // Attention sinks: append one learnable logit per head as an extra key
        // column, softmax over [scores | sink], then drop the sink column from
        // the value-weighted sum (it only absorbs probability mass).
        let n_keys = scores.dim(D::Minus1)?;
        let sinks = self
            .sinks
            .reshape((1, self.num_heads, 1, 1))?
            .broadcast_as((b, self.num_heads, seq_len, 1))?
            .to_dtype(scores.dtype())?
            .contiguous()?;
        let logits = Tensor::cat(&[&scores, &sinks], D::Minus1)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs = probs.narrow(D::Minus1, 0, n_keys)?; // drop sink col
        let out = probs.contiguous()?.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        self.o_proj.forward(&out)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))
}

// --- MoE: router + clamped-SwiGLU experts -----------------------------------
// gpt-oss expert activation: gate, up = deinterleave(gate_up); with gate clamped
// to `limit` and up clamped to [-limit, limit]; glu = gate*sigmoid(a*gate);
// out = (up + 1) * glu.
const SWIGLU_ALPHA: f64 = 1.702;

// FP4 (e2m1) code -> value lookup, codes 0..7 positive, 8..15 negative.
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// Dequantize MXFP4 expert weights to `dtype`. Each byte in `blocks` packs two
// 4-bit e2m1 codes; `scales` holds one e8m0 exponent per 32-value block:
// value = FP4_LUT[code] * 2^(scale - 127). Done once at load on CPU.
//   blocks: u8 [E, out, nb, 16], scales: u8 [E, out, nb] -> dtype [E, out, nb*32]
fn dequant_mxfp4(blocks: &Tensor, scales: &Tensor, dtype: DType) -> Result<Tensor> {
    let dev = blocks.device().clone();
    let (e, out, nb, bytes) = blocks.dims4()?;
    let vals = bytes * 2; // 32 values per block
    let in_dim = nb * vals;
    let blk = blocks.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let scl = scales.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let rows = e * out;
    let mut data = vec![0f32; rows * in_dim];
    for row in 0..rows {
        for b in 0..nb {
            let scale = 2f32.powi(scl[row * nb + b] as i32 - 127);
            let blk_off = (row * nb + b) * bytes;
            let out_off = row * in_dim + b * vals;
            for j in 0..bytes {
                let byte = blk[blk_off + j];
                data[out_off + 2 * j] = FP4_LUT[(byte & 0x0f) as usize] * scale;
                data[out_off + 2 * j + 1] = FP4_LUT[(byte >> 4) as usize] * scale;
            }
        }
    }
    Tensor::from_vec(data, (e, out, in_dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(&dev)
}

#[derive(Debug, Clone)]
struct Expert {
    gate_up_proj: Linear, // hidden -> 2*intermediate (gate/up interleaved)
    down_proj: Linear,    // intermediate -> hidden
    limit: f64,
}

impl Expert {
    // gate_up_w [2*inter, hidden], down_w [hidden, inter], each with its bias.
    fn from_weights(
        gate_up_w: Tensor,
        gate_up_b: Tensor,
        down_w: Tensor,
        down_b: Tensor,
        limit: f64,
    ) -> Self {
        Self {
            gate_up_proj: Linear::from_weights(gate_up_w, Some(gate_up_b)),
            down_proj: Linear::from_weights(down_w, Some(down_b)),
            limit,
        }
    }
}

impl Module for Expert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // gate_up output is interleaved [g0, u0, g1, u1, ...]; reshape (.., I, 2)
        // so [.., i, 0] = gate_i and [.., i, 1] = up_i.
        let gate_up = self.gate_up_proj.forward(xs)?;
        let (n, two_i) = gate_up.dims2()?;
        let gate_up = gate_up.reshape((n, two_i / 2, 2))?;
        let gate = gate_up.i((.., .., 0))?.contiguous()?;
        let up = gate_up.i((.., .., 1))?.contiguous()?;
        let gate = gate.clamp(f64::NEG_INFINITY, self.limit)?;
        let up = up.clamp(-self.limit, self.limit)?;
        let glu = (&gate * candle_nn::ops::sigmoid(&(&gate * SWIGLU_ALPHA)?)?)?;
        let act = ((up + 1.0)? * glu)?;
        self.down_proj.forward(&act)
    }
}

#[derive(Debug, Clone)]
struct SparseMoe {
    router: Linear, // gpt-oss router has bias
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
}

impl SparseMoe {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let router = linear(cfg.hidden_size, cfg.num_local_experts, vb.pp("router"))?;
        let e = cfg.num_local_experts;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let dtype = vb.dtype();
        let vb_e = vb.pp("experts");
        // Fused MXFP4 expert weights, dequantized once: blocks/scales load as raw
        // u8 (get_unchecked_dtype avoids the dtype coercion get() would apply).
        let gate_up_w = dequant_mxfp4(
            &vb_e.get_unchecked_dtype("gate_up_proj_blocks", DType::U8)?,
            &vb_e.get_unchecked_dtype("gate_up_proj_scales", DType::U8)?,
            dtype,
        )?; // [E, 2*inter, hidden]
        let gate_up_b = vb_e.get((e, 2 * i), "gate_up_proj_bias")?;
        let down_w = dequant_mxfp4(
            &vb_e.get_unchecked_dtype("down_proj_blocks", DType::U8)?,
            &vb_e.get_unchecked_dtype("down_proj_scales", DType::U8)?,
            dtype,
        )?; // [E, hidden, inter]
        let down_b = vb_e.get((e, h), "down_proj_bias")?;
        let experts = (0..e)
            .map(|x| {
                Ok(Expert::from_weights(
                    gate_up_w.i(x)?.contiguous()?,
                    gate_up_b.i(x)?.contiguous()?,
                    down_w.i(x)?.contiguous()?,
                    down_b.i(x)?.contiguous()?,
                    cfg.swiglu_limit,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            router,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }
}

impl Module for SparseMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;
        let xs = xs.reshape(((), hidden))?;
        let router_logits = self.router.forward(&xs)?;
        let routing = candle_nn::ops::softmax_last_dim(&router_logits)?;
        let sel = routing
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let weights = routing.gather(&sel, D::Minus1)?;
        let weights = weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let sel = sel.to_vec2::<u32>()?;

        let mut top_x = vec![vec![]; self.experts.len()];
        let mut top_w = vec![vec![]; self.experts.len()];
        for (row, (w, idxs)) in weights.iter().zip(sel.iter()).enumerate() {
            let sum: f32 = w.iter().sum();
            for (&w, &e) in w.iter().zip(idxs.iter()) {
                top_x[e as usize].push(row as u32);
                top_w[e as usize].push(w / sum); // normalize top-k
            }
        }
        let mut ys = xs.zeros_like()?;
        for (e, expert) in self.experts.iter().enumerate() {
            if top_x[e].is_empty() {
                continue;
            }
            let idx = Tensor::new(top_x[e].as_slice(), xs.device())?;
            let w = Tensor::new(top_w[e].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;
            let state = xs.index_select(&idx, 0)?;
            let out = expert.forward(&state)?.broadcast_mul(&w)?;
            ys = ys.index_add(&idx, &out, 0)?;
        }
        ys.reshape((b, seq_len, hidden))
    }
}

// --- Decoder layer / Model --------------------------------------------------
#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: SparseMoe,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, rotary: &RotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: SparseMoe::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            is_sliding: cfg.is_sliding(layer_idx),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rotary, mask, offset)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let h = self.post_attention_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = RotaryEmbedding::new(cfg, vb.device(), vb.dtype())?;
        let vb_l = vb_m.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(cfg, i, &rotary, vb_l.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    // Additive causal mask, optionally sliding-window-limited. Query i (absolute
    // position offset+i) may attend to key j when j <= offset+i and, when `window`
    // is set, (offset+i) - j < window (gpt-oss sliding layers).
    // Shape [1, 1, seq_len, offset+seq_len]; -inf where masked, 0 otherwise.
    fn causal_mask(&self, seq_len: usize, offset: usize, window: Option<usize>) -> Result<Tensor> {
        let kv = seq_len + offset;
        let mut data = vec![0f32; seq_len * kv];
        for qi in 0..seq_len {
            let q_abs = qi + offset;
            for kj in 0..kv {
                let out_of_window = window.is_some_and(|w| q_abs >= kj + w);
                if kj > q_abs || out_of_window {
                    data[qi * kv + kj] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_slice(&data, (1, 1, seq_len, kv), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        // Full-causal mask for full-attention layers, sliding-window mask for the
        // alternating sliding layers; selected per layer below.
        let (full, sliding) = if seq_len <= 1 {
            (None, None)
        } else {
            (
                Some(self.causal_mask(seq_len, offset, None)?),
                Some(self.causal_mask(seq_len, offset, Some(self.sliding_window))?),
            )
        };
        for layer in self.layers.iter() {
            let mask = if layer.is_sliding {
                sliding.as_ref()
            } else {
                full.as_ref()
            };
            xs = layer.forward(&xs, &self.rotary, mask, offset)?;
        }
        self.norm.forward(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    model: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let model = Model::new(cfg, vb.clone())?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self { model, lm_head })
    }

    pub fn forward(&self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let xs = self.model.forward(input_ids, offset)?;
        let (_b, seq_len, _) = xs.dims3()?;
        xs.i((.., seq_len - 1, ..))?.apply(&self.lm_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Dequantize a single 16-byte block (32 values) at the given e8m0 scale.
    fn one_block(bytes: Vec<u8>, scale: u8) -> Vec<f32> {
        let dev = Device::Cpu;
        let blocks = Tensor::from_vec(bytes, (1usize, 1, 1, 16), &dev).unwrap();
        let scales = Tensor::from_vec(vec![scale], (1usize, 1, 1), &dev).unwrap();
        let out = dequant_mxfp4(&blocks, &scales, DType::F32).unwrap();
        assert_eq!(out.dims(), &[1, 1, 32]);
        out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn mxfp4_codes_and_nibble_order() {
        // byte = (high << 4) | low; low nibble -> even index, high -> odd index.
        // 0x21: low 1 -> 0.5, high 2 -> 1.0
        // 0xF7: low 7 -> 6.0, high 15 -> -6.0
        // 0x8A: low 10 -> -1.0, high 8 -> -0.0
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x21;
        bytes[1] = 0xF7;
        bytes[2] = 0x8A;
        let v = one_block(bytes, 127); // 2^(127-127) = 1.0
        assert_eq!(&v[..6], &[0.5, 1.0, 6.0, -6.0, -1.0, -0.0]);
        assert!(v[6..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn mxfp4_scale_exponent() {
        let mut bytes = vec![0u8; 16];
        bytes[0] = 0x21; // 0.5, 1.0 pre-scale
        assert_eq!(&one_block(bytes.clone(), 128)[..2], &[1.0, 2.0]); // x2
        assert_eq!(&one_block(bytes, 126)[..2], &[0.25, 0.5]); // x0.5
    }

    #[test]
    fn mxfp4_shape_multi_block() {
        let dev = Device::Cpu;
        let (e, out, nb) = (2usize, 3usize, 2usize);
        let blocks = Tensor::zeros((e, out, nb, 16), DType::U8, &dev).unwrap();
        let scales = Tensor::zeros((e, out, nb), DType::U8, &dev).unwrap();
        let t = dequant_mxfp4(&blocks, &scales, DType::F32).unwrap();
        assert_eq!(t.dims(), &[e, out, nb * 32]);
    }
}
