//! Phi3 model implementation with quantization support.
//!
//! Phi3 is a language model intended for research purposes.
//! This implementation provides quantization for reduced memory usage.
//!
//! Key characteristics:
//! - Multi-head attention
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for quantization
//!
//! References:
//! - [Model Card](https://huggingface.co/microsoft/phi-3)
//!

use std::collections::HashMap;

use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::quantized::QuantizedBackend;
#[cfg(not(feature = "flash-attn"))]
use candle::BackendStorage;
#[cfg(feature = "flash-attn")]
use candle::CudaStorage;
#[cfg(feature = "flash-attn")]
use candle::CudaStorage;
use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{kv_cache::KvCache, Embedding, RmsNorm};

#[derive(Debug, Clone)]
pub struct QLinear<QB: QuantizedBackend> {
    inner: candle::quantized::QMatMul<QB>,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> QLinear<QB> {
    fn new<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        name: &str,
        device: &QB::Device,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        let w = ct.tensor(r, &format!("{name}.weight"), device)?;
        let inner = candle::quantized::QMatMul::from_qtensor(w)?;
        Ok(Self { inner, span })
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for QLinear<QB>
where
    candle::quantized::QMatMul<QB>: Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp<QB: QuantizedBackend> {
    ffn_up: QLinear<QB>,
    ffn_down: QLinear<QB>,
    i_size: usize,
}

impl<QB: QuantizedBackend> Module<QB::Storage> for Mlp<QB>
where
    QLinear<QB>: Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let up_states = xs.apply(&self.ffn_up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.silu()?)?;
        up_states.apply(&self.ffn_down)
    }
}

fn rms_norm<QB: QuantizedBackend>(w: QTensor<QB>, eps: f64) -> Result<RmsNorm<QB::Storage>> {
    let w = w.dequantize()?;
    let rms = RmsNorm::new(w, eps);
    Ok(rms)
}

#[derive(Debug, Clone)]
struct LayerWeights<QB: QuantizedBackend> {
    attn_qkv: QLinear<QB>,
    attn_output: QLinear<QB>,
    attn_norm: RmsNorm<QB::Storage>,
    ffn_norm: RmsNorm<QB::Storage>,
    mlp: Mlp<QB>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor<QB::Storage>,
    sin: Tensor<QB::Storage>,
    neg_inf: Tensor<QB::Storage>,
    kv_cache: KvCache<QB::Storage>,
    use_flash_attn: bool,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
}

fn masked_fill<QB: QuantizedBackend>(
    on_false: &Tensor<QB::Storage>,
    mask: &Tensor<QB::Storage>,
    on_true: &Tensor<QB::Storage>,
) -> Result<Tensor<QB::Storage>> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl<QB: QuantizedBackend> LayerWeights<QB> {
    fn apply_rotary_emb(
        &self,
        xs: &Tensor<QB::Storage>,
        index_pos: usize,
    ) -> Result<Tensor<QB::Storage>> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _h, seq_len, _n_embd) = xs.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor<QB::Storage>,
        mask: Option<&Tensor<QB::Storage>>,
        index_pos: usize,
    ) -> Result<Tensor<QB::Storage>>
    where
        QLinear<QB>: Module<QB::Storage>,
    {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let qkv = self.attn_qkv.forward(x)?;

        let query_pos = self.n_head * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.n_kv_head * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.n_kv_head * self.head_dim,
            self.n_kv_head * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos)?.contiguous()?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        if index_pos == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        let k = crate::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = crate::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.to_dtype(DType::BF16)?.transpose(1, 2)?;
            let k = k.to_dtype(DType::BF16)?.transpose(1, 2)?;
            let v = v.to_dtype(DType::BF16)?.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?
                .to_dtype(DType::F32)?
                .transpose(1, 2)?
        } else {
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill::<QB>(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attn_output.forward(&y)?;
        Ok(y)
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor<CudaStorage>,
    k: &Tensor<CudaStorage>,
    v: &Tensor<CudaStorage>,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor<CudaStorage>> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn<B: BackendStorage>(
    _: &Tensor<B>,
    _: &Tensor<B>,
    _: &Tensor<B>,
    _: f32,
    _: bool,
) -> Result<Tensor<B>> {
    unimplemented!("compile with '--features flash-attn'")
}

#[derive(Debug, Clone)]
pub struct ModelWeights<QB: QuantizedBackend> {
    tok_embeddings: Embedding<QB::Storage>,
    layers: Vec<LayerWeights<QB>>,
    output_norm: RmsNorm<QB::Storage>,
    output: QLinear<QB>,
    masks: HashMap<usize, Tensor<QB::Storage>>,
    span: tracing::Span,
    span_output: tracing::Span,
}

type CosSin<QB> = (
    Tensor<<QB as QuantizedBackend>::Storage>,
    Tensor<<QB as QuantizedBackend>::Storage>,
);
fn precomput_freqs_cis<QB: QuantizedBackend>(
    head_dim: usize,
    max_seq_len: usize,
    freq_base: f32,
    device: &QB::Device,
) -> Result<CosSin<QB>> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl<QB: QuantizedBackend> ModelWeights<QB> {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        use_flash_attn: bool,
        ct: gguf_file::Content,
        reader: &mut R,
        device: &QB::Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let head_count = md_get("phi3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("phi3.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("phi3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("phi3.embedding_length")?.to_u32()? as usize;
        let max_seq_len = md_get("phi3.context_length")?.to_u32()? as usize;
        let head_dim = embedding_length / head_count;
        let i_size = md_get("phi3.feed_forward_length")?.to_u32()? as usize;
        let rope_dim = md_get("phi3.rope.dimension_count")?.to_u32()? as usize;
        let rms_eps = md_get("phi3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let (cos, sin) = precomput_freqs_cis::<QB>(rope_dim, max_seq_len, 10_000., device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings: QTensor<QB> = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize()?;
        let output_norm =
            rms_norm::<QB>(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;
        let output = QLinear::new(&ct, reader, "output", device)?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let ffn_up = QLinear::new(&ct, reader, &format!("{prefix}.ffn_up"), device)?;
            let ffn_down = QLinear::new(&ct, reader, &format!("{prefix}.ffn_down"), device)?;
            let mlp = Mlp {
                ffn_up,
                ffn_down,
                i_size,
            };
            let attn_norm = rms_norm::<QB>(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let ffn_norm = rms_norm::<QB>(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let kv_cache = KvCache::new(2, max_seq_len);
            layers.push(LayerWeights {
                attn_qkv: QLinear::new(&ct, reader, &format!("{prefix}.attn_qkv"), device)?,
                attn_output: QLinear::new(&ct, reader, &format!("{prefix}.attn_output"), device)?,
                attn_norm,
                ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache,
                use_flash_attn,
                span_attn,
                span_rot,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, t: usize, device: &QB::Device) -> Result<Tensor<QB::Storage>> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(
        &mut self,
        xs: &Tensor<QB::Storage>,
        index_pos: usize,
    ) -> Result<Tensor<QB::Storage>>
    where
        QLinear<QB>: Module<QB::Storage>,
    {
        let (_b_sz, seq_len) = xs.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, xs.device())?)
        };
        let _enter = self.span.enter();
        let mut xs = self.tok_embeddings.forward(xs)?;
        for layer in self.layers.iter_mut() {
            let residual = &xs;
            let ys = xs.apply(&layer.attn_norm)?;
            let ys = layer.forward_attn(&ys, mask.as_ref(), index_pos)?;
            let ys = (ys + residual)?;
            let residual = &ys;
            let ys = ys.apply(&layer.ffn_norm)?;
            let ys = layer.mlp.forward(&ys)?;
            xs = (ys + residual)?
        }
        let xs = xs.apply(&self.output_norm)?.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&xs)
    }
}
