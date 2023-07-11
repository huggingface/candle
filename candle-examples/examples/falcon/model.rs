use anyhow::Result;
use candle::{safetensors::SafeTensors, DType, Device, IndexOp, Shape, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear};
use std::collections::HashMap;

const MAX_SEQ_LEN: usize = 5000;

pub struct VarBuilder<'a> {
    safetensors: Option<(HashMap<String, usize>, Vec<SafeTensors<'a>>)>,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn from_safetensors(
        safetensors: Vec<SafeTensors<'a>>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        Self {
            safetensors: Some((routing, safetensors)),
            device: device.clone(),
            dtype,
        }
    }

    pub fn zeros(dtype: DType, device: &Device) -> Self {
        Self {
            safetensors: None,
            device: device.clone(),
            dtype,
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let s: Shape = s.into();
        match &self.safetensors {
            None => Tensor::zeros(s, self.dtype, &self.device),
            Some((routing, safetensors)) => {
                // Unwrap or 0  just to let the proper error flow.
                let index = routing.get(tensor_name).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(tensor_name, &self.device)?
                    .to_dtype(self.dtype)?;
                if *tensor.shape() != s {
                    let msg = format!("shape mismatch for {tensor_name}");
                    Err(candle::Error::UnexpectedShape {
                        msg,
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}

fn linear(size1: usize, size2: usize, bias: bool, p: &str, vb: &VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
    let bias = if bias {
        Some(vb.get(size2, &format!("{p}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

fn layer_norm(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (
        vb.get(size, &format!("{p}.weight")),
        vb.get(size, &format!("{p}.bias")),
    ) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (
                vb.get(size, &format!("{p}.gamma")),
                vb.get(size, &format!("{p}.beta")),
            ) {
                (weight, bias)
            } else {
                return Err(err.into());
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug)]
struct Dropout {
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
    Ok(Embedding::new(embeddings, hidden_size))
}

// https://raw.githubusercontent.com/huggingface/transformers/030c863aaa0165e98352b61697430bf69bf33755/src/transformers/models/falcon/configuration_falcon.py
#[derive(Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_epsilon: f64,
    pub initializer_range: f64,
    pub use_cache: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_dropout: f64,
    pub attention_dropout: f64,
    pub n_head_kv: Option<usize>,
    pub alibi: bool,
    pub new_decoder_architecture: bool,
    pub multi_query: bool,
    pub parallel_attn: bool,
    pub bias: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 11,
            eos_token_id: 11,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            n_head_kv: None,
            alibi: false,
            new_decoder_architecture: false,
            multi_query: true,
            parallel_attn: true,
            bias: false,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.alibi {
            anyhow::bail!("alibi is not supported");
        }
        if self.new_decoder_architecture {
            anyhow::bail!("new_decoder_architecture is not supported");
        }
        if self.n_head_kv.is_some() {
            anyhow::bail!("n_head_kv is not supported");
        }
        Ok(())
    }

    // https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    pub fn falcon7b() -> Self {
        // This is currently on par with the defaults, the defaults come from the Python default
        // arguments for the config initialization whereas the following come from the json config.
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 11,
            eos_token_id: 11,
            hidden_dropout: 0.,
            attention_dropout: 0.,
            n_head_kv: None,
            alibi: false,
            new_decoder_architecture: false,
            multi_query: true,
            parallel_attn: true,
            bias: false,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn rotary(&self) -> bool {
        !self.alibi
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let l = x.dim(D::Minus1)?;
    let x1 = x.i((.., .., ..l / 2))?;
    let x2 = x.i((.., .., l / 2..))?;
    let x21 = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    Ok(x21)
}

#[derive(Debug)]
struct FalconRotaryEmbedding {
    inv_freq: Tensor,
    cache: Option<(usize, Tensor, Tensor)>,
}

impl FalconRotaryEmbedding {
    fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv_freq.as_slice(), &vb.device)?,
            cache: None,
        })
    }

    fn cos_sin(
        &mut self,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        match &self.cache {
            Some((s, cos, sin)) if *s == seq_len => {
                return Ok((cos.clone(), sin.clone()));
            }
            _ => {}
        }
        let t: Vec<_> = (0..seq_len).map(|c| c as u32).collect();
        let t = Tensor::new(t.as_slice(), device)?.to_dtype(dtype)?;
        let inv_freq = self.inv_freq.to_dtype(dtype)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        self.cache = Some((seq_len, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }

    fn forward(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        past_kv_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_batch, seq_len, _head_dim) = query.shape().r3()?;
        let (cos, sin) = self.cos_sin(MAX_SEQ_LEN, &query.device(), query.dtype())?;
        let cos = cos.i(past_kv_len..past_kv_len + seq_len)?;
        let sin = sin.i(past_kv_len..past_kv_len + seq_len)?;
        let qs = (query.broadcast_mul(&cos)? + &rotate_half(query)?.broadcast_mul(&sin)?)?;
        let ks = (key.broadcast_mul(&cos)? + &rotate_half(key)?.broadcast_mul(&sin)?)?;
        Ok((qs, ks))
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, &on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug)]
struct FalconAttention {
    query_key_value: Linear,
    dense: Linear,
    maybe_rotary: Option<FalconRotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    inv_norm_factor: f64,
    multi_query: bool,
    use_cache: bool,
    num_heads: usize,
    head_dim: usize,
    n_head_kv: usize,
}

impl FalconAttention {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let maybe_rotary = if cfg.rotary() {
            let rotary = FalconRotaryEmbedding::load(vb, cfg)?;
            Some(rotary)
        } else {
            None
        };
        let head_dim = cfg.head_dim();
        let hidden_size = cfg.hidden_size;
        let qkv_out_dim = if cfg.multi_query {
            hidden_size + 2 * head_dim
        } else {
            3 * hidden_size
        };
        let query_key_value = linear(
            hidden_size,
            qkv_out_dim,
            cfg.bias,
            &format!("{p}.query_key_value"),
            vb,
        )?;
        let dense = linear(
            hidden_size,
            hidden_size,
            cfg.bias,
            &format!("{p}.dense"),
            vb,
        )?;
        Ok(Self {
            query_key_value,
            dense,
            maybe_rotary,
            kv_cache: None,
            inv_norm_factor: 1. / (head_dim as f64).sqrt(),
            multi_query: cfg.multi_query,
            use_cache: cfg.use_cache,
            num_heads: cfg.num_attention_heads,
            n_head_kv: cfg.n_head_kv.unwrap_or(1),
            head_dim,
        })
    }

    fn split_heads(&self, fused_qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = fused_qkv.shape().r3()?;
        if !self.multi_query {
            let fused_qkv = fused_qkv.reshape((b_sz, seq_len, self.num_heads, 3, self.head_dim))?;
            let q = fused_qkv.i((.., .., .., 0))?;
            let k = fused_qkv.i((.., .., .., 1))?;
            let v = fused_qkv.i((.., .., .., 2))?;
            Ok((q, k, v))
        } else {
            let fused_qkv =
                fused_qkv.reshape((b_sz, seq_len, self.num_heads + 2, self.head_dim))?;
            let d = fused_qkv.dim(D::Minus2)?;
            let q = fused_qkv.i((.., .., ..d - 2))?;
            let k = fused_qkv.i((.., .., d - 2..d - 1))?;
            let v = fused_qkv.i((.., .., d - 1..))?;
            Ok((q, k, v))
        }
    }

    fn forward(&mut self, x: &Tensor, mask: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let fused_qkv = self.query_key_value.forward(x)?;
        let head_dim = self.head_dim;
        let (query, key, value) = self.split_heads(&fused_qkv)?;
        let (b_sz, seq_len, _, _) = query.shape().r4()?;
        let query = query
            .transpose(1, 2)?
            .reshape((b_sz * self.num_heads, seq_len, head_dim))?;
        let key = key
            .transpose(1, 2)?
            .reshape((b_sz * self.n_head_kv, seq_len, head_dim))?;
        let value = value
            .transpose(1, 2)?
            .reshape((b_sz * self.n_head_kv, seq_len, head_dim))?;
        let (query, key) = if let Some(r) = &mut self.maybe_rotary {
            r.forward(&query, &key, past_kv_len)?
        } else {
            (query, key)
        };
        let (mut key, mut value) = (key, value);
        let mask = masked_fill(&mask.to_dtype(DType::F32)?, mask, -1e9)?.to_dtype(query.dtype())?;
        if self.use_cache {
            if let Some((cache_k, cache_v)) = &self.kv_cache {
                // TODO: we could trim the tensors to MAX_SEQ_LEN so that this would work for
                // arbitrarily large sizes.
                key = Tensor::cat(&[cache_k, &key], 1)?.contiguous()?;
                value = Tensor::cat(&[cache_v, &value], 1)?.contiguous()?;
            }
            self.kv_cache = Some((key.clone(), value.clone()))
        }
        let query = query.reshape((b_sz * self.num_heads, seq_len, head_dim))?;
        let all_len = past_kv_len + seq_len;
        let key = key.reshape((b_sz * self.n_head_kv, all_len, head_dim))?;
        let value = value.reshape((b_sz * self.n_head_kv, all_len, head_dim))?;

        let (key, value) = if self.n_head_kv == 1 {
            (
                key.broadcast_as((b_sz * self.num_heads, all_len, head_dim))?,
                value.broadcast_as((b_sz * self.num_heads, all_len, head_dim))?,
            )
        } else {
            (key, value)
        };

        // Only handle the case where alibi is None here, and non-flash attention.
        let attention_scores = (query.matmul(&key.t()?)? * self.inv_norm_factor)?;
        let attention_scores = attention_scores
            .broadcast_add(&mask.squeeze(1)?)?
            .to_dtype(DType::F32)?
            .softmax(D::Minus1)?
            .to_dtype(x.dtype())?;
        let attn_output = attention_scores
            .matmul(&value)?
            .reshape((b_sz, self.num_heads, seq_len, head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * head_dim))?;
        let attn_output = self.dense.forward(&attn_output)?;
        Ok(attn_output)
    }
}

#[derive(Debug)]
struct FalconMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    dropout: Dropout,
}

impl FalconMlp {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let b = cfg.bias;
        let dense_h_to_4h = linear(h, 4 * h, b, &format!("{p}.dense_h_to_4h"), vb)?;
        let dense_4h_to_h = linear(4 * h, h, b, &format!("{p}.dense_4h_to_h"), vb)?;
        let dropout = Dropout::new(cfg.hidden_dropout);
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            dropout,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense_h_to_4h.forward(x)?.gelu()?;
        let x = self.dense_4h_to_h.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug)]
struct FalconDecoderLayer {
    inp_layernorm: LayerNorm,
    self_attention: FalconAttention,
    post_attention_layernorm: Option<LayerNorm>,
    mlp: FalconMlp,
    parallel_attn: bool,
}

impl FalconDecoderLayer {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let mlp = FalconMlp::load(&format!("{p}.mlp"), vb, cfg)?;
        let inp_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            &format!("{p}.input_layernorm"),
            vb,
        )?;
        let self_attention = FalconAttention::load(&format!("{p}.self_attention"), vb, cfg)?;
        let post_attention_layernorm = if cfg.parallel_attn {
            None
        } else {
            let ln = layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                &format!("{p}.post_attention_layernorm"),
                vb,
            )?;
            Some(ln)
        };
        Ok(Self {
            inp_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            parallel_attn: cfg.parallel_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let residual = x.clone();
        let ln_attn = self.inp_layernorm.forward(x)?;
        let attn_output = self.self_attention.forward(&ln_attn, mask, past_kv_len)?;
        let (residual, ln_mlp) = match &self.post_attention_layernorm {
            None => (residual, ln_attn),
            Some(pal) => {
                // This should include some dropout.
                let residual = (&attn_output + &residual)?;
                let ln_mlp = pal.forward(&residual)?;
                (residual, ln_mlp)
            }
        };
        let mlp_output = self.mlp.forward(&ln_mlp)?;

        let mlp_output = if self.parallel_attn {
            (mlp_output + attn_output)?
        } else {
            mlp_output
        };
        let output = (mlp_output + residual)?;
        Ok(output)
    }
}

#[derive(Debug)]
pub struct Falcon {
    word_embeddings: Embedding,
    blocks: Vec<FalconDecoderLayer>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: Config,
}

fn make_causal_mask(t: usize) -> Result<Tensor> {
    let mask: Vec<_> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u32::from(j > i)))
        .collect();
    let mask = Tensor::from_slice(&mask, (t, t), &Device::Cpu)?;
    Ok(mask)
}

fn prepare_attn_mask(b_sz: usize, seq_len: usize) -> Result<Tensor> {
    // let mask = Tensor::ones((b_sz, seq_len), DType::U32, &Device::Cpu)?;
    let mask = make_causal_mask(seq_len)?;
    let mask = mask.broadcast_as((b_sz, 1, seq_len, seq_len))?;
    Ok(mask)
}

impl Falcon {
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn load(vb: &VarBuilder, cfg: Config) -> Result<Self> {
        let word_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            "transformer.word_embeddings",
            vb,
        )?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| FalconDecoderLayer::load(&format!("transformer.h.{i}"), vb, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            "transformer.ln_f",
            vb,
        )?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, false, "lm_head", vb)?;
        Ok(Self {
            word_embeddings,
            blocks,
            ln_f,
            lm_head,
            config: cfg,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.shape().r2()?;
        let mut hidden_state = self.word_embeddings.forward(input_ids)?;
        let past_kv_len = match &self.blocks[0].self_attention.kv_cache {
            Some((k, _)) => k.dim(1)?,
            None => 0,
        };
        let causal_mask = prepare_attn_mask(b_sz, seq_len)?.to_device(&input_ids.device())?;
        for block in self.blocks.iter_mut() {
            hidden_state = block.forward(&hidden_state, &causal_mask, past_kv_len)?;
        }
        let hidden_state = self.ln_f.forward(&hidden_state)?;
        let hidden_state = hidden_state.i((.., seq_len - 1..))?;
        let logits = self.lm_head.forward(&hidden_state)?.squeeze(1)?;
        Ok(logits)
    }
}
