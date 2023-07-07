use anyhow::Result;
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor, D};
use std::collections::HashMap;

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

#[derive(Debug)]
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn load(size1: usize, size2: usize, bias: bool, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        let bias = if bias {
            Some(vb.get(size2, &format!("{p}.bias"))?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let (bsize, _, _) = x.shape().r3()?;
        let w = self.weight.broadcast_left(bsize)?.t()?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

#[derive(Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    fn load(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<Self> {
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
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_bsize, _seq_len, hidden_size) = x.shape().r3()?;
        let mean_x = (x.sum(&[2])? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = ((&x * &x)?.sum(&[2])? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
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

#[derive(Debug)]
struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    fn load(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
        Ok(Self::new(embeddings, hidden_size))
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = Tensor::embedding(&indexes, &self.embeddings)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

// https://raw.githubusercontent.com/huggingface/transformers/030c863aaa0165e98352b61697430bf69bf33755/src/transformers/models/falcon/configuration_falcon.py
#[derive(Debug)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    layer_norm_epsilon: f64,
    initializer_range: f64,
    use_cache: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    hidden_dropout: f64,
    attention_dropout: f64,
    n_head_kv: Option<usize>,
    alibi: bool,
    new_decoder_architecture: bool,
    multi_query: bool,
    parallel_attn: bool,
    bias: bool,
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
    let x1 = x.narrow(D::Minus1, 0, l / 2)?;
    let x2 = x.narrow(D::Minus1, l / 2, l - l / 2)?;
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
        let inv_freq = Tensor::new(inv_freq.as_slice(), &vb.device)?;
        let cache = None;
        Ok(Self { inv_freq, cache })
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

    fn forward(&mut self, query: &Tensor, key: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch, seq_len, _head_dim) = query.shape().r3()?;
        let (cos, sin) = self.cos_sin(seq_len, &query.device(), query.dtype())?;
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
    inv_norm_factor: f64,
    multi_query: bool,
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
        let query_key_value = Linear::load(
            hidden_size,
            qkv_out_dim,
            cfg.bias,
            &format!("{p}.query_key_value"),
            vb,
        )?;
        let dense = Linear::load(
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
            inv_norm_factor: 1. / (head_dim as f64).sqrt(),
            multi_query: cfg.multi_query,
            num_heads: cfg.num_attention_heads,
            n_head_kv: cfg.n_head_kv.unwrap_or(1),
            head_dim,
        })
    }

    fn split_heads(&self, fused_qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = fused_qkv.shape().r3()?;
        if !self.multi_query {
            let fused_qkv = fused_qkv.reshape((b_sz, seq_len, self.num_heads, 3, self.head_dim))?;
            let q = fused_qkv.narrow(D::Minus2, 0, 1)?.squeeze(D::Minus2)?;
            let k = fused_qkv.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?;
            let v = fused_qkv.narrow(D::Minus2, 2, 1)?.squeeze(D::Minus2)?;
            Ok((q, k, v))
        } else {
            let fused_qkv =
                fused_qkv.reshape((b_sz, seq_len, self.num_heads + 2, self.head_dim))?;
            let d = fused_qkv.dim(D::Minus2)?;
            let q = fused_qkv.narrow(D::Minus2, 0, d - 2)?;
            let k = fused_qkv.narrow(D::Minus2, d - 2, 1)?;
            let v = fused_qkv.narrow(D::Minus2, d - 1, 1)?;
            Ok((q, k, v))
        }
    }

    fn forward(&mut self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let fused_qkv = self.query_key_value.forward(x)?;
        let head_dim = self.head_dim;
        let (query, key, value) = self.split_heads(&fused_qkv)?;
        let (b_sz, q_len, _, _) = query.shape().r4()?;
        let query = query
            .transpose(1, 2)?
            .reshape((b_sz * self.num_heads, q_len, head_dim))?;
        let key = key
            .transpose(1, 2)?
            .reshape((b_sz * self.n_head_kv, q_len, head_dim))?;
        let value = value
            .transpose(1, 2)?
            .reshape((b_sz * self.n_head_kv, q_len, head_dim))?;
        let (query, key) = if let Some(r) = &mut self.maybe_rotary {
            r.forward(&query, &key)?
        } else {
            (query, key)
        };
        let mask = masked_fill(&mask.to_dtype(DType::F32)?, mask, -1e9)?.to_dtype(query.dtype())?;
        // TODO: layer_past, use_cache?
        let query = query.reshape((b_sz * self.num_heads, q_len, head_dim))?;
        let key = key.reshape((b_sz * self.n_head_kv, q_len, head_dim))?;
        let value = value.reshape((b_sz * self.n_head_kv, q_len, head_dim))?;

        let (key, value) = if self.n_head_kv == 1 {
            (
                key.broadcast_as(query.dims())?,
                value.broadcast_as(query.dims())?,
            )
        } else {
            (key, value)
        };

        // Only handle alibi is None here, and non-flash attention.
        let attention_scores = (query.matmul(&key.t()?)? * self.inv_norm_factor)?;
        let attention_scores = attention_scores
            .broadcast_add(&mask.squeeze(1)?)?
            .softmax(D::Minus1)?;
        let attn_output = attention_scores
            .matmul(&value)?
            .reshape((b_sz, self.num_heads, q_len, head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * head_dim))?;
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
        let dense_h_to_4h = Linear::load(h, 4 * h, b, &format!("{p}.dense_h_to_4h"), vb)?;
        let dense_4h_to_h = Linear::load(4 * h, h, b, &format!("{p}.dense_4h_to_h"), vb)?;
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
        let inp_layernorm = LayerNorm::load(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            &format!("{p}.input_layernorm"),
            vb,
        )?;
        let self_attention = FalconAttention::load(&format!("{p}.self_attention"), vb, cfg)?;
        let post_attention_layernorm = if cfg.parallel_attn {
            None
        } else {
            let ln = LayerNorm::load(
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

    fn forward(&mut self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let ln_attn = self.inp_layernorm.forward(x)?;
        let attn_output = self.self_attention.forward(&ln_attn, mask)?;
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
    pub fn load(vb: &VarBuilder, cfg: Config) -> Result<Self> {
        let word_embeddings = Embedding::load(
            cfg.vocab_size,
            cfg.hidden_size,
            "transformer.word_embeddings",
            vb,
        )?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| FalconDecoderLayer::load(&format!("transformer.h.{i}"), vb, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = LayerNorm::load(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            "transformer.ln_f",
            vb,
        )?;
        let lm_head = Linear::load(cfg.hidden_size, cfg.vocab_size, false, "lm_head", vb)?;
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
        let causal_mask = prepare_attn_mask(b_sz, seq_len)?.to_device(&input_ids.device())?;
        for block in self.blocks.iter_mut() {
            hidden_state = block.forward(&hidden_state, &causal_mask)?;
        }
        let hidden_state = self.ln_f.forward(&hidden_state)?;
        let hidden_state = hidden_state.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?.squeeze(1)?;
        Ok(logits)
    }
}
