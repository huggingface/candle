#![allow(dead_code)]

use anyhow::Result;
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor, D};
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
        let dtype = x.dtype();
        let (_bsize, _seq_len, hidden_size) = x.shape().r3()?;
        let x = x.to_dtype(DType::F32)?;
        let mean_x = (x.sum(&[2])? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = ((&x * &x)?.sum(&[2])? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .to_dtype(dtype)?
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConvConfig {
    padding: usize,
    stride: usize,
}

#[derive(Debug)]
struct Conv1D {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig,
}

impl Conv1D {
    // Applies weight norm for inference by recomputing the weight tensor. This
    // does not apply to training.
    // https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    fn load_weight_norm(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight_g = vb.get((out_c, 1, 1), &format!("{p}.weight_g"))?;
        let weight_v = vb.get((out_c, in_c, kernel_size), &format!("{p}.weight_v"))?;
        let norm_v = (&weight_v * &weight_v)?.sum(&[1, 2])?.sqrt()?;
        let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
        let bias = vb.get(out_c, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            config,
        })
    }

    fn load(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((out_c, in_c, kernel_size), &format!("{p}.weight"))?;
        let bias = vb.get(out_c, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            config,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}

// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/configuration_musicgen.py#L83
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    vocab_size: usize,
    max_position_embeddings: usize,
    num_hidden_layers: usize,
    ffn_dim: usize,
    num_attention_heads: usize,
    layerdrop: f64,
    use_cache: bool,
    activation_function: HiddenAct,
    hidden_size: usize,
    dropout: f64,
    attention_dropout: f64,
    activation_dropout: f64,
    initializer_factor: f64,
    scale_embedding: bool,
    num_codebooks: usize,
    pad_token_id: usize,
    bos_token_id: usize,
    eos_token_id: Option<usize>,
    tie_word_embeddings: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 24,
            ffn_dim: 4096,
            num_attention_heads: 16,
            layerdrop: 0.0,
            use_cache: true,
            activation_function: HiddenAct::Gelu, // TODO: Handle old style gelu.
            hidden_size: 1024,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            initializer_factor: 0.02,
            scale_embedding: false,
            num_codebooks: 4,
            pad_token_id: 2048,
            bos_token_id: 2048,
            eos_token_id: None,
            tie_word_embeddings: false,
        }
    }
}

fn get_embedding(num_embeddings: usize, embedding_dim: usize) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let emb = f64::ln(10000.) / (half_dim - 1) as f64;
    let xs: Vec<_> = (0..num_embeddings).map(|v| v as f32).collect();
    let xs = Tensor::from_vec(xs, (num_embeddings, 1), &Device::Cpu)?;
    let ys: Vec<_> = (0..half_dim)
        .map(|v| f64::exp(v as f64 * -emb) as f32)
        .collect();
    let ys = Tensor::from_vec(ys, (1, half_dim), &Device::Cpu)?;
    let shape = (num_embeddings, half_dim);
    let emb = (xs.broadcast_as(shape)? * ys.broadcast_as(shape)?)?;
    let emb =
        Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1)?.reshape((num_embeddings, 2 * half_dim))?;
    let emb = if embedding_dim % 2 == 1 {
        let zeros = Tensor::zeros((num_embeddings, 1), DType::F32, &Device::Cpu)?;
        Tensor::cat(&[&emb, &zeros], 1)?
    } else {
        emb
    };
    Ok(emb)
}

#[derive(Debug)]
struct MusicgenSinusoidalPositionalEmbedding {
    num_positions: usize,
    embedding_dim: usize,
    weights: Tensor,
}

impl MusicgenSinusoidalPositionalEmbedding {
    fn load(_vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let num_positions = cfg.max_position_embeddings;
        let embedding_dim = cfg.hidden_size;
        let weights = get_embedding(num_positions, embedding_dim)?;
        Ok(Self {
            num_positions,
            embedding_dim,
            weights,
        })
    }

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, _codebooks, seq_len) = input_ids.shape().r3()?;
        if seq_len > self.weights.dim(0)? {
            self.weights = get_embedding(seq_len, self.embedding_dim)?
        }
        Ok(self.weights.narrow(0, 0, seq_len)?)
    }
}

#[derive(Debug)]
struct MusicgenAttention {
    scaling: f64,
    is_decoder: bool,
    num_heads: usize,
    head_dim: usize,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
}

impl MusicgenAttention {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = h / num_heads;
        let k_proj = Linear::load(h, h, false, &format!("{p}.k_proj"), vb)?;
        let v_proj = Linear::load(h, h, false, &format!("{p}.v_proj"), vb)?;
        let q_proj = Linear::load(h, h, false, &format!("{p}.q_proj"), vb)?;
        let out_proj = Linear::load(h, h, false, &format!("{p}.out_proj"), vb)?;
        Ok(Self {
            scaling: 1. / (head_dim as f64).sqrt(),
            is_decoder: true,
            num_heads,
            head_dim,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.shape().r3()?;
        let query_states = (self.q_proj.forward(xs)? * self.scaling)?;

        let kv_states = kv_states.unwrap_or(xs);
        let key_states = self.k_proj.forward(kv_states)?;
        let value_states = self.v_proj.forward(kv_states)?;

        let tgt = (b_sz, tgt_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(tgt)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(tgt)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(tgt)?.transpose(1, 2)?.contiguous()?;

        let src_len = key_states.dim(1)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let attn_weights = attn_weights
            .reshape((b_sz, self.num_heads, tgt_len, src_len))?
            .broadcast_add(attention_mask)?;
        let attn_weights = attn_weights.softmax(D::Minus1)?;
        // TODO: layer_head_mask?
        let attn_output = attn_weights
            .matmul(&value_states)?
            .reshape((b_sz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.num_heads * self.head_dim))?;
        let attn_output = self.out_proj.forward(&attn_output)?;
        Ok(attn_output)
    }
}

#[derive(Debug)]
struct MusicgenDecoderLayer {
    self_attn: MusicgenAttention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: MusicgenAttention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation_fn: HiddenAct,
}

impl MusicgenDecoderLayer {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let self_attn = MusicgenAttention::load(&format!("{p}.self_attn"), vb, cfg)?;
        let self_attn_layer_norm =
            LayerNorm::load(h, 1e-5, &format!("{p}.self_attn_layer_norm"), vb)?;
        let encoder_attn = MusicgenAttention::load(&format!("{p}.encoder_attn"), vb, cfg)?;
        let encoder_attn_layer_norm =
            LayerNorm::load(h, 1e-5, &format!("{p}.encoder_attn_layer_norm"), vb)?;
        let fc1 = Linear::load(h, cfg.ffn_dim, false, &format!("{p}.fc1"), vb)?;
        let fc2 = Linear::load(cfg.ffn_dim, h, false, &format!("{p}.fc2"), vb)?;
        let final_layer_norm = LayerNorm::load(h, 1e-5, &format!("{p}.final_layer_norm"), vb)?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation_fn: cfg.activation_function,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None, attention_mask)?;
        let mut xs = (xs + residual)?;
        if let Some(encoder_hidden_states) = &encoder_hidden_states {
            let residual = xs.clone();
            let encoder_attention_mask = attention_mask.clone(); // TODO
            xs = self.encoder_attn.forward(
                &xs,
                Some(encoder_hidden_states),
                &encoder_attention_mask,
            )?;
            xs = (xs + residual)?
        }
        let residual = xs.clone();
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct MusicgenDecoder {
    embed_tokens: Vec<Embedding>,
    embed_positions: MusicgenSinusoidalPositionalEmbedding,
    layers: Vec<MusicgenDecoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
    num_codebooks: usize,
    d_model: usize,
}

impl MusicgenDecoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let embed_scale = if cfg.scale_embedding {
            (h as f64).sqrt()
        } else {
            1.
        };
        let embed_dim = cfg.vocab_size + 1;
        let embed_tokens = (0..cfg.num_codebooks)
            .map(|i| Embedding::load(embed_dim, h, &format!("{p}.embed_tokens.{i}"), vb))
            .collect::<Result<Vec<_>>>()?;
        let embed_positions = MusicgenSinusoidalPositionalEmbedding::load(vb, cfg)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| MusicgenDecoderLayer::load(&format!("{p}.layers.{i}"), vb, cfg))
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = LayerNorm::load(h, 1e-5, &format!("{p}.layer_norm"), vb)?;
        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            embed_scale,
            num_codebooks: cfg.num_codebooks,
            d_model: cfg.hidden_size,
        })
    }

    fn prepare_decoder_attention_mask(&self, _b_sz: usize, _seq_len: usize) -> Result<Tensor> {
        todo!()
    }

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let dev = input_ids.device();
        let (b_sz_times_codebooks, seq_len) = input_ids.shape().r2()?;
        let b_sz = b_sz_times_codebooks / self.num_codebooks;
        let input = input_ids.reshape((b_sz, self.num_codebooks, seq_len))?;
        let mut inputs_embeds = Tensor::zeros((b_sz, seq_len, self.d_model), DType::F32, &dev)?;
        for (idx, codebook) in self.embed_tokens.iter().enumerate() {
            let inp = input.narrow(1, idx, 1)?.squeeze(1)?;
            inputs_embeds = (inputs_embeds + codebook.forward(&inp)?)?
        }
        let inputs_embeds = inputs_embeds;
        let positions = self.embed_positions.forward(&input)?.to_device(&dev)?;
        let mut xs = inputs_embeds.broadcast_add(&positions)?;
        let attention_mask = self.prepare_decoder_attention_mask(b_sz, seq_len)?;
        for (_layer_idx, decoder_layer) in self.layers.iter_mut().enumerate() {
            xs = decoder_layer.forward(&xs, &attention_mask, None)?;
        }
        let xs = self.layer_norm.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct MusicgenForCausalLM {
    decoder: MusicgenDecoder,
    lm_heads: Vec<Linear>,
    num_codebooks: usize,
    vocab_size: usize,
}

impl MusicgenForCausalLM {
    pub fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let decoder = MusicgenDecoder::load(&format!("{p}.model.decoder"), vb, cfg)?;
        let lm_heads = (0..cfg.num_codebooks)
            .map(|i| Linear::load(h, cfg.vocab_size, false, &format!("{p}.lm_heads.{i}"), vb))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            decoder,
            lm_heads,
            num_codebooks: cfg.num_codebooks,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.shape().r2()?;
        let hidden_states = self.decoder.forward(input_ids)?;
        let lm_logits = self
            .lm_heads
            .iter()
            .map(|h| Ok(h.forward(&hidden_states)?))
            .collect::<Result<Vec<_>>>()?;
        let lm_logits = Tensor::stack(&lm_logits, 1)?.reshape((
            b_sz * self.num_codebooks,
            seq_len,
            self.vocab_size,
        ))?;
        Ok(lm_logits)
    }
}

// T5 Text Encoder
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

#[derive(Debug, Clone, PartialEq)]
pub struct T5Config {
    vocab_size: usize,
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    num_layers: usize,
    num_decoder_layers: Option<usize>,
    num_heads: usize,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    dropout_rate: f64,
    layer_norm_epsilon: f64,
    initializer_factor: f64,
    feed_forward_proj: HiddenAct,
    is_decoder: bool,
    is_encoder_decoder: bool,
    use_cache: bool,
    pad_token_id: usize,
    eos_token_id: usize,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: HiddenAct::Relu,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
        }
    }
}

impl T5Config {
    // https://huggingface.co/facebook/musicgen-small/blob/495da4ad086b3416a27c6187f9239f9fd96f3962/config.json#L184
    pub fn musicgen_small() -> Self {
        Self {
            d_ff: 3072,
            d_kv: 64,
            d_model: 768,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: HiddenAct::Relu,
            initializer_factor: 1.0,
            is_decoder: false,
            is_encoder_decoder: true,
            layer_norm_epsilon: 1e-6,
            num_decoder_layers: Some(12),
            num_heads: 12,
            num_layers: 12,
            pad_token_id: 0,
            relative_attention_max_distance: 128,
            relative_attention_num_buckets: 32,
            use_cache: true,
            vocab_size: 32128,
        }
    }
}

#[derive(Debug)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(h, &format!("{p}.weight"))?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
        })
    }
}

#[derive(Debug)]
struct T5DenseActDense {
    wi: Linear,
    wo: Linear,
    dropout: Dropout,
    act: HiddenAct,
}

impl T5DenseActDense {
    fn load(p: &str, vb: &VarBuilder, cfg: &T5Config) -> Result<Self> {
        let wi = Linear::load(cfg.d_model, cfg.d_ff, false, &format!("{p}.wi"), vb)?;
        let wo = Linear::load(cfg.d_ff, cfg.d_model, false, &format!("{p}.wo"), vb)?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            wi,
            wo,
            dropout,
            act: HiddenAct::Relu,
        })
    }
}

#[derive(Debug)]
struct T5LayerFF {
    dense_relu_dense: T5DenseActDense,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5LayerFF {
    fn load(p: &str, vb: &VarBuilder, cfg: &T5Config) -> Result<Self> {
        // is_gated_act is not supported.
        let dense_relu_dense = T5DenseActDense::load(&format!("{p}.DenseReluDense"), vb, cfg)?;
        let layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            &format!("{p}.layer_norm"),
            vb,
        )?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            dense_relu_dense,
            layer_norm,
            dropout,
        })
    }
}

#[derive(Debug)]
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative_attention_bias: Option<Embedding>,
}

impl T5Attention {
    fn load(h: bool, p: &str, vb: &VarBuilder, cfg: &T5Config) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = Linear::load(cfg.d_model, inner_dim, false, &format!("{p}.q"), vb)?;
        let k = Linear::load(cfg.d_model, inner_dim, false, &format!("{p}.k"), vb)?;
        let v = Linear::load(cfg.d_model, inner_dim, false, &format!("{p}.v"), vb)?;
        let o = Linear::load(inner_dim, cfg.d_model, false, &format!("{p}.o"), vb)?;
        let relative_attention_bias = if h {
            let emb = Embedding::load(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                &format!("{p}.relative_attention_bias"),
                vb,
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            relative_attention_bias,
        })
    }
}

#[derive(Debug)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5LayerSelfAttention {
    fn load(h: bool, p: &str, vb: &VarBuilder, cfg: &T5Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, &format!("{p}.SelfAttention"), vb, cfg)?;
        let layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            &format!("{p}.layer_norm"),
            vb,
        )?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            self_attention,
            layer_norm,
            dropout,
        })
    }
}

#[derive(Debug)]
struct T5LayerCrossAttention {}

impl T5LayerCrossAttention {
    fn load(_p: &str, _vb: &VarBuilder, _cfg: &T5Config) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(
        has_relative_attention_bias: bool,
        p: &str,
        vb: &VarBuilder,
        cfg: &T5Config,
    ) -> Result<Self> {
        let p = &format!("{p}.layer");
        let self_attn =
            T5LayerSelfAttention::load(has_relative_attention_bias, &format!("{p}.0"), vb, cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(&format!("{p}.1"), vb, cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(&format!("{p}.{ff_i}"), vb, cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
        })
    }
}

#[derive(Debug)]
struct T5Stack {
    // TODO: Add embed_tokens if needed (shared embedding layer).
    block: Vec<T5Block>,
    final_layer_norm: T5LayerNorm,
    dropout: Dropout,
}

impl T5Stack {
    fn load(p: &str, vb: &VarBuilder, cfg: &T5Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, &format!("{p}.block.{i}"), vb, cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            &format!("{p}.final_layer_norm"),
            vb,
        )?;
        let dropout = Dropout::new(cfg.dropout_rate);
        Ok(Self {
            block,
            final_layer_norm,
            dropout,
        })
    }
}

#[derive(Debug)]
struct T5EncoderModel {
    shared: Embedding,
    encoder: T5Stack,
}

impl T5EncoderModel {
    fn load(p: &str, vb: &VarBuilder, cfg: &T5Config) -> Result<Self> {
        let shared = Embedding::load(cfg.vocab_size, cfg.d_model, &format!("{p}.shared"), vb)?;
        let encoder = T5Stack::load(&format!("{p}.encoder"), vb, cfg)?;
        Ok(Self { shared, encoder })
    }
}

// Encodec Model
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py

#[derive(Debug, Clone, PartialEq)]
enum NormType {
    WeightNorm,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodecConfig {
    target_bandwidths: Vec<f64>,
    sampling_rate: usize,
    audio_channels: usize,
    normalize: bool,
    chunk_length_s: Option<usize>,
    overlap: Option<usize>,
    hidden_size: usize,
    num_filters: usize,
    num_residual_layers: usize,
    upsampling_ratios: Vec<usize>,
    norm_type: NormType,
    kernel_size: usize,
    last_kernel_size: usize,
    residual_kernel_size: usize,
    dilation_growth_rate: usize,
    use_causal_conv: bool,
    pad_mode: &'static str,
    compress: usize,
    num_lstm_layers: usize,
    trim_right_ratio: f64,
    codebook_size: usize,
    codebook_dim: Option<usize>,
    use_conv_shortcut: bool,
}

impl Default for EncodecConfig {
    fn default() -> Self {
        Self {
            target_bandwidths: vec![1.5, 3.0, 6.0, 12.0, 24.0],
            sampling_rate: 24_000,
            audio_channels: 1,
            normalize: false,
            chunk_length_s: None,
            overlap: None,
            hidden_size: 128,
            num_filters: 32,
            num_residual_layers: 1,
            upsampling_ratios: vec![8, 5, 4, 2],
            norm_type: NormType::WeightNorm,
            kernel_size: 7,
            last_kernel_size: 7,
            residual_kernel_size: 3,
            dilation_growth_rate: 2,
            use_causal_conv: true,
            pad_mode: "reflect",
            compress: 2,
            num_lstm_layers: 2,
            trim_right_ratio: 1.0,
            codebook_size: 1024,
            codebook_dim: None,
            use_conv_shortcut: true,
        }
    }
}

impl EncodecConfig {
    // https://huggingface.co/facebook/musicgen-small/blob/495da4ad086b3416a27c6187f9239f9fd96f3962/config.json#L6
    fn musicgen_small() -> Self {
        Self {
            audio_channels: 1,
            chunk_length_s: None,
            codebook_dim: Some(128),
            codebook_size: 2048,
            compress: 2,
            dilation_growth_rate: 2,
            hidden_size: 128,
            kernel_size: 7,
            last_kernel_size: 7,
            norm_type: NormType::WeightNorm,
            normalize: false,
            num_filters: 64,
            num_lstm_layers: 2,
            num_residual_layers: 1,
            overlap: None,
            pad_mode: "reflect",
            residual_kernel_size: 3,
            sampling_rate: 32_000,
            target_bandwidths: vec![2.2],
            trim_right_ratio: 1.0,
            upsampling_ratios: vec![8, 5, 4, 4],
            use_causal_conv: false,
            use_conv_shortcut: false,
        }
    }
}

#[derive(Debug)]
struct EncodecEuclideanCodebook {}

impl EncodecEuclideanCodebook {
    fn load(_p: &str, _vb: &VarBuilder, _cfg: &EncodecConfig) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug)]
struct EncodecVectorQuantization {}

impl EncodecVectorQuantization {
    fn load(_p: &str, _vb: &VarBuilder, _cfg: &EncodecConfig) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug)]
struct EncodecResidualVectorQuantizer {}

impl EncodecResidualVectorQuantizer {
    fn load(_p: &str, _vb: &VarBuilder, _cfg: &EncodecConfig) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug)]
struct EncodecLSTM {}

impl EncodecLSTM {
    fn load(_dimension: usize, _p: &str, _vb: &VarBuilder, cfg: &EncodecConfig) -> Result<Self> {
        let _ = cfg.num_lstm_layers;
        todo!()
    }
}

#[derive(Debug)]
struct EncodecConvTranspose1d {}

impl EncodecConvTranspose1d {
    fn load(
        _in_channels: usize,
        _out_channels: usize,
        _kernel_size: usize,
        _stride: usize,
        _p: &str,
        _vb: &VarBuilder,
    ) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug)]
struct EncodecConv1d {
    conv: Conv1D,
}

impl EncodecConv1d {
    fn load(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        p: &str,
        vb: &VarBuilder,
        cfg: &EncodecConfig,
    ) -> Result<Self> {
        let conv = match cfg.norm_type {
            NormType::WeightNorm => Conv1D::load_weight_norm(
                in_c,
                out_c,
                kernel_size,
                ConvConfig { padding: 0, stride },
                &format!("{p}.conv"),
                vb,
            )?,
            NormType::None => Conv1D::load(
                in_c,
                out_c,
                kernel_size,
                ConvConfig { padding: 0, stride },
                &format!("{p}.conv"),
                vb,
            )?,
        };
        Ok(Self { conv })
    }
}

#[derive(Debug)]
struct EncodecResnetBlock {
    block: Vec<EncodecConv1d>,
    shortcut: Option<EncodecConv1d>,
}

impl EncodecResnetBlock {
    fn load(
        dim: usize,
        _dilations: &[usize],
        p: &str,
        vb: &VarBuilder,
        cfg: &EncodecConfig,
    ) -> Result<Self> {
        let block = vec![];
        // TODO: Add the conv1d layers in block.
        let shortcut = if cfg.use_conv_shortcut {
            let conv = EncodecConv1d::load(dim, dim, 1, 1, &format!("{p}.shortcut"), vb, cfg)?;
            Some(conv)
        } else {
            None
        };
        Ok(Self { block, shortcut })
    }
}

#[derive(Debug)]
struct Layer {
    prefix: String,
    cnt: usize,
}

impl Layer {
    fn new(prefix: String) -> Self {
        Self { prefix, cnt: 0 }
    }

    fn inc(&mut self) {
        self.cnt += 1;
    }

    fn next_name(&mut self) -> String {
        let name = format!("{}.{}", self.prefix, self.cnt);
        self.cnt += 1;
        name
    }
}

#[derive(Debug)]
struct EncodecEncoder {
    init_conv: EncodecConv1d,
    sampling_layers: Vec<(Vec<EncodecResnetBlock>, EncodecConv1d)>,
    final_lstm: EncodecLSTM,
    final_conv: EncodecConv1d,
}

impl EncodecEncoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &EncodecConfig) -> Result<Self> {
        let mut layer = Layer::new(format!("{p}.layers"));
        let init_conv = EncodecConv1d::load(
            cfg.audio_channels,
            cfg.num_filters,
            cfg.kernel_size,
            1,
            &layer.next_name(),
            vb,
            cfg,
        )?;
        let mut sampling_layers = vec![];
        let mut scaling = 1;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let current_scale = scaling * cfg.num_filters;
            let mut resnets = vec![];
            for j in 0..(cfg.num_residual_layers as u32) {
                let resnet = EncodecResnetBlock::load(
                    current_scale,
                    &[cfg.dilation_growth_rate.pow(j), 1],
                    &layer.next_name(),
                    vb,
                    cfg,
                )?;
                resnets.push(resnet)
            }
            layer.inc(); // ELU
            let conv1d = EncodecConv1d::load(
                current_scale,
                current_scale * 2,
                ratio * 2,
                ratio,
                &layer.next_name(),
                vb,
                cfg,
            )?;
            sampling_layers.push((resnets, conv1d));
            scaling *= 2;
        }
        let final_lstm = EncodecLSTM::load(cfg.num_filters * scaling, &layer.next_name(), vb, cfg)?;
        layer.inc(); // ELU
        let final_conv = EncodecConv1d::load(
            cfg.num_filters * scaling,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
            &layer.next_name(),
            vb,
            cfg,
        )?;
        Ok(Self {
            init_conv,
            sampling_layers,
            final_conv,
            final_lstm,
        })
    }
}

#[derive(Debug)]
struct EncodecDecoder {
    init_conv: EncodecConv1d,
    init_lstm: EncodecLSTM,
    sampling_layers: Vec<(EncodecConvTranspose1d, Vec<EncodecResnetBlock>)>,
    final_conv: EncodecConv1d,
}

impl EncodecDecoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &EncodecConfig) -> Result<Self> {
        let mut layer = Layer::new(format!("{p}.layers"));
        let mut scaling = usize::pow(2, cfg.upsampling_ratios.len() as u32);
        let init_conv = EncodecConv1d::load(
            cfg.hidden_size,
            cfg.num_filters * scaling,
            cfg.last_kernel_size,
            1,
            &layer.next_name(),
            vb,
            cfg,
        )?;
        let init_lstm = EncodecLSTM::load(cfg.num_filters * scaling, &layer.next_name(), vb, cfg)?;
        let mut sampling_layers = vec![];
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let current_scale = scaling * cfg.num_filters;
            layer.inc(); // ELU
            let conv1d = EncodecConvTranspose1d::load(
                current_scale,
                current_scale / 2,
                ratio * 2,
                ratio,
                &layer.next_name(),
                vb,
            )?;
            let mut resnets = vec![];
            for j in 0..(cfg.num_residual_layers as u32) {
                let resnet = EncodecResnetBlock::load(
                    current_scale / 2,
                    &[cfg.dilation_growth_rate.pow(j), 1],
                    &layer.next_name(),
                    vb,
                    cfg,
                )?;
                resnets.push(resnet)
            }
            sampling_layers.push((conv1d, resnets));
            scaling /= 2;
        }
        layer.inc(); // ELU
        let final_conv = EncodecConv1d::load(
            cfg.num_filters,
            cfg.audio_channels,
            cfg.last_kernel_size,
            1,
            &layer.next_name(),
            vb,
            cfg,
        )?;
        Ok(Self {
            init_conv,
            init_lstm,
            sampling_layers,
            final_conv,
        })
    }
}

#[derive(Debug)]
struct EncodecModel {
    encoder: EncodecEncoder,
    decoder: EncodecDecoder,
    quantizer: EncodecResidualVectorQuantizer,
}

impl EncodecModel {
    fn load(p: &str, vb: &VarBuilder, cfg: &EncodecConfig) -> Result<Self> {
        let encoder = EncodecEncoder::load(&format!("{p}.encoder"), vb, cfg)?;
        let decoder = EncodecDecoder::load(&format!("{p}.decoder"), vb, cfg)?;
        let quantizer = EncodecResidualVectorQuantizer::load(&format!("{p}.quantizer"), vb, cfg)?;
        Ok(Self {
            encoder,
            decoder,
            quantizer,
        })
    }
}

#[derive(Debug)]
pub struct MusicgenForConditionalGeneration {
    text_encoder: T5EncoderModel,
    audio_encoder: EncodecModel,
    decoder: MusicgenForCausalLM,
    cfg: Config,
}

impl MusicgenForConditionalGeneration {
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn load(vb: &VarBuilder, cfg: Config) -> Result<Self> {
        let t5_cfg = T5Config::musicgen_small(); // TODO: Get as argument.
        let encodec_cfg = EncodecConfig::musicgen_small(); // TODO
        let text_encoder = T5EncoderModel::load("text_encoder", vb, &t5_cfg)?;
        let audio_encoder = EncodecModel::load("audio_encoder", vb, &encodec_cfg)?;
        let decoder = MusicgenForCausalLM::load("decoder", vb, &cfg)?;
        Ok(Self {
            text_encoder,
            audio_encoder,
            decoder,
            cfg,
        })
    }
}
