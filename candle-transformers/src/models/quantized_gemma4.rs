//! Gemma 4 text model implementation with GGUF quantization support.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle::quantized::{gguf_file, QMatMul};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::kv_cache::{ConcatKvCache, RotatingKvCache};
use candle_nn::Activation;

use crate::utils::repeat_kv;

#[derive(Clone, Debug)]
struct Config {
    block_count: usize,
    context_length: usize,
    hidden_size: usize,
    attention_heads: Vec<usize>,
    kv_heads: Vec<usize>,
    global_head_dim: usize,
    local_head_dim: usize,
    rms_norm_eps: f64,
    global_rope_base: f64,
    local_rope_base: f64,
    final_logit_softcap: f64,
    sliding_window: usize,
    shared_kv_layers: usize,
    per_layer_input_dim: usize,
    has_per_layer_inputs: bool,
    sliding_pattern: Vec<bool>,
    dtype: DType,
}

impl Config {
    fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let architecture = metadata(content, "general.architecture")?.to_string()?;
        if architecture != "gemma4" {
            candle::bail!("model architecture is {architecture}, expected gemma4")
        }
        let dtype = match content.metadata.get("general.dtype") {
            Some(value) => match value.to_u32()? {
                0 => DType::F32,
                1 => DType::F16,
                value => candle::bail!("unsupported general.dtype value {value}"),
            },
            None => DType::F16,
        };
        let sliding_pattern = metadata(content, "gemma4.attention.sliding_window_pattern")?
            .to_vec()?
            .iter()
            .map(|value| value.to_bool())
            .collect::<Result<Vec<_>>>()?;
        let block_count = usize_metadata(content, "gemma4.block_count")?;
        if sliding_pattern.len() != block_count {
            candle::bail!(
                "Gemma 4 sliding pattern has {} entries for {block_count} blocks",
                sliding_pattern.len()
            )
        }
        let shared_kv_layers = usize_metadata(content, "gemma4.attention.shared_kv_layers")?;
        if shared_kv_layers + 2 > block_count {
            candle::bail!(
                "Gemma 4 requires at least two KV-owning layers, got {block_count} blocks and {shared_kv_layers} shared layers"
            )
        }
        let kv_layer_count = block_count - shared_kv_layers;
        if !sliding_pattern[kv_layer_count - 2] || sliding_pattern[kv_layer_count - 1] {
            candle::bail!(
                "Gemma 4 shared KV source layers must end with one sliding and one global layer"
            )
        }
        let per_layer_input_dim =
            usize_metadata(content, "gemma4.embedding_length_per_layer_input")?;
        Ok(Self {
            block_count,
            context_length: usize_metadata(content, "gemma4.context_length")?,
            hidden_size: usize_metadata(content, "gemma4.embedding_length")?,
            attention_heads: per_layer_usize_metadata(
                content,
                "gemma4.attention.head_count",
                block_count,
            )?,
            kv_heads: per_layer_usize_metadata(
                content,
                "gemma4.attention.head_count_kv",
                block_count,
            )?,
            global_head_dim: usize_metadata(content, "gemma4.attention.key_length")?,
            local_head_dim: usize_metadata(content, "gemma4.attention.key_length_swa")?,
            rms_norm_eps: metadata(content, "gemma4.attention.layer_norm_rms_epsilon")?.to_f32()?
                as f64,
            global_rope_base: metadata(content, "gemma4.rope.freq_base")?.to_f32()? as f64,
            local_rope_base: metadata(content, "gemma4.rope.freq_base_swa")?.to_f32()? as f64,
            final_logit_softcap: metadata(content, "gemma4.final_logit_softcapping")?.to_f32()?
                as f64,
            sliding_window: usize_metadata(content, "gemma4.attention.sliding_window")?,
            shared_kv_layers,
            per_layer_input_dim: per_layer_input_dim.max(1),
            has_per_layer_inputs: per_layer_input_dim > 0,
            sliding_pattern,
            dtype,
        })
    }

    fn is_sliding(&self, layer: usize) -> bool {
        self.sliding_pattern[layer]
    }
}

struct Embeddings {
    token_embedding: QMatMul,
    per_layer_token_embedding: Option<QMatMul>,
    per_layer_model_projection: Option<QMatMul>,
    per_layer_projection_norm: Option<GemmaRmsNorm>,
    hidden_size: usize,
    layer_count: usize,
    per_layer_input_dim: usize,
}

impl Embeddings {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        config: &Config,
    ) -> Result<Self> {
        Ok(Self {
            token_embedding: qmatmul(content, reader, "token_embd.weight", device)?,
            per_layer_token_embedding: config
                .has_per_layer_inputs
                .then(|| qmatmul(content, reader, "per_layer_token_embd.weight", device))
                .transpose()?,
            per_layer_model_projection: config
                .has_per_layer_inputs
                .then(|| qmatmul(content, reader, "per_layer_model_proj.weight", device))
                .transpose()?,
            per_layer_projection_norm: config
                .has_per_layer_inputs
                .then(|| {
                    GemmaRmsNorm::load(
                        content,
                        reader,
                        "per_layer_proj_norm.weight",
                        device,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            hidden_size: config.hidden_size,
            layer_count: config.block_count,
            per_layer_input_dim: config.per_layer_input_dim,
        })
    }

    fn forward(&self, tokens: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, sequence) = tokens.dims2()?;
        let hidden = (self.token_embedding.embedding(tokens)? * (self.hidden_size as f64).sqrt())?;
        let per_layer_inputs = match (
            &self.per_layer_token_embedding,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(token_embedding), Some(model_projection), Some(projection_norm)) => {
                let token_inputs = (token_embedding.embedding(tokens)?
                    * (self.per_layer_input_dim as f64).sqrt())?
                .reshape((batch, sequence, self.layer_count, self.per_layer_input_dim))?;
                let projected_inputs = (model_projection.forward(&hidden)?
                    * (1.0 / (self.hidden_size as f64).sqrt()))?
                .reshape((batch, sequence, self.layer_count, self.per_layer_input_dim))?;
                let projected_inputs = projection_norm.forward(&projected_inputs)?;
                ((token_inputs + projected_inputs)? * (1.0 / 2f64.sqrt()))?
            }
            (None, None, None) => Tensor::zeros(
                (batch, sequence, self.layer_count, self.per_layer_input_dim),
                hidden.dtype(),
                hidden.device(),
            )?,
            _ => candle::bail!("Gemma 4 per-layer embedding tensors are incomplete"),
        };
        Ok((hidden, per_layer_inputs))
    }
}

#[derive(Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_base: f64,
        freq_factors: Option<&Tensor>,
        context_length: usize,
        device: &Device,
    ) -> Result<Self> {
        let inverse_frequencies = (0..head_dim / 2)
            .map(|index| 1f32 / rope_base.powf((index * 2) as f64 / head_dim as f64) as f32)
            .collect::<Vec<_>>();
        let mut inverse_frequencies =
            Tensor::from_vec(inverse_frequencies, (1, head_dim / 2), device)?;
        if let Some(freq_factors) = freq_factors {
            inverse_frequencies =
                inverse_frequencies.broadcast_div(&freq_factors.reshape((1, head_dim / 2))?)?;
        }
        let positions = Tensor::arange(0u32, context_length as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((context_length, 1))?;
        let frequencies = positions.matmul(&inverse_frequencies)?;
        Ok(Self {
            sin: frequencies.sin()?.to_dtype(dtype)?,
            cos: frequencies.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(
        &self,
        query: &Tensor,
        key: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let sequence = query.dim(2)?;
        let cosine = self
            .cos
            .narrow(0, offset, sequence)?
            .to_dtype(query.dtype())?;
        let sine = self
            .sin
            .narrow(0, offset, sequence)?
            .to_dtype(query.dtype())?;
        let query = candle_nn::rotary_emb::rope(&query.contiguous()?, &cosine, &sine)?;
        let key = key
            .map(|key| candle_nn::rotary_emb::rope(&key.contiguous()?, &cosine, &sine))
            .transpose()?;
        Ok((query, key))
    }
}

#[derive(Debug)]
enum KvCache {
    Global(ConcatKvCache),
    Local(RotatingKvCache),
}

impl KvCache {
    fn new(is_sliding: bool, sliding_window: usize) -> Self {
        if is_sliding {
            Self::Local(RotatingKvCache::new(2, sliding_window))
        } else {
            Self::Global(ConcatKvCache::new(2))
        }
    }

    fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Global(cache) => cache.append(key, value),
            Self::Local(cache) => cache.append(key, value),
        }
    }

    fn tensors(&self) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Global(cache) => match (cache.k(), cache.v()) {
                (Some(key), Some(value)) => Ok((key.clone(), value.clone())),
                _ => candle::bail!("Gemma 4 shared global KV cache is empty"),
            },
            Self::Local(cache) => match (cache.k()?, cache.v()?) {
                (Some(key), Some(value)) => Ok((key, value)),
                _ => candle::bail!("Gemma 4 shared local KV cache is empty"),
            },
        }
    }

    fn key_positions(&self) -> Vec<usize> {
        match self {
            Self::Global(cache) => (0..cache.current_seq_len()).collect(),
            Self::Local(cache) => cache.positions(0),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Global(cache) => cache.reset(),
            Self::Local(cache) => cache.reset(),
        }
    }
}

struct Attention {
    query: QMatMul,
    key: Option<QMatMul>,
    value: Option<QMatMul>,
    output: QMatMul,
    query_norm: GemmaRmsNorm,
    key_norm: Option<GemmaRmsNorm>,
    attention_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    is_sliding: bool,
    cache_index: usize,
    rotary: Arc<RotaryEmbedding>,
    rms_norm_eps: f64,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        config: &Config,
        layer_index: usize,
        kv_layer_count: usize,
        global_rotary: Arc<RotaryEmbedding>,
        local_rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_index}");
        let is_sliding = config.is_sliding(layer_index);
        let owns_kv = layer_index < kv_layer_count;
        let cache_index = if owns_kv {
            layer_index
        } else if is_sliding {
            kv_layer_count - 2
        } else {
            kv_layer_count - 1
        };
        Ok(Self {
            query: qmatmul(content, reader, &format!("{prefix}.attn_q.weight"), device)?,
            key: owns_kv
                .then(|| qmatmul(content, reader, &format!("{prefix}.attn_k.weight"), device))
                .transpose()?,
            value: (owns_kv
                && content
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_v.weight")))
            .then(|| qmatmul(content, reader, &format!("{prefix}.attn_v.weight"), device))
            .transpose()?,
            output: qmatmul(
                content,
                reader,
                &format!("{prefix}.attn_output.weight"),
                device,
            )?,
            query_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.attn_q_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            key_norm: owns_kv
                .then(|| {
                    GemmaRmsNorm::load(
                        content,
                        reader,
                        &format!("{prefix}.attn_k_norm.weight"),
                        device,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            attention_heads: config.attention_heads[layer_index],
            kv_heads: config.kv_heads[layer_index],
            head_dim: if is_sliding {
                config.local_head_dim
            } else {
                config.global_head_dim
            },
            is_sliding,
            cache_index,
            rotary: if is_sliding {
                local_rotary
            } else {
                global_rotary
            },
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        offset: usize,
        cache: &mut KvCache,
        device: &Device,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let query = self
            .query
            .forward(hidden)?
            .reshape((batch, sequence, self.attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query = self.query_norm.forward(&query)?;
        let key = self
            .key
            .as_ref()
            .map(|key| {
                key.forward(hidden)?
                    .reshape((batch, sequence, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)
            })
            .transpose()?;
        let value = self
            .value
            .as_ref()
            .map(|value| {
                value
                    .forward(hidden)?
                    .reshape((batch, sequence, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)
            })
            .transpose()?;
        let value = match (value, key.as_ref()) {
            (Some(value), _) => Some(value),
            (None, Some(key)) => Some(key.clone()),
            (None, None) => None,
        };
        let key = match (key, self.key_norm.as_ref()) {
            (Some(key), Some(norm)) => Some(norm.forward(&key)?),
            (None, None) => None,
            _ => candle::bail!("Gemma 4 key projection and key norm ownership differ"),
        };
        let value = value
            .map(|value| value_norm(&value, self.rms_norm_eps))
            .transpose()?;
        let (query, key) = self.rotary.apply(&query, key.as_ref(), offset)?;
        let (key, value) = match (key, value) {
            (Some(key), Some(value)) => cache.append(&key, &value)?,
            (None, None) => cache.tensors()?,
            _ => candle::bail!("Gemma 4 key and value ownership differ"),
        };
        let key_positions = cache.key_positions();
        let key = repeat_kv(key, self.attention_heads / self.kv_heads)?.contiguous()?;
        let value = repeat_kv(value, self.attention_heads / self.kv_heads)?.contiguous()?;
        let mut scores = query.matmul(&key.transpose(2, 3)?)?;
        if sequence > 1 {
            let mask = attention_mask(
                sequence,
                offset,
                &key_positions,
                self.is_sliding.then_some(sliding_window),
                device,
                scores.dtype(),
            )?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probabilities.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            sequence,
            self.attention_heads * self.head_dim,
        ))?;
        self.output.forward(&context)
    }
}

struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate: qmatmul(
                content,
                reader,
                &format!("{prefix}.ffn_gate.weight"),
                device,
            )?,
            up: qmatmul(content, reader, &format!("{prefix}.ffn_up.weight"), device)?,
            down: qmatmul(
                content,
                reader,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(hidden)?.apply(&Activation::Gelu)?;
        self.down.forward(&(gate * self.up.forward(hidden)?)?)
    }
}

struct Layer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: GemmaRmsNorm,
    post_attention_norm: GemmaRmsNorm,
    ffn_norm: GemmaRmsNorm,
    post_ffn_norm: GemmaRmsNorm,
    per_layer_input_gate: Option<QMatMul>,
    per_layer_projection: Option<QMatMul>,
    per_layer_post_norm: Option<GemmaRmsNorm>,
    output_scale: Option<Tensor>,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        config: &Config,
        layer_index: usize,
        kv_layer_count: usize,
        global_rotary: Arc<RotaryEmbedding>,
        local_rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_index}");
        Ok(Self {
            attention: Attention::load(
                content,
                reader,
                device,
                config,
                layer_index,
                kv_layer_count,
                global_rotary,
                local_rotary,
            )?,
            mlp: Mlp::load(content, reader, device, &prefix)?,
            attention_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.attn_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            post_attention_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            ffn_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.ffn_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            post_ffn_norm: GemmaRmsNorm::load(
                content,
                reader,
                &format!("{prefix}.post_ffw_norm.weight"),
                device,
                config.rms_norm_eps,
            )?,
            per_layer_input_gate: config
                .has_per_layer_inputs
                .then(|| {
                    qmatmul(
                        content,
                        reader,
                        &format!("{prefix}.inp_gate.weight"),
                        device,
                    )
                })
                .transpose()?,
            per_layer_projection: config
                .has_per_layer_inputs
                .then(|| qmatmul(content, reader, &format!("{prefix}.proj.weight"), device))
                .transpose()?,
            per_layer_post_norm: config
                .has_per_layer_inputs
                .then(|| {
                    GemmaRmsNorm::load(
                        content,
                        reader,
                        &format!("{prefix}.post_norm.weight"),
                        device,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            output_scale: content
                .tensor_infos
                .contains_key(&format!("{prefix}.layer_output_scale.weight"))
                .then(|| {
                    content
                        .tensor(
                            reader,
                            &format!("{prefix}.layer_output_scale.weight"),
                            device,
                        )?
                        .dequantize(device)
                })
                .transpose()?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        per_layer_input: &Tensor,
        offset: usize,
        cache: &mut KvCache,
        device: &Device,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let attention = self.attention.forward(
            &self.attention_norm.forward(hidden)?,
            offset,
            cache,
            device,
            sliding_window,
        )?;
        let attention = self.post_attention_norm.forward(&attention)?;
        let hidden = (hidden + attention)?;
        let mlp = self.mlp.forward(&self.ffn_norm.forward(&hidden)?)?;
        let mlp = self.post_ffn_norm.forward(&mlp)?;
        let hidden = (hidden + mlp)?;
        let hidden = match (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.per_layer_post_norm,
        ) {
            (Some(input_gate), Some(projection), Some(post_norm)) => {
                let gated_input = input_gate.forward(&hidden)?.apply(&Activation::Gelu)?;
                let per_layer_output = projection.forward(&(gated_input * per_layer_input)?)?;
                hidden + post_norm.forward(&per_layer_output)?
            }
            (None, None, None) => Ok(hidden),
            _ => candle::bail!("Gemma 4 per-layer residual tensors are incomplete"),
        }?;
        match &self.output_scale {
            Some(output_scale) => hidden.broadcast_mul(output_scale),
            None => Ok(hidden),
        }
    }
}

/// Quantized Gemma 4 text model loaded from a GGUF file.
pub struct ModelWeights {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    caches: Vec<KvCache>,
    norm: GemmaRmsNorm,
    output: QMatMul,
    final_logit_softcap: f64,
    device: Device,
    sliding_window: usize,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let config = Config::from_gguf(&content)?;
        let kv_layer_count = config.block_count - config.shared_kv_layers;
        let global_rope_factors = content
            .tensor(reader, "rope_freqs.weight", device)?
            .dequantize(device)?;
        let global_rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.global_head_dim,
            config.global_rope_base,
            Some(&global_rope_factors),
            config.context_length,
            device,
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.local_head_dim,
            config.local_rope_base,
            None,
            config.context_length,
            device,
        )?);
        let embeddings = Embeddings::load(&content, reader, device, &config)?;
        let mut layers = Vec::with_capacity(config.block_count);
        for index in 0..config.block_count {
            layers.push(Layer::load(
                &content,
                reader,
                device,
                &config,
                index,
                kv_layer_count,
                global_rotary.clone(),
                local_rotary.clone(),
            )?);
        }
        let caches = (0..kv_layer_count)
            .map(|index| KvCache::new(config.is_sliding(index), config.sliding_window))
            .collect();
        let output_name = if content.tensor_infos.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        Ok(Self {
            embeddings,
            layers,
            caches,
            norm: GemmaRmsNorm::load(
                &content,
                reader,
                "output_norm.weight",
                device,
                config.rms_norm_eps,
            )?,
            output: qmatmul(&content, reader, output_name, device)?,
            final_logit_softcap: config.final_logit_softcap,
            device: device.clone(),
            sliding_window: config.sliding_window,
        })
    }

    pub fn forward(&mut self, tokens: &Tensor, offset: usize) -> Result<Tensor> {
        let (mut hidden, per_layer_inputs) = self.embeddings.forward(tokens)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer_inputs.narrow(2, index, 1)?.squeeze(2)?;
            hidden = layer.forward(
                &hidden,
                &per_layer_input,
                offset,
                &mut self.caches[layer.attention.cache_index],
                &self.device,
                self.sliding_window,
            )?;
        }
        let sequence = hidden.dim(1)?;
        let hidden = self.norm.forward(&hidden.narrow(1, sequence - 1, 1)?)?;
        let logits = self.output.forward(&hidden)?.squeeze(1)?;
        (logits / self.final_logit_softcap)?.tanh()? * self.final_logit_softcap
    }

    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }
}

fn value_norm(value: &Tensor, epsilon: f64) -> Result<Tensor> {
    let original_dtype = value.dtype();
    let value_f32 = value.to_dtype(DType::F32)?;
    let variance = value_f32.sqr()?.mean_keepdim(D::Minus1)?;
    value_f32
        .broadcast_div(&(variance + epsilon)?.sqrt()?)?
        .to_dtype(original_dtype)
}

fn attention_mask(
    sequence: usize,
    offset: usize,
    key_positions: &[usize],
    sliding_window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask = (0..sequence)
        .flat_map(|query_index| {
            let query_position = offset + query_index;
            key_positions.iter().map(move |&key_position| {
                let outside_window =
                    sliding_window.is_some_and(|window| key_position + window < query_position);
                if key_position > query_position || outside_window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_slice(&mask, (1, 1, sequence, key_positions.len()), device)?.to_dtype(dtype)
}

struct GemmaRmsNorm {
    weight: Tensor,
    epsilon: f64,
}

impl GemmaRmsNorm {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
        epsilon: f64,
    ) -> Result<Self> {
        Ok(Self {
            weight: content.tensor(reader, name, device)?.dequantize(device)?,
            epsilon,
        })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let original_dtype = hidden.dtype();
        let hidden_f32 = hidden.to_dtype(DType::F32)?;
        let variance = hidden_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = hidden_f32.broadcast_div(&(variance + self.epsilon)?.sqrt()?)?;
        normalized
            .to_dtype(original_dtype)?
            .broadcast_mul(&self.weight)
    }
}

fn metadata<'a>(content: &'a gguf_file::Content, name: &str) -> Result<&'a gguf_file::Value> {
    content
        .metadata
        .get(name)
        .ok_or_else(|| candle::Error::Msg(format!("cannot find {name} in metadata")))
}

fn usize_metadata(content: &gguf_file::Content, name: &str) -> Result<usize> {
    Ok(metadata(content, name)?.to_u32()? as usize)
}

fn per_layer_usize_metadata(
    content: &gguf_file::Content,
    name: &str,
    block_count: usize,
) -> Result<Vec<usize>> {
    let value = metadata(content, name)?;
    let values = match value {
        gguf_file::Value::U32(value) => vec![*value as usize; block_count],
        gguf_file::Value::I32(value) if *value >= 0 => vec![*value as usize; block_count],
        gguf_file::Value::Array(values) => values
            .iter()
            .map(|value| match value {
                gguf_file::Value::U32(value) => Ok(*value as usize),
                gguf_file::Value::I32(value) if *value >= 0 => Ok(*value as usize),
                value => candle::bail!("{name} contains a non-negative 32-bit integer: {value:?}"),
            })
            .collect::<Result<Vec<_>>>()?,
        value => candle::bail!("{name} is not a scalar or array of 32-bit integers: {value:?}"),
    };
    if values.len() != block_count {
        candle::bail!(
            "{name} has {} entries for {block_count} blocks",
            values.len()
        )
    }
    Ok(values)
}

fn qmatmul<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<QMatMul> {
    QMatMul::from_qtensor(content.tensor(reader, name, device)?)
}
