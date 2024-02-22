use crate::models::with_tracing::{linear_b as linear, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Config {
    pub num_layers: usize,
    pub padded_vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub kv_channels: usize,
    pub num_attention_heads: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
    pub rmsnorm: bool,
    pub apply_residual_connection_post_layernorm: bool,
    pub post_layer_norm: bool,
    pub add_bias_linear: bool,
    pub add_qkv_bias: bool,
    pub bias_dropout_fusion: bool,
    pub multi_query_attention: bool,
    pub multi_query_group_num: usize,
    pub apply_query_key_layer_scaling: bool,
    pub attention_softmax_in_fp32: bool,
    pub fp32_residual_connection: bool,
}

impl Config {
    pub fn glm3_6b() -> Self {
        Self {
            num_layers: 28,
            padded_vocab_size: 65024,
            hidden_size: 4096,
            ffn_hidden_size: 13696,
            kv_channels: 128,
            num_attention_heads: 32,
            seq_length: 8192,
            layernorm_epsilon: 1e-5,
            rmsnorm: true,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            add_bias_linear: false,
            add_qkv_bias: true,
            bias_dropout_fusion: true,
            multi_query_attention: true,
            multi_query_group_num: 2,
            apply_query_key_layer_scaling: true,
            attention_softmax_in_fp32: true,
            fp32_residual_connection: false,
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cache: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dtype: DType, dev: &Device) -> Result<Self> {
        let rotary_dim = cfg.kv_channels;
        let n_elem = rotary_dim / 2;
        let inv_freq: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / 10_000f64.powf(i as f64 / n_elem as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.seq_length as u32, dev)?
            .to_dtype(dtype)?
            .reshape((cfg.seq_length, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cache = Tensor::stack(&[&freqs.cos()?, &freqs.sin()?], D::Minus1)?;
        Ok(Self { cache })
    }

    fn apply(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (seqlen, _b, np, _hn) = xs.dims4()?;
        let cache = self.cache.narrow(0, seqlen_offset, seqlen)?;
        let rot_dim = cache.dim(D::Minus2)? * 2;
        let (xs, xs_pass) = (
            xs.narrow(D::Minus1, 0, rot_dim)?,
            xs.narrow(D::Minus1, rot_dim, rot_dim)?,
        );
        let xshaped = xs.reshape((seqlen, (), np, rot_dim / 2, 2))?;
        let cache = cache.reshape((seqlen, (), 1, rot_dim / 2, 2))?;
        let (xshaped0, xshaped1) = (
            xshaped.i((.., .., .., .., 0))?,
            xshaped.i((.., .., .., .., 1))?,
        );
        let (cache0, cache1) = (cache.i((.., .., .., .., 0))?, cache.i((.., .., .., .., 1))?);
        let xs_out = Tensor::stack(
            &[
                (xshaped0.broadcast_mul(&cache0)? - xshaped1.broadcast_mul(&cache1)?)?,
                (xshaped1.broadcast_mul(&cache0)? + xshaped0.broadcast_mul(&cache1)?)?,
            ],
            D::Minus1,
        )?;
        let xs_out = xs_out.flatten_from(3)?;
        Tensor::cat(&[xs_out, xs_pass], D::Minus1)
    }
}

#[derive(Debug, Clone)]
struct CoreAttention {
    coeff: Option<f64>,
    norm_factor: f64,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl CoreAttention {
    fn new(layer_number: usize, cfg: &Config) -> Result<Self> {
        let norm_factor = (cfg.kv_channels as f64).sqrt();
        let (norm_factor, coeff) = if cfg.apply_query_key_layer_scaling {
            let coeff = f64::max(1.0, layer_number as f64);
            (norm_factor * coeff, Some(coeff))
        } else {
            (norm_factor, None)
        };
        Ok(Self { coeff, norm_factor })
    }

    fn forward(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        value_layer: &Tensor,
        attention_mask: &Option<Tensor>,
    ) -> Result<Tensor> {
        let output_size = (
            query_layer.dim(1)?, // b
            query_layer.dim(2)?, // np
            query_layer.dim(0)?, // sq
            key_layer.dim(0)?,   // sk
        );
        let query_layer =
            query_layer.reshape((output_size.2, output_size.0 * output_size.1, ()))?;
        let key_layer = key_layer.reshape((output_size.3, output_size.0 * output_size.1, ()))?;
        let matmul_result = Tensor::matmul(
            &query_layer.transpose(0, 1)?,
            &key_layer.transpose(0, 1)?.transpose(1, 2)?,
        )?;
        let matmul_result = (matmul_result / self.norm_factor)?.reshape(output_size)?;
        let matmul_result = match self.coeff {
            None => matmul_result,
            Some(coeff) => (matmul_result * coeff)?,
        };
        let attention_scores = match attention_mask {
            Some(mask) => masked_fill(
                &matmul_result,
                &mask.broadcast_left((matmul_result.dim(0)?, matmul_result.dim(1)?))?,
                f32::NEG_INFINITY,
            )?,
            None => matmul_result,
        };
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        let output_size = (
            value_layer.dim(1)?,
            value_layer.dim(2)?,
            query_layer.dim(0)?,
            value_layer.dim(3)?,
        );
        let value_layer =
            value_layer.reshape((value_layer.dim(0)?, output_size.0 * output_size.1, ()))?;
        let attention_probs =
            attention_probs.reshape((output_size.0 * output_size.1, output_size.2, ()))?;
        let context_layer = Tensor::matmul(&attention_probs, &value_layer.transpose(0, 1)?)?;
        let context_layer = context_layer.reshape(output_size)?;
        let context_layer = context_layer.permute((2, 0, 1, 3))?.contiguous()?;
        context_layer.flatten_from(D::Minus2)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    query_key_value: Linear,
    core_attention: CoreAttention,
    dense: Linear,
    multi_query_attention: bool,
    num_attention_heads_per_partition: usize,
    num_multi_query_groups_per_partition: usize,
    hidden_size_per_attention_head: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl SelfAttention {
    fn new(layer_number: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let projection_size = cfg.kv_channels * cfg.num_attention_heads;
        let hidden_size_per_attention_head = projection_size / cfg.num_attention_heads;
        let qkv_hidden_size = if cfg.multi_query_attention {
            projection_size + 2 * hidden_size_per_attention_head * cfg.multi_query_group_num
        } else {
            3 * projection_size
        };
        let query_key_value = linear(
            cfg.hidden_size,
            qkv_hidden_size,
            cfg.add_bias_linear || cfg.add_qkv_bias,
            vb.pp("query_key_value"),
        )?;
        let core_attention = CoreAttention::new(layer_number, cfg)?;
        let dense = linear(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense"),
        )?;
        Ok(Self {
            query_key_value,
            core_attention,
            dense,
            multi_query_attention: cfg.multi_query_attention,
            num_attention_heads_per_partition: cfg.num_attention_heads,
            num_multi_query_groups_per_partition: cfg.multi_query_group_num,
            hidden_size_per_attention_head: cfg.kv_channels,
            kv_cache: None,
        })
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: &Option<Tensor>,
        rotary_emb: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let mixed_x_layer = xs.apply(&self.query_key_value)?;
        if !self.multi_query_attention {
            candle::bail!("only multi_query_attention=true is supported")
        }
        let hpa = self.hidden_size_per_attention_head;
        let query_layer =
            mixed_x_layer.narrow(D::Minus1, 0, self.num_attention_heads_per_partition * hpa)?;
        let key_layer = mixed_x_layer.narrow(
            D::Minus1,
            self.num_attention_heads_per_partition * hpa,
            self.num_multi_query_groups_per_partition * hpa,
        )?;
        let value_layer = mixed_x_layer.narrow(
            D::Minus1,
            self.num_attention_heads_per_partition * hpa
                + self.num_multi_query_groups_per_partition * hpa,
            self.num_multi_query_groups_per_partition * hpa,
        )?;
        let query_layer = query_layer.reshape((
            query_layer.dim(0)?,
            query_layer.dim(1)?,
            self.num_attention_heads_per_partition,
            hpa,
        ))?;
        let key_layer = key_layer.reshape((
            key_layer.dim(0)?,
            key_layer.dim(1)?,
            self.num_multi_query_groups_per_partition,
            hpa,
        ))?;
        let value_layer = value_layer.reshape((
            value_layer.dim(0)?,
            value_layer.dim(1)?,
            self.num_multi_query_groups_per_partition,
            hpa,
        ))?;

        // Rotary embeddings.
        let seqlen_offset = match &self.kv_cache {
            None => 0,
            Some((prev_k, _)) => prev_k.dim(0)?,
        };
        let query_layer = rotary_emb.apply(&query_layer, seqlen_offset)?;
        let key_layer = rotary_emb.apply(&key_layer, seqlen_offset)?;

        // KV cache.
        let (key_layer, value_layer) = match &self.kv_cache {
            None => (key_layer, value_layer),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_layer], 0)?;
                let v = Tensor::cat(&[prev_v, &value_layer], 0)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key_layer.clone(), value_layer.clone()));

        // Repeat KV.
        let ratio =
            self.num_attention_heads_per_partition / self.num_multi_query_groups_per_partition;
        let key_layer = {
            let (d0, d1, d2, d3) = key_layer.dims4()?;
            key_layer
                .unsqueeze(D::Minus2)?
                .expand((d0, d1, d2, ratio, d3))?
                .reshape((
                    d0,
                    d1,
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ))?
        };
        let value_layer = {
            let (d0, d1, d2, d3) = value_layer.dims4()?;
            value_layer
                .unsqueeze(D::Minus2)?
                .expand((d0, d1, d2, ratio, d3))?
                .reshape((
                    d0,
                    d1,
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ))?
        };

        let context_layer =
            self.core_attention
                .forward(&query_layer, &key_layer, &value_layer, attention_mask)?;
        let output = context_layer.apply(&self.dense)?;
        Ok(output)
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
struct MLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense_h_to_4h = linear(
            cfg.hidden_size,
            cfg.ffn_hidden_size * 2,
            cfg.add_bias_linear,
            vb.pp("dense_h_to_4h"),
        )?;
        let dense_4h_to_h = linear(
            cfg.ffn_hidden_size,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense_4h_to_h"),
        )?;
        Ok(Self {
            dense_4h_to_h,
            dense_h_to_4h,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense_h_to_4h)?
            .apply(&candle_nn::Activation::Swiglu)?
            .apply(&self.dense_4h_to_h)
    }
}

#[derive(Debug, Clone)]
struct Block {
    input_layernorm: candle_nn::LayerNorm,
    self_attention: SelfAttention,
    post_attention_layernorm: candle_nn::LayerNorm,
    mlp: MLP,
    apply_residual_connection_post_layernorm: bool,
}

impl Block {
    fn new(layer_number: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = if cfg.rmsnorm {
            candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("input_layernorm"),
            )?
            .into_inner()
        } else {
            candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("input_layernorm"),
            )?
        };
        let post_attention_layernorm = if cfg.rmsnorm {
            candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?
            .into_inner()
        } else {
            candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?
        };
        let self_attention = SelfAttention::new(layer_number, cfg, vb.pp("self_attention"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            apply_residual_connection_post_layernorm: cfg.apply_residual_connection_post_layernorm,
        })
    }

    fn reset_kv_cache(&mut self) {
        self.self_attention.reset_kv_cache()
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: &Option<Tensor>,
        rotary_emb: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let layernorm_output = xs.apply(&self.input_layernorm)?;
        let attention_output =
            self.self_attention
                .forward(&layernorm_output, attention_mask, rotary_emb)?;
        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            xs
        };
        let layernorm_input = (residual + attention_output)?;
        let layernorm_output = layernorm_input.apply(&self.post_attention_layernorm)?;
        let mlp_output = layernorm_output.apply(&self.mlp)?;
        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &layernorm_input
        };
        mlp_output + residual
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    layers: Vec<Block>,
    final_layernorm: Option<candle_nn::LayerNorm>,
    rotary_emb: RotaryEmbedding,
}

impl Transformer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_index in 0..cfg.num_layers {
            let block = Block::new(layer_index + 1, cfg, vb_l.pp(layer_index))?;
            layers.push(block)
        }
        let final_layernorm = if cfg.post_layer_norm {
            let ln = if cfg.rmsnorm {
                candle_nn::rms_norm(
                    cfg.hidden_size,
                    cfg.layernorm_epsilon,
                    vb.pp("final_layernorm"),
                )?
                .into_inner()
            } else {
                candle_nn::layer_norm(
                    cfg.hidden_size,
                    cfg.layernorm_epsilon,
                    vb.pp("final_layernorm"),
                )?
            };
            Some(ln)
        } else {
            None
        };
        let rotary_emb = RotaryEmbedding::new(cfg, vb.dtype(), vb.device())?;
        Ok(Self {
            layers,
            final_layernorm,
            rotary_emb,
        })
    }

    fn reset_kv_cache(&mut self) {
        for block in self.layers.iter_mut() {
            block.reset_kv_cache()
        }
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: &Option<Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for block in self.layers.iter_mut() {
            xs = block.forward(&xs, attention_mask, &self.rotary_emb)?
        }
        match self.final_layernorm.as_ref() {
            None => Ok(xs),
            Some(ln) => xs.apply(ln),
        }
    }
}

#[derive(Debug, Clone)]
struct Embedding {
    word_embeddings: candle_nn::Embedding,
    fp32_residual_connection: bool,
}

impl Embedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.padded_vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        Ok(Self {
            word_embeddings,
            fp32_residual_connection: cfg.fp32_residual_connection,
        })
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.word_embeddings.forward(xs)?.transpose(0, 1)?; // b,s,h -> s,b,h
        if self.fp32_residual_connection {
            xs.to_dtype(candle::DType::F32)
        } else {
            xs.contiguous()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embedding: Embedding,
    encoder: Transformer,
    output_layer: Linear,
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("transformer");
        let embedding = Embedding::new(cfg, vb.pp("embedding"))?;
        let encoder = Transformer::new(cfg, vb.pp("encoder"))?;
        let output_layer = linear(
            cfg.hidden_size,
            cfg.padded_vocab_size,
            false,
            vb.pp("output_layer"),
        )?;
        Ok(Self {
            embedding,
            encoder,
            output_layer,
        })
    }

    pub fn reset_kv_cache(&mut self) {
        self.encoder.reset_kv_cache()
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, seq_len) = xs.dims2()?;
        let input_embeds = xs.apply(&self.embedding)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len, xs.device())?)
        };
        let xs = self.encoder.forward(&input_embeds, &attention_mask)?;
        let lm_logits = xs.i(seq_len - 1)?.apply(&self.output_layer)?;
        Ok(lm_logits)
    }
}
