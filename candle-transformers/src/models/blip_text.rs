//! Implementation of BLIP text encoder/decoder.
//!
//! - 📝 [Paper](https://arxiv.org/abs/2201.12086). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"
//!
//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/radames/Candle-BLIP-Image-Captioning)
//! - 💻 [GH Link](https://github.com/salesforce/BLIP)
//! - 🤗 [HF Link](https://huggingface.co/Salesforce/blip-image-captioning-base)
//! - 📝 [Paper](https://arxiv.org/abs/2201.12086)
//!
use super::with_tracing::{linear, Embedding, Linear};
use candle::{BackendStorage, Module, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub encoder_hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
    pub is_decoder: bool,
}

#[derive(Debug, Clone)]
struct TextEmbeddings<B: BackendStorage> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    position_ids: Tensor<B>,
}

impl<B: BackendStorage> TextEmbeddings<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let word_embeddings =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?;
        let position_embeddings = Embedding::new(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        let position_ids =
            Tensor::arange(0, cfg.max_position_embeddings as u32, vb.device())?.unsqueeze(0)?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
            position_ids,
        })
    }

    fn forward(&self, xs: &Tensor<B>, past_kv_len: usize) -> Result<Tensor<B>> {
        let seq_len = xs.dim(1)?;
        let position_ids = self.position_ids.narrow(1, past_kv_len, seq_len)?;
        let embeddings = self.word_embeddings.forward(xs)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        (embeddings + position_embeddings)?.apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextSelfAttention<B: BackendStorage> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    attention_head_size: usize,
    num_attention_heads: usize,
    attention_scale: f64,
    kv_cache: Option<(Tensor<B>, Tensor<B>)>,
}

impl<B: BackendStorage> TextSelfAttention<B> {
    fn new(cfg: &Config, is_cross_attention: bool, vb: VarBuilder<B>) -> Result<Self> {
        let num_attention_heads = cfg.num_attention_heads;
        let attention_head_size = cfg.hidden_size / num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        let query = linear(cfg.hidden_size, all_head_size, vb.pp("query"))?;
        let in_size = if is_cross_attention {
            cfg.encoder_hidden_size
        } else {
            cfg.hidden_size
        };
        let key = linear(in_size, all_head_size, vb.pp("key"))?;
        let value = linear(in_size, all_head_size, vb.pp("value"))?;
        let attention_scale = 1f64 / (attention_head_size as f64).sqrt();
        Ok(Self {
            query,
            key,
            value,
            attention_head_size,
            num_attention_heads,
            attention_scale,
            kv_cache: None,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let (b_size, seq_len, _) = xs.dims3()?;
        xs.reshape((
            b_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .permute((0, 2, 1, 3))
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None
    }

    fn forward(
        &mut self,
        xs: &Tensor<B>,
        encoder_hidden_states: Option<&Tensor<B>>,
        attention_mask: Option<&Tensor<B>>,
    ) -> Result<Tensor<B>> {
        let query = self
            .transpose_for_scores(&self.query.forward(xs)?)?
            .contiguous()?;
        let (key, value) = match encoder_hidden_states {
            None => {
                let key = self.transpose_for_scores(&self.key.forward(xs)?)?;
                let value = self.transpose_for_scores(&self.value.forward(xs)?)?;
                let (key, value) = match &self.kv_cache {
                    None => (key, value),
                    Some((prev_key, prev_value)) => {
                        let key = Tensor::cat(&[prev_key, &key], 2)?;
                        let value = Tensor::cat(&[prev_value, &value], 2)?;
                        (key, value)
                    }
                };
                self.kv_cache = Some((key.clone(), value.clone()));
                (key, value)
            }
            Some(xs) => {
                let key = self.transpose_for_scores(&self.key.forward(xs)?)?;
                let value = self.transpose_for_scores(&self.value.forward(xs)?)?;
                // no kv-cache in this case, but the results could probably be memoized.
                (key, value)
            }
        };
        let key = key.contiguous()?;
        let value = value.contiguous()?;
        let attention_scores = query.matmul(&key.t()?)?;
        let attention_scores = (attention_scores * self.attention_scale)?;
        let attention_scores = match attention_mask {
            Some(mask) => attention_scores.broadcast_add(mask)?,
            None => attention_scores,
        };
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        attention_probs
            .matmul(&value)?
            .permute((0, 2, 1, 3))?
            .flatten_from(D::Minus2)
    }
}

#[derive(Debug, Clone)]
struct TextSelfOutput<B: BackendStorage> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: BackendStorage> TextSelfOutput<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, xs: &Tensor<B>, input_tensor: &Tensor<B>) -> Result<Tensor<B>> {
        (xs.apply(&self.dense) + input_tensor)?.apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextAttention<B: BackendStorage> {
    self_: TextSelfAttention<B>,
    output: TextSelfOutput<B>,
}

impl<B: BackendStorage> TextAttention<B> {
    fn new(cfg: &Config, is_cross_attention: bool, vb: VarBuilder<B>) -> Result<Self> {
        let self_ = TextSelfAttention::new(cfg, is_cross_attention, vb.pp("self"))?;
        let output = TextSelfOutput::new(cfg, vb.pp("output"))?;
        Ok(Self { self_, output })
    }

    fn reset_kv_cache(&mut self) {
        self.self_.reset_kv_cache()
    }

    fn forward(
        &mut self,
        xs: &Tensor<B>,
        encoder_hidden_states: Option<&Tensor<B>>,
        attention_mask: Option<&Tensor<B>>,
    ) -> Result<Tensor<B>> {
        let self_outputs = self
            .self_
            .forward(xs, encoder_hidden_states, attention_mask)?;
        self.output.forward(&self_outputs, xs)
    }
}

#[derive(Debug, Clone)]
struct TextIntermediate<B: BackendStorage> {
    dense: Linear<B>,
    intermediate_act_fn: candle_nn::Activation,
}

impl<B: BackendStorage> TextIntermediate<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act_fn: cfg.hidden_act,
        })
    }
}

impl<B: BackendStorage> Module<B> for TextIntermediate<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.dense)?.apply(&self.intermediate_act_fn)
    }
}

#[derive(Debug, Clone)]
struct TextOutput<B: BackendStorage> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: BackendStorage> TextOutput<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, xs: &Tensor<B>, input_tensor: &Tensor<B>) -> Result<Tensor<B>> {
        (xs.apply(&self.dense)? + input_tensor)?.apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextLayer<B: BackendStorage> {
    attention: TextAttention<B>,
    cross_attention: Option<TextAttention<B>>,
    intermediate: TextIntermediate<B>,
    output: TextOutput<B>,
}

impl<B: BackendStorage> TextLayer<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let attention = TextAttention::new(cfg, false, vb.pp("attention"))?;
        let cross_attention = if cfg.is_decoder {
            Some(TextAttention::new(cfg, true, vb.pp("crossattention"))?)
        } else {
            None
        };
        let intermediate = TextIntermediate::new(cfg, vb.pp("intermediate"))?;
        let output = TextOutput::new(cfg, vb.pp("output"))?;
        Ok(Self {
            attention,
            cross_attention,
            intermediate,
            output,
        })
    }

    fn reset_kv_cache(&mut self) {
        self.attention.reset_kv_cache();
        if let Some(ca) = &mut self.cross_attention {
            ca.reset_kv_cache()
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor<B>,
        encoder_hidden_states: &Tensor<B>,
        attention_mask: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let attention_output = self.attention.forward(xs, None, Some(attention_mask))?;
        let attention_output = match &mut self.cross_attention {
            Some(ca) => ca.forward(&attention_output, Some(encoder_hidden_states), None)?,
            None => candle::bail!("expected some cross-attn"),
        };
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

#[derive(Debug, Clone)]
struct TextEncoder<B: BackendStorage> {
    layers: Vec<TextLayer<B>>,
}

impl<B: BackendStorage> TextEncoder<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let vb = vb.pp("layer");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = TextLayer::new(cfg, vb.pp(i))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn reset_kv_cache(&mut self) {
        self.layers.iter_mut().for_each(|l| l.reset_kv_cache())
    }

    fn forward(
        &mut self,
        xs: &Tensor<B>,
        encoder_hidden_states: &Tensor<B>,
        attention_mask: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, encoder_hidden_states, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct TextPooler<B: BackendStorage> {
    dense: Linear<B>,
}

impl<B: BackendStorage> TextPooler<B> {
    pub fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl<B: BackendStorage> Module<B> for TextPooler<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.narrow(D::Minus1, 0, 1)?
            .squeeze(D::Minus1)?
            .apply(&self.dense)?
            .tanh()
    }
}

#[derive(Debug, Clone)]
struct TextPredictionHeadTransform<B: BackendStorage> {
    dense: Linear<B>,
    transform_act_fn: candle_nn::Activation,
    layer_norm: LayerNorm<B>,
}

impl<B: BackendStorage> TextPredictionHeadTransform<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            dense,
            transform_act_fn: cfg.hidden_act,
            layer_norm,
        })
    }
}

impl<B: BackendStorage> Module<B> for TextPredictionHeadTransform<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.dense)?
            .apply(&self.transform_act_fn)?
            .apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextLMPredictionHead<B: BackendStorage> {
    transform: TextPredictionHeadTransform<B>,
    decoder: Linear<B>,
}

impl<B: BackendStorage> TextLMPredictionHead<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let transform = TextPredictionHeadTransform::new(cfg, vb.pp("transform"))?;
        let weight = vb.get((cfg.vocab_size, cfg.hidden_size), "decoder.weight")?;
        let bias = vb.get(cfg.vocab_size, "bias")?;
        let decoder = Linear::from_weights(weight, Some(bias));
        Ok(Self { transform, decoder })
    }
}

impl<B: BackendStorage> Module<B> for TextLMPredictionHead<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.transform)?.apply(&self.decoder)
    }
}

#[derive(Debug, Clone)]
struct TextOnlyMLMHead<B: BackendStorage> {
    predictions: TextLMPredictionHead<B>,
}

impl<B: BackendStorage> TextOnlyMLMHead<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let predictions = TextLMPredictionHead::new(cfg, vb.pp("predictions"))?;
        Ok(Self { predictions })
    }
}

impl<B: BackendStorage> Module<B> for TextOnlyMLMHead<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        self.predictions.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct TextModel<B: BackendStorage> {
    embeddings: TextEmbeddings<B>,
    encoder: TextEncoder<B>,
    past_kv_len: usize,
    // We do not need the pooler for caption generation
}

impl<B: BackendStorage> TextModel<B> {
    pub fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let embeddings = TextEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = TextEncoder::new(cfg, vb.pp("encoder"))?;
        Ok(Self {
            embeddings,
            encoder,
            past_kv_len: 0,
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor<B>,
        encoder_hidden_states: &Tensor<B>,
        attention_mask: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let embedding_output = self.embeddings.forward(input_ids, self.past_kv_len)?;
        let sequence_output =
            self.encoder
                .forward(&embedding_output, encoder_hidden_states, attention_mask)?;
        self.past_kv_len += seq_len;
        // We're interested in the sequence-output rather than the pooled-output.
        Ok(sequence_output)
    }

    fn reset_kv_cache(&mut self) {
        self.past_kv_len = 0;
        self.encoder.reset_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct TextLMHeadModel<B: BackendStorage> {
    bert: TextModel<B>,
    cls: TextOnlyMLMHead<B>,
}

impl<B: BackendStorage> TextLMHeadModel<B> {
    pub fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let bert = TextModel::new(cfg, vb.pp("bert"))?;
        let cls = TextOnlyMLMHead::new(cfg, vb.pp("cls"))?;
        Ok(Self { bert, cls })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor<B>,
        encoder_hidden_states: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let seq_len = input_ids.dim(1)?;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), input_ids.device())?;
        let sequence_output = self.bert.forward(input_ids, encoder_hidden_states, &mask)?;
        let prediction_scores = self.cls.forward(&sequence_output)?;
        // return_logits is false so we don't discard the last sequence element.
        Ok(prediction_scores)
    }

    pub fn reset_kv_cache(&mut self) {
        self.bert.reset_kv_cache()
    }
}
