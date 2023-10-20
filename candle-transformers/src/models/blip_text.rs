#![allow(unused)]
use super::with_tracing::{linear, linear_no_bias, Embedding, Linear};
use candle::{Module, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

#[derive(Debug, Clone)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    encoder_hidden_size: usize,
    intermediate_size: usize,
    projection_dim: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    max_position_embeddings: usize,
    hidden_act: candle_nn::Activation,
    layer_norm_eps: f64,
    is_decoder: bool,
}

#[derive(Debug, Clone)]
struct TextEmbeddings {
    word_embedddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    position_ids: Tensor,
}

impl TextEmbeddings {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let word_embedddings =
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
            word_embedddings,
            position_embeddings,
            layer_norm,
            position_ids,
        })
    }
}

#[derive(Debug, Clone)]
struct TextSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    all_head_size: usize,
    attention_head_size: usize,
    num_attention_heads: usize,
}

impl TextSelfAttention {
    fn new(cfg: &Config, is_cross_attention: bool, vb: VarBuilder) -> Result<Self> {
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
        Ok(Self {
            query,
            key,
            value,
            all_head_size,
            attention_head_size,
            num_attention_heads,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        xs.reshape((
            b_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .permute((0, 2, 1, 3))
    }
}

#[derive(Debug, Clone)]
struct TextSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl TextSelfOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.dense) + input_tensor)?.apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextAttention {
    self_: TextSelfAttention,
    output: TextSelfOutput,
}

impl TextAttention {
    fn new(cfg: &Config, is_cross_attention: bool, vb: VarBuilder) -> Result<Self> {
        let self_ = TextSelfAttention::new(cfg, is_cross_attention, vb.pp("self"))?;
        let output = TextSelfOutput::new(cfg, vb.pp("output"))?;
        Ok(Self { self_, output })
    }
}

#[derive(Debug, Clone)]
struct TextIntermediate {
    dense: Linear,
    intermediate_act_fn: candle_nn::Activation,
}

impl TextIntermediate {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act_fn: cfg.hidden_act,
        })
    }
}

impl Module for TextIntermediate {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)?.apply(&self.intermediate_act_fn)
    }
}

#[derive(Debug, Clone)]
struct TextOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl TextOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.dense)? + input_tensor)?.apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextLayer {
    attention: TextAttention,
    cross_attention: Option<TextAttention>,
    intermediate: TextIntermediate,
    output: TextOutput,
}

impl TextLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = TextAttention::new(cfg, false, vb.pp("attention"))?;
        let cross_attention = if cfg.is_decoder {
            Some(TextAttention::new(cfg, true, vb.pp("attention"))?)
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
}

impl Module for TextLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct TextEncoder {
    layers: Vec<TextLayer>,
}

impl TextEncoder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("layer");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = TextLayer::new(cfg, vb.pp(i))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }
}

impl Module for TextEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = xs.apply(layer)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct TextPooler {
    dense: Linear,
}

impl TextPooler {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl Module for TextPooler {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.narrow(D::Minus1, 0, 1)?
            .squeeze(D::Minus1)?
            .apply(&self.dense)?
            .tanh()
    }
}

#[derive(Debug, Clone)]
struct TextPredictionHeadTransform {
    dense: Linear,
    transform_act_fn: candle_nn::Activation,
    layer_norm: LayerNorm,
}

impl TextPredictionHeadTransform {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            dense,
            transform_act_fn: cfg.hidden_act,
            layer_norm,
        })
    }
}

impl Module for TextPredictionHeadTransform {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)?
            .apply(&self.transform_act_fn)?
            .apply(&self.layer_norm)
    }
}

#[derive(Debug, Clone)]
struct TextLMPredictionHead {
    transform: TextPredictionHeadTransform,
    decoder: Linear,
    bias: Tensor,
}

impl TextLMPredictionHead {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let transform = TextPredictionHeadTransform::new(cfg, vb.pp("transform"))?;
        let decoder = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("decoder"))?;
        let bias = vb.get(cfg.vocab_size, "bias")?;
        Ok(Self {
            transform,
            decoder,
            bias,
        })
    }
}

impl Module for TextLMPredictionHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.transform)?.apply(&self.decoder)
    }
}

#[derive(Debug, Clone)]
struct TextOnlyMLMHead {
    predictions: TextLMPredictionHead,
}

impl TextOnlyMLMHead {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let predictions = TextLMPredictionHead::new(cfg, vb.pp("predictions"))?;
        Ok(Self { predictions })
    }
}

impl Module for TextOnlyMLMHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.predictions.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct TextModel {
    embeddings: TextEmbeddings,
    encoder: TextEncoder,
    pooler: Option<TextPooler>,
}

#[derive(Debug, Clone)]
pub struct TextLMHeadModel {
    bert: TextModel,
    cls: TextOnlyMLMHead,
}
