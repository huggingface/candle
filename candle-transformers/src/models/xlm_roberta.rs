use crate::models::with_tracing::{linear, Linear};
use candle::{tensor::TryToDevice, BackendStorage, CpuStorage, DType, Module, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, ops::softmax_last_dim, Activation, Embedding, LayerNorm, VarBuilder,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub layer_norm_eps: f64,
    pub attention_probs_dropout_prob: f32,
    pub hidden_dropout_prob: f32,
    pub num_attention_heads: usize,
    pub position_embedding_type: String,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub pad_token_id: u32,
}

struct XLMRobertaEmbeddings<B: BackendStorage> {
    word_embeddings: Embedding<B>,
    position_embeddings: Option<Embedding<B>>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    padding_idx: u32,
    span: tracing::Span,
}

impl<B: BackendStorage> XLMRobertaEmbeddings<B> {
    fn load(vb: VarBuilder<B>, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            padding_idx: config.pad_token_id,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor<B>, token_type_ids: &Tensor<B>) -> Result<Tensor<B>>
    where
        Tensor<B>: TryToDevice<CpuStorage, B>,
    {
        let _enter = self.span.enter();
        let (_bsize, _) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            let mask = input_ids
                .ne(self.padding_idx)?
                .to_dtype(input_embeddings.dtype())?;
            let cumsum = mask.cumsum(1)?;
            let position_ids = (cumsum * mask)?
                .broadcast_add(
                    &Tensor::try_from(self.padding_idx)?
                        .to_dtype(input_embeddings.dtype())?
                        .to_device::<B>(input_embeddings.device())?,
                )?
                .to_dtype(candle::DType::U32)?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?;
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        Ok(embeddings)
    }
}

struct XLMRobertaSelfAttention<B: BackendStorage> {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
}

impl<B: BackendStorage> XLMRobertaSelfAttention<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        Ok(Self {
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
            all_head_size,
            query: linear(cfg.hidden_size, all_head_size, vb.pp("query"))?,
            key: linear(cfg.hidden_size, all_head_size, vb.pp("key"))?,
            value: linear(cfg.hidden_size, all_head_size, vb.pp("value"))?,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let mut new_x_shape = x.dims().to_vec();
        new_x_shape[2] = self.num_attention_heads;
        new_x_shape.push(self.attention_head_size);
        let x = x.reshape(new_x_shape)?;
        x.permute((0, 2, 1, 3))?.contiguous()
    }

    fn forward(
        &self,
        hidden_states: &Tensor<B>,
        encoder_hidden_states: Option<&Tensor<B>>,
        attention_mask: &Tensor<B>,
        past_key_value: Option<(&Tensor<B>, &Tensor<B>)>,
        encoder_attention_mask: Option<&Tensor<B>>,
    ) -> Result<Tensor<B>> {
        let mixed_query_layer = self.query.forward(hidden_states)?;
        let is_cross_attention = encoder_hidden_states.is_some();
        let (key_layer, value_layer, attention_mask) = if is_cross_attention {
            if let Some((past_key, past_value)) = past_key_value {
                let key_layer = past_key.clone();
                let value_layer = past_value.clone();
                let attention_mask = encoder_attention_mask.unwrap().clone();
                (key_layer, value_layer, Some(attention_mask))
            } else {
                let key_layer =
                    self.transpose_for_scores(&self.key.forward(encoder_hidden_states.unwrap())?)?;
                let value_layer = self
                    .transpose_for_scores(&self.value.forward(encoder_hidden_states.unwrap())?)?;
                let attention_mask = encoder_attention_mask.unwrap();
                (key_layer, value_layer, Some(attention_mask.clone()))
            }
        } else if let Some((past_key, past_value)) = past_key_value {
            let mut key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
            let mut value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
            key_layer = Tensor::cat(&[past_key.clone(), key_layer], 2)?;
            value_layer = Tensor::cat(&[past_value.clone(), value_layer], 2)?;
            (key_layer, value_layer, Some(attention_mask.clone()))
        } else {
            let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
            let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
            (key_layer, value_layer, Some(attention_mask.clone()))
        };

        let query_layer = self.transpose_for_scores(&mixed_query_layer)?;
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        let scale = 1f64 / f64::sqrt(self.attention_head_size as f64);

        attention_scores = (attention_scores * scale)?;
        attention_scores = match attention_mask {
            None => attention_scores,
            Some(mask) => {
                attention_scores.broadcast_add(&mask.to_dtype(attention_scores.dtype())?)?
            }
        };
        let attention_probs = softmax_last_dim(&attention_scores)?;

        let context_layer = attention_probs
            .matmul(&value_layer)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let mut new_context_layer_shape =
            context_layer.dims()[..context_layer.dims().len() - 2].to_vec();
        new_context_layer_shape.push(self.all_head_size);
        let context_layer = context_layer.reshape(new_context_layer_shape)?;

        Ok(context_layer)
    }
}

struct XLMRobertaSelfOutput<B: BackendStorage> {
    dense: Linear<B>,
    layernorm: LayerNorm<B>,
}

impl<B: BackendStorage> XLMRobertaSelfOutput<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layernorm })
    }

    fn forward(&self, hidden_states: &Tensor<B>, input_tensor: &Tensor<B>) -> Result<Tensor<B>> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.layernorm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

struct XLMRobertaAttention<B: BackendStorage> {
    output: XLMRobertaSelfOutput<B>,
    self_attention: XLMRobertaSelfAttention<B>,
}

impl<B: BackendStorage> XLMRobertaAttention<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let output = XLMRobertaSelfOutput::new(cfg, vb.pp("output"))?;
        let self_attention = XLMRobertaSelfAttention::new(cfg, vb.pp("self"))?;
        Ok(Self {
            output,
            self_attention,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor<B>,
        attention_mask: &Tensor<B>,
        encoder_hidden_states: Option<&Tensor<B>>,
        encoder_attention_mask: Option<&Tensor<B>>,
        past_key_value: Option<(&Tensor<B>, &Tensor<B>)>,
    ) -> Result<(Tensor<B>, Tensor<B>)> {
        let self_outputs = self.self_attention.forward(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            past_key_value,
            encoder_attention_mask,
        )?;
        let attention_output = self.output.forward(&self_outputs, hidden_states)?;
        Ok((attention_output, self_outputs))
    }
}

struct XLMRobertaOutput<B: BackendStorage> {
    dense: Linear<B>,
    layernorm: LayerNorm<B>,
}

impl<B: BackendStorage> XLMRobertaOutput<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layernorm })
    }

    fn forward(&self, hidden_states: &Tensor<B>, input_tensor: &Tensor<B>) -> Result<Tensor<B>> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.layernorm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

struct XLMRobertaIntermediate<B: BackendStorage> {
    dense: Linear<B>,
    intermediate_act_fn: Activation,
}

impl<B: BackendStorage> XLMRobertaIntermediate<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        let intermediate_act_fn = cfg.hidden_act;
        Ok(Self {
            dense,
            intermediate_act_fn,
        })
    }

    fn forward(&self, hidden_states: &Tensor<B>) -> Result<Tensor<B>> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.intermediate_act_fn.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

struct XLMRobertaLayer<B: BackendStorage> {
    attention: XLMRobertaAttention<B>,
    intermediate: XLMRobertaIntermediate<B>,
    output: XLMRobertaOutput<B>,
}

impl<B: BackendStorage> XLMRobertaLayer<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let attention = XLMRobertaAttention::new(cfg, vb.pp("attention"))?;
        let intermediate = XLMRobertaIntermediate::new(cfg, vb.pp("intermediate"))?;
        let output = XLMRobertaOutput::new(cfg, vb.pp("output"))?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor<B>,
        attention_mask: &Tensor<B>,
        encoder_hidden_states: Option<&Tensor<B>>,
        encoder_attention_mask: Option<&Tensor<B>>,
        past_key_value: Option<(&Tensor<B>, &Tensor<B>)>,
    ) -> Result<(Tensor<B>, Tensor<B>)> {
        let self_attention_outputs = self.attention.forward(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
        )?;
        let attention_output = self_attention_outputs.0;
        let outputs = self_attention_outputs.1;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok((layer_output, outputs))
    }
}

struct XLMRobertaEncoder<B: BackendStorage> {
    layers: Vec<XLMRobertaLayer<B>>,
}

impl<B: BackendStorage> XLMRobertaEncoder<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| XLMRobertaLayer::new(cfg, vb.pp(format!("layer.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor<B>,
        attention_mask: &Tensor<B>,
        encoder_hidden_states: Option<&Tensor<B>>,
        encoder_attention_mask: Option<&Tensor<B>>,
        past_key_value: Option<(&Tensor<B>, &Tensor<B>)>,
    ) -> Result<Tensor<B>> {
        let mut hidden_states = hidden_states.clone();
        for layer_module in self.layers.iter() {
            let layer_outputs = layer_module.forward(
                &hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
            )?;
            hidden_states = layer_outputs.0;
        }
        Ok(hidden_states)
    }
}

pub struct XLMRobertaModel<B: BackendStorage> {
    encoder: XLMRobertaEncoder<B>,
    embeddings: XLMRobertaEmbeddings<B>,
}

impl<B: BackendStorage> XLMRobertaModel<B> {
    pub fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let encoder = XLMRobertaEncoder::new(cfg, vb.pp("encoder"))?;
        let embeddings = XLMRobertaEmbeddings::load(vb.pp("embeddings"), cfg)?;
        Ok(Self {
            encoder,
            embeddings,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor<B>,
        attention_mask: &Tensor<B>,
        token_type_ids: &Tensor<B>,
        past_key_value: Option<(&Tensor<B>, &Tensor<B>)>,
        encoder_hidden_states: Option<&Tensor<B>>,
        encoder_attention_mask: Option<&Tensor<B>>,
    ) -> Result<Tensor<B>> {
        let hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = prepare_4d_attention_mask(attention_mask, DType::F32, None)?
            .to_device(hidden_states.device())?;
        let hidden_states = self.encoder.forward(
            &hidden_states,
            &attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
        )?;
        Ok(hidden_states)
    }
}

struct XLMRobertaLMHead<B: BackendStorage> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: BackendStorage> XLMRobertaLMHead<B> {
    fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(
        &self,
        hidden_states: &Tensor<B>,
        shared_embeddings: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = candle_nn::Activation::Gelu.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.broadcast_matmul(shared_embeddings)?;
        Ok(hidden_states)
    }
}

pub struct XLMRobertaForMaskedLM<B: BackendStorage> {
    roberta: XLMRobertaModel<B>,
    lm_head: XLMRobertaLMHead<B>,
}

impl<B: BackendStorage> XLMRobertaForMaskedLM<B> {
    pub fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let roberta = XLMRobertaModel::new(cfg, vb.pp("roberta"))?;
        let lm_head = XLMRobertaLMHead::new(cfg, vb.pp("lm_head"))?;
        Ok(Self { roberta, lm_head })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor<B>,
        attention_mask: &Tensor<B>,
        token_type_ids: &Tensor<B>,
        past_key_value: Option<(&Tensor<B>, &Tensor<B>)>,
        encoder_hidden_states: Option<&Tensor<B>>,
        encoder_attention_mask: Option<&Tensor<B>>,
    ) -> Result<Tensor> {
        let hidden_states = self.roberta.forward(
            input_ids,
            attention_mask,
            token_type_ids,
            past_key_value,
            encoder_hidden_states,
            encoder_attention_mask,
        )?;
        let lm_logits = self.lm_head.forward(
            &hidden_states,
            &self
                .roberta
                .embeddings
                .word_embeddings
                .embeddings()
                .t()?
                .unsqueeze(0)?,
        )?;
        Ok(lm_logits)
    }
}

struct XLMRobertaClassificationHead<B: BackendStorage> {
    dense: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: BackendStorage> XLMRobertaClassificationHead<B> {
    fn new(num_labels: usize, cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let out_proj = linear(cfg.hidden_size, num_labels, vb.pp("out_proj"))?;
        Ok(Self { dense, out_proj })
    }

    fn forward(&self, hidden_states: &Tensor<B>) -> Result<Tensor<B>> {
        let cls_states = hidden_states.get_on_dim(1, 0)?.contiguous()?;
        let hidden_states = self.dense.forward(&cls_states)?;
        // The activation used in the classification head is tanh, as per the original
        // implementation.
        // https://github.com/huggingface/transformers/blob/6e3063422c4b1c014aa60c32b9254fd2902f0f28/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1454
        let hidden_states = self.out_proj.forward(&hidden_states.tanh()?)?;
        Ok(hidden_states)
    }
}

pub struct XLMRobertaForSequenceClassification<B: BackendStorage> {
    roberta: XLMRobertaModel<B>,
    classifier: XLMRobertaClassificationHead<B>,
}

impl<B: BackendStorage> XLMRobertaForSequenceClassification<B> {
    pub fn new(num_labels: usize, cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let roberta = XLMRobertaModel::new(cfg, vb.pp("roberta"))?;
        let classifier = XLMRobertaClassificationHead::new(num_labels, cfg, vb.pp("classifier"))?;
        Ok(Self {
            roberta,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor<B>,
        attention_mask: &Tensor<B>,
        token_type_ids: &Tensor<B>,
    ) -> Result<Tensor<B>> {
        let hidden_states =
            self.roberta
                .forward(input_ids, attention_mask, token_type_ids, None, None, None)?;
        self.classifier.forward(&hidden_states)
    }
}

fn prepare_4d_attention_mask<B: BackendStorage>(
    mask: &Tensor<B>,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor<B>> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?
        .to_dtype(dtype)?;

    let inverted_mask = (1.0 - expanded_mask)?;

    (inverted_mask * get_dtype_min_val(dtype))?.to_dtype(dtype)
}

fn get_dtype_min_val(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::MIN as f64,
        DType::F64 => f64::MIN,
        _ => panic!("Unsupported data type"),
    }
}
