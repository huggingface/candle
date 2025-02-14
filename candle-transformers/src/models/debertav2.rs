use std::collections::HashMap;

use candle::{bail, Context, DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, embedding, layer_norm, Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder,
};
use serde::{Deserialize, Deserializer};

pub const DTYPE: DType = DType::F32;

// NOTE: HiddenAct and HiddenActLayer are both direct copies from bert.rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

pub struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

pub type Id2Label = HashMap<u32, String>;
pub type Label2Id = HashMap<String, u32>;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub relative_attention: bool,
    pub max_relative_positions: isize,
    pub pad_token_id: Option<usize>,
    pub position_biased_input: bool,
    #[serde(deserialize_with = "deserialize_pos_att_type")]
    pub pos_att_type: Vec<String>,
    pub position_buckets: Option<isize>,
    pub share_att_key: Option<bool>,
    pub attention_head_size: Option<usize>,
    pub embedding_size: Option<usize>,
    pub norm_rel_ebd: Option<String>,
    pub conv_kernel_size: Option<usize>,
    pub conv_groups: Option<usize>,
    pub conv_act: Option<String>,
    pub id2label: Option<Id2Label>,
    pub label2id: Option<Label2Id>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<HiddenAct>,
    pub pooler_hidden_size: Option<usize>,
    pub cls_dropout: Option<f64>,
}

fn deserialize_pos_att_type<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize, Debug)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(s) => Ok(s.split('|').map(String::from).collect()),
        StringOrVec::Vec(v) => Ok(v),
    }
}

// NOTE: Dropout is probably not needed for now since this will primarily be used
// in inferencing. However, for training/fine-tuning it will be necessary.
pub struct StableDropout {
    _drop_prob: f64,
    _count: usize,
}

impl StableDropout {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            _drop_prob: drop_prob,
            _count: 0,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L823
pub struct DebertaV2Embeddings {
    device: Device,
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    dropout: StableDropout,
    position_ids: Tensor,
    config: Config,
    embedding_size: usize,
    embed_proj: Option<candle_nn::Linear>,
}

impl DebertaV2Embeddings {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let device = vb.device().clone();
        let config = config.clone();

        let embedding_size = config.embedding_size.unwrap_or(config.hidden_size);

        let word_embeddings =
            embedding(config.vocab_size, embedding_size, vb.pp("word_embeddings"))?;

        let position_embeddings = if config.position_biased_input {
            Some(embedding(
                config.max_position_embeddings,
                embedding_size,
                vb.pp("position_embeddings"),
            )?)
        } else {
            None
        };

        let token_type_embeddings: Option<Embedding> = if config.type_vocab_size > 0 {
            Some(candle_nn::embedding(
                config.type_vocab_size,
                config.hidden_size,
                vb.pp("token_type_embeddings"),
            )?)
        } else {
            None
        };

        let embed_proj: Option<candle_nn::Linear> = if embedding_size != config.hidden_size {
            Some(candle_nn::linear_no_bias(
                embedding_size,
                config.hidden_size,
                vb.pp("embed_proj"),
            )?)
        } else {
            None
        };

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = StableDropout::new(config.hidden_dropout_prob);

        let position_ids =
            Tensor::arange(0, config.max_position_embeddings as u32, &device)?.unsqueeze(0)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            position_ids,
            device,
            config,
            embedding_size,
            embed_proj,
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (input_shape, input_embeds) = match (input_ids, inputs_embeds) {
            (Some(ids), None) => {
                let embs = self.word_embeddings.forward(ids)?;
                (ids.dims(), embs)
            }
            (None, Some(e)) => (e.dims(), e.clone()),
            (None, None) => {
                bail!("Must specify either input_ids or inputs_embeds")
            }
            (Some(_), Some(_)) => {
                bail!("Can't specify both input_ids and inputs_embeds")
            }
        };

        let seq_length = match input_shape.last() {
            Some(v) => *v,
            None => bail!("DebertaV2Embeddings invalid input shape"),
        };

        let position_ids = match position_ids {
            Some(v) => v.clone(),
            None => self.position_ids.narrow(1, 0, seq_length)?,
        };

        let token_type_ids = match token_type_ids {
            Some(ids) => ids.clone(),
            None => Tensor::zeros(input_shape, DType::U32, &self.device)?,
        };

        let position_embeddings = match &self.position_embeddings {
            Some(emb) => emb.forward(&position_ids)?,
            None => Tensor::zeros_like(&input_embeds)?,
        };

        let mut embeddings = input_embeds;

        if self.config.position_biased_input {
            embeddings = embeddings.add(&position_embeddings)?;
        }

        if self.config.type_vocab_size > 0 {
            embeddings = self.token_type_embeddings.as_ref().map_or_else(
                || bail!("token_type_embeddings must be set when type_vocab_size > 0"),
                |token_type_embeddings| {
                    embeddings.add(&token_type_embeddings.forward(&token_type_ids)?)
                },
            )?;
        }

        if self.embedding_size != self.config.hidden_size {
            embeddings = if let Some(embed_proj) = &self.embed_proj {
                embed_proj.forward(&embeddings)?
            } else {
                bail!("embed_proj must exist if embedding_size != config.hidden_size");
            }
        }

        embeddings = self.layer_norm.forward(&embeddings)?;

        if let Some(mask) = mask {
            let mut mask = mask.clone();
            if mask.dims() != embeddings.dims() {
                if mask.dims().len() == 4 {
                    mask = mask.squeeze(1)?.squeeze(1)?;
                }
                mask = mask.unsqueeze(2)?;
            }

            mask = mask.to_dtype(embeddings.dtype())?;
            embeddings = embeddings.broadcast_mul(&mask)?;
        }

        self.dropout.forward(&embeddings)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L72
struct XSoftmax {}

impl XSoftmax {
    pub fn apply(input: &Tensor, mask: &Tensor, dim: D, device: &Device) -> Result<Tensor> {
        // NOTE: At the time of this writing, candle does not have a logical-not operator.
        let mut rmask = mask.broadcast_as(input.shape())?.to_dtype(DType::F32)?;

        rmask = rmask
            .broadcast_lt(&Tensor::new(&[1.0_f32], device)?)?
            .to_dtype(DType::U8)?;

        let min_value_tensor = Tensor::new(&[f32::MIN], device)?.broadcast_as(input.shape())?;
        let mut output = rmask.where_cond(&min_value_tensor, input)?;

        output = candle_nn::ops::softmax(&output, dim)?;

        let t_zeroes = Tensor::new(&[0f32], device)?.broadcast_as(input.shape())?;
        output = rmask.where_cond(&t_zeroes, &output)?;

        Ok(output)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L605
pub struct DebertaV2DisentangledSelfAttention {
    config: Config,
    num_attention_heads: usize,
    query_proj: candle_nn::Linear,
    key_proj: candle_nn::Linear,
    value_proj: candle_nn::Linear,
    dropout: StableDropout,
    device: Device,
    relative_attention: bool,
    pos_dropout: Option<StableDropout>,
    position_buckets: isize,
    max_relative_positions: isize,
    pos_ebd_size: isize,
    share_att_key: bool,
    pos_key_proj: Option<candle_nn::Linear>,
    pos_query_proj: Option<candle_nn::Linear>,
}

impl DebertaV2DisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let config = config.clone();
        let vb = vb.clone();

        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(candle::Error::Msg(format!(
                "The hidden size {} is not a multiple of the number of attention heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }

        let num_attention_heads = config.num_attention_heads;

        let attention_head_size = config
            .attention_head_size
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let all_head_size = num_attention_heads * attention_head_size;

        let query_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("query_proj"))?;
        let key_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("key_proj"))?;
        let value_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("value_proj"))?;

        let share_att_key = config.share_att_key.unwrap_or(false);
        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let mut pos_ebd_size: isize = 0;
        let position_buckets = config.position_buckets.unwrap_or(-1);
        let mut pos_dropout: Option<StableDropout> = None;
        let mut pos_key_proj: Option<candle_nn::Linear> = None;
        let mut pos_query_proj: Option<candle_nn::Linear> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }
            pos_ebd_size = max_relative_positions;
            if position_buckets > 0 {
                pos_ebd_size = position_buckets
            }

            pos_dropout = Some(StableDropout::new(config.hidden_dropout_prob));

            if !share_att_key {
                if config.pos_att_type.iter().any(|s| s == "c2p") {
                    pos_key_proj = Some(candle_nn::linear(
                        config.hidden_size,
                        all_head_size,
                        vb.pp("pos_key_proj"),
                    )?);
                }
                if config.pos_att_type.iter().any(|s| s == "p2c") {
                    pos_query_proj = Some(candle_nn::linear(
                        config.hidden_size,
                        all_head_size,
                        vb.pp("pos_query_proj"),
                    )?);
                }
            }
        }

        let dropout = StableDropout::new(config.attention_probs_dropout_prob);
        let device = vb.device().clone();

        Ok(Self {
            config,
            num_attention_heads,
            query_proj,
            key_proj,
            value_proj,
            dropout,
            device,
            relative_attention,
            pos_dropout,
            position_buckets,
            max_relative_positions,
            pos_ebd_size,
            share_att_key,
            pos_key_proj,
            pos_query_proj,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query_states = match query_states {
            Some(qs) => qs,
            None => hidden_states,
        };

        let query_layer = self.transpose_for_scores(&self.query_proj.forward(query_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key_proj.forward(query_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value_proj.forward(query_states)?)?;

        let mut rel_att: Option<Tensor> = None;

        let mut scale_factor: usize = 1;

        if self.config.pos_att_type.iter().any(|s| s == "c2p") {
            scale_factor += 1;
        }

        if self.config.pos_att_type.iter().any(|s| s == "p2c") {
            scale_factor += 1;
        }

        let scale = {
            let q_size = query_layer.dim(D::Minus1)?;
            Tensor::new(&[(q_size * scale_factor) as f32], &self.device)?.sqrt()?
        };

        let mut attention_scores: Tensor = {
            let key_layer_transposed = key_layer.t()?;
            let div = key_layer_transposed
                .broadcast_div(scale.to_dtype(query_layer.dtype())?.as_ref())?;
            query_layer.matmul(&div)?
        };

        if self.relative_attention {
            if let Some(rel_embeddings) = rel_embeddings {
                let rel_embeddings = self
                    .pos_dropout
                    .as_ref()
                    .context("relative_attention requires pos_dropout")?
                    .forward(rel_embeddings)?;
                rel_att = Some(self.disentangled_attention_bias(
                    query_layer,
                    key_layer,
                    relative_pos,
                    rel_embeddings,
                    scale_factor,
                )?);
            }
        }

        if let Some(rel_att) = rel_att {
            attention_scores = attention_scores.broadcast_add(&rel_att)?;
        }

        attention_scores = attention_scores.reshape((
            (),
            self.num_attention_heads,
            attention_scores.dim(D::Minus2)?,
            attention_scores.dim(D::Minus1)?,
        ))?;

        let mut attention_probs =
            XSoftmax::apply(&attention_scores, attention_mask, D::Minus1, &self.device)?;

        attention_probs = self.dropout.forward(&attention_probs)?;

        let mut context_layer = attention_probs
            .reshape((
                (),
                attention_probs.dim(D::Minus2)?,
                attention_probs.dim(D::Minus1)?,
            ))?
            .matmul(&value_layer)?;

        context_layer = context_layer
            .reshape((
                (),
                self.num_attention_heads,
                context_layer.dim(D::Minus2)?,
                context_layer.dim(D::Minus1)?,
            ))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let dims = context_layer.dims();

        context_layer = match dims.len() {
            2 => context_layer.reshape(())?,
            3 => context_layer.reshape((dims[0], ()))?,
            4 => context_layer.reshape((dims[0], dims[1], ()))?,
            5 => context_layer.reshape((dims[0], dims[1], dims[2], ()))?,
            _ => {
                bail!(
                    "Invalid shape for DisentabgledSelfAttention context layer: {:?}",
                    dims
                )
            }
        };

        Ok(context_layer)
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        match dims.len() {
            3 => {
                let reshaped = xs.reshape((dims[0], dims[1], self.num_attention_heads, ()))?;

                reshaped.transpose(1, 2)?.contiguous()?.reshape((
                    (),
                    reshaped.dim(1)?,
                    reshaped.dim(D::Minus1)?,
                ))
            }
            shape => {
                bail!("Invalid shape for transpose_for_scores. Expected 3 dimensions, got {shape}")
            }
        }
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: Tensor,
        key_layer: Tensor,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Tensor,
        scale_factor: usize,
    ) -> Result<Tensor> {
        let mut relative_pos = relative_pos.map_or(
            build_relative_position(
                query_layer.dim(D::Minus2)?,
                key_layer.dim(D::Minus2)?,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?,
            |pos| pos.clone(),
        );

        relative_pos = match relative_pos.dims().len() {
            2 => relative_pos.unsqueeze(0)?.unsqueeze(0)?,
            3 => relative_pos.unsqueeze(1)?,
            other => {
                bail!("Relative position ids must be of dim 2 or 3 or 4. Got dim of size {other}")
            }
        };

        let att_span = self.pos_ebd_size;

        let rel_embeddings = rel_embeddings
            .narrow(0, 0, (att_span * 2) as usize)?
            .unsqueeze(0)?;

        let mut pos_query_layer: Option<Tensor> = None;
        let mut pos_key_layer: Option<Tensor> = None;

        let repeat_with = query_layer.dim(0)? / self.num_attention_heads;
        if self.share_att_key {
            pos_query_layer = Some(
                self.transpose_for_scores(&self.query_proj.forward(&rel_embeddings)?)?
                    .repeat(repeat_with)?,
            );

            pos_key_layer = Some(
                self.transpose_for_scores(&self.key_proj.forward(&rel_embeddings)?)?
                    .repeat(repeat_with)?,
            )
        } else {
            if self.config.pos_att_type.iter().any(|s| s == "c2p") {
                pos_key_layer = Some(
                    self.transpose_for_scores(
                        &self
                            .pos_key_proj
                            .as_ref()
                            .context(
                                "Need pos_key_proj when share_att_key is false or not specified",
                            )?
                            .forward(&rel_embeddings)?,
                    )?
                    .repeat(repeat_with)?,
                )
            }
            if self.config.pos_att_type.iter().any(|s| s == "p2c") {
                pos_query_layer = Some(self.transpose_for_scores(&self
                    .pos_query_proj
                    .as_ref()
                    .context("Need a pos_query_proj when share_att_key is false or not specified")?
                    .forward(&rel_embeddings)?)?.repeat(repeat_with)?)
            }
        }

        let mut score = Tensor::new(&[0 as f32], &self.device)?;

        if self.config.pos_att_type.iter().any(|s| s == "c2p") {
            let pos_key_layer = pos_key_layer.context("c2p without pos_key_layer")?;

            let scale = Tensor::new(
                &[(pos_key_layer.dim(D::Minus1)? * scale_factor) as f32],
                &self.device,
            )?
            .sqrt()?;

            let mut c2p_att = query_layer.matmul(&pos_key_layer.t()?)?;

            let c2p_pos = relative_pos
                .broadcast_add(&Tensor::new(&[att_span as i64], &self.device)?)?
                .clamp(0 as f32, (att_span * 2 - 1) as f32)?;

            c2p_att = c2p_att.gather(
                &c2p_pos
                    .squeeze(0)?
                    .expand(&[
                        query_layer.dim(0)?,
                        query_layer.dim(1)?,
                        relative_pos.dim(D::Minus1)?,
                    ])?
                    .contiguous()?,
                D::Minus1,
            )?;

            score = score.broadcast_add(
                &c2p_att.broadcast_div(scale.to_dtype(c2p_att.dtype())?.as_ref())?,
            )?;
        }

        if self.config.pos_att_type.iter().any(|s| s == "p2c") {
            let pos_query_layer = pos_query_layer.context("p2c without pos_key_layer")?;

            let scale = Tensor::new(
                &[(pos_query_layer.dim(D::Minus1)? * scale_factor) as f32],
                &self.device,
            )?
            .sqrt()?;

            let r_pos = {
                if key_layer.dim(D::Minus2)? != query_layer.dim(D::Minus2)? {
                    build_relative_position(
                        key_layer.dim(D::Minus2)?,
                        key_layer.dim(D::Minus2)?,
                        &self.device,
                        Some(self.position_buckets),
                        Some(self.max_relative_positions),
                    )?
                    .unsqueeze(0)?
                } else {
                    relative_pos
                }
            };

            let p2c_pos = r_pos
                .to_dtype(DType::F32)?
                .neg()?
                .broadcast_add(&Tensor::new(&[att_span as f32], &self.device)?)?
                .clamp(0f32, (att_span * 2 - 1) as f32)?;

            let p2c_att = key_layer
                .matmul(&pos_query_layer.t()?)?
                .gather(
                    &p2c_pos
                        .squeeze(0)?
                        .expand(&[
                            query_layer.dim(0)?,
                            key_layer.dim(D::Minus2)?,
                            key_layer.dim(D::Minus2)?,
                        ])?
                        .contiguous()?
                        .to_dtype(DType::U32)?,
                    D::Minus1,
                )?
                .t()?;

            score =
                score.broadcast_add(&p2c_att.broadcast_div(&scale.to_dtype(p2c_att.dtype())?)?)?;
        }

        Ok(score)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L270
pub struct DebertaV2Attention {
    dsa: DebertaV2DisentangledSelfAttention,
    output: DebertaV2SelfOutput,
}

impl DebertaV2Attention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dsa = DebertaV2DisentangledSelfAttention::load(vb.pp("attention.self"), config)?;
        let output = DebertaV2SelfOutput::load(vb.pp("attention.output"), config)?;
        Ok(Self { dsa, output })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let self_output = self.dsa.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        self.output
            .forward(&self_output, query_states.unwrap_or(hidden_states))
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L255
pub struct DebertaV2SelfOutput {
    dense: candle_nn::Linear,
    layer_norm: LayerNorm,
    dropout: StableDropout,
}

impl DebertaV2SelfOutput {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = StableDropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.dense.forward(hidden_states)?;
        hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm
            .forward(&hidden_states.broadcast_add(input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L307
pub struct DebertaV2Intermediate {
    dense: candle_nn::Linear,
    intermediate_act: HiddenActLayer,
}

impl DebertaV2Intermediate {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("intermediate.dense"),
        )?;
        let intermediate_act = HiddenActLayer::new(config.hidden_act);
        Ok(Self {
            dense,
            intermediate_act,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.intermediate_act
            .forward(&self.dense.forward(hidden_states)?)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L323
pub struct DebertaV2Output {
    dense: candle_nn::Linear,
    layer_norm: LayerNorm,
    dropout: StableDropout,
}

impl DebertaV2Output {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("output.dense"),
        )?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("output.LayerNorm"),
        )?;
        let dropout = StableDropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.dense.forward(hidden_states)?;
        hidden_states = self.dropout.forward(&hidden_states)?;
        hidden_states = {
            let to_norm = hidden_states.broadcast_add(input_tensor)?;
            self.layer_norm.forward(&to_norm)?
        };
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L339
pub struct DebertaV2Layer {
    attention: DebertaV2Attention,
    intermediate: DebertaV2Intermediate,
    output: DebertaV2Output,
}

impl DebertaV2Layer {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = DebertaV2Attention::load(vb.clone(), config)?;
        let intermediate = DebertaV2Intermediate::load(vb.clone(), config)?;
        let output = DebertaV2Output::load(vb.clone(), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        let intermediate_output = self.intermediate.forward(&attention_output)?;

        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;

        Ok(layer_output)
    }
}

// TODO: In order to fully test ConvLayer a model needs to be found has a configuration where `conv_kernel_size` exists and is > 0
// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L373
pub struct ConvLayer {
    _conv_act: String,
    _conv: Conv1d,
    _layer_norm: LayerNorm,
    _dropout: StableDropout,
    _config: Config,
}

impl ConvLayer {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let config = config.clone();
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);
        let conv_act: String = config.conv_act.clone().unwrap_or("tanh".to_string());

        let conv_conf = Conv1dConfig {
            padding: (kernel_size - 1) / 2,
            groups,
            ..Default::default()
        };

        let conv = conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_conf,
            vb.pp("conv"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = StableDropout::new(config.hidden_dropout_prob);

        Ok(Self {
            _conv_act: conv_act,
            _conv: conv,
            _layer_norm: layer_norm,
            _dropout: dropout,
            _config: config,
        })
    }

    pub fn forward(
        &self,
        _hidden_states: &Tensor,
        _residual_states: &Tensor,
        _input_mask: &Tensor,
    ) -> Result<Tensor> {
        todo!("Need a model that contains a conv layer to test against.")
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L409
pub struct DebertaV2Encoder {
    layer: Vec<DebertaV2Layer>,
    relative_attention: bool,
    max_relative_positions: isize,
    position_buckets: isize,
    rel_embeddings: Option<Embedding>,
    norm_rel_ebd: String,
    layer_norm: Option<LayerNorm>,
    conv: Option<ConvLayer>,
    device: Device,
}

impl DebertaV2Encoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layer = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let position_buckets = config.position_buckets.unwrap_or(-1);

        let mut rel_embeddings: Option<Embedding> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }

            let mut pos_ebd_size = max_relative_positions * 2;

            if position_buckets > 0 {
                pos_ebd_size = position_buckets * 2;
            }

            rel_embeddings = Some(embedding(
                pos_ebd_size as usize,
                config.hidden_size,
                vb.pp("rel_embeddings"),
            )?);
        }

        // NOTE: The Python code assumes that the config attribute "norm_rel_ebd" is an array of some kind, but most examples have it as a string.
        // So it might need to be updated at some point.
        let norm_rel_ebd = match config.norm_rel_ebd.as_ref() {
            Some(nre) => nre.trim().to_string(),
            None => "none".to_string(),
        };

        let layer_norm: Option<LayerNorm> = if norm_rel_ebd == "layer_norm" {
            Some(layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?)
        } else {
            None
        };

        let conv: Option<ConvLayer> = if config.conv_kernel_size.unwrap_or(0) > 0 {
            Some(ConvLayer::load(vb.pp("conv"), config)?)
        } else {
            None
        };

        Ok(Self {
            layer,
            relative_attention,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            norm_rel_ebd,
            layer_norm,
            conv,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_mask = if attention_mask.dims().len() <= 2 {
            attention_mask.clone()
        } else {
            attention_mask
                .sum_keepdim(attention_mask.rank() - 2)?
                .gt(0.)?
        };

        let attention_mask = self.get_attention_mask(attention_mask.clone())?;

        let relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)?;

        let mut next_kv: Tensor = hidden_states.clone();
        let rel_embeddings = self.get_rel_embedding()?;
        let mut output_states = next_kv.to_owned();
        let mut query_states: Option<Tensor> = query_states.cloned();

        for (i, layer_module) in self.layer.iter().enumerate() {
            // NOTE: The original python code branches here if this model is being
            // used for training vs. inferencing. For now, we will only handle the
            // inferencing side of things

            output_states = layer_module.forward(
                next_kv.as_ref(),
                &attention_mask,
                query_states.as_ref(),
                relative_pos.as_ref(),
                rel_embeddings.as_ref(),
            )?;

            if i == 0 {
                if let Some(conv) = &self.conv {
                    output_states = conv.forward(hidden_states, &output_states, &input_mask)?;
                }
            }

            if query_states.is_some() {
                query_states = Some(output_states.clone());
            } else {
                next_kv = output_states.clone();
            }
        }

        Ok(output_states)
    }

    fn get_attention_mask(&self, mut attention_mask: Tensor) -> Result<Tensor> {
        match attention_mask.dims().len() {
            0..=2 => {
                let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
                attention_mask = extended_attention_mask.broadcast_mul(
                    &extended_attention_mask
                        .squeeze(D::Minus2)?
                        .unsqueeze(D::Minus1)?,
                )?;
            }
            3 => attention_mask = attention_mask.unsqueeze(1)?,
            len => bail!("Unsupported attentiom mask size length: {len}"),
        }

        Ok(attention_mask)
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if self.relative_attention && relative_pos.is_none() {
            let q = if let Some(query_states) = query_states {
                query_states.dim(D::Minus2)?
            } else {
                hidden_states.dim(D::Minus2)?
            };

            return Ok(Some(build_relative_position(
                q,
                hidden_states.dim(D::Minus2)?,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?));
        }

        if relative_pos.is_some() {
            Ok(relative_pos.cloned())
        } else {
            Ok(None)
        }
    }
    fn get_rel_embedding(&self) -> Result<Option<Tensor>> {
        if !self.relative_attention {
            return Ok(None);
        }

        let rel_embeddings = self
            .rel_embeddings
            .as_ref()
            .context("self.rel_embeddings not present when using relative_attention")?
            .embeddings()
            .clone();

        if !self.norm_rel_ebd.contains("layer_norm") {
            return Ok(Some(rel_embeddings));
        }

        let layer_normed_embeddings = self
            .layer_norm
            .as_ref()
            .context("DebertaV2Encoder layer_norm is None when norm_rel_ebd contains layer_norm")?
            .forward(&rel_embeddings)?;

        Ok(Some(layer_normed_embeddings))
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L991
pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    z_steps: usize,
    pub device: Device,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let vb = vb.clone();
        let embeddings = DebertaV2Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV2Encoder::load(vb.pp("encoder"), config)?;
        let z_steps: usize = 0;

        Ok(Self {
            embeddings,
            encoder,
            z_steps,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let input_ids_shape = input_ids.shape();

        let attention_mask = match attention_mask {
            Some(mask) => mask,
            None => Tensor::ones(input_ids_shape, DType::I64, &self.device)?,
        };

        let token_type_ids = match token_type_ids {
            Some(ids) => ids,
            None => Tensor::zeros(input_ids_shape, DType::U32, &self.device)?,
        };

        let embedding_output = self.embeddings.forward(
            Some(input_ids),
            Some(&token_type_ids),
            None,
            Some(&attention_mask),
            None,
        )?;

        let encoder_output =
            self.encoder
                .forward(&embedding_output, &attention_mask, None, None)?;

        if self.z_steps > 1 {
            todo!("Complete DebertaV2Model forward() when z_steps > 1 -- Needs a model to test this situation.")
        }

        Ok(encoder_output)
    }
}

#[derive(Debug)]
pub struct NERItem {
    pub entity: String,
    pub word: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
    pub index: usize,
}

#[derive(Debug)]
pub struct TextClassificationItem {
    pub label: String,
    pub score: f32,
}

pub struct DebertaV2NERModel {
    pub device: Device,
    deberta: DebertaV2Model,
    dropout: candle_nn::Dropout,
    classifier: candle_nn::Linear,
}

fn id2label_len(config: &Config, id2label: Option<HashMap<u32, String>>) -> Result<usize> {
    let id2label_len = match (&config.id2label, id2label) {
        (None, None) => bail!("Id2Label is either not present in the model configuration or not passed into DebertaV2NERModel::load as a parameter"),
        (None, Some(id2label_p)) => id2label_p.len(),
        (Some(id2label_c), None) => id2label_c.len(),
        (Some(id2label_c), Some(id2label_p)) => {
          if *id2label_c == id2label_p {
            id2label_c.len()
          } else {
            bail!("Id2Label is both present in the model configuration and provided as a parameter, and they are different.")
          }
        }
    };
    Ok(id2label_len)
}

impl DebertaV2NERModel {
    pub fn load(vb: VarBuilder, config: &Config, id2label: Option<Id2Label>) -> Result<Self> {
        let id2label_len = id2label_len(config, id2label)?;

        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let dropout = candle_nn::Dropout::new(config.hidden_dropout_prob as f32);
        let classifier: candle_nn::Linear = candle_nn::linear_no_bias(
            config.hidden_size,
            id2label_len,
            vb.root().pp("classifier"),
        )?;

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            dropout,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let output = self
            .deberta
            .forward(input_ids, token_type_ids, attention_mask)?;
        let output = self.dropout.forward(&output, false)?;
        self.classifier.forward(&output)
    }
}

pub struct DebertaV2SeqClassificationModel {
    pub device: Device,
    deberta: DebertaV2Model,
    dropout: StableDropout,
    pooler: DebertaV2ContextPooler,
    classifier: candle_nn::Linear,
}

impl DebertaV2SeqClassificationModel {
    pub fn load(vb: VarBuilder, config: &Config, id2label: Option<Id2Label>) -> Result<Self> {
        let id2label_len = id2label_len(config, id2label)?;
        let deberta = DebertaV2Model::load(vb.clone(), config)?;
        let pooler = DebertaV2ContextPooler::load(vb.clone(), config)?;
        let output_dim = pooler.output_dim()?;
        let classifier = candle_nn::linear(output_dim, id2label_len, vb.root().pp("classifier"))?;
        let dropout = match config.cls_dropout {
            Some(cls_dropout) => StableDropout::new(cls_dropout),
            None => StableDropout::new(config.hidden_dropout_prob),
        };

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            dropout,
            pooler,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let encoder_layer = self
            .deberta
            .forward(input_ids, token_type_ids, attention_mask)?;
        let pooled_output = self.pooler.forward(&encoder_layer)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }
}

pub struct DebertaV2ContextPooler {
    dense: candle_nn::Linear,
    dropout: StableDropout,
    config: Config,
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L49
impl DebertaV2ContextPooler {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let pooler_hidden_size = config
            .pooler_hidden_size
            .context("config.pooler_hidden_size is required for DebertaV2ContextPooler")?;

        let pooler_dropout = config
            .pooler_dropout
            .context("config.pooler_dropout is required for DebertaV2ContextPooler")?;

        let dense = candle_nn::linear(
            pooler_hidden_size,
            pooler_hidden_size,
            vb.root().pp("pooler.dense"),
        )?;

        let dropout = StableDropout::new(pooler_dropout);

        Ok(Self {
            dense,
            dropout,
            config: config.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let context_token = hidden_states.narrow(1, 0, 1)?.squeeze(1)?;
        let context_token = self.dropout.forward(&context_token)?;

        let pooled_output = self.dense.forward(&context_token.contiguous()?)?;
        let pooler_hidden_act = self
            .config
            .pooler_hidden_act
            .context("Could not obtain pooler hidden act from config")?;

        HiddenActLayer::new(pooler_hidden_act).forward(&pooled_output)
    }

    pub fn output_dim(&self) -> Result<usize> {
        self.config.pooler_hidden_size.context("DebertaV2ContextPooler cannot return output_dim (pooler_hidden_size) since it is not specified in the model config")
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L557
pub(crate) fn build_relative_position(
    query_size: usize,
    key_size: usize,
    device: &Device,
    bucket_size: Option<isize>,
    max_position: Option<isize>,
) -> Result<Tensor> {
    let q_ids = Tensor::arange(0, query_size as i64, device)?.unsqueeze(0)?;
    let k_ids: Tensor = Tensor::arange(0, key_size as i64, device)?.unsqueeze(D::Minus1)?;
    let mut rel_pos_ids = k_ids.broadcast_sub(&q_ids)?;
    let bucket_size = bucket_size.unwrap_or(-1);
    let max_position = max_position.unwrap_or(-1);

    if bucket_size > 0 && max_position > 0 {
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position, device)?;
    }

    rel_pos_ids = rel_pos_ids.to_dtype(DType::I64)?;
    rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;
    rel_pos_ids.unsqueeze(0)
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L542
pub(crate) fn make_log_bucket_position(
    relative_pos: Tensor,
    bucket_size: isize,
    max_position: isize,
    device: &Device,
) -> Result<Tensor> {
    let sign = relative_pos.to_dtype(DType::F32)?.sign()?;

    let mid = bucket_size / 2;

    let lt_mid = relative_pos.lt(mid as i64)?;
    let gt_neg_mid = relative_pos.gt(-mid as i64)?;

    let condition = lt_mid
        .to_dtype(candle::DType::F32)?
        .mul(&gt_neg_mid.to_dtype(candle::DType::F32)?)?
        .to_dtype(DType::U8)?;

    let on_true = Tensor::new(&[(mid - 1) as u32], device)?
        .broadcast_as(relative_pos.shape())?
        .to_dtype(relative_pos.dtype())?;

    let on_false = relative_pos
        .to_dtype(DType::F32)?
        .abs()?
        .to_dtype(DType::I64)?;

    let abs_pos = condition.where_cond(&on_true, &on_false)?;

    let mid_as_tensor = Tensor::from_slice(&[mid as f32], (1,), device)?;

    let log_pos = {
        let first_log = abs_pos
            .to_dtype(DType::F32)?
            .broadcast_div(&mid_as_tensor)?
            .log()?;

        let second_log =
            Tensor::from_slice(&[((max_position as f32 - 1.0) / mid as f32)], (1,), device)?
                .log()?;

        let first_div_second = first_log.broadcast_div(&second_log)?;

        let to_ceil = first_div_second
            .broadcast_mul(Tensor::from_slice(&[(mid - 1) as f32], (1,), device)?.as_ref())?;

        let ceil = to_ceil.ceil()?;

        ceil.broadcast_add(&mid_as_tensor)?
    };

    Ok({
        let abs_pos_lte_mid = abs_pos.to_dtype(DType::F32)?.broadcast_le(&mid_as_tensor)?;
        let relative_pos = relative_pos.to_dtype(relative_pos.dtype())?;
        let log_pos_mul_sign = log_pos.broadcast_mul(&sign.to_dtype(DType::F32)?)?;
        abs_pos_lte_mid.where_cond(&relative_pos.to_dtype(DType::F32)?, &log_pos_mul_sign)?
    })
}
