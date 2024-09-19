use candle::{DType, Device, IndexOp, Module, Result, Tensor};

use crate::models::{metavoice::tokenizers, stable_diffusion::embeddings};

use super::Activation;

/// Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
/// positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
/// [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
/// For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
/// with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
#[derive(Clone, Debug)]
pub enum PositionEmbeddingType {
    // TODO: 2024/09/19 11:04:35 使用 serde 转成下划线命名
    Absolute,
    RelativeKey,
    RelativeKeyQuery,
}

#[derive(Clone, Debug)]
pub struct ChineseClipTextConfig {
    //  vocab_size=30522,
    //     hidden_size=768,
    //     num_hidden_layers=12,
    //     num_attention_heads=12,
    //     intermediate_size=3072,
    //     hidden_act="gelu",
    //     hidden_dropout_prob=0.1,
    //     attention_probs_dropout_prob=0.1,
    //     max_position_embeddings=512,
    //     type_vocab_size=2,
    //     initializer_range=0.02,
    //     initializer_factor=1.0,
    //     layer_norm_eps=1e-12,
    //     pad_token_id=0,
    //     position_embedding_type="absolute",
    //     use_cache=True,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub initializer_factor: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: usize,
    pub position_embedding_type: PositionEmbeddingType,
    pub use_cache: bool,
}

impl Default for ChineseClipTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
        }
    }
}

impl ChineseClipTextConfig {
    // "architectures": [
    //   "ChineseCLIPTextModel"
    // ],
    // "attention_probs_dropout_prob": 0.1,
    // "bos_token_id": 0,
    // "directionality": "bidi",
    // "eos_token_id": 2,
    // "hidden_act": "gelu",
    // "hidden_dropout_prob": 0.1,
    // "hidden_size": 768,
    // "initializer_range": 0.02,
    // "intermediate_size": 3072,
    // "layer_norm_eps": 1e-12,
    // "max_position_embeddings": 512,
    // "model_type": "chinese_clip_text_model",
    // "num_attention_heads": 12,
    // "num_hidden_layers": 12,
    // "output_past": true,
    // "pad_token_id": 0,
    // "pooler_fc_size": 768,
    // "pooler_num_attention_heads": 12,
    // "pooler_num_fc_layers": 3,
    // "pooler_size_per_head": 128,
    // "pooler_type": "first_token_transform",
    // "type_vocab_size": 2,
    // "vocab_size": 21128

    /// referer: https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json
    pub fn clip_vit_base_patch16() -> Self {
        Self {
            vocab_size: 21128,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
        }
    }
}

pub struct ChineseClipTextEmbeddings {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    token_type_embeddings: candle_nn::Embedding,
    layer_norm: candle_nn::LayerNorm,
    dropout: candle_nn::Dropout,
    position_embedding_type: PositionEmbeddingType,
    position_ids: candle::Tensor,
    token_type_ids: candle::Tensor,
}

impl ChineseClipTextEmbeddings {
    pub fn new(vs: candle_nn::VarBuilder, c: &ChineseClipTextConfig) -> Result<Self> {
        let word_embeddings =
            candle_nn::embedding(c.vocab_size, c.hidden_size, vs.pp("word_embeddings"))?;
        let position_embeddings = candle_nn::embedding(
            c.max_position_embeddings,
            c.hidden_size,
            vs.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            c.type_vocab_size,
            c.hidden_size,
            vs.pp("token_type_embeddings"),
        )?;
        let layer_norm = candle_nn::layer_norm::<f64>(
            c.hidden_size,
            c.layer_norm_eps.into(),
            vs.pp("layer_norm"),
        )?;
        let dropout = candle_nn::Dropout::new(c.hidden_dropout_prob);
        let position_ids =
            Tensor::arange(0u32, c.max_position_embeddings as u32, vs.device())?.unsqueeze(0)?;
        let token_type_ids = Tensor::zeros(position_ids.shape(), DType::I64, vs.device())?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            position_embedding_type: c.position_embedding_type.clone(),
            position_ids,
            token_type_ids: token_type_ids,
        })
    }
}

impl Module for ChineseClipTextEmbeddings {
    fn forward(&self, xs: &candle::Tensor) -> Result<Tensor> {
        let input_shape = xs.shape();
        let seq_length = input_shape.dims1()?;
        // let batch_size = input_shape.dims0()?;

        // self.position_ids.slice, dim, offset)
        let position_ids = (0..seq_length as u32).collect::<Vec<_>>();
        let position_ids = self.position_ids.index_select(
            &Tensor::new(&position_ids[..], self.position_ids.device())?,
            1,
        )?;

        let word_embeddings = self.word_embeddings.forward(&xs)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&self.token_type_ids)?;
        let embeddings = (word_embeddings + token_type_embeddings)?;
        let embeddings = match self.position_embedding_type {
            PositionEmbeddingType::Absolute => {
                let position_embeddings = self.position_embeddings.forward(&position_ids)?;
                (embeddings + position_embeddings)?
            }
            _ => embeddings,
        };
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, false)?;
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]

    pub fn test_tmp() {
        let data = candle::Tensor::arange(0.0, 100.0, &Device::Cpu).unwrap();
        println!("{:?}", data);
        println!("{:?}", data.shape());

        let data = data.unsqueeze(0).unwrap();
        println!("{:?}", data);
        println!("{:?}", data.shape());

        let seq_length = 10;
        let position_ids = (0..seq_length as u32).collect::<Vec<_>>();
        let position_ids = data
            .index_select(&Tensor::new(&position_ids[..], data.device()).unwrap(), 1)
            .unwrap();
        println!("{:?}", position_ids);
        println!("{:?}", position_ids.shape());
    }
}
