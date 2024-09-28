use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L27
#[derive(serde::Deserialize, Clone, Debug)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L132
#[derive(serde::Deserialize, Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
}

trait TransformerConfig {
    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn layer_norm_eps(&self) -> f64;
    fn hidden_act(&self) -> candle_nn::Activation;
}

impl TransformerConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> candle_nn::Activation {
        self.hidden_act
    }
}

impl TransformerConfig for VisionConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> candle_nn::Activation {
        self.hidden_act
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L228
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size();
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let num_heads = cfg.num_attention_heads();
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, q_len, _) = xs.dims3()?;
        let query_states = xs.apply(&self.q_proj)?;
        let key_states = xs.apply(&self.k_proj)?;
        let value_states = xs.apply(&self.v_proj)?;

        let shape = (batch_size, q_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?;

        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // The original implementation upcasts to f32 but candle_nn::ops::softmax should handle this properly.
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_outputs = attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?
            .apply(&self.out_proj)?;
        Ok(attn_outputs)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L599
#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: candle_nn::Activation,
}

impl Mlp {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let intermediate_size = cfg.intermediate_size();
        let fc1 = candle_nn::linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(intermediate_size, hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation_fn: cfg.hidden_act(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &candle::Tensor) -> Result<candle::Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L614
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let layer_norm_eps = cfg.layer_norm_eps();
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm1"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        xs + residual
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let mut layers = vec![];
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers() {
            let layer = EncoderLayer::new(cfg, vb.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_embedding: candle_nn::Embedding,
    position_ids: Tensor,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = candle_nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_embedding"),
        )?;
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
        let position_ids = Tensor::arange(0, num_patches as i64, vb.device())?;
        let position_embedding =
            candle_nn::embedding(num_patches, cfg.hidden_size(), vb.pp("position_embedding"))?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for VisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_batch, _channels, _height, _width) = xs.dims4()?;
        let embeddings = xs
            .apply(&self.patch_embedding)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

#[derive(Debug, Clone)]
struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    // head: Option<MultiheadAttentionPoolingHead>,
}

impl VisionTransformer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.embeddings)?;
        let xs = self.encoder.forward(&xs, None)?;
        xs.apply(&self.post_layernorm)
    }
}

#[derive(Debug, Clone)]
pub struct VisionModel {
    vision_model: VisionTransformer,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let vision_model = VisionTransformer::new(cfg, vb.pp("vision_model"))?;
        Ok(Self { vision_model })
    }
}

impl Module for VisionModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.vision_model)
    }
}

#[derive(Debug, Clone)]
struct TextEmbeddings {
    token_embedding: candle_nn::Embedding,
    position_embedding: candle_nn::Embedding,
    position_ids: Tensor,
}

impl TextEmbeddings {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embedding"))?;
        let position_embedding = candle_nn::embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?.unsqueeze(0)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for TextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

#[derive(Debug, Clone)]
struct TextTransformer {
    embeddings: TextEmbeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
    head: Linear,
}

impl TextTransformer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = TextEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let final_layer_norm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;
        let head = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            head,
        })
    }

    // TODO: rewrite to newer version
    fn build_causal_attention_mask(
        bsz: usize,
        seq_len: usize,
        mask_after: usize,
        device: &candle::Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i || j > mask_after {
                        f32::MIN
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        mask.broadcast_as((bsz, 1, seq_len, seq_len))
    }

    pub fn forward_with_mask(&self, input_ids: &Tensor, mask_after: usize) -> Result<Tensor> {
        let (bsz, seq_len) = input_ids.dims2()?;
        let input_ids = self.embeddings.forward(input_ids)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, input_ids.device())?;
        let input_ids = self
            .encoder
            .forward(&input_ids, Some(&causal_attention_mask))?;
        let last_hidden_state = self.final_layer_norm.forward(&input_ids)?;
        last_hidden_state
            .i((.., seq_len - 1, ..))?
            .apply(&self.head)
    }
}

impl Module for TextTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(xs, usize::MAX)
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    text_model: TextTransformer,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let text_model = TextTransformer::new(cfg, vb.pp("text_model"))?;
        Ok(Self { text_model })
    }
}

impl Module for TextModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.text_model)
    }
}
