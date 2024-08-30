//! Text encoder as used in most OpenCLIP pretrained models
//! https://github.com/mlfoundations/open_clip

use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, ops::softmax_last_dim, Embedding, LayerNorm, Linear, Module,
    VarBuilder,
};

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub pad_with: Option<String>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
}

impl Config {
    pub fn vit_base_patch32() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 512,
            intermediate_size: 2048,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            projection_dim: 512,
        }
    }
}

#[derive(Clone, Debug)]
struct TextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Tensor,
}

impl TextEmbeddings {
    fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        let token_embedding = embedding(c.vocab_size, c.embed_dim, vs.pp("token_embedding"))?;
        let position_embedding = vs.get(
            (c.max_position_embeddings, c.embed_dim),
            "positional_embedding",
        )?;
        Ok(TextEmbeddings {
            token_embedding,
            position_embedding,
        })
    }
}

impl Module for TextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;

        let position_embedding = self.position_embedding.narrow(0, 0, seq_length)?;

        inputs_embeds.broadcast_add(&position_embedding)
    }
}

#[derive(Clone, Debug)]
struct Attention {
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    q_proj: candle_nn::Linear,
    out_proj: Linear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl Attention {
    fn new(vs: candle_nn::VarBuilder, c: &Config) -> Result<Self> {
        let embed_dim = c.embed_dim;
        let num_attention_heads = c.num_attention_heads;

        let in_proj_weights = vs
            .get((embed_dim * 3, embed_dim), "in_proj_weight")?
            .chunk(3, 0)?;
        let (q_w, k_w, v_w) = (
            &in_proj_weights[0],
            &in_proj_weights[1],
            &in_proj_weights[2],
        );
        let in_proj_biases = vs.get(embed_dim * 3, "in_proj_bias")?.chunk(3, 0)?;
        let (q_b, k_b, v_b) = (&in_proj_biases[0], &in_proj_biases[1], &in_proj_biases[2]);

        let q_proj = Linear::new(q_w.clone(), Some(q_b.clone()));
        let k_proj = Linear::new(k_w.clone(), Some(k_b.clone()));
        let v_proj = Linear::new(v_w.clone(), Some(v_b.clone()));
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vs.pp("out_proj"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(Attention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape_multihead(&self, xs: &Tensor, bsz: usize, seq_len: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;

        let q = self.shape_multihead(&self.q_proj.forward(xs)?, bsz, seq_len)?;
        let k = self.shape_multihead(&self.k_proj.forward(xs)?, bsz, seq_len)?;
        let v = self.shape_multihead(&self.v_proj.forward(xs)?, bsz, seq_len)?;
        let q = (q * self.scale)?;

        let attn_weights = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;

        let attn_weights = softmax_last_dim(&attn_weights)?;

        let attn_output = attn_weights.matmul(&v)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, seq_len, embed_dim))?;
        let out = self.out_proj.forward(&attn_output)?;
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        let fc1 = linear(c.embed_dim, c.intermediate_size, vs.pp("c_fc"))?;
        let fc2 = linear(c.intermediate_size, c.embed_dim, vs.pp("c_proj"))?;

        Ok(Mlp { fc1, fc2 })
    }
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&xs.gelu_erf()?)
    }
}

#[derive(Clone, Debug)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        let self_attn = Attention::new(vs.pp("attn"), c)?;
        let layer_norm1 = layer_norm(c.embed_dim, 1e-5, vs.pp("ln_1"))?;
        let mlp = Mlp::new(vs.pp("mlp"), c)?;
        let layer_norm2 = layer_norm(c.embed_dim, 1e-5, vs.pp("ln_2"))?;

        Ok(EncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let out = (xs + residual)?;
        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        let vs = vs.pp("resblocks");
        let mut layers: Vec<EncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers {
            let layer = EncoderLayer::new(vs.pp(index.to_string()), c)?;
            layers.push(layer)
        }
        Ok(Encoder { layers })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

/// A text transformer as used in CLIP variants.
#[derive(Clone, Debug)]
pub struct OpenClipTextTransformer {
    embeddings: TextEmbeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
}

impl OpenClipTextTransformer {
    pub fn new(vs: VarBuilder, c: &Config) -> Result<Self> {
        let embeddings = TextEmbeddings::new(vs.clone(), c)?;
        let final_layer_norm = layer_norm(c.embed_dim, 1e-5, vs.pp("ln_final"))?;
        let encoder = Encoder::new(vs.pp("transformer"), c)?;
        Ok(OpenClipTextTransformer {
            embeddings,
            encoder,
            final_layer_norm,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_ids = self.embeddings.forward(input_ids)?;
        let input_ids = self.encoder.forward(&input_ids)?;
        self.final_layer_norm.forward(&input_ids)
    }
}

impl Module for OpenClipTextTransformer {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let output = self.forward(input_ids)?;
        let sequence_max_indices = input_ids.argmax(D::Minus1)?.to_dtype(DType::I64)?;

        let mut indices = Vec::new();
        for (batch_idx, &seq_idx) in sequence_max_indices.to_vec1::<i64>()?.iter().enumerate() {
            let index = output.i((batch_idx, seq_idx as usize))?.unsqueeze(0)?;
            indices.push(index);
        }
        Tensor::cat(&indices, 0)
    }
}
