#![allow(unused)]
use candle::{IndexOp, Result, Tensor};
use candle_nn::{layer_norm, linear_b as linear, Activation, LayerNorm, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub num_cross_attention_key_value_heads: Option<usize>,
    pub activation_function: Activation,
    pub hidden_size: usize,
    pub scale_embedding: bool,
    pub num_codebooks: usize,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub tie_word_embeddings: bool,
    pub rope_embeddings: bool,
    pub rope_theta: f64,
}

#[derive(Debug, Clone)]
pub struct Attention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    is_causal: bool,
}

impl Attention {
    fn new(num_kv_heads: usize, is_causal: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.rope_embeddings {
            candle::bail!("rope embeddings are not supported");
        }
        let embed_dim = cfg.hidden_size;
        let head_dim = embed_dim / cfg.num_attention_heads;
        let kv_out_dim = num_kv_heads * head_dim;
        let k_proj = linear(embed_dim, kv_out_dim, false, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, kv_out_dim, false, vb.pp("v_proj"))?;
        let q_proj = linear(embed_dim, embed_dim, false, vb.pp("q_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, false, vb.pp("out_proj"))?;
        Ok(Self {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            is_causal,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct DecoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: Activation,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        let kv_heads_cross = cfg.num_cross_attention_key_value_heads.unwrap_or(kv_heads);

        let self_attn = Attention::new(kv_heads, true, cfg, vb.pp("self_attn"))?;
        let encoder_attn = Attention::new(kv_heads_cross, false, cfg, vb.pp("encoder_attn"))?;
        let self_attn_layer_norm =
            layer_norm(cfg.hidden_size, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn_layer_norm =
            layer_norm(cfg.hidden_size, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear(cfg.hidden_size, cfg.ffn_dim, false, vb.pp("fc1"))?;
        let fc2 = linear(cfg.ffn_dim, cfg.hidden_size, false, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.hidden_size, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation: cfg.activation_function,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: &Tensor,
        encoder_xs: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        // Self attention
        let residual = xs;
        let xs = xs.apply(&self.self_attn_layer_norm)?;
        let xs = self.self_attn.forward(&xs, None, Some(attention_mask))?;
        let xs = (residual + xs)?;

        // Cross attention
        let residual = &xs;
        let xs = xs.apply(&self.encoder_attn_layer_norm)?;
        let xs = self
            .encoder_attn
            .forward(&xs, Some(encoder_xs), Some(encoder_attention_mask))?;
        let xs = (residual + xs)?;

        // Fully connected
        let residual = &xs;
        let xs = xs
            .apply(&self.final_layer_norm)?
            .apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Vec<candle_nn::Embedding>,
    embed_positions: Tensor,
    layers: Vec<DecoderLayer>,
    layer_norm: LayerNorm,
    num_codebooks: usize,
    lm_heads: Vec<Linear>,
    dtype: candle::DType,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_d = vb.pp("decoder");
        let mut embed_tokens = Vec::with_capacity(cfg.num_codebooks);
        let vb_e = vb_d.pp("embed_tokens");
        for embed_idx in 0..cfg.num_codebooks {
            let e = candle_nn::embedding(cfg.vocab_size + 1, cfg.hidden_size, vb_d.pp(embed_idx))?;
            embed_tokens.push(e)
        }
        let embed_positions = vb_d.get(
            (cfg.max_position_embeddings, cfg.hidden_size),
            "embed_positions.weights",
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_d.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let layer_norm = layer_norm(cfg.hidden_size, 1e-5, vb_d.pp("layer_norm"))?;

        let mut lm_heads = Vec::with_capacity(cfg.num_codebooks);
        let vb_l = vb.pp("lm_heads");
        for lm_idx in 0..cfg.num_codebooks {
            let lm_head = linear(cfg.hidden_size, cfg.vocab_size, false, vb.pp(lm_idx))?;
            lm_heads.push(lm_head)
        }
        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            num_codebooks: cfg.num_codebooks,
            lm_heads,
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        encoder_xs: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let (b_sz, num_codebooks, seq_len) = input_ids.dims3()?;
        let mut inputs_embeds = Tensor::zeros((0, seq_len), self.dtype, input_ids.device())?;
        for (idx, embs) in self.embed_tokens.iter().enumerate() {
            let e = input_ids.i((.., idx))?.apply(embs)?;
            inputs_embeds = (inputs_embeds + e)?
        }
        // TODO: embed_positions
        let mut xs = inputs_embeds;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask, encoder_xs, encoder_attention_mask)?;
        }
        let xs = xs.apply(&self.layer_norm)?;
        let mut lm_logits = Vec::with_capacity(self.num_codebooks);
        for lm_head in self.lm_heads.iter() {
            let logits = xs.apply(lm_head)?;
            lm_logits.push(logits)
        }
        Ok(lm_logits)
    }
}
