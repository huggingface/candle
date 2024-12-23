use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b, rms_norm, Linear, RmsNorm, VarBuilder};

fn default_act() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}

fn default_hidden_size() -> usize {
    1024
}

fn default_intermediate_size() -> usize {
    4096
}

fn default_num_channels() -> usize {
    3
}

fn default_num_hidden_layers() -> usize {
    24
}

fn default_num_attention_heads() -> usize {
    16
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    pub head_dim: Option<usize>,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_act")]
    pub hidden_act: candle_nn::Activation,
}

impl Config {
    pub fn pixtral_12b_2409() -> Self {
        Self {
            hidden_size: 1024,
            num_channels: 3,
            image_size: 1024,
            patch_size: 16,
            rope_theta: 10000.0,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            head_dim: None,
            // Default
            hidden_act: candle_nn::Activation::Silu,
        }
    }

    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let q_proj = linear_b(h, h, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(h, h, false, vb.pp("k_proj"))?;
        let v_proj = linear_b(h, h, false, vb.pp("v_proj"))?;
        let o_proj = linear_b(h, h, false, vb.pp("o_proj"))?;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        emb: &RotaryEmbedding,
        subsampled_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, patches, _) = xs.dims3()?;
        let query_states = xs.apply(&self.q_proj)?;
        let key_states = xs.apply(&self.k_proj)?;
        let value_states = xs.apply(&self.v_proj)?;

        let shape = (b, patches, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let (query_states, key_states) =
            emb.apply_rotary_emb_qkv(&query_states, &key_states, subsampled_positions)?;
        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((b, patches, ()))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_proj = linear_b(h, i, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(h, i, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(i, h, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.gate_proj)?.apply(&self.act_fn)? * xs.apply(&self.up_proj))?
            .apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    attention_norm: RmsNorm,
    feed_forward: Mlp,
    attention: Attention,
    ffn_norm: RmsNorm,
}

impl AttentionLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("attention_norm"))?;
        let feed_forward = Mlp::new(cfg, vb.pp("feed_forward"))?;
        let attention = Attention::new(cfg, vb.pp("attention"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention_norm,
            feed_forward,
            attention,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        emb: &RotaryEmbedding,
        subsampled_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention.forward(
            &xs.apply(&self.attention_norm)?,
            emb,
            subsampled_positions,
            attention_mask,
        )?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.ffn_norm)?.apply(&self.feed_forward)?;
        xs + residual
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    layers: Vec<AttentionLayer>,
}

impl Transformer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = AttentionLayer::new(cfg, vb.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor,
        emb: &RotaryEmbedding,
        subsampled_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, emb, subsampled_positions, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device();
        let dim = cfg.head_dim();
        let rope_theta = cfg.rope_theta as f32;
        let max_patches_per_side = cfg.image_size / cfg.patch_size;
        let freqs: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let freqs_h = freqs.iter().step_by(2).copied().collect::<Vec<_>>();
        let freqs_h = Tensor::new(freqs_h, dev)?;
        let freqs_w = freqs.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
        let freqs_w = Tensor::new(freqs_w, dev)?;
        let h = Tensor::arange(0u32, max_patches_per_side as u32, dev)?.to_dtype(DType::F32)?;
        let w = Tensor::arange(0u32, max_patches_per_side as u32, dev)?.to_dtype(DType::F32)?;
        let freqs_h = h.unsqueeze(1)?.matmul(&freqs_h.unsqueeze(0)?)?;
        let freqs_w = w.unsqueeze(1)?.matmul(&freqs_w.unsqueeze(0)?)?;
        let inv_freq = Tensor::cat(
            &[
                freqs_h.unsqueeze(1)?.repeat((1, max_patches_per_side, 1))?,
                freqs_w.unsqueeze(0)?.repeat((max_patches_per_side, 1, 1))?,
            ],
            D::Minus1,
        )?
        .reshape(((), dim / 2))?;
        let cos = inv_freq.cos()?.to_dtype(dtype)?;
        let sin = inv_freq.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        subsampled_positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, _seq_len, _n_embd) = q.dims4()?;
        let (cos, sin) = match subsampled_positions {
            None => (&self.cos, &self.sin),
            Some(pos) => (
                &self.cos.index_select(pos, 0)?,
                &self.sin.index_select(pos, 0)?,
            ),
        };
        let q_embed = candle_nn::rotary_emb::rope(q, cos, sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, cos, sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    patch_conv: candle_nn::Conv2d,
    ln_pre: RmsNorm,
    transformer: Transformer,
    patch_positional_embedding: RotaryEmbedding,
    max_image_width: u32,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_conv = candle_nn::conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_conv"),
        )?;
        let ln_pre = candle_nn::rms_norm(cfg.hidden_size, 1e-5, vb.pp("ln_pre"))?;
        let transformer = Transformer::new(cfg, vb.pp("transformer"))?;
        let patch_positional_embedding =
            RotaryEmbedding::new(cfg, vb.pp("patch_positional_embedding"))?;
        let max_image_width = (cfg.image_size / cfg.patch_size) as u32;
        Ok(Self {
            patch_conv,
            ln_pre,
            transformer,
            patch_positional_embedding,
            max_image_width,
        })
    }

    pub fn position_ids_in_meshgrid(
        &self,
        num_patches_h: usize,
        num_patches_w: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let idx = Tensor::arange(0, num_patches_h as u32, device)?;
        let idy = Tensor::arange(0, num_patches_w as u32, device)?;
        let mesh = Tensor::meshgrid(&[idx, idy], false)?;
        let ids = (&mesh[0] * (self.max_image_width as f64) + &mesh[1])?.flatten_all()?;
        Ok(ids)
    }
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let patch_embeds = xs.apply(&self.patch_conv)?;
        let subsampled_positions = Some(self.position_ids_in_meshgrid(
            patch_embeds.dim(2)?,
            patch_embeds.dim(3)?,
            patch_embeds.device(),
        )?);
        let patch_embeds = patch_embeds.flatten_from(2)?.t()?.apply(&self.ln_pre)?;
        self.transformer.forward(
            &patch_embeds,
            &self.patch_positional_embedding,
            subsampled_positions.as_ref(),
            None,
        )
    }
}
