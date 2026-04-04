//! Window Q-Former downsampler for Granite 4.0 Vision.
//!
//! Processes vision features through windowed cross-attention to compress spatial tokens.
//! Supports two downsampling strategies:
//! - InterpolateDownsampler: area interpolation (for deepstack projectors)
//! - SpatialOffsetDownsampler: 2x2 block offset sampling (for spatial projectors)

use candle::{Result, Tensor};
use candle_nn::{layer_norm, Linear, Module, VarBuilder};

use super::config::Config;

// ---------------------------------------------------------------------------
// BERT-style attention block (used for both self-attention and cross-attention)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: candle_nn::LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl BertAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let attn_vb = vb.pp("attention");
        let query = candle_nn::linear(hidden_size, hidden_size, attn_vb.pp("query"))?;
        let key = candle_nn::linear(hidden_size, hidden_size, attn_vb.pp("key"))?;
        let value = candle_nn::linear(hidden_size, hidden_size, attn_vb.pp("value"))?;
        let out_vb = vb.pp("output");
        let dense = candle_nn::linear(hidden_size, hidden_size, out_vb.pp("dense"))?;
        let layer_norm = layer_norm(hidden_size, 1e-6, out_vb.pp("LayerNorm"))?;
        Ok(Self {
            query,
            key,
            value,
            dense,
            layer_norm,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass. For self-attention, pass encoder_hidden_states = None.
    /// For cross-attention, pass the encoder states (key/value source).
    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let context = encoder_hidden_states.unwrap_or(hidden_states);
        let (b, seq_q, _) = hidden_states.dims3()?;
        let (_, seq_k, _) = context.dims3()?;

        let q = hidden_states
            .apply(&self.query)?
            .reshape((b, seq_q, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = context
            .apply(&self.key)?
            .reshape((b, seq_k, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = context
            .apply(&self.value)?
            .reshape((b, seq_k, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((b, seq_q, ()))?;

        let out = out.apply(&self.dense)?;
        self.layer_norm.forward(&(out + hidden_states)?)
    }
}

// ---------------------------------------------------------------------------
// Q-Former layer: self-attention → cross-attention → FFN
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct QFormerLayer {
    attention: BertAttention,
    crossattention: BertAttention,
    intermediate_dense: Linear,
    output_dense: Linear,
    output_layer_norm: candle_nn::LayerNorm,
}

impl QFormerLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention = BertAttention::new(hidden_size, num_heads, vb.pp("attention"))?;
        let crossattention =
            BertAttention::new(hidden_size, num_heads, vb.pp("crossattention"))?;
        let intermediate_dense = candle_nn::linear(
            hidden_size,
            intermediate_size,
            vb.pp("intermediate_query.dense"),
        )?;
        let output_dense =
            candle_nn::linear(intermediate_size, hidden_size, vb.pp("output_query.dense"))?;
        let output_layer_norm =
            layer_norm(hidden_size, 1e-6, vb.pp("output_query.LayerNorm"))?;
        Ok(Self {
            attention,
            crossattention,
            intermediate_dense,
            output_dense,
            output_layer_norm,
        })
    }

    fn forward(&self, query: &Tensor, encoder_states: &Tensor) -> Result<Tensor> {
        let hidden = self.attention.forward(query, None)?;
        let hidden = self.crossattention.forward(&hidden, Some(encoder_states))?;
        let intermediate = hidden
            .apply(&self.intermediate_dense)?
            .gelu_erf()?;
        let output = intermediate.apply(&self.output_dense)?;
        self.output_layer_norm.forward(&(output + hidden)?)
    }
}

// ---------------------------------------------------------------------------
// Q-Former model (single layer + layernorm)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct QFormer {
    layer_norm: candle_nn::LayerNorm,
    layer: QFormerLayer,
}

impl QFormer {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let layer_norm = layer_norm(hidden_size, 1e-6, vb.pp("layernorm"))?;
        let layer = QFormerLayer::new(hidden_size, num_heads, 3072, vb.pp("encoder.layer.0"))?;
        Ok(Self { layer_norm, layer })
    }

    fn forward(&self, query_embeds: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let query = self.layer_norm.forward(query_embeds)?;
        self.layer.forward(&query, encoder_hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Downsamplers
// ---------------------------------------------------------------------------

/// Area interpolation downsampler. Requires integer scaling factor.
fn area_downsample(features: &Tensor, orig_side: usize, new_side: usize) -> Result<Tensor> {
    let (b, _hw, c) = features.dims3()?;
    let factor = orig_side / new_side;
    if orig_side != new_side * factor {
        candle::bail!(
            "area_downsample: orig_side {orig_side} not evenly divisible to new_side {new_side}"
        );
    }
    // (B, orig, orig, C) → (B, new, factor, new, factor, C)
    let x = features
        .reshape((b, orig_side, orig_side, c))?
        .reshape((b, new_side, factor, new_side, factor, c))?;
    // Average over factor dimensions (sum + divide)
    let x = x.sum_keepdim(4)?.squeeze(4)?;
    let x = x.sum_keepdim(2)?.squeeze(2)?;
    let scale = (factor * factor) as f64;
    (x / scale)?.reshape((b, new_side * new_side, c))
}

/// Spatial offset downsampler: samples one position from each 2x2 block.
fn spatial_offset_downsample(
    features: &Tensor,
    orig_side: usize,
    offset: usize,
) -> Result<Tensor> {
    let (b, _hw, c) = features.dims3()?;
    let new_side = orig_side / 2;
    let offsets = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)];
    let (oh, ow) = offsets[offset];
    // (B, orig, orig, C) → (B, new, 2, new, 2, C)
    let x = features
        .reshape((b, orig_side, orig_side, c))?
        .reshape((b, new_side, 2, new_side, 2, c))?;
    // Select offset position from each 2x2 block
    // After first narrow+squeeze, dims shift: (B, new, new, 2, C)
    let x = x.narrow(2, oh, 1)?.squeeze(2)?;
    let x = x.narrow(3, ow, 1)?.squeeze(3)?;
    // (B, new, new, C) → (B, new*new, C)
    x.reshape((b, new_side * new_side, c))
}

// ---------------------------------------------------------------------------
// Window Q-Former Downsampler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WindowQFormerDownsampler {
    norm: candle_nn::LayerNorm,
    qformer: QFormer,
    out_linear: Linear,
    query: Tensor,
    image_positions: Tensor,
    image_side: usize,
    query_side: usize,
    window_side: usize,
    spatial_offset: Option<usize>,
}

impl WindowQFormerDownsampler {
    pub fn new(cfg: &Config, spatial_offset: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let vision_hidden_size = cfg.vision_config.hidden_size;
        let llm_hidden_size = cfg.text_config.hidden_size;
        let num_heads = vision_hidden_size / 64;
        let (query_side, window_side) = cfg.query_and_window_side();
        let query_length = query_side * query_side;

        let norm = layer_norm(vision_hidden_size, 1e-6, vb.pp("norm"))?;
        let out_linear = candle_nn::linear(vision_hidden_size, llm_hidden_size, vb.pp("out_linear"))?;
        let query = vb.get((1, query_length, vision_hidden_size), "query")?;
        let image_positions =
            vb.get((1, window_side * window_side, vision_hidden_size), "image_positions")?;
        let qformer = QFormer::new(vision_hidden_size, num_heads, vb.pp("qformer"))?;

        let image_side = cfg.patches_per_side();

        Ok(Self {
            norm,
            qformer,
            out_linear,
            query,
            image_positions,
            image_side,
            query_side,
            window_side,
            spatial_offset,
        })
    }

    /// Window: (B, side*side, C) raster → (B*n*n, win*win, C)
    fn win(&self, x: &Tensor, side: usize, win: usize) -> Result<Tensor> {
        let (b, _, c) = x.dims3()?;
        let n = side / win;
        x.reshape((b, side, side, c))?
            .reshape((b, n, win, n, win, c))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((b * n * n, win * win, c))
    }

    /// Unwindow: (B*n*n, win*win, C) → (B, (n*win)^2, C)
    fn unwin(&self, xw: &Tensor, batch_size: usize, n: usize, win: usize) -> Result<Tensor> {
        let (_, _, c) = xw.dims3()?;
        let side = n * win;
        xw.reshape((batch_size, n, n, win, win, c))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((batch_size, side * side, c))
    }

    pub fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let (b, hw, _c) = image_features.dims3()?;
        if hw != self.image_side * self.image_side {
            candle::bail!(
                "Expected {} patches, got {hw}",
                self.image_side * self.image_side
            );
        }
        let n = self.image_side / self.window_side;

        // Normalize
        let normed = self.norm.forward(image_features)?;

        // Window the encoder features for cross-attention
        let enc = self.win(&normed, self.image_side, self.window_side)?;

        // Downsample to create query seeds
        let downsampled = match self.spatial_offset {
            Some(offset) => spatial_offset_downsample(&normed, self.image_side, offset)?,
            None => area_downsample(&normed, self.image_side, self.image_side * self.query_side / self.window_side)?,
        };

        let new_side = n * self.query_side;
        let downsampled_w = self.win(&downsampled, new_side, self.query_side)?;

        // Combine learned query with downsampled features
        let query_embeds = self.query.broadcast_add(&downsampled_w)?;
        let encoder_embeds = enc.broadcast_add(&self.image_positions)?;

        // Run Q-Former cross-attention
        let out_w = self.qformer.forward(&query_embeds, &encoder_embeds)?;

        // Unwindow back to spatial layout
        let out = self.unwin(&out_w, b, n, self.query_side)?;

        // Project to LLM hidden size
        out.apply(&self.out_linear)
    }
}
