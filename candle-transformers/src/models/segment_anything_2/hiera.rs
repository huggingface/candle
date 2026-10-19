//! Hiera image trunk as used by Segment Anything 2.
//!
//! This differs from the timm flavoured [`crate::models::hiera`] implementation: it keeps the
//! intermediate per-stage feature maps (SAM 2 consumes a feature pyramid rather than a single
//! stride 16 map) and it pools the queries at the stage transitions.
//!
//! - 💻 [SAM 2](https://github.com/facebookresearch/sam2)
//! - 📝 [Paper](https://arxiv.org/abs/2408.00714)
use candle::{Device, IndexOp, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

/// Coefficients of the cubic convolution kernel used by torch, `A` is -0.75 there.
fn cubic_weights(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let c1 = |x: f64| ((A + 2.) * x - (A + 3.)) * x * x + 1.;
    let c2 = |x: f64| ((A * x - 5. * A) * x + 8. * A) * x - 4. * A;
    [c2(t + 1.), c1(t), c1(1. - t), c2(2. - t)]
}

/// `(out_sz, in_sz)` resampling matrix matching `F.interpolate(mode="bicubic")` with
/// `align_corners=False`. Out of bound taps are clamped to the border like torch does.
fn bicubic_matrix(in_sz: usize, out_sz: usize, device: &Device) -> Result<Tensor> {
    let scale = in_sz as f64 / out_sz as f64;
    let mut m = vec![0f32; out_sz * in_sz];
    for i in 0..out_sz {
        let src = (i as f64 + 0.5) * scale - 0.5;
        let base = src.floor();
        let ws = cubic_weights(src - base);
        for (k, w) in ws.iter().enumerate() {
            let idx = (base as i64 - 1 + k as i64).clamp(0, in_sz as i64 - 1) as usize;
            m[i * in_sz + idx] += *w as f32;
        }
    }
    Tensor::from_vec(m, (out_sz, in_sz), device)
}

/// Bicubic resize of a `(b, c, h, w)` tensor, performed as two separable matmuls.
fn bicubic_resize(xs: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (_b, _c, in_h, in_w) = xs.dims4()?;
    let device = xs.device();
    let dtype = xs.dtype();
    let m_w = bicubic_matrix(in_w, out_w, device)?
        .t()?
        .contiguous()?
        .reshape((1, 1, in_w, out_w))?
        .to_dtype(dtype)?;
    let xs = xs.contiguous()?.broadcast_matmul(&m_w)?;
    let m_h = bicubic_matrix(in_h, out_h, device)?
        .t()?
        .contiguous()?
        .reshape((1, 1, in_h, out_h))?
        .to_dtype(dtype)?;
    xs.transpose(2, 3)?
        .contiguous()?
        .broadcast_matmul(&m_h)?
        .transpose(2, 3)?
        .contiguous()
}

/// Split a `(b, h, w, c)` tensor into `(b * num_windows, window, window, c)`, zero padding the
/// bottom/right edges when needed. Also returns the padded spatial size.
fn window_partition(xs: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = xs.dims4()?;
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let xs = if pad_h > 0 {
        xs.pad_with_zeros(1, 0, pad_h)?
    } else {
        xs.clone()
    };
    let xs = if pad_w > 0 {
        xs.pad_with_zeros(2, 0, pad_w)?
    } else {
        xs
    };
    let (h_p, w_p) = (h + pad_h, w + pad_w);
    let windows = xs
        .reshape((
            b,
            h_p / window_size,
            window_size,
            w_p / window_size,
            window_size,
            c,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape(((), window_size, window_size, c))?;
    Ok((windows, (h_p, w_p)))
}

/// Inverse of [`window_partition`], dropping the padding that was added.
fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
) -> Result<Tensor> {
    let c = windows.dim(3)?;
    let b = windows.dim(0)? / (h_p * w_p / window_size / window_size);
    let xs = windows
        .reshape((
            b,
            h_p / window_size,
            w_p / window_size,
            window_size,
            window_size,
            c,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b, h_p, w_p, c))?;
    if h_p > h || w_p > w {
        xs.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()
    } else {
        Ok(xs)
    }
}

/// 2x2 max pooling over a `(b, h, w, c)` tensor.
fn do_pool(xs: &Tensor) -> Result<Tensor> {
    xs.permute((0, 3, 1, 2))?
        .contiguous()?
        .max_pool2d(2)?
        .permute((0, 2, 3, 1))?
        .contiguous()
}

#[derive(Debug)]
struct MultiScaleAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    q_pool: bool,
}

impl MultiScaleAttention {
    fn new(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        q_pool: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let qkv = candle_nn::linear(dim, dim_out * 3, vb.pp("qkv"))?;
        let proj = candle_nn::linear(dim_out, dim_out, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            q_pool,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = xs.dims4()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, h * w, 3, self.num_heads, ()))?;
        let head_dim = qkv.dim(4)?;
        let q = qkv.i((.., .., 0))?;
        let k = qkv.i((.., .., 1))?;
        let v = qkv.i((.., .., 2))?;
        // Query pooling, this is what downsamples the feature map at the stage transitions.
        let (q, out_h, out_w) = if self.q_pool {
            let q = q
                .reshape((b, h, w, self.num_heads * head_dim))?
                .contiguous()?;
            let q = do_pool(&q)?;
            let (out_h, out_w) = (q.dim(1)?, q.dim(2)?);
            let q = q.reshape((b, out_h * out_w, self.num_heads, head_dim))?;
            (q, out_h, out_w)
        } else {
            (q, h, w)
        };
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let attn = (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let xs = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, out_h, out_w, ()))?;
        self.proj.forward(&xs)
    }
}

#[derive(Debug)]
struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.lin1.forward(xs)?.gelu_erf()?.apply(&self.lin2)
    }
}

#[derive(Debug)]
struct MultiScaleBlock {
    norm1: LayerNorm,
    attn: MultiScaleAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    proj: Option<Linear>,
    q_stride: bool,
    window_size: usize,
    span: tracing::Span,
}

impl MultiScaleBlock {
    fn new(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        q_stride: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim_out, 1e-6, vb.pp("norm2"))?;
        let attn = MultiScaleAttention::new(dim, dim_out, num_heads, q_stride, vb.pp("attn"))?;
        let vb_m = vb.pp("mlp").pp("layers");
        let mlp = Mlp {
            lin1: candle_nn::linear(dim_out, dim_out * 4, vb_m.pp(0))?,
            lin2: candle_nn::linear(dim_out * 4, dim_out, vb_m.pp(1))?,
        };
        let proj = if dim != dim_out {
            Some(candle_nn::linear(dim, dim_out, vb.pp("proj"))?)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "hiera-block");
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            proj,
            q_stride,
            window_size,
            span,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let shortcut = xs.clone();
        let xs = self.norm1.forward(xs)?;
        // On a stage transition the residual has to be projected and pooled to match.
        let shortcut = match self.proj.as_ref() {
            None => shortcut,
            Some(proj) => {
                let p = proj.forward(&xs)?;
                if self.q_stride {
                    do_pool(&p)?
                } else {
                    p
                }
            }
        };
        let (h, w) = (xs.dim(1)?, xs.dim(2)?);
        let (xs, pad_hw) = if self.window_size > 0 {
            window_partition(&xs, self.window_size)?
        } else {
            (xs, (h, w))
        };
        let xs = self.attn.forward(&xs)?;
        // Query pooling halves the spatial extent, so the windows used to stitch the output back
        // together are half the size of the ones used to split the input.
        let (window_size, (h, w), pad_hw) = if self.q_stride {
            let window_size = self.window_size / 2;
            let (h, w) = (shortcut.dim(1)?, shortcut.dim(2)?);
            let pad_h = (window_size - h % window_size) % window_size;
            let pad_w = (window_size - w % window_size) % window_size;
            (window_size, (h, w), (h + pad_h, w + pad_w))
        } else {
            (self.window_size, (h, w), pad_hw)
        };
        let xs = if self.window_size > 0 {
            window_unpartition(&xs, window_size, pad_hw, (h, w))?
        } else {
            xs
        };
        let xs = (shortcut + xs)?;
        let residual = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs + residual
    }
}

#[derive(Debug)]
pub struct Hiera {
    patch_embed: candle_nn::Conv2d,
    pos_embed: Tensor,
    pos_embed_window: Tensor,
    blocks: Vec<MultiScaleBlock>,
    stage_ends: Vec<usize>,
    /// Number of channels of each stage output, from the coarsest to the finest resolution.
    channel_list: Vec<usize>,
    span: tracing::Span,
}

impl Hiera {
    pub fn new(cfg: &super::Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: 4,
            padding: 3,
            ..Default::default()
        };
        let patch_embed =
            candle_nn::conv2d(3, cfg.embed_dim, 7, conv_cfg, vb.pp("patch_embed.proj"))?;
        let (pe_h, pe_w) = cfg.window_pos_embed_bkg_spatial_size;
        let pos_embed = vb.get((1, cfg.embed_dim, pe_h, pe_w), "pos_embed")?;
        let pos_embed_window = vb.get(
            (1, cfg.embed_dim, cfg.window_spec[0], cfg.window_spec[0]),
            "pos_embed_window",
        )?;

        let depth: usize = cfg.stages.iter().sum();
        let stage_ends = (1..=cfg.stages.len())
            .map(|i| cfg.stages[..i].iter().sum::<usize>() - 1)
            .collect::<Vec<usize>>();
        // The three blocks that immediately follow a stage boundary pool their queries.
        let q_pool_blocks = stage_ends[..stage_ends.len() - 1]
            .iter()
            .map(|x| x + 1)
            .collect::<Vec<usize>>();

        let mut blocks = Vec::with_capacity(depth);
        let mut channel_list = Vec::with_capacity(cfg.stages.len());
        let vb_b = vb.pp("blocks");
        let mut embed_dim = cfg.embed_dim;
        let mut num_heads = cfg.num_heads;
        let mut cur_stage = 1;
        for i in 0..depth {
            // The window size lags by one block: the first block of a stage still uses the window
            // size of the previous stage.
            let window_size = if cfg.global_att_blocks.contains(&i) {
                0
            } else {
                cfg.window_spec[cur_stage - 1]
            };
            let dim_out = if i > 0 && stage_ends.contains(&(i - 1)) {
                num_heads *= 2;
                cur_stage += 1;
                embed_dim * 2
            } else {
                embed_dim
            };
            let block = MultiScaleBlock::new(
                embed_dim,
                dim_out,
                num_heads,
                q_pool_blocks.contains(&i),
                window_size,
                vb_b.pp(i),
            )?;
            blocks.push(block);
            embed_dim = dim_out;
            if stage_ends.contains(&i) {
                channel_list.push(dim_out)
            }
        }
        channel_list.reverse();
        let span = tracing::span!(tracing::Level::TRACE, "hiera");
        Ok(Self {
            patch_embed,
            pos_embed,
            pos_embed_window,
            blocks,
            stage_ends,
            channel_list,
            span,
        })
    }

    pub fn channel_list(&self) -> &[usize] {
        &self.channel_list
    }

    /// Interpolated background embedding plus the tiled per-window embedding.
    fn get_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let pos_embed = bicubic_resize(&self.pos_embed, h, w)?;
        let (_, c, wh, ww) = self.pos_embed_window.dims4()?;
        let window = self
            .pos_embed_window
            .reshape((1, c, 1, wh, 1, ww))?
            .broadcast_as((1, c, h / wh, wh, w / ww, ww))?
            .contiguous()?
            .reshape((1, c, h, w))?;
        (pos_embed + window)?.permute((0, 2, 3, 1))
    }

    /// Returns the output of each stage as `(b, c, h, w)` tensors, from stride 4 to stride 32.
    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();
        let xs = xs.apply(&self.patch_embed)?.permute((0, 2, 3, 1))?;
        let (h, w) = (xs.dim(1)?, xs.dim(2)?);
        let mut xs = xs.broadcast_add(&self.get_pos_embed(h, w)?)?;
        let mut outputs = Vec::with_capacity(self.stage_ends.len());
        for (i, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs)?;
            if self.stage_ends.contains(&i) {
                outputs.push(xs.permute((0, 3, 1, 2))?.contiguous()?)
            }
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_matches_torch() -> Result<()> {
        // Reference values from `F.interpolate(x, size=(7, 7), mode="bicubic")`, which is what
        // SAM 2 uses to resize the background positional embedding.
        let expected: [f32; 49] = [
            -0.496083, -0.117621, 0.499726, 1.103134, 1.706543, 2.323889, 2.702351, 1.017763,
            1.396224, 2.013572, 2.616979, 3.220387, 3.837733, 4.216195, 3.487154, 3.865613,
            4.482962, 5.086369, 5.689778, 6.307123, 6.685585, 5.900784, 6.279243, 6.896592, 7.5,
            8.103409, 8.720752, 9.099215, 8.31442, 8.692878, 9.310227, 9.913634, 10.517044,
            11.134387, 11.51285, 10.783802, 11.162258, 11.779609, 12.383016, 12.986426, 13.603767,
            13.98223, 12.297651, 12.676106, 13.293458, 13.896865, 14.500276, 15.117616, 15.496078,
        ];
        let xs = Tensor::arange(0f32, 16f32, &Device::Cpu)?.reshape((1, 1, 4, 4))?;
        let ys = bicubic_resize(&xs, 7, 7)?;
        assert_eq!(ys.dims(), &[1, 1, 7, 7]);
        for (got, want) in ys.flatten_all()?.to_vec1::<f32>()?.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
        Ok(())
    }
}
