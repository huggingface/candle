//! Hiera inference implementation based on timm.
//!
//!
//! - 💻 [Hiera](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hiera.py)
//! - 📝 [Paper](https://arxiv.org/abs/2306.00989). Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

use candle::{Result, D};
use candle_nn::{conv2d, layer_norm, linear, ops::softmax, Conv2dConfig, Func, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    channels: usize,
    heads: usize,
    stages: [usize; 4],
}

impl Config {
    pub fn tiny() -> Self {
        Self {
            channels: 96,
            heads: 1,
            stages: [1, 2, 7, 2],
        }
    }
    pub fn small() -> Self {
        Self {
            channels: 96,
            heads: 1,
            stages: [1, 2, 11, 2],
        }
    }
    pub fn base() -> Self {
        Self {
            channels: 96,
            heads: 1,
            stages: [2, 3, 16, 3],
        }
    }
    pub fn base_plus() -> Self {
        Self {
            channels: 112,
            heads: 2,
            stages: [2, 3, 16, 3],
        }
    }
    pub fn large() -> Self {
        Self {
            channels: 144,
            heads: 2,
            stages: [2, 6, 36, 4],
        }
    }
    pub fn huge() -> Self {
        Self {
            channels: 256,
            heads: 4,
            stages: [2, 6, 36, 4],
        }
    }
}

const NUM_TOKENS: usize = 56 * 56;

fn hiera_embeddings(channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv_cfg = Conv2dConfig {
        stride: 4,
        padding: 3,
        ..Default::default()
    };
    let proj = conv2d(3, channels, 7, conv_cfg, vb.pp("patch_embed.proj"))?;

    let pos_embed = vb.get((1, NUM_TOKENS, channels), "pos_embed")?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&proj)?;
        let (b, c, _, _) = xs.dims4()?;
        let xs = xs.reshape((b, c, ()))?.transpose(1, 2)?;
        let xs = xs.broadcast_add(&pos_embed)?;
        Ok(xs)
    }))
}

fn hiera_unroll() -> Result<Func<'static>> {
    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        let (mut b, _, c) = xs.dims3()?;
        let mut size = 56;

        xs = xs.reshape((b, size, size, c))?;
        for _ in 0..3 {
            size /= 2;
            let new_shape = &[b, size, 2, size, 2, c];
            xs = xs.reshape(new_shape)?;
            xs = xs.permute((0, 2, 4, 1, 3, 5))?;
            xs = xs.flatten(0, 2)?;
            b *= 4;
        }
        xs = xs.reshape(((), NUM_TOKENS, c))?;

        Ok(xs)
    }))
}

fn hiera_mlp(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let fc1 = linear(in_channels, out_channels, vb.pp("fc1"))?;
    let fc2 = linear(out_channels, in_channels, vb.pp("fc2"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&fc1)?.gelu()?.apply(&fc2)?;
        Ok(xs)
    }))
}

fn hiera_attention(
    in_channels: usize,
    out_channels: usize,
    heads: usize,
    q_stride: usize,
    window_size: usize,
    use_mask_attention: bool,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let head_dim = out_channels / heads;

    let scale = (head_dim as f64).powf(-0.5);

    let proj = linear(out_channels, out_channels, vb.pp("proj"))?;
    let qkv = linear(in_channels, out_channels * 3, vb.pp("qkv"))?;

    Ok(Func::new(move |xs| {
        let (b, n, _) = xs.dims3()?;

        let num_windows = if use_mask_attention {
            n / (q_stride * window_size)
        } else {
            1
        };
        let qkv = xs.apply(&qkv)?;

        let ec = qkv.elem_count();
        let s = ec / (b * num_windows * 3 * heads * head_dim);
        let qkv = qkv
            .reshape((b, s, num_windows, 3, heads, head_dim))?
            .permute((3, 0, 4, 2, 1, 5))?;

        let mut q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        if q_stride > 1 {
            let ec = q.elem_count();
            let s = ec / (b * num_windows * q_stride * heads * head_dim);
            q = q
                .reshape((b, heads, num_windows, q_stride, s, head_dim))?
                .max(3)?;
        }

        let q = (q * scale)?;

        // Q, K and V are 6 dimensional with the first dimension being 1.
        // Squeeze them for the attention calculation since 6 dimensional matmuls are not supported.
        let att = q
            .squeeze(0)?
            .matmul(&k.squeeze(0)?.transpose(D::Minus2, D::Minus1)?)?;
        let att = softmax(&att, D::Minus1)?;
        let xs = att.matmul(&v.squeeze(0)?)?.unsqueeze(0)?;

        let xs = xs.transpose(1, 3)?.reshape((b, (), out_channels))?;
        let xs = xs.apply(&proj)?;

        Ok(xs)
    }))
}

fn hiera_block(
    heads: usize,
    in_channels: usize,
    out_channels: usize,
    q_stride: usize,
    window_size: usize,
    use_mask_attention: bool,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let norm1 = layer_norm(in_channels, 1e-6, vb.pp("norm1"))?;
    let norm2 = layer_norm(out_channels, 1e-6, vb.pp("norm2"))?;
    let proj = linear(in_channels, out_channels, vb.pp("proj"));
    let stride = 4;
    let mlp = hiera_mlp(out_channels, out_channels * 4, vb.pp("mlp"))?;
    let attn = hiera_attention(
        in_channels,
        out_channels,
        heads,
        q_stride,
        window_size,
        use_mask_attention,
        vb.pp("attn"),
    )?;

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        let xs_norm = xs.apply_t(&norm1, false)?;
        if let Ok(p) = &proj {
            xs = xs_norm.apply(p)?;
            let (a, _, d) = xs.dims3()?;
            xs = xs.reshape((a, stride, (), d))?.max(1)?;
        }
        let xs = (xs + &xs_norm.apply(&attn)?)?;

        let xs = (&xs + &xs.apply_t(&norm2, false)?.apply(&mlp)?)?;

        Ok(xs)
    }))
}

fn hiera_blocks(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    let nblocks = cfg.stages.iter().sum();
    let mut blocks = Vec::with_capacity(nblocks);

    let mut out_channels = cfg.channels;
    let mut in_channels = out_channels;
    let mut heads = cfg.heads;
    let mut b = 0;

    let mut q_stride = 1;
    let mut window_size = 64;

    for s in 0..4 {
        let use_mask_attention = s < 2;

        for _ in 0..cfg.stages[s] {
            blocks.push(hiera_block(
                heads,
                in_channels,
                out_channels,
                q_stride,
                window_size,
                use_mask_attention,
                vb.pp(b),
            )?);
            b += 1;
            in_channels = out_channels;
            q_stride = 1;
        }
        q_stride = 4;
        out_channels *= 2;
        heads *= 2;
        window_size /= 4;
    }

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for block in blocks.iter() {
            xs = xs.apply(block)?
        }
        Ok(xs)
    }))
}

fn hiera_head(outputs: usize, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let norm = layer_norm(outputs, 1e-6, vb.pp("norm"))?;
    let linear = linear(outputs, nclasses, vb.pp("fc"))?;
    Ok(Func::new(move |xs| {
        xs.apply_t(&norm, false)?.apply(&linear)
    }))
}

// Build a hiera model for a given configuration.
fn hiera_model(cfg: &Config, nclasses: Option<usize>, vb: VarBuilder) -> Result<Func<'static>> {
    let cls = match nclasses {
        None => None,
        Some(nclasses) => {
            let outputs = cfg.channels * 8;
            let head = hiera_head(outputs, nclasses, vb.pp("head"))?;
            Some(head)
        }
    };

    let embeddings = hiera_embeddings(cfg.channels, vb.clone())?;
    let unroll = hiera_unroll()?;
    let blocks = hiera_blocks(cfg, vb.pp("blocks"))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&embeddings)?
            .apply(&unroll)?
            .apply(&blocks)?
            .mean(1)?;
        match &cls {
            None => Ok(xs),
            Some(cls) => xs.apply(cls),
        }
    }))
}

pub fn hiera(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    hiera_model(cfg, Some(nclasses), vb)
}

pub fn hiera_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    hiera_model(cfg, None, vb)
}
