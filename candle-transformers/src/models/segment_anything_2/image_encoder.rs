//! SAM 2 image encoder: a Hiera trunk followed by a simple FPN neck.
use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::hiera::Hiera;

/// Feature pyramid neck. The laterals are 1x1 convolutions mapping every trunk stage to `d_model`
/// channels, and only the levels listed in `fpn_top_down_levels` get a top down contribution.
#[derive(Debug)]
pub struct FpnNeck {
    convs: Vec<candle_nn::Conv2d>,
    fpn_top_down_levels: Vec<usize>,
}

impl FpnNeck {
    pub fn new(
        d_model: usize,
        backbone_channel_list: &[usize],
        fpn_top_down_levels: Vec<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_c = vb.pp("convs");
        let mut convs = Vec::with_capacity(backbone_channel_list.len());
        for (i, &dim) in backbone_channel_list.iter().enumerate() {
            let conv =
                candle_nn::conv2d(dim, d_model, 1, Default::default(), vb_c.pp(i).pp("conv"))?;
            convs.push(conv)
        }
        Ok(Self {
            convs,
            fpn_top_down_levels,
        })
    }

    /// `xs` runs from the finest to the coarsest resolution, and so does the returned pyramid.
    pub fn forward(&self, xs: &[Tensor]) -> Result<Vec<Tensor>> {
        let n = self.convs.len() - 1;
        let mut out = vec![None; self.convs.len()];
        let mut prev: Option<Tensor> = None;
        for i in (0..=n).rev() {
            // `backbone_channel_list` is ordered coarse to fine while `xs` is fine to coarse.
            let lateral = xs[i].apply(&self.convs[n - i])?;
            let cur = match &prev {
                Some(p) if self.fpn_top_down_levels.contains(&i) => {
                    let (_, _, h, w) = p.dims4()?;
                    (lateral + p.upsample_nearest2d(h * 2, w * 2)?)?
                }
                _ => lateral,
            };
            prev = Some(cur.clone());
            out[i] = Some(cur);
        }
        Ok(out.into_iter().map(|x| x.unwrap()).collect())
    }
}

#[derive(Debug)]
pub struct ImageEncoder {
    trunk: Hiera,
    neck: FpnNeck,
    /// Number of coarse levels dropped from the top of the pyramid.
    scalp: usize,
    span: tracing::Span,
}

impl ImageEncoder {
    pub fn new(cfg: &super::Config, vb: VarBuilder) -> Result<Self> {
        let trunk = Hiera::new(cfg, vb.pp("trunk"))?;
        let neck = FpnNeck::new(
            super::PROMPT_EMBED_DIM,
            trunk.channel_list(),
            cfg.fpn_top_down_levels.clone(),
            vb.pp("neck"),
        )?;
        let span = tracing::span!(tracing::Level::TRACE, "image-encoder");
        Ok(Self {
            trunk,
            neck,
            scalp: 1,
            span,
        })
    }

    /// Returns the stride 4, 8 and 16 feature maps, all with `PROMPT_EMBED_DIM` channels.
    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();
        let features = self.neck.forward(&self.trunk.forward(xs)?)?;
        let n = features.len() - self.scalp;
        Ok(features.into_iter().take(n).collect())
    }
}
