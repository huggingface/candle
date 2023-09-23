//! 2D UNet Building Blocks
//!
use super::attention::{
    AttentionBlock, AttentionBlockConfig, SpatialTransformer, SpatialTransformerConfig,
};
use super::resnet::{ResnetBlock2D, ResnetBlock2DConfig};
use crate::models::with_tracing::{conv2d, Conv2d};
use candle::{Module, Result, Tensor, D};
use candle_nn as nn;

#[derive(Debug)]
struct Downsample2D {
    conv: Option<Conv2d>,
    padding: usize,
    span: tracing::Span,
}

impl Downsample2D {
    fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        use_conv: bool,
        out_channels: usize,
        padding: usize,
    ) -> Result<Self> {
        let conv = if use_conv {
            let config = nn::Conv2dConfig {
                stride: 2,
                padding,
                ..Default::default()
            };
            let conv = conv2d(in_channels, out_channels, 3, config, vs.pp("conv"))?;
            Some(conv)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "downsample2d");
        Ok(Self {
            conv,
            padding,
            span,
        })
    }
}

impl Module for Downsample2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match &self.conv {
            None => xs.avg_pool2d(2),
            Some(conv) => {
                if self.padding == 0 {
                    let xs = xs
                        .pad_with_zeros(D::Minus1, 0, 1)?
                        .pad_with_zeros(D::Minus2, 0, 1)?;
                    conv.forward(&xs)
                } else {
                    conv.forward(xs)
                }
            }
        }
    }
}

// This does not support the conv-transpose mode.
#[derive(Debug)]
struct Upsample2D {
    conv: Conv2d,
    span: tracing::Span,
}

impl Upsample2D {
    fn new(vs: nn::VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let config = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(in_channels, out_channels, 3, config, vs.pp("conv"))?;
        let span = tracing::span!(tracing::Level::TRACE, "upsample2d");
        Ok(Self { conv, span })
    }
}

impl Upsample2D {
    fn forward(&self, xs: &Tensor, size: Option<(usize, usize)>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = match size {
            None => {
                let (_bsize, _channels, h, w) = xs.dims4()?;
                xs.upsample_nearest2d(2 * h, 2 * w)?
            }
            Some((h, w)) => xs.upsample_nearest2d(h, w)?,
        };
        self.conv.forward(&xs)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DownEncoderBlock2DConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_downsample: bool,
    pub downsample_padding: usize,
}

impl Default for DownEncoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
pub struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    span: tracing::Span,
    pub config: DownEncoderBlock2DConfig,
}

impl DownEncoderBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: DownEncoderBlock2DConfig,
    ) -> Result<Self> {
        let resnets: Vec<_> = {
            let vs = vs.pp("resnets");
            let conv_cfg = ResnetBlock2DConfig {
                eps: config.resnet_eps,
                out_channels: Some(out_channels),
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                temb_channels: None,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(vs.pp(&i.to_string()), in_channels, conv_cfg)
                })
                .collect::<Result<Vec<_>>>()?
        };
        let downsampler = if config.add_downsample {
            let downsample = Downsample2D::new(
                vs.pp("downsamplers").pp("0"),
                out_channels,
                true,
                out_channels,
                config.downsample_padding,
            )?;
            Some(downsample)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "down-enc2d");
        Ok(Self {
            resnets,
            downsampler,
            span,
            config,
        })
    }
}

impl Module for DownEncoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, None)?
        }
        match &self.downsampler {
            Some(downsampler) => downsampler.forward(&xs),
            None => Ok(xs),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UpDecoderBlock2DConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_upsample: bool,
}

impl Default for UpDecoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
pub struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    span: tracing::Span,
    pub config: UpDecoderBlock2DConfig,
}

impl UpDecoderBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: UpDecoderBlock2DConfig,
    ) -> Result<Self> {
        let resnets: Vec<_> = {
            let vs = vs.pp("resnets");
            let conv_cfg = ResnetBlock2DConfig {
                out_channels: Some(out_channels),
                eps: config.resnet_eps,
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                temb_channels: None,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(vs.pp(&i.to_string()), in_channels, conv_cfg)
                })
                .collect::<Result<Vec<_>>>()?
        };
        let upsampler = if config.add_upsample {
            let upsample =
                Upsample2D::new(vs.pp("upsamplers").pp("0"), out_channels, out_channels)?;
            Some(upsample)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "up-dec2d");
        Ok(Self {
            resnets,
            upsampler,
            span,
            config,
        })
    }
}

impl Module for UpDecoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, None)?
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&xs, None),
            None => Ok(xs),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UNetMidBlock2DConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: Option<usize>,
    pub attn_num_head_channels: Option<usize>,
    // attention_type "default"
    pub output_scale_factor: f64,
}

impl Default for UNetMidBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: Some(1),
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct UNetMidBlock2D {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(AttentionBlock, ResnetBlock2D)>,
    span: tracing::Span,
    pub config: UNetMidBlock2DConfig,
}

impl UNetMidBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        temb_channels: Option<usize>,
        config: UNetMidBlock2DConfig,
    ) -> Result<Self> {
        let vs_resnets = vs.pp("resnets");
        let vs_attns = vs.pp("attentions");
        let resnet_groups = config
            .resnet_groups
            .unwrap_or_else(|| usize::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(vs_resnets.pp("0"), in_channels, resnet_cfg)?;
        let attn_cfg = AttentionBlockConfig {
            num_head_channels: config.attn_num_head_channels,
            num_groups: resnet_groups,
            rescale_output_factor: config.output_scale_factor,
            eps: config.resnet_eps,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = AttentionBlock::new(vs_attns.pp(&index.to_string()), in_channels, attn_cfg)?;
            let resnet = ResnetBlock2D::new(
                vs_resnets.pp(&(index + 1).to_string()),
                in_channels,
                resnet_cfg,
            )?;
            attn_resnets.push((attn, resnet))
        }
        let span = tracing::span!(tracing::Level::TRACE, "mid2d");
        Ok(Self {
            resnet,
            attn_resnets,
            span,
            config,
        })
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = self.resnet.forward(xs, temb)?;
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&attn.forward(&xs)?, temb)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UNetMidBlock2DCrossAttnConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: Option<usize>,
    pub attn_num_head_channels: usize,
    // attention_type "default"
    pub output_scale_factor: f64,
    pub cross_attn_dim: usize,
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
    pub transformer_layers_per_block: usize,
}

impl Default for UNetMidBlock2DCrossAttnConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: 1,
            output_scale_factor: 1.,
            cross_attn_dim: 1280,
            sliced_attention_size: None, // Sliced attention disabled
            use_linear_projection: false,
            transformer_layers_per_block: 1,
        }
    }
}

#[derive(Debug)]
pub struct UNetMidBlock2DCrossAttn {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(SpatialTransformer, ResnetBlock2D)>,
    span: tracing::Span,
    pub config: UNetMidBlock2DCrossAttnConfig,
}

impl UNetMidBlock2DCrossAttn {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        temb_channels: Option<usize>,
        use_flash_attn: bool,
        config: UNetMidBlock2DCrossAttnConfig,
    ) -> Result<Self> {
        let vs_resnets = vs.pp("resnets");
        let vs_attns = vs.pp("attentions");
        let resnet_groups = config
            .resnet_groups
            .unwrap_or_else(|| usize::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(vs_resnets.pp("0"), in_channels, resnet_cfg)?;
        let n_heads = config.attn_num_head_channels;
        let attn_cfg = SpatialTransformerConfig {
            depth: config.transformer_layers_per_block,
            num_groups: resnet_groups,
            context_dim: Some(config.cross_attn_dim),
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = SpatialTransformer::new(
                vs_attns.pp(&index.to_string()),
                in_channels,
                n_heads,
                in_channels / n_heads,
                use_flash_attn,
                attn_cfg,
            )?;
            let resnet = ResnetBlock2D::new(
                vs_resnets.pp(&(index + 1).to_string()),
                in_channels,
                resnet_cfg,
            )?;
            attn_resnets.push((attn, resnet))
        }
        let span = tracing::span!(tracing::Level::TRACE, "xa-mid2d");
        Ok(Self {
            resnet,
            attn_resnets,
            span,
            config,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = self.resnet.forward(xs, temb)?;
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&attn.forward(&xs, encoder_hidden_states)?, temb)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DownBlock2DConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_downsample: bool,
    pub downsample_padding: usize,
}

impl Default for DownBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
pub struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    span: tracing::Span,
    pub config: DownBlock2DConfig,
}

impl DownBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
        config: DownBlock2DConfig,
    ) -> Result<Self> {
        let vs_resnets = vs.pp("resnets");
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(out_channels),
            eps: config.resnet_eps,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnets = (0..config.num_layers)
            .map(|i| {
                let in_channels = if i == 0 { in_channels } else { out_channels };
                ResnetBlock2D::new(vs_resnets.pp(&i.to_string()), in_channels, resnet_cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let downsampler = if config.add_downsample {
            let downsampler = Downsample2D::new(
                vs.pp("downsamplers").pp("0"),
                out_channels,
                true,
                out_channels,
                config.downsample_padding,
            )?;
            Some(downsampler)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "down2d");
        Ok(Self {
            resnets,
            downsampler,
            span,
            config,
        })
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<(Tensor, Vec<Tensor>)> {
        let _enter = self.span.enter();
        let mut xs = xs.clone();
        let mut output_states = vec![];
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, temb)?;
            output_states.push(xs.clone());
        }
        let xs = match &self.downsampler {
            Some(downsampler) => {
                let xs = downsampler.forward(&xs)?;
                output_states.push(xs.clone());
                xs
            }
            None => xs,
        };
        Ok((xs, output_states))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CrossAttnDownBlock2DConfig {
    pub downblock: DownBlock2DConfig,
    pub attn_num_head_channels: usize,
    pub cross_attention_dim: usize,
    // attention_type: "default"
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
    pub transformer_layers_per_block: usize,
}

impl Default for CrossAttnDownBlock2DConfig {
    fn default() -> Self {
        Self {
            downblock: Default::default(),
            attn_num_head_channels: 1,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
            transformer_layers_per_block: 1,
        }
    }
}

#[derive(Debug)]
pub struct CrossAttnDownBlock2D {
    downblock: DownBlock2D,
    attentions: Vec<SpatialTransformer>,
    span: tracing::Span,
    pub config: CrossAttnDownBlock2DConfig,
}

impl CrossAttnDownBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
        use_flash_attn: bool,
        config: CrossAttnDownBlock2DConfig,
    ) -> Result<Self> {
        let downblock = DownBlock2D::new(
            vs.clone(),
            in_channels,
            out_channels,
            temb_channels,
            config.downblock,
        )?;
        let n_heads = config.attn_num_head_channels;
        let cfg = SpatialTransformerConfig {
            depth: config.transformer_layers_per_block,
            context_dim: Some(config.cross_attention_dim),
            num_groups: config.downblock.resnet_groups,
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let vs_attn = vs.pp("attentions");
        let attentions = (0..config.downblock.num_layers)
            .map(|i| {
                SpatialTransformer::new(
                    vs_attn.pp(&i.to_string()),
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    use_flash_attn,
                    cfg,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "xa-down2d");
        Ok(Self {
            downblock,
            attentions,
            span,
            config,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let _enter = self.span.enter();
        let mut output_states = vec![];
        let mut xs = xs.clone();
        for (resnet, attn) in self.downblock.resnets.iter().zip(self.attentions.iter()) {
            xs = resnet.forward(&xs, temb)?;
            xs = attn.forward(&xs, encoder_hidden_states)?;
            output_states.push(xs.clone());
        }
        let xs = match &self.downblock.downsampler {
            Some(downsampler) => {
                let xs = downsampler.forward(&xs)?;
                output_states.push(xs.clone());
                xs
            }
            None => xs,
        };
        Ok((xs, output_states))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UpBlock2DConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_upsample: bool,
}

impl Default for UpBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
pub struct UpBlock2D {
    pub resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    span: tracing::Span,
    pub config: UpBlock2DConfig,
}

impl UpBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        prev_output_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
        config: UpBlock2DConfig,
    ) -> Result<Self> {
        let vs_resnets = vs.pp("resnets");
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(out_channels),
            temb_channels,
            eps: config.resnet_eps,
            output_scale_factor: config.output_scale_factor,
            ..Default::default()
        };
        let resnets = (0..config.num_layers)
            .map(|i| {
                let res_skip_channels = if i == config.num_layers - 1 {
                    in_channels
                } else {
                    out_channels
                };
                let resnet_in_channels = if i == 0 {
                    prev_output_channels
                } else {
                    out_channels
                };
                let in_channels = resnet_in_channels + res_skip_channels;
                ResnetBlock2D::new(vs_resnets.pp(&i.to_string()), in_channels, resnet_cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let upsampler = if config.add_upsample {
            let upsampler =
                Upsample2D::new(vs.pp("upsamplers").pp("0"), out_channels, out_channels)?;
            Some(upsampler)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "up2d");
        Ok(Self {
            resnets,
            upsampler,
            span,
            config,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        res_xs: &[Tensor],
        temb: Option<&Tensor>,
        upsample_size: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.clone();
        for (index, resnet) in self.resnets.iter().enumerate() {
            xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1)?;
            xs = xs.contiguous()?;
            xs = resnet.forward(&xs, temb)?;
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&xs, upsample_size),
            None => Ok(xs),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CrossAttnUpBlock2DConfig {
    pub upblock: UpBlock2DConfig,
    pub attn_num_head_channels: usize,
    pub cross_attention_dim: usize,
    // attention_type: "default"
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
    pub transformer_layers_per_block: usize,
}

impl Default for CrossAttnUpBlock2DConfig {
    fn default() -> Self {
        Self {
            upblock: Default::default(),
            attn_num_head_channels: 1,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
            transformer_layers_per_block: 1,
        }
    }
}

#[derive(Debug)]
pub struct CrossAttnUpBlock2D {
    pub upblock: UpBlock2D,
    pub attentions: Vec<SpatialTransformer>,
    span: tracing::Span,
    pub config: CrossAttnUpBlock2DConfig,
}

impl CrossAttnUpBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        prev_output_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
        use_flash_attn: bool,
        config: CrossAttnUpBlock2DConfig,
    ) -> Result<Self> {
        let upblock = UpBlock2D::new(
            vs.clone(),
            in_channels,
            prev_output_channels,
            out_channels,
            temb_channels,
            config.upblock,
        )?;
        let n_heads = config.attn_num_head_channels;
        let cfg = SpatialTransformerConfig {
            depth: config.transformer_layers_per_block,
            context_dim: Some(config.cross_attention_dim),
            num_groups: config.upblock.resnet_groups,
            sliced_attention_size: config.sliced_attention_size,
            use_linear_projection: config.use_linear_projection,
        };
        let vs_attn = vs.pp("attentions");
        let attentions = (0..config.upblock.num_layers)
            .map(|i| {
                SpatialTransformer::new(
                    vs_attn.pp(&i.to_string()),
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    use_flash_attn,
                    cfg,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "xa-up2d");
        Ok(Self {
            upblock,
            attentions,
            span,
            config,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        res_xs: &[Tensor],
        temb: Option<&Tensor>,
        upsample_size: Option<(usize, usize)>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.clone();
        for (index, resnet) in self.upblock.resnets.iter().enumerate() {
            xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1)?;
            xs = xs.contiguous()?;
            xs = resnet.forward(&xs, temb)?;
            xs = self.attentions[index].forward(&xs, encoder_hidden_states)?;
        }
        match &self.upblock.upsampler {
            Some(upsampler) => upsampler.forward(&xs, upsample_size),
            None => Ok(xs),
        }
    }
}
