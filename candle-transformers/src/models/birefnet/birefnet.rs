//! BiRefNet main model implementation
//!
//! BiRefNet (Bilateral Reference Network) for high-resolution image segmentation.

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::blocks::BasicDecBlk;
use super::config::{BackboneType, Config, MultiScaleInputType, SqueezeBlockType};
use super::decoder::Decoder;
use super::swin::SwinTransformer;

/// BiRefNet model for image segmentation
#[derive(Debug, Clone)]
pub struct BiRefNet {
    config: Config,
    backbone: SwinTransformer,
    squeeze_module: Option<BasicDecBlk>,
    decoder: Decoder,
}

impl BiRefNet {
    /// Create a new BiRefNet model
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        // Create backbone
        let backbone = match config.backbone {
            BackboneType::SwinV1L => SwinTransformer::swin_v1_l(vb.pp("bb"))?,
            BackboneType::SwinV1B => SwinTransformer::swin_v1_b(vb.pp("bb"))?,
        };

        // Calculate channels (doubled when mul_scl_ipt = "cat")
        let channels: Vec<usize> = match config.mul_scl_ipt {
            MultiScaleInputType::Cat => config.lateral_channels.iter().map(|c| c * 2).collect(),
            _ => config.lateral_channels.clone(),
        };

        // Add context channels
        let squeeze_in_channels = if config.cxt_num > 0 {
            let cxt: Vec<usize> = channels[1..]
                .iter()
                .rev()
                .take(config.cxt_num)
                .cloned()
                .collect();
            channels[0] + cxt.iter().sum::<usize>()
        } else {
            channels[0]
        };

        // Squeeze module
        let squeeze_module = match config.squeeze_block {
            SqueezeBlockType::BasicDecBlkX1 => Some(BasicDecBlk::new(
                squeeze_in_channels,
                channels[0],
                config.dec_att, // Use config's dec_att instead of None
                true,
                vb.pp("squeeze_module.0"),
            )?),
            SqueezeBlockType::None => None,
        };

        // Decoder
        let decoder = Decoder::new(&channels, &config, vb.pp("decoder"))?;

        Ok(Self {
            config,
            backbone,
            squeeze_module,
            decoder,
        })
    }

    /// Load model from safetensors file
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let config = Config::default();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
        Self::new(config, vb)
    }

    /// Encoder forward pass
    fn forward_enc(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let features = self.backbone.forward_features(x)?;
        let (mut x1, mut x2, mut x3, mut x4) = (
            features[0].clone(),
            features[1].clone(),
            features[2].clone(),
            features[3].clone(),
        );

        // Multi-scale input processing
        match self.config.mul_scl_ipt {
            MultiScaleInputType::Cat => {
                let (_, _, h, w) = x.dims4()?;
                let x_half = x.upsample_bilinear2d(h / 2, w / 2, true)?;
                let features_half = self.backbone.forward_features(&x_half)?;

                let (_, _, h1, w1) = x1.dims4()?;
                let (_, _, h2, w2) = x2.dims4()?;
                let (_, _, h3, w3) = x3.dims4()?;
                let (_, _, h4, w4) = x4.dims4()?;

                x1 = Tensor::cat(
                    &[x1, features_half[0].upsample_bilinear2d(h1, w1, true)?],
                    1,
                )?;
                x2 = Tensor::cat(
                    &[x2, features_half[1].upsample_bilinear2d(h2, w2, true)?],
                    1,
                )?;
                x3 = Tensor::cat(
                    &[x3, features_half[2].upsample_bilinear2d(h3, w3, true)?],
                    1,
                )?;
                x4 = Tensor::cat(
                    &[x4, features_half[3].upsample_bilinear2d(h4, w4, true)?],
                    1,
                )?;
            }
            MultiScaleInputType::Add => {
                let (_, _, h, w) = x.dims4()?;
                let x_half = x.upsample_bilinear2d(h / 2, w / 2, true)?;
                let features_half = self.backbone.forward_features(&x_half)?;

                let (_, _, h1, w1) = x1.dims4()?;
                let (_, _, h2, w2) = x2.dims4()?;
                let (_, _, h3, w3) = x3.dims4()?;
                let (_, _, h4, w4) = x4.dims4()?;

                x1 = (x1 + features_half[0].upsample_bilinear2d(h1, w1, true)?)?;
                x2 = (x2 + features_half[1].upsample_bilinear2d(h2, w2, true)?)?;
                x3 = (x3 + features_half[2].upsample_bilinear2d(h3, w3, true)?)?;
                x4 = (x4 + features_half[3].upsample_bilinear2d(h4, w4, true)?)?;
            }
            MultiScaleInputType::None => {}
        }

        // Context aggregation
        if self.config.cxt_num > 0 {
            let (_, _, h4, w4) = x4.dims4()?;
            let mut cxt_features = vec![x4.clone()];

            let cxt_sources = [&x1, &x2, &x3];
            for src in cxt_sources.iter().rev().take(self.config.cxt_num) {
                cxt_features.insert(0, src.upsample_bilinear2d(h4, w4, true)?);
            }

            x4 = Tensor::cat(&cxt_features, 1)?;
        }

        Ok((x1, x2, x3, x4))
    }
}

impl Module for BiRefNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Encode
        let (x1, x2, x3, mut x4) = self.forward_enc(x)?;

        // Squeeze
        if let Some(squeeze) = &self.squeeze_module {
            x4 = squeeze.forward(&x4)?;
        }

        // Decode
        let features = vec![x.clone(), x1, x2, x3, x4];
        let outputs = self.decoder.forward(&features)?;

        // Return the last output (highest resolution)
        outputs
            .last()
            .cloned()
            .ok_or_else(|| candle::Error::Msg("Decoder produced no outputs".to_string()))
    }
}
