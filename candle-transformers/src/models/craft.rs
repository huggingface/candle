//! CRAFT model implementation.
//!
//! See Character Region Awareness for Text Detection
//! <https://arxiv.org/abs/1904.01941>
//! <https://github.com/clovaai/CRAFT-pytorch>
use candle::{Module, ModuleT, Result, Tensor};
use candle_nn::{seq, BatchNormConfig, Sequential, VarBuilder};

use super::vgg::{Models, Vgg};

pub struct Craft<'a> {
    basenet: Vgg<'a>, // Vgg16 specifically
    upconvs: [Sequential; 4],
    conv_class: Sequential,
}

impl<'a> Craft<'a> {
    /// Constructs a new CRAFT model
    ///
    /// # Arguments
    ///
    /// * `vb` - Variable Builder for the CRAFT model
    /// * `vgg_vb` - Variable Builder for the inner Vgg16 model
    pub fn new(vb: VarBuilder<'a>, vgg_vb: VarBuilder<'a>) -> Result<Self> {
        // Ok(Self { blocks })
        let basenet = Vgg::new(vgg_vb, Models::Vgg16)?;
        let upconvs = [
            double_conv(1024, 512, 256, "upconv1", &vb)?,
            double_conv(512, 256, 128, "upconv2", &vb)?,
            double_conv(256, 128, 64, "upconv2", &vb)?,
            double_conv(128, 64, 32, "upconv3", &vb)?,
        ];
        let conv_class = build_conv_class("conv_cls", &vb)?;

        Ok(Self {
            basenet,
            upconvs,
            conv_class,
        })
    }
}

impl Module for Craft<'_> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // sources = self.basenet(x)
        let sources = self.basenet.forward_t(xs, false)?;

        /*
         * y = torch.cat([sources[0], sources[1]], dim=1)
         * y = self.upconv1(y)
         */
        let mut xs = Tensor::cat(&[sources.get(0)?, sources.get(1)?], 1)?;
        xs = self.upconvs[0].forward(&xs)?;

        for i in 1..self.upconvs.len() {
            /*
             * y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
             * y = torch.cat([y, sources[2]], dim=1)
             * y = self.upconv2(y)
             */
            let source_layer = sources.get(i + 1)?;
            xs = xs.interpolate2d(source_layer.dim(2)?, source_layer.dim(3)?)?;
            xs = self.upconvs[i].forward(&xs)?;
        }

        // y = self.conv_cls(feature)
        xs = self.conv_class.forward(&xs)?;

        // return y.permute(0,2,3,1)
        xs.permute([0, 2, 3, 1])
    }
}

fn double_conv(
    in_c: usize,
    mid_c: usize,
    out_c: usize,
    name: &str,
    vb: &VarBuilder,
) -> Result<Sequential> {
    let seq = seq();

    let seq = seq.add(candle_nn::conv2d(
        in_c + mid_c,
        mid_c,
        1,
        candle_nn::Conv2dConfig {
            ..Default::default()
        },
        vb.pp(name),
    )?);

    let batch = candle_nn::batch_norm(mid_c, BatchNormConfig::default(), vb.pp(name))?;
    let seq = seq.add(move |xs: &_| batch.forward_t(xs, false));

    let seq = seq.add(candle_nn::Activation::Relu);
    let seq = seq.add(candle_nn::conv2d(
        mid_c,
        out_c,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp(name),
    )?);

    let batch = candle_nn::batch_norm(out_c, BatchNormConfig::default(), vb.pp(name))?;
    let seq = seq.add(move |xs: &_| batch.forward_t(xs, false));

    let seq = seq.add(candle_nn::Activation::Relu);
    Ok(seq)
}

fn build_conv_class(name: &str, vb: &VarBuilder) -> Result<Sequential> {
    let num_class = 2;

    let seq = seq();

    let seq = seq.add(candle_nn::conv2d(
        32,
        32,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp(name),
    )?);
    let seq = seq.add(candle_nn::Activation::Relu);

    let seq = seq.add(candle_nn::conv2d(
        32,
        32,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp(name),
    )?);
    let seq = seq.add(candle_nn::Activation::Relu);

    let seq = seq.add(candle_nn::conv2d(
        32,
        16,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp(name),
    )?);
    let seq = seq.add(candle_nn::Activation::Relu);

    let seq = seq.add(candle_nn::conv2d(
        16,
        num_class,
        1,
        candle_nn::Conv2dConfig {
            ..Default::default()
        },
        vb.pp(name),
    )?);
    let seq = seq.add(candle_nn::Activation::Relu);

    Ok(seq)
}
