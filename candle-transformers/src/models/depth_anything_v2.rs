//! Implementation of the Depth Anything model from FAIR.
//!
//! See:
//! - ["Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"](https://github.com/LiheYoung/Depth-Anything)
//!

use std::sync::Arc;

use candle::D::Minus1;
use candle::{Module, Result, Tensor};
use candle_nn::ops::Identity;
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, conv_transpose2d, linear, seq, Activation, BatchNorm,
    BatchNormConfig, Conv2d, Conv2dConfig, ConvTranspose2dConfig, Sequential, VarBuilder,
};

use crate::models::dinov2::DinoVisionTransformer;

pub struct DepthAnythingV2Config {
    out_channel_sizes: [usize; 4],
    in_channel_size: usize, // embed_dim in the Dino model
    num_features: usize,
    use_batch_norm: bool,
    use_class_token: bool,
    layer_ids_vits: Vec<usize>,
    input_image_size: usize,
    target_patch_size: usize,
}

impl DepthAnythingV2Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        out_channel_sizes: [usize; 4],
        in_channel_size: usize,
        num_features: usize,
        use_batch_norm: bool,
        use_class_token: bool,
        layer_ids_vits: Vec<usize>,
        input_image_size: usize,
        target_patch_size: usize,
    ) -> Self {
        Self {
            out_channel_sizes,
            in_channel_size,
            num_features,
            use_batch_norm,
            use_class_token,
            layer_ids_vits,
            input_image_size,
            target_patch_size,
        }
    }

    pub fn vit_small() -> Self {
        Self {
            out_channel_sizes: [48, 96, 192, 384],
            in_channel_size: 384,
            num_features: 64,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![2, 5, 8, 11],
            input_image_size: 518,
            target_patch_size: 518 / 14,
        }
    }

    pub fn vit_base() -> Self {
        Self {
            out_channel_sizes: [96, 192, 384, 768],
            in_channel_size: 768,
            num_features: 128,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![2, 5, 8, 11],
            input_image_size: 518,
            target_patch_size: 518 / 14,
        }
    }

    pub fn vit_large() -> Self {
        Self {
            out_channel_sizes: [256, 512, 1024, 1024],
            in_channel_size: 1024,
            num_features: 256,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![4, 11, 17, 23],
            input_image_size: 518,
            target_patch_size: 518 / 14,
        }
    }

    pub fn vit_giant() -> Self {
        Self {
            out_channel_sizes: [1536, 1536, 1536, 1536],
            in_channel_size: 1536,
            num_features: 384,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![9, 19, 29, 39],
            input_image_size: 518,
            target_patch_size: 518 / 14,
        }
    }
}

pub struct ResidualConvUnit {
    activation: Activation,
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: Option<BatchNorm>,
    batch_norm2: Option<BatchNorm>,
}

impl ResidualConvUnit {
    pub fn new(
        conf: &DepthAnythingV2Config,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        const KERNEL_SIZE: usize = 3;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv1 = conv2d(
            conf.num_features,
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("conv1"),
        )?;
        let conv2 = conv2d(
            conf.num_features,
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("conv2"),
        )?;

        let (batch_norm1, batch_norm2) = match conf.use_batch_norm {
            true => {
                let batch_norm_cfg = BatchNormConfig {
                    eps: 1e-05,
                    remove_mean: false,
                    affine: true,
                    momentum: 0.1,
                };
                (
                    Some(batch_norm(conf.num_features, batch_norm_cfg, vb.pp("bn1"))?),
                    Some(batch_norm(conf.num_features, batch_norm_cfg, vb.pp("bn2"))?),
                )
            }
            false => (None, None),
        };

        Ok(Self {
            activation,
            conv1,
            conv2,
            batch_norm1,
            batch_norm2,
        })
    }
}

impl Module for ResidualConvUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.activation.forward(xs)?;
        let out = self.conv1.forward(&out)?;
        let out = if let Some(batch_norm1) = &self.batch_norm1 {
            batch_norm1.forward_train(&out)?
        } else {
            out
        };

        let out = self.activation.forward(&out)?;
        let out = self.conv2.forward(&out)?;
        let out = if let Some(batch_norm2) = &self.batch_norm2 {
            batch_norm2.forward_train(&out)?
        } else {
            out
        };

        out + xs
    }
}

pub struct FeatureFusionBlock {
    res_conv_unit1: ResidualConvUnit,
    res_conv_unit2: ResidualConvUnit,
    output_conv: Conv2d,
    target_patch_size: usize,
}

impl FeatureFusionBlock {
    pub fn new(
        conf: &DepthAnythingV2Config,
        target_patch_size: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        const KERNEL_SIZE: usize = 1;
        let conv_cfg = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let output_conv = conv2d(
            conf.num_features,
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("out_conv"),
        )?;
        let res_conv_unit1 = ResidualConvUnit::new(conf, activation, vb.pp("resConfUnit1"))?;
        let res_conv_unit2 = ResidualConvUnit::new(conf, activation, vb.pp("resConfUnit2"))?;

        Ok(Self {
            res_conv_unit1,
            res_conv_unit2,
            output_conv,
            target_patch_size,
        })
    }
}

impl Module for FeatureFusionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.res_conv_unit2.forward(xs)?;
        let out = out.interpolate2d(self.target_patch_size, self.target_patch_size)?;

        self.output_conv.forward(&out)
    }
}

pub struct Scratch {
    layer1_rn: Conv2d,
    layer2_rn: Conv2d,
    layer3_rn: Conv2d,
    layer4_rn: Conv2d,
    refine_net1: FeatureFusionBlock,
    refine_net2: FeatureFusionBlock,
    refine_net3: FeatureFusionBlock,
    refine_net4: FeatureFusionBlock,
    output_conv1: Conv2d,
    output_conv2: Sequential,
}

impl Scratch {
    pub fn new(conf: &DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        const KERNEL_SIZE: usize = 3;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        let layer1_rn = conv2d_no_bias(
            conf.out_channel_sizes[0],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer1_rn"),
        )?;
        let layer2_rn = conv2d_no_bias(
            conf.out_channel_sizes[1],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer2_rn"),
        )?;
        let layer3_rn = conv2d_no_bias(
            conf.out_channel_sizes[2],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer3_rn"),
        )?;
        let layer4_rn = conv2d_no_bias(
            conf.out_channel_sizes[3],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer4_rn"),
        )?;

        let refine_net1 = FeatureFusionBlock::new(
            conf,
            conf.target_patch_size * 8,
            Activation::Relu,
            vb.pp("refinenet1"),
        )?;
        let refine_net2 = FeatureFusionBlock::new(
            conf,
            conf.target_patch_size * 4,
            Activation::Relu,
            vb.pp("refinenet2"),
        )?;
        let refine_net3 = FeatureFusionBlock::new(
            conf,
            conf.target_patch_size * 2,
            Activation::Relu,
            vb.pp("refinenet3"),
        )?;
        let refine_net4 = FeatureFusionBlock::new(
            conf,
            conf.target_patch_size,
            Activation::Relu,
            vb.pp("refinenet4"),
        )?;

        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let output_conv1 = conv2d(
            conf.num_features,
            conf.num_features / 2,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("output_conv1"),
        )?;

        let output_conv2 = seq();
        const HEAD_FEATURES_2: usize = 32;
        const OUT_CHANNELS_2: usize = 1;
        const KERNEL_SIZE_2: usize = 1;
        let output_conv2 = output_conv2.add(conv2d(
            conf.num_features / 2,
            HEAD_FEATURES_2,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("output_conv2").pp("0"),
        )?);
        let output_conv2 = output_conv2
            .add(Activation::Relu)
            .add(conv2d(
                HEAD_FEATURES_2,
                OUT_CHANNELS_2,
                KERNEL_SIZE_2,
                conv_cfg,
                vb.pp("output_conv2").pp("2"),
            )?)
            .add(Activation::Relu);

        Ok(Self {
            layer1_rn,
            layer2_rn,
            layer3_rn,
            layer4_rn,
            refine_net1,
            refine_net2,
            refine_net3,
            refine_net4,
            output_conv1,
            output_conv2,
        })
    }
}

const NUM_CHANNELS: usize = 4;

pub struct DPTHead {
    projections: Vec<Conv2d>,
    resize_layers: Vec<Box<dyn Module>>,
    readout_projections: Vec<Sequential>,
    scratch: Scratch,
    use_class_token: bool,
    input_image_size: usize,
    target_patch_size: usize,
}

impl DPTHead {
    pub fn new(conf: &DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        let mut projections: Vec<Conv2d> = Vec::with_capacity(conf.out_channel_sizes.len());
        for (conv_index, out_channel_size) in conf.out_channel_sizes.iter().enumerate() {
            projections.push(conv2d(
                conf.in_channel_size,
                *out_channel_size,
                1,
                Default::default(),
                vb.pp("projects").pp(conv_index.to_string()),
            )?);
        }

        let resize_layers: Vec<Box<dyn Module>> = vec![
            Box::new(conv_transpose2d(
                conf.out_channel_sizes[0],
                conf.out_channel_sizes[0],
                4,
                ConvTranspose2dConfig {
                    padding: 0,
                    stride: 4,
                    dilation: 1,
                    output_padding: 0,
                },
                vb.pp("resize_layers").pp("0"),
            )?),
            Box::new(conv_transpose2d(
                conf.out_channel_sizes[1],
                conf.out_channel_sizes[1],
                2,
                ConvTranspose2dConfig {
                    padding: 0,
                    stride: 2,
                    dilation: 1,
                    output_padding: 0,
                },
                vb.pp("resize_layers").pp("1"),
            )?),
            Box::new(Identity::new()),
            Box::new(conv2d(
                conf.out_channel_sizes[3],
                conf.out_channel_sizes[3],
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("resize_layers").pp("3"),
            )?),
        ];

        let readout_projections = if conf.use_class_token {
            let rop = Vec::with_capacity(NUM_CHANNELS);
            for rop_index in 0..NUM_CHANNELS {
                seq()
                    .add(linear(
                        2 * conf.in_channel_size,
                        conf.in_channel_size,
                        vb.pp("readout_projects").pp(rop_index.to_string()),
                    )?)
                    .add(Activation::Gelu);
            }
            rop
        } else {
            vec![]
        };

        let scratch = Scratch::new(conf, vb.pp("scratch"))?;

        Ok(Self {
            projections,
            resize_layers,
            readout_projections,
            scratch,
            use_class_token: conf.use_class_token,
            input_image_size: conf.input_image_size,
            target_patch_size: conf.target_patch_size,
        })
    }
}

impl Module for DPTHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out: Vec<Tensor> = Vec::with_capacity(NUM_CHANNELS);
        for i in 0..NUM_CHANNELS {
            let x = if self.use_class_token {
                let x = xs.get(i)?.get(0)?;
                let class_token = xs.get(i)?.get(1)?;
                let readout = class_token.unsqueeze(1)?.expand(x.shape())?;
                let to_cat = [x, readout];
                let cat = Tensor::cat(&to_cat, Minus1)?;
                self.readout_projections[i].forward(&cat)?
            } else {
                xs.get(i)?
            };
            let x_dims = x.dims();

            let x = x.permute((0, 2, 1))?.reshape((
                x_dims[0],
                x_dims[x_dims.len() - 1],
                self.target_patch_size,
                self.target_patch_size,
            ))?;
            let x = self.projections[i].forward(&x)?;

            let x = self.resize_layers[i].forward(&x)?;
            out.push(x);
        }

        let layer_1_rn = self.scratch.layer1_rn.forward(&out[0])?;
        let layer_2_rn = self.scratch.layer2_rn.forward(&out[1])?;
        let layer_3_rn = self.scratch.layer3_rn.forward(&out[2])?;
        let layer_4_rn = self.scratch.layer4_rn.forward(&out[3])?;

        let path4 = self.scratch.refine_net4.forward(&layer_4_rn)?;

        let res3_out = self
            .scratch
            .refine_net3
            .res_conv_unit1
            .forward(&layer_3_rn)?;
        let res3_out = path4.add(&res3_out)?;
        let path3 = self.scratch.refine_net3.forward(&res3_out)?;

        let res2_out = self
            .scratch
            .refine_net2
            .res_conv_unit1
            .forward(&layer_2_rn)?;
        let res2_out = path3.add(&res2_out)?;
        let path2 = self.scratch.refine_net2.forward(&res2_out)?;

        let res1_out = self
            .scratch
            .refine_net1
            .res_conv_unit1
            .forward(&layer_1_rn)?;
        let res1_out = path2.add(&res1_out)?;
        let path1 = self.scratch.refine_net1.forward(&res1_out)?;

        let out = self.scratch.output_conv1.forward(&path1)?;

        let out = out.interpolate2d(self.input_image_size, self.input_image_size)?;

        self.scratch.output_conv2.forward(&out)
    }
}

pub struct DepthAnythingV2 {
    pretrained: Arc<DinoVisionTransformer>,
    depth_head: DPTHead,
    conf: DepthAnythingV2Config,
}

impl DepthAnythingV2 {
    pub fn new(
        pretrained: Arc<DinoVisionTransformer>,
        conf: DepthAnythingV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depth_head = DPTHead::new(&conf, vb.pp("depth_head"))?;

        Ok(Self {
            pretrained,
            depth_head,
            conf,
        })
    }
}

impl Module for DepthAnythingV2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let features = self.pretrained.get_intermediate_layers(
            xs,
            &self.conf.layer_ids_vits,
            false,
            false,
            true,
        )?;
        let depth = self.depth_head.forward(&features)?;

        depth.relu()
    }
}
