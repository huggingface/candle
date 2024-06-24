use candle::D::Minus1;
use candle::{Module, Result, Tensor};
use candle_nn::ops::Identity;
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, conv_transpose2d, linear, seq, Activation, BatchNorm,
    BatchNormConfig, Conv2d, Conv2dConfig, ConvTranspose2dConfig, Sequential, VarBuilder,
};

use crate::models::dinov2::DinoVisionTransformer;

pub struct ResidualConvUnit {
    activation: Activation,
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: Option<BatchNorm>,
    batch_norm2: Option<BatchNorm>,
}

impl ResidualConvUnit {
    pub fn new(
        num_features: usize,
        activation: Activation,
        use_batch_norm: bool,
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
            num_features,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("conv1"),
        )?;
        let conv2 = conv2d(
            num_features,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("conv2"),
        )?;

        let (batch_norm1, batch_norm2) = match use_batch_norm {
            true => {
                let batch_norm_cfg = BatchNormConfig {
                    eps: 1e-05,
                    remove_mean: false,
                    affine: true,
                    momentum: 0.1,
                };
                (
                    Some(batch_norm(num_features, batch_norm_cfg, vb.pp("bn1"))?),
                    Some(batch_norm(num_features, batch_norm_cfg, vb.pp("bn2"))?),
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
        num_features: usize,
        activation: Activation,
        use_batch_norm: bool,
        target_patch_size: usize,
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
            num_features,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("out_conv"),
        )?;
        let res_conv_unit1 = ResidualConvUnit::new(
            num_features,
            activation,
            use_batch_norm,
            vb.pp("resConfUnit1"),
        )?;
        let res_conv_unit2 = ResidualConvUnit::new(
            num_features,
            activation,
            use_batch_norm,
            vb.pp("resConfUnit2"),
        )?;

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
        let out = self.res_conv_unit2.forward(&xs)?;
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
    pub fn new(
        channel_sizes: Vec<usize>,
        num_features: usize,
        use_batch_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        const KERNEL_SIZE: usize = 3;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        let layer1_rn = conv2d_no_bias(
            *channel_sizes.get(0).unwrap(),
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer1_rn"),
        )?;
        let layer2_rn = conv2d_no_bias(
            *channel_sizes.get(1).unwrap(),
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer2_rn"),
        )?;
        let layer3_rn = conv2d_no_bias(
            *channel_sizes.get(2).unwrap(),
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer3_rn"),
        )?;
        let layer4_rn = conv2d_no_bias(
            *channel_sizes.get(3).unwrap(),
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer4_rn"),
        )?;

        let refine_net1 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            PATCH_SIZE * 8,
            vb.pp("refinenet1"),
        )?;
        let refine_net2 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            PATCH_SIZE * 4,
            vb.pp("refinenet2"),
        )?;
        let refine_net3 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            PATCH_SIZE * 2,
            vb.pp("refinenet3"),
        )?;
        let refine_net4 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            PATCH_SIZE,
            vb.pp("refinenet4"),
        )?;

        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let output_conv1 = conv2d(
            num_features,
            num_features / 2,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("output_conv1"),
        )?;

        let output_conv2 = seq();
        const HEAD_FEATURES_2: usize = 32;
        const OUT_CHANNELS_2: usize = 1;
        const KERNEL_SIZE_2: usize = 1;
        let output_conv2 = output_conv2.add(conv2d(
            num_features / 2,
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
        // TODO currently skipping the identity() call, doesn't seem necessary

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
    use_class_token: bool,
    projections: Vec<Conv2d>,
    resize_layers: Vec<Box<dyn Module>>,
    readout_projections: Vec<Sequential>,
    scratch: Scratch,
}

impl DPTHead {
    pub fn new(
        out_channel_sizes: Vec<usize>,
        in_channel_size: usize,
        num_features: usize,
        use_batch_norm: bool,
        use_class_token: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let projections = out_channel_sizes
            .iter()
            .enumerate()
            .map(|(conv_index, out_channel_size)| {
                conv2d(
                    in_channel_size,
                    *out_channel_size,
                    1,
                    Conv2dConfig {
                        padding: 0,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    vb.pp("projects").pp(conv_index.to_string()),
                )
                .unwrap()
            })
            .collect();

        let resize_layers: Vec<Box<dyn Module>> = vec![
            Box::new(conv_transpose2d(
                *out_channel_sizes.get(0).unwrap(),
                *out_channel_sizes.get(0).unwrap(),
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
                *out_channel_sizes.get(1).unwrap(),
                *out_channel_sizes.get(1).unwrap(),
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
                *out_channel_sizes.get(3).unwrap(),
                *out_channel_sizes.get(3).unwrap(),
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

        let readout_projections = if use_class_token {
            (0..NUM_CHANNELS)
                .map(|rop_index| {
                    seq()
                        .add(
                            linear(
                                2 * in_channel_size,
                                in_channel_size,
                                vb.pp("readout_projects").pp(rop_index.to_string()),
                            )
                            .unwrap(),
                        )
                        .add(Activation::Gelu)
                })
                .collect()
        } else {
            vec![]
        };

        let scratch = Scratch::new(
            out_channel_sizes,
            num_features,
            use_batch_norm,
            vb.pp("scratch"),
        )?;

        Ok(Self {
            use_class_token,
            projections,
            resize_layers,
            readout_projections,
            scratch,
        })
    }
}
pub const DINO_IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 37; //  518 // 14 TODO see how to solve this dynamically

impl Module for DPTHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = (0..NUM_CHANNELS)
            .map(|i| {
                let x = if self.use_class_token {
                    let x = xs.get(i).unwrap().get(0).unwrap();
                    let class_token = xs.get(i).unwrap().get(1).unwrap();
                    let readout = class_token.unsqueeze(1).unwrap().expand(x.shape()).unwrap();
                    let to_cat = [x, readout];
                    let cat = Tensor::cat(&to_cat, Minus1).unwrap();
                    self.readout_projections[i].forward(&cat).unwrap()
                } else {
                    xs.get(i).unwrap()
                };
                let x_dims = x.dims();

                let x = x
                    .permute((0, 2, 1))
                    .unwrap()
                    .reshape((x_dims[0], x_dims[x_dims.len() - 1], PATCH_SIZE, PATCH_SIZE))
                    .unwrap();
                let x = self.projections[i].forward(&x).unwrap();

                self.resize_layers[i].forward(&x).unwrap()
            })
            .collect::<Vec<Tensor>>();

        let layer_1_rn = self.scratch.layer1_rn.forward(out.get(0).unwrap())?;
        let layer_2_rn = self.scratch.layer2_rn.forward(out.get(1).unwrap())?;
        let layer_3_rn = self.scratch.layer3_rn.forward(out.get(2).unwrap())?;
        let layer_4_rn = self.scratch.layer4_rn.forward(out.get(3).unwrap())?;

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

        let out = out.interpolate2d(DINO_IMG_SIZE, DINO_IMG_SIZE)?;

        self.scratch.output_conv2.forward(&out)
    }
}

pub struct DepthAnythingV2<'a> {
    pretrained: &'a DinoVisionTransformer,
    depth_head: DPTHead,
}

impl<'a> DepthAnythingV2<'a> {
    pub fn new(
        pretrained: &'a DinoVisionTransformer,
        in_channel_size: usize,
        out_channel_sizes: Vec<usize>,
        num_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depth_head = DPTHead::new(
            out_channel_sizes,
            in_channel_size,
            num_features,
            false,
            false,
            vb.pp("depth_head"),
        )?;

        Ok(Self {
            pretrained,
            depth_head,
        })
    }
}

impl<'a> Module for DepthAnythingV2<'a> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.shape().dims();
        println!("DepthAnythingV2 got dims: {:?}", dims);

        let layer_ids_vits = vec![2, 5, 8, 11];
        // let layer_ids_vitb = vec![2, 5, 8, 11];
        // let layer_ids_vitl = vec![4, 11, 17, 23];
        // let layer_ids_vitg = vec![9, 19, 29, 39];

        let features =
            self.pretrained
                .get_intermediate_layers(xs, layer_ids_vits, false, false, true)?;
        let depth = self.depth_head.forward(&features)?;

        depth.relu()
    }
}
