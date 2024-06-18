use candle::{Module, Result, Tensor};
use candle_nn::ops::Identity;
use candle_nn::{
    batch_norm, conv2d, conv_transpose2d, linear, seq, Activation, BatchNorm, BatchNormConfig,
    Conv2d, Conv2dConfig, ConvTranspose2dConfig, Sequential, VarBuilder,
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
        var_builder: VarBuilder,
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
            var_builder.push_prefix("conv1"),
        )?;
        let conv2 = conv2d(
            num_features,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            var_builder.push_prefix("conv2"),
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
                    Some(batch_norm(
                        num_features,
                        batch_norm_cfg,
                        var_builder.push_prefix("batch_norm1"),
                    )?),
                    Some(batch_norm(
                        num_features,
                        batch_norm_cfg,
                        var_builder.push_prefix("batch_norm2"),
                    )?),
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
        let out = self.conv1(&out)?;
        let out = if let Some(batch_norm1) = &self.batch_norm1 {
            batch_norm1.forward_train(&out)?
        } else {
            out
        };

        let out = self.activation.forward(out)?;
        let out = self.conv2(&out)?;
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
    use_scaling: bool,
}

impl FeatureFusionBlock {
    pub fn new(
        num_features: usize,
        activation: Activation,
        use_batch_norm: bool,
        use_scaling: bool,
        var_builder: VarBuilder,
    ) -> Result<Self> {
        const KERNEL_SIZE: usize = 1;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let output_conv = conv2d(
            num_features,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            var_builder.push_prefix("output_conv"),
        )?;
        let res_conv_unit1 = ResidualConvUnit::new(
            num_features,
            activation,
            use_batch_norm,
            var_builder.push_prefix("res_conv_unit1"),
        )?;
        let res_conv_unit2 = ResidualConvUnit::new(
            num_features,
            activation,
            use_batch_norm,
            var_builder.push_prefix("res_conv_unit2"),
        )?;

        Ok(Self {
            res_conv_unit1,
            res_conv_unit2,
            output_conv,
            use_scaling,
        })
    }
}

impl Module for FeatureFusionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // TODO for now this call is moved to Scratch. Not great....
        // let out = xs.get(0)?;
        // let out = match xs.elem_count() {
        //     2 => self.res_conv_unit1.forward(&xs.get(1)?)?,
        //     _ => out
        // };

        let out = self.res_conv_unit2.forward(&xs)?;
        let size = xs.shape();
        let dims = size.dims();
        let target_h = if self.use_scaling {
            dims[-1] * 2
        } else {
            dims[-1]
        };
        let target_w = if self.use_scaling {
            dims[-2] * 2
        } else {
            dims[-2]
        };

        let out = out.interpolate2d(target_h, target_w)?;

        self.output_conv.forward(&out)
    }
}

pub struct Scratch {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Option<Conv2d>,
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
        var_builder: VarBuilder,
    ) -> Result<Self> {
        const KERNEL_SIZE: usize = 3;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        let conv1 = conv2d(
            *channel_sizes.get(0)?,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            var_builder.push_prefix("conv1"),
        )?;
        let conv2 = conv2d(
            *channel_sizes.get(1)?,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            var_builder.push_prefix("conv2"),
        )?;
        let conv3 = conv2d(
            *channel_sizes.get(2)?,
            num_features,
            KERNEL_SIZE,
            conv_cfg,
            var_builder.push_prefix("conv3"),
        )?;
        let conv4 = channel_sizes.get(3).map(|in_shape| {
            conv2d(
                *channel_sizes.get(2)?,
                num_features,
                KERNEL_SIZE,
                conv_cfg,
                var_builder.push_prefix("conv4"),
            )?
        });

        let refine_net1 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            true,
            var_builder.push_prefix("refine_net1"),
        )?;
        let refine_net2 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            false,
            var_builder.push_prefix("refine_net2"),
        )?;
        let refine_net3 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            false,
            var_builder.push_prefix("refine_net3"),
        )?;
        let refine_net4 = FeatureFusionBlock::new(
            num_features,
            Activation::Relu,
            use_batch_norm,
            false,
            var_builder.push_prefix("refine_net4"),
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
            var_builder.push_prefix("output_conv1"),
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
            var_builder.push_prefix("output_conv2_conv1"),
        )?);
        let output_conv2 = output_conv2
            .add(Activation::Relu)
            .add(conv2d(
                HEAD_FEATURES_2,
                OUT_CHANNELS_2,
                KERNEL_SIZE_2,
                conv_cfg,
                var_builder.push_prefix("output_conv2_conv2"),
            )?)
            .add(Activation::Relu);
        // TODO currently skipping the identity() call, doesn't seem necessary

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv4,
            refine_net1,
            refine_net2,
            refine_net3,
            refine_net4,
            output_conv1,
            output_conv2,
        })
    }
}

impl Module for Scratch {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // these are called layer_*_rn in the Python impl
        let conv1_out = self.conv1.forward(&xs.get(0)?)?;
        let conv2_out = self.conv2.forward(&xs.get(1)?)?;
        let conv3_out = self.conv3.forward(&xs.get(2)?)?;
        let conv4_out = self.conv4.forward(&xs.get(3)?)?;

        let path4 = self.refine_net4.forward(&conv4_out)?;

        let res3_out = self.refine_net3.res_conv_unit1.forward(&conv3_out)?;
        let res3_out = path4.add(&res3_out)?;
        let path3 = self.refine_net3.forward(&res3_out)?;

        let res2_out = self.refine_net2.res_conv_unit1.forward(&conv2_out)?;
        let res2_out = path3.add(&res2_out)?;
        let path2 = self.refine_net2.forward(&res2_out)?;

        let res1_out = self.refine_net1.res_conv_unit1.forward(&conv1_out)?;
        let res1_out = path2.add(&res1_out)?;
        let path1 = self.refine_net1.forward(&res1_out)?;

        let out = self.output_conv1.forward(&path1)?;

        // TODO this needs to be scaled to the correct width and height
        // which the Python implementation does somewhere else
        let dims = xs.shape().dims()?;
        println!("Got dims: {:?}", dims);
        let [height, width] = dims[-2..];

        let out = out.interpolate2d(height, width)?;

        self.output_conv2.forward(&out)
    }
}

const NUM_CHANNELS: usize = 4;

pub struct DPTHead {
    use_class_token: bool,
    projections: Vec<Conv2d>,
    resize_layers: Vec<dyn Module>,
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
        var_builder: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };

        let projections = out_channel_sizes
            .iter()
            .enumerate()
            .map(|(conv_index, out_channel_size)| {
                conv2d(
                    in_channel_size,
                    *out_channel_size,
                    1,
                    conv_cfg,
                    var_builder.push_prefix(format!("projection_{}", conv_index)),
                )?
            })
            .collect();

        let resize_layers: Vec<dyn Module> = vec![
            conv_transpose2d(
                *out_channel_sizes.get(0)?,
                *out_channel_sizes.get(0)?,
                4,
                ConvTranspose2dConfig {
                    padding: 0,
                    stride: 4,
                    dilation: 1,
                    output_padding: 0,
                },
                var_builder.push_prefix("resize_layer1"),
            )?,
            conv_transpose2d(
                *out_channel_sizes.get(1)?,
                *out_channel_sizes.get(1)?,
                2,
                ConvTranspose2dConfig {
                    padding: 0,
                    stride: 2,
                    dilation: 1,
                    output_padding: 0,
                },
                var_builder.push_prefix("resize_layer2"),
            )?,
            Identity::new(),
            conv2d(
                *out_channel_sizes.get(3)?,
                *out_channel_sizes.get(3)?,
                2,
                Conv2dConfig {
                    padding: 0,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                },
                var_builder.push_prefix("resize_layer4"),
            )?,
        ];

        let readout_projections = if use_class_token {
            (0..NUM_CHANNELS)
                .map(|rop_index| {
                    seq()
                        .add(linear(
                            2 * in_channel_size,
                            in_channel_size,
                            var_builder
                                .push_prefix(format!("readout_projections_linear_{}", rop_index)),
                        )?)
                        .add(Activation::Relu)
                })
                .collect()
        } else {
            vec![]
        };

        let scratch = Scratch::new(
            out_channel_sizes,
            num_features,
            use_batch_norm,
            var_builder.push_prefix("scratch"),
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

impl Module for DPTHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // TODO this needs to be scaled to the correct width and height
        // which the Python implementation does somewhere else
        let dims = xs.shape().dims()?;
        println!("Got dims: {:?}", dims);
        let [height, width] = dims[-2..];
        const PATCH_DENOMINATOR: i32 = 14;
        let patch_height = height / PATCH_DENOMINATOR;
        let patch_width = width / PATCH_DENOMINATOR;

        let out = Tensor::stack(
            (0..NUM_CHANNELS)
                .map(|i| {
                    let x = if self.use_class_token {
                        let x = xs.get(0)?;
                        let class_token = xs.get(1)?;
                        let readout = class_token.unsqueeze(1).unwrap().expand(x.shape())?;
                        let cat = Tensor::cat(&*[x, readout], -1)?;
                        self.readout_projections[i].forward(&cat)?
                    } else {
                        xs.get(0)?
                    };
                    let x_dims = x.dims();
                    let x = x.permute((0, 2, 1))?.reshape((
                        x_dims[0],
                        x_dims[-1],
                        patch_height,
                        patch_width,
                    ))?;
                    let x = self.projections[i].forward(&x)?;

                    self.resize_layers[i].forward(&x)
                })
                .collect(),
            0,
        )?;

        self.scratch.forward(&out)
    }
}

pub struct DepthAnythingV2 {
    pretrained: DinoVisionTransformer,
    depth_head: DPTHead,
}
