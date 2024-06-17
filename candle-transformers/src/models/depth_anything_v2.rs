use candle::Result;
use candle_nn::{Activation, batch_norm, BatchNorm, BatchNormConfig, Conv2d, conv2d, Conv2dConfig, Sequential, VarBuilder};

use crate::models::dinov2::DinoVisionTransformer;

pub struct ResidualConvUnit {
    activation: Activation,
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: Option<BatchNorm>,
    batch_norm2: Option<BatchNorm>,
}

impl ResidualConvUnit {
    pub fn new(num_features: usize, activation: Activation, use_batch_norm: bool, var_builder: VarBuilder) -> Result<Self> {
        const KERNEL_SIZE: usize = 3;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv1 = conv2d(num_features, num_features, KERNEL_SIZE, conv_cfg, var_builder.push_prefix("conv1"))?;
        let conv2 = conv2d(num_features, num_features, KERNEL_SIZE, conv_cfg, var_builder.push_prefix("conv2"))?;

        
        let (batch_norm1, batch_norm2) = match use_batch_norm {
            true => {
                let batch_norm_cfg = BatchNormConfig {
                    eps: 1e-05,
                    remove_mean: false,
                    affine: true,
                    momentum: 0.1,
                };
                (Some(batch_norm(num_features, batch_norm_cfg, var_builder.push_prefix("batch_norm1"))?), Some(batch_norm(num_features, batch_norm_cfg, var_builder.push_prefix("batch_norm2"))?))
            },
            false => (None, None)
        };

        Ok(Self {
          activation,
            conv1,
            conv2,
            batch_norm1,
            batch_norm2
        })

    }
}

pub struct FeatureFusionBlock {
    res_conv_unit1: ResidualConvUnit,
    res_conv_unit2: ResidualConvUnit,
    output_conv: Conv2d,

}

pub struct Scratch {
    layer1_rn: Conv2d,
    layer2_rn: Conv2d,
    layer3_rn: Conv2d,
    layer4_rn: Option<Conv2d>,
    refine_net1: FeatureFusionBlock,
    refine_net2: FeatureFusionBlock,
    refine_net3: FeatureFusionBlock,
    refine_net4: FeatureFusionBlock,
    output_conv1: Conv2d,
    output_conv2: Sequential,
}

pub struct DPTHead {
    num_classes: usize,
    use_class_token: bool,
    projections: Sequential,
    resize_layers: Sequential,
    readout_projections: Option<Sequential>,
}

pub struct DepthAnythingV2 {
    pretrained: DinoVisionTransformer,
    depth_head: DPTHead,
}