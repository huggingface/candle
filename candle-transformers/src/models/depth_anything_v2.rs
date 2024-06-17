use candle_nn::{Activation, BatchNorm, Conv2d, Sequential};

use crate::models::dinov2::DinoVisionTransformer;

pub struct ResidualConvUnit {
    activation: Activation,
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: Option<BatchNorm>,
    batch_norm2: Option<BatchNorm>,
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