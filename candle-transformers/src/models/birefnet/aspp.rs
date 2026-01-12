//! ASPP (Atrous Spatial Pyramid Pooling) modules for BiRefNet
//!
//! This module implements ASPP and ASPPDeformable using deformable convolutions.

use candle::{Module, Result, Tensor, D};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, ModuleT, VarBuilder};

/// Deformable Convolution v2 module
///
/// Implements DCNv2 with learnable offsets and modulation masks.
/// The modulation mask is scaled by 2.0 after sigmoid, following the BiRefNet design.
#[derive(Debug, Clone)]
pub struct DeformableConv2d {
    offset_conv: Conv2d,
    modulator_conv: Conv2d,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl DeformableConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let k2 = kernel_size * kernel_size;
        let cfg = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };

        // Offset convolution: outputs 2 * k * k channels (x, y offset per sampling point)
        let offset_conv =
            candle_nn::conv2d(in_channels, 2 * k2, kernel_size, cfg, vb.pp("offset_conv"))?;

        // Modulator convolution: outputs k * k channels (weight per sampling point)
        let modulator_conv =
            candle_nn::conv2d(in_channels, k2, kernel_size, cfg, vb.pp("modulator_conv"))?;

        // Convolution weights
        let weight = vb.get(
            (out_channels, in_channels, kernel_size, kernel_size),
            "regular_conv.weight",
        )?;

        let bias = if use_bias {
            Some(vb.get(out_channels, "regular_conv.bias")?)
        } else {
            None
        };

        Ok(Self {
            offset_conv,
            modulator_conv,
            weight,
            bias,
            stride,
            padding,
        })
    }
}

impl Module for DeformableConv2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Compute offsets
        let offset = self.offset_conv.forward(xs)?;

        // Compute modulation weights (sigmoid * 2.0, range [0, 2])
        let modulator = (candle_nn::ops::sigmoid(&self.modulator_conv.forward(xs)?)? * 2.0)?;

        // Call candle-core's deform_conv2d operation
        xs.deform_conv2d(
            &offset,
            &self.weight,
            Some(&modulator),
            self.bias.as_ref(),
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1), // dilation
            1,      // groups
            1,      // offset_groups
        )
    }
}

/// ASPP module (single branch)
#[derive(Debug, Clone)]
pub struct ASPPModule {
    atrous_conv: Conv2d,
    bn: Option<BatchNorm>,
}

impl ASPPModule {
    pub fn new(
        in_channels: usize,
        planes: usize,
        kernel_size: usize,
        padding: usize,
        dilation: usize,
        use_bn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            dilation,
            ..Default::default()
        };
        let atrous_conv =
            candle_nn::conv2d(in_channels, planes, kernel_size, cfg, vb.pp("atrous_conv"))?;
        let bn = if use_bn {
            Some(candle_nn::batch_norm(planes, 1e-5, vb.pp("bn"))?)
        } else {
            None
        };
        Ok(Self { atrous_conv, bn })
    }
}

impl Module for ASPPModule {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.atrous_conv.forward(xs)?;
        let xs = if let Some(bn) = &self.bn {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };
        xs.relu()
    }
}

/// Global Average Pooling branch for ASPP
///
/// PyTorch structure: nn.Sequential(AdaptiveAvgPool2d, Conv2d, BatchNorm2d)
/// - Index 0: AdaptiveAvgPool2d (no parameters)
/// - Index 1: Conv2d (weight only, no bias)
/// - Index 2: BatchNorm2d
#[derive(Debug, Clone)]
pub struct GlobalAvgPool {
    conv: Conv2d,
    bn: Option<BatchNorm>,
}

impl GlobalAvgPool {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        use_bn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Conv2d at index 1 (no bias in original PyTorch implementation)
        let conv = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            1,
            Default::default(),
            vb.pp("1"),
        )?;
        // BatchNorm at index 2
        let bn = if use_bn {
            Some(candle_nn::batch_norm(out_channels, 1e-5, vb.pp("2"))?)
        } else {
            None
        };
        Ok(Self { conv, bn })
    }
}

impl Module for GlobalAvgPool {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // AdaptiveAvgPool2d((1, 1))
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = self.conv.forward(&xs)?;
        let xs = if let Some(bn) = &self.bn {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };
        xs.relu()
    }
}

/// Standard ASPP module
#[derive(Debug, Clone)]
pub struct ASPP {
    aspp1: ASPPModule,
    aspp2: ASPPModule,
    aspp3: ASPPModule,
    aspp4: ASPPModule,
    global_avg_pool: GlobalAvgPool,
    conv1: Conv2d,
    bn1: Option<BatchNorm>,
}

impl ASPP {
    pub fn new(
        in_channels: usize,
        out_channels: Option<usize>,
        use_bn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(in_channels);
        let inter_channels = 256;
        let dilations = [1, 6, 12, 18];

        let aspp1 = ASPPModule::new(
            in_channels,
            inter_channels,
            1,
            0,
            dilations[0],
            use_bn,
            vb.pp("aspp1"),
        )?;
        let aspp2 = ASPPModule::new(
            in_channels,
            inter_channels,
            3,
            dilations[1],
            dilations[1],
            use_bn,
            vb.pp("aspp2"),
        )?;
        let aspp3 = ASPPModule::new(
            in_channels,
            inter_channels,
            3,
            dilations[2],
            dilations[2],
            use_bn,
            vb.pp("aspp3"),
        )?;
        let aspp4 = ASPPModule::new(
            in_channels,
            inter_channels,
            3,
            dilations[3],
            dilations[3],
            use_bn,
            vb.pp("aspp4"),
        )?;

        let global_avg_pool = GlobalAvgPool::new(
            in_channels,
            inter_channels,
            use_bn,
            vb.pp("global_avg_pool"),
        )?;

        let conv1 = candle_nn::conv2d(
            inter_channels * 5,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv1"),
        )?;
        let bn1 = if use_bn {
            Some(candle_nn::batch_norm(out_channels, 1e-5, vb.pp("bn1"))?)
        } else {
            None
        };

        Ok(Self {
            aspp1,
            aspp2,
            aspp3,
            aspp4,
            global_avg_pool,
            conv1,
            bn1,
        })
    }
}

impl Module for ASPP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x1 = self.aspp1.forward(xs)?;
        let x2 = self.aspp2.forward(xs)?;
        let x3 = self.aspp3.forward(xs)?;
        let x4 = self.aspp4.forward(xs)?;

        let x5 = self.global_avg_pool.forward(xs)?;
        let (_, _, h, w) = x1.dims4()?;
        let x5 = x5.upsample_bilinear2d(h, w, true)?;

        let xs = Tensor::cat(&[x1, x2, x3, x4, x5], 1)?;
        let xs = self.conv1.forward(&xs)?;
        let xs = if let Some(bn) = &self.bn1 {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };
        xs.relu()
    }
}

/// ASPP module with Deformable Convolution (single branch)
#[derive(Debug, Clone)]
struct ASPPModuleDeformable {
    atrous_conv: DeformableConv2d,
    bn: Option<BatchNorm>,
}

impl ASPPModuleDeformable {
    pub fn new(
        in_channels: usize,
        planes: usize,
        kernel_size: usize,
        padding: usize,
        use_bn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let atrous_conv = DeformableConv2d::new(
            in_channels,
            planes,
            kernel_size,
            1, // stride
            padding,
            false, // bias
            vb.pp("atrous_conv"),
        )?;
        let bn = if use_bn {
            Some(candle_nn::batch_norm(planes, 1e-5, vb.pp("bn"))?)
        } else {
            None
        };
        Ok(Self { atrous_conv, bn })
    }
}

impl Module for ASPPModuleDeformable {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.atrous_conv.forward(xs)?;
        let xs = if let Some(bn) = &self.bn {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };
        xs.relu()
    }
}

/// ASPPDeformable - ASPP variant using Deformable Convolution
///
/// Default parallel_block_sizes = [1, 3, 7]
///
/// Note: Dropout is omitted in inference mode as it becomes identity mapping.
#[derive(Debug, Clone)]
pub struct ASPPDeformable {
    aspp1: ASPPModuleDeformable,
    aspp_deforms: Vec<ASPPModuleDeformable>,
    global_avg_pool: GlobalAvgPool,
    conv1: Conv2d,
    bn1: Option<BatchNorm>,
}

impl ASPPDeformable {
    pub fn new(
        in_channels: usize,
        out_channels: Option<usize>,
        parallel_block_sizes: Vec<usize>,
        use_bn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(in_channels);
        let inter_channels = 256;

        // First ASPP module (1x1 convolution)
        let aspp1 = ASPPModuleDeformable::new(
            in_channels,
            inter_channels,
            1, // kernel_size
            0, // padding
            use_bn,
            vb.pp("aspp1"),
        )?;

        // Multiple Deformable ASPP modules
        let aspp_deforms: Vec<_> = parallel_block_sizes
            .iter()
            .enumerate()
            .map(|(i, &conv_size)| {
                ASPPModuleDeformable::new(
                    in_channels,
                    inter_channels,
                    conv_size,
                    conv_size / 2, // padding = conv_size // 2
                    use_bn,
                    vb.pp(format!("aspp_deforms.{}", i)),
                )
            })
            .collect::<Result<_>>()?;

        // Global average pooling branch
        let global_avg_pool = GlobalAvgPool::new(
            in_channels,
            inter_channels,
            use_bn,
            vb.pp("global_avg_pool"),
        )?;

        // Output convolution (no bias in original PyTorch implementation)
        // channels = inter_channels * (1 + len(aspp_deforms) + 1)
        let num_branches = 2 + parallel_block_sizes.len();
        let conv1 = candle_nn::conv2d_no_bias(
            inter_channels * num_branches,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv1"),
        )?;
        let bn1 = if use_bn {
            Some(candle_nn::batch_norm(out_channels, 1e-5, vb.pp("bn1"))?)
        } else {
            None
        };

        Ok(Self {
            aspp1,
            aspp_deforms,
            global_avg_pool,
            conv1,
            bn1,
        })
    }
}

impl Module for ASPPDeformable {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Debug helper
        #[cfg(debug_assertions)]
        fn print_stats(name: &str, t: &Tensor) {
            if let Ok(t_cpu) = t.to_device(&candle::Device::Cpu) {
                if let Ok(flat) = t_cpu.flatten_all() {
                    if let Ok(data) = flat.to_vec1::<f32>() {
                        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                        eprintln!("      {}: shape={:?}, min={:.4}, max={:.4}, mean={:.4}", name, t.dims(), min, max, mean);
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        eprintln!("    ASPPDeformable forward:");
        #[cfg(debug_assertions)]
        print_stats("input", xs);

        // First branch (1x1 deformable conv)
        let x1 = self.aspp1.forward(xs)?;
        #[cfg(debug_assertions)]
        print_stats("aspp1", &x1);

        // Deformable branches
        let mut features = vec![x1.clone()];
        for (i, aspp) in self.aspp_deforms.iter().enumerate() {
            let feat = aspp.forward(xs)?;
            #[cfg(debug_assertions)]
            print_stats(&format!("aspp_deforms[{}]", i), &feat);
            features.push(feat);
        }

        // Global average pooling branch
        let x_global = self.global_avg_pool.forward(xs)?;
        #[cfg(debug_assertions)]
        print_stats("global_avg_pool", &x_global);
        
        let (_, _, h, w) = x1.dims4()?;
        let x_global_up = x_global.upsample_bilinear2d(h, w, true)?;
        features.push(x_global_up);

        // Concatenate all branches
        let xs = Tensor::cat(&features, 1)?;
        #[cfg(debug_assertions)]
        print_stats("after cat", &xs);

        // Output convolution
        let xs = self.conv1.forward(&xs)?;
        #[cfg(debug_assertions)]
        print_stats("after conv1", &xs);
        
        let xs = if let Some(bn) = &self.bn1 {
            bn.forward_t(&xs, false)?
        } else {
            xs
        };
        #[cfg(debug_assertions)]
        print_stats("after bn1", &xs);
        
        xs.relu()
    }
}
