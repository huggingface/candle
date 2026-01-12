//! Decoder module for BiRefNet
//!
//! This module implements the multi-scale decoder with skip connections.

use candle::{Module, Result, Tensor};
use candle_nn::{batch_norm, BatchNorm, Conv2d, VarBuilder};

use super::blocks::{BasicDecBlk, BasicLatBlk, SimpleConvs};
use super::config::Config;

/// Gradient convolution block: Conv2d -> BatchNorm -> ReLU
#[derive(Debug, Clone)]
struct GdtConvs {
    conv: Conv2d,
    bn: BatchNorm,
}

impl GdtConvs {
    fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = candle_nn::conv2d(
            in_channels,
            out_channels,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("0"),
        )?;
        let bn = batch_norm(out_channels, 1e-5, vb.pp("1"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_train(&x)?;
        x.relu()
    }
}

/// BiRefNet Decoder
#[derive(Debug, Clone)]
pub struct Decoder {
    config: Config,

    // IPT blocks (decoder input)
    ipt_blk5: Option<SimpleConvs>,
    ipt_blk4: Option<SimpleConvs>,
    ipt_blk3: Option<SimpleConvs>,
    ipt_blk2: Option<SimpleConvs>,
    ipt_blk1: Option<SimpleConvs>,

    // Decoder blocks
    decoder_block4: BasicDecBlk,
    decoder_block3: BasicDecBlk,
    decoder_block2: BasicDecBlk,
    decoder_block1: BasicDecBlk,

    // Lateral blocks
    lateral_block4: BasicLatBlk,
    lateral_block3: BasicLatBlk,
    lateral_block2: BasicLatBlk,

    // Output
    conv_out1: Conv2d,

    // Multi-scale supervision
    conv_ms_spvn_4: Option<Conv2d>,
    conv_ms_spvn_3: Option<Conv2d>,
    conv_ms_spvn_2: Option<Conv2d>,

    // Gradient attention (out_ref)
    gdt_convs_4: Option<GdtConvs>,
    gdt_convs_3: Option<GdtConvs>,
    gdt_convs_2: Option<GdtConvs>,
    gdt_convs_attn_4: Option<Conv2d>,
    gdt_convs_attn_3: Option<Conv2d>,
    gdt_convs_attn_2: Option<Conv2d>,
}

impl Decoder {
    pub fn new(channels: &[usize], config: &Config, vb: VarBuilder) -> Result<Self> {
        // channels = [3072, 1536, 768, 384] for Swin-L with mul_scl_ipt='cat'
        let use_bn = true;

        // IPT blocks output channel configuration
        // ipt_cha_opt = 1 means using channels[x]//8
        // For Swin-L + mul_scl_ipt='cat':
        //   ipt_blk5 output: 3072 // 8 = 384
        //   ipt_blk4 output: 3072 // 8 = 384
        //   ipt_blk3 output: 1536 // 8 = 192
        //   ipt_blk2 output: 768 // 8 = 96
        //   ipt_blk1 output: 384 // 8 = 48
        let (ipt_blk5, ipt_blk4, ipt_blk3, ipt_blk2, ipt_blk1, ipt_cha_tuple) = if config.dec_ipt {
            let ic = 64; // inter_channels

            // Output channels: channels[x] // 8
            let out_ch_5 = channels[0] / 8;
            let out_ch_4 = channels[0] / 8;
            let out_ch_3 = channels[1] / 8;
            let out_ch_2 = channels[2] / 8;
            let out_ch_1 = channels[3] / 8;

            // Input channels depend on dec_ipt_split
            let split_factor = if config.dec_ipt_split { 1024 } else { 1 };

            (
                Some(SimpleConvs::new(
                    split_factor * 3,
                    out_ch_5,
                    ic,
                    vb.pp("ipt_blk5"),
                )?),
                Some(SimpleConvs::new(256 * 3, out_ch_4, ic, vb.pp("ipt_blk4"))?),
                Some(SimpleConvs::new(64 * 3, out_ch_3, ic, vb.pp("ipt_blk3"))?),
                Some(SimpleConvs::new(16 * 3, out_ch_2, ic, vb.pp("ipt_blk2"))?),
                Some(SimpleConvs::new(3, out_ch_1, ic, vb.pp("ipt_blk1"))?),
                (out_ch_5, out_ch_4, out_ch_3, out_ch_2, out_ch_1),
            )
        } else {
            (None, None, None, None, None, (0, 0, 0, 0, 0))
        };

        // Decoder blocks (using ASPPDeformable)
        let decoder_block4 = BasicDecBlk::new(
            channels[0] + ipt_cha_tuple.0,
            channels[1],
            config.dec_att,
            use_bn,
            vb.pp("decoder_block4"),
        )?;
        let decoder_block3 = BasicDecBlk::new(
            channels[1] + ipt_cha_tuple.1,
            channels[2],
            config.dec_att,
            use_bn,
            vb.pp("decoder_block3"),
        )?;
        let decoder_block2 = BasicDecBlk::new(
            channels[2] + ipt_cha_tuple.2,
            channels[3],
            config.dec_att,
            use_bn,
            vb.pp("decoder_block2"),
        )?;
        let decoder_block1 = BasicDecBlk::new(
            channels[3] + ipt_cha_tuple.3,
            channels[3] / 2,
            config.dec_att,
            use_bn,
            vb.pp("decoder_block1"),
        )?;

        // Lateral blocks
        let lateral_block4 = BasicLatBlk::new(channels[1], channels[1], vb.pp("lateral_block4"))?;
        let lateral_block3 = BasicLatBlk::new(channels[2], channels[2], vb.pp("lateral_block3"))?;
        let lateral_block2 = BasicLatBlk::new(channels[3], channels[3], vb.pp("lateral_block2"))?;

        // Output (Sequential with index 0)
        let conv_out1 = candle_nn::conv2d(
            channels[3] / 2 + ipt_cha_tuple.4,
            1,
            1,
            Default::default(),
            vb.pp("conv_out1.0"),
        )?;

        // Multi-scale supervision
        let (conv_ms_spvn_4, conv_ms_spvn_3, conv_ms_spvn_2) = if config.ms_supervision {
            (
                Some(candle_nn::conv2d(
                    channels[1],
                    1,
                    1,
                    Default::default(),
                    vb.pp("conv_ms_spvn_4"),
                )?),
                Some(candle_nn::conv2d(
                    channels[2],
                    1,
                    1,
                    Default::default(),
                    vb.pp("conv_ms_spvn_3"),
                )?),
                Some(candle_nn::conv2d(
                    channels[3],
                    1,
                    1,
                    Default::default(),
                    vb.pp("conv_ms_spvn_2"),
                )?),
            )
        } else {
            (None, None, None)
        };

        // Gradient attention (out_ref)
        // _N = 16 in Python
        let gdt_n = 16;
        let (gdt_convs_4, gdt_convs_3, gdt_convs_2, gdt_convs_attn_4, gdt_convs_attn_3, gdt_convs_attn_2) =
            if config.out_ref {
                (
                    Some(GdtConvs::new(channels[1], gdt_n, vb.pp("gdt_convs_4"))?),
                    Some(GdtConvs::new(channels[2], gdt_n, vb.pp("gdt_convs_3"))?),
                    Some(GdtConvs::new(channels[3], gdt_n, vb.pp("gdt_convs_2"))?),
                    Some(candle_nn::conv2d(gdt_n, 1, 1, Default::default(), vb.pp("gdt_convs_attn_4.0"))?),
                    Some(candle_nn::conv2d(gdt_n, 1, 1, Default::default(), vb.pp("gdt_convs_attn_3.0"))?),
                    Some(candle_nn::conv2d(gdt_n, 1, 1, Default::default(), vb.pp("gdt_convs_attn_2.0"))?),
                )
            } else {
                (None, None, None, None, None, None)
            };

        Ok(Self {
            config: config.clone(),
            ipt_blk5,
            ipt_blk4,
            ipt_blk3,
            ipt_blk2,
            ipt_blk1,
            decoder_block4,
            decoder_block3,
            decoder_block2,
            decoder_block1,
            lateral_block4,
            lateral_block3,
            lateral_block2,
            conv_out1,
            conv_ms_spvn_4,
            conv_ms_spvn_3,
            conv_ms_spvn_2,
            gdt_convs_4,
            gdt_convs_3,
            gdt_convs_2,
            gdt_convs_attn_4,
            gdt_convs_attn_3,
            gdt_convs_attn_2,
        })
    }

    /// Get patches batch for dec_ipt_split
    ///
    /// Splits input image into patches based on reference feature map size,
    /// then concatenates all patches along channel dimension.
    ///
    /// # Arguments
    /// * `x` - Original input image [B, C, H, W]
    /// * `p` - Reference feature map [B, C', H', W']
    ///
    /// # Returns
    /// patches: [B, num_patches * C, H', W']
    fn get_patches_batch(&self, x: &Tensor, p: &Tensor) -> Result<Tensor> {
        let (_, _, size_h, size_w) = p.dims4()?;
        let (b, c, h, w) = x.dims4()?;

        let num_patches_h = h / size_h;
        let num_patches_w = w / size_w;

        // Step 1: reshape to decompose spatial dimensions
        // [B, C, H, W] -> [B, C, num_h, size_h, num_w, size_w]
        let patches = x.reshape((b, c, num_patches_h, size_h, num_patches_w, size_w))?;

        // Step 2: permute to move patch indices before channel
        // [B, C, num_h, size_h, num_w, size_w] -> [B, num_h, num_w, C, size_h, size_w]
        let patches = patches.permute((0, 2, 4, 1, 3, 5))?;

        // Step 3: reshape to merge patch indices and channel
        // [B, num_h, num_w, C, size_h, size_w] -> [B, num_h * num_w * C, size_h, size_w]
        patches.reshape((b, num_patches_h * num_patches_w * c, size_h, size_w))
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `features` - [x, x1, x2, x3, x4] where x is original input
    ///
    /// # Returns
    /// Multi-scale outputs (last one is highest resolution)
    pub fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        // Debug helper
        #[cfg(debug_assertions)]
        fn print_stats(name: &str, t: &Tensor) {
            if let Ok(t_cpu) = t.to_device(&candle::Device::Cpu) {
                if let Ok(flat) = t_cpu.flatten_all() {
                    if let Ok(data) = flat.to_vec1::<f32>() {
                        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                        eprintln!("  Decoder {}: shape={:?}, min={:.4}, max={:.4}, mean={:.4}", name, t.dims(), min, max, mean);
                    }
                }
            }
        }

        let (x, x1, x2, x3, x4) = (
            &features[0],
            &features[1],
            &features[2],
            &features[3],
            &features[4],
        );
        let mut outs = Vec::new();

        // Stage 4
        let x4 = if let Some(ipt_blk5) = &self.ipt_blk5 {
            let patches = if self.config.dec_ipt_split {
                self.get_patches_batch(x, x4)?
            } else {
                x.clone()
            };
            let (_, _, h, w) = x4.dims4()?;
            let ipt = ipt_blk5.forward(&patches.upsample_bilinear2d(h, w, true)?)?;
            #[cfg(debug_assertions)]
            print_stats("ipt_blk5", &ipt);
            Tensor::cat(&[x4.clone(), ipt], 1)?
        } else {
            x4.clone()
        };

        #[cfg(debug_assertions)]
        eprintln!("  Decoder decoder_block4 input:");
        let p4 = self.decoder_block4.forward(&x4)?;
        #[cfg(debug_assertions)]
        print_stats("decoder_block4 output", &p4);
        
        if let Some(conv) = &self.conv_ms_spvn_4 {
            outs.push(conv.forward(&p4)?);
        }

        // Apply gradient attention for stage 4
        let p4 = if let (Some(gdt_convs), Some(gdt_attn)) = (&self.gdt_convs_4, &self.gdt_convs_attn_4) {
            let p4_gdt = gdt_convs.forward(&p4)?;
            let attn = candle_nn::ops::sigmoid(&gdt_attn.forward(&p4_gdt)?)?;
            #[cfg(debug_assertions)]
            print_stats("gdt_attn_4", &attn);
            let dims = p4.dims().to_vec();
            (p4 * attn.broadcast_as(dims)?)?
        } else {
            p4
        };

        let (_, _, h3, w3) = x3.dims4()?;
        let _p4 = p4.upsample_bilinear2d(h3, w3, true)?;
        let lateral4 = self.lateral_block4.forward(x3)?;
        #[cfg(debug_assertions)]
        print_stats("lateral_block4", &lateral4);
        let _p3 = (_p4 + lateral4)?;

        // Stage 3
        let _p3 = if let Some(ipt_blk4) = &self.ipt_blk4 {
            let patches = if self.config.dec_ipt_split {
                self.get_patches_batch(x, &_p3)?
            } else {
                x.clone()
            };
            let ipt = ipt_blk4.forward(&patches.upsample_bilinear2d(h3, w3, true)?)?;
            #[cfg(debug_assertions)]
            print_stats("ipt_blk4", &ipt);
            Tensor::cat(&[_p3, ipt], 1)?
        } else {
            _p3
        };

        #[cfg(debug_assertions)]
        {
            fn print_stats_inline(name: &str, t: &Tensor) {
                if let Ok(t_cpu) = t.to_device(&candle::Device::Cpu) {
                    if let Ok(flat) = t_cpu.flatten_all() {
                        if let Ok(data) = flat.to_vec1::<f32>() {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                            eprintln!("  Decoder {}: shape={:?}, min={:.4}, max={:.4}, mean={:.4}", name, t.dims(), min, max, mean);
                        }
                    }
                }
            }
            print_stats_inline("decoder_block3 actual input (_p3)", &_p3);
        }
        #[cfg(debug_assertions)]
        eprintln!("  Decoder decoder_block3 input:");
        let p3 = self.decoder_block3.forward(&_p3)?;
        #[cfg(debug_assertions)]
        print_stats("decoder_block3 output", &p3);
        
        if let Some(conv) = &self.conv_ms_spvn_3 {
            outs.push(conv.forward(&p3)?);
        }

        // Apply gradient attention for stage 3
        let p3 = if let (Some(gdt_convs), Some(gdt_attn)) = (&self.gdt_convs_3, &self.gdt_convs_attn_3) {
            let p3_gdt = gdt_convs.forward(&p3)?;
            let attn = candle_nn::ops::sigmoid(&gdt_attn.forward(&p3_gdt)?)?;
            #[cfg(debug_assertions)]
            print_stats("gdt_attn_3", &attn);
            let dims = p3.dims().to_vec();
            (p3 * attn.broadcast_as(dims)?)?
        } else {
            p3
        };

        let (_, _, h2, w2) = x2.dims4()?;
        let _p3 = p3.upsample_bilinear2d(h2, w2, true)?;
        let lateral3 = self.lateral_block3.forward(x2)?;
        #[cfg(debug_assertions)]
        print_stats("lateral_block3", &lateral3);
        let _p2 = (_p3 + lateral3)?;

        // Stage 2
        let _p2 = if let Some(ipt_blk3) = &self.ipt_blk3 {
            let patches = if self.config.dec_ipt_split {
                self.get_patches_batch(x, &_p2)?
            } else {
                x.clone()
            };
            let ipt = ipt_blk3.forward(&patches.upsample_bilinear2d(h2, w2, true)?)?;
            #[cfg(debug_assertions)]
            print_stats("ipt_blk3", &ipt);
            Tensor::cat(&[_p2, ipt], 1)?
        } else {
            _p2
        };

        #[cfg(debug_assertions)]
        eprintln!("  Decoder decoder_block2 input:");
        let p2 = self.decoder_block2.forward(&_p2)?;
        #[cfg(debug_assertions)]
        print_stats("decoder_block2 output", &p2);
        
        if let Some(conv) = &self.conv_ms_spvn_2 {
            outs.push(conv.forward(&p2)?);
        }

        // Apply gradient attention for stage 2
        let p2 = if let (Some(gdt_convs), Some(gdt_attn)) = (&self.gdt_convs_2, &self.gdt_convs_attn_2) {
            let p2_gdt = gdt_convs.forward(&p2)?;
            let attn = candle_nn::ops::sigmoid(&gdt_attn.forward(&p2_gdt)?)?;
            #[cfg(debug_assertions)]
            print_stats("gdt_attn_2", &attn);
            let dims = p2.dims().to_vec();
            (p2 * attn.broadcast_as(dims)?)?
        } else {
            p2
        };

        let (_, _, h1, w1) = x1.dims4()?;
        let _p2 = p2.upsample_bilinear2d(h1, w1, true)?;
        let lateral2 = self.lateral_block2.forward(x1)?;
        #[cfg(debug_assertions)]
        print_stats("lateral_block2", &lateral2);
        let _p1 = (_p2 + lateral2)?;

        // Stage 1
        let _p1 = if let Some(ipt_blk2) = &self.ipt_blk2 {
            let patches = if self.config.dec_ipt_split {
                self.get_patches_batch(x, &_p1)?
            } else {
                x.clone()
            };
            let ipt = ipt_blk2.forward(&patches.upsample_bilinear2d(h1, w1, true)?)?;
            #[cfg(debug_assertions)]
            print_stats("ipt_blk2", &ipt);
            Tensor::cat(&[_p1, ipt], 1)?
        } else {
            _p1
        };

        #[cfg(debug_assertions)]
        eprintln!("  Decoder decoder_block1 input:");
        let _p1 = self.decoder_block1.forward(&_p1)?;
        #[cfg(debug_assertions)]
        print_stats("decoder_block1 output", &_p1);
        
        let (_, _, h, w) = x.dims4()?;
        let _p1 = _p1.upsample_bilinear2d(h, w, true)?;

        // Final output
        let _p1 = if let Some(ipt_blk1) = &self.ipt_blk1 {
            let patches = if self.config.dec_ipt_split {
                self.get_patches_batch(x, &_p1)?
            } else {
                x.clone()
            };
            let ipt = ipt_blk1.forward(&patches.upsample_bilinear2d(h, w, true)?)?;
            #[cfg(debug_assertions)]
            print_stats("ipt_blk1", &ipt);
            Tensor::cat(&[_p1, ipt], 1)?
        } else {
            _p1
        };

        let p1_out = self.conv_out1.forward(&_p1)?;
        #[cfg(debug_assertions)]
        print_stats("conv_out1", &p1_out);
        outs.push(p1_out);

        Ok(outs)
    }
}
