use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT,
    VarBuilder,
};

#[derive(Debug)]
struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    fn new(scale_factor: usize) -> Self {
        Self { scale_factor }
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_batch, _channels, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(h * self.scale_factor, w * self.scale_factor)
    }
}

#[derive(Debug)]
struct ConvBnSiLu {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBnSiLu {
    fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        k: usize,
        stride: usize,
        padding: Option<usize>,
    ) -> Result<Self> {
        let padding = padding.unwrap_or(k / 2);
        let cfg = Conv2dConfig {
            stride,
            padding,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        let conv = conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?;
        Ok(Self { conv, bn })
    }
}

impl Module for ConvBnSiLu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.bn.forward_t(&xs, false)?;
        candle_nn::ops::silu(&xs)
    }
}

#[derive(Debug)]
struct Bottleneck {
    cv1: ConvBnSiLu,
    cv2: ConvBnSiLu,
    residual: bool,
}

impl Bottleneck {
    fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool, expansion: f32) -> Result<Self> {
        let hidden = (c2 as f32 * expansion) as usize;
        let cv1 = ConvBnSiLu::load(vb.pp("cv1"), c1, hidden, 1, 1, None)?;
        let cv2 = ConvBnSiLu::load(vb.pp("cv2"), hidden, c2, 3, 1, None)?;
        Ok(Self {
            cv1,
            cv2,
            residual: shortcut && c1 == c2,
        })
    }
}

impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.cv2.forward(&self.cv1.forward(xs)?)?;
        if self.residual {
            xs + ys
        } else {
            Ok(ys)
        }
    }
}

#[derive(Debug)]
struct C3 {
    cv1: ConvBnSiLu,
    cv2: ConvBnSiLu,
    cv3: ConvBnSiLu,
    m: Vec<Bottleneck>,
}

impl C3 {
    fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        n: usize,
        shortcut: bool,
        expansion: f32,
    ) -> Result<Self> {
        let hidden = (c2 as f32 * expansion) as usize;
        let cv1 = ConvBnSiLu::load(vb.pp("cv1"), c1, hidden, 1, 1, None)?;
        let cv2 = ConvBnSiLu::load(vb.pp("cv2"), c1, hidden, 1, 1, None)?;
        let cv3 = ConvBnSiLu::load(vb.pp("cv3"), hidden * 2, c2, 1, 1, None)?;
        let mut m = Vec::with_capacity(n);
        for idx in 0..n {
            let b = Bottleneck::load(vb.pp(format!("m.{idx}")), hidden, hidden, shortcut, 1.0)?;
            m.push(b);
        }
        Ok(Self { cv1, cv2, cv3, m })
    }
}

impl Module for C3 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y1 = self.cv1.forward(xs)?;
        let mut y = y1.clone();
        for b in self.m.iter() {
            y = b.forward(&y)?;
        }
        let y2 = self.cv2.forward(xs)?;
        let out = Tensor::cat(&[&y, &y2], 1)?;
        self.cv3.forward(&out)
    }
}

#[derive(Debug)]
struct Sppf {
    cv1: ConvBnSiLu,
    cv2: ConvBnSiLu,
    k: usize,
}

impl Sppf {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let hidden = c1 / 2;
        let cv1 = ConvBnSiLu::load(vb.pp("cv1"), c1, hidden, 1, 1, None)?;
        let cv2 = ConvBnSiLu::load(vb.pp("cv2"), hidden * 4, c2, 1, 1, None)?;
        Ok(Self { cv1, cv2, k })
    }
}

impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.cv1.forward(xs)?;
        let pad = self.k / 2;
        let pooled = |t: &Tensor| -> Result<Tensor> {
            if pad == 0 {
                return t.max_pool2d_with_stride(self.k, 1);
            }
            t.pad_with_zeros(2, pad, pad)?
                .pad_with_zeros(3, pad, pad)?
                .max_pool2d_with_stride(self.k, 1)
        };
        let xs2 = pooled(&xs)?;
        let xs3 = pooled(&xs2)?;
        let xs4 = pooled(&xs3)?;
        self.cv2.forward(&Tensor::cat(&[&xs, &xs2, &xs3, &xs4], 1)?)
    }
}

#[derive(Debug)]
struct CspDarknet53 {
    l0: ConvBnSiLu,
    l1: ConvBnSiLu,
    l2: C3,
    l3: ConvBnSiLu,
    l4: C3,
    l5: ConvBnSiLu,
    l6: C3,
    l7: ConvBnSiLu,
    l8: C3,
    l9: Sppf,
}

impl CspDarknet53 {
    fn load(vb: VarBuilder) -> Result<Self> {
        let l0 = ConvBnSiLu::load(vb.pp("model.0"), 3, 32, 6, 2, Some(2))?;
        let l1 = ConvBnSiLu::load(vb.pp("model.1"), 32, 64, 3, 2, None)?;
        let l2 = C3::load(vb.pp("model.2"), 64, 64, 1, true, 0.5)?;
        let l3 = ConvBnSiLu::load(vb.pp("model.3"), 64, 128, 3, 2, None)?;
        let l4 = C3::load(vb.pp("model.4"), 128, 128, 2, true, 0.5)?;
        let l5 = ConvBnSiLu::load(vb.pp("model.5"), 128, 256, 3, 2, None)?;
        let l6 = C3::load(vb.pp("model.6"), 256, 256, 3, true, 0.5)?;
        let l7 = ConvBnSiLu::load(vb.pp("model.7"), 256, 512, 3, 2, None)?;
        let l8 = C3::load(vb.pp("model.8"), 512, 512, 1, true, 0.5)?;
        let l9 = Sppf::load(vb.pp("model.9"), 512, 512, 5)?;
        Ok(Self {
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Vec<Tensor>)> {
        let x0 = self.l0.forward(xs)?;
        let x1 = self.l1.forward(&x0)?;
        let x2 = self.l2.forward(&x1)?;
        let x3 = self.l3.forward(&x2)?;
        let x4 = self.l4.forward(&x3)?;
        let x5 = self.l5.forward(&x4)?;
        let x6 = self.l6.forward(&x5)?;
        let x7 = self.l7.forward(&x6)?;
        let x8 = self.l8.forward(&x7)?;
        let x9 = self.l9.forward(&x8)?;

        let feature_maps = vec![x1.clone(), x3.clone(), x5.clone(), x7.clone(), x9.clone()];
        Ok((x4, x6, x9, feature_maps))
    }
}

#[derive(Debug)]
struct PanetNeck {
    up: Upsample,
    l10: ConvBnSiLu,
    l13: C3,
    l14: ConvBnSiLu,
    l17: C3,
    l18: ConvBnSiLu,
    l20: C3,
    l21: ConvBnSiLu,
    l23: C3,
}

impl PanetNeck {
    fn load(vb: VarBuilder) -> Result<Self> {
        let up = Upsample::new(2);
        let l10 = ConvBnSiLu::load(vb.pp("model.10"), 512, 256, 1, 1, None)?;
        let l13 = C3::load(vb.pp("model.13"), 512, 256, 1, false, 0.5)?;
        let l14 = ConvBnSiLu::load(vb.pp("model.14"), 256, 128, 1, 1, None)?;
        let l17 = C3::load(vb.pp("model.17"), 256, 128, 1, false, 0.5)?;
        let l18 = ConvBnSiLu::load(vb.pp("model.18"), 128, 128, 3, 2, None)?;
        let l20 = C3::load(vb.pp("model.20"), 256, 256, 1, false, 0.5)?;
        let l21 = ConvBnSiLu::load(vb.pp("model.21"), 256, 256, 3, 2, None)?;
        let l23 = C3::load(vb.pp("model.23"), 512, 512, 1, false, 0.5)?;
        Ok(Self {
            up,
            l10,
            l13,
            l14,
            l17,
            l18,
            l20,
            l21,
            l23,
        })
    }

    fn forward(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<[Tensor; 3]> {
        let x10 = self.l10.forward(p5)?;
        let x11 = self.up.forward(&x10)?;
        let x12 = Tensor::cat(&[&x11, p4], 1)?;
        let x13 = self.l13.forward(&x12)?;
        let x14 = self.l14.forward(&x13)?;
        let x15 = self.up.forward(&x14)?;
        let x16 = Tensor::cat(&[&x15, p3], 1)?;
        let x17 = self.l17.forward(&x16)?;
        let x18 = self.l18.forward(&x17)?;
        let x19 = Tensor::cat(&[&x18, &x14], 1)?;
        let x20 = self.l20.forward(&x19)?;
        let x21 = self.l21.forward(&x20)?;
        let x22 = Tensor::cat(&[&x21, &x10], 1)?;
        let x23 = self.l23.forward(&x22)?;
        Ok([x17, x20, x23])
    }
}

#[derive(Debug)]
struct YoloV3Head {
    convs: [Conv2d; 3],
    anchors: Tensor,
    strides: [f32; 3],
    num_outputs: usize,
    num_anchors: usize,
}

impl YoloV3Head {
    fn load(vb: VarBuilder, num_classes: usize, num_anchors: usize) -> Result<Self> {
        let num_outputs = num_classes + 5;

        let conv0 = conv2d(
            128,
            num_outputs * num_anchors,
            1,
            Default::default(),
            vb.pp("model.24.m.0"),
        )?;
        let conv1 = conv2d(
            256,
            num_outputs * num_anchors,
            1,
            Default::default(),
            vb.pp("model.24.m.1"),
        )?;
        let conv2 = conv2d(
            512,
            num_outputs * num_anchors,
            1,
            Default::default(),
            vb.pp("model.24.m.2"),
        )?;
        let anchors = vb.pp("model.24").get((3, num_anchors, 2), "anchors")?;
        Ok(Self {
            convs: [conv0, conv1, conv2],
            anchors,
            strides: [8.0, 16.0, 32.0],
            num_outputs,
            num_anchors,
        })
    }

    fn make_grid(
        &self,
        layer_idx: usize,
        nx: usize,
        ny: usize,
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let gx = Tensor::arange(0, nx as u32, dev)?.to_dtype(DType::F32)?;
        let gy = Tensor::arange(0, ny as u32, dev)?.to_dtype(DType::F32)?;

        let gx = gx.reshape((1, 1, 1, nx))?.repeat((1, 1, ny, 1))?;
        let gy = gy.reshape((1, 1, ny, 1))?.repeat((1, 1, 1, nx))?;
        let grid = Tensor::stack(&[&gx, &gy], 4)?;

        let anchor = self.anchors.to_device(dev)?.i(layer_idx)?;
        let anchor_grid = (anchor
            .unsqueeze(0)?
            .unsqueeze(2)?
            .unsqueeze(3)?
            .repeat((1, 1, ny, nx, 1))?
            * self.strides[layer_idx] as f64)?;
        Ok((grid, anchor_grid))
    }

    fn forward(&self, inputs: &[Tensor; 3]) -> Result<(Tensor, Vec<Tensor>)> {
        let mut outputs = Vec::with_capacity(self.convs.len());
        let mut maps = Vec::with_capacity(self.convs.len());
        for (idx, (conv, xs)) in self.convs.iter().zip(inputs).enumerate() {
            let xs = conv.forward(xs)?;
            let (b, _, h, w) = xs.dims4()?;
            let xs = xs
                .reshape((b, self.num_anchors, self.num_outputs, h, w))?
                .permute((0, 1, 3, 4, 2))?;
            let raw = xs.clone();
            let (grid, anchor_grid) = self.make_grid(idx, w, h, xs.device())?;
            let y = candle_nn::ops::sigmoid(&xs)?;
            let xy = y.i((.., .., .., .., 0..2))?;
            let xy = (xy * 2.0f64)?;
            let xy = (xy - 0.5f64)?;
            let xy = xy.broadcast_add(&grid)?;
            let xy = (xy * self.strides[idx] as f64)?;
            let wh = y.i((.., .., .., .., 2..4))?;
            let wh = (wh * 2.0f64)?.sqr()?.broadcast_mul(&anchor_grid)?;
            let rest = y.i((.., .., .., .., 4..))?;
            let y = Tensor::cat(&[&xy, &wh, &rest], 4)?;
            outputs.push(y.reshape((b, self.num_anchors * h * w, self.num_outputs))?);
            maps.push(raw);
        }
        let pred = Tensor::cat(outputs.as_slice(), 1)?;
        Ok((pred, maps))
    }
}

#[derive(Debug)]
pub struct YoloV5 {
    backbone: CspDarknet53,
    neck: PanetNeck,
    head: YoloV3Head,
}

impl YoloV5 {
    pub fn load(vb: VarBuilder, num_classes: usize, num_anchors: usize) -> Result<Self> {
        let backbone = CspDarknet53::load(vb.clone())?;
        let neck = PanetNeck::load(vb.clone())?;
        let head = YoloV3Head::load(vb, num_classes, num_anchors)?;

        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (p3, p4, p5, feature_maps) = self.backbone.forward(xs)?;
        let detection_features = self.neck.forward(&p3, &p4, &p5)?;
        let (predictions, _) = self.head.forward(&detection_features)?;

        Ok((predictions, feature_maps))
    }
}
