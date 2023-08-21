#![allow(dead_code)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod coco_classes;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, BatchNorm, Conv2d, Conv2dConfig, Module, VarBuilder,
};
use clap::{Parser, ValueEnum};
use image::{DynamicImage, ImageBuffer};

const CONFIDENCE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;

// Model architecture from https://github.com/ultralytics/ultralytics/issues/189
// https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py

#[derive(Clone, Copy, PartialEq, Debug)]
struct Multiples {
    depth: f64,
    width: f64,
    ratio: f64,
}

impl Multiples {
    fn n() -> Self {
        Self {
            depth: 0.33,
            width: 0.25,
            ratio: 2.0,
        }
    }
    fn s() -> Self {
        Self {
            depth: 0.33,
            width: 0.50,
            ratio: 2.0,
        }
    }
    fn m() -> Self {
        Self {
            depth: 0.67,
            width: 0.75,
            ratio: 1.5,
        }
    }
    fn l() -> Self {
        Self {
            depth: 1.00,
            width: 1.00,
            ratio: 1.0,
        }
    }
    fn x() -> Self {
        Self {
            depth: 1.00,
            width: 1.25,
            ratio: 1.0,
        }
    }

    fn filters(&self) -> (usize, usize, usize) {
        let f1 = (256. * self.width) as usize;
        let f2 = (512. * self.width) as usize;
        let f3 = (512. * self.width * self.ratio) as usize;
        (f1, f2, f3)
    }
}

#[derive(Debug)]
struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    fn new(scale_factor: usize) -> Result<Self> {
        Ok(Upsample { scale_factor })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let (_b_size, _channels, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(self.scale_factor * h, self.scale_factor * w)
    }
}

#[derive(Debug)]
struct ConvBlock {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBlock {
    fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        k: usize,
        stride: usize,
        padding: Option<usize>,
    ) -> Result<Self> {
        let padding = padding.unwrap_or(k / 2);
        let cfg = Conv2dConfig { padding, stride };
        let conv = conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?;
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        let xs = self.bn.forward(&xs)?;
        candle_nn::ops::silu(&xs)
    }
}

#[derive(Debug)]
struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
}

impl Bottleneck {
    fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool) -> Result<Self> {
        let channel_factor = 1.;
        let c_ = (c2 as f64 * channel_factor) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 3, 1, None)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_, c2, 3, 1, None)?;
        let residual = c1 == c2 && shortcut;
        Ok(Self { cv1, cv2, residual })
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
struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
    c: usize,
}

impl C2f {
    fn load(vb: VarBuilder, c1: usize, c2: usize, n: usize, shortcut: bool) -> Result<Self> {
        let c = (c2 as f64 * 0.5) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, 2 * c, 1, 1, None)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1, None)?;
        let mut bottleneck = Vec::with_capacity(n);
        for idx in 0..n {
            let b = Bottleneck::load(vb.pp(&format!("bottleneck.{idx}")), c, c, shortcut)?;
            bottleneck.push(b)
        }
        Ok(Self {
            cv1,
            cv2,
            bottleneck,
            c,
        })
    }
}

impl Module for C2f {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.cv1.forward(xs)?;
        let mut ys = ys.chunk(2, 1)?;
        for m in self.bottleneck.iter() {
            ys.push(m.forward(ys.last().unwrap())?)
        }
        let zs = Tensor::cat(ys.as_slice(), 1)?;
        self.cv2.forward(&zs)
    }
}

#[derive(Debug)]
struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
}

impl Sppf {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let c_ = c1 / 2;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 1, 1, None)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_ * 4, c2, 1, 1, None)?;
        Ok(Self { cv1, cv2, k })
    }
}

impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, _, _) = xs.dims4()?;
        let xs = self.cv1.forward(xs)?;
        let xs2 = xs
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d((self.k, self.k), (1, 1))?;
        let xs3 = xs2
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d((self.k, self.k), (1, 1))?;
        let xs4 = xs3
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d((self.k, self.k), (1, 1))?;
        self.cv2.forward(&Tensor::cat(&[&xs, &xs2, &xs3, &xs4], 1)?)
    }
}

#[derive(Debug)]
struct Dfl {
    conv: Conv2d,
    num_classes: usize,
}

impl Dfl {
    fn load(vb: VarBuilder, num_classes: usize) -> Result<Self> {
        let conv = conv2d_no_bias(num_classes, 1, 1, Default::default(), vb.pp("conv"))?;
        Ok(Self { conv, num_classes })
    }
}

impl Module for Dfl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, _channels, anchors) = xs.dims3()?;
        let xs = xs
            .reshape((b_sz, 4, self.num_classes, anchors))?
            .transpose(2, 1)?;
        let xs = candle_nn::ops::softmax(&xs, 1)?;
        self.conv.forward(&xs)?.reshape((b_sz, 4, anchors))
    }
}

#[derive(Debug)]
struct DarkNet {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: ConvBlock,
    b3_1: C2f,
    b4_0: ConvBlock,
    b4_1: C2f,
    b5: Sppf,
}

impl DarkNet {
    fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let b1_0 = ConvBlock::load(vb.pp("b1.0"), 3, (64. * w) as usize, 3, 2, Some(1))?;
        let b1_1 = ConvBlock::load(
            vb.pp("b1.1"),
            (64. * w) as usize,
            (128. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let b2_0 = C2f::load(
            vb.pp("b2.0"),
            (128. * w) as usize,
            (128. * w) as usize,
            (3. * d).round() as usize,
            true,
        )?;
        let b2_1 = ConvBlock::load(
            vb.pp("b2.1"),
            (128. * w) as usize,
            (256. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let b2_2 = C2f::load(
            vb.pp("b2.2"),
            (256. * w) as usize,
            (256. * w) as usize,
            (6. * d).round() as usize,
            true,
        )?;
        let b3_0 = ConvBlock::load(
            vb.pp("b3.0"),
            (256. * w) as usize,
            (512. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let b3_1 = C2f::load(
            vb.pp("b3.1"),
            (512. * w) as usize,
            (512. * w) as usize,
            (6. * d).round() as usize,
            true,
        )?;
        let b4_0 = ConvBlock::load(
            vb.pp("b4.0"),
            (512. * w) as usize,
            (512. * w * r) as usize,
            3,
            2,
            Some(1),
        )?;
        let b4_1 = C2f::load(
            vb.pp("b4.1"),
            (512. * w * r) as usize,
            (512. * w * r) as usize,
            (3. * d).round() as usize,
            true,
        )?;
        let b5 = Sppf::load(
            vb.pp("b5.0"),
            (512. * w * r) as usize,
            (512. * w * r) as usize,
            5,
        )?;
        Ok(Self {
            b1_0,
            b1_1,
            b2_0,
            b2_1,
            b2_2,
            b3_0,
            b3_1,
            b4_0,
            b4_1,
            b5,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x1 = self.b1_1.forward(&self.b1_0.forward(xs)?)?;
        let x2 = self
            .b2_2
            .forward(&self.b2_1.forward(&self.b2_0.forward(&x1)?)?)?;
        let x3 = self.b3_1.forward(&self.b3_0.forward(&x2)?)?;
        let x4 = self.b4_1.forward(&self.b4_0.forward(&x3)?)?;
        let x5 = self.b5.forward(&x4)?;
        Ok((x2, x3, x5))
    }
}

#[derive(Debug)]
struct YoloV8Neck {
    up: Upsample,
    n1: C2f,
    n2: C2f,
    n3: ConvBlock,
    n4: C2f,
    n5: ConvBlock,
    n6: C2f,
}

impl YoloV8Neck {
    fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let up = Upsample::new(2)?;
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let n = (3. * d).round() as usize;
        let n1 = C2f::load(
            vb.pp("n1"),
            (512. * w * (1. + r)) as usize,
            (512. * w) as usize,
            n,
            false,
        )?;
        let n2 = C2f::load(
            vb.pp("n2"),
            (768. * w) as usize,
            (256. * w) as usize,
            n,
            false,
        )?;
        let n3 = ConvBlock::load(
            vb.pp("n3"),
            (256. * w) as usize,
            (256. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let n4 = C2f::load(
            vb.pp("n4"),
            (768. * w) as usize,
            (512. * w) as usize,
            n,
            false,
        )?;
        let n5 = ConvBlock::load(
            vb.pp("n5"),
            (512. * w) as usize,
            (512. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let n6 = C2f::load(
            vb.pp("n6"),
            (512. * w * (1. + r)) as usize,
            (512. * w * r) as usize,
            n,
            false,
        )?;
        Ok(Self {
            up,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
        })
    }

    fn forward(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self
            .n1
            .forward(&Tensor::cat(&[&self.up.forward(p5)?, p4], 1)?)?;
        let head_1 = self
            .n2
            .forward(&Tensor::cat(&[&self.up.forward(&x)?, p3], 1)?)?;
        let head_2 = self
            .n4
            .forward(&Tensor::cat(&[&self.n3.forward(&head_1)?, &x], 1)?)?;
        let head_3 = self
            .n6
            .forward(&Tensor::cat(&[&self.n5.forward(&head_2)?, p5], 1)?)?;
        Ok((head_1, head_2, head_3))
    }
}

#[derive(Debug)]
struct DetectionHead {
    dfl: Dfl,
    cv2: [(ConvBlock, ConvBlock, Conv2d); 3],
    cv3: [(ConvBlock, ConvBlock, Conv2d); 3],
    ch: usize,
    no: usize,
}

fn make_anchors(
    xs0: &Tensor,
    xs1: &Tensor,
    xs2: &Tensor,
    (s0, s1, s2): (usize, usize, usize),
    grid_cell_offset: f64,
) -> Result<(Tensor, Tensor)> {
    let dev = xs0.device();
    let mut anchor_points = vec![];
    let mut stride_tensor = vec![];
    for (xs, stride) in [(xs0, s0), (xs1, s1), (xs2, s2)] {
        // xs is only used to extract the h and w dimensions.
        let (_, _, h, w) = xs.dims4()?;
        let sx = (Tensor::arange(0, w as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sy = (Tensor::arange(0, h as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sx = sx
            .reshape((1, sx.elem_count()))?
            .repeat((h, 1))?
            .flatten_all()?;
        let sy = sy
            .reshape((sy.elem_count(), 1))?
            .repeat((1, w))?
            .flatten_all()?;
        anchor_points.push(Tensor::stack(&[&sx, &sy], D::Minus1)?);
        stride_tensor.push((Tensor::ones(h * w, DType::F32, dev)? * stride as f64)?);
    }
    let anchor_points = Tensor::cat(anchor_points.as_slice(), 0)?;
    let stride_tensor = Tensor::cat(stride_tensor.as_slice(), 0)?.unsqueeze(1)?;
    Ok((anchor_points, stride_tensor))
}
fn dist2bbox(distance: &Tensor, anchor_points: &Tensor) -> Result<Tensor> {
    let chunks = distance.chunk(2, 1)?;
    let lt = &chunks[0];
    let rb = &chunks[1];
    let x1y1 = anchor_points.sub(lt)?;
    let x2y2 = anchor_points.add(rb)?;
    let c_xy = ((&x1y1 + &x2y2)? * 0.5)?;
    let wh = (&x2y2 - &x1y1)?;
    Tensor::cat(&[c_xy, wh], 1)
}

impl DetectionHead {
    fn load(vb: VarBuilder, nc: usize, filters: (usize, usize, usize)) -> Result<Self> {
        let ch = 16;
        let dfl = Dfl::load(vb.pp("dfl"), ch)?;
        let c1 = usize::max(filters.0, nc);
        let c2 = usize::max(filters.0 / 4, ch * 4);
        let cv3 = [
            Self::load_cv3(vb.pp("cv3.0"), c1, nc, filters.0)?,
            Self::load_cv3(vb.pp("cv3.1"), c1, nc, filters.1)?,
            Self::load_cv3(vb.pp("cv3.2"), c1, nc, filters.2)?,
        ];
        let cv2 = [
            Self::load_cv2(vb.pp("cv2.0"), c2, ch, filters.0)?,
            Self::load_cv2(vb.pp("cv2.1"), c2, ch, filters.1)?,
            Self::load_cv2(vb.pp("cv2.2"), c2, ch, filters.2)?,
        ];
        let no = nc + ch * 4;
        Ok(Self {
            dfl,
            cv2,
            cv3,
            ch,
            no,
        })
    }

    fn load_cv3(
        vb: VarBuilder,
        c1: usize,
        nc: usize,
        filter: usize,
    ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        let block0 = ConvBlock::load(vb.pp("0"), filter, c1, 3, 1, None)?;
        let block1 = ConvBlock::load(vb.pp("1"), c1, c1, 3, 1, None)?;
        let conv = conv2d(c1, nc, 1, Default::default(), vb.pp("2"))?;
        Ok((block0, block1, conv))
    }

    fn load_cv2(
        vb: VarBuilder,
        c2: usize,
        ch: usize,
        filter: usize,
    ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        let block0 = ConvBlock::load(vb.pp("0"), filter, c2, 3, 1, None)?;
        let block1 = ConvBlock::load(vb.pp("1"), c2, c2, 3, 1, None)?;
        let conv = conv2d(c2, 4 * ch, 1, Default::default(), vb.pp("2"))?;
        Ok((block0, block1, conv))
    }

    fn forward(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<Tensor> {
        let forward_cv = |xs, i: usize| {
            let xs_2 = self.cv2[i].0.forward(xs)?;
            let xs_2 = self.cv2[i].1.forward(&xs_2)?;
            let xs_2 = self.cv2[i].2.forward(&xs_2)?;

            let xs_3 = self.cv3[i].0.forward(xs)?;
            let xs_3 = self.cv3[i].1.forward(&xs_3)?;
            let xs_3 = self.cv3[i].2.forward(&xs_3)?;
            Tensor::cat(&[&xs_2, &xs_3], 1)
        };
        let xs0 = forward_cv(xs0, 0)?;
        let xs1 = forward_cv(xs1, 1)?;
        let xs2 = forward_cv(xs2, 2)?;

        let (anchors, strides) = make_anchors(&xs0, &xs1, &xs2, (8, 16, 32), 0.5)?;
        let anchors = anchors.transpose(0, 1)?;
        let strides = strides.transpose(0, 1)?;

        let reshape = |xs: &Tensor| {
            let d = xs.dim(0)?;
            let el = xs.elem_count();
            xs.reshape((d, self.no, el / (d * self.no)))
        };
        let ys0 = reshape(&xs0)?;
        let ys1 = reshape(&xs1)?;
        let ys2 = reshape(&xs2)?;

        let x_cat = Tensor::cat(&[ys0, ys1, ys2], 2)?;
        let box_ = x_cat.i((.., ..self.ch * 4))?;
        let cls = x_cat.i((.., self.ch * 4..))?;

        let dbox = dist2bbox(&self.dfl.forward(&box_)?, &anchors.unsqueeze(0)?)?;
        let dbox = dbox.broadcast_mul(&strides)?;
        Tensor::cat(&[dbox, candle_nn::ops::sigmoid(&cls)?], 1)
    }
}

#[derive(Debug)]
struct YoloV8 {
    net: DarkNet,
    fpn: YoloV8Neck,
    head: DetectionHead,
}

impl YoloV8 {
    fn load(vb: VarBuilder, m: Multiples, num_classes: usize) -> Result<Self> {
        let net = DarkNet::load(vb.pp("net"), m)?;
        let fpn = YoloV8Neck::load(vb.pp("fpn"), m)?;
        let head = DetectionHead::load(vb.pp("head"), num_classes, m.filters())?;
        Ok(Self { net, fpn, head })
    }
}

impl Module for YoloV8 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (xs1, xs2, xs3) = self.net.forward(xs)?;
        let (xs1, xs2, xs3) = self.fpn.forward(&xs1, &xs2, &xs3)?;
        self.head.forward(&xs1, &xs2, &xs3)
    }
}

#[derive(Debug, Clone, Copy)]
struct Bbox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    confidence: f32,
}

// Intersection over union of two bounding boxes.
fn iou(b1: &Bbox, b2: &Bbox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

// Assumes x1 <= x2 and y1 <= y2
pub fn draw_rect(
    img: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    x1: u32,
    x2: u32,
    y1: u32,
    y2: u32,
) {
    for x in x1..=x2 {
        let pixel = img.get_pixel_mut(x, y1);
        *pixel = image::Rgb([255, 0, 0]);
        let pixel = img.get_pixel_mut(x, y2);
        *pixel = image::Rgb([255, 0, 0]);
    }
    for y in y1..=y2 {
        let pixel = img.get_pixel_mut(x1, y);
        *pixel = image::Rgb([255, 0, 0]);
        let pixel = img.get_pixel_mut(x2, y);
        *pixel = image::Rgb([255, 0, 0]);
    }
}

pub fn report(pred: &Tensor, img: DynamicImage, w: usize, h: usize) -> Result<DynamicImage> {
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > CONFIDENCE_THRESHOLD {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > NMS_THRESHOLD {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", coco_classes::NAMES[class_index], b);
            let xmin = ((b.xmin * w_ratio) as u32).clamp(0, initial_w - 1);
            let ymin = ((b.ymin * h_ratio) as u32).clamp(0, initial_h - 1);
            let xmax = ((b.xmax * w_ratio) as u32).clamp(0, initial_w - 1);
            let ymax = ((b.ymax * h_ratio) as u32).clamp(0, initial_h - 1);
            draw_rect(&mut img, xmin, xmax, ymin, ymax);
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    which: Which,

    images: Vec<String>,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-yolo-v8".to_string());
                let filename = match self.which {
                    Which::N => "yolov8n.safetensors",
                    Which::S => "yolov8s.safetensors",
                    Which::M => "yolov8m.safetensors",
                    Which::L => "yolov8l.safetensors",
                    Which::X => "yolov8x.safetensors",
                };
                api.get(filename)?
            }
        };
        Ok(path)
    }
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Create the model and load the weights from the file.
    let multiples = match args.which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model = args.model()?;
    let weights = unsafe { candle::safetensors::MmapedFile::new(model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &Device::Cpu);
    let model = YoloV8::load(vb, multiples, /* num_classes=*/ 80)?;
    println!("model loaded");
    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_name = std::path::PathBuf::from(image_name);
        let original_image = image::io::Reader::open(&image_name)?
            .decode()
            .map_err(candle::Error::wrap)?;
        let image = {
            let data = original_image
                .resize_exact(640, 640, image::imageops::FilterType::Triangle)
                .to_rgb8()
                .into_raw();
            Tensor::from_vec(data, (640, 640, 3), &Device::Cpu)?.permute((2, 0, 1))?
        };
        let image = (image.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = model.forward(&image)?.squeeze(0)?;
        println!("generated predictions {predictions:?}");
        let image = report(&predictions, original_image, 640, 640)?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image.save(image_name)?
    }

    Ok(())
}
