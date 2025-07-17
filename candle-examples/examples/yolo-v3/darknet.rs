use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{batch_norm, conv2d, conv2d_no_bias, Func, Module, VarBuilder};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
struct Block {
    block_type: String,
    parameters: BTreeMap<String, String>,
}

impl Block {
    fn get(&self, key: &str) -> Result<&str> {
        match self.parameters.get(key) {
            None => candle::bail!("cannot find {} in {}", key, self.block_type),
            Some(value) => Ok(value),
        }
    }
}

#[derive(Debug)]
pub struct Darknet {
    blocks: Vec<Block>,
    parameters: BTreeMap<String, String>,
}

impl Darknet {
    fn get(&self, key: &str) -> Result<&str> {
        match self.parameters.get(key) {
            None => candle::bail!("cannot find {} in net parameters", key),
            Some(value) => Ok(value),
        }
    }
}

struct Accumulator {
    block_type: Option<String>,
    parameters: BTreeMap<String, String>,
    net: Darknet,
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            block_type: None,
            parameters: BTreeMap::new(),
            net: Darknet {
                blocks: vec![],
                parameters: BTreeMap::new(),
            },
        }
    }

    fn finish_block(&mut self) {
        match &self.block_type {
            None => (),
            Some(block_type) => {
                if block_type == "net" {
                    self.net.parameters = self.parameters.clone();
                } else {
                    let block = Block {
                        block_type: block_type.to_string(),
                        parameters: self.parameters.clone(),
                    };
                    self.net.blocks.push(block);
                }
                self.parameters.clear();
            }
        }
        self.block_type = None;
    }
}

pub fn parse_config<T: AsRef<Path>>(path: T) -> Result<Darknet> {
    let file = File::open(path.as_ref())?;
    let mut acc = Accumulator::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.trim();
        if line.starts_with('[') {
            if !line.ends_with(']') {
                candle::bail!("line does not end with ']' {line}")
            }
            let line = &line[1..line.len() - 1];
            acc.finish_block();
            acc.block_type = Some(line.to_string());
        } else {
            let key_value: Vec<&str> = line.splitn(2, '=').collect();
            if key_value.len() != 2 {
                candle::bail!("missing equal {line}")
            }
            let prev = acc.parameters.insert(
                key_value[0].trim().to_owned(),
                key_value[1].trim().to_owned(),
            );
            if prev.is_some() {
                candle::bail!("multiple value for key {}", line)
            }
        }
    }
    acc.finish_block();
    Ok(acc.net)
}

enum Bl {
    Layer(Box<dyn candle_nn::Module + Send + Sync>),
    Route(Vec<usize>),
    Shortcut(usize),
    Yolo(usize, Vec<(usize, usize)>),
}

fn conv(vb: VarBuilder, index: usize, p: usize, b: &Block) -> Result<(usize, Bl)> {
    let activation = b.get("activation")?;
    let filters = b.get("filters")?.parse::<usize>()?;
    let pad = b.get("pad")?.parse::<usize>()?;
    let size = b.get("size")?.parse::<usize>()?;
    let stride = b.get("stride")?.parse::<usize>()?;
    let padding = if pad != 0 { (size - 1) / 2 } else { 0 };
    let (bn, bias) = match b.parameters.get("batch_normalize") {
        Some(p) if p.parse::<usize>()? != 0 => {
            let bn = batch_norm(filters, 1e-5, vb.pp(format!("batch_norm_{index}")))?;
            (Some(bn), false)
        }
        Some(_) | None => (None, true),
    };
    let conv_cfg = candle_nn::Conv2dConfig {
        stride,
        padding,
        groups: 1,
        dilation: 1,
        cudnn_fwd_algo: None,
    };
    let conv = if bias {
        conv2d(p, filters, size, conv_cfg, vb.pp(format!("conv_{index}")))?
    } else {
        conv2d_no_bias(p, filters, size, conv_cfg, vb.pp(format!("conv_{index}")))?
    };
    let leaky = match activation {
        "leaky" => true,
        "linear" => false,
        otherwise => candle::bail!("unsupported activation {}", otherwise),
    };
    let func = candle_nn::func(move |xs| {
        let xs = conv.forward(xs)?;
        let xs = match &bn {
            Some(bn) => xs.apply_t(bn, false)?,
            None => xs,
        };
        let xs = if leaky {
            xs.maximum(&(&xs * 0.1)?)?
        } else {
            xs
        };
        Ok(xs)
    });
    Ok((filters, Bl::Layer(Box::new(func))))
}

fn upsample(prev_channels: usize) -> Result<(usize, Bl)> {
    let layer = candle_nn::func(|xs| {
        let (_n, _c, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(2 * h, 2 * w)
    });
    Ok((prev_channels, Bl::Layer(Box::new(layer))))
}

fn int_list_of_string(s: &str) -> Result<Vec<i64>> {
    let res: std::result::Result<Vec<_>, _> =
        s.split(',').map(|xs| xs.trim().parse::<i64>()).collect();
    Ok(res?)
}

fn usize_of_index(index: usize, i: i64) -> usize {
    if i >= 0 {
        i as usize
    } else {
        (index as i64 + i) as usize
    }
}

fn route(index: usize, p: &[(usize, Bl)], block: &Block) -> Result<(usize, Bl)> {
    let layers = int_list_of_string(block.get("layers")?)?;
    let layers: Vec<usize> = layers
        .into_iter()
        .map(|l| usize_of_index(index, l))
        .collect();
    let channels = layers.iter().map(|&l| p[l].0).sum();
    Ok((channels, Bl::Route(layers)))
}

fn shortcut(index: usize, p: usize, block: &Block) -> Result<(usize, Bl)> {
    let from = block.get("from")?.parse::<i64>()?;
    Ok((p, Bl::Shortcut(usize_of_index(index, from))))
}

fn yolo(p: usize, block: &Block) -> Result<(usize, Bl)> {
    let classes = block.get("classes")?.parse::<usize>()?;
    let flat = int_list_of_string(block.get("anchors")?)?;
    if flat.len() % 2 != 0 {
        candle::bail!("even number of anchors");
    }
    let flat = flat.into_iter().map(|i| i as usize).collect::<Vec<_>>();
    let anchors: Vec<_> = (0..(flat.len() / 2))
        .map(|i| (flat[2 * i], flat[2 * i + 1]))
        .collect();
    let mask = int_list_of_string(block.get("mask")?)?;
    let anchors = mask.into_iter().map(|i| anchors[i as usize]).collect();
    Ok((p, Bl::Yolo(classes, anchors)))
}

fn detect(
    xs: &Tensor,
    image_height: usize,
    classes: usize,
    anchors: &[(usize, usize)],
) -> Result<Tensor> {
    let (bsize, _channels, height, _width) = xs.dims4()?;
    let stride = image_height / height;
    let grid_size = image_height / stride;
    let bbox_attrs = 5 + classes;
    let nanchors = anchors.len();
    let xs = xs
        .reshape((bsize, bbox_attrs * nanchors, grid_size * grid_size))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((bsize, grid_size * grid_size * nanchors, bbox_attrs))?;
    let grid = Tensor::arange(0u32, grid_size as u32, &Device::Cpu)?;
    let a = grid.repeat((grid_size, 1))?;
    let b = a.t()?.contiguous()?;
    let x_offset = a.flatten_all()?.unsqueeze(1)?;
    let y_offset = b.flatten_all()?.unsqueeze(1)?;
    let xy_offset = Tensor::cat(&[&x_offset, &y_offset], 1)?
        .repeat((1, nanchors))?
        .reshape((grid_size * grid_size * nanchors, 2))?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;
    let anchors: Vec<f32> = anchors
        .iter()
        .flat_map(|&(x, y)| vec![x as f32 / stride as f32, y as f32 / stride as f32].into_iter())
        .collect();
    let anchors = Tensor::new(anchors.as_slice(), &Device::Cpu)?
        .reshape((anchors.len() / 2, 2))?
        .repeat((grid_size * grid_size, 1))?
        .unsqueeze(0)?;
    let ys02 = xs.i((.., .., 0..2))?;
    let ys24 = xs.i((.., .., 2..4))?;
    let ys4 = xs.i((.., .., 4..))?;
    let ys02 = (candle_nn::ops::sigmoid(&ys02)?.add(&xy_offset)? * stride as f64)?;
    let ys24 = (ys24.exp()?.mul(&anchors)? * stride as f64)?;
    let ys4 = candle_nn::ops::sigmoid(&ys4)?;
    let ys = Tensor::cat(&[ys02, ys24, ys4], 2)?;
    Ok(ys)
}

impl Darknet {
    pub fn height(&self) -> Result<usize> {
        let image_height = self.get("height")?.parse::<usize>()?;
        Ok(image_height)
    }

    pub fn width(&self) -> Result<usize> {
        let image_width = self.get("width")?.parse::<usize>()?;
        Ok(image_width)
    }

    pub fn build_model(&self, vb: VarBuilder) -> Result<Func> {
        let mut blocks: Vec<(usize, Bl)> = vec![];
        let mut prev_channels: usize = 3;
        for (index, block) in self.blocks.iter().enumerate() {
            let channels_and_bl = match block.block_type.as_str() {
                "convolutional" => conv(vb.pp(index.to_string()), index, prev_channels, block)?,
                "upsample" => upsample(prev_channels)?,
                "shortcut" => shortcut(index, prev_channels, block)?,
                "route" => route(index, &blocks, block)?,
                "yolo" => yolo(prev_channels, block)?,
                otherwise => candle::bail!("unsupported block type {}", otherwise),
            };
            prev_channels = channels_and_bl.0;
            blocks.push(channels_and_bl);
        }
        let image_height = self.height()?;
        let func = candle_nn::func(move |xs| {
            let mut prev_ys: Vec<Tensor> = vec![];
            let mut detections: Vec<Tensor> = vec![];
            for (_, b) in blocks.iter() {
                let ys = match b {
                    Bl::Layer(l) => {
                        let xs = prev_ys.last().unwrap_or(xs);
                        l.forward(xs)?
                    }
                    Bl::Route(layers) => {
                        let layers: Vec<_> = layers.iter().map(|&i| &prev_ys[i]).collect();
                        Tensor::cat(&layers, 1)?
                    }
                    Bl::Shortcut(from) => (prev_ys.last().unwrap() + prev_ys.get(*from).unwrap())?,
                    Bl::Yolo(classes, anchors) => {
                        let xs = prev_ys.last().unwrap_or(xs);
                        detections.push(detect(xs, image_height, *classes, anchors)?);
                        Tensor::new(&[0u32], &Device::Cpu)?
                    }
                };
                prev_ys.push(ys);
            }
            Tensor::cat(&detections, 1)
        });
        Ok(func)
    }
}
