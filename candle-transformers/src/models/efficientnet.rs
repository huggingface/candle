use candle::{Result, Tensor, D};
use candle_nn as nn;
use nn::{Module, VarBuilder};

// Based on the Python version from torchvision.
// https://github.com/pytorch/vision/blob/0d75d9e5516f446c9c0ef93bd4ed9fea13992d06/torchvision/models/efficientnet.py#L47
#[derive(Debug, Clone, Copy)]
pub struct MBConvConfig {
    expand_ratio: f64,
    kernel: usize,
    stride: usize,
    input_channels: usize,
    out_channels: usize,
    num_layers: usize,
}

fn make_divisible(v: f64, divisor: usize) -> usize {
    let min_value = divisor;
    let new_v = usize::max(
        min_value,
        (v + divisor as f64 * 0.5) as usize / divisor * divisor,
    );
    if (new_v as f64) < 0.9 * v {
        new_v + divisor
    } else {
        new_v
    }
}

fn bneck_confs(width_mult: f64, depth_mult: f64) -> Vec<MBConvConfig> {
    let bneck_conf = |e, k, s, i, o, n| {
        let input_channels = make_divisible(i as f64 * width_mult, 8);
        let out_channels = make_divisible(o as f64 * width_mult, 8);
        let num_layers = (n as f64 * depth_mult).ceil() as usize;
        MBConvConfig {
            expand_ratio: e,
            kernel: k,
            stride: s,
            input_channels,
            out_channels,
            num_layers,
        }
    };
    vec![
        bneck_conf(1., 3, 1, 32, 16, 1),
        bneck_conf(6., 3, 2, 16, 24, 2),
        bneck_conf(6., 5, 2, 24, 40, 2),
        bneck_conf(6., 3, 2, 40, 80, 3),
        bneck_conf(6., 5, 1, 80, 112, 3),
        bneck_conf(6., 5, 2, 112, 192, 4),
        bneck_conf(6., 3, 1, 192, 320, 1),
    ]
}

impl MBConvConfig {
    pub fn b0() -> Vec<Self> {
        bneck_confs(1.0, 1.0)
    }
    pub fn b1() -> Vec<Self> {
        bneck_confs(1.0, 1.1)
    }
    pub fn b2() -> Vec<Self> {
        bneck_confs(1.1, 1.2)
    }
    pub fn b3() -> Vec<Self> {
        bneck_confs(1.2, 1.4)
    }
    pub fn b4() -> Vec<Self> {
        bneck_confs(1.4, 1.8)
    }
    pub fn b5() -> Vec<Self> {
        bneck_confs(1.6, 2.2)
    }
    pub fn b6() -> Vec<Self> {
        bneck_confs(1.8, 2.6)
    }
    pub fn b7() -> Vec<Self> {
        bneck_confs(2.0, 3.1)
    }
}

/// Conv2D with same padding.
#[derive(Debug)]
struct Conv2DSame {
    conv2d: nn::Conv2d,
    s: usize,
    k: usize,
}

impl Conv2DSame {
    fn new(
        vb: VarBuilder,
        i: usize,
        o: usize,
        k: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv_config = nn::Conv2dConfig {
            stride,
            groups,
            ..Default::default()
        };
        let conv2d = if bias {
            nn::conv2d(i, o, k, conv_config, vb)?
        } else {
            nn::conv2d_no_bias(i, o, k, conv_config, vb)?
        };
        Ok(Self {
            conv2d,
            s: stride,
            k,
        })
    }
}

impl Module for Conv2DSame {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let s = self.s;
        let k = self.k;
        let (_, _, ih, iw) = xs.dims4()?;
        let oh = (ih + s - 1) / s;
        let ow = (iw + s - 1) / s;
        let pad_h = usize::max((oh - 1) * s + k - ih, 0);
        let pad_w = usize::max((ow - 1) * s + k - iw, 0);
        if pad_h > 0 || pad_w > 0 {
            let xs = xs.pad_with_zeros(2, pad_h / 2, pad_h - pad_h / 2)?;
            let xs = xs.pad_with_zeros(3, pad_w / 2, pad_w - pad_w / 2)?;
            self.conv2d.forward(&xs)
        } else {
            self.conv2d.forward(xs)
        }
    }
}

#[derive(Debug)]
struct ConvNormActivation {
    conv2d: Conv2DSame,
    bn2d: nn::BatchNorm,
    activation: bool,
}

impl ConvNormActivation {
    fn new(
        vb: VarBuilder,
        i: usize,
        o: usize,
        k: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let conv2d = Conv2DSame::new(vb.pp("0"), i, o, k, stride, groups, false)?;
        let bn2d = nn::batch_norm(o, 1e-3, vb.pp("1"))?;
        Ok(Self {
            conv2d,
            bn2d,
            activation: true,
        })
    }

    fn no_activation(self) -> Self {
        Self {
            activation: false,
            ..self
        }
    }
}

impl Module for ConvNormActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv2d.forward(xs)?.apply_t(&self.bn2d, false)?;
        if self.activation {
            swish(&xs)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug)]
struct SqueezeExcitation {
    fc1: Conv2DSame,
    fc2: Conv2DSame,
}

impl SqueezeExcitation {
    fn new(vb: VarBuilder, in_channels: usize, squeeze_channels: usize) -> Result<Self> {
        let fc1 = Conv2DSame::new(vb.pp("fc1"), in_channels, squeeze_channels, 1, 1, 1, true)?;
        let fc2 = Conv2DSame::new(vb.pp("fc2"), squeeze_channels, in_channels, 1, 1, 1, true)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for SqueezeExcitation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        // equivalent to adaptive_avg_pool2d([1, 1])
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = swish(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = nn::ops::sigmoid(&xs)?;
        residual.broadcast_mul(&xs)
    }
}

#[derive(Debug)]
struct MBConv {
    expand_cna: Option<ConvNormActivation>,
    depthwise_cna: ConvNormActivation,
    squeeze_excitation: SqueezeExcitation,
    project_cna: ConvNormActivation,
    config: MBConvConfig,
}

impl MBConv {
    fn new(vb: VarBuilder, c: MBConvConfig) -> Result<Self> {
        let vb = vb.pp("block");
        let exp = make_divisible(c.input_channels as f64 * c.expand_ratio, 8);
        let expand_cna = if exp != c.input_channels {
            Some(ConvNormActivation::new(
                vb.pp("0"),
                c.input_channels,
                exp,
                1,
                1,
                1,
            )?)
        } else {
            None
        };
        let start_index = if expand_cna.is_some() { 1 } else { 0 };
        let depthwise_cna =
            ConvNormActivation::new(vb.pp(start_index), exp, exp, c.kernel, c.stride, exp)?;
        let squeeze_channels = usize::max(1, c.input_channels / 4);
        let squeeze_excitation =
            SqueezeExcitation::new(vb.pp(start_index + 1), exp, squeeze_channels)?;
        let project_cna =
            ConvNormActivation::new(vb.pp(start_index + 2), exp, c.out_channels, 1, 1, 1)?
                .no_activation();
        Ok(Self {
            expand_cna,
            depthwise_cna,
            squeeze_excitation,
            project_cna,
            config: c,
        })
    }
}

impl Module for MBConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let use_res_connect =
            self.config.stride == 1 && self.config.input_channels == self.config.out_channels;
        let ys = match &self.expand_cna {
            Some(expand_cna) => expand_cna.forward(xs)?,
            None => xs.clone(),
        };
        let ys = self.depthwise_cna.forward(&ys)?;
        let ys = self.squeeze_excitation.forward(&ys)?;
        let ys = self.project_cna.forward(&ys)?;
        if use_res_connect {
            ys + xs
        } else {
            Ok(ys)
        }
    }
}

fn swish(s: &Tensor) -> Result<Tensor> {
    s * nn::ops::sigmoid(s)?
}

#[derive(Debug)]
pub struct EfficientNet {
    init_cna: ConvNormActivation,
    blocks: Vec<MBConv>,
    final_cna: ConvNormActivation,
    classifier: nn::Linear,
}

impl EfficientNet {
    pub fn new(p: VarBuilder, configs: Vec<MBConvConfig>, nclasses: usize) -> Result<Self> {
        let f_p = p.pp("features");
        let first_in_c = configs[0].input_channels;
        let last_out_c = configs.last().unwrap().out_channels;
        let final_out_c = 4 * last_out_c;
        let init_cna = ConvNormActivation::new(f_p.pp(0), 3, first_in_c, 3, 2, 1)?;
        let nconfigs = configs.len();
        let mut blocks = vec![];
        for (index, cnf) in configs.into_iter().enumerate() {
            let f_p = f_p.pp(index + 1);
            for r_index in 0..cnf.num_layers {
                let cnf = if r_index == 0 {
                    cnf
                } else {
                    MBConvConfig {
                        input_channels: cnf.out_channels,
                        stride: 1,
                        ..cnf
                    }
                };
                blocks.push(MBConv::new(f_p.pp(r_index), cnf)?)
            }
        }
        let final_cna =
            ConvNormActivation::new(f_p.pp(nconfigs + 1), last_out_c, final_out_c, 1, 1, 1)?;
        let classifier = nn::linear(final_out_c, nclasses, p.pp("classifier.1"))?;
        Ok(Self {
            init_cna,
            blocks,
            final_cna,
            classifier,
        })
    }
}

impl Module for EfficientNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.init_cna.forward(xs)?;
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?
        }
        let xs = self.final_cna.forward(&xs)?;
        // Equivalent to adaptive_avg_pool2d([1, 1]) -> squeeze(-1) -> squeeze(-1)
        let xs = xs.mean(D::Minus1)?.mean(D::Minus1)?;
        self.classifier.forward(&xs)
    }
}
