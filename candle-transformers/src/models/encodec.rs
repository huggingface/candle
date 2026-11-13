//! EnCodec neural audio codec based on the Encodec implementation.
//!
//! See ["High Fidelity Neural Audio Compression"](https://arxiv.org/abs/2210.13438)
//!
//! Based on implementation from [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py)

use candle::{DType, IndexOp, Layout, Module, Result, Shape, Tensor, D};
use candle_nn::{conv1d, Conv1d, ConvTranspose1d, VarBuilder};

// Encodec Model
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Deserialize)]
pub enum NormType {
    #[serde(rename = "weight_norm", alias = "WeightNorm")]
    WeightNorm,
    #[serde(rename = "time_group_norm", alias = "TimeGroupNorm")]
    TimeGroupNorm,
    #[serde(rename = "none", alias = "None")]
    None,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Deserialize)]
pub enum PadMode {
    #[serde(rename = "constant", alias = "Constant")]
    Constant,
    #[serde(rename = "reflect", alias = "Reflect")]
    Reflect,
    #[serde(rename = "replicate", alias = "Replicate")]
    Replicate,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    pub target_bandwidths: Vec<f64>,
    pub sampling_rate: usize,
    pub audio_channels: usize,
    pub normalize: bool,
    pub chunk_length_s: Option<f64>,
    pub overlap: Option<f64>,
    pub hidden_size: usize,
    pub num_filters: usize,
    pub num_residual_layers: usize,
    pub upsampling_ratios: Vec<usize>,
    pub norm_type: NormType,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub use_causal_conv: bool,
    pub pad_mode: PadMode,
    pub compress: usize,
    pub num_lstm_layers: usize,
    pub trim_right_ratio: f64,
    pub codebook_size: usize,
    pub codebook_dim: Option<usize>,
    pub use_conv_shortcut: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            target_bandwidths: vec![1.5, 3.0, 6.0, 12.0, 24.0],
            sampling_rate: 24_000,
            audio_channels: 1,
            normalize: false,
            chunk_length_s: None,
            overlap: None,
            hidden_size: 128,
            num_filters: 32,
            num_residual_layers: 1,
            upsampling_ratios: vec![8, 5, 4, 2],
            norm_type: NormType::WeightNorm,
            kernel_size: 7,
            last_kernel_size: 7,
            residual_kernel_size: 3,
            dilation_growth_rate: 2,
            use_causal_conv: true,
            pad_mode: PadMode::Reflect,
            compress: 2,
            num_lstm_layers: 2,
            trim_right_ratio: 1.0,
            codebook_size: 1024,
            codebook_dim: None,
            use_conv_shortcut: true,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.target_bandwidths.is_empty()
            || self
                .target_bandwidths
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.)
            || self.sampling_rate == 0
            || self.audio_channels == 0
            || self.hidden_size == 0
            || self.num_filters == 0
            || self.upsampling_ratios.is_empty()
            || self.upsampling_ratios.contains(&0)
            || self.compress == 0
            || self.codebook_size < 2
            || self.codebook_dim == Some(0)
            || self.kernel_size == 0
            || self.last_kernel_size == 0
            || self.residual_kernel_size == 0
        {
            candle::bail!("invalid EnCodec configuration");
        }
        if self.upsampling_ratios.len() >= usize::BITS as usize
            || self
                .num_filters
                .checked_shl(self.upsampling_ratios.len() as u32)
                .is_none()
        {
            candle::bail!("EnCodec scale overflow");
        }
        let hop = self
            .upsampling_ratios
            .iter()
            .try_fold(1usize, |n, value| n.checked_mul(*value))
            .ok_or_else(|| candle::Error::Msg("EnCodec hop overflow".into()))?;
        if hop > self.sampling_rate || !self.codebook_size.is_power_of_two() {
            candle::bail!("EnCodec requires a supported hop and power-of-two codebooks");
        }
        if self
            .chunk_length_s
            .is_some_and(|v| !v.is_finite() || v <= 0.)
            || self
                .overlap
                .is_some_and(|v| !v.is_finite() || !(0. ..1.).contains(&v))
        {
            candle::bail!("invalid EnCodec chunking configuration");
        }
        if self.num_quantizers() == 0 {
            candle::bail!("EnCodec bandwidth is too small for one quantizer");
        }
        Ok(())
    }
    fn codebook_dim(&self) -> usize {
        self.codebook_dim.unwrap_or(self.hidden_size)
    }

    pub fn frame_rate(&self) -> usize {
        let hop_length: usize = self.upsampling_ratios.iter().product();
        self.sampling_rate.div_ceil(hop_length)
    }

    pub fn num_quantizers(&self) -> usize {
        let num = 1000f64
            * self
                .target_bandwidths
                .last()
                .expect("empty target_bandwidths");
        (num as usize) / (self.frame_rate() * self.codebook_size.ilog2() as usize)
    }
}

fn get_extra_padding_for_conv1d(
    xs: &Tensor,
    k_size: usize,
    stride: usize,
    padding_total: usize,
) -> Result<usize> {
    let len = xs.dim(D::Minus1)?;
    let n_frames = (len + padding_total).saturating_sub(k_size) as f64 / stride as f64 + 1.0;
    let ideal_len =
        ((n_frames.ceil() as usize - 1) * stride + k_size).saturating_sub(padding_total);
    Ok(ideal_len.saturating_sub(len))
}

fn pad1d(xs: &Tensor, pad_l: usize, pad_r: usize, mode: PadMode) -> Result<Tensor> {
    match mode {
        PadMode::Constant => xs.pad_with_zeros(D::Minus1, pad_l, pad_r),
        PadMode::Reflect => {
            // Match Transformers' small-input wrapper: zero-extend before
            // reflection, then remove that temporary extension from the right.
            let length = xs.dim(D::Minus1)?;
            if length == 0 {
                candle::bail!("cannot reflect-pad empty audio");
            }
            let max_pad = pad_l.max(pad_r);
            let extra = if length <= max_pad {
                max_pad - length + 1
            } else {
                0
            };
            let xs = xs.pad_with_zeros(D::Minus1, 0, extra)?;
            let length = length + extra;
            let indices: Vec<u32> = (0..length + pad_l + pad_r)
                .map(|index| {
                    let index = index as isize - pad_l as isize;
                    if index < 0 {
                        (-index) as u32
                    } else if index >= length as isize {
                        (2 * length as isize - 2 - index) as u32
                    } else {
                        index as u32
                    }
                })
                .collect();
            let indices = Tensor::new(indices.as_slice(), xs.device())?;
            let padded = xs.contiguous()?.index_select(&indices, D::Minus1)?;
            padded.narrow(D::Minus1, 0, padded.dim(D::Minus1)? - extra)
        }
        PadMode::Replicate => xs.pad_with_same(D::Minus1, pad_l, pad_r),
    }
}

// Applies weight norm for inference by recomputing the weight tensor. This
// does not apply to training.
// https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    if vb.contains_tensor("weight") {
        return candle_nn::conv1d(in_c, out_c, kernel_size, config, vb);
    }
    let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

pub fn conv1d_weight_norm_no_bias(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    if vb.contains_tensor("weight") {
        return candle_nn::conv1d_no_bias(in_c, out_c, kernel_size, config, vb);
    }
    let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    Ok(Conv1d::new(weight, None, config))
}

pub fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    if vb.contains_tensor("weight") {
        let weight = vb.get((in_c, out_c, kernel_size), "weight")?;
        let bias = if bias {
            Some(vb.get(out_c, "bias")?)
        } else {
            None
        };
        return Ok(ConvTranspose1d::new(weight, bias, config));
    }
    let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((in_c, out_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, config))
}

struct CodebookEncode;

impl candle::CustomOp2 for CodebookEncode {
    fn name(&self) -> &'static str {
        "cb"
    }

    fn cpu_fwd(
        &self,
        lhs_storage: &candle::CpuStorage,
        lhs_layout: &Layout,
        rhs_storage: &candle::CpuStorage,
        rhs_layout: &Layout,
    ) -> Result<(candle::CpuStorage, Shape)> {
        use rayon::prelude::*;

        let (lhs_dim1, lhs_dim2) = lhs_layout.shape().dims2()?;
        let (rhs_dim1, rhs_dim2) = rhs_layout.shape().dims2()?;
        if lhs_dim2 != rhs_dim2 {
            candle::bail!("CodebookEncode, mismatch on last dim, {lhs_layout:?} {rhs_layout:?}");
        }
        if lhs_dim2 == 0 {
            candle::bail!("CodebookEncode, empty last dim {lhs_layout:?}")
        }
        let lhs = match lhs_layout.contiguous_offsets() {
            None => candle::bail!("CodebookEncode, lhs has to be contiguous, got {lhs_layout:?}"),
            Some((o1, o2)) => {
                let slice = lhs_storage.as_slice::<f32>()?;
                &slice[o1..o2]
            }
        };
        let rhs = match rhs_layout.contiguous_offsets() {
            None => candle::bail!("CodebookEncode, rhs has to be contiguous, got {rhs_layout:?}"),
            Some((o1, o2)) => {
                let slice = rhs_storage.as_slice::<f32>()?;
                &slice[o1..o2]
            }
        };
        let dst = (0..lhs_dim1)
            .into_par_iter()
            .map(|idx1| {
                let mut where_min = 0;
                let mut min_dist = f32::INFINITY;
                let lhs = &lhs[idx1 * lhs_dim2..(idx1 + 1) * lhs_dim2];
                for idx2 in 0..rhs_dim1 {
                    let rhs = &rhs[idx2 * rhs_dim2..(idx2 + 1) * rhs_dim2];
                    let mut dist = 0f32;
                    for (a, b) in lhs.iter().zip(rhs.iter()) {
                        dist += (a - b) * (a - b)
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        where_min = idx2;
                    }
                }
                where_min as u32
            })
            .collect();
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, (lhs_dim1,).into()))
    }
}

// https://github.com/huggingface/transformers/blob/abaca9f9432a84cfaa95531de4c72334f38a42f2/src/transformers/models/encodec/modeling_encodec.py#L340
#[allow(unused)]
#[derive(Clone, Debug)]
pub struct EuclideanCodebook {
    inited: Tensor,
    cluster_size: Tensor,
    embed: candle_nn::Embedding,
    embed_avg: Tensor,
    c2: Tensor,
}

impl EuclideanCodebook {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let inited = vb.get(1, "inited")?;
        let cluster_size = vb.get(cfg.codebook_size, "cluster_size")?;
        let e_shape = (cfg.codebook_size, cfg.codebook_dim());
        let embed = vb.get(e_shape, "embed")?;
        let c2 = ((&embed * &embed)?.sum(D::Minus1)? / 2.0)?;
        let embed_avg = vb.get(e_shape, "embed_avg")?;
        Ok(Self {
            inited,
            cluster_size,
            embed: candle_nn::Embedding::new(embed, cfg.codebook_dim()),
            embed_avg,
            c2,
        })
    }

    pub fn encode_slow(&self, xs: &Tensor) -> Result<Tensor> {
        let mut target_shape = xs.dims().to_vec();
        target_shape.pop();
        let xs = xs.flatten_to(D::Minus2)?;
        let _ = xs.dims2()?;
        let dot_prod = xs.matmul(&self.embed.embeddings().t()?)?;
        let codes = self.c2.broadcast_sub(&dot_prod)?.argmin(D::Minus1)?;
        codes.reshape(target_shape)
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut target_shape = xs.dims().to_vec();
        target_shape.pop();
        let xs = xs.flatten_to(D::Minus2)?;
        let _ = xs.dims2()?;
        let codes = Tensor::apply_op2(&xs, self.embed.embeddings(), CodebookEncode)?;
        codes.reshape(target_shape)
    }

    pub fn decode(&self, embed_ind: &Tensor) -> Result<Tensor> {
        let quantize = self.embed.forward(embed_ind)?;
        Ok(quantize)
    }
}

#[derive(Clone, Debug)]
pub struct VectorQuantization {
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let codebook = EuclideanCodebook::new(cfg, vb.pp("codebook"))?;
        Ok(Self { codebook })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.transpose(1, 2)?;
        self.codebook.encode_slow(&xs)
    }

    pub fn decode(&self, embed_ind: &Tensor) -> Result<Tensor> {
        let quantize = self.codebook.decode(embed_ind)?;
        let quantize = quantize.transpose(1, 2)?;
        Ok(quantize)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    layers: Vec<VectorQuantization>,
    dtype: DType,
}

impl ResidualVectorQuantizer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = &vb.pp("layers");
        let layers = (0..cfg.num_quantizers())
            .map(|i| VectorQuantization::new(cfg, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            dtype: vb.dtype(),
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut codes = Vec::with_capacity(self.layers.len());
        let mut residual = xs.clone();
        for layer in self.layers.iter() {
            let indices = layer.encode(&residual)?;
            let quantized = layer.decode(&indices)?;
            residual = (residual - quantized)?;
            codes.push(indices)
        }
        Tensor::stack(&codes, 0)
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut quantized_out = Tensor::zeros((), self.dtype, codes.device())?;
        let ncodes = codes.dim(0)?;
        if ncodes > self.layers.len() {
            candle::bail!(
                "codes shape {:?} does not match the number of quantization layers {}",
                codes.shape(),
                self.layers.len()
            )
        }
        for (i, layer) in self.layers.iter().take(ncodes).enumerate() {
            let quantized = layer.decode(&codes.i(i)?)?;
            quantized_out = quantized.broadcast_add(&quantized_out)?;
        }
        Ok(quantized_out)
    }
}

// https://github.com/huggingface/transformers/blob/abaca9f9432a84cfaa95531de4c72334f38a42f2/src/transformers/models/encodec/modeling_encodec.py#L226
#[derive(Clone, Debug)]
pub struct EncodecLSTM {
    layers: Vec<candle_nn::LSTM>,
}

impl EncodecLSTM {
    pub fn new(dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = &vb.pp("lstm");
        let mut layers = vec![];
        for layer_idx in 0..cfg.num_lstm_layers {
            let config = candle_nn::LSTMConfig {
                layer_idx,
                ..Default::default()
            };
            let lstm = candle_nn::lstm(dim, dim, config, vb.clone())?;
            layers.push(lstm)
        }
        Ok(Self { layers })
    }
}

impl Module for EncodecLSTM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        use candle_nn::RNN;
        // This is different from the Python transformers version as candle LSTM is batch first.
        let xs = xs.t()?;
        let residual = &xs;
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            let states = layer.seq(&xs)?;
            xs = layer.states_to_tensor(&states)?;
        }
        let xs = (xs + residual)?.t()?;
        Ok(xs)
    }
}

#[derive(Clone, Debug)]
pub struct EncodecConvTranspose1d {
    conv: ConvTranspose1d,
    norm: Option<candle_nn::GroupNorm>,
    padding_left: usize,
    padding_right: usize,
}
impl EncodecConvTranspose1d {
    fn new(
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = candle_nn::ConvTranspose1dConfig {
            stride,
            ..Default::default()
        };
        let conv = match cfg.norm_type {
            NormType::WeightNorm => {
                conv_transpose1d_weight_norm(in_c, out_c, k, true, conv_cfg, vb.pp("conv"))?
            }
            NormType::None | NormType::TimeGroupNorm => {
                candle_nn::conv_transpose1d(in_c, out_c, k, conv_cfg, vb.pp("conv"))?
            }
        };
        let norm = if cfg.norm_type == NormType::TimeGroupNorm {
            Some(candle_nn::group_norm(1, out_c, 1e-5, vb.pp("norm"))?)
        } else {
            None
        };
        let total = k
            .checked_sub(stride)
            .ok_or_else(|| candle::Error::Msg("codec kernel is smaller than stride".into()))?;
        if !(0. ..=1.).contains(&cfg.trim_right_ratio)
            || (!cfg.use_causal_conv && cfg.trim_right_ratio != 1.)
        {
            candle::bail!("invalid codec trim_right_ratio");
        }
        let padding_right = if cfg.use_causal_conv {
            (total as f64 * cfg.trim_right_ratio).ceil() as usize
        } else {
            total / 2
        };
        Ok(Self {
            conv,
            norm,
            padding_left: total - padding_right,
            padding_right,
        })
    }
}
impl Module for EncodecConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv)?;
        if let Some(norm) = &self.norm {
            xs = xs.apply(norm)?;
        }
        let size = xs
            .dim(D::Minus1)?
            .checked_sub(self.padding_left + self.padding_right)
            .ok_or_else(|| {
                candle::Error::Msg("codec transposed-convolution output is too short".into())
            })?;
        xs.narrow(D::Minus1, self.padding_left, size)
    }
}

#[derive(Clone, Debug)]
pub struct EncodecConv1d {
    causal: bool,
    conv: Conv1d,
    norm: Option<candle_nn::GroupNorm>,
    pad_mode: PadMode,
}

impl EncodecConv1d {
    pub fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = match cfg.norm_type {
            NormType::WeightNorm => conv1d_weight_norm(
                in_c,
                out_c,
                kernel_size,
                candle_nn::Conv1dConfig {
                    stride,
                    dilation,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
            NormType::None | NormType::TimeGroupNorm => conv1d(
                in_c,
                out_c,
                kernel_size,
                candle_nn::Conv1dConfig {
                    padding: 0,
                    stride,
                    groups: 1,
                    dilation,
                    cudnn_fwd_algo: None,
                },
                vb.pp("conv"),
            )?,
        };
        let norm = match cfg.norm_type {
            NormType::None | NormType::WeightNorm => None,
            NormType::TimeGroupNorm => {
                let gn = candle_nn::group_norm(1, out_c, 1e-5, vb.pp("norm"))?;
                Some(gn)
            }
        };
        Ok(Self {
            causal: cfg.use_causal_conv,
            conv,
            norm,
            pad_mode: cfg.pad_mode,
        })
    }
}

impl Module for EncodecConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _t, _c) = xs.dims3()?;
        let k_size = self.conv.weight().dim(D::Minus1)?;
        let conv_cfg = self.conv.config();
        // Effective kernel size with dilations.
        let k_size = (k_size - 1) * conv_cfg.dilation + 1;
        let padding_total = k_size - conv_cfg.stride;
        let extra_padding =
            get_extra_padding_for_conv1d(xs, k_size, conv_cfg.stride, padding_total)?;
        let xs = if self.causal {
            pad1d(xs, padding_total, extra_padding, self.pad_mode)?
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            pad1d(
                xs,
                padding_left,
                padding_right + extra_padding,
                self.pad_mode,
            )?
        };
        let xs = self.conv.forward(&xs)?;
        match &self.norm {
            None => Ok(xs),
            Some(norm) => xs.apply(norm),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodecResnetBlock {
    block_conv1: EncodecConv1d,
    block_conv2: EncodecConv1d,
    shortcut: Option<EncodecConv1d>,
}

impl EncodecResnetBlock {
    pub fn new(
        dim: usize,
        (dilation1, dilation2): (usize, usize),
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let h = dim / cfg.compress;
        let mut layer = Layer::new(vb.pp("block"));
        // TODO: Apply dilations!
        layer.inc();
        let block_conv1 = EncodecConv1d::new(
            dim,
            h,
            cfg.residual_kernel_size,
            1,
            dilation1,
            cfg,
            layer.next(),
        )?;
        layer.inc();
        let block_conv2 = EncodecConv1d::new(h, dim, 1, 1, dilation2, cfg, layer.next())?;
        let shortcut = if cfg.use_conv_shortcut {
            let conv = EncodecConv1d::new(dim, dim, 1, 1, 1, cfg, vb.pp("shortcut"))?;
            Some(conv)
        } else {
            None
        };
        Ok(Self {
            block_conv1,
            block_conv2,
            shortcut,
        })
    }
}

impl Module for EncodecResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = xs.elu(1.)?;
        let xs = self.block_conv1.forward(&xs)?;
        let xs = xs.elu(1.)?;
        let xs = self.block_conv2.forward(&xs)?;
        let xs = match &self.shortcut {
            None => (xs + residual)?,
            Some(shortcut) => xs.add(&shortcut.forward(&residual)?)?,
        };
        Ok(xs)
    }
}

struct Layer<'a> {
    vb: VarBuilder<'a>,
    cnt: usize,
}

impl<'a> Layer<'a> {
    fn new(vb: VarBuilder<'a>) -> Self {
        Self { vb, cnt: 0 }
    }

    fn inc(&mut self) {
        self.cnt += 1;
    }

    fn next(&mut self) -> VarBuilder<'_> {
        let vb = self.vb.pp(self.cnt.to_string());
        self.cnt += 1;
        vb
    }
}

#[derive(Clone, Debug)]
pub struct Encoder {
    init_conv: EncodecConv1d,
    sampling_layers: Vec<(Vec<EncodecResnetBlock>, EncodecConv1d)>,
    final_lstm: EncodecLSTM,
    final_conv: EncodecConv1d,
}

impl Encoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layer = Layer::new(vb.pp("layers"));
        let init_conv = EncodecConv1d::new(
            cfg.audio_channels,
            cfg.num_filters,
            cfg.kernel_size,
            1,
            1,
            cfg,
            layer.next(),
        )?;
        let mut sampling_layers = vec![];
        let mut scaling = 1;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let current_scale = scaling * cfg.num_filters;
            let mut resnets = vec![];
            for j in 0..(cfg.num_residual_layers as u32) {
                let resnet = EncodecResnetBlock::new(
                    current_scale,
                    (cfg.dilation_growth_rate.pow(j), 1),
                    cfg,
                    layer.next(),
                )?;
                resnets.push(resnet)
            }
            layer.inc(); // ELU
            let conv1d = EncodecConv1d::new(
                current_scale,
                current_scale * 2,
                ratio * 2,
                ratio,
                1,
                cfg,
                layer.next(),
            )?;
            sampling_layers.push((resnets, conv1d));
            scaling *= 2;
        }
        let final_lstm = EncodecLSTM::new(cfg.num_filters * scaling, cfg, layer.next())?;
        layer.inc(); // ELU
        let final_conv = EncodecConv1d::new(
            cfg.num_filters * scaling,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
            1,
            cfg,
            layer.next(),
        )?;
        Ok(Self {
            init_conv,
            sampling_layers,
            final_conv,
            final_lstm,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.init_conv)?;
        for (resnets, conv) in self.sampling_layers.iter() {
            for resnet in resnets.iter() {
                xs = xs.apply(resnet)?;
            }
            xs = xs.elu(1.0)?.apply(conv)?;
        }
        xs.apply(&self.final_lstm)?
            .elu(1.0)?
            .apply(&self.final_conv)
    }
}

#[derive(Clone, Debug)]
pub struct Decoder {
    init_conv: EncodecConv1d,
    init_lstm: EncodecLSTM,
    sampling_layers: Vec<(EncodecConvTranspose1d, Vec<EncodecResnetBlock>)>,
    final_conv: EncodecConv1d,
}

impl Decoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layer = Layer::new(vb.pp("layers"));
        let mut scaling = usize::pow(2, cfg.upsampling_ratios.len() as u32);
        let init_conv = EncodecConv1d::new(
            cfg.hidden_size,
            cfg.num_filters * scaling,
            cfg.last_kernel_size,
            1,
            1,
            cfg,
            layer.next(),
        )?;
        let init_lstm = EncodecLSTM::new(cfg.num_filters * scaling, cfg, layer.next())?;
        let mut sampling_layers = vec![];
        for &ratio in cfg.upsampling_ratios.iter() {
            let current_scale = scaling * cfg.num_filters;
            layer.inc(); // ELU
            let conv1d = EncodecConvTranspose1d::new(
                current_scale,
                current_scale / 2,
                ratio * 2,
                ratio,
                cfg,
                layer.next(),
            )?;
            let mut resnets = vec![];
            for j in 0..(cfg.num_residual_layers as u32) {
                let resnet = EncodecResnetBlock::new(
                    current_scale / 2,
                    (cfg.dilation_growth_rate.pow(j), 1),
                    cfg,
                    layer.next(),
                )?;
                resnets.push(resnet)
            }
            sampling_layers.push((conv1d, resnets));
            scaling /= 2;
        }
        layer.inc(); // ELU
        let final_conv = EncodecConv1d::new(
            cfg.num_filters,
            cfg.audio_channels,
            cfg.last_kernel_size,
            1,
            1,
            cfg,
            layer.next(),
        )?;
        Ok(Self {
            init_conv,
            init_lstm,
            sampling_layers,
            final_conv,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.init_conv)?.apply(&self.init_lstm)?;
        for (conv, resnets) in self.sampling_layers.iter() {
            xs = xs.elu(1.)?.apply(conv)?;
            for resnet in resnets.iter() {
                xs = xs.apply(resnet)?
            }
        }
        xs.elu(1.)?.apply(&self.final_conv)
    }
}

#[derive(Debug)]
pub struct Model {
    encoder: Encoder,
    decoder: Decoder,
    quantizer: ResidualVectorQuantizer,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        let quantizer = ResidualVectorQuantizer::new(cfg, vb.pp("quantizer"))?;
        Ok(Self {
            encoder,
            decoder,
            quantizer,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.encoder.forward(xs)?;
        let codes = self.quantizer.encode(&xs)?;
        codes.transpose(0, 1)
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (_b_sz, _codebooks, _seqlen) = codes.dims3()?;
        let codes = codes.transpose(0, 1)?;
        let embeddings = self.quantizer.decode(&codes)?;
        let outputs = self.decoder.forward(&embeddings)?;
        Ok(outputs)
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/encodec_padding.json")).unwrap()
    }
    fn values(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect()
    }
    #[test]
    fn reflect_padding_matches_torch_including_short_input_extension() -> Result<()> {
        for case in fixture()["reflect"].as_array().unwrap() {
            let source = values(&case["input"]);
            let left = case["left"].as_u64().unwrap() as usize;
            let right = case["right"].as_u64().unwrap() as usize;
            let expected = values(&case["expected"]);
            let length = source.len();
            let source = Tensor::from_vec(source, (1, 1, length), &candle::Device::Cpu)?;
            let actual = pad1d(&source, left, right, PadMode::Reflect)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }
    #[test]
    fn transpose_convolution_removes_reference_padding_on_correct_side() -> Result<()> {
        let device = candle::Device::Cpu;
        for case in fixture()["transpose"].as_array().unwrap() {
            let causal = case["causal"].as_bool().unwrap();
            let right_ratio = case["ratio"].as_f64().unwrap();
            let expected = values(&case["expected"]);
            let mut weights = std::collections::HashMap::new();
            weights.insert(
                "conv.weight".into(),
                Tensor::ones((1, 1, 4), DType::F32, &device)?,
            );
            weights.insert("conv.bias".into(), Tensor::zeros(1, DType::F32, &device)?);
            let config = Config {
                use_causal_conv: causal,
                trim_right_ratio: right_ratio,
                norm_type: NormType::None,
                ..Default::default()
            };
            let conv = EncodecConvTranspose1d::new(
                1,
                1,
                4,
                2,
                &config,
                VarBuilder::from_tensors(weights, DType::F32, &device),
            )?;
            let actual = conv
                .forward(&Tensor::new(&[[[1f32, 2.]]], &device)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }
}
