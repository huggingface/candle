use candle::{DType, ModuleT, Result, Tensor, D};
use candle_nn::{
    BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder,
};

use crate::models::parakeet::attention::{
    MultiHeadAttention, RelPositionMultiHeadAttention, RelPositionalEncoding,
};
use crate::models::parakeet::cache::{CacheLike, ConformerCache};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConformerArgs {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub subsampling_factor: usize,
    pub self_attention_model: String,
    pub subsampling: String,
    pub conv_kernel_size: usize,
    pub subsampling_conv_channels: usize,
    pub pos_emb_max_len: usize,
    #[serde(default)]
    pub causal_downsampling: bool,
    #[serde(default = "default_true")]
    pub use_bias: bool,
    #[serde(default)]
    pub xscaling: bool,
    #[serde(default)]
    pub pos_bias_u: Option<Vec<f32>>,
    #[serde(default)]
    pub pos_bias_v: Option<Vec<f32>>,
    #[serde(default = "default_subsampling_conv_chunking_factor")]
    pub subsampling_conv_chunking_factor: isize,
    #[serde(default)]
    pub att_context_size: Option<Vec<i64>>,
}

fn default_true() -> bool {
    true
}

fn default_subsampling_conv_chunking_factor() -> isize {
    1
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(
        size,
        candle_nn::LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        },
        vb,
    )
}

#[derive(Debug, Clone)]
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.silu()?;
        self.linear2.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct Convolution {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: BatchNorm,
    pointwise_conv2: Conv1d,
    padding: usize,
}

impl Convolution {
    fn load(args: &ConformerArgs, vb: VarBuilder) -> Result<Self> {
        let padding = (args.conv_kernel_size - 1) / 2;
        let cfg_pw = Conv1dConfig {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg_dw = Conv1dConfig {
            padding: 0,
            stride: 1,
            groups: args.d_model,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let pw1_w = vb.get(
            (args.d_model * 2, args.d_model, 1),
            "pointwise_conv1.weight",
        )?;
        let pw1_b = vb.get((args.d_model * 2,), "pointwise_conv1.bias")?;
        let pointwise_conv1 = Conv1d::new(pw1_w, Some(pw1_b), cfg_pw);

        let dw_w = vb.get(
            (args.d_model, 1, args.conv_kernel_size),
            "depthwise_conv.weight",
        )?;
        let dw_b = vb.get((args.d_model,), "depthwise_conv.bias")?;
        let depthwise_conv = Conv1d::new(dw_w, Some(dw_b), cfg_dw);

        let batch_norm = candle_nn::batch_norm(args.d_model, 1e-5, vb.pp("batch_norm"))?;

        let pw2_w = vb.get((args.d_model, args.d_model, 1), "pointwise_conv2.weight")?;
        let pw2_b = vb.get((args.d_model,), "pointwise_conv2.bias")?;
        let pointwise_conv2 = Conv1d::new(pw2_w, Some(pw2_b), cfg_pw);

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            batch_norm,
            pointwise_conv2,
            padding,
        })
    }

    fn forward(&self, x: &Tensor, cache: Option<&mut dyn CacheLike>) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?; // (B, C, T)
        x = self.pointwise_conv1.forward(&x)?; // (B, 2C, T)
        x = x.transpose(1, 2)?; // (B, T, 2C)
        let (_, _, c2) = x.dims3()?;
        let c = c2 / 2;
        let a = x.narrow(2, 0, c)?;
        let b_part = x.narrow(2, c, c)?;
        let gate = candle_nn::ops::sigmoid(&b_part)?;
        let mut x = (&a * &gate)?;

        if let Some(cache) = cache {
            x = cache.update_and_fetch_conv(&x, self.padding)?;
        } else if self.padding > 0 {
            x = x.pad_with_zeros(D::Minus2, self.padding, self.padding)?;
        }

        x = x.transpose(1, 2)?; // (B, C, T)
        x = self.depthwise_conv.forward(&x)?;
        x = self.batch_norm.forward_t(&x, false)?;
        x = x.silu()?;
        x = self.pointwise_conv2.forward(&x)?;
        x.transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
enum SelfAttention {
    RelPos {
        attn: RelPositionMultiHeadAttention,
        local_context: Option<(usize, usize)>,
    },
    Normal(MultiHeadAttention),
}

#[derive(Debug, Clone)]
struct ConformerBlock {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: SelfAttention,
    norm_conv: LayerNorm,
    conv: Convolution,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerBlock {
    fn load(args: &ConformerArgs, vb: VarBuilder) -> Result<Self> {
        let ff_hidden_dim = args.d_model * args.ff_expansion_factor;
        let norm_ff1 = layer_norm(args.d_model, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(args.d_model, ff_hidden_dim, vb.pp("feed_forward1"))?;

        let norm_self_att = layer_norm(args.d_model, vb.pp("norm_self_att"))?;
        let self_attn = match args.self_attention_model.as_str() {
            "rel_pos" | "rel_pos_local_attn" => SelfAttention::RelPos {
                attn: RelPositionMultiHeadAttention::load(
                    args.n_heads,
                    args.d_model,
                    args.use_bias,
                    vb.pp("self_attn"),
                )?,
                local_context: if args.self_attention_model == "rel_pos_local_attn" {
                    args.att_context_size.as_ref().and_then(|v| {
                        if v.len() == 2 && v[0] >= 0 && v[1] >= 0 {
                            Some((v[0] as usize, v[1] as usize))
                        } else {
                            None
                        }
                    })
                } else {
                    None
                },
            },
            _ => SelfAttention::Normal(MultiHeadAttention::load(
                args.n_heads,
                args.d_model,
                true,
                vb.pp("self_attn"),
            )?),
        };

        let norm_conv = layer_norm(args.d_model, vb.pp("norm_conv"))?;
        let conv = Convolution::load(args, vb.pp("conv"))?;

        let norm_ff2 = layer_norm(args.d_model, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(args.d_model, ff_hidden_dim, vb.pp("feed_forward2"))?;

        let norm_out = layer_norm(args.d_model, vb.pp("norm_out"))?;

        Ok(Self {
            norm_ff1,
            ff1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_ff2,
            ff2,
            norm_out,
        })
    }

    fn set_attention_model(&mut self, name: &str, context_size: Option<(usize, usize)>) {
        if let SelfAttention::RelPos { local_context, .. } = &mut self.self_attn {
            if name == "rel_pos_local_attn" {
                *local_context = context_size;
            } else {
                *local_context = None;
            }
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        pos_emb: Option<&Tensor>,
        mut cache: Option<&mut dyn CacheLike>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&x)?)?;
        x = (&x + &(&ff1 * 0.5)?)?;

        let x_norm = self.norm_self_att.forward(&x)?;
        let attn_out = if let Some(cache) = cache.as_deref_mut() {
            match &self.self_attn {
                SelfAttention::RelPos {
                    attn,
                    local_context,
                } => {
                    let pos_emb = pos_emb
                        .ok_or_else(|| candle::Error::Msg("pos_emb required".to_string()))?;
                    let mask = if let Some((left, right)) = local_context {
                        let q_len = x_norm.dims3()?.1;
                        let k_len = cache.offset() + q_len;
                        let offset = k_len.saturating_sub(q_len);
                        let mut data = vec![0f32; q_len * k_len];
                        for i in 0..q_len {
                            let center = i + offset;
                            let start = center.saturating_sub(*left);
                            let end = (center + *right).min(k_len.saturating_sub(1));
                            for j in 0..k_len {
                                if j < start || j > end {
                                    data[i * k_len + j] = f32::NEG_INFINITY;
                                }
                            }
                        }
                        Some(Tensor::from_vec(data, (1, 1, q_len, k_len), x.device())?)
                    } else {
                        None
                    };
                    attn.forward(
                        &x_norm,
                        &x_norm,
                        &x_norm,
                        pos_emb,
                        mask.as_ref(),
                        Some(cache),
                    )?
                }
                SelfAttention::Normal(attn) => {
                    attn.forward(&x_norm, &x_norm, &x_norm, None, Some(cache))?
                }
            }
        } else {
            match &self.self_attn {
                SelfAttention::RelPos {
                    attn,
                    local_context,
                } => {
                    let pos_emb = pos_emb
                        .ok_or_else(|| candle::Error::Msg("pos_emb required".to_string()))?;
                    let mask = if let Some((left, right)) = local_context {
                        let q_len = x_norm.dims3()?.1;
                        let k_len = q_len;
                        let offset = 0usize;
                        let mut data = vec![0f32; q_len * k_len];
                        for i in 0..q_len {
                            let center = i + offset;
                            let start = center.saturating_sub(*left);
                            let end = (center + *right).min(k_len.saturating_sub(1));
                            for j in 0..k_len {
                                if j < start || j > end {
                                    data[i * k_len + j] = f32::NEG_INFINITY;
                                }
                            }
                        }
                        Some(Tensor::from_vec(data, (1, 1, q_len, k_len), x.device())?)
                    } else {
                        None
                    };
                    attn.forward(&x_norm, &x_norm, &x_norm, pos_emb, mask.as_ref(), None)?
                }
                SelfAttention::Normal(attn) => {
                    attn.forward(&x_norm, &x_norm, &x_norm, None, None)?
                }
            }
        };

        x = (&x + &attn_out)?;
        let conv_out = if let Some(cache) = cache.as_deref_mut() {
            self.conv
                .forward(&self.norm_conv.forward(&x)?, Some(cache))?
        } else {
            self.conv.forward(&self.norm_conv.forward(&x)?, None)?
        };
        x = (&x + &conv_out)?;
        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&x)?)?;
        x = (&x + &(&ff2 * 0.5)?)?;
        self.norm_out.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct DwStridingSubsampling {
    conv: Vec<Conv2d>,
    out: Linear,
    stride: usize,
    kernel_size: usize,
    padding: usize,
    sampling_num: usize,
    subsampling_conv_chunking_factor: isize,
}

impl DwStridingSubsampling {
    fn load(args: &ConformerArgs, vb: VarBuilder) -> Result<Self> {
        let sampling_num = (args.subsampling_factor as f64).log2() as usize;
        let stride = 2;
        let kernel_size = 3;
        let padding = (kernel_size - 1) / 2;

        let mut conv = Vec::new();
        let mut in_channels = 1;
        let mut final_freq_dim = args.feat_in;
        for _ in 0..sampling_num {
            final_freq_dim = ((final_freq_dim + 2 * padding - kernel_size) / stride) + 1;
        }
        let cfg = Conv2dConfig {
            padding,
            stride,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let first_w = vb.get(
            (
                args.subsampling_conv_channels,
                in_channels,
                kernel_size,
                kernel_size,
            ),
            "conv.0.weight",
        )?;
        let first_b = vb.get((args.subsampling_conv_channels,), "conv.0.bias")?;
        conv.push(Conv2d::new(first_w, Some(first_b), cfg));
        in_channels = args.subsampling_conv_channels;

        for i in 0..(sampling_num - 1) {
            let dw_name = format!("conv.{}", 2 + i * 3);
            let dw_w = vb.get((in_channels, 1, kernel_size, kernel_size), dw_name.as_str())?;
            let dw_b = vb.get((in_channels,), dw_name.as_str())?;
            let dw_cfg = Conv2dConfig {
                padding,
                stride,
                groups: in_channels,
                dilation: 1,
                cudnn_fwd_algo: None,
            };
            conv.push(Conv2d::new(dw_w, Some(dw_b), dw_cfg));

            let pw_name = format!("conv.{}", 2 + i * 3 + 1);
            let pw_w = vb.get(
                (args.subsampling_conv_channels, in_channels, 1, 1),
                pw_name.as_str(),
            )?;
            let pw_b = vb.get((args.subsampling_conv_channels,), pw_name.as_str())?;
            let pw_cfg = Conv2dConfig {
                padding: 0,
                stride: 1,
                groups: 1,
                dilation: 1,
                cudnn_fwd_algo: None,
            };
            conv.push(Conv2d::new(pw_w, Some(pw_b), pw_cfg));
        }

        let out = candle_nn::linear(
            args.subsampling_conv_channels * final_freq_dim,
            args.d_model,
            vb.pp("out"),
        )?;

        Ok(Self {
            conv,
            out,
            stride,
            kernel_size,
            padding,
            sampling_num,
            subsampling_conv_chunking_factor: args.subsampling_conv_chunking_factor,
        })
    }

    fn forward(&self, x: &Tensor, lengths: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut lengths = lengths.to_dtype(DType::F32)?;
        for _ in 0..self.sampling_num {
            lengths = (&lengths + (2 * self.padding) as f64)?;
            lengths = (&lengths - self.kernel_size as f64)?;
            lengths = (&lengths / self.stride as f64)?;
            lengths = lengths.floor()?;
            lengths = (&lengths + 1.0)?;
        }
        lengths = lengths.to_dtype(DType::I64)?;

        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for conv in &self.conv {
            x = conv.forward(&x)?;
            x = x.relu()?;
        }
        let x = x.transpose(1, 2)?; // (B, T, C, F)
        let (b, t, c, f) = x.dims4()?;
        let x = x.reshape((b, t, c * f))?;
        let x = self.out.forward(&x)?;
        Ok((x, lengths))
    }
}

#[derive(Debug, Clone)]
enum PreEncode {
    Subsampling(DwStridingSubsampling),
    Linear(Linear),
}

#[derive(Debug, Clone)]
pub struct Conformer {
    pub args: ConformerArgs,
    pos_enc: Option<RelPositionalEncoding>,
    pre_encode: PreEncode,
    layers: Vec<ConformerBlock>,
}

impl Conformer {
    pub fn load(args: ConformerArgs, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let pos_enc = match args.self_attention_model.as_str() {
            "rel_pos" | "rel_pos_local_attn" => Some(RelPositionalEncoding::new(
                args.d_model,
                args.pos_emb_max_len,
                args.xscaling,
                &device,
            )?),
            _ => None,
        };

        let pre_encode = if args.subsampling_factor > 1 {
            if args.subsampling == "dw_striding" && !args.causal_downsampling {
                PreEncode::Subsampling(DwStridingSubsampling::load(&args, vb.pp("pre_encode"))?)
            } else {
                return Err(candle::Error::Msg(
                    "unsupported subsampling type".to_string(),
                ));
            }
        } else {
            PreEncode::Linear(candle_nn::linear(
                args.feat_in,
                args.d_model,
                vb.pp("pre_encode"),
            )?)
        };

        let mut layers = Vec::with_capacity(args.n_layers);
        for i in 0..args.n_layers {
            layers.push(ConformerBlock::load(&args, vb.pp(format!("layers.{i}")))?);
        }

        Ok(Self {
            args,
            pos_enc,
            pre_encode,
            layers,
        })
    }

    pub fn set_attention_model(&mut self, name: &str, context_size: Option<(usize, usize)>) {
        if name == "rel_pos_local_attn" {
            for layer in &mut self.layers {
                layer.set_attention_model(name, context_size);
            }
        } else {
            for layer in &mut self.layers {
                layer.set_attention_model(name, None);
            }
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(&mut self, x: &Tensor, lengths: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        self.forward_inner::<ConformerCache>(x, lengths, None)
    }

    pub fn forward_with_cache<C: CacheLike>(
        &mut self,
        x: &Tensor,
        lengths: Option<&Tensor>,
        cache: &mut [C],
    ) -> Result<(Tensor, Tensor)> {
        self.forward_inner::<C>(x, lengths, Some(cache))
    }

    fn forward_inner<C: CacheLike>(
        &mut self,
        x: &Tensor,
        lengths: Option<&Tensor>,
        mut cache: Option<&mut [C]>,
    ) -> Result<(Tensor, Tensor)> {
        let lengths = if let Some(lengths) = lengths {
            lengths.clone()
        } else {
            let b = x.dims3()?.0;
            let len = x.dims3()?.1 as i64;
            Tensor::from_vec(vec![len; b], (b,), x.device())?
        };

        let (mut x, out_lengths) = match &self.pre_encode {
            PreEncode::Subsampling(pre) => pre.forward(x, &lengths)?,
            PreEncode::Linear(linear) => (linear.forward(x)?, lengths),
        };

        let mut pos_emb = None;
        if let Some(pos_enc) = &mut self.pos_enc {
            let offset = cache
                .as_ref()
                .and_then(|c| c.first().map(|c| c.offset()))
                .unwrap_or(0);
            let (x_scaled, pos) = pos_enc.forward(&x, offset)?;
            x = x_scaled;
            pos_emb = Some(pos);
        }

        if let Some(cache) = cache.as_mut() {
            for (layer, cache) in self.layers.iter().zip(cache.iter_mut()) {
                x = layer.forward(&x, pos_emb.as_ref(), Some(cache))?;
            }
        } else {
            for layer in &self.layers {
                x = layer.forward(&x, pos_emb.as_ref(), None)?;
            }
        }

        Ok((x, out_lengths))
    }
}
