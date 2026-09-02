// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use super::{conv, quantization, seanet, transformer};
use candle::{DType, Device, Module, Result, StreamTensor, StreamingModule, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    pub seanet: seanet::Config,
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/b7d2bd5a/.hydra/config.yaml
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: candle_nn::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: 0,
            norm: conv::Norm::WeightNorm,
            pad_mode: conv::PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer_cfg = transformer::Config {
            d_model: seanet_cfg.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            conv_kernel_size: 5,
            use_conv_bias: true,
            use_conv_block: false,
            cross_attention: false,
            max_period: 10000,
            gating: None,
            norm: super::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,

            dim_feedforward: 2048,
            kv_repeat: 1,
            conv_layout: true, // see builders.py
            max_seq_len: 8192, // the transformer works at 25hz so this is ~5 mins.
        };
        Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Encodec {
    encoder: seanet::SeaNetEncoder,
    decoder: seanet::SeaNetDecoder,
    encoder_transformer: transformer::ProjectedTransformer,
    decoder_transformer: transformer::ProjectedTransformer,
    downsample: conv::ConvDownsample1d,
    upsample: conv::ConvTrUpsample1d,
    quantizer: quantization::SplitResidualVectorQuantizer,
    config: Config,
}

impl Encodec {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.seanet.dimension;
        let encoder = seanet::SeaNetEncoder::new(&cfg.seanet, vb.pp("encoder"))?;
        let decoder = seanet::SeaNetDecoder::new(&cfg.seanet, vb.pp("decoder"))?;
        let encoder_transformer = transformer::ProjectedTransformer::new(
            dim,
            &[dim],
            &cfg.transformer,
            vb.pp("encoder_transformer"),
        )?;
        let decoder_transformer = transformer::ProjectedTransformer::new(
            dim,
            &[dim],
            &cfg.transformer,
            vb.pp("decoder_transformer"),
        )?;
        let quantizer = quantization::SplitResidualVectorQuantizer::new(
            /* dim */ cfg.quantizer_dim,
            /* input_dim */ Some(dim),
            /* output_dim */ Some(dim),
            /* n_q */ cfg.quantizer_n_q,
            /* bins */ cfg.quantizer_bins,
            vb.pp("quantizer"),
        )?;
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;

        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
        // `upsample` and `downsample` only apply if frame_rate is different from encoder_frame_rate.
        let downsample = conv::ConvDownsample1d::new(
            /* stride */ downsample_stride,
            /* dim */ dim,
            /* causal */ true,
            /* learnt */ true,
            vb.pp("downsample"),
        )?;
        let upsample = conv::ConvTrUpsample1d::new(
            /* stride */ downsample_stride,
            /* dim */ dim,
            /* causal */ true,
            /* learnt */ true,
            vb.pp("upsample"),
        )?;

        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            config: cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn encode_pre_quantize(&mut self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.encoder.forward(xs)?;
        self.encoder_transformer.reset_state();
        let xs = self.encoder_transformer.forward(&xs)?;
        let xs = &xs[0];
        xs.apply(&self.downsample)
    }

    pub fn encode(&mut self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.encoder.forward(xs)?;
        self.encoder_transformer.reset_state();
        let xs = self.encoder_transformer.forward(&xs)?;
        let xs = &xs[0];
        let xs = xs.apply(&self.downsample)?;
        let codes = self.quantizer.encode(&xs)?;
        Ok(codes)
    }

    pub fn encode_step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let xs = self.encoder.step(xs)?;
        let xs = self.encoder_transformer.step(&xs)?;
        let xs = self.downsample.step(&xs)?;
        match xs.as_option() {
            None => Ok(().into()),
            Some(xs) => {
                let codes = self.quantizer.encode(xs)?;
                Ok(codes.into())
            }
        }
    }

    pub fn decode(&mut self, codes: &Tensor) -> Result<Tensor> {
        let emb = self.quantizer.decode(codes)?;
        let emb = emb.apply(&self.upsample)?;
        self.decoder_transformer.reset_state();
        let outs = self.decoder_transformer.forward(&emb)?;
        let out = &outs[0];
        self.decoder.forward(out)
    }

    pub fn decode_step(&mut self, codes: &StreamTensor) -> Result<StreamTensor> {
        let emb = match codes.as_option() {
            Some(codes) => StreamTensor::from_tensor(self.quantizer.decode(codes)?),
            None => StreamTensor::empty(),
        };
        let emb = self.upsample.step(&emb)?;
        let out = self.decoder_transformer.step(&emb)?;
        self.decoder.step(&out)
    }

    pub fn reset_state(&mut self) {
        self.encoder.reset_state();
        self.encoder_transformer.reset_state();
        self.decoder.reset_state();
        self.decoder_transformer.reset_state();
        self.upsample.reset_state();
    }
}

pub fn load(model_file: &str, num_codebooks: Option<usize>, dev: &Device) -> Result<Encodec> {
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, dev)? };
    let cfg = Config::v0_1(num_codebooks);
    let encodec = Encodec::new(cfg, vb)?;
    Ok(encodec)
}
