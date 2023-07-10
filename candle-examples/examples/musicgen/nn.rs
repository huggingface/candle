#![allow(dead_code)]

use anyhow::Result;
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor};
use std::collections::HashMap;

const MAX_SEQ_LEN: usize = 5000;

pub struct VarBuilder<'a> {
    safetensors: Option<(HashMap<String, usize>, Vec<SafeTensors<'a>>)>,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn from_safetensors(
        safetensors: Vec<SafeTensors<'a>>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        Self {
            safetensors: Some((routing, safetensors)),
            device: device.clone(),
            dtype,
        }
    }

    pub fn zeros(dtype: DType, device: &Device) -> Self {
        Self {
            safetensors: None,
            device: device.clone(),
            dtype,
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let s: Shape = s.into();
        match &self.safetensors {
            None => Tensor::zeros(s, self.dtype, &self.device),
            Some((routing, safetensors)) => {
                // Unwrap or 0  just to let the proper error flow.
                let index = routing.get(tensor_name).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(tensor_name, &self.device)?
                    .to_dtype(self.dtype)?;
                if *tensor.shape() != s {
                    let msg = format!("shape mismatch for {tensor_name}");
                    Err(candle::Error::UnexpectedShape {
                        msg,
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}

pub type Linear = candle_nn::Linear;

pub fn linear(size1: usize, size2: usize, bias: bool, p: &str, vb: &VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
    let bias = if bias {
        Some(vb.get(size2, &format!("{p}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

pub type LayerNorm = candle_nn::LayerNorm;

pub fn layer_norm(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (
        vb.get(size, &format!("{p}.weight")),
        vb.get(size, &format!("{p}.bias")),
    ) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (
                vb.get(size, &format!("{p}.gamma")),
                vb.get(size, &format!("{p}.beta")),
            ) {
                (weight, bias)
            } else {
                return Err(err.into());
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug)]
pub struct Dropout {
    pr: f64,
}

impl Dropout {
    pub fn new(pr: f64) -> Self {
        Self { pr }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

pub type Embedding = candle_nn::Embedding;

pub fn embedding(
    vocab_size: usize,
    hidden_size: usize,
    p: &str,
    vb: &VarBuilder,
) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
    Ok(Embedding::new(embeddings, hidden_size))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvConfig {
    pub padding: usize,
    pub stride: usize,
}

#[derive(Debug)]
pub struct Conv1D {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig,
}

impl Conv1D {
    // Applies weight norm for inference by recomputing the weight tensor. This
    // does not apply to training.
    // https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    pub fn load_weight_norm(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight_g = vb.get((out_c, 1, 1), &format!("{p}.weight_g"))?;
        let weight_v = vb.get((out_c, in_c, kernel_size), &format!("{p}.weight_v"))?;
        let norm_v = (&weight_v * &weight_v)?.sum(&[1, 2])?.sqrt()?;
        let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
        let bias = vb.get(out_c, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            config,
        })
    }

    pub fn load(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((out_c, in_c, kernel_size), &format!("{p}.weight"))?;
        let bias = vb.get(out_c, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            config,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    pub fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}
