use candle::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Module, Result, Tensor,
};
use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;

use super::ops;

#[derive(Debug, Clone, Deserialize)]
pub enum QuantMethodType {
    #[serde(rename = "fp8")]
    Fp8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizedConfig {
    pub weight_block_size: Option<Vec<usize>>,
    pub quant_method: QuantMethodType,
}

pub enum BlockwiseFP8Linear {
    Quantized { w: QMatMul, b: Option<Tensor> },
    Unquantized(Linear),
}

impl Module for BlockwiseFP8Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized { w, b } => {
                let in_ty = xs.dtype();
                let mut xs = xs.to_dtype(DType::F32)?;

                xs = w.forward(&xs)?;
                if let Some(bias) = b {
                    xs = xs.broadcast_add(bias)?;
                }

                xs.to_dtype(in_ty)
            }
            Self::Unquantized(l) => xs.apply(l),
        }
    }
}

/// Load a blockwise quantized FP8 layer and optionally quantize it in-place for faster inference.
pub fn blockwise_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    bias: bool,
    quant: Option<GgmlDType>,
    vb: VarBuilder,
) -> Result<BlockwiseFP8Linear> {
    let Some(config) = config else {
        return Ok(BlockwiseFP8Linear::Unquantized(candle_nn::linear_b(
            in_dim, out_dim, bias, vb,
        )?));
    };

    if !matches!(config.quant_method, QuantMethodType::Fp8) {
        candle::bail!("Expected FP8 quant method!");
    }

    let weight_block_size = config
        .weight_block_size
        .as_ref()
        .expect("Blockwise FP8 requires weight_block_size in config");
    if weight_block_size.len() != 2 {
        candle::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        Default::default(),
        DType::F32,
    )?;

    let bias_ty = if quant.is_some() {
        DType::F32
    } else {
        vb.dtype()
    };

    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?.to_dtype(bias_ty)?)
    } else {
        None
    };

    let dequant = ops::fp8_blockwise_dequantize(
        &weight,
        &weight_scale_inv,
        weight_block_size.to_vec(),
        vb.dtype(),
    )?;

    let layer = match quant {
        Some(q) => BlockwiseFP8Linear::Quantized {
            w: QMatMul::from_qtensor(QTensor::quantize(&dequant, q)?)?,
            b: bias,
        },
        None => BlockwiseFP8Linear::Unquantized(Linear::new(dequant, None)),
    };

    Ok(layer)
}
