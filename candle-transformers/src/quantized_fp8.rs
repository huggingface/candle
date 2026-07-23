//! Loading block-wise FP8 (E4M3) checkpoints from safetensors.
//!
//! Checkpoints such as DeepSeek-V3's store each linear weight as an `F8E4M3` tensor plus a
//! per-block `f32` scale (`weight_scale_inv` in the original checkpoints, typically with a
//! 128x128 block size): `weight[i, j] = fp8_weight[i, j] * scale[i / block_size, j / block_size]`.
//! This module dequantizes such weights into a dense `f32` tensor at load time using Candle's
//! existing `F8E4M3` dtype; there is no fused block-wise FP8 matmul kernel here.

use candle::{DType, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Block-wise FP8 quantization parameters.
#[derive(Debug, Clone, Copy)]
pub struct Fp8BlockConfig {
    pub block_size: usize,
}

/// Dequantize a block-wise FP8 weight into a dense `[out_dim, in_dim]` `f32` tensor.
///
/// `weight` is an `F8E4M3` tensor of shape `[out_dim, in_dim]` and `scale` is an `f32` (or
/// castable) tensor of shape `[ceil(out_dim / block_size), ceil(in_dim / block_size)]` holding
/// one scale per `block_size x block_size` block of `weight`.
pub fn dequantize_fp8_blockwise(
    weight: &Tensor,
    scale: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    let (out_dim, in_dim) = weight.dims2()?;
    let (scale_rows, scale_cols) = scale.dims2()?;
    if scale_rows != out_dim.div_ceil(block_size) || scale_cols != in_dim.div_ceil(block_size) {
        candle::bail!(
            "fp8: scale shape {:?} does not match weight shape {:?} for block_size {block_size}",
            (scale_rows, scale_cols),
            (out_dim, in_dim)
        )
    }
    let weight = weight.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let scale = scale.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let mut out = vec![0f32; out_dim * in_dim];
    for (i, weight_row) in weight.iter().enumerate() {
        let scale_row = &scale[i / block_size];
        for (j, &w) in weight_row.iter().enumerate() {
            out[i * in_dim + j] = w * scale_row[j / block_size];
        }
    }
    Tensor::from_vec(out, (out_dim, in_dim), &candle::Device::Cpu)
}

/// Build a [`candle_nn::Linear`] layer from a block-wise FP8 checkpoint, reading `weight`
/// (`F8E4M3`), `weight_scale_inv` (`f32`) and (optional) `bias` tensors at the current
/// `VarBuilder` path.
pub fn fp8_block_linear(
    in_dim: usize,
    out_dim: usize,
    cfg: Fp8BlockConfig,
    bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;
    let scale = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(cfg.block_size),
            in_dim.div_ceil(cfg.block_size),
        ),
        "weight_scale_inv",
        Default::default(),
        DType::F32,
    )?;
    let weight =
        dequantize_fp8_blockwise(&weight, &scale, cfg.block_size)?.to_device(vb.device())?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// Fused dequantize+GEMM CUDA path: keeps the FP8 weight in `F8E4M3` and runs the kernel from
/// `candle-fp8-kernels` on every forward pass instead of dequantizing once at load time.
#[cfg(feature = "fp8-cuda")]
pub mod cuda {
    use super::Fp8BlockConfig;
    use candle::{DType, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[derive(Debug, Clone)]
    pub struct Fp8BlockLinearCuda {
        weight: Tensor,
        scale: Tensor,
        bias: Option<Tensor>,
        block_size: usize,
    }

    impl Fp8BlockLinearCuda {
        pub fn new(
            in_dim: usize,
            out_dim: usize,
            cfg: Fp8BlockConfig,
            bias: bool,
            vb: VarBuilder,
        ) -> Result<Self> {
            let weight = vb.get_with_hints_dtype(
                (out_dim, in_dim),
                "weight",
                Default::default(),
                DType::F8E4M3,
            )?;
            let scale = vb.get_with_hints_dtype(
                (
                    out_dim.div_ceil(cfg.block_size),
                    in_dim.div_ceil(cfg.block_size),
                ),
                "weight_scale_inv",
                Default::default(),
                DType::F32,
            )?;
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(Self {
                weight,
                scale,
                bias,
                block_size: cfg.block_size,
            })
        }
    }

    impl Module for Fp8BlockLinearCuda {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let in_dims = xs.dims();
            let in_dim = *in_dims.last().unwrap();
            let m: usize = in_dims[..in_dims.len() - 1].iter().product();
            let xs2 = xs
                .reshape((m, in_dim))?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let ys = candle_fp8_kernels::fp8_block_gemm(
                &xs2,
                &self.weight,
                &self.scale,
                self.block_size,
            )?;
            let out_dim = ys.dim(1)?;
            let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
            out_dims.push(out_dim);
            let ys = ys.reshape(out_dims)?;
            match &self.bias {
                None => Ok(ys),
                Some(bias) => ys.broadcast_add(bias),
            }
        }
    }
}

/// Fused dequantize+GEMM Metal path: keeps the block-wise FP8 checkpoint packed and runs the
/// Metal kernel from `candle-fp8-kernels` on every forward pass (scalar tiled GEMM, E4M3 decoded
/// in-shader).
#[cfg(feature = "fp8-metal")]
pub mod metal {
    use super::Fp8BlockConfig;
    use candle::{DType, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[derive(Debug, Clone)]
    pub struct Fp8BlockLinearMetal {
        weight: Tensor,
        scale: Tensor,
        bias: Option<Tensor>,
        block_size: usize,
    }

    impl Fp8BlockLinearMetal {
        pub fn new(
            in_dim: usize,
            out_dim: usize,
            cfg: Fp8BlockConfig,
            bias: bool,
            vb: VarBuilder,
        ) -> Result<Self> {
            let weight = vb.get_with_hints_dtype(
                (out_dim, in_dim),
                "weight",
                Default::default(),
                DType::F8E4M3,
            )?;
            let scale = vb.get_with_hints_dtype(
                (
                    out_dim.div_ceil(cfg.block_size),
                    in_dim.div_ceil(cfg.block_size),
                ),
                "weight_scale_inv",
                Default::default(),
                DType::F32,
            )?;
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(Self {
                weight,
                scale,
                bias,
                block_size: cfg.block_size,
            })
        }
    }

    impl Module for Fp8BlockLinearMetal {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let in_dims = xs.dims();
            let in_dim = *in_dims.last().unwrap();
            let m: usize = in_dims[..in_dims.len() - 1].iter().product();
            let xs2 = xs
                .reshape((m, in_dim))?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let ys = candle_fp8_kernels::fp8_block_gemm(
                &xs2,
                &self.weight,
                &self.scale,
                self.block_size,
            )?;
            let out_dim = ys.dim(1)?;
            let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
            out_dims.push(out_dim);
            let ys = ys.reshape(out_dims)?;
            match &self.bias {
                None => Ok(ys),
                Some(bias) => ys.broadcast_add(bias),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn dequantize_fp8_blockwise_roundtrip() -> Result<()> {
        let block_size = 2;
        let out_dim = 4;
        let in_dim = 4;

        // Values chosen to be exactly representable in E4M3 so the round-trip is exact.
        let w: [[f32; 4]; 4] = [
            [1.0, 2.0, -1.0, 0.5],
            [4.0, -2.0, 1.5, -0.5],
            [0.25, 0.25, 8.0, -8.0],
            [-0.25, 1.0, -4.0, 2.0],
        ];
        let scale: [[f32; 2]; 2] = [[2.0, 0.5], [1.0, 4.0]];

        let w_flat: Vec<f32> = w.iter().flatten().copied().collect();
        let weight_f32 = Tensor::from_vec(w_flat, (out_dim, in_dim), &Device::Cpu)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;
        let scale_flat: Vec<f32> = scale.iter().flatten().copied().collect();
        let scale_tensor = Tensor::from_vec(scale_flat, (2, 2), &Device::Cpu)?;

        let dequant = dequantize_fp8_blockwise(&weight, &scale_tensor, block_size)?;
        let dequant = dequant.to_vec2::<f32>()?;

        for i in 0..out_dim {
            for j in 0..in_dim {
                let expected = w[i][j] * scale[i / block_size][j / block_size];
                assert!(
                    (dequant[i][j] - expected).abs() < 1e-5,
                    "mismatch at ({i},{j}): {} vs {expected}",
                    dequant[i][j]
                );
            }
        }
        Ok(())
    }
}
