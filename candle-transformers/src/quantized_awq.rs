//! Loading and dequantizing AWQ checkpoints (AutoAWQ "GEMM" layout) from safetensors.
//!
//! AWQ packs `bits`-wide integers into `i32` words along the *output*-feature axis (the
//! opposite axis from GPTQ), with the nibbles within each word permuted by a fixed reorder
//! table inherited from the original llm-awq CUDA kernel. This module unpacks those checkpoints
//! into a dense `f32` weight matrix at load time; there is no fused AWQ GEMM kernel here.

use candle::{DType, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// AWQ quantization parameters, as found in the checkpoint's `quantize_config.json`.
#[derive(Debug, Clone, Copy)]
pub struct AwqConfig {
    pub bits: usize,
    pub group_size: usize,
}

/// For 4-bit AWQ (GEMM layout), the value stored at in-word position `i` (bits `[i*4, i*4+4)`)
/// belongs to output column `col_group * 8 + AWQ_ORDER_4BIT[i]`. This is the inverse of the
/// `order_map = [0, 2, 4, 6, 1, 3, 5, 7]` permutation AutoAWQ uses when packing.
const AWQ_ORDER_4BIT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

fn awq_order(bits: usize) -> Result<&'static [usize]> {
    match bits {
        4 => Ok(&AWQ_ORDER_4BIT),
        _ => candle::bail!("awq: unsupported bits {bits}, only 4-bit GEMM layout is supported"),
    }
}

/// Unpack an AWQ-quantized (GEMM layout) weight into a dense `[in_dim, out_dim]` `f32` tensor.
///
/// `qweight` has shape `[in_dim, out_dim / pack_factor]`, `qzeros` has shape
/// `[n_groups, out_dim / pack_factor]`, and `scales` has shape `[n_groups, out_dim]`, where
/// `pack_factor = 32 / bits` and `n_groups = ceil(in_dim / group_size)`. Unlike GPTQ, AWQ zero
/// points are used as-is, with no `+1` offset.
pub fn dequantize_awq(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<Tensor> {
    let order = awq_order(bits)?;
    let pack_factor = 32 / bits;
    let mask = (1i32 << bits) - 1;

    let (in_dim, packed_out) = qweight.dims2()?;
    let out_dim = packed_out * pack_factor;
    let qweight = qweight.to_dtype(DType::I32)?.to_vec2::<i32>()?;
    let qzeros = qzeros.to_dtype(DType::I32)?.to_vec2::<i32>()?;
    let scales = scales.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let mut weight = vec![0f32; in_dim * out_dim];
    for (row, qweight_row) in qweight.iter().enumerate() {
        let group = row / group_size;
        let qzeros_row = &qzeros[group];
        let scales_row = &scales[group];
        for (col_group, (&w_word, &z_word)) in qweight_row.iter().zip(qzeros_row.iter()).enumerate()
        {
            for (j, &shift_pos) in order.iter().enumerate() {
                let shift = shift_pos * bits;
                let q = (w_word >> shift) & mask;
                let z = (z_word >> shift) & mask;
                let col = col_group * pack_factor + j;
                weight[row * out_dim + col] = (q - z) as f32 * scales_row[col];
            }
        }
    }
    Tensor::from_vec(weight, (in_dim, out_dim), &candle::Device::Cpu)
}

/// Build a [`candle_nn::Linear`] layer from an AWQ-quantized (GEMM layout) checkpoint, reading
/// `qweight`/`qzeros`/`scales`/(optional `bias`) tensors at the current `VarBuilder` path.
pub fn awq_linear(
    in_dim: usize,
    out_dim: usize,
    cfg: AwqConfig,
    bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    let pack_factor = 32 / cfg.bits;
    let n_groups = in_dim.div_ceil(cfg.group_size);

    let qweight = vb.get_with_hints_dtype(
        (in_dim, out_dim / pack_factor),
        "qweight",
        Default::default(),
        DType::I32,
    )?;
    let qzeros = vb.get_with_hints_dtype(
        (n_groups, out_dim / pack_factor),
        "qzeros",
        Default::default(),
        DType::I32,
    )?;
    let scales = vb.get_with_hints_dtype(
        (n_groups, out_dim),
        "scales",
        Default::default(),
        DType::F32,
    )?;

    let weight = dequantize_awq(&qweight, &qzeros, &scales, cfg.bits, cfg.group_size)?;
    let weight = weight.t()?.contiguous()?.to_device(vb.device())?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// Fused dequantize+GEMM CUDA path: keeps the AWQ checkpoint packed and runs the kernel from
/// `candle-awq-kernels` on every forward pass instead of dequantizing once at load time.
#[cfg(feature = "awq-cuda")]
pub mod cuda {
    use super::AwqConfig;
    use candle::{DType, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[derive(Debug, Clone)]
    pub struct AwqLinearCuda {
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    }

    impl AwqLinearCuda {
        pub fn new(
            in_dim: usize,
            out_dim: usize,
            cfg: AwqConfig,
            bias: bool,
            vb: VarBuilder,
        ) -> Result<Self> {
            if cfg.bits != 4 {
                candle::bail!(
                    "awq-cuda: only 4-bit GEMM layout is supported, got {}",
                    cfg.bits
                );
            }
            let pack_factor = 32 / cfg.bits;
            let n_groups = in_dim.div_ceil(cfg.group_size);
            let qweight = vb.get_with_hints_dtype(
                (in_dim, out_dim / pack_factor),
                "qweight",
                Default::default(),
                DType::I32,
            )?;
            let qzeros = vb.get_with_hints_dtype(
                (n_groups, out_dim / pack_factor),
                "qzeros",
                Default::default(),
                DType::I32,
            )?;
            let scales = vb.get_with_hints_dtype(
                (n_groups, out_dim),
                "scales",
                Default::default(),
                DType::F32,
            )?;
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(Self {
                qweight,
                qzeros,
                scales,
                bias,
                group_size: cfg.group_size,
            })
        }
    }

    impl Module for AwqLinearCuda {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let in_dims = xs.dims();
            let in_dim = *in_dims.last().unwrap();
            let m: usize = in_dims[..in_dims.len() - 1].iter().product();
            let xs2 = xs
                .reshape((m, in_dim))?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let ys = candle_awq_kernels::awq_gemm(
                &xs2,
                &self.qweight,
                &self.qzeros,
                &self.scales,
                self.group_size,
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

/// Fused dequantize+GEMM Metal path: keeps the AWQ checkpoint packed and runs the Metal kernel
/// from `candle-awq-kernels` on every forward pass (scalar tiled GEMM, 4-bit GEMM layout).
#[cfg(feature = "awq-metal")]
pub mod metal {
    use super::AwqConfig;
    use candle::{DType, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[derive(Debug, Clone)]
    pub struct AwqLinearMetal {
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    }

    impl AwqLinearMetal {
        pub fn new(
            in_dim: usize,
            out_dim: usize,
            cfg: AwqConfig,
            bias: bool,
            vb: VarBuilder,
        ) -> Result<Self> {
            if cfg.bits != 4 {
                candle::bail!(
                    "awq-metal: only 4-bit GEMM layout is supported, got {}",
                    cfg.bits
                );
            }
            let pack_factor = 32 / cfg.bits;
            let n_groups = in_dim.div_ceil(cfg.group_size);
            let qweight = vb.get_with_hints_dtype(
                (in_dim, out_dim / pack_factor),
                "qweight",
                Default::default(),
                DType::I32,
            )?;
            let qzeros = vb.get_with_hints_dtype(
                (n_groups, out_dim / pack_factor),
                "qzeros",
                Default::default(),
                DType::I32,
            )?;
            let scales = vb.get_with_hints_dtype(
                (n_groups, out_dim),
                "scales",
                Default::default(),
                DType::F32,
            )?;
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(Self {
                qweight,
                qzeros,
                scales,
                bias,
                group_size: cfg.group_size,
            })
        }
    }

    impl Module for AwqLinearMetal {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let in_dims = xs.dims();
            let in_dim = *in_dims.last().unwrap();
            let m: usize = in_dims[..in_dims.len() - 1].iter().product();
            let xs2 = xs
                .reshape((m, in_dim))?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let ys = candle_awq_kernels::awq_gemm(
                &xs2,
                &self.qweight,
                &self.qzeros,
                &self.scales,
                self.group_size,
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

    // Build a 4-bit AWQ (GEMM layout) tensor set for a single [in_dim=8, out_dim=8] weight with
    // one group, packing with AutoAWQ's forward `order_map`, and check that dequantization
    // (which uses the inverse `AWQ_ORDER_4BIT`) recovers the original values.
    #[test]
    fn dequantize_awq_4bit_roundtrip() -> Result<()> {
        let bits = 4;
        let group_size = 8;
        let in_dim = 8;
        let out_dim = 8;
        let pack_factor = 32 / bits;
        let order_map = [0usize, 2, 4, 6, 1, 3, 5, 7];

        let q: [[i32; 8]; 8] =
            std::array::from_fn(|row| std::array::from_fn(|col| ((row * 5 + col * 7) % 16) as i32));
        let zero_point: [i32; 8] = std::array::from_fn(|col| (col % 5) as i32);
        let scale: [f32; 8] = std::array::from_fn(|col| 0.25 * (1 + col as i32) as f32);

        let mut qweight_data = vec![0i32; in_dim * (out_dim / pack_factor)];
        let mut qzeros_data = vec![0i32; out_dim / pack_factor];
        for row in 0..in_dim {
            let mut w_word = 0i32;
            for (i, &col_full) in order_map.iter().enumerate().take(pack_factor) {
                w_word |= q[row][col_full] << (i * bits);
            }
            qweight_data[row] = w_word;
        }
        for (i, &col_full) in order_map.iter().enumerate().take(pack_factor) {
            qzeros_data[0] |= zero_point[col_full] << (i * bits);
        }

        let qweight =
            Tensor::from_vec(qweight_data, (in_dim, out_dim / pack_factor), &Device::Cpu)?;
        let qzeros = Tensor::from_vec(qzeros_data, (1, out_dim / pack_factor), &Device::Cpu)?;
        let scales = Tensor::from_vec(scale.to_vec(), (1, out_dim), &Device::Cpu)?;

        let weight = dequantize_awq(&qweight, &qzeros, &scales, bits, group_size)?;
        let weight = weight.to_vec2::<f32>()?;

        for row in 0..in_dim {
            for col in 0..out_dim {
                let expected = (q[row][col] - zero_point[col]) as f32 * scale[col];
                assert!((weight[row][col] - expected).abs() < 1e-6);
            }
        }
        Ok(())
    }
}
