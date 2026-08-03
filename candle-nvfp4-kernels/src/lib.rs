//! CUDA execution for NVIDIA ModelOpt W4A16 NVFP4 linear layers.
//!
//! Weights use two E2M1 values per byte, one E4M3FN scale per 16 input values,
//! and a tensor-wide `f32` scale. The host API is intentionally batched: it
//! avoids launching and re-uploading a weight matrix once per token during prefill.

#[cfg(feature = "cuda")]
use std::ffi::{c_int, c_void};

#[cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
use candle_core::{DType, Device, Layout, Module, Result as CandleResult, Shape, Tensor};

pub const BLOCK_SIZE: usize = 16;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn candle_nvfp4_cuda_is_available() -> c_int;
    fn candle_nvfp4_linear_f32_host_batched(
        input: *const f32,
        weight: *const u8,
        scales: *const u8,
        tensor_scale: f32,
        output: *mut f32,
        tokens: i64,
        rows: i64,
        cols: i64,
    ) -> c_int;
    fn candle_nvfp4_linear_f32_device(
        input: *const f32,
        weight: *const u8,
        scales: *const u8,
        tensor_scale: f32,
        output: *mut f32,
        tokens: i64,
        rows: i64,
        cols: i64,
        stream: *mut c_void,
    ) -> c_int;
}

/// Whether the active CUDA device meets the native NVFP4 execution contract
/// (compute capability 10.0 / Blackwell or newer).
#[cfg(feature = "cuda")]
pub fn is_available() -> bool {
    unsafe { candle_nvfp4_cuda_is_available() == 1 }
}

#[cfg(not(feature = "cuda"))]
pub fn is_available() -> bool {
    false
}

fn validate(
    input: &[f32],
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    tokens: usize,
) -> Result<(), String> {
    if rows == 0 || cols == 0 || tokens == 0 || !cols.is_multiple_of(BLOCK_SIZE) {
        return Err(format!(
            "NVFP4 requires non-zero rows/tokens and cols divisible by {BLOCK_SIZE}"
        ));
    }
    if [rows, cols, tokens]
        .into_iter()
        .any(|dimension| dimension > i64::MAX as usize)
    {
        return Err("NVFP4 dimensions exceed the CUDA kernel's i64 range".into());
    }
    if input.len() != tokens * cols {
        return Err(format!(
            "NVFP4 input has {} values, expected {}",
            input.len(),
            tokens * cols
        ));
    }
    if weight.len() != rows * cols / 2 {
        return Err(format!(
            "NVFP4 packed weight has {} bytes, expected {}",
            weight.len(),
            rows * cols / 2
        ));
    }
    if scales.len() != rows * cols / BLOCK_SIZE {
        return Err(format!(
            "NVFP4 scale tensor has {} bytes, expected {}",
            scales.len(),
            rows * cols / BLOCK_SIZE
        ));
    }
    Ok(())
}

#[cfg(test)]
fn e2m1_to_f32(nibble: u8) -> f32 {
    [
        0., 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ][(nibble & 0xf) as usize]
}

#[cfg(test)]
fn e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1. } else { -1. };
    let exponent = (byte >> 3) & 0xf;
    let mantissa = byte & 0x7;
    if exponent == 0 {
        sign * mantissa as f32 / 512.
    } else if exponent == 0xf && mantissa == 0x7 {
        f32::NAN
    } else {
        sign * (1. + mantissa as f32 / 8.) * 2f32.powi(exponent as i32 - 7)
    }
}

/// Computes `[tokens, cols] @ dequant([rows, cols]).T` on CUDA.
pub fn linear_f32_host_batched(
    input: &[f32],
    packed_weight: &[u8],
    weight_scale_e4m3: &[u8],
    tensor_scale: f32,
    rows: usize,
    cols: usize,
    tokens: usize,
) -> Result<Vec<f32>, String> {
    validate(input, packed_weight, weight_scale_e4m3, rows, cols, tokens)?;
    let _ = tensor_scale;
    if !is_available() {
        return Err(
            "NVFP4 CUDA kernels require compute capability 10.0 (Blackwell) or newer".into(),
        );
    }
    #[cfg(feature = "cuda")]
    {
        let mut output = vec![0.; tokens * rows];
        let status = unsafe {
            candle_nvfp4_linear_f32_host_batched(
                input.as_ptr(),
                packed_weight.as_ptr(),
                weight_scale_e4m3.as_ptr(),
                tensor_scale,
                output.as_mut_ptr(),
                tokens as i64,
                rows as i64,
                cols as i64,
            )
        };
        if status == 0 {
            Ok(output)
        } else {
            Err(format!("NVFP4 CUDA execution failed with status {status}"))
        }
    }
    #[cfg(not(feature = "cuda"))]
    unreachable!("availability is false without the cuda feature")
}

/// Single-token convenience wrapper around [`linear_f32_host_batched`].
pub fn linear_f32_host(
    input: &[f32],
    packed_weight: &[u8],
    weight_scale_e4m3: &[u8],
    tensor_scale: f32,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    linear_f32_host_batched(
        input,
        packed_weight,
        weight_scale_e4m3,
        tensor_scale,
        rows,
        cols,
        1,
    )
}

/// A ModelOpt NVFP4 linear operator whose immutable weight and scales reside on
/// the CUDA device. Inputs and results never leave the device.
#[derive(Clone, Debug)]
pub struct Nvfp4Linear {
    packed_weight: Tensor,
    weight_scale_e4m3: Tensor,
    tensor_scale: f32,
    rows: usize,
    cols: usize,
}

impl Nvfp4Linear {
    /// Uploads a packed ModelOpt operator to `device` once.
    pub fn new(
        packed_weight: &[u8],
        weight_scale_e4m3: &[u8],
        tensor_scale: f32,
        rows: usize,
        cols: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        validate(
            &vec![0.; cols],
            packed_weight,
            weight_scale_e4m3,
            rows,
            cols,
            1,
        )
        .map_err(candle_core::Error::Msg)?;
        if !device.is_cuda() {
            candle_core::bail!("Nvfp4Linear requires a CUDA device")
        }
        if !is_available() {
            candle_core::bail!(
                "NVFP4 CUDA kernels require compute capability 10.0 (Blackwell) or newer"
            )
        }
        Ok(Self {
            packed_weight: Tensor::from_slice(packed_weight, (rows, cols / 2), device)?,
            weight_scale_e4m3: Tensor::from_slice(
                weight_scale_e4m3,
                (rows, cols / BLOCK_SIZE),
                device,
            )?,
            tensor_scale,
            rows,
            cols,
        })
    }

    /// Applies `[... , cols] @ dequant([rows, cols]).T`, preserving the leading
    /// dimensions of `xs` and replacing its final dimension with `rows`.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let dims = xs.dims();
        if dims.is_empty() || dims[dims.len() - 1] != self.cols {
            candle_core::bail!(
                "Nvfp4Linear expected input with final dimension {}, got {:?}",
                self.cols,
                dims
            )
        }
        if xs.dtype() != DType::F32 {
            candle_core::bail!("Nvfp4Linear currently requires f32 activations")
        }
        let tokens = dims[..dims.len() - 1].iter().product::<usize>();
        let input = xs.contiguous()?.reshape((tokens, self.cols))?;
        let output = input.apply_op3(
            &self.packed_weight,
            &self.weight_scale_e4m3,
            Nvfp4Matmul {
                rows: self.rows,
                cols: self.cols,
                tensor_scale: self.tensor_scale,
            },
        )?;
        let mut output_dims = dims[..dims.len() - 1].to_vec();
        output_dims.push(self.rows);
        output.reshape(output_dims)
    }
}

impl Module for Nvfp4Linear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward(xs)
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Nvfp4Matmul {
    rows: usize,
    cols: usize,
    tensor_scale: f32,
}

impl candle_core::CustomOp3 for Nvfp4Matmul {
    fn name(&self) -> &'static str {
        "nvfp4-linear"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> CandleResult<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Nvfp4Linear is CUDA-only; use linear_f32_host for the host fallback")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scales_layout: &Layout,
    ) -> CandleResult<(CudaStorage, Shape)> {
        if input.dtype() != DType::F32 || weight.dtype() != DType::U8 || scales.dtype() != DType::U8
        {
            candle_core::bail!("Nvfp4Linear expects f32 input and u8 packed weight/scales")
        }
        if !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !scales_layout.is_contiguous()
        {
            candle_core::bail!("Nvfp4Linear requires contiguous input, weight, and scale tensors")
        }
        let (tokens, cols) = input_layout.shape().dims2()?;
        if cols != self.cols
            || weight_layout.shape().dims2()? != (self.rows, self.cols / 2)
            || scales_layout.shape().dims2()? != (self.rows, self.cols / BLOCK_SIZE)
        {
            candle_core::bail!("Nvfp4Linear received tensors with incompatible shapes")
        }
        let dev = input.device();
        let stream = dev.cuda_stream();
        let input = input
            .as_cuda_slice::<f32>()?
            .slice(input_layout.start_offset()..);
        let weight = weight
            .as_cuda_slice::<u8>()?
            .slice(weight_layout.start_offset()..);
        let scales = scales
            .as_cuda_slice::<u8>()?
            .slice(scales_layout.start_offset()..);
        let mut output = unsafe { dev.alloc::<f32>(tokens * self.rows)? };
        unsafe {
            let (input_ptr, _input_guard) = input.device_ptr(&stream);
            let (weight_ptr, _weight_guard) = weight.device_ptr(&stream);
            let (scales_ptr, _scales_guard) = scales.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
            let status = candle_nvfp4_linear_f32_device(
                input_ptr as *const f32,
                weight_ptr as *const u8,
                scales_ptr as *const u8,
                self.tensor_scale,
                output_ptr as *mut f32,
                tokens as i64,
                self.rows as i64,
                self.cols as i64,
                stream.cu_stream() as *mut c_void,
            );
            if status != 0 {
                candle_core::bail!("NVFP4 CUDA execution failed with status {status}")
            }
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((tokens, self.rows)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shape_validation_rejects_malformed_packed_tensors() {
        assert!(validate(&[0.; 16], &[0; 7], &[0; 1], 1, 16, 1).is_err());
        assert!(validate(&[0.; 15], &[0; 8], &[0; 1], 1, 16, 1).is_err());
        assert!(validate(&[0.; 16], &[0; 8], &[0; 2], 1, 16, 1).is_err());
        assert!(validate(&[0.; 15], &[0; 8], &[0; 1], 1, 15, 1).is_err());
    }

    #[test]
    fn nvfp4_encoding_matches_modelopt_layout() {
        assert_eq!(e2m1_to_f32(0x0), 0.0);
        assert_eq!(e2m1_to_f32(0x7), 6.0);
        assert_eq!(e2m1_to_f32(0xd), -3.0);
        assert_eq!(e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(e4m3fn_to_f32(0x40), 2.0);
        assert_eq!(e4m3fn_to_f32(0xb8), -1.0);
    }
}
