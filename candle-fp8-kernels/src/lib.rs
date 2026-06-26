//! Fused dequantize+GEMM CUDA kernel for block-wise FP8 (E4M3) quantized linear layers
//! (DeepSeek-V3 style checkpoints).
//!
//! Unlike `candle_transformers::quantized_fp8`, which dequantizes a block-wise FP8 checkpoint
//! into a dense `f32` weight once at load time and then runs the regular matmul, this crate
//! fuses the per-element dequantization into a shared-memory-tiled GEMM kernel so the dense
//! weight is never materialized. The GEMM itself is a straightforward shared-memory-tiled
//! kernel with scalar FP32 accumulation; it does not use tensor cores.

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "cuda")]
use candle::{
    backend::BackendStorage,
    cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut},
    CudaStorage, DType,
};
use candle::{CpuStorage, Layout, Result, Shape, Tensor};

pub struct Fp8BlockGemm {
    pub block_size: usize,
}

impl candle::CustomOp3 for Fp8BlockGemm {
    fn name(&self) -> &'static str {
        "fp8-block-gemm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!(
            "no cpu support for the fused fp8-block-gemm kernel, use candle_transformers::quantized_fp8 instead"
        )
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle::MetalStorage,
        x_l: &Layout,
        w: &candle::MetalStorage,
        w_l: &Layout,
        scale: &candle::MetalStorage,
        scale_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        metal::fp8_block_gemm_metal_fwd(self, x, x_l, w, w_l, scale, scale_l)
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        w: &CudaStorage,
        w_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if x.dtype() != DType::F32 {
            candle::bail!(
                "fp8-block-gemm only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        if w.dtype() != DType::F8E4M3 {
            candle::bail!(
                "fp8-block-gemm expects an F8E4M3 weight, got {:?}",
                w.dtype()
            );
        }
        if scale.dtype() != DType::F32 {
            candle::bail!(
                "fp8-block-gemm expects an f32 scale, got {:?}",
                scale.dtype()
            );
        }

        let (m, k) = x_l.shape().dims2()?;
        let (n, wk) = w_l.shape().dims2()?;
        if wk != k {
            candle::bail!("fp8-block-gemm: weight cols {wk} != x cols {k}");
        }
        let (scale_rows, scale_cols) = scale_l.shape().dims2()?;
        if scale_rows != n.div_ceil(self.block_size) || scale_cols != k.div_ceil(self.block_size) {
            candle::bail!(
                "fp8-block-gemm: scale shape {:?} does not match weight shape {:?} for block_size {}",
                (scale_rows, scale_cols),
                (n, k),
                self.block_size
            );
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("fp8-block-gemm: x must be contiguous"),
        };
        let w_slice = w.as_cuda_slice::<float8::F8E4M3>()?;
        let w_slice = match w_l.contiguous_offsets() {
            Some((o1, o2)) => w_slice.slice(o1..o2),
            None => candle::bail!("fp8-block-gemm: weight must be contiguous"),
        };
        let scale_slice = scale.as_cuda_slice::<f32>()?;
        let scale_slice = match scale_l.contiguous_offsets() {
            Some((o1, o2)) => scale_slice.slice(o1..o2),
            None => candle::bail!("fp8-block-gemm: scale must be contiguous"),
        };

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (w_ptr, _guard) = w_slice.device_ptr(&stream);
            let (scale_ptr, _guard) = scale_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_fp8_block_gemm_f32(
                x_ptr as *const f32,
                w_ptr as *const core::ffi::c_void,
                scale_ptr as *const f32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                self.block_size as i32,
                scale_cols as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the fused block-wise FP8 dequant+GEMM kernel: `y = x @ dequant(weight, scale)^T`.
///
/// * `x` - activations, `f32`, shape `[m, k]`.
/// * `weight` - `F8E4M3` weight, shape `[n, k]` (standard `nn.Linear` layout).
/// * `scale` - per-block `f32` scale, shape `[ceil(n / block_size), ceil(k / block_size)]`.
pub fn fp8_block_gemm(
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    let op = Fp8BlockGemm { block_size };
    x.apply_op3(weight, scale, op)
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use candle::{DType, Device};

    /// Numeric correctness test for the block-wise FP8 Metal kernel, cross-checked against an
    /// independent f64 reference. Weight values are chosen to be exactly representable in E4M3 so
    /// the byte round-trip is lossless and the shader's E4M3 decode must match exactly. Requires a
    /// Metal device; run on the macOS GPU CI runner.
    #[test]
    fn fp8_block_gemm_metal_matches_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        let m = 17; // not a multiple of TILE=16
        let k: usize = 40; // not a multiple of TILE=16
        let n: usize = 24; // not a multiple of TILE=16
        let block_size = 16;
        let scale_rows = n.div_ceil(block_size);
        let scale_cols = k.div_ceil(block_size);

        // E4M3-exact magnitudes so f32 -> F8E4M3 -> f32 is lossless.
        let reps = [
            -8.0f32, -4.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0,
        ];
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let w: Vec<f32> = (0..n * k).map(|i| reps[(i * 7) % reps.len()]).collect();
        let scale: Vec<f32> = (0..scale_rows * scale_cols)
            .map(|i| 0.5 + (i % 5) as f32 * 0.25)
            .collect();

        let mut expected = vec![0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    let wv = w[col * k + kk];
                    let s = scale[(col / block_size) * scale_cols + kk / block_size];
                    acc += x[row * k + kk] as f64 * (wv * s) as f64;
                }
                expected[row * n + col] = acc as f32;
            }
        }

        let x_t = Tensor::from_vec(x, (m, k), &device)?;
        // Metal has no f32 -> F8E4M3 cast kernel, so quantize on CPU and move the bytes to the GPU
        // (the same E4M3 bytes a real checkpoint would carry).
        let w_t = Tensor::from_vec(w, (n, k), &Device::Cpu)?
            .to_dtype(DType::F8E4M3)?
            .to_device(&device)?;
        let scale_t = Tensor::from_vec(scale, (scale_rows, scale_cols), &device)?;

        let y = fp8_block_gemm(&x_t, &w_t, &scale_t, block_size)?.to_vec2::<f32>()?;

        for row in 0..m {
            for col in 0..n {
                let exp = expected[row * n + col];
                assert!(
                    (y[row][col] - exp).abs() < 1e-3 + 1e-4 * exp.abs(),
                    "metal mismatch at ({row},{col}): {} vs {exp}",
                    y[row][col]
                );
            }
        }
        Ok(())
    }
}
