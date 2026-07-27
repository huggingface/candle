//! Fused dequantize+GEMM CUDA kernel for AWQ (AutoAWQ "GEMM" layout, 4-bit) quantized linear
//! layers.
//!
//! Unlike `candle_transformers::quantized_awq`, which dequantizes an AWQ checkpoint into a dense
//! `f32` weight once at load time and then runs the regular matmul, this crate fuses the
//! per-element dequantization (including AWQ's output-axis nibble permutation) into a
//! shared-memory-tiled GEMM kernel so the dense weight is never materialized. The GEMM itself
//! is a straightforward shared-memory-tiled kernel with scalar FP32 accumulation; it does not
//! use tensor cores.

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

/// `x.apply_op3(qweight, qzeros, op)`: `scales` rides along as an extra field, mirroring the
/// pattern `candle-flash-attn` uses for `alibi_slopes` (CustomOp only supports 3 tensors).
pub struct AwqGemm {
    pub scales: Tensor,
    pub group_size: usize,
}

const AWQ_BITS: usize = 4;
pub(crate) const AWQ_PACK_FACTOR: usize = 32 / AWQ_BITS;

impl candle::CustomOp3 for AwqGemm {
    fn name(&self) -> &'static str {
        "awq-gemm"
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
            "no cpu support for the fused awq-gemm kernel, use candle_transformers::quantized_awq instead"
        )
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle::MetalStorage,
        x_l: &Layout,
        qweight: &candle::MetalStorage,
        qweight_l: &Layout,
        qzeros: &candle::MetalStorage,
        qzeros_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        metal::awq_gemm_metal_fwd(self, x, x_l, qweight, qweight_l, qzeros, qzeros_l)
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        qzeros: &CudaStorage,
        qzeros_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if x.dtype() != DType::F32 {
            candle::bail!(
                "awq-gemm only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        let (m, k) = x_l.shape().dims2()?;
        let (qk, n_packed) = qweight_l.shape().dims2()?;
        if qk != k {
            candle::bail!("awq-gemm: qweight rows {qk} != x cols {k}");
        }
        let n = n_packed * AWQ_PACK_FACTOR;
        let (_n_groups, qz_packed) = qzeros_l.shape().dims2()?;
        if qz_packed != n_packed {
            candle::bail!("awq-gemm: qzeros cols {qz_packed} != qweight cols {n_packed}");
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: x must be contiguous"),
        };
        let qweight_slice = qweight.as_cuda_slice::<i32>()?;
        let qweight_slice = match qweight_l.contiguous_offsets() {
            Some((o1, o2)) => qweight_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: qweight must be contiguous"),
        };
        let qzeros_slice = qzeros.as_cuda_slice::<i32>()?;
        let qzeros_slice = match qzeros_l.contiguous_offsets() {
            Some((o1, o2)) => qzeros_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: qzeros must be contiguous"),
        };

        if self.scales.dtype() != DType::F32 {
            candle::bail!(
                "awq-gemm: scales must be f32, got {:?}",
                self.scales.dtype()
            );
        }
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let scales_slice = match &*scales_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("awq-gemm: scales must be a cuda tensor"),
        };
        let scales_slice = match scales_layout.contiguous_offsets() {
            Some((o1, o2)) => scales_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: scales must be contiguous"),
        };

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (qweight_ptr, _guard) = qweight_slice.device_ptr(&stream);
            let (qzeros_ptr, _guard) = qzeros_slice.device_ptr(&stream);
            let (scales_ptr, _guard) = scales_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_awq_gemm_f32(
                x_ptr as *const f32,
                qweight_ptr as *const i32,
                qzeros_ptr as *const i32,
                scales_ptr as *const f32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                self.group_size as i32,
                n_packed as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the fused AWQ (4-bit GEMM layout) dequant+GEMM kernel:
/// `y = x @ dequant(qweight, qzeros, scales)`.
///
/// * `x` - activations, `f32`, shape `[m, k]`.
/// * `qweight` - packed weight, `i32`, shape `[k, n / pack_factor]`.
/// * `qzeros` - packed zero points, `i32`, shape `[n_groups, n / pack_factor]`.
/// * `scales` - per-group scales, `f32`, shape `[n_groups, n]`.
pub fn awq_gemm(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let op = AwqGemm {
        scales: scales.clone(),
        group_size,
    };
    x.apply_op3(qweight, qzeros, op)
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use candle::Device;

    const AWQ_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

    /// Plain-Rust reference (f64 accumulation), independent of the Metal kernel.
    fn dequant_matmul_ref(
        x: &[f32],
        qweight: &[i32],
        qzeros: &[i32],
        scales: &[f32],
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let n_packed = n / AWQ_PACK_FACTOR;
        let mut y = vec![0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let col_group = col / AWQ_PACK_FACTOR;
                let j = col % AWQ_PACK_FACTOR;
                let shift = (AWQ_ORDER[j] * AWQ_BITS) as i32;
                let mut acc = 0f64;
                for kk in 0..k {
                    let g = kk / group_size;
                    let q = (qweight[kk * n_packed + col_group] >> shift) & 0xF;
                    let z = (qzeros[g * n_packed + col_group] >> shift) & 0xF;
                    let s = scales[g * n + col];
                    let w = (q - z) as f32 * s;
                    acc += x[row * k + kk] as f64 * w as f64;
                }
                y[row * n + col] = acc as f32;
            }
        }
        y
    }

    /// Pack 8 output-axis nibbles into one i32 using AWQ's order map.
    fn pack_awq(vals: [i32; 8]) -> i32 {
        let mut w = 0i32;
        for (j, &v) in vals.iter().enumerate() {
            w |= (v & 0xF) << (AWQ_ORDER[j] * AWQ_BITS);
        }
        w
    }

    /// Numeric correctness test for the AWQ Metal kernel, cross-checked against an independent
    /// f64 reference. Requires a Metal device; run on the macOS GPU CI runner.
    #[test]
    fn awq_gemm_metal_matches_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        let m = 17; // not a multiple of TILE=16
        let k = 32;
        let n = 24; // multiple of pack_factor=8, not a multiple of TILE=16
        let group_size = 16;
        let n_groups = k / group_size;
        let n_packed = n / AWQ_PACK_FACTOR;

        let x: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qweight: Vec<i32> = (0..k * n_packed)
            .map(|i| {
                let mut vals = [0i32; 8];
                for (j, slot) in vals.iter_mut().enumerate() {
                    *slot = ((i * 7 + j * 3) % 16) as i32;
                }
                pack_awq(vals)
            })
            .collect();
        let qzeros: Vec<i32> = (0..n_groups * n_packed)
            .map(|i| {
                let mut vals = [0i32; 8];
                for (j, slot) in vals.iter_mut().enumerate() {
                    *slot = ((i * 5 + j) % 16) as i32;
                }
                pack_awq(vals)
            })
            .collect();
        let scales: Vec<f32> = (0..n_groups * n)
            .map(|i| 0.05 + (i % 7) as f32 * 0.01)
            .collect();

        let expected = dequant_matmul_ref(&x, &qweight, &qzeros, &scales, m, k, n, group_size);

        let x_t = Tensor::from_vec(x, (m, k), &device)?;
        let qweight_t = Tensor::from_vec(qweight, (k, n_packed), &device)?;
        let qzeros_t = Tensor::from_vec(qzeros, (n_groups, n_packed), &device)?;
        let scales_t = Tensor::from_vec(scales, (n_groups, n), &device)?;

        let y = awq_gemm(&x_t, &qweight_t, &qzeros_t, &scales_t, group_size)?.to_vec2::<f32>()?;

        for row in 0..m {
            for col in 0..n {
                let exp = expected[row * n + col];
                assert!(
                    (y[row][col] - exp).abs() < 1e-3 + 1e-5 * exp.abs(),
                    "metal mismatch at ({row},{col}): {} vs {exp}",
                    y[row][col]
                );
            }
        }
        Ok(())
    }
}
