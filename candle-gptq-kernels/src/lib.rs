//! Fused dequantize+GEMM CUDA kernel for GPTQ (AutoGPTQ/GPTQModel "old" CUDA layout)
//! quantized linear layers.
//!
//! Unlike `candle_transformers::quantized_gptq`, which dequantizes a GPTQ checkpoint into a
//! dense `f32` weight once at load time and then runs the regular matmul, this crate fuses the
//! per-element dequantization into the GEMM kernel itself so the dense weight is never
//! materialized. Two kernels are provided: [`gptq_gemm`], a shared-memory-tiled kernel with
//! scalar FP32 accumulation (any `bits` dividing 32), and [`gptq_gemm_tensor_core`], a 4-bit-only
//! kernel that runs the GEMM inner product on tensor cores via the WMMA API instead.
//!
//! For the common 4-bit symmetric case there is also [`marlin_gemm`], an FFI binding to the real
//! Marlin tensor-core kernel (vendored CUDA C++ under `kernels/marlin/`); use
//! [`marlin_repack_gptq`] once at load time to convert a checkpoint into Marlin's layout.
//!
//! With the `metal` feature the same [`gptq_gemm`] entry point runs a Metal port of the scalar
//! tiled kernel on Apple Silicon (see `src/metal.rs` / `kernels/gptq_gemm.metal`); the tensor-core
//! and Marlin paths remain CUDA-only.

#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "cuda")]
mod marlin;

#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "cuda")]
pub use marlin::{marlin_gemm, marlin_repack_gptq};

#[cfg(feature = "cuda")]
use candle::{
    backend::BackendStorage,
    cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut},
    CudaStorage, DType,
};
use candle::{CpuStorage, Layout, Result, Shape, Tensor};

/// `x.apply_op3(qweight, qzeros, op)`: `scales` and `g_idx` ride along as extra fields, mirroring
/// the pattern `candle-flash-attn` uses for `alibi_slopes` (CustomOp only supports 3 tensors).
pub struct GptqGemm {
    pub scales: Tensor,
    pub g_idx: Tensor,
    pub bits: usize,
    pub pack_factor: usize,
}

#[cfg(feature = "cuda")]
impl GptqGemm {
    fn extra_cuda_slice<'a, T: candle::cuda_backend::CudaDType>(
        storage: &'a candle::Storage,
        layout: &Layout,
        name: &'static str,
    ) -> Result<candle::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let slice = match storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("{name} must be a cuda tensor"),
        };
        match layout.contiguous_offsets() {
            Some((o1, o2)) => Ok(slice.slice(o1..o2)),
            None => candle::bail!("{name} has to be contiguous"),
        }
    }
}

impl candle::CustomOp3 for GptqGemm {
    fn name(&self) -> &'static str {
        "gptq-gemm"
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
        candle::bail!("no cpu support for the fused gptq-gemm kernel, use candle_transformers::quantized_gptq instead")
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
        metal::gptq_gemm_metal_fwd(self, x, x_l, qweight, qweight_l, qzeros, qzeros_l)
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
                "gptq-gemm only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        let (m, k) = x_l.shape().dims2()?;
        let (packed_k, n) = qweight_l.shape().dims2()?;
        if packed_k * self.pack_factor != k {
            candle::bail!(
                "gptq-gemm: qweight rows {packed_k} * pack_factor {} != x cols {k}",
                self.pack_factor
            );
        }
        let (_n_groups, n_packed) = qzeros_l.shape().dims2()?;
        if n_packed * self.pack_factor != n {
            candle::bail!(
                "gptq-gemm: qzeros cols {n_packed} * pack_factor {} != out dim {n}",
                self.pack_factor
            );
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm: x must be contiguous"),
        };
        let qweight_slice = qweight.as_cuda_slice::<i32>()?;
        let qweight_slice = match qweight_l.contiguous_offsets() {
            Some((o1, o2)) => qweight_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm: qweight must be contiguous"),
        };
        let qzeros_slice = qzeros.as_cuda_slice::<i32>()?;
        let qzeros_slice = match qzeros_l.contiguous_offsets() {
            Some((o1, o2)) => qzeros_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm: qzeros must be contiguous"),
        };
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let scales_slice = Self::extra_cuda_slice::<f32>(&scales_storage, scales_layout, "scales")?;
        let (g_idx_storage, g_idx_layout) = self.g_idx.storage_and_layout();
        let g_idx_slice = Self::extra_cuda_slice::<i32>(&g_idx_storage, g_idx_layout, "g_idx")?;

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (qweight_ptr, _guard) = qweight_slice.device_ptr(&stream);
            let (qzeros_ptr, _guard) = qzeros_slice.device_ptr(&stream);
            let (scales_ptr, _guard) = scales_slice.device_ptr(&stream);
            let (g_idx_ptr, _guard) = g_idx_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_gptq_gemm_f32(
                x_ptr as *const f32,
                qweight_ptr as *const i32,
                qzeros_ptr as *const i32,
                scales_ptr as *const f32,
                g_idx_ptr as *const i32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                self.bits as i32,
                self.pack_factor as i32,
                n_packed as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the fused GPTQ dequant+GEMM kernel: `y = x @ dequant(qweight, qzeros, scales, g_idx)`.
///
/// * `x` - activations, `f32`, shape `[m, k]`.
/// * `qweight` - packed weight, `i32`, shape `[k / pack_factor, n]`.
/// * `qzeros` - packed zero points, `i32`, shape `[n_groups, n / pack_factor]`.
/// * `scales` - per-group scales, `f32`, shape `[n_groups, n]`.
/// * `g_idx` - per-row group index, `i32`, shape `[k]`.
pub fn gptq_gemm(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<Tensor> {
    let _ = group_size;
    let pack_factor = 32 / bits;
    let op = GptqGemm {
        scales: scales.clone(),
        g_idx: g_idx.clone(),
        bits,
        pack_factor,
    };
    x.apply_op3(qweight, qzeros, op)
}

/// `x.apply_op3(qweight, qzeros, op)`: `scales` and `g_idx` ride along as extra fields, same
/// pattern as [`GptqGemm`]. CUDA-only (tensor cores / WMMA have no Metal equivalent here).
#[cfg(feature = "cuda")]
pub struct GptqGemmTensorCore {
    pub scales: Tensor,
    pub g_idx: Tensor,
}

#[cfg(feature = "cuda")]
const GPTQ_TC_BITS: usize = 4;
#[cfg(feature = "cuda")]
const GPTQ_TC_PACK_FACTOR: usize = 32 / GPTQ_TC_BITS;

#[cfg(feature = "cuda")]
impl candle::CustomOp3 for GptqGemmTensorCore {
    fn name(&self) -> &'static str {
        "gptq-gemm-tensor-core"
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
        candle::bail!("no cpu support for the fused gptq-gemm-tensor-core kernel, use candle_transformers::quantized_gptq instead")
    }

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
                "gptq-gemm-tensor-core only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        let (m, k) = x_l.shape().dims2()?;
        let (packed_k, n) = qweight_l.shape().dims2()?;
        if packed_k * GPTQ_TC_PACK_FACTOR != k {
            candle::bail!(
                "gptq-gemm-tensor-core: qweight rows {packed_k} * pack_factor {GPTQ_TC_PACK_FACTOR} != x cols {k}"
            );
        }
        let (_n_groups, n_packed) = qzeros_l.shape().dims2()?;
        if n_packed * GPTQ_TC_PACK_FACTOR != n {
            candle::bail!(
                "gptq-gemm-tensor-core: qzeros cols {n_packed} * pack_factor {GPTQ_TC_PACK_FACTOR} != out dim {n}"
            );
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm-tensor-core: x must be contiguous"),
        };
        let qweight_slice = qweight.as_cuda_slice::<i32>()?;
        let qweight_slice = match qweight_l.contiguous_offsets() {
            Some((o1, o2)) => qweight_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm-tensor-core: qweight must be contiguous"),
        };
        let qzeros_slice = qzeros.as_cuda_slice::<i32>()?;
        let qzeros_slice = match qzeros_l.contiguous_offsets() {
            Some((o1, o2)) => qzeros_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm-tensor-core: qzeros must be contiguous"),
        };
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let scales_slice =
            GptqGemm::extra_cuda_slice::<f32>(&scales_storage, scales_layout, "scales")?;
        let (g_idx_storage, g_idx_layout) = self.g_idx.storage_and_layout();
        let g_idx_slice = GptqGemm::extra_cuda_slice::<i32>(&g_idx_storage, g_idx_layout, "g_idx")?;

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (qweight_ptr, _guard) = qweight_slice.device_ptr(&stream);
            let (qzeros_ptr, _guard) = qzeros_slice.device_ptr(&stream);
            let (scales_ptr, _guard) = scales_slice.device_ptr(&stream);
            let (g_idx_ptr, _guard) = g_idx_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_gptq_gemm_tc_f32(
                x_ptr as *const f32,
                qweight_ptr as *const i32,
                qzeros_ptr as *const i32,
                scales_ptr as *const f32,
                g_idx_ptr as *const i32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                n_packed as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the tensor-core (WMMA `mma.sync`) fused GPTQ dequant+GEMM kernel, 4-bit only:
/// `y = x @ dequant(qweight, qzeros, scales, g_idx)`.
///
/// Same checkpoint layout as [`gptq_gemm`], but the GEMM itself runs on tensor cores instead
/// of scalar FMAs (see `kernels/gptq_gemm_tc.cu`). Only `bits == 4` is supported; use
/// [`gptq_gemm`] for other bit widths.
#[cfg(feature = "cuda")]
pub fn gptq_gemm_tensor_core(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: &Tensor,
    bits: usize,
) -> Result<Tensor> {
    if bits != GPTQ_TC_BITS {
        candle::bail!(
            "gptq-gemm-tensor-core only supports 4-bit weights, got {bits}-bit; use gptq_gemm instead"
        );
    }
    let op = GptqGemmTensorCore {
        scales: scales.clone(),
        g_idx: g_idx.clone(),
    };
    x.apply_op3(qweight, qzeros, op)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::Device;

    /// Plain-Rust reference, computed independently of either CUDA kernel: dequantize on the
    /// fly and accumulate in `f64` to get a value both fused kernels should agree with.
    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul_ref(
        x: &[f32],
        qweight: &[i32],
        qzeros: &[i32],
        scales: &[f32],
        g_idx: &[i32],
        m: usize,
        k: usize,
        n: usize,
        pack_factor: usize,
        bits: usize,
    ) -> Vec<f32> {
        let mask = (1i32 << bits) - 1;
        let n_packed = n / pack_factor;
        let mut y = vec![0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    let g = g_idx[kk] as usize;
                    let w_word = qweight[(kk / pack_factor) * n + col];
                    let shift_q = (kk % pack_factor) * bits;
                    let q = (w_word >> shift_q) & mask;
                    let z_word = qzeros[g * n_packed + col / pack_factor];
                    let shift_z = (col % pack_factor) * bits;
                    let z = ((z_word >> shift_z) & mask) + 1;
                    let s = scales[g * n + col];
                    let w = (q - z) as f32 * s;
                    acc += x[row * k + kk] as f64 * w as f64;
                }
                y[row * n + col] = acc as f32;
            }
        }
        y
    }

    /// Numeric correctness test for the 4-bit tensor-core kernel (`gptq_gemm_tensor_core`),
    /// cross-checked against both a plain-Rust reference and the scalar fused kernel
    /// (`gptq_gemm`). Dimensions are deliberately not multiples of the kernel's 16x16x16 tile
    /// to exercise the zero-padding paths for `M`, `K`, and `N`. Requires a CUDA device; this
    /// is run on real GPU CI (see `.github/workflows/ci_cuda.yaml`), not locally.
    #[test]
    fn gptq_gemm_tensor_core_matches_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let bits = 4;
        let pack_factor = 32 / bits;
        let m = 17; // not a multiple of WMMA_M=16
        let k: usize = 48; // multiple of pack_factor=8, not a multiple of WMMA_K=16
        let n = 24; // not a multiple of WMMA_N=16
        let group_size = 16;
        let n_groups = k.div_ceil(group_size);

        let x: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qweight: Vec<i32> = (0..(k / pack_factor) * n)
            .map(|i| {
                let mut packed = 0i32;
                for sub in 0..pack_factor {
                    let v = ((i * 7 + sub * 3) % 16) as i32;
                    packed |= v << (sub * bits);
                }
                packed
            })
            .collect();
        let qzeros: Vec<i32> = (0..n_groups * (n / pack_factor))
            .map(|i| {
                let mut packed = 0i32;
                for sub in 0..pack_factor {
                    let v = ((i * 5 + sub) % 16) as i32;
                    packed |= v << (sub * bits);
                }
                packed
            })
            .collect();
        let scales: Vec<f32> = (0..n_groups * n)
            .map(|i| 0.05 + (i % 7) as f32 * 0.01)
            .collect();
        let g_idx: Vec<i32> = (0..k as i32).map(|i| i / group_size as i32).collect();

        let expected = dequant_matmul_ref(
            &x,
            &qweight,
            &qzeros,
            &scales,
            &g_idx,
            m,
            k,
            n,
            pack_factor,
            bits,
        );

        let x_t = Tensor::from_vec(x, (m, k), &device)?;
        let qweight_t = Tensor::from_vec(qweight, (k / pack_factor, n), &device)?;
        let qzeros_t = Tensor::from_vec(qzeros, (n_groups, n / pack_factor), &device)?;
        let scales_t = Tensor::from_vec(scales, (n_groups, n), &device)?;
        let g_idx_t = Tensor::from_vec(g_idx, k, &device)?;

        let y_tc = gptq_gemm_tensor_core(&x_t, &qweight_t, &qzeros_t, &scales_t, &g_idx_t, bits)?
            .to_vec2::<f32>()?;
        let y_scalar = gptq_gemm(
            &x_t, &qweight_t, &qzeros_t, &scales_t, &g_idx_t, bits, group_size,
        )?
        .to_vec2::<f32>()?;

        for row in 0..m {
            for col in 0..n {
                let exp = expected[row * n + col];
                // The tensor-core path rounds operands to fp16 before `mma.sync`, so allow a
                // wider tolerance than the fp32 scalar kernel; this still catches indexing /
                // layout bugs, which produce errors far larger than fp16 rounding noise.
                let tc_tol = 0.05 + 0.02 * exp.abs();
                assert!(
                    (y_tc[row][col] - exp).abs() < tc_tol,
                    "tensor-core mismatch at ({row},{col}): {} vs {exp}",
                    y_tc[row][col]
                );
                assert!(
                    (y_scalar[row][col] - exp).abs() < 1e-3 + 1e-5 * exp.abs(),
                    "scalar mismatch at ({row},{col}): {} vs {exp}",
                    y_scalar[row][col]
                );
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use candle::Device;

    /// Plain-Rust reference (f64 accumulation), independent of the Metal kernel.
    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul_ref(
        x: &[f32],
        qweight: &[i32],
        qzeros: &[i32],
        scales: &[f32],
        g_idx: &[i32],
        m: usize,
        k: usize,
        n: usize,
        pack_factor: usize,
        bits: usize,
    ) -> Vec<f32> {
        let mask = (1i32 << bits) - 1;
        let n_packed = n / pack_factor;
        let mut y = vec![0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    let g = g_idx[kk] as usize;
                    let w_word = qweight[(kk / pack_factor) * n + col];
                    let shift_q = (kk % pack_factor) * bits;
                    let q = (w_word >> shift_q) & mask;
                    let z_word = qzeros[g * n_packed + col / pack_factor];
                    let shift_z = (col % pack_factor) * bits;
                    let z = ((z_word >> shift_z) & mask) + 1;
                    let s = scales[g * n + col];
                    let w = (q - z) as f32 * s;
                    acc += x[row * k + kk] as f64 * w as f64;
                }
                y[row * n + col] = acc as f32;
            }
        }
        y
    }

    /// Numeric correctness test for the Metal scalar kernel. Dimensions are deliberately not
    /// multiples of the 16x16 tile to exercise the zero-padding paths. Requires a Metal device;
    /// run on the macOS GPU CI runner (see `.github/workflows/ci_metal.yaml`).
    #[test]
    fn gptq_gemm_metal_matches_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        let bits = 4;
        let pack_factor = 32 / bits;
        let m = 17; // not a multiple of TILE=16
        let k: usize = 48; // multiple of pack_factor=8, not a multiple of TILE=16
        let n = 24; // not a multiple of TILE=16
        let group_size = 16;
        let n_groups = k.div_ceil(group_size);

        let x: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qweight: Vec<i32> = (0..(k / pack_factor) * n)
            .map(|i| {
                let mut packed = 0i32;
                for sub in 0..pack_factor {
                    let v = ((i * 7 + sub * 3) % 16) as i32;
                    packed |= v << (sub * bits);
                }
                packed
            })
            .collect();
        let qzeros: Vec<i32> = (0..n_groups * (n / pack_factor))
            .map(|i| {
                let mut packed = 0i32;
                for sub in 0..pack_factor {
                    let v = ((i * 5 + sub) % 16) as i32;
                    packed |= v << (sub * bits);
                }
                packed
            })
            .collect();
        let scales: Vec<f32> = (0..n_groups * n)
            .map(|i| 0.05 + (i % 7) as f32 * 0.01)
            .collect();
        let g_idx: Vec<i32> = (0..k as i32).map(|i| i / group_size as i32).collect();

        let expected = dequant_matmul_ref(
            &x,
            &qweight,
            &qzeros,
            &scales,
            &g_idx,
            m,
            k,
            n,
            pack_factor,
            bits,
        );

        let x_t = Tensor::from_vec(x, (m, k), &device)?;
        let qweight_t = Tensor::from_vec(qweight, (k / pack_factor, n), &device)?;
        let qzeros_t = Tensor::from_vec(qzeros, (n_groups, n / pack_factor), &device)?;
        let scales_t = Tensor::from_vec(scales, (n_groups, n), &device)?;
        let g_idx_t = Tensor::from_vec(g_idx, k, &device)?;

        let y = gptq_gemm(
            &x_t, &qweight_t, &qzeros_t, &scales_t, &g_idx_t, bits, group_size,
        )?
        .to_vec2::<f32>()?;

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
