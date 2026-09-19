//! Vulkan quantized storage: raw GGUF quantized bytes in VRAM.
//!
//! `QVulkanStorage` holds the raw quantized bytes (as read from the GGUF
//! file) in a `Subbuffer<[u8]>`. The decode (m == 1) matmul runs the Q4_K
//! GEMV kernel directly on the raw bytes; the prefill (m > 1) path
//! dequantizes to f32 in VRAM and runs the f32 GEMM. Q6_K weights are
//! dequantized to f32 on the fly and run the f32 kernels.
use vulkano::buffer::Subbuffer;
use crate::quantized::GgmlDType;
use crate::vulkan_backend::{VBuf, VulkanDevice, VulkanStorage};
use crate::{DType, Layout, Result, Shape};
pub struct QVulkanStorage {
    bytes: Subbuffer<[u8]>,
    device: VulkanDevice,
    dtype: GgmlDType,
}
impl std::fmt::Debug for QVulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QVulkanStorage")
            .field("size_in_bytes", &self.size_in_bytes())
            .field("dtype", &self.dtype)
            .finish()
    }
}
impl QVulkanStorage {
    /// Uploads `data` (raw quantized bytes) to VRAM.
    pub fn new(device: &VulkanDevice, dtype: GgmlDType, data: &[u8]) -> Result<Self> {
        let bytes = device.upload_u8(data)?;
        Ok(Self {
            bytes,
            device: device.clone(),
            dtype,
        })
    }
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    pub fn size_in_bytes(&self) -> usize {
        self.bytes.len() as usize
    }
    pub fn data(&self) -> Result<Vec<u8>> {
        self.device.download_u8(&self.bytes)
    }
    /// Dequantize to f32 in VRAM: Q4_K/Q6_K through the GPU dequant
    /// kernels, f32 weights by reinterpreting the buffer (no copy).
    pub fn dequantize_f32(&self, elem_count: usize) -> Result<VulkanStorage> {
        match self.dtype {
            GgmlDType::F32 => {
                let buf = self.bytes.clone().reinterpret::<[f32]>();
                Ok(VulkanStorage::new(
                    VBuf::F32(buf),
                    &self.device,
                    elem_count,
                    DType::F32,
                ))
            }
            GgmlDType::Q4K => {
                let device = self.device.clone();
                let out = device.new_f32_buffer(elem_count)?;
                let w = self.bytes.clone();
                let o = out.clone();
                let kernels = device.kernels();
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q4k_dequant_f32(
                        cbb,
                        &kernels,
                        &w,
                        &o,
                        elem_count,
                    )
                    .map_err(|e| e.to_string())
                })?;
                Ok(VulkanStorage::new(
                    VBuf::F32(out),
                    &self.device,
                    elem_count,
                    DType::F32,
                ))
            }
            GgmlDType::Q6K => {
                let device = self.device.clone();
                if std::env::var("QWV_DEBUG").is_ok() {
                    let db = self.data()?;
                    let mut nz = 0usize; for b in db.iter().take(2048) { if *b != 0 { nz += 1; } }
                    eprintln!("Q6K_WEIGHTS uploaded bytes len={} first2048 nonzero={}/2048 first16={:?}", db.len(), nz, db.iter().take(16).collect::<Vec<_>>());
                }
                let out = device.new_f32_buffer(elem_count)?;
                let w = self.bytes.clone();
                let o = out.clone();
                let kernels = device.kernels();
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q6k_dequant_f32(
                        cbb,
                        &kernels,
                        &w,
                        &o,
                        elem_count,
                    )
                    .map_err(|e| e.to_string())
                })?;
                if std::env::var("QWV_DEBUG").is_ok() {
                    let dv = device.download_f32(&out)?;
                    let mut nz = 0usize; for b in dv.iter().take(2048) { if *b != 0.0 { nz += 1; } }
                    eprintln!("Q6K_DEQUANT_OUT len={} first2048 nonzero={}/2048 max={:?} first8={:?}", dv.len(), nz, dv.iter().take(2048).cloned().fold(f32::NEG_INFINITY, f32::max), dv.iter().take(8).collect::<Vec<_>>());
                }
                Ok(VulkanStorage::new(
                    VBuf::F32(out),
                    &self.device,
                    elem_count,
                    DType::F32,
                ))
            }
            other => {
                crate::bail!("vulkan: dequantize of {:?} not implemented", other)
            }
        }
    }
    /// Quantized matmul: `shape` is the weight shape `(n, k)`, `storage` is
    /// the input activation (f32, `rows * k` elements, contiguous). The
    /// output is f32 with the last dim replaced by `n`.
    pub fn fwd(
        &self,
        shape: &Shape,
        storage: &VulkanStorage,
        layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        if self.dtype != GgmlDType::Q4K && self.dtype != GgmlDType::Q6K {
            crate::bail!(
                "vulkan: qmatmul only supports Q4_K/Q6_K (got {:?})",
                self.dtype
            );
        }
        let (n, k) = shape.dims2()?;
        if !layout.is_contiguous() {
            crate::bail!("vulkan: qmatmul requires a contiguous input");
        }
        let src_dims = layout.shape().dims();
        if src_dims.is_empty() || *src_dims.last().unwrap() != k {
            crate::bail!("vulkan: qmatmul input/weight shape mismatch");
        }
        let elems = layout.shape().elem_count();
        let m = elems / k;
        let offset = layout.start_offset();
        let Some(input) = storage.as_f32() else {
            crate::bail!("vulkan: qmatmul input must be f32");
        };
        // A view (e.g. `narrow` of the final position) shares the full
        // buffer; slice it so the kernel reads from the view's start.
        let input = if offset > 0 {
            input.clone().slice(offset as u64..(offset + elems) as u64)
        } else {
            input.clone()
        };
        let device = self.device.clone();
        let out = device.new_f32_buffer(m * n)?;
        let w = self.bytes.clone();
        let o = out.clone();
        let in_buf = input.clone();
        let kernels = device.kernels();
        let q6k = self.dtype == GgmlDType::Q6K;
        if m == 1 {
            if q6k {
                let tmp = device.new_f32_buffer(n * k)?;
                let t = tmp.clone();
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q6k_dequant_f32(
                        cbb,
                        &kernels,
                        &w,
                        &t,
                        n * k,
                    )
                    .map_err(|e| e.to_string())?;
                    candle_vulkan_kernels::call_gemv_f32(
                        cbb,
                        &kernels,
                        &in_buf,
                        &t,
                        &o,
                        k,
                        n,
                    )
                    .map_err(|e| e.to_string())
                })?;
            } else {
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q4k_qmatvec_f32(
                        cbb,
                        &kernels,
                        &in_buf,
                        &w,
                        &o,
                        k,
                        n,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
        } else {
            // Prefill: dequant the weight to f32 in VRAM, then f32 GEMM.
            let tmp = device.new_f32_buffer(n * k)?;
            let t = tmp.clone();
            if q6k {
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q6k_dequant_f32(
                        cbb,
                        &kernels,
                        &w,
                        &t,
                        n * k,
                    )
                    .map_err(|e| e.to_string())?;
                    candle_vulkan_kernels::call_gemm_f32(
                        cbb,
                        &kernels,
                        &in_buf,
                        &t,
                        &o,
                        1,
                        m,
                        n,
                        k,
                        true,
                    )
                    .map_err(|e| e.to_string())
                })?;
            } else {
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q4k_dequant_f32(
                        cbb,
                        &kernels,
                        &w,
                        &t,
                        n * k,
                    )
                    .map_err(|e| e.to_string())?;
                    candle_vulkan_kernels::call_gemm_f32(
                        cbb,
                        &kernels,
                        &in_buf,
                        &t,
                        &o,
                        1,
                        m,
                        n,
                        k,
                        true,
                    )
                    .map_err(|e| e.to_string())
                })?;
            }
        }
        let mut dst_dims = src_dims.to_vec();
        dst_dims.pop();
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        Ok((
            VulkanStorage::new(VBuf::F32(out), &self.device, m * n, DType::F32),
            dst_shape,
        ))
    }
}
/// Uploads raw quantized bytes to the Vulkan device.
pub fn load_quantized(
    device: &VulkanDevice,
    data: &[u8],
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    Ok(super::QStorage::Vulkan(QVulkanStorage::new(
        device, dtype, data,
    )?))
}
