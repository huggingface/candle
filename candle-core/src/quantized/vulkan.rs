//! Vulkan quantized storage: raw GGUF quantized bytes in VRAM.
//!
//! `QVulkanStorage` holds the raw quantized bytes (as read from the GGUF
//! file) in a `Subbuffer<[u8]>`. The decode (m == 1) matmul runs the Q4_K
//! GEMV kernel directly on the raw bytes; the prefill (m > 1) path
//! dequantizes to f32 in VRAM and runs the f32 GEMM. The other dtypes
//! (Q6_K, Q8_0, Q5_0, Q5_K) dequantize to f32 on the fly and run the f32
//! kernels.
use crate::quantized::GgmlDType;
use crate::vulkan_backend::{VBuf, VulkanDevice, VulkanStorage};
use crate::{DType, Layout, Result, Shape};
use vulkano::buffer::Subbuffer;
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
    /// Allocate a zero-filled quantized buffer for `elem_count` elements.
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let bytes = device.upload_u8(&vec![0u8; size])?;
        Ok(Self {
            bytes,
            device: device.clone(),
            dtype,
        })
    }
    /// Vulkan kernels take buffer handles; raw device pointers are not exposed.
    pub fn device_ptr(&self) -> Result<*const u8> {
        crate::bail!("vulkan: raw device pointers are not exposed by this backend")
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
    /// Dequantize to f32 in VRAM: Q4_K/Q6_K/Q8_0/Q5_0/Q5_K through the
    /// GPU dequant kernels, f32 weights by reinterpreting the buffer (no
    /// copy).
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
            GgmlDType::Q4K
            | GgmlDType::Q6K
            | GgmlDType::Q8_0
            | GgmlDType::Q5_0
            | GgmlDType::Q5K => {
                let device = self.device.clone();
                let out = device.new_f32_buffer(elem_count)?;
                let w = self.bytes.clone();
                let o = out.clone();
                let kernels = device.kernels();
                let dtype = self.dtype;
                device.execute(move |cbb| {
                    dequant_dispatch(dtype, cbb, &kernels, &w, &o, elem_count)
                })?;
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
        if !matches!(
            self.dtype,
            GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q8_0 | GgmlDType::Q5_0 | GgmlDType::Q5K
        ) {
            crate::bail!(
                "vulkan: qmatmul only supports Q4_K/Q6_K/Q8_0/Q5_0/Q5_K (got {:?})",
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
        let dtype = self.dtype;
        if m == 1 {
            if dtype == GgmlDType::Q4K {
                device.execute(move |cbb| {
                    candle_vulkan_kernels::call_q4k_qmatvec_f32(
                        cbb, &kernels, &in_buf, &w, &o, k, n,
                    )
                    .map_err(|e| e.to_string())
                })?;
            } else {
                let tmp = device.new_f32_buffer(n * k)?;
                let t = tmp.clone();
                device.execute(move |cbb| {
                    dequant_dispatch(dtype, cbb, &kernels, &w, &t, n * k)?;
                    candle_vulkan_kernels::call_gemv_f32(cbb, &kernels, &in_buf, &t, &o, k, n)
                        .map_err(|e| e.to_string())
                })?;
            }
        } else {
            // Prefill: dequant the weight to f32 in VRAM, then f32 GEMM.
            let tmp = device.new_f32_buffer(n * k)?;
            let t = tmp.clone();
            device.execute(move |cbb| {
                dequant_dispatch(dtype, cbb, &kernels, &w, &t, n * k)?;
                candle_vulkan_kernels::call_gemm_f32(
                    cbb, &kernels, &in_buf, &t, &o, 1, m, n, k, true,
                )
                .map_err(|e| e.to_string())
            })?;
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
/// Records a dequant dispatch for the dtypes handled by this backend
/// (Q4_K, Q6_K, Q8_0, Q5_0, Q5_K).
fn dequant_dispatch(
    dtype: GgmlDType,
    cbb: &mut vulkano::command_buffer::AutoCommandBufferBuilder<
        vulkano::command_buffer::PrimaryAutoCommandBuffer,
    >,
    kernels: &candle_vulkan_kernels::Kernels,
    w: &Subbuffer<[u8]>,
    out: &Subbuffer<[f32]>,
    elems: usize,
) -> std::result::Result<(), String> {
    let r: std::result::Result<(), candle_vulkan_kernels::VulkanKernelError> = match dtype {
        GgmlDType::Q4K => candle_vulkan_kernels::call_q4k_dequant_f32(cbb, kernels, w, out, elems),
        GgmlDType::Q6K => candle_vulkan_kernels::call_q6k_dequant_f32(cbb, kernels, w, out, elems),
        GgmlDType::Q8_0 => candle_vulkan_kernels::call_q80_dequant_f32(cbb, kernels, w, out, elems),
        GgmlDType::Q5_0 => candle_vulkan_kernels::call_q50_dequant_f32(cbb, kernels, w, out, elems),
        GgmlDType::Q5K => candle_vulkan_kernels::call_q5k_dequant_f32(cbb, kernels, w, out, elems),
        other => return Err(format!("vulkan: dequantize of {:?} not implemented", other)),
    };
    r.map_err(|e| e.to_string())
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
