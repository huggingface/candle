//! Vulkan quantized storage: raw GGUF quantized bytes in VRAM.
//!
//! `QVulkanStorage` holds the raw quantized bytes (as read from the GGUF
//! file) in a `U8` `VulkanStorage`. Dequantization runs a per-`GgmlDType`
//! Slang kernel that reads the raw bytes and writes f32 values; the
//! matmul (`fwd`) dequantizes the weights to f32 in VRAM and runs the f32
//! GEMM; the embedding (`embedding`) dequantizes and gathers the requested
use crate::quantized::{GgmlDType, GgmlType};
use crate::{
    backend::{BackendDevice, BackendStorage},
    CpuStorage, DType, Layout, Result, Shape, VulkanDevice, VulkanStorage,
};
use candle_vulkan_kernels::{KernelName, Source};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

pub struct QVulkanStorage {
    /// The raw quantized bytes (a `DType::U8` `VulkanStorage`).
    bytes: VulkanStorage,
    device: VulkanDevice,
    dtype: GgmlDType,
}

impl std::fmt::Debug for QVulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QVulkanStorage")
            .field("storage_size_in_bytes", &self.storage_size_in_bytes())
            .field("dtype", &self.dtype)
            .finish()
    }
}

/// Maps a quantized dtype to the dequantize kernel (source + name).
fn dequantize_kernel(dtype: GgmlDType) -> (Source, KernelName) {
    match dtype {
        GgmlDType::F32 => (Source::DequantF32, KernelName::DequantF32),
        GgmlDType::F16 => (Source::DequantF16, KernelName::DequantF16),
        GgmlDType::BF16 => (Source::DequantBf16, KernelName::DequantBf16),
        GgmlDType::Q4_0 => (Source::DequantQ40, KernelName::DequantQ40),
        GgmlDType::Q4_1 => (Source::DequantQ41, KernelName::DequantQ41),
        GgmlDType::Q5_0 => (Source::DequantQ50, KernelName::DequantQ50),
        GgmlDType::Q5_1 => (Source::DequantQ51, KernelName::DequantQ51),
        GgmlDType::Q8_0 => (Source::DequantQ80, KernelName::DequantQ80),
        GgmlDType::Q8_1 => (Source::DequantQ81, KernelName::DequantQ81),
        GgmlDType::Q2K => (Source::DequantQ2K, KernelName::DequantQ2K),
        GgmlDType::Q3K => (Source::DequantQ3K, KernelName::DequantQ3K),
        GgmlDType::Q4K => (Source::DequantQ4K, KernelName::DequantQ4K),
        GgmlDType::Q5K => (Source::DequantQ5K, KernelName::DequantQ5K),
        GgmlDType::Q6K => (Source::DequantQ6K, KernelName::DequantQ6K),
        GgmlDType::Q8K => (Source::DequantQ8K, KernelName::DequantQ8K),
    }
}

impl QVulkanStorage {
    /// Creates a zeroed quantized buffer holding `elem_count` elements of
    /// `dtype`.
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let block = dtype.block_size();
        let byte_len = elem_count / block * dtype.type_size();
        let bytes = VulkanStorage::new(device, byte_len, DType::U8)?;
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

    pub fn storage_size_in_bytes(&self) -> usize {
        self.bytes.buffer().as_u8().len() as usize
    }

    /// The raw quantized bytes as a byte-view `Subbuffer`.
    fn src_bytes(&self) -> Subbuffer<[u8]> {
        self.bytes.buffer().as_u8()
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let cpu = self.bytes.to_cpu_storage()?;
        match cpu {
            CpuStorage::U8(b) => Ok(b),
            _ => crate::bail!("vulkan quantized storage is not a U8 buffer"),
        }
    }

    /// Dequantizes `elem_count` elements to an f32 `VulkanStorage`.
    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        let block = self.dtype.block_size();
        if !elem_count.is_multiple_of(block) {
            crate::bail!(
                "dequantize: elem_count {elem_count} is not a multiple of block size {block}"
            );
        }
        let num_blocks = elem_count / block;
        let (source, name) = dequantize_kernel(self.dtype);
        let out = VulkanStorage::new(&self.device, elem_count, DType::F32)?;
        let dst = match out.buffer() {
            crate::vulkan_backend::VulkanStorageBuffer::F32(b) => b.clone(),
            _ => unreachable!("out was created as F32"),
        };
        let src = self.src_bytes();
        // params[0] = num_blocks
        let params_buf = Buffer::from_iter(
            self.device.mem_alloc(),
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![num_blocks as f32],
        )
        .map_err(|e| crate::Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[f32]> = params_buf;
        let kernels = self.device.kernels();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_dequantize_slang(
                cbb, &kernels, source, name, &src, &dst, &params, num_blocks,
            )
            .map_err(|e| e.to_string())
        })?;
        self.device.synchronize()?;
        Ok(out)
    }

    /// Gathers the requested embedding rows by dequantizing the whole
    /// `(rows, hidden)` weight matrix to f32 and gathering along dim 0.
    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids_storage: &VulkanStorage,
        _ids_l: &Layout,
    ) -> Result<VulkanStorage> {
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let data_f32 = self.dequantize(rows * hidden)?;
        let data_l = Layout::new((rows, hidden).into(), vec![hidden, 1], 0);
        // The Vulkan gather does per-element row selection (output shape = ids
        // shape), so build 2D ids `(num_ids, hidden)` with each row the id
        // repeated to gather full rows: `out[i][j] = data[ids[i]][j]`.
        let ids_cpu = ids_storage.to_cpu_storage()?;
        let ids_vals: Vec<u32> = match ids_cpu {
            CpuStorage::U32(v) => v,
            _ => crate::bail!("quantized embedding: ids must be u32"),
        };
        let num_ids = ids_vals.len();
        let ids2d: Vec<u32> = ids_vals
            .iter()
            .flat_map(|&id| std::iter::repeat_n(id, hidden))
            .collect();
        let ids2d_storage = self
            .device
            .storage_from_cpu_storage(&CpuStorage::U32(ids2d))?;
        let ids2d_l = Layout::new((num_ids, hidden).into(), vec![hidden, 1], 0);
        let out = data_f32.gather(&data_l, &ids2d_storage, &ids2d_l, 0)?;
        Ok(out)
    }

    /// Quantized matmul: dequantizes the `(n, k)` weights to f32 and runs the
    /// f32 GEMM with the `(b, m, k)` lhs.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &VulkanStorage,
        layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        let (n, k) = self_shape.dims2()?;
        let dims = layout.shape().dims().to_vec();
        let (b, m) = match dims.len() {
            2 => (1usize, dims[0]),
            3 => (dims[0], dims[1]),
            _ => crate::bail!("quantized matmul: unsupported input rank {}", dims.len()),
        };
        let k2 = *dims.last().unwrap();
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {layout:?}");
        }
        let m_total = b * m;
        let data_f32 = self.dequantize(n * k)?;
        // The weight is (n, k) contiguous; view it as the (k, n) rhs (transposed).
        let rhs_l = Layout::new((k, n).into(), vec![1, k], 0);
        let out = if storage.dtype() == DType::F32 {
            let lhs_l = Layout::new((m_total, k).into(), vec![k, 1], layout.start_offset());
            storage.matmul(&data_f32, (1, m_total, n, k), &lhs_l, &rhs_l)?
        } else {
            let orig = if storage.dtype() == DType::BF16 {
                DType::BF16
            } else {
                DType::F16
            };
            let lhs = storage.to_dtype(layout, DType::F32)?;
            let lhs_l = Layout::new((m_total, k).into(), vec![k, 1], 0);
            let out = lhs.matmul(&data_f32, (1, m_total, n, k), &lhs_l, &rhs_l)?;
            let out_l = Layout::new((m_total, n).into(), vec![n, 1], 0);
            out.to_dtype(&out_l, orig)?
        };
        let mut out_shape = dims;
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

/// Uploads raw quantized bytes to the Vulkan device.
pub fn load_quantized<T: GgmlType + Send + Sync + 'static>(
    device: &VulkanDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let dtype = T::DTYPE;
    let raw = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
        )
    };
    let bytes = device.storage_from_cpu_storage(&CpuStorage::U8(raw.to_vec()))?;
    Ok(super::QStorage::Vulkan(Box::new(QVulkanStorage {
        bytes,
        device: device.clone(),
        dtype,
    })))
}
