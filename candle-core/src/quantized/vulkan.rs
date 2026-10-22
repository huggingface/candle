//! Vulkan quantized storage: raw GGUF quantized bytes in VRAM.
//!
//! `QVulkanStorage` holds the raw quantized bytes (as read from the GGUF
//! file) in a `U8` `VulkanStorage`. Dequantization runs a per-`GgmlDType`
//! Slang kernel that reads the raw bytes and writes f32 values; the
//! matmul (`fwd`) dequantizes the weights to f32 in VRAM and runs the f32
//! GEMM; the embedding (`embedding`) dequantizes and gathers the requested
use crate::quantized::{GgmlDType, GgmlType, QStorage};
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
        // Storage is allocated uninitialized; zero it on the GPU.
        bytes.zero_fill()?;
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

    /// Replaces the quantized bytes with `bytes` (same dtype and size).
    pub fn set_bytes(&mut self, bytes: VulkanStorage) {
        self.bytes = bytes;
    }

    /// Quantizes the f32 values in `src` (on the Vulkan device) into this
    /// storage.
    ///
    /// Quantization only happens on the CPU for now: the f32 source is read
    /// back, quantized with the CPU path, and the quantized bytes are
    /// uploaded.
    pub fn quantize(&mut self, src: &VulkanStorage) -> Result<()> {
        let f32_data = match src.to_cpu_storage()? {
            CpuStorage::F32(b) => b,
            _ => crate::bail!("vulkan quantize: expected an f32 source"),
        };
        let elem_count = f32_data.len();
        let mut cpu_q = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        cpu_q.quantize(&crate::Storage::Cpu(CpuStorage::F32(f32_data)))?;
        let bytes = cpu_q.data()?.into_owned();
        let bytes = self
            .device
            .storage_from_cpu_storage(&CpuStorage::U8(bytes))?;
        self.set_bytes(bytes);
        Ok(())
    }

    /// Quantizes the f32 values in `src` (on the Vulkan device) into this
    /// storage using imatrix weights.
    ///
    /// Quantization only happens on the CPU for now: the f32 source is read
    /// back, quantized with the CPU path, and the quantized bytes are
    /// uploaded.
    pub fn quantize_imatrix(
        &mut self,
        src: &VulkanStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let f32_data = match src.to_cpu_storage()? {
            CpuStorage::F32(b) => b,
            _ => crate::bail!("vulkan quantize_imatrix: expected an f32 source"),
        };
        let elem_count = f32_data.len();
        let mut cpu_q = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        cpu_q.quantize_imatrix(
            &crate::Storage::Cpu(CpuStorage::F32(f32_data)),
            imatrix_weights,
            n_per_row,
        )?;
        let bytes = cpu_q.data()?.into_owned();
        let bytes = self
            .device
            .storage_from_cpu_storage(&CpuStorage::U8(bytes))?;
        self.set_bytes(bytes);
        Ok(())
    }

    /// Quantizes the f32 values in `src` (on the CPU) into this storage
    /// using imatrix weights.
    ///
    /// Quantization only happens on the CPU for now.
    pub fn quantize_imatrix_onto(
        &mut self,
        src: &CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let elem_count = src.as_slice::<f32>()?.len();
        let mut cpu_q = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut cpu_q {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let bytes = cpu_q.data()?.into_owned();
        let bytes = self
            .device
            .storage_from_cpu_storage(&CpuStorage::U8(bytes))?;
        self.set_bytes(bytes);
        Ok(())
    }

    /// Quantizes the f32 values in `src` (on the CPU) into this storage.
    ///
    /// Quantization only happens on the CPU for now.
    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        let elem_count = src.as_slice::<f32>()?.len();
        let mut cpu_q = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut cpu_q {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let bytes = cpu_q.data()?.into_owned();
        let bytes = self
            .device
            .storage_from_cpu_storage(&CpuStorage::U8(bytes))?;
        self.set_bytes(bytes);
        Ok(())
    }

    /// Dequantizes `elem_count` elements to an f32 `VulkanStorage`.
    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        let block = self.dtype.block_size();
        if !elem_count.is_multiple_of(block) {
            crate::bail!(
                "dequantize: elem_count {elem_count} is not a multiple of block size {block}"
            );
        }
        self.dequantize_src(self.src_bytes(), elem_count)
    }

    /// Dequantizes `elem_count` elements from `src` (a byte-view of the
    /// quantized data holding exactly `elem_count / block` blocks) to an f32
    /// `VulkanStorage`. `elem_count` must be a multiple of the block size.
    fn dequantize_src(&self, src: Subbuffer<[u8]>, elem_count: usize) -> Result<VulkanStorage> {
        let num_blocks = elem_count / self.dtype.block_size();
        let (source, name) = dequantize_kernel(self.dtype);
        let out = VulkanStorage::new(&self.device, elem_count, DType::F32)?;
        let dst = match out.buffer() {
            crate::vulkan_backend::VulkanStorageBuffer::F32(b) => b.clone(),
            _ => crate::bail!("dequantize: out was created as F32"),
        };
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
            vec![num_blocks as u64],
        )
        .map_err(|e| crate::Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[u64]> = params_buf;
        let kernels = self.device.kernels();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_dequantize_slang(
                cbb, &kernels, source, name, &src, &dst, &params, num_blocks,
            )
            .map_err(|e| e.to_string())
        })?;
        // No pipeline sync: consumers either enqueue further work on the
        // same queue (ordered after this kernel) or read back, which syncs.
        Ok(out)
    }

    /// Gathers the requested embedding rows.
    ///
    /// A single id (decode) dequantizes only the requested row: each row is a
    /// contiguous range of quantized blocks, so the per-type dequantize kernel
    /// runs over a byte-view of that row (`hidden / block` blocks) instead of
    /// the whole `(rows, hidden)` matrix. Multiple ids (prefill) dequantize
    /// the whole matrix to f32 and gather along dim 0.
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
        let ids_cpu = ids_storage.to_cpu_storage()?;
        let ids_vals: Vec<u32> = match ids_cpu {
            CpuStorage::U32(v) => v,
            _ => crate::bail!("quantized embedding: ids must be u32"),
        };
        // Decode fast path: dequantize the single requested row.
        if ids_vals.len() == 1 {
            let id = ids_vals[0] as usize;
            if id >= rows {
                crate::bail!("quantized embedding: id {id} out of range [0, {rows})");
            }
            let block = self.dtype.block_size();
            let row_bytes = (hidden / block) * self.dtype.type_size();
            let start = id as u64 * row_bytes as u64;
            let src = self.src_bytes().slice(start..start + row_bytes as u64);
            return self.dequantize_src(src, hidden);
        }
        let num_ids = ids_vals.len();
        let data_f32 = self.dequantize(rows * hidden)?;
        let data_l = Layout::new((rows, hidden).into(), vec![hidden, 1], 0);
        // The Vulkan gather does per-element row selection (output shape = ids
        // shape), so build 2D ids `(num_ids, hidden)` with each row the id
        // repeated to gather full rows: `out[i][j] = data[ids[i]][j]`.
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
    /// f32 GEMM with the `(b, m, k)` lhs. Single-row inputs (decode) with
    /// Q4_0 or Q4_K weights use the fused dequant+GEMM kernel instead, which
    /// keeps the weights quantized in VRAM (no f32 materialization); larger
    /// inputs materialize the weights because the fused kernel dequantizes
    /// once per output row.
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
        let k2 = *dims.last().ok_or_else(|| {
            crate::Error::Vulkan("quantized matmul: empty dims".to_string().into())
        })?;
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {layout:?}");
        }
        let m_total = b * m;
        let orig = storage.dtype();
        let lhs = if orig == DType::F32 {
            storage.clone()
        } else {
            storage.to_dtype(layout, DType::F32)?
        };
        let lhs_offset = if orig == DType::F32 {
            layout.start_offset()
        } else {
            0
        };
        // Fused dequant+GEMM: single-row lhs (decode) uses the per-column
        // decode kernels, multi-row lhs (prefill) the per-element prefill
        // kernels, for the quantization formats they support.
        let fused = match (self.dtype, m_total == 1) {
            (GgmlDType::Q4_0, true) if k.is_multiple_of(32) => {
                Some((Source::FusedQ40Gemm, KernelName::FusedQ40GemmF32))
            }
            (GgmlDType::Q4_0, false) if k.is_multiple_of(32) => {
                Some((Source::FusedQ40GemmM, KernelName::FusedQ40GemmMF32))
            }
            (GgmlDType::Q4K, true) if k.is_multiple_of(256) => {
                Some((Source::FusedQ4kGemm, KernelName::FusedQ4kGemmF32))
            }
            (GgmlDType::Q4K, false) if k.is_multiple_of(256) => {
                Some((Source::FusedQ4kGemmM, KernelName::FusedQ4kGemmMF32))
            }
            (GgmlDType::Q5_0, true) if k.is_multiple_of(32) => {
                Some((Source::FusedQ50Gemm, KernelName::FusedQ50GemmF32))
            }
            (GgmlDType::Q5_1, true) if k.is_multiple_of(32) => {
                Some((Source::FusedQ51Gemm, KernelName::FusedQ51GemmF32))
            }
            (GgmlDType::Q8_0, true) if k.is_multiple_of(32) => {
                Some((Source::FusedQ80Gemm, KernelName::FusedQ80GemmF32))
            }
            (GgmlDType::Q8_1, true) if k.is_multiple_of(32) => {
                Some((Source::FusedQ81Gemm, KernelName::FusedQ81GemmF32))
            }
            (GgmlDType::Q5K, true) if k.is_multiple_of(256) => {
                Some((Source::FusedQ5kGemm, KernelName::FusedQ5kGemmF32))
            }
            (GgmlDType::Q8K, true) if k.is_multiple_of(256) => {
                Some((Source::FusedQ8kGemm, KernelName::FusedQ8kGemmF32))
            }
            _ => None,
        };
        let out = match fused {
            Some((source, name)) => {
                self.fused_quant_gemm(&lhs, lhs_offset, m_total, n, k, source, name)?
            }
            None => {
                let data_f32 = self.dequantize(n * k)?;
                // The weight is (n, k) contiguous; view it as the (k, n) rhs (transposed).
                let rhs_l = Layout::new((k, n).into(), vec![1, k], 0);
                let lhs_l = Layout::new((m_total, k).into(), vec![k, 1], lhs_offset);
                lhs.matmul(&data_f32, (1, m_total, n, k), &lhs_l, &rhs_l)?
            }
        };
        let out = if orig == DType::F32 {
            out
        } else {
            let out_l = Layout::new((m_total, n).into(), vec![n, 1], 0);
            out.to_dtype(&out_l, orig)?
        };
        let mut out_shape = dims;
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }

    /// Fused dequant+matmul: `out[m, n] = lhs[m, k] @ dequant(W[n, k])`
    /// with the weights dequantized in the kernel (no f32
    /// materialization). `m == 1` selects the decode kernels, `m > 1` the
    /// prefill kernels.
    #[allow(clippy::too_many_arguments)]
    fn fused_quant_gemm(
        &self,
        lhs: &VulkanStorage,
        lhs_offset: usize,
        m: usize,
        n: usize,
        k: usize,
        source: Source,
        name: KernelName,
    ) -> Result<VulkanStorage> {
        let lhs_buf = match lhs.buffer() {
            crate::vulkan_backend::VulkanStorageBuffer::F32(b) => b
                .clone()
                .slice(lhs_offset as u64..(lhs_offset + m * k) as u64),
            _ => crate::bail!("fused quant gemm: lhs must be f32"),
        };
        let out = VulkanStorage::new(&self.device, m * n, DType::F32)?;
        let dst = match out.buffer() {
            crate::vulkan_backend::VulkanStorageBuffer::F32(b) => b.clone(),
            _ => crate::bail!("fused quant gemm: out was created as F32"),
        };
        let src = self.src_bytes();
        // params = [m, n, k]
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
            vec![m as u64, n as u64, k as u64],
        )
        .map_err(|e| crate::Error::Vulkan(e.to_string().into()))?;
        let params: Subbuffer<[u64]> = params_buf;
        let kernels = self.device.kernels();
        self.device.execute(move |cbb| {
            candle_vulkan_kernels::call_fused_quant_gemm_slang(
                cbb, &kernels, source, name, &lhs_buf, &src, &dst, &params, m, n,
            )
            .map_err(|e| e.to_string())
        })?;
        Ok(out)
    }
}

/// Uploads raw quantized bytes to the Vulkan device.
pub fn load_quantized<T: GgmlType + Send + Sync + 'static>(
    device: &VulkanDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let dtype = T::DTYPE;
    let raw = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    let bytes = device.storage_from_cpu_storage(&CpuStorage::U8(raw.to_vec()))?;
    Ok(super::QStorage::Vulkan(Box::new(QVulkanStorage {
        bytes,
        device: device.clone(),
        dtype,
    })))
}
