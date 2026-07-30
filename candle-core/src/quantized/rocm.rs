//! Quantized (GGUF/GGML) storage backed by a ROCm device.
//!
//! Modelled on [`crate::quantized::cuda`]: the ROCm backend compiles the same
//! `candle-kernels/src/quantized.cu` sources, so the on-device layout — padded
//! rows above all — has to match byte for byte.
//!
//! Matrix multiplication currently dequantizes the weights and hands the result
//! to the regular rocBLAS GEMM. That is slower than the DMMV/MMVQ/MMQ kernels
//! the CUDA backend picks between, but it is correct for every dtype and needs
//! no quantized matmul kernel of its own.

use super::{GgmlDType, QStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::quantized::k_quants::GgmlType;
use crate::rocm_backend::{RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory};
use crate::{CpuStorage, DType, Layout, Result, Shape};

mod kernels;

use kernels::MATRIX_ROW_PADDING;

#[cfg(test)]
mod tests;

pub struct QRocmStorage {
    /// Quantized payload followed by `MATRIX_ROW_PADDING` elements worth of
    /// zeroed blocks. See [`padded_len`].
    data: SendSyncDeviceMemory<u8>,
    /// Length of the *logical* payload in bytes; `data` is longer.
    len: usize,
    dtype: GgmlDType,
    device: RocmDevice,
}

/// Allocation size for a payload of `len` bytes.
///
/// The quantized kernels read whole padded rows, so every buffer carries
/// `MATRIX_ROW_PADDING` extra elements. Mirrors `quantized/cuda.rs`.
fn padded_len(len: usize, dtype: GgmlDType) -> usize {
    len + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size()
}

/// Upload `data` into a freshly allocated, zero-padded device buffer.
fn upload(device: &RocmDevice, data: &[u8], dtype: GgmlDType) -> Result<SendSyncDeviceMemory<u8>> {
    let mut inner = device.alloc_zeros::<u8>(padded_len(data.len(), dtype))?;
    // `copy_from_host` clamps to the shorter of the two, so the padding stays
    // zeroed.
    inner
        .copy_from_host(data)
        .map_err(|e| crate::Error::Msg(format!("failed to upload quantized data: {e}")))?;
    Ok(inner)
}

/// Dequantize `n` blocks of `buffer` on the CPU.
///
/// `buffer` comes back from the device as a plain `Vec<u8>` with no alignment
/// guarantee beyond a byte, hence the unaligned reads.
fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
    let size = std::mem::size_of::<T>();
    if buffer.len() < n * size {
        crate::bail!(
            "quantized buffer holds {} bytes, expected at least {}",
            buffer.len(),
            n * size
        )
    }
    let mut blocks = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: the bounds check above covers `n * size` bytes and
        // `read_unaligned` accepts any alignment. Every ggml block type is
        // plain old data, so the copy is a bit-for-bit move.
        blocks.push(unsafe { (buffer.as_ptr().add(i * size) as *const T).read_unaligned() });
    }
    T::to_float(&blocks, dst);
    Ok(())
}

impl QRocmStorage {
    pub fn zeros(device: &RocmDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let len = el_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let padded =
            (el_count + MATRIX_ROW_PADDING).div_ceil(dtype.block_size()) * dtype.type_size();
        Ok(Self {
            data: device.alloc_zeros::<u8>(padded)?,
            len,
            dtype,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.len
    }

    /// The quantized payload, without the trailing padding.
    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.len];
        // `copy_to_host` clamps to the shorter of the two buffers.
        self.data
            .copy_to_host(&mut out)
            .map_err(|e| crate::Error::Msg(format!("failed to read quantized data: {e}")))?;
        Ok(out)
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<RocmStorage> {
        if kernels::has_dequantize_kernel(self.dtype) {
            let dst = kernels::dequantize_f32(&self.data, self.dtype, elem_count, &self.device)?;
            return Ok(RocmStorage {
                slice: RocmStorageSlice::F32(dst),
                device: self.device.clone(),
            });
        }
        // No kernel for this dtype — round-trip through the host.
        let buffer = self.data()?;
        let mut out = vec![0f32; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out)?,
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out)?,
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out)?,
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out)?,
            dtype => crate::bail!("unsupported dtype for ROCm dequantize {dtype:?}"),
        }
        self.device.storage_from_cpu_storage(&CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<RocmStorage> {
        if !kernels::has_dequantize_kernel(self.dtype) {
            let f32 = self.dequantize(elem_count)?;
            return f32.to_dtype(&Layout::contiguous(elem_count), DType::F16);
        }
        let dst = kernels::dequantize_f16(&self.data, self.dtype, elem_count, &self.device)?;
        Ok(RocmStorage {
            slice: RocmStorageSlice::F16(dst),
            device: self.device.clone(),
        })
    }

    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()> {
        let src = match &src.slice {
            RocmStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            slice => crate::bail!("only f32 can be quantized, got {:?}", slice.dtype()),
        };
        self.quantize_from_f32(&src)
    }

    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        self.quantize_from_f32(src.as_slice::<f32>()?)
    }

    /// Quantization itself runs on the CPU; only the upload is device side.
    fn quantize_from_f32(&mut self, src: &[f32]) -> Result<()> {
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src.len(), self.dtype)?;
        match &mut qcpu_storage {
            QStorage::Cpu(storage) => storage.from_float(src),
            _ => crate::bail!("Device::Cpu.qzeros did not return a cpu storage"),
        }
        let data = qcpu_storage.data()?;
        self.data = upload(&self.device, &data, self.dtype)?;
        self.len = data.len();
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &RocmStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        crate::bail!("imatrix quantization is not implemented for the ROCm backend")
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        crate::bail!("imatrix quantization is not implemented for the ROCm backend")
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &RocmStorage,
        ids_l: &Layout,
    ) -> Result<RocmStorage> {
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.len != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.len
            )
        }
        let ids_mem = match &ids.slice {
            RocmStorageSlice::U32(mem) => mem,
            slice => crate::bail!(
                "quantized embedding expects u32 ids, got {:?}",
                slice.dtype()
            ),
        };
        let (o1, o2) = match ids_l.contiguous_offsets() {
            Some(offsets) => offsets,
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-embedding",
            }
            .bt())?,
        };
        let dst = kernels::get_rows(
            &self.data,
            self.dtype,
            hidden,
            ids_mem,
            o1,
            o2 - o1,
            &self.device,
        )?;
        Ok(RocmStorage {
            slice: RocmStorageSlice::F32(dst),
            device: self.device.clone(),
        })
    }

    /// `self` is the transposed weight matrix of shape `(n, k)`.
    ///
    /// Dequantize, then run the regular GEMM. The dequantized weights are laid
    /// out as `n` rows of `k`, so the `(k, n)` operand the GEMM wants is that
    /// same buffer viewed with swapped strides.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for quantized matmul input {s:?}"),
        };
        if k2 != k {
            crate::bail!(
                "mismatch on quantized matmul dim {self_shape:?} {:?}",
                layout.shape()
            )
        }
        let weights = self.dequantize(n * k)?;
        let weights = match storage.dtype() {
            DType::F32 => weights,
            dtype @ (DType::F16 | DType::BF16 | DType::F64) => {
                weights.to_dtype(&Layout::contiguous((n, k)), dtype)?
            }
            dtype => crate::bail!("quantized matmul expects a float input, got {dtype:?}"),
        };
        let rhs_l = Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
        let out = storage.matmul(&weights, (b, m, n, k), layout, &rhs_l)?;
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }

    pub fn indexed_moe_forward(
        &self,
        _self_shape: &Shape,
        _input: &RocmStorage,
        _input_l: &Layout,
        _ids: &RocmStorage,
        _ids_l: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        crate::bail!("indexed_moe_forward is not implemented for the ROCm backend")
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        crate::bail!("device_ptr is not implemented for the ROCm backend")
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &RocmDevice,
    data: &[T],
) -> Result<QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    Ok(QStorage::Rocm(QRocmStorage {
        data: upload(device, data, dtype)?,
        len: data.len(),
        dtype,
        device: device.clone(),
    }))
}
