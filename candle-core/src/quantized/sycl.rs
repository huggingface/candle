//! GGUF quantized storage for the SYCL backend.
//!
//! Phase 4 + 4b: `load_quantized`, GPU `dequantize` for all 12 GGUF quant types,
//! `quantize`/`quantize_onto` (candle's CPU quantizer + upload), `embedding`.
//! `QMatMul::fwd` dispatches: **fused MMVQ** (`m <= 8`, weight stays quantized —
//! the decode path) or **dequantize + oneMKL GEMM** (prefill / fallback). A DP4a
//! / Q8_1 integer MMQ path is the remaining perf work.
#![allow(unused)]
use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::sycl_backend::{k, storage_from_buffer, SyclError};
use crate::{DType, Layout, Result, Shape, SyclDevice, SyclStorage};

fn nyi<T>(what: &str) -> Result<T> {
    Err(crate::Error::Sycl(
        SyclError::msg(format!("quantized/sycl: {what} not implemented yet")).into(),
    ))
}

fn to_k_dtype(dt: GgmlDType) -> k::GgmlDType {
    use k::GgmlDType as K;
    match dt {
        GgmlDType::F32 => K::F32,
        GgmlDType::F16 => K::F16,
        GgmlDType::BF16 => K::BF16,
        GgmlDType::Q4_0 => K::Q4_0,
        GgmlDType::Q4_1 => K::Q4_1,
        GgmlDType::Q5_0 => K::Q5_0,
        GgmlDType::Q5_1 => K::Q5_1,
        GgmlDType::Q8_0 => K::Q8_0,
        GgmlDType::Q8_1 => K::Q8_1,
        GgmlDType::Q2K => K::Q2K,
        GgmlDType::Q3K => K::Q3K,
        GgmlDType::Q4K => K::Q4K,
        GgmlDType::Q5K => K::Q5K,
        GgmlDType::Q6K => K::Q6K,
        GgmlDType::Q8K => K::Q8K,
    }
}

pub struct QSyclStorage {
    data: k::DeviceBuffer,
    dtype: GgmlDType,
    elem_count: usize,
    device: SyclDevice,
}

impl QSyclStorage {
    fn from_cpu_quant(
        device: &SyclDevice,
        dtype: GgmlDType,
        cpu: &dyn super::QuantizedType,
        elem_count: usize,
    ) -> Result<Self> {
        let bytes = cpu.storage_size_in_bytes();
        let host = unsafe { std::slice::from_raw_parts(cpu.as_ptr(), bytes) };
        let data = device.alloc_bytes(bytes)?;
        data.copy_from_host(host)
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(Self {
            data,
            dtype,
            elem_count,
            device: device.clone(),
        })
    }
}

impl QSyclStorage {
    pub fn zeros(device: &SyclDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let n_blocks = elem_count / dtype.block_size();
        let bytes = n_blocks * dtype.type_size();
        let data = device.alloc_bytes(bytes)?;
        data.memset_zero()
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(Self {
            data,
            dtype,
            elem_count,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &SyclDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len_bytes()
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len_bytes()];
        self.data
            .copy_to_host(&mut out)
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.as_ptr() as *const u8)
    }

    /// Dequantize `elem_count` values into a fresh f32 `SyclStorage`.
    pub fn dequantize(&self, elem_count: usize) -> Result<SyclStorage> {
        let out = self.device.new_storage(DType::F32, elem_count)?;
        let w = |e: k::SyclError| crate::Error::Sycl(SyclError::msg(e.to_string()).into());
        match self.dtype {
            GgmlDType::F32 => {
                out.buf()
                    .copy_from_device(&self.data, elem_count * 4)
                    .map_err(w)?;
            }
            GgmlDType::F16 | GgmlDType::BF16 => {
                // reinterpret the raw halves as a SyclStorage, then cast to f32.
                let half = storage_from_buffer(
                    &self.device,
                    self.raw_clone()?,
                    if self.dtype == GgmlDType::F16 {
                        DType::F16
                    } else {
                        DType::BF16
                    },
                    elem_count,
                );
                let f32s = half.to_dtype_raw(&Layout::contiguous(elem_count), DType::F32)?;
                out.buf()
                    .copy_from_device(f32s.buf(), elem_count * 4)
                    .map_err(w)?;
            }
            other => {
                let n_blocks = elem_count / other.block_size();
                k::dequantize(
                    self.device.q(),
                    to_k_dtype(other),
                    &self.data,
                    out.buf(),
                    n_blocks,
                )
                .map_err(w)?;
            }
        }
        Ok(out)
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<SyclStorage> {
        let f32s = self.dequantize(elem_count)?;
        f32s.to_dtype_raw(&Layout::contiguous(elem_count), DType::F16)
    }

    fn raw_clone(&self) -> Result<k::DeviceBuffer> {
        let b = self.device.alloc_bytes(self.data.len_bytes())?;
        b.copy_from_device(&self.data, self.data.len_bytes())
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(b)
    }

    /// `QMatMul` forward. `self_shape = (n, k)` (weight is stored transposed);
    /// `storage`/`layout` is the activation with shape `(.., k)`, result `(.., n)`.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &SyclStorage,
        layout: &Layout,
    ) -> Result<(SyclStorage, Shape)> {
        let (n, kk) = self_shape.dims2()?;
        let src_dims = layout.shape().dims().to_vec();
        let m: usize = src_dims[..src_dims.len() - 1].iter().product();
        if src_dims[src_dims.len() - 1] != kk {
            crate::bail!("qmatmul: input {layout:?} incompatible with weight {self_shape:?}");
        }
        let mut out_dims = src_dims.clone();
        *out_dims.last_mut().unwrap() = n;

        // Need a dense f32 (m, k) activation. Skip the cast entirely when the
        // input is already f32, contiguous and zero-offset (the common case).
        let act_owned;
        let act: &SyclStorage = if storage.dtype() == DType::F32
            && layout.start_offset() == 0
            && layout.is_contiguous()
        {
            storage
        } else {
            act_owned = storage.to_dtype_raw(layout, DType::F32)?;
            &act_owned
        };
        let werr = |e: k::SyclError| crate::Error::Sycl(SyclError::msg(e.to_string()).into());
        let blk = self.dtype.block_size();
        let out_f32 = if m <= 8 && blk != 1 && kk % blk == 0 {
            // Decode path: fused mat-vec, the weight stays quantized in memory.
            let out = self.device.new_storage(DType::F32, m * n)?;
            k::mmvq(
                self.device.q(),
                to_k_dtype(self.dtype),
                &self.data,
                act.buf(),
                out.buf(),
                n,
                kk,
                m,
            )
            .map_err(werr)?;
            out
        } else {
            // Prefill / fallback: dequantize the whole weight, one oneMKL GEMM.
            // w is stored (n, k); pass a transposed view so `mm_operand` sees `transb`.
            let w = self.dequantize(n * kk)?;
            let w_t = Layout::new((kk, n).into(), vec![1, kk], 0);
            act.matmul_raw(&w, (1, m, n, kk), &Layout::contiguous((m, kk)), &w_t)?
        };
        let out = if storage.dtype() == DType::F32 {
            out_f32
        } else {
            out_f32.to_dtype_raw(&Layout::contiguous((m, n)), storage.dtype())?
        };
        Ok((out, Shape::from(out_dims)))
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &SyclStorage,
        ids_l: &Layout,
    ) -> Result<SyclStorage> {
        let w = self.dequantize(rows * hidden)?;
        w.index_select_raw(ids, &Layout::contiguous((rows, hidden)), ids_l, 0)
    }

    pub fn quantize(&mut self, src: &SyclStorage) -> Result<()> {
        // No on-device quantizer yet: pull to host, use candle's CPU quantizer,
        // upload the blocks. Correct, and quantization is a one-off at load time.
        let cpu = src.to_cpu_storage()?;
        self.quantize_onto(&cpu)
    }
    pub fn quantize_imatrix(&mut self, _src: &SyclStorage, _w: &[f32], _n: usize) -> Result<()> {
        nyi("quantize_imatrix")
    }
    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _w: &[f32],
        _n: usize,
    ) -> Result<()> {
        nyi("quantize_imatrix_onto")
    }
    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        let f32s = src.as_slice::<f32>()?;
        let mut cpu = self.dtype.cpu_zeros(f32s.len());
        cpu.from_float(f32s);
        *self = QSyclStorage::from_cpu_quant(&self.device, self.dtype, cpu.as_ref(), f32s.len())?;
        Ok(())
    }

    pub fn fwd_via_dequant(&self) -> Result<()> {
        Ok(())
    }

    pub fn indexed_moe_forward(
        &self,
        _: &Shape,
        _: &SyclStorage,
        _: &Layout,
        _: &SyclStorage,
        _: &Layout,
    ) -> Result<(SyclStorage, Shape)> {
        nyi("indexed_moe_forward")
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &SyclDevice,
    data: &[T],
) -> Result<QStorage> {
    let bytes = std::mem::size_of_val(data);
    let host = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
    let buf = device.alloc_bytes(bytes)?;
    buf.copy_from_host(host)
        .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
    Ok(QStorage::Sycl(QSyclStorage {
        data: buf,
        dtype: T::DTYPE,
        elem_count: data.len() * T::DTYPE.block_size(),
        device: device.clone(),
    }))
}
