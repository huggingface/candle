use super::{GgmlDType, QStorage};
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{CudaDevice, CudaStorage, Result};

use cudarc::driver::{CudaSlice, DeviceSlice};

pub struct QCudaStorage {
    data: CudaSlice<u8>,
    dtype: GgmlDType,
    device: CudaDevice,
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = el_count * dtype.type_size() / dtype.block_size();
        let data = device.alloc_zeros::<u8>(size_in_bytes).w()?;
        Ok(QCudaStorage {
            data,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        // Run the dequantization on cpu.
        use crate::quantized::k_quants::GgmlType;

        let buffer = self.device.dtoh_sync_copy(&self.data).w()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let slice =
                    unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const f32, block_len) };
                out.copy_from_slice(slice)
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out)?;
            }
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => {
                self.device.dtoh_sync_copy(data).w()?
            }
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let data = self.device.htod_sync_copy(data.as_ref()).w()?;
        self.data = data;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use cudarc::driver::LaunchAsync;

        if self.dtype != GgmlDType::Q4_0 {
            crate::bail!(
                "cuda quantized fwd is not implemented for {:?} ({:?} {:?})",
                self.dtype,
                self_shape,
                layout
            )
        }
        let (n, k) = self_shape.dims2()?;
        let (dst_shape, m) = match layout.shape().dims() {
            &[b, m, k2] => {
                if k2 != k {
                    crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
                }
                (vec![b, m, n], b * m)
            }
            &[m, k2] => {
                if k2 != k {
                    crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
                }
                (vec![m, n], m)
            }
            s => crate::bail!("unexpected shape for input {s:?}"),
        };

        let dev = storage.device().clone();
        let slice = storage.as_cuda_slice::<f32>()?;
        let slice = match layout.contiguous_offsets() {
            None => crate::bail!("input has to be contiguous"),
            Some((o1, o2)) => slice.slice(o1..o2),
        };
        let func = dev.get_or_load_func("mul_mat_q4_0_check", candle_kernels::QUANTIZED)?;
        let elem_count = m * n;
        let dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
        let (nrows_x, ncols_x) = (n, k);
        let (nrows_y, ncols_y) = (m, k);
        let params = (
            &self.data, &slice, &dst, /* ncols_x */ ncols_x, /* nrows_x */ nrows_x,
            /* ncols_y */ ncols_y, /* nrows_y */ nrows_y, /* nrows_dst */ m,
        );
        let block_num_x = (nrows_x + MMQ_Y_Q4_0_AMPERE - 1) / MMQ_Y_Q4_0_AMPERE;
        let block_num_y = (ncols_y + MMQ_X_Q4_0_AMPERE - 1) / MMQ_X_Q4_0_AMPERE;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (block_num_x as u32, block_num_y as u32, 1),
            block_dim: (WARP_SIZE as u32, NWARPS_Q4_0_AMPERE as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, params) }.w()?;

        let dst = CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, dst_shape.into()))
    }
}

fn read_to_vec<T: Clone>(buffer: &[u8], n: usize) -> Vec<T> {
    let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
    slice.to_vec()
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let data = device.htod_sync_copy(data).w()?;
    Ok(QStorage::Cuda(QCudaStorage {
        data,
        device: device.clone(),
        dtype: T::DTYPE,
    }))
}
