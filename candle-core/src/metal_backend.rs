use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose2D};
use crate::error::Error;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use candle_metal_kernels;
use core::mem;
use half::{bf16, f16};
use metal;
use metal::mps::matrix::encode_gemm;
use metal::mps::Float32;
use metal::{Buffer, MTLResourceOptions, NSUInteger};
use std::sync::Arc;

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("metal error")]
    Metal,
}

#[derive(Clone)]
pub struct MetalDevice {
    device: metal::Device,
    _command_queue: metal::CommandQueue,
    command_buffer: metal::CommandBuffer,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.device.registry_id())
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    pub fn metal_device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }

    pub fn id(&self) -> u64 {
        self.registry_id()
    }

    fn new_buffer(&self, element_count: usize, dtype: DType) -> Buffer {
        let size = (element_count * dtype.size_in_bytes()) as u64;
        self.device.new_buffer(size, MTLResourceOptions::empty())
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: Arc<metal::Buffer>,
    device: MetalDevice,
    dtype: DType,
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::F32 => Ok(CpuStorage::F32(
                self.buffer.read_to_vec(self.buffer.length() as usize / 4),
            )),
            dtype => todo!("Unsupported dtype {dtype:?}"),
        }
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        println!("TODO Affine");
        Ok(self.clone())
        // todo!()
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        println!("TODO reduce_op");
        Ok(self.clone())
        // todo!()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        todo!("Implement {:?} {layout:?}  - {dtype:?}", self.dtype)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let dtype = self.dtype;
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let mut buffer = device.new_buffer(el_count, dtype);
        //todo!("Implement the kernel calling");
        // device.kernels.call_unary(U::KERNEL, &self.buffer, &mut buffer, el_count, dtype);
        Ok(Self {
            buffer: Arc::new(buffer),
            device,
            dtype,
        })
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        println!("TODO Binary {:?}", B::NAME);
        Ok(self.clone())
        // todo!()
    }

    fn where_cond(&self, _: &Layout, rhs: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        println!("TODO where_cond");
        Ok(rhs.clone())
        // todo!()
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        todo!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        todo!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        println!("TODO Index select");
        Ok(self.clone())
        // todo!()
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let transpose_left = false;
        let transpose_right = false;
        let alpha = 1.0;
        let beta = 0.0;
        self.matmul_generic(
            rhs,
            (b, m, n, k),
            lhs_l,
            rhs_l,
            transpose_left,
            transpose_right,
            alpha,
            beta,
        )
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        println!("TODO Copy strided");
        Ok(())
    }
}

impl MetalStorage {
    pub(crate) fn matmul_t(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let transpose_left = false;
        let transpose_right = true;
        let alpha = 1.0;
        let beta = 0.0;
        self.matmul_generic(
            rhs,
            (b, m, n, k),
            lhs_l,
            rhs_l,
            transpose_left,
            transpose_right,
            alpha,
            beta,
        )
    }
    pub(crate) fn matmul_generic(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
        transpose_left: bool,
        transpose_right: bool,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                let mut out_buffer = self.device.new_buffer(elem_count, self.dtype);
                if b != 1 {
                    println!("TODO implement batched matmul for B={b}");
                    // bail!("Didn't implemented strided matmul yet");
                    return Ok(Self {
                        buffer: Arc::new(out_buffer),
                        device: self.device.clone(),
                        dtype: self.dtype(),
                    });
                }
                if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
                    println!(
                        "Didn't implemented non contiguous matmul yet {:?} {:?}",
                        lhs_l.is_contiguous(),
                        rhs_l.is_contiguous()
                    );
                    return Ok(Self {
                        buffer: Arc::new(out_buffer),
                        device: self.device.clone(),
                        dtype: self.dtype(),
                    });
                }

                encode_gemm::<Float32, Float32, Float32>(
                    &self.device,
                    &self.device.command_buffer,
                    transpose_left,
                    transpose_right,
                    &self.buffer,
                    &rhs.buffer,
                    &mut out_buffer,
                    m as NSUInteger,
                    n as NSUInteger,
                    k as NSUInteger,
                    alpha,
                    beta,
                )
                .map_err(|e| Error::Metal(e))?;

                println!("lhs {:?} {m} {k}", self.buffer.length());
                println!("rhs {:?} {k} {n}", rhs.buffer.length());
                println!("out {:?} {m} {n}", out_buffer.length());
                println!("lhs {:?}", lhs_l.shape());

                Ok(Self {
                    buffer: Arc::new(out_buffer),
                    device: self.device.clone(),
                    dtype: self.dtype(),
                })
            }
            _ => todo!("Unimplemented matmul for this pair"),
        }
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = metal::Device::all().swap_remove(ordinal);
        let _command_queue = device.new_command_queue();
        let command_buffer = _command_queue.new_owned_command_buffer();
        Ok(Self {
            device,
            _command_queue,
            command_buffer,
        })
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        todo!("set_seed")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.device.registry_id() == rhs.device.registry_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        // TODO Is there a faster way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.zeros_impl(shape, dtype)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // TODO Is there a faster way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.ones_impl(shape, dtype)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let option = metal::MTLResourceOptions::CPUCacheModeDefaultCache;
        let buffer = match storage {
            CpuStorage::U8(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<u8>()) as u64,
                option,
            ),
            CpuStorage::U32(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<u32>()) as u64,
                option,
            ),
            CpuStorage::I64(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<i64>()) as u64,
                option,
            ),
            CpuStorage::BF16(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<bf16>()) as u64,
                option,
            ),
            CpuStorage::F16(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<f16>()) as u64,
                option,
            ),
            CpuStorage::F32(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<f32>()) as u64,
                option,
            ),
            CpuStorage::F64(storage) => self.device.new_buffer_with_data(
                storage.as_ptr() as *const core::ffi::c_void,
                (storage.len() * mem::size_of::<f64>()) as u64,
                option,
            ),
        };
        Ok(Self::Storage {
            buffer: Arc::new(buffer),
            device: self.clone(),
            dtype: storage.dtype(),
        })
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        // TODO is there a better way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, mean, stddev)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        // TODO is there a better way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, stddev)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }
}
