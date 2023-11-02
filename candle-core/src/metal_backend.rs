use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
pub use candle_metal_kernels;
use core::mem;
use half::{bf16, f16};
use metal;
use metal::mps::matrix::{MatrixMultiplication, Matrix, MatrixDescriptor};
use metal::mps::{Float32, MPSDataType};
use metal::MTLResourceOptions;
use crate::bail;

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
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: metal::Buffer,
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
        todo!()
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        todo!()
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        todo!()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        todo!("Implement {:?} {layout:?}  - {dtype:?}", self.dtype)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        todo!()
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
        todo!()
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
        let elem_count = b * m * n;
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                if b != 1 {
                    bail!("Didn't implemented strided matmul yet");
                }
                if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
                    bail!("Didn't implemented non contiguous matmul yet");
                }
                let out_buffer = self.device.new_buffer(
                    (elem_count * mem::size_of::<f32>()) as u64,
                    MTLResourceOptions::empty(),
                );
                let m : u64 = m.try_into().expect("usize should fit u64");
                let n: u64 = n.try_into().expect("usize should fit u64");
                let k: u64 = k.try_into().expect("usize should fit u64");
                // Create descriptors
                let left_descriptor =
                    MatrixDescriptor::init_single(m, k, k * Float32::SIZE, Float32::TYPE_ID);
                let right_descriptor =
                    MatrixDescriptor::init_single(k, n, n * Float32::SIZE, Float32::TYPE_ID);
                let result_descriptor =
                    MatrixDescriptor::init_single(m, n, n * Float32::SIZE, Float32::TYPE_ID);

                // Create matrix objects
                let left_matrix =
                    Matrix::init_with_buffer_descriptor(&self.buffer, &left_descriptor)
                        .expect("Failed to create left matrix");
                let right_matrix =
                    Matrix::init_with_buffer_descriptor(&rhs.buffer, &right_descriptor)
                        .expect("Failed to create left matrix");

                let result_matrix =
                    Matrix::init_with_buffer_descriptor(&out_buffer, &result_descriptor)
                        .expect("Failed to create left matrix");
                
                let transpose_left = false;
                let transpose_right = false;
                let alpha = 1.0;
                let beta = 0.0;


                // Create kernel
                let matrix_multiplication = MatrixMultiplication::init(
                    &self.device,
                    transpose_left,
                    transpose_right,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                )
                .expect("Failed to create matrix multiplication kernel");

                // Encode kernel to command buffer
                matrix_multiplication.encode_to_command_buffer(
                    &self.device.command_buffer,
                    &left_matrix,
                    &right_matrix,
                    &result_matrix,
                );
        Ok(Self{
            buffer: out_buffer,
            device: self.device.clone(),
            dtype: self.dtype(),
        })

            }
            _ => todo!("Unimplemented matmul for this pair"),
        }
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        todo!()
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = metal::Device::all().swap_remove(ordinal);
        let _command_queue = device.new_command_queue();
        let command_buffer = _command_queue.new_owned_command_buffer();
        Ok(Self { device, _command_queue, command_buffer })
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

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<MetalStorage> {
        todo!()
    }

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
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
            buffer,
            device: self.clone(),
            dtype: storage.dtype(),
        })
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }
}
