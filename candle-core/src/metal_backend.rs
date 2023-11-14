use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use candle_metal_kernels;
use candle_metal_kernels::Kernels;
use half::f16;
use metal;
use metal::{Buffer, CommandBuffer, CommandQueue, HeapDescriptor, MTLResourceOptions, NSUInteger};
use std::sync::{Arc, RwLock};

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    heap: metal::Heap,
    command_buffer: Arc<RwLock<metal::CommandBuffer>>,
    kernels: Arc<candle_metal_kernels::Kernels>,
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
    pub fn id(&self) -> NSUInteger {
        self.registry_id()
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn command_buffer(&self) -> std::sync::RwLockReadGuard<CommandBuffer> {
        self.command_buffer.read().unwrap()
    }

    pub fn commit_wait_until_completed(&self) {
        let mut old = self.command_buffer.try_write().unwrap();
        let status = old.status();
        use metal::MTLCommandBufferStatus::{
            Committed, Completed, Enqueued, Error, NotEnqueued, Scheduled,
        };
        // match old.status() {}
        if old.status() == metal::MTLCommandBufferStatus::Completed {
            return;
        }
        old.commit();
        old.wait_until_completed();
        // let count = old.retain_count();
        // println!("Count {count:?}");
        let command_buffer = self.command_queue.new_command_buffer().to_owned();

        *old = command_buffer;
        // let count = old.retain_count();
        // // println!("Count after {count:?}");
        // old.release();
        // let count = old.retain_count();
        // println!("Count after release {count:?}");
        // self.command_buffer.replace_with(|_| command_buffer)
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    pub fn new_buffer(&self, element_count: usize, dtype: DType) -> Buffer {
        let size = (element_count * dtype.size_in_bytes()) as NSUInteger;
        // println!("Creating buffer {size}");
        let buffer = self
            .heap
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
            .expect("New buffer");
        // println!("{:?}", self.heap.used_size());
        buffer
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Buffer {
        let size = core::mem::size_of_val(data) as NSUInteger;
        let option = metal::MTLResourceOptions::StorageModeShared;
        // println!("Creating data buffer {size}");
        self.device
            .new_buffer_with_data(data.as_ptr() as *const core::ffi::c_void, size, option)
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
        self.device.commit_wait_until_completed();

        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(
                self.buffer.read_to_vec(self.buffer.length() as usize),
            )),
            DType::U32 => Ok(CpuStorage::U32(
                self.buffer.read_to_vec(self.buffer.length() as usize / 4),
            )),
            DType::I64 => Ok(CpuStorage::I64(
                self.buffer.read_to_vec(self.buffer.length() as usize / 8),
            )),
            DType::F16 => Ok(CpuStorage::F16(
                self.buffer.read_to_vec(self.buffer.length() as usize / 2),
            )),
            DType::BF16 => Ok(CpuStorage::BF16(
                self.buffer.read_to_vec(self.buffer.length() as usize / 2),
            )),
            DType::F32 => Ok(CpuStorage::F32(
                self.buffer.read_to_vec(self.buffer.length() as usize / 4),
            )),
            DType::F64 => Ok(CpuStorage::F64(
                self.buffer.read_to_vec(self.buffer.length() as usize / 8),
            )),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let mut buffer = device.new_buffer(el, self.dtype);
        let command_buffer = self.device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "affine_float",
                DType::F16 => "affine_half",
                dtype => todo!("Affine {dtype:?}"),
            };
            candle_metal_kernels::call_affine(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &mut buffer,
                mul as f32,
                add as f32,
            )
            .unwrap();
        } else {
            let name = match self.dtype {
                DType::F32 => "affine_float_strided",
                DType::F16 => "affine_half_strided",
                dtype => todo!("Affine {dtype:?}"),
            };
            candle_metal_kernels::call_affine_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &mut buffer,
                mul as f32,
                add as f32,
            )
            .unwrap();
        }
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        assert!(sum_dims.len() == 1);
        assert!(sum_dims[0] == layout.shape().rank() - 1);
        assert!(layout.is_contiguous());
        assert!(layout.start_offset() == 0);
        let device = self.device.clone();
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !sum_dims.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in sum_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }

        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let (name, check_empty, return_index) = match (op, self.dtype) {
            (ReduceOp::Sum, DType::F32) => ("fast_sum_float", false, false),
            (ReduceOp::Min, DType::F32) => ("fast_min_float", true, false),
            (ReduceOp::Max, DType::F32) => ("fast_max_float", true, false),
            (ReduceOp::ArgMin, DType::F32) => ("fast_argmin_float", true, true),
            (ReduceOp::ArgMax, DType::F32) => ("fast_argmax_float", true, true),
            _ => todo!("Reduce op for non float"),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dtype = if return_index { DType::U32 } else { self.dtype };
        let mut buffer = device.new_buffer(dst_el, dtype);
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_reduce_contiguous(
            &device.device,
            &command_buffer,
            &device.kernels,
            name,
            src_el,
            dst_el,
            &self.buffer,
            &mut buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self {
            buffer,
            device,
            dtype,
        })
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let device = self.device();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let mut buffer = device.new_buffer(el_count, dtype);
        let command_buffer = device.command_buffer();
        if layout.is_contiguous() {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32",
                (DType::F32, DType::F16) => "cast_f32_f16",
                (DType::F16, DType::F32) => "cast_f16_f32",
                (left, right) => todo!("to dtype {left:?} - {right:?}"),
            };
            candle_metal_kernels::call_cast_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &mut buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32_strided",
                (DType::F32, DType::F16) => "cast_f32_f16_strided",
                (DType::F16, DType::F32) => "cast_f16_f32_strided",
                (left, right) => todo!("to dtype {left:?} - {right:?}"),
            };
            candle_metal_kernels::call_cast_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
                &mut buffer,
            )
            .map_err(MetalError::from)?;
        }

        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let mut buffer = device.new_buffer(el_count, dtype);
        {
            let command_buffer = device.command_buffer();
            if layout.is_contiguous() && layout.start_offset() == 0 {
                use candle_metal_kernels::unary::contiguous;

                let kernel_name = match (B::KERNEL, dtype) {
                    ("ucos", DType::F32) => contiguous::cos::FLOAT,
                    ("usin", DType::F32) => contiguous::sin::FLOAT,
                    ("usqr", DType::F32) => contiguous::sqr::FLOAT,
                    ("usqrt", DType::F32) => contiguous::sqrt::FLOAT,
                    ("uneg", DType::F32) => contiguous::neg::FLOAT,
                    ("uexp", DType::F32) => contiguous::exp::FLOAT,
                    ("ulog", DType::F32) => contiguous::log::FLOAT,
                    ("ugelu", DType::F32) => contiguous::gelu::FLOAT,
                    ("ugelu_erf", DType::F32) => contiguous::gelu_erf::FLOAT,
                    ("uerf", DType::F32) => contiguous::erf::FLOAT,
                    ("uceil", DType::F32) => contiguous::ceil::FLOAT,
                    ("ufloor", DType::F32) => contiguous::floor::FLOAT,
                    ("uround", DType::F32) => contiguous::round::FLOAT,
                    ("ucos", DType::F16) => contiguous::cos::HALF,
                    ("usin", DType::F16) => contiguous::sin::HALF,
                    ("usqr", DType::F16) => contiguous::sqr::HALF,
                    ("usqrt", DType::F16) => contiguous::sqrt::HALF,
                    ("uneg", DType::F16) => contiguous::neg::HALF,
                    ("uexp", DType::F16) => contiguous::exp::HALF,
                    ("ulog", DType::F16) => contiguous::log::HALF,
                    ("ugelu", DType::F16) => contiguous::gelu::HALF,
                    ("ugelu_erf", DType::F16) => contiguous::gelu_erf::HALF,
                    ("uerf", DType::F16) => contiguous::erf::HALF,
                    ("uceil", DType::F16) => contiguous::ceil::HALF,
                    ("ufloor", DType::F16) => contiguous::floor::HALF,
                    ("uround", DType::F16) => contiguous::round::HALF,
                    (name, dtype) => todo!("Match {name} - {dtype:?}"),
                };
                candle_metal_kernels::call_unary_contiguous(
                    &device.device,
                    &command_buffer,
                    &device.kernels,
                    kernel_name,
                    el_count,
                    &self.buffer,
                    &mut buffer,
                )
                .map_err(MetalError::from)?;
            } else {
                use candle_metal_kernels::unary::strided;
                let kernel_name = match (B::KERNEL, dtype) {
                    ("ucos", DType::F32) => strided::cos::FLOAT,
                    ("usin", DType::F32) => strided::sin::FLOAT,
                    ("usqr", DType::F32) => strided::sqr::FLOAT,
                    ("usqrt", DType::F32) => strided::sqrt::FLOAT,
                    ("uneg", DType::F32) => strided::neg::FLOAT,
                    ("uexp", DType::F32) => strided::exp::FLOAT,
                    ("ulog", DType::F32) => strided::log::FLOAT,
                    ("ugelu", DType::F32) => strided::gelu::FLOAT,
                    ("ugelu_erf", DType::F32) => strided::gelu_erf::FLOAT,
                    ("uerf", DType::F32) => strided::erf::FLOAT,
                    ("uceil", DType::F32) => strided::ceil::FLOAT,
                    ("ufloor", DType::F32) => strided::floor::FLOAT,
                    ("uround", DType::F32) => strided::round::FLOAT,
                    ("ucos", DType::F16) => strided::cos::HALF,
                    ("usin", DType::F16) => strided::sin::HALF,
                    ("usqr", DType::F16) => strided::sqr::HALF,
                    ("usqrt", DType::F16) => strided::sqrt::HALF,
                    ("uneg", DType::F16) => strided::neg::HALF,
                    ("uexp", DType::F16) => strided::exp::HALF,
                    ("ulog", DType::F16) => strided::log::HALF,
                    ("ugelu", DType::F16) => strided::gelu::HALF,
                    ("ugelu_erf", DType::F16) => strided::gelu_erf::HALF,
                    ("uerf", DType::F16) => strided::erf::HALF,
                    ("uceil", DType::F16) => strided::ceil::HALF,
                    ("ufloor", DType::F16) => strided::floor::HALF,
                    ("uround", DType::F16) => strided::round::HALF,
                    (name, dtype) => todo!("Match {name} - {dtype:?}"),
                };
                candle_metal_kernels::call_unary_strided(
                    &device.device,
                    &command_buffer,
                    &device.kernels,
                    kernel_name,
                    layout.dims(),
                    &self.buffer,
                    layout.stride(),
                    layout.start_offset() * self.dtype.size_in_bytes(),
                    &mut buffer,
                    0,
                )
                .map_err(MetalError::from)?;
            }
        }
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = lhs_l.shape();
        let el_count = shape.elem_count();
        let mut buffer = device.new_buffer(el_count, dtype);
        let command_buffer = device.command_buffer();
        if (lhs_l.is_contiguous() && lhs_l.start_offset() == 0)
            && (rhs_l.is_contiguous() && rhs_l.start_offset() == 0)
        {
            use candle_metal_kernels::binary::contiguous;

            let kernel_name = match (B::KERNEL, dtype) {
                ("add", DType::F32) => contiguous::add::FLOAT,
                ("badd", DType::F32) => contiguous::add::FLOAT,
                ("sub", DType::F32) => contiguous::sub::FLOAT,
                ("bsub", DType::F32) => contiguous::sub::FLOAT,
                ("mul", DType::F32) => contiguous::mul::FLOAT,
                ("bmul", DType::F32) => contiguous::mul::FLOAT,
                ("div", DType::F32) => contiguous::div::FLOAT,
                ("bdiv", DType::F32) => contiguous::div::FLOAT,
                ("add", DType::F16) => contiguous::add::HALF,
                ("badd", DType::F16) => contiguous::add::HALF,
                ("sub", DType::F16) => contiguous::sub::HALF,
                ("bsub", DType::F16) => contiguous::sub::HALF,
                ("mul", DType::F16) => contiguous::mul::HALF,
                ("bmul", DType::F16) => contiguous::mul::HALF,
                ("div", DType::F16) => contiguous::div::HALF,
                ("bdiv", DType::F16) => contiguous::div::HALF,
                (name, dtype) => todo!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_binary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &rhs.buffer,
                &mut buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::binary::strided;

            let kernel_name = match (B::KERNEL, dtype) {
                ("badd", DType::F32) => strided::add::FLOAT,
                ("bsub", DType::F32) => strided::sub::FLOAT,
                ("bmul", DType::F32) => strided::mul::FLOAT,
                ("bdiv", DType::F32) => strided::div::FLOAT,
                ("badd", DType::F16) => strided::add::HALF,
                ("bsub", DType::F16) => strided::sub::HALF,
                ("bmul", DType::F16) => strided::mul::HALF,
                ("bdiv", DType::F16) => strided::div::HALF,
                (name, dtype) => todo!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_binary_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                lhs_l.dims(),
                &self.buffer,
                lhs_l.stride(),
                lhs_l.start_offset() * self.dtype.size_in_bytes(),
                &rhs.buffer,
                rhs_l.stride(),
                rhs_l.start_offset() * rhs.dtype.size_in_bytes(),
                &mut buffer,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device.clone();
        let shape = t_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dtype = t.dtype;
        let mut buffer = self.device.new_buffer(el, dtype);
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_where_cond_strided(
            &device.device,
            &command_buffer,
            &device.kernels,
            "where_u8_f32",
            dims,
            &self.buffer,
            (
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
            ),
            &t.buffer,
            (&t_l.stride(), t_l.start_offset() * t.dtype.size_in_bytes()),
            &f.buffer,
            (&f_l.stride(), f_l.start_offset() * f.dtype.size_in_bytes()),
            &mut buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self {
            buffer,
            device,
            dtype,
        })
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

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose1D,
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

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        assert!(src_l.is_contiguous());
        assert!(src_l.start_offset() == 0);
        assert!(ids_l.is_contiguous());
        assert!(ids_l.start_offset() == 0);
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;
        let dtype = self.dtype;
        let device = self.device();
        let mut buffer = device.new_buffer(dst_el, dtype);
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "is_u32_f32",
            (DType::U32, DType::F16) => "is_u32_f16",
            (left, right) => todo!("index select metal {left:?} {right:?}"),
        };
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_index_select(
            &device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            src_l.dims(),
            ids_el,
            dim,
            &self.buffer,
            &ids.buffer,
            &mut buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
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
        // Create descriptors
        use metal::mps::matrix::*;

        let (type_id, size) = match self.dtype {
            DType::F32 => (
                metal::mps::MPS_FLOATBIT_ENCODING | 32,
                core::mem::size_of::<f32>() as NSUInteger,
            ),
            DType::F16 => (
                metal::mps::MPS_FLOATBIT_ENCODING | 16,
                core::mem::size_of::<f16>() as NSUInteger,
            ),
            dtype => todo!("Dtype for matmul {dtype:?} is not supported"),
        };

        let elem_count = b * m * n;

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
        // The a tensor has dims batching, k, n (rhs)
        let transpose_left = if lhs_m1 == 1 && lhs_m2 == k {
            false
        } else if lhs_m1 == m && lhs_m2 == 1 {
            true
        } else {
            Err(MetalError::MatMulNonContiguous {
                lhs_stride: lhs_stride.to_vec(),
                rhs_stride: rhs_stride.to_vec(),
                mnk: (m, n, k),
            })?
        };
        let transpose_right = if rhs_m1 == 1 && rhs_m2 == n {
            false
        } else if rhs_m1 == k && rhs_m2 == 1 {
            true
        } else {
            Err(MetalError::MatMulNonContiguous {
                lhs_stride: lhs_stride.to_vec(),
                rhs_stride: rhs_stride.to_vec(),
                mnk: (m, n, k),
            })?
        };
        let stride_left: u64 = match lhs_stride[..lhs_stride.len() - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(MetalError::MatMulNonContiguous {
                lhs_stride: lhs_stride.to_vec(),
                rhs_stride: rhs_stride.to_vec(),
                mnk: (m, n, k),
            })?,
        } as u64;
        let stride_right: u64 = match rhs_stride[..rhs_stride.len() - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(MetalError::MatMulNonContiguous {
                lhs_stride: lhs_stride.to_vec(),
                rhs_stride: rhs_stride.to_vec(),
                mnk: (m, n, k),
            })?,
        } as u64;

        let b = b as NSUInteger;
        let m = m as NSUInteger;
        let n = n as NSUInteger;
        let k = k as NSUInteger;

        let left_descriptor = if transpose_left {
            MatrixDescriptor::init_single(k, m, m * size, type_id)
        } else {
            MatrixDescriptor::init_single(m, k, k * size, type_id)
        };
        let right_descriptor = if transpose_right {
            MatrixDescriptor::init_single(n, k, k * size, type_id)
        } else {
            MatrixDescriptor::init_single(k, n, n * size, type_id)
        };
        let result_descriptor = MatrixDescriptor::init_single(m, n, n * size, type_id);

        let out_buffer = self.device.new_buffer(elem_count, self.dtype);

        {
            let command_buffer = self.device.command_buffer();
            for bi in 0..b {
                // Create matrix objects
                let left_matrix = Matrix::init_with_buffer_descriptor(
                    &self.buffer,
                    (bi * stride_left + lhs_l.start_offset() as u64) * size,
                    &left_descriptor,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })?;
                let right_matrix = Matrix::init_with_buffer_descriptor(
                    &rhs.buffer,
                    (bi * stride_right + rhs_l.start_offset() as u64) * size,
                    &right_descriptor,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })?;

                let result_matrix = Matrix::init_with_buffer_descriptor(
                    &out_buffer,
                    bi * m * n * size,
                    &result_descriptor,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })?;

                let alpha = 1.0f64;
                let beta = 0.0f64;
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
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })?;

                // Encode kernel to command buffer
                matrix_multiplication.encode_to_command_buffer(
                    &command_buffer,
                    &left_matrix,
                    &right_matrix,
                    &result_matrix,
                );
            }
        }

        Ok(Self {
            buffer: out_buffer,
            device: self.device.clone(),
            dtype: self.dtype(),
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        let command_buffer = self.device.command_buffer();
        let kernel_name = match self.dtype {
            DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
            DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
            DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
            DType::U32 => candle_metal_kernels::unary::strided::copy::U32,
            dtype => todo!("copy_strided not implemented for {dtype:?}"),
        };
        candle_metal_kernels::call_unary_strided(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            kernel_name,
            src_l.dims(),
            &self.buffer,
            src_l.stride(),
            src_l.start_offset() * self.dtype.size_in_bytes(),
            &mut dst.buffer,
            dst_offset * dst.dtype.size_in_bytes(),
        )
        .map_err(MetalError::from)?;
        Ok(())
    }
}

impl MetalStorage {
    pub fn new(buffer: Buffer, device: MetalDevice, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = metal::Device::all().swap_remove(ordinal);

        let command_queue = device.new_command_queue();

        let descriptor = HeapDescriptor::new();
        let mut size =
            device.heap_buffer_size_and_align(100_000_000, MTLResourceOptions::StorageModeShared);
        size.size += (size.size & (size.align - 1)) + size.align;
        descriptor.set_size(size.size);
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let heap = device.new_heap(&descriptor);
        let command_buffer = Arc::new(RwLock::new(command_queue.new_command_buffer().to_owned()));
        let kernels = Arc::new(Kernels::new());
        Ok(Self {
            device,
            heap,
            command_queue,
            command_buffer,
            kernels,
        })
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        todo!("set_seed")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal {
            gpu_id: self.registry_id() as usize,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.device.registry_id() == rhs.device.registry_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let buffer = self.new_buffer(shape.elem_count(), dtype);
        Ok(MetalStorage {
            buffer,
            device: self.clone(),
            dtype,
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // TODO Is there a faster way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.ones_impl(shape, dtype)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let buffer = match storage {
            CpuStorage::U8(storage) => self.new_buffer_with_data(storage),
            CpuStorage::U32(storage) => self.new_buffer_with_data(storage),
            CpuStorage::I64(storage) => self.new_buffer_with_data(storage),
            CpuStorage::BF16(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F16(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F32(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F64(storage) => self.new_buffer_with_data(storage),
        };
        Ok(Self::Storage {
            buffer,
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
