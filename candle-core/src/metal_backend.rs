use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use candle_metal_kernels;
use candle_metal_kernels::Kernels;
use core::mem;
use half::{bf16, f16};
use metal;
use metal::mps::matrix::encode_gemm;
use metal::mps::Float32;
use metal::{Buffer, CommandQueue, MTLResourceOptions, NSUInteger};
use std::sync::Arc;

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),
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
    // pub fn metal_device(&self) -> &metal::DeviceRef {
    //     self.device.as_ref()
    // }

    pub fn id(&self) -> u64 {
        self.registry_id()
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    pub fn new_buffer(&self, element_count: usize, dtype: DType) -> Buffer {
        let size = (element_count * dtype.size_in_bytes()) as u64;
        // debug!("Allocate 1 - buffer size {size}");
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeManaged)
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
        match self.dtype {
            DType::F32 => Ok(CpuStorage::F32(
                self.buffer.read_to_vec(self.buffer.length() as usize / 4),
            )),
            dtype => todo!("Unsupported dtype {dtype:?}"),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        assert!(layout.is_contiguous());
        assert_eq!(dtype, DType::F32);

        let mut buffer = device.new_buffer(el, self.dtype);
        let command_buffer = self.device.command_queue.new_command_buffer();
        candle_metal_kernels::call_affine(
            &device.device,
            &command_buffer,
            &device.kernels,
            el,
            &self.buffer,
            &mut buffer,
            mul as f32,
            add as f32,
        )
        .unwrap();
        command_buffer.commit();
        return Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        });
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        // debug!("TODO reduce_op {op:?} {sum_dims:?}");
        assert!(sum_dims.len() == 1);
        assert!(sum_dims[0] == layout.shape().rank() - 1);
        assert!(layout.is_contiguous());
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
        let command_buffer = self.device.command_queue.new_command_buffer();
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
        command_buffer.commit();

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
        let command_buffer = device.command_queue.new_command_buffer();
        if layout.is_contiguous() {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32",
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
            todo!(
                "TODO Implement the kernel calling cast {:?}-{:?}",
                self.dtype,
                dtype
            );
        }

        command_buffer.commit();
        // command_buffer.wait_until_scheduled();
        // debug!(
        //     "cast {:?} - {:?} - {:?}",
        //     dtype,
        //     self.buffer.length(),
        //     buffer.length()
        // );
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
        let command_buffer = device.command_queue.new_command_buffer();
        if layout.is_contiguous() {
            use candle_metal_kernels::unary::contiguous;

            let kernel_name = match (B::KERNEL, dtype) {
                ("ucos", DType::F32) => contiguous::cos::FLOAT,
                ("usin", DType::F32) => contiguous::sin::FLOAT,
                ("usqr", DType::F32) => contiguous::sqr::FLOAT,
                ("usqrt", DType::F32) => contiguous::sqrt::FLOAT,
                ("uneg", DType::F32) => contiguous::neg::FLOAT,
                ("uexp", DType::F32) => contiguous::exp::FLOAT,
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
            todo!("TODO Implement the kernel calling {}", B::KERNEL);
        }
        command_buffer.commit();

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
        let command_buffer = device.command_queue.new_command_buffer();
        if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
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
                (name, dtype) => todo!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_binary_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                lhs_l.dims(),
                &self.buffer,
                &lhs_l.stride(),
                lhs_l.start_offset(),
                &rhs.buffer,
                &rhs_l.stride(),
                rhs_l.start_offset(),
                &mut buffer,
            )
            .map_err(MetalError::from)?;
        }
        command_buffer.commit();

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
        let command_buffer = self.device.command_queue.new_command_buffer();
        candle_metal_kernels::call_where_cond_strided(
            &device.device,
            &command_buffer,
            &device.kernels,
            "where_u8_f32",
            &dims,
            &self.buffer,
            (layout.stride(), layout.start_offset()),
            &t.buffer,
            (&t_l.stride(), t_l.start_offset()),
            &f.buffer,
            (&f_l.stride(), f_l.start_offset()),
            &mut buffer,
        )
        .map_err(MetalError::from)?;
        command_buffer.commit();
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

    fn index_select(
        &self,
        _ids: &Self,
        _src_l: &Layout,
        _ids_l: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        todo!("Index select");
        // let ids_shape = ids_l.shape();
        // let left_size: usize = src_l.dims()[..dim].iter().product();
        // let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        // let src_dim_size = src_l.dims()[dim];
        // let ids_dim_size = ids_shape.elem_count();
        // let dst_el = ids_shape.elem_count() * left_size * right_size;
        // let dtype = self.dtype;
        // let device = self.device();
        // let buffer = device.new_buffer(dst_el, dtype);
        // Ok(Self {
        //     buffer,
        //     device: device.clone(),
        //     dtype,
        // })
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
        let transpose_right = !rhs_l.is_contiguous();
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

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        if src_l.is_contiguous() {
            let command_buffer = self.device.command_queue.new_command_buffer();
            let blip = command_buffer.new_blit_command_encoder();
            blip.copy_from_buffer(
                &self.buffer,
                src_l.start_offset() as u64,
                &dst.buffer,
                dst_offset as u64,
                self.buffer.length(),
            );
        } else {
            let command_buffer = self.device.command_queue.new_command_buffer();
            let kernel_name = match self.dtype {
                DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
                DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
                DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
                dtype => todo!("copy_strided not implemented for {dtype:?}"),
            };
            candle_metal_kernels::call_unary_strided(
                &self.device.device,
                &command_buffer,
                &self.device.kernels,
                kernel_name,
                src_l.dims(),
                &self.buffer,
                &src_l.stride(),
                src_l.start_offset(),
                &mut dst.buffer,
                dst_offset,
            )
            .map_err(MetalError::from)?;
            command_buffer.commit();
        }
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
                    // debug!("TODO implement batched matmul for B={b}");
                    // bail!("Didn't implemented strided matmul yet");
                    return Ok(Self {
                        buffer: out_buffer,
                        device: self.device.clone(),
                        dtype: self.dtype(),
                    });
                }
                if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
                    // debug!(
                    //     "TODO non contiguous matmul yet {:?} {:?} - {:?} - {transpose_right}",
                    //     lhs_l.is_contiguous(),
                    //     rhs_l.is_contiguous(),
                    //     rhs_l
                    // );
                    return Ok(Self {
                        buffer: out_buffer,
                        device: self.device.clone(),
                        dtype: self.dtype(),
                    });
                }

                // debug!("TODO GEMM");
                let command_buffer = self.device.command_queue.new_command_buffer();
                encode_gemm::<Float32, Float32, Float32>(
                    &self.device,
                    &command_buffer,
                    transpose_left,
                    transpose_right,
                    &self.buffer,
                    &rhs.buffer,
                    &mut out_buffer,
                    m as NSUInteger,
                    n as NSUInteger,
                    k as NSUInteger,
                    alpha as f32,
                    beta as f32,
                    Some(b as NSUInteger),
                )
                .map_err(MetalError::from)?;

                command_buffer.commit();
                // command_buffer.wait_until_scheduled();

                Ok(Self {
                    buffer: out_buffer,
                    device: self.device.clone(),
                    dtype: self.dtype(),
                })
            }
            _ => todo!("Unimplemented matmul for this pair"),
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

        // let capture = metal::CaptureManager::shared();
        // let descriptor = metal::CaptureDescriptor::new();
        // descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        // descriptor.set_capture_device(&device);
        // let mut dir = std::env::current_dir()?;
        // dir.push("out.gputrace");
        // descriptor.set_output_url(dir);

        // capture
        //     .start_capture(&descriptor)
        //     .map_err(MetalError::from)?;
        let command_queue = device.new_command_queue();
        // let command_buffer = _command_queue.new_owned_command_buffer();
        let kernels = Arc::new(Kernels::new());
        Ok(Self {
            device,
            command_queue,
            // command_buffer,
            kernels,
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
        let option = metal::MTLResourceOptions::StorageModeManaged;
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
        // debug!("Allocate 2 - buffer size {}", buffer.length());
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
