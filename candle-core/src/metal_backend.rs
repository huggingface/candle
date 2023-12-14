use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use candle_metal_kernels;
use candle_metal_kernels::Kernels;
use metal;
use metal::{Buffer, CommandBuffer, CommandQueue, MTLResourceOptions, NSUInteger};
use std::collections::HashMap;
use std::path::Path;
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
    command_buffers: Arc<RwLock<Vec<metal::CommandBuffer>>>,
    command_buffer_index: Arc<RwLock<usize>>,
    fence: metal::Fence,
    kernels: Arc<candle_metal_kernels::Kernels>,
    buffers: Arc<RwLock<HashMap<(NSUInteger, MTLResourceOptions), Vec<Arc<Buffer>>>>>,
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

    pub fn metal_device(&self) -> &metal::Device {
        &self.device
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn command_buffer(&self) -> CommandBuffer {
        let mut command_buffers = self.command_buffers.try_write().unwrap();
        let mut command_buffer = command_buffers[0].to_owned();
        let mut index = self.command_buffer_index.try_write().unwrap();
        if *index > 20 {
            command_buffer.commit();
            command_buffer = self.command_queue.new_command_buffer().to_owned();
            *command_buffers = vec![command_buffer.clone()];
            *index = 0;
        }
        *index += 1;
        command_buffer
    }

    pub fn wait_until_completed(&self) {
        let mut command_buffers = self.command_buffers.try_write().unwrap();
        let command_buffer = &command_buffers[0];
        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                panic!("Alredy committed");
            }
            _ => {}
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        *command_buffers = vec![self.command_queue.new_command_buffer().to_owned()];
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    pub fn new_buffer(&self, element_count: usize, dtype: DType, name: &str) -> Arc<Buffer> {
        let size = (element_count * dtype.size_in_bytes()) as NSUInteger;
        self._new_buffer(size, MTLResourceOptions::StorageModePrivate, name)
    }

    fn _new_buffer(&self, size: NSUInteger, option: MTLResourceOptions, name: &str) -> Arc<Buffer> {
        // println!("Creating new buffer {name}");
        let mut buffers = self.buffers.try_write().unwrap();
        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);

        for sub in &mut *subbuffers {
            if Arc::strong_count(sub) == 1 {
                // println!("Reusing tensor {size} {name}");
                return sub.clone();
            }
        }
        let new_buffer = self.device.new_buffer(size as NSUInteger, option);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        // println!("Created tensor {size} {name}");
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(s) > 1)
                .map(|s| Arc::clone(s))
                .collect();
            *subbuffers = newbuffers;
        }

        new_buffer
    }

    pub fn new_buffer_managed(&self, size: NSUInteger) -> Arc<Buffer> {
        self._new_buffer(size, MTLResourceOptions::StorageModeManaged, "managed")
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Arc<Buffer> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        let tmp = self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            size,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let real = self._new_buffer(
            size,
            metal::MTLResourceOptions::StorageModePrivate,
            "with_data",
        );
        let command_buffer = self.command_buffer();
        command_buffer.set_label("with_data");
        let blit = command_buffer.new_blit_command_encoder();
        blit.wait_for_fence(&self.fence);
        blit.set_label("with_data_blit");
        blit.copy_from_buffer(&tmp, 0, &real, 0, tmp.length());
        blit.update_fence(&self.fence);
        blit.end_encoding();
        // drop(command_buffer);
        // real.did_modify_range(metal::NSRange::new(0, real.length()));
        // println!("Command {:?}", command.status());

        // This is necessary, for mmaped safetensors
        // Because of the unsafe slice cast we're doing.
        // The slice might not live long enough for metal
        // To actually fill the GPU buffer.
        // Putting this wait forces the GPU buffer to be filled
        // with the actual data allowing the CPU storage todo
        // deallocate properly.
        self.wait_until_completed();
        real
    }

    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(&self);
        descriptor.set_output_url(path);

        capture
            .start_capture(&descriptor)
            .map_err(MetalError::from)?;
        Ok(())
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
        let length = self.buffer.length() as usize;
        let size = self.dtype.size_in_bytes();
        if length % size != 0 {
            crate::bail!(
                "The Metal buffer length is not aligned with dtype {:?}",
                self.dtype
            );
        }
        let buffer = self.device.new_buffer_managed(self.buffer.length());
        {
            let command_buffer = self.device.command_buffer();
            command_buffer.set_label("to_cpu");
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("blit_to_cpu");
            blit.wait_for_fence(&self.device.fence);
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.update_fence(&self.device.fence);
            blit.end_encoding();
        }
        self.device.wait_until_completed();

        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(read_to_vec(&buffer, length / size))),
            DType::U32 => Ok(CpuStorage::U32(read_to_vec(&buffer, length / size))),
            DType::I64 => Ok(CpuStorage::I64(read_to_vec(&buffer, length / size))),
            DType::F16 => Ok(CpuStorage::F16(read_to_vec(&buffer, length / size))),
            DType::BF16 => Ok(CpuStorage::BF16(read_to_vec(&buffer, length / size))),
            DType::F32 => {
                let vec = read_to_vec(&buffer, length / size);
                // println!("Got back {:?}", &vec[..1]);
                Ok(CpuStorage::F32(vec))
            }
            DType::F64 => Ok(CpuStorage::F64(read_to_vec(&buffer, length / size))),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype, "affine");
        let command_buffer = self.device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "affine_float",
                DType::F16 => "affine_half",
                dtype => crate::bail!("Affine {dtype:?}"),
            };
            candle_metal_kernels::call_affine(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "affine_float_strided",
                DType::F16 => "affine_half_strided",
                dtype => crate::bail!("Affine {dtype:?}"),
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
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        }
        // buffer.did_modify_range(metal::NSRange::new(0, buffer.length()));
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn powf(&self, layout: &Layout, pow: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype, "powf");
        let command_buffer = self.device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "powf_float",
                DType::F16 => "powf_half",
                dtype => crate::bail!("Powf {dtype:?}"),
            };
            candle_metal_kernels::call_powf(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                pow as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "powf_float_strided",
                DType::F16 => "powf_half_strided",
                dtype => crate::bail!("Powf {dtype:?}"),
            };
            candle_metal_kernels::call_powf_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &buffer,
                pow as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype, "elu");
        let command_buffer = self.device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "elu_float",
                DType::F16 => "elu_half",
                dtype => crate::bail!("Powf {dtype:?}"),
            };
            candle_metal_kernels::call_elu(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                alpha as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "elu_float_strided",
                DType::F16 => "elu_half_strided",
                dtype => crate::bail!("Powf {dtype:?}"),
            };
            candle_metal_kernels::call_elu_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &buffer,
                alpha as f32,
            )
            .map_err(MetalError::from)?;
        }
        buffer.did_modify_range(metal::NSRange::new(0, buffer.length()));
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        if !(sum_dims.len() == 1
            && sum_dims[0] == layout.shape().rank() - 1
            && layout.stride()[sum_dims[0]] == 1)
        {
            crate::bail!("Non last dim reduce op not supported yet");
        }

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
            _ => crate::bail!("Reduce op for non float"),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dtype = if return_index { DType::U32 } else { self.dtype };
        if dtype == DType::U32 {
            crate::bail!("Implement return index reduce op");
        }
        let buffer = device.new_buffer(dst_el, dtype, "reduce");
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_reduce_contiguous(
            &device.device,
            &command_buffer,
            &device.kernels,
            name,
            src_el,
            dst_el,
            &self.buffer,
            layout.start_offset() * self.dtype.size_in_bytes(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::new(buffer, device, dtype))
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        crate::bail!("cmp metal")
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let device = self.device();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, "todtype");
        let command_buffer = device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32",
                (DType::U32, DType::U8) => "cast_u32_u8",
                (DType::U8, DType::U32) => "cast_u8_u32",
                (DType::F32, DType::F16) => "cast_f32_f16",
                (DType::F16, DType::F32) => "cast_f16_f32",
                (left, right) => crate::bail!("to dtype {left:?} - {right:?}"),
            };
            candle_metal_kernels::call_cast_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                layout.start_offset() * self.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32_strided",
                (DType::U32, DType::U8) => "cast_u32_u8_strided",
                (DType::U8, DType::U32) => "cast_u8_u32_strided",
                (DType::F32, DType::F16) => "cast_f32_f16_strided",
                (DType::F16, DType::F32) => "cast_f16_f32_strided",
                (left, right) => crate::bail!("to dtype {left:?} - {right:?}"),
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
                &buffer,
            )
            .map_err(MetalError::from)?;
        }
        command_buffer.set_label("to_dtype");
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, B::KERNEL);
        let command_buffer = device.command_buffer();
        command_buffer.set_label(B::KERNEL);
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
                ("utanh", DType::F32) => contiguous::tanh::FLOAT,
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
                ("utanh", DType::F16) => contiguous::tanh::HALF,
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_unary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &buffer,
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
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
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
                &buffer,
                0,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
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
        let buffer = device.new_buffer(el_count, dtype, B::KERNEL);
        let command_buffer = device.command_buffer();
        if (lhs_l.is_contiguous() && lhs_l.start_offset() == 0)
            && (rhs_l.is_contiguous() && rhs_l.start_offset() == 0)
            && &B::KERNEL[..1] != "b"
        {
            use candle_metal_kernels::binary::contiguous;

            let kernel_name = match (B::KERNEL, dtype) {
                ("add", DType::F32) => contiguous::add::FLOAT,
                // ("badd", DType::F32) => contiguous::add::FLOAT,
                ("sub", DType::F32) => contiguous::sub::FLOAT,
                //("bsub", DType::F32) => contiguous::sub::FLOAT,
                ("mul", DType::F32) => contiguous::mul::FLOAT,
                // ("bmul", DType::F32) => contiguous::mul::FLOAT,
                ("div", DType::F32) => contiguous::div::FLOAT,
                // ("bdiv", DType::F32) => contiguous::div::FLOAT,
                ("add", DType::F16) => contiguous::add::HALF,
                // ("badd", DType::F16) => contiguous::add::HALF,
                ("sub", DType::F16) => contiguous::sub::HALF,
                // ("bsub", DType::F16) => contiguous::sub::HALF,
                ("mul", DType::F16) => contiguous::mul::HALF,
                // ("bmul", DType::F16) => contiguous::mul::HALF,
                ("div", DType::F16) => contiguous::div::HALF,
                // ("bdiv", DType::F16) => contiguous::div::HALF,
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_binary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &rhs.buffer,
                &buffer,
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
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
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
                &buffer,
            )
            .map_err(MetalError::from)?;
        }
        command_buffer.set_label("binary");
        Ok(Self::new(buffer, device.clone(), dtype))
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
        let buffer = self.device.new_buffer(el, dtype, "where");
        let command_buffer = self.device.command_buffer();
        if t.dtype() != f.dtype() {
            crate::bail!("Invalid ternary different dtypes for values");
        }
        let name = match (self.dtype, t.dtype()) {
            (DType::U8, DType::F32) => "where_u8_f32",
            (DType::U8, DType::F16) => "where_u8_f16",
            (left, right) => crate::bail!("Ternary {left:?} - {right:?} not implemented"),
        };
        candle_metal_kernels::call_where_cond_strided(
            &device.device,
            &command_buffer,
            &device.kernels,
            name,
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
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device, dtype))
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv1D,
    ) -> Result<Self> {
        crate::bail!("conv1d metal")
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("conv_transpose1d metal")
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv2D,
    ) -> Result<Self> {
        crate::bail!("conv2d metal")
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("conv_tranpose2d metal")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("avg_pool2d metal")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("max_pool2d metal")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("upsample_nearest1d metal")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        crate::bail!("upsample_nearest2d metal")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("gather metal")
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
        crate::bail!("scatter_add metal")
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if !(src_l.is_contiguous()
            && src_l.start_offset() == 0
            && ids_l.is_contiguous()
            && ids_l.start_offset() == 0)
        {
            crate::bail!("Non contiguous index select not implemented");
        }
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;
        let dtype = self.dtype;
        let device = self.device();
        let buffer = device.new_buffer(dst_el, dtype, "index_select");
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "is_u32_f32",
            (DType::U32, DType::F16) => "is_u32_f16",
            (left, right) => crate::bail!("index select metal {left:?} {right:?}"),
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
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device.clone(), dtype))
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
        crate::bail!("index_add metal")
    }
    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // Create descriptors

        let buffer = self.device.new_buffer(b * m * n, self.dtype, "matmul");
        let name = match self.dtype {
            DType::F32 => "sgemm",
            DType::F16 => "hgemm",
            dtype => {
                return Err(MetalError::Message(format!("matmul doesn't support {dtype:?}")).into())
            }
        };

        let command_buffer = self.device.command_buffer();
        // println!("MATMUL {b} {m} {n} {k}");
        // println!("strides {:?} {:?}", lhs_l.stride(), rhs_l.stride());
        command_buffer.set_label("matmul");
        candle_metal_kernels::call_gemm(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            (b, m, n, k),
            &lhs_l.stride(),
            lhs_l.start_offset() * self.dtype.size_in_bytes(),
            &self.buffer,
            &rhs_l.stride(),
            rhs_l.start_offset() * rhs.dtype.size_in_bytes(),
            &rhs.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        // Create kernel

        Ok(Self::new(buffer, self.device.clone(), self.dtype()))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let command_buffer = self.device.command_buffer();
        // println!("Copy strided");
        if src_l.is_contiguous() && self.dtype == dst.dtype() {
            command_buffer.set_label("copy_contiguous");
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("copy_contiguous");
            let src_offset = (src_l.start_offset() * self.dtype.size_in_bytes()) as NSUInteger;
            let length = (src_l.shape().elem_count() * self.dtype.size_in_bytes()) as NSUInteger;
            let dst_offset = (dst_offset * dst.dtype().size_in_bytes()) as NSUInteger;
            blit.copy_from_buffer(&self.buffer, src_offset, dst.buffer(), dst_offset, length);
            blit.end_encoding();
        } else {
            let src_shape = src_l.shape();
            let el_count = src_shape.elem_count();
            if el_count == 0 {
                return Ok(());
            }
            let kernel_name = match self.dtype {
                DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
                DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
                DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
                DType::U32 => candle_metal_kernels::unary::strided::copy::U32,
                DType::U8 => candle_metal_kernels::unary::strided::copy::U8,
                dtype => crate::bail!("copy_strided not implemented for {dtype:?}"),
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
                &dst.buffer,
                dst_offset * dst.dtype.size_in_bytes(),
            )
            .map_err(MetalError::from)?;
            command_buffer.set_label("copy_strided");
        }
        Ok(())
    }
}

impl MetalStorage {
    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, dtype: DType) -> Self {
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
        // println!("CREATING DEVICE");
        let device = metal::Device::all().swap_remove(ordinal);

        let n = 1;
        let command_queue = device.new_command_queue();

        let command_buffers = (0..n)
            .map(|i| {
                let command_buffer = command_queue.new_command_buffer().to_owned();
                command_buffer.enqueue();
                command_buffer.set_label(&format!("num {i}"));
                command_buffer
            })
            .collect();
        let command_buffers = Arc::new(RwLock::new(command_buffers));
        let command_buffer_index = Arc::new(RwLock::new(0));
        let fence = device.new_fence();
        let kernels = Arc::new(Kernels::new(fence.clone()));
        let buffers = Arc::new(RwLock::new(HashMap::new()));
        Ok(Self {
            device,
            fence,
            command_queue,
            command_buffers,
            command_buffer_index,
            buffers,
            kernels,
        })
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("set_seed")
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
        let buffer = self.new_buffer(shape.elem_count(), dtype, "zeros");
        let command_buffer = self.command_buffer();
        command_buffer.set_label("zeros");
        let blit = command_buffer.new_blit_command_encoder();
        blit.fill_buffer(
            &buffer,
            metal::NSRange {
                location: 0,
                length: buffer.length(),
            },
            0,
        );
        blit.end_encoding();
        Ok(MetalStorage::new(buffer, self.clone(), dtype))
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
        Ok(Self::Storage::new(
            buffer.into(),
            self.clone(),
            storage.dtype(),
        ))
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

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}
