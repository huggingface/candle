use crate::{DType, Layout, Shape};

use wgpu_compute_layer::cache::BufferReferenceId;
use super::{
    device::WgpuDevice,
    wgpu_functions::{
        self, binary::BinaryOperation, cmp::CmpOperation, matmul::SGEMMParams,
        reduce::ReduceOperations, unary::UnaryOperation,
        WgpuTensor,
    },
};

#[derive(Debug)]
pub struct WgpuStorage(pub (crate)wgpu_compute_layer::WgpuStorage, pub (crate)WgpuDevice);

impl WgpuStorage{
    pub fn device(&self) -> &WgpuDevice {
        &self.1
    }

    pub fn dtype(&self) -> crate::DType {
        self.0.wgpu_dtype().into()
    }

    pub fn wgpu_dtype(&self) -> wgpu_compute_layer::DType {
        self.0.wgpu_dtype()
    }

    pub fn buffer(&self) -> BufferReferenceId {
        self.0.buffer()
    }

    pub fn size(&self) -> u64 {
        self.0.size_in_bytes() as u64
    }

    pub fn size_in_bytes(&self) -> usize {
        self.0.size_in_bytes()
    }
}

impl WgpuStorage {
    pub fn new(
        buffer: BufferReferenceId,
        wgpu_device: super::WgpuDevice,
        dtype: crate::DType,
        size: u64,
    ) -> Self {
        Self(wgpu_compute_layer::WgpuStorage::new(buffer, wgpu_device.inner_device().clone(), dtype.into(), size), wgpu_device)
    }

    pub(crate) fn temporary_clone(&self) -> Self {
        unsafe {
            Self(self.0.temporary_clone(), self.1.clone())
        }
    }

    pub async fn to_cpu_storage_async(&self) -> crate::Result<crate::CpuStorage> {
        match self.dtype() {
            crate::DType::U32 => Ok(crate::CpuStorage::U32(
               self.0.read_from_buffer_reference_async().await?,
            )),
            crate::DType::F32 => Ok(crate::CpuStorage::F32(
                self.0.read_from_buffer_reference_async().await?,
            )),
            crate::DType::U8 => Ok(crate::CpuStorage::U8(
                self.0.read_from_buffer_reference_async().await?,
            )),
            crate::DType::I64 => Ok(crate::CpuStorage::I64(
                self.0.read_from_buffer_reference_async().await?,
            )),
            crate::DType::F64 => Ok(crate::CpuStorage::F64(
                self.0.read_from_buffer_reference_async().await?,
            )),
            crate::DType::F16 => Ok(crate::CpuStorage::F16(
                self.0.read_from_buffer_reference_async().await?,
            )),
            _ => todo!(),
        }
    }

    fn try_clone_layout(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), layout.shape().elem_count());
        self.copy_strided_src(&buffer_dest, 0, layout)?;
        Ok(buffer_dest)
    }

    fn copy_strided_src(
        &self,
        dst: &WgpuStorage,
        dst_offset: usize,
        src_l: &crate::Layout,
    ) -> crate::Result<()> {
        match src_l.contiguous_offsets() {
            Some((start, end)) => {
                let len = end - start;
                let to_copy = ((dst.size() as usize / 4) - dst_offset).min(len);
                wgpu_functions::queue_copy(
                    self.device(),
                    dst.buffer(),
                    self.buffer(),
                    dst_offset,
                    start,
                    to_copy,
                    self.dtype(),
                )?;
            }
            None => {
                wgpu_functions::queue_copy_strided(
                    self.device(),
                    dst.buffer(),
                    self.buffer(),
                    self.dtype(),
                    src_l,
                    dst_offset as u32,
                )?;
            }
        }
        Ok(())
    }
}

impl crate::backend::BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), self.size() / self.dtype().size_in_bytes() as u64);
        wgpu_functions::queue_copy(
            self.device(),
            buffer_dest.buffer(),
            self.buffer(),
            0,
            0,
            (self.size() / 4) as usize,
            self.dtype(),
        )?;

        Ok(buffer_dest)
    }

    fn dtype(&self) -> crate::DType {
        self.0.wgpu_dtype().into()
    }

    fn device(&self) -> &Self::Device {
        &self.1
    }

    #[cfg(target_arch = "wasm32")]
    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        panic!("Sync copy to CpuStorage is not allowed for wgpu device in WebAssembly. First copy the date asynchronously to a CpuStorage");
        //panic, so we get a stacktrace and see where we wanted to copy
        //return Err(crate::Error::Wgpu("Sync copy to CpuStorage is not allowed for wgpu device in WebAssembly. First copy the date asynchronously to a CpuStorage".to_owned().into()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        pollster::block_on(self.to_cpu_storage_async())
    }

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), layout.shape().elem_count());
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(layout, self.buffer()),
            UnaryOperation::Affine,
            mul as f32,
            add as f32,
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn powf(&self, layout: &crate::Layout, e: f64) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), layout.shape().elem_count());
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(layout, self.buffer()),
            UnaryOperation::PowScalar,
            e as f32,
            0.0,
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn elu(&self, layout: &crate::Layout, alpha: f64) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), layout.shape().elem_count());
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(layout, self.buffer()),
            UnaryOperation::Elu,
            alpha as f32,
            0.0,
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn reduce_op(
        &self,
        reduce_op: crate::op::ReduceOp,
        layout: &crate::Layout,
        reduce_dims: &[usize],
    ) -> crate::Result<Self> {
        let src_dims = layout.dims();
        let mut dst_dims = src_dims.to_vec();
        for &dim in reduce_dims.iter() {
            dst_dims[dim] = 1;
        }
        let dst_shape = Shape::from(dst_dims);
        let mut reduce_dims = reduce_dims.to_vec();

        fn calculate_stride(shape: &[usize]) -> Vec<usize> {
            // Reverse the shape vector and fold over it
            let mut strides = shape
                .iter()
                .rev()
                .scan(1, |state, &dim| {
                    let current_stride = *state;
                    *state *= dim;
                    Some(current_stride)
                })
                .collect::<Vec<usize>>();
            // Reverse the strides to get them in the correct order
            strides.reverse();
            strides
        }

        let output_dtype = match reduce_op {
            crate::op::ReduceOp::ArgMin | crate::op::ReduceOp::ArgMax => crate::DType::U32,
            _ => self.dtype(),
        };

        let buffer_dest = self
            .device()
            .alloc_uninit_size(output_dtype, dst_shape.elem_count());

        let op = match reduce_op {
            crate::op::ReduceOp::Sum => ReduceOperations::Sum,
            crate::op::ReduceOp::Min => ReduceOperations::Min,
            crate::op::ReduceOp::Max => ReduceOperations::Max,
            crate::op::ReduceOp::ArgMin => ReduceOperations::ArgMin,
            crate::op::ReduceOp::ArgMax => ReduceOperations::ArgMax,
        };

        // Sort the reduce_dims as they have to be processed from left to right when converting the
        // indexes.
        reduce_dims.sort();
        let mut start_reduce_dim = 0;
        let mut end_reduce_dim = 1;
        let mut current_shape = layout.shape().clone().into_dims();
        let input_stride = calculate_stride(&current_shape[..]);
        let mut current_buffer = None;

        let call_reduce = |output_buffer: BufferReferenceId,
                           output_size: u32,
                           start_reduce_dim: usize,
                           end_reduce_dim: usize,
                           reduce_dims: &Vec<usize>,
                           prev_buffer: BufferReferenceId,
                           current_shape: &Vec<usize>,
                           layout: &Layout|
         -> crate::Result<()> {
            let start_dim = reduce_dims[start_reduce_dim];
            let end_dim = reduce_dims[end_reduce_dim - 1];
            let output_to_start_shape_stride2 = src_dims[(end_dim + 1)..]
                .iter()
                .fold(1, |prev, c| prev * *c)
                as u32;

            let output_to_start_stride1;
            if let Some(index) = current_shape.iter().rposition(|c| *c != 1) {
                output_to_start_stride1 = input_stride[index] as u32;
            } else {
                //All Other Elements have a Shape of 1?
                output_to_start_stride1 = 1_u32;
            }
            let output_to_start_stride2 =
                src_dims[start_dim..].iter().fold(1, |prev, c| prev * *c) as u32;
            let output_to_start_stride2 =
                output_to_start_stride2 - output_to_start_shape_stride2 * output_to_start_stride1;
            let reduction_length = src_dims[start_dim..(end_dim + 1)]
                .iter()
                .fold(1, |prev, c| prev * *c);
            let stride_reduction = *input_stride[start_dim..(end_dim + 1)].iter().min().unwrap();
            wgpu_functions::queue_reduce_from_buffer_op(
                self.device(),
                output_buffer,
                prev_buffer,
                op,
                self.dtype(),
                layout,
                wgpu_functions::reduce::ReduceParams {
                    dest_size: output_size,
                    output_to_start_shape_stride2, //Multiply all Shapes after EndDim
                    output_to_start_stride1, //Find Stride of last dimension(that was not reduced)
                    output_to_start_stride2, //(Multiply all Shapes from StartDim until end) - output_to_start_shape_stride2 * output_to_start_stride1
                    reduction_length: reduction_length as u32,
                    stride_reduction: stride_reduction as u32, //length of elements to reduce per output
                },
            )?;
            Ok(())
        };

        loop {
            if end_reduce_dim < reduce_dims.len() {
                if reduce_dims[end_reduce_dim] == reduce_dims[end_reduce_dim - 1] + 1 {
                    //the current end, is handled for the same block
                    end_reduce_dim += 1;
                } else {
                    let start_dim = reduce_dims[start_reduce_dim];
                    let end_dim = reduce_dims[end_reduce_dim - 1];

                    let l = Layout::contiguous(Shape::from_dims(&current_shape));

                    for c in current_shape.iter_mut().take(end_dim + 1).skip(start_dim) {
                        *c = 1;
                    }

                    let output_count = current_shape.iter().product::<usize>();

                    let buffer_temp = self.device().inner_device()
                        .create_buffer_reference(output_count * self.dtype().size_in_bytes(), false);

                    let (prev_buffer, l) = match current_buffer {
                        Some(buffer) => (buffer, &l),
                        None => (self.buffer(), layout),
                    };

                    call_reduce(
                        buffer_temp,
                        output_count as u32,
                        start_reduce_dim,
                        end_reduce_dim,
                        &reduce_dims,
                        prev_buffer,
                        &current_shape,
                        l,
                    )?;

                    current_buffer = Some(buffer_temp);

                    start_reduce_dim = end_reduce_dim;
                    end_reduce_dim += 1;
                }
            } else {
                //end was outside of range,
                let start_dim = reduce_dims[start_reduce_dim];
                let end_dim = reduce_dims[end_reduce_dim - 1];

                let l = Layout::contiguous(Shape::from_dims(&current_shape));

                for c in current_shape.iter_mut().take(end_dim + 1).skip(start_dim) {
                    *c = 1;
                }

                let (prev_buffer, l) = match current_buffer {
                    Some(buffer) => (buffer, &l),
                    None => (self.buffer(), layout),
                };

                call_reduce(
                    buffer_dest.buffer(),
                    dst_shape.elem_count() as u32,
                    start_reduce_dim,
                    end_reduce_dim,
                    &reduce_dims,
                    prev_buffer,
                    &current_shape,
                    l,
                )?;

                break;
            }
        }
        Ok(buffer_dest)
    }

    fn cmp(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_size = lhs_l.shape().elem_count().div_ceil(4) * 4;
        let buffer_dest = self
            .device()
            .alloc_uninit_size(DType::U8, buffer_size);

        let op2 = match op {
            crate::op::CmpOp::Eq => CmpOperation::Eq,
            crate::op::CmpOp::Ne => CmpOperation::Ne,
            crate::op::CmpOp::Le => CmpOperation::Le,
            crate::op::CmpOp::Ge => CmpOperation::Ge,
            crate::op::CmpOp::Lt => CmpOperation::Lt,
            crate::op::CmpOp::Gt => CmpOperation::Gt,
        };

        wgpu_functions::queue_cmp_buffer_from_buffer(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(lhs_l, self.buffer()),
            WgpuTensor::new(rhs_l, rhs.buffer()),
            op2,
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn to_dtype(&self, layout: &crate::Layout, dtype: crate::DType) -> crate::Result<Self> {
        match (self.dtype(), dtype) {
            (DType::F32, DType::F32) => self.try_clone_layout(layout),
            (DType::U32, DType::U32) => self.try_clone_layout(layout),
            (DType::U8, DType::F32) => {
                let buffer_dest = self
                    .device()
                    .alloc_uninit_size(DType::F32, layout.shape().elem_count());
                wgpu_functions::queue_convert_u8_to_f32(
                    self.device(),
                    buffer_dest.buffer(),
                    self.buffer(),
                    layout,
                )?;
                Ok(buffer_dest)
            }
            (DType::F32, DType::U8) => {
                if !layout.is_contiguous() {
                    panic!(
                        "conversion from {:?} to {:?} not suported for non contiguous matrix",
                        self.dtype(), dtype
                    );
                }
                let buffer_dest = self
                    .device()
                    .alloc_uninit_size(DType::U8, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_f32_to_u8(
                    self.device(),
                    buffer_dest.buffer(),
                    self.buffer(),
                    layout.start_offset() as u32,
                    layout.shape().elem_count() as u32,
                )?;
                Ok(buffer_dest)
            }
            (DType::F32, DType::F16) => {
                if !layout.is_contiguous() {
                    panic!(
                        "conversion from {:?} to {:?} not suported for non contiguous matrix",
                        self.dtype(), dtype
                    );
                }
                let buffer_dest = self
                    .device()
                    .alloc_uninit_size(DType::F16, layout.shape().elem_count());
                wgpu_functions::queue_convert_f32_to_f16(
                    self.device(),
                    buffer_dest.buffer(),
                    self.buffer(),
                    layout.start_offset() as u32,
                    layout.shape().elem_count() as u32,
                )?;
                Ok(buffer_dest)
            }
            (DType::F16, DType::F32) => {
                if !layout.is_contiguous() {
                    panic!(
                        "conversion from {:?} to {:?} not suported for non contiguous matrix",
                        self.dtype(), dtype
                    );
                }
                let buffer_dest = self
                    .device()
                    .alloc_uninit_size(DType::F32, layout.shape().elem_count());
                wgpu_functions::queue_convert_f16_to_f32(
                    self.device(),
                    buffer_dest.buffer(),
                    self.buffer(),
                    layout.start_offset() as u32,
                    layout.shape().elem_count() as u32,
                )?;
                Ok(buffer_dest)
            }
            (DType::U32, DType::U8) => {
                if !layout.is_contiguous() {
                    panic!(
                        "conversion from {:?} to {:?} not suported for non contiguous matrix",
                        self.dtype(), dtype
                    );
                }
                let buffer_dest = self
                    .device()
                    .alloc_uninit_size(DType::U8, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_u32_to_u8(
                    self.device(),
                    buffer_dest.buffer(),
                    self.buffer(),
                    layout.start_offset() as u32,
                    layout.shape().elem_count() as u32,
                )?;
                Ok(buffer_dest)
            }
            (input_type, output_type) => {
                let buffer_dest = self
                    .device()
                    .alloc_uninit_size(output_type, layout.shape().elem_count());
                wgpu_functions::queue_convert(
                    self.device(),
                    buffer_dest.buffer(),
                    self.buffer(),
                    layout,
                    output_type,
                    input_type,
                )?;
                Ok(buffer_dest)
            }
        }
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), layout.shape().elem_count());

        let op = match B::NAME {
            "gelu" => UnaryOperation::Gelu,
            "erf" => UnaryOperation::Erf,
            "silu" => UnaryOperation::SiLu,
            "ceil" => UnaryOperation::Ceil,
            "floor" => UnaryOperation::Floor,
            "round" => UnaryOperation::Round,
            "gelu_erf" => UnaryOperation::GeluErf,
            "sign" => UnaryOperation::Sign,
            "abs" => UnaryOperation::Abs,

            "exp" => UnaryOperation::Exp,
            "log" => UnaryOperation::Log,
            "sin" => UnaryOperation::Sin,
            "cos" => UnaryOperation::Cos,
            "neg" => UnaryOperation::Neg,
            "recip" => UnaryOperation::Inverse,
            "sqr" => UnaryOperation::Square,
            "sqrt" => UnaryOperation::Sqrt,
            "tanh" => UnaryOperation::Tanh,
            "relu" => UnaryOperation::Relu,
            "sigmoid" => UnaryOperation::Sigmoid,
            _ => {
                panic!("Operation {} is not supported on wgpu", B::NAME)
            }
        };
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(layout, self.buffer()),
            op,
            0.0,
            0.0,
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &crate::Layout,
        rhs_layout: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), lhs_layout.shape().elem_count());

        let op = match B::NAME {
            "add" => BinaryOperation::Add,
            "sub" => BinaryOperation::Minus,
            "mul" => BinaryOperation::Mult,
            "div" => BinaryOperation::Div,
            "minimum" => BinaryOperation::Min,
            "maximum" => BinaryOperation::Max,
            _ => {
                panic!("Operation {} is not supported on wgpu", B::NAME)
            }
        };

        wgpu_functions::queue_binary_buffer_from_buffer(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(lhs_layout, self.buffer()),
            WgpuTensor::new(rhs_layout, rhs.buffer()),
            op,
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn where_cond(
        &self,
        input_layout: &crate::Layout,
        t: &Self, //true values
        t_layout: &crate::Layout,
        f: &Self, //false values
        f_layout: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(t.dtype(), input_layout.shape().elem_count());

        wgpu_functions::where_cond::queue_where_cond(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(input_layout, self.buffer()),
            WgpuTensor::new(t_layout, t.buffer()),
            WgpuTensor::new(f_layout, f.buffer()),
            self.dtype(),
            t.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn conv1d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), params.b_size * params.c_out * params.l_out());

        wgpu_functions::queue_conv1d(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(l, self.buffer()),
            WgpuTensor::new(kernel_l, kernel.buffer()),
            self.dtype(),
            params,
        )?;
        Ok(buffer_dest)
    }

    fn conv_transpose1d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), params.b_size * params.c_out * params.l_out());
        wgpu_functions::queue_conv1d_transpose(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(l, self.buffer()),
            WgpuTensor::new(kernel_l, kernel.buffer()),
            self.dtype(),
            params,
        )?;
        Ok(buffer_dest)
    }

    fn conv2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        let buffer_dest = self.device().alloc_uninit_size(
            self.dtype(),
            params.b_size * params.c_out * params.out_h() * params.out_w(),
        );
        wgpu_functions::queue_conv2d(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(l, self.buffer()),
            WgpuTensor::new(kernel_l, kernel.buffer()),
            self.dtype(),
            params,
        )?;
        Ok(buffer_dest)
    }

    fn conv_transpose2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        let buffer_dest = self.device().alloc_uninit_size(
            self.dtype(),
            params.b_size * params.c_out * params.out_h() * params.out_w(),
        );
        wgpu_functions::queue_conv2d_transpose(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(l, self.buffer()),
            WgpuTensor::new(kernel_l, kernel.buffer()),
            self.dtype(),
            params,
        )?;
        Ok(buffer_dest)
    }

    fn avg_pool2d(
        &self,
        layout: &crate::Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> crate::Result<Self> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let h_out = (h - kernel_size.1) / stride.1 + 1;
        let w_out = (w - kernel_size.0) / stride.0 + 1;

        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), b * c * h_out * w_out);

        wgpu_functions::queue_avg_pool2d(
            self.device(),
            buffer_dest.buffer(),
            self.buffer(),
            layout,
            self.dtype(),
            kernel_size,
            stride,
        )?;

        Ok(buffer_dest)
    }

    fn max_pool2d(
        &self,
        layout: &crate::Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> crate::Result<Self> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let h_out = (h - kernel_size.1) / stride.1 + 1;
        let w_out = (w - kernel_size.0) / stride.0 + 1;

        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), b * c * h_out * w_out);

        wgpu_functions::queue_max_pool2d(
            self.device(),
            buffer_dest.buffer(),
            self.buffer(),
            layout,
            self.dtype(),
            kernel_size,
            stride,
        )?;

        Ok(buffer_dest)
    }

    fn upsample_nearest1d(
        &self,
        layout: &crate::Layout,
        target_size: usize,
    ) -> crate::Result<Self> {
        let (b, c, _) = layout.shape().dims3()?;

        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), b * c * target_size);

        wgpu_functions::queue_upsample1d(
            self.device(),
            buffer_dest.buffer(),
            self.buffer(),
            layout,
            self.dtype(),
            target_size,
        )?;

        Ok(buffer_dest)
    }

    fn upsample_nearest2d(
        &self,
        layout: &crate::Layout,
        target_size_y: usize,
        target_size_x: usize,
    ) -> crate::Result<Self> {
        let (b, c, _, _) = layout.shape().dims4()?;

        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), b * c * target_size_x * target_size_y);

        wgpu_functions::queue_upsample2d(
            self.device(),
            buffer_dest.buffer(),
            self.buffer(),
            layout,
            self.dtype(),
            (target_size_y, target_size_x),
        )?;

        Ok(buffer_dest)
    }

    fn gather(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        d: usize,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), indexes_l.shape().elem_count());

        wgpu_functions::queue_gather(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(l, self.buffer()),
            WgpuTensor::new(indexes_l, indexes.buffer()),
            self.dtype(),
            d,
        )?;

        Ok(buffer_dest)
    }

    fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
        d: usize,
    ) -> crate::Result<Self> {
        let mut new_shape = lhs_l.shape().clone().into_dims();
        new_shape[d] = rhs_l.shape().elem_count();
        let new_shape = Shape::from_dims(&new_shape[..]);

        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), new_shape.elem_count());

        wgpu_functions::queue_index_select(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(lhs_l, self.buffer()),
            WgpuTensor::new(rhs_l, rhs.buffer()),
            self.dtype(),
            rhs.dtype(),
            d,
        )?;
        Ok(buffer_dest)
    }

    fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), l.shape().elem_count());

        self.copy_strided_src(&buffer_dest, 0, l)?;

        wgpu_functions::queue_index_add_inplace(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(indexes_l, indexes.buffer()),
            WgpuTensor::new(source_l, source.buffer()),
            self.dtype(),
            &Layout::contiguous(l.shape().clone()),
            d,
        )?;

        Ok(buffer_dest)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (batching, m, n, k): (usize, usize, usize, usize),
        layout1: &crate::Layout,
        layout2: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = self
            .device()
            .alloc_uninit_size(self.dtype(), batching * (m * n));

        wgpu_functions::queue_matmul_buffer(
            self.device(),
            buffer_dest.buffer(),
            WgpuTensor::new(layout1, self.buffer()),
            WgpuTensor::new(layout2, rhs.buffer()),
            SGEMMParams::new(batching, m, k, n),
            self.dtype(),
        )?;
        Ok(buffer_dest)
    }

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &crate::Layout,
    ) -> crate::Result<()> {
        self.copy_strided_src(dst, dst_offset, src_l)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> crate::Result<()> {
        wgpu_functions::queue_copy2d(
            self.device(),
            (dst.buffer(), dst_stride1 as u32, dst_offset as u32),
            (self.buffer(), src_stride1 as u32, src_offset as u32),
            self.dtype(),
            d1 as u32,
            d2 as u32,
        )?;
        Ok(())
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> crate::Result<()> {
        wgpu_functions::queue_scatter_set_inplace(
            self.device(),
            self.buffer(),
            WgpuTensor::new(indexes_l, indexes.buffer()),
            WgpuTensor::new(source_l, source.buffer()),
            self.dtype(),
            &Layout::contiguous(l.shape().clone()),
            d,
        )?;

        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> crate::Result<()> {
        wgpu_functions::queue_scatter_add_inplace(
            self.device(),
            self.buffer(),
            WgpuTensor::new(indexes_l, indexes.buffer()),
            WgpuTensor::new(source_l, source.buffer()),
            self.dtype(),
            &Layout::contiguous(l.shape().clone()),
            d,
        )?;

        Ok(())
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> crate::Result<()> {
        if scalar.to_f64() == 0.0 {
            wgpu_functions::queue_unary_inplace_op(
                self.device(),
                self.buffer(),
                UnaryOperation::SetZero,
                0.0,
                0.0,
                scalar.dtype(),
                layout,
            )
        } else if scalar.to_f64() == 1.0 {
            wgpu_functions::queue_unary_inplace_op(
                self.device(),
                self.buffer(),
                UnaryOperation::SetOne,
                0.0,
                0.0,
                scalar.dtype(),
                layout,
            )
        } else {
            wgpu_functions::queue_unary_inplace_op(
                self.device(),
                self.buffer(),
                UnaryOperation::SetScalar,
                scalar.to_f64() as f32,
                0.0,
                scalar.dtype(),
                layout,
            )
        }
    }

    fn upsample_bilinear2d(
        &self,
        l_src: &Layout,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> crate::Result<Self> {
        use crate::wgpu::wgpu_functions;

        if !l_src.is_contiguous() {
            crate::bail!("input must be contiguous");
        }

        let (n, c, in_h, in_w) = l_src.shape().dims4()?;

        let out_h = out_h as u32;
        let out_w = out_w as u32;

        // Allocate output
        let el = (n as u32 * c as u32 * out_h * out_w) as usize;
        let output_buffer = self.device().alloc_uninit_size(self.dtype(), el);

        wgpu_functions::queue_upsample_bilinear2d(
            self.device(),
            (self.buffer(), l_src.start_offset() as u32),
            self.dtype(),
            output_buffer.buffer(),
            n as u32,
            c as u32,
            in_h as u32,
            in_w as u32,
            out_h,
            out_w,
            align_corners,
            scale_h,
            scale_w,
        )?;

        Ok(output_buffer)
    }
}

