use crate::{notImplemented, Shape};

use super::{device::WgpuDevice, wgpu_functions::{self, read_data_from_gpu_async, BinaryOperation, UnaryOperation}};





#[derive(Debug)]
pub struct WgpuStorage {
    pub buffer : wgpu::Buffer,
    pub wgpu_device : WgpuDevice,
    pub dtype: crate::DType,
}


impl WgpuStorage {
    // pub (crate) fn new(buffer: wgpu::Buffer, wgpu_device: WgpuDevice) -> Self {
    //     Self { buffer, wgpu_device,dtype : crate::DType::F32}
    // }

    pub (crate) fn new(buffer: wgpu::Buffer, wgpu_device: WgpuDevice, dtype : crate::DType) -> Self {
        Self { buffer, wgpu_device,dtype : dtype}
    }

    pub async fn to_cpu_storage_async(&self) -> crate::Result<crate::CpuStorage> {
        match self.dtype{
            crate::DType::U32 => return Ok(crate::CpuStorage::U32(read_data_from_gpu_async(&self.wgpu_device, &self.buffer).await)),
            crate::DType::F32 => return Ok(crate::CpuStorage::F32(read_data_from_gpu_async(&self.wgpu_device, &self.buffer).await)),
            _ => notImplemented!(to_cpu_storage_async),
        }
    }

    pub fn get_length(&self) -> usize{
        return (self.buffer.size() / 4) as usize; //f32
    }

}


impl crate::backend::BackendStorage for WgpuStorage{
    type Device = WgpuDevice;

    fn try_clone(&self, _: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), self.buffer.size() as usize);
        wgpu_functions::queue_copy(self.device(), &buffer_dest, &self.buffer, 0, 0, self.buffer.size() as usize);
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn dtype(&self) -> crate::DType {
        return self.dtype;
    }

    fn device(&self) -> &Self::Device {
        return &self.wgpu_device;
    }

    #[cfg(target_arch = "wasm32")]
    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage>{
        return Err(Error::WebGpu("Sync copy to CpuStorage is not allowed for WebGpu device in WebAssembly. First copy the date asynchronously to a CpuStorage".to_owned().into()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        match self.dtype{
            crate::DType::U32 => return Ok(crate::CpuStorage::U32(pollster::block_on(read_data_from_gpu_async(&self.wgpu_device, &self.buffer)))),
            crate::DType::F32 => return Ok(crate::CpuStorage::F32(pollster::block_on(read_data_from_gpu_async(&self.wgpu_device, &self.buffer)))),
            _ => todo!(),
        }
    }

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), layout.shape().elem_count() * 4);
        wgpu_functions::queue_unary_from_buffer_op(self.device(), &buffer_dest, &self.buffer,  UnaryOperation::Affine, mul as f32, add as f32, self.dtype, layout)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn powf(&self, layout: &crate::Layout, e: f64) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), layout.shape().elem_count() * 4);
        wgpu_functions::queue_unary_from_buffer_op(self.device(), &buffer_dest, &self.buffer,  UnaryOperation::PowScalar, e as f32, 0.0, self.dtype,layout)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn elu(&self, layout: &crate::Layout, alpha: f64) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), layout.shape().elem_count() * 4);
        wgpu_functions::queue_unary_from_buffer_op(self.device(), &buffer_dest, &self.buffer,  UnaryOperation::Elu, alpha as f32, 0.0, self.dtype,layout)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn reduce_op(&self, reduce_op: crate::op::ReduceOp, layout: &crate::Layout, reduce_dims: &[usize]) -> crate::Result<Self> {
        let src_dims = layout.dims();
        let mut dst_dims = src_dims.to_vec();
        for &dim in reduce_dims.iter() {
            dst_dims[dim] = 1;
        }
        let dst_shape = Shape::from(dst_dims);
        let mut reduce_dims = reduce_dims.to_vec();
        // Sort the reduce_dims as they have to be processed from left to right when converting the
        // indexes.
        reduce_dims.sort();

        let mut reduce_dims_bit = 0;
        if reduce_dims.contains(&0){
            reduce_dims_bit |= 0b1
        }
        if reduce_dims.contains(&1){
            reduce_dims_bit |= 0b10
        }
        if reduce_dims.contains(&2){
            reduce_dims_bit |= 0b100
        }
        if reduce_dims.contains(&3){
            reduce_dims_bit |= 0b1000
        }
        if reduce_dims.contains(&4){
            reduce_dims_bit |= 0b10000
        }

        let reduce_dims_and_stride: Vec<_> = reduce_dims
            .iter()
            .map(|&d| (src_dims[d], src_dims[d + 1..].iter().product::<usize>()))
            .collect();
        

        
        let buffer_dest = wgpu_functions::create_buffer(self.device(), dst_shape.elem_count() * 4);
        let op = match reduce_op{
            crate::op::ReduceOp::Sum => wgpu_functions::ReduceOperations::Sum,
            crate::op::ReduceOp::Min => wgpu_functions::ReduceOperations::Min,
            crate::op::ReduceOp::Max => wgpu_functions::ReduceOperations::Max,
            crate::op::ReduceOp::ArgMin => wgpu_functions::ReduceOperations::ArgMin,
            crate::op::ReduceOp::ArgMax => wgpu_functions::ReduceOperations::ArgMax,
        };
        wgpu_functions::queue_reduce_from_buffer_op(self.device(), &buffer_dest, &self.buffer, op, self.dtype, layout, reduce_dims_bit, dst_shape)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn cmp(&self, op: crate::op::CmpOp, rhs: &Self, lhs_l: &crate::Layout, rhs_l: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), lhs_l.shape().elem_count() * 4);
        
        let op2 =match op{
            crate::op::CmpOp::Eq => wgpu_functions::CmpOperation::Eq,
            crate::op::CmpOp::Ne => wgpu_functions::CmpOperation::Ne,
            crate::op::CmpOp::Le => wgpu_functions::CmpOperation::Le,
            crate::op::CmpOp::Ge => wgpu_functions::CmpOperation::Ge,
            crate::op::CmpOp::Lt => wgpu_functions::CmpOperation::Lt,
            crate::op::CmpOp::Gt => wgpu_functions::CmpOperation::Gt,
        };

        wgpu_functions::queue_cmp_buffer_from_buffer(self.device(), &buffer_dest, &self.buffer, &rhs.buffer, op2, self.dtype, lhs_l, rhs_l)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), crate::DType::U32));
    }

    fn to_dtype(&self, _layout: &crate::Layout, dtype: crate::DType) -> crate::Result<Self> {
        match dtype{
            //crate::DType::F32 => return Ok(WgpuStorage::new(self.buffer,self.device().clone())) ,
            _ =>  panic!("conversion of dtype to {:?} not suported on webgpu", dtype),
        }

       
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), layout.shape().elem_count() * 4);
        let op = match B::NAME{
            "gelu" => UnaryOperation::Gelu,
            //"erf" => UnaryOperation::Erf,
            "silu" => UnaryOperation::SiLu,
            "ceil" => UnaryOperation::Ceil,
            "floor" => UnaryOperation::Floor,
            "round" => UnaryOperation::Round,
            //"gelu_erf" => UnaryOperation::GeluErf,
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
            "sigmoid" => UnaryOperation::Sigmoid,
            _ =>{panic!("Operation {} is not supported on wgpu", B::NAME)}
        };
        wgpu_functions::queue_unary_from_buffer_op(self.device(), &buffer_dest, &self.buffer, op, 0.0, 0.0, self.dtype, layout)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn binary_impl<B: crate::op::BinaryOpT>(&self, rhs: &Self, lhs_layout: &crate::Layout, rhs_layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), lhs_layout.shape().elem_count() * 4);
        
        let op = match B::NAME{
            "add" => BinaryOperation::Add,
            "sub" => BinaryOperation::Minus,
            "mul" => BinaryOperation::Mult,
            "div" => BinaryOperation::Div,
            "minimum" => BinaryOperation::Min,
            "maximum" => BinaryOperation::Max,
            _ =>{panic!("Operation {} is not supported on wgpu", B::NAME)}
        };
        
        wgpu_functions::queue_binary_buffer_from_buffer(self.device(), &buffer_dest, &self.buffer, &rhs.buffer,op, self.dtype,lhs_layout, rhs_layout)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(),self.dtype));
    }

    fn where_cond(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: &Self, _: &crate::Layout) -> crate::Result<Self> {
        notImplemented!(where_cont)
    }

    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        notImplemented!(conv1d)
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        notImplemented!(conv_transpose1d)
    }

    fn conv2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), (params.b_size * params.c_out * params.out_h() * params.out_w()) * 4);
        wgpu_functions::queue_conv2d(self.device(), &buffer_dest, &self.buffer, &kernel.buffer, self.dtype,params,l)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(),self.dtype));
    }

    fn conv_transpose2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        
        let buffer_dest = wgpu_functions::create_buffer(self.device(), (params.b_size * params.c_out * params.out_h() * params.out_w()) * 4);
        wgpu_functions::queue_conv2d_transpose(self.device(), &buffer_dest, &self.buffer, &kernel.buffer, self.dtype,params,l)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(),self.dtype));
    }

    fn avg_pool2d(&self, _: &crate::Layout, _: (usize, usize), _: (usize, usize)) -> crate::Result<Self> {
        notImplemented!(avg_pool2d)
    }

    fn max_pool2d(&self, _: &crate::Layout, _: (usize, usize), _: (usize, usize)) -> crate::Result<Self> {
        notImplemented!(max_pool2d)
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        notImplemented!(upsample_nearest1d)
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> crate::Result<Self> {
        notImplemented!(upsample_nearest2d)
    }

    fn gather(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        notImplemented!(gather)
    }

    fn scatter_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        notImplemented!(scatter_add)
    }

    fn index_select(&self, _: &Self, _: &crate::Layout, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        notImplemented!(index_select)
    }

    fn index_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        notImplemented!(index_add)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (batching,m,n,k): (usize, usize, usize, usize),
        layout1: &crate::Layout,
        layout2: &crate::Layout,
    ) -> crate::Result<Self> {

        //mXk * kXn -> mXn * nXk
        let m2  = m;
        let n2 = k;
        let k2 = n;

        let buffer_dest = wgpu_functions::create_buffer(self.device(), batching * (m2 * k2) * 4);
        wgpu_functions::queue_matmul_buffer(self.device(), &buffer_dest, &self.buffer, &rhs.buffer,batching as u32, m2 as u32, n2 as u32, k2 as u32,layout1,layout2, self.dtype)?;
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone(), self.dtype));
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &crate::Layout) -> crate::Result<()> {
        match src_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => {
                let to_copy = (dst.get_length() - dst_offset).min(len);
                wgpu_functions::queue_copy(self.device(), &dst.buffer, &self.buffer, dst_offset, start_offset, to_copy);
            }
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            } => {
                let mut dst_index = dst_offset;
                for src_index in block_start_index {
                    let next_dst_index = dst_index + block_len;
                    if dst_index >= dst.get_length() {
                        break;
                    }
                    let to_copy = usize::min(block_len, dst.get_length() - dst_index);
                    wgpu_functions::queue_copy(self.device(), &dst.buffer, &self.buffer, dst_index, src_index, to_copy);
                    dst_index = next_dst_index
                }
            }
        }
        return Ok(());
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
        for i1 in 0..d1 {
            let dst_idx = i1 * dst_stride1 + dst_offset;
            let src_idx = i1 * src_stride1 + src_offset;
            wgpu_functions::queue_copy(self.device(), &dst.buffer, &self.buffer, dst_idx, src_idx, d2);
        }
        Ok(())
    }
}