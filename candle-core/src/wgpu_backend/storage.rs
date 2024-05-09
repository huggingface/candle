use crate::Error;

use super::{device::WgpuDevice, wgpu_functions::{self, read_data_from_gpu_async, BinaryOperation, UnaryOperation}};




#[derive(Debug)]
pub struct WgpuStorage {
    pub buffer : wgpu::Buffer,
    pub wgpu_device : WgpuDevice
}


impl WgpuStorage {
    pub (crate) fn new(buffer: wgpu::Buffer, wgpu_device: WgpuDevice) -> Self {
        Self { buffer, wgpu_device }
    }

    pub async fn to_cpu_storage_async(&self) -> crate::Result<crate::CpuStorage> {
        let data = read_data_from_gpu_async(&self.wgpu_device, &self.buffer).await;
        return Ok(crate::CpuStorage::F32(data));
    }

}



impl crate::backend::BackendStorage for WgpuStorage{
    type Device = WgpuDevice;

    fn try_clone(&self, _: &crate::Layout) -> crate::Result<Self> {
        todo!()
    }

    fn dtype(&self) -> crate::DType {
        return crate::DType::F32; //only f32 is supported for now
    }

    fn device(&self) -> &Self::Device {
        return &self.wgpu_device;
    }


    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        return Err(Error::WebGpu("Sync copy to CpuStorage not allowed for WebGpu device. First copy the date asynchronously to a CpuStorage".to_owned().into()));
    }

    fn affine(&self, _: &crate::Layout, _: f64, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn powf(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        
        todo!()
    }

    fn elu(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn reduce_op(&self, _: crate::op::ReduceOp, _: &crate::Layout, _: &[usize]) -> crate::Result<Self> {
        todo!()
    }

    fn cmp(&self, _: crate::op::CmpOp, _: &Self, _: &crate::Layout, _: &crate::Layout) -> crate::Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &crate::Layout, _: crate::DType) -> crate::Result<Self> {
        panic!("Not Supported")
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), layout.shape().elem_count() * 4);
        let op = match B::NAME{
            //"gelu" => UnaryOperation::Gelu,
            //"erf" => UnaryOperation::Erf,
            "silu" => UnaryOperation::SiLu,
            "ceil" => UnaryOperation::Ceil,
            "floor" => UnaryOperation::Floor,
            //"round" => UnaryOperation::Round,
            //"gelu_erf" => UnaryOperation::GeluErf,
            "sign" => UnaryOperation::Sign,
            "abs" => UnaryOperation::Abs,

            "exp" => UnaryOperation::Exp,
            "log" => UnaryOperation::Log,
            "sin" => UnaryOperation::Sin,
            "cos" => UnaryOperation::Cos,
            "neg" => UnaryOperation::Neg,
            //"recip" => UnaryOperation::Recip,
            //"sqr" => UnaryOperation::Sqr,
            "sqrt" => UnaryOperation::Sqrt,
            "tanh" => UnaryOperation::Tanh,
            _ =>{panic!("Operation {} is not supported on wgpu", B::NAME)}
        };
        wgpu_functions::queue_unary_from_buffer_op(self.device(), &buffer_dest, &self.buffer, layout.shape().elem_count() as u32, op);
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone()));
    }

    fn binary_impl<B: crate::op::BinaryOpT>(&self, rhs: &Self, lhs_layout: &crate::Layout, _: &crate::Layout) -> crate::Result<Self> {
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
        
        wgpu_functions::queue_binary_buffer_from_buffer(self.device(), &buffer_dest, &self.buffer, &rhs.buffer, lhs_layout.shape().elem_count() as u32, op);
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone()));
    }

    fn where_cond(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: &Self, _: &crate::Layout) -> crate::Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn avg_pool2d(&self, _: &crate::Layout, _: (usize, usize), _: (usize, usize)) -> crate::Result<Self> {
        todo!()
    }

    fn max_pool2d(&self, _: &crate::Layout, _: (usize, usize), _: (usize, usize)) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn gather(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        todo!()
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
        todo!()
    }

    fn index_select(&self, _: &Self, _: &crate::Layout, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        todo!()
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
        todo!()
    }

    fn matmul(
        &self,
        rhs: &Self,
        (_batching,m,n,k): (usize, usize, usize, usize),
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = wgpu_functions::create_buffer(self.device(), (m * k) * 4);
        wgpu_functions::queue_matmul_buffer(self.device(), &buffer_dest, &self.buffer, &rhs.buffer, m as u32, n as u32, k as u32);
        return Ok(WgpuStorage::new(buffer_dest,self.device().clone()));
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> crate::Result<()> {
        todo!()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> crate::Result<()> {
        todo!()
    }
}