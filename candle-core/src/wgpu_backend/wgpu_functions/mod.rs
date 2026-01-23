pub mod binary;
pub mod cmp;
pub mod conv2d;
pub mod convert;
pub mod copy;
pub mod gather;
pub mod index_select;
pub mod matmul;
pub mod pool2d;
pub mod reduce;
pub mod rms_norm;
pub mod softmax;
pub mod unary;
pub mod upsample;
pub mod where_cond;
pub mod rotary_emb;

use wgpu_compute_layer::{
    cache::{
        BindgroupAlignmentLayout, BufferReferenceId
    }, queue_buffer::{OpIsInplaceable, PipelineReference, QueueBuffer}, shader_loader::PipelineIndex, util::ToU32
};

use super::WgpuDevice;

use wgpu_compute_layer::cache::BindgroupAlignment;

pub use candle_wgpu_kernels::DType;
pub use candle_wgpu_kernels::Pipelines;
use candle_wgpu_kernels::Constants;

use crate::Layout;

/**************** FUNCTIONS ****************/ 
pub use binary::queue_binary_buffer_from_buffer;
pub use cmp::queue_cmp_buffer_from_buffer;
pub use conv2d::{queue_conv1d, queue_conv1d_transpose, queue_conv2d, queue_conv2d_transpose};
pub use convert::{
    queue_convert, queue_convert_f32_to_u8,
    queue_convert_u32_to_u8, queue_convert_u8_to_f32,
    queue_convert_f32_to_f16, queue_convert_f16_to_f32
};
pub use copy::{queue_copy, queue_copy2d, queue_copy3d, queue_copy3d_padded, queue_copy_strided, queue_transpose3d};
pub use gather::{queue_gather, queue_index_add_inplace, queue_scatter_add_inplace, queue_scatter_set_inplace};
pub use index_select::queue_index_select;
pub use matmul::queue_matmul_buffer;
pub use pool2d::{queue_avg_pool2d, queue_max_pool2d};
pub use reduce::queue_reduce_from_buffer_op;
pub use rms_norm::{queue_rms_norm, queue_layer_norm};
pub use softmax::queue_softmax;
pub use unary::{queue_unary_from_buffer_op, queue_unary_inplace_op};
pub use upsample::{queue_upsample1d, queue_upsample2d, queue_upsample_bilinear2d};
pub use where_cond::queue_where_cond;
pub use rotary_emb::{queue_rotary_emb_i, queue_rotary_emb_c, queue_rotary_emb_thd};

#[derive(Debug, Copy, Clone)]
pub struct WgpuTensor<'a>{
    layout : &'a Layout,
    buffer : BufferReferenceId,
}

impl<'a> WgpuTensor<'a> {
    pub fn new(layout: &'a Layout, buffer: BufferReferenceId) -> Self {
        Self { layout, buffer }
    }
    
    pub fn layout(&self) -> &Layout {
        self.layout
    }
    
    pub fn buffer(&self) -> BufferReferenceId {
        self.buffer
    }
}


impl From<crate::DType> for wgpu_compute_layer::cache::BindgroupAlignment {
    fn from(val: crate::DType) -> Self {
        let wgpu_type : wgpu_compute_layer::DType = val.into();
        wgpu_type.into() 
    }
}

pub(crate) trait QueueLayouts{
    fn add_layout1(&mut self, layout: &crate::Layout);
    fn add_layout2(&mut self, layout: &crate::Layout);
    fn add_layout3(&mut self, layout: &crate::Layout);
    fn add_layout1_non_contiguous(&mut self, layout: &crate::Layout);
    fn add_layout2_non_contiguous(&mut self, layout: &crate::Layout);
    fn add_layout3_non_contiguous(&mut self, layout: &crate::Layout);
    fn get_pipeline_const<T: ToU32>(
        &mut self,
        pipeline: impl Into<PipelineIndex>,
        const_vec: Vec<T>,
    ) -> PipelineReference;

    fn get_pipeline_const_inplace<T: ToU32>(
        &mut self,
        pipeline: impl Into<PipelineIndex>,
        const_vec: Vec<T>,
        inplaceable: OpIsInplaceable,
    ) -> PipelineReference;
        
}

fn add_layout<'a>(
    queue: &mut QueueBuffer<'a>,
    layout: &Layout,
    is_contiguous: bool,
    constant_dims: Constants,
    constant_is_startofsset_zero: Constants,
    constant_is_contiguous: Constants,
) {
    let shape = layout.shape().dims();
    let stride = layout.stride();

    queue.add_const(constant_dims, shape.len());
    if layout.start_offset() != 0 {
        queue.add_const(constant_is_startofsset_zero, false);
        queue.add(layout.start_offset());
    }

    if is_contiguous {
        queue.add(layout.shape().elem_count());
    } else {
        queue.add_const(constant_is_contiguous, false);

        queue.get_meta_mut().extend(shape.iter().map(|&x| x as u32));
        queue.get_meta_mut().extend(stride.iter().map(|&x| x as u32));
    }
}

impl<'a> QueueLayouts for QueueBuffer<'a>
{
    fn add_layout1(&mut self, layout: &crate::Layout) {
        add_layout(self,
           layout,
           layout.is_contiguous(),
           Constants::ConstDims1,
           Constants::ConstIsStartoffsetZero1,
           Constants::ConstIsContiguous1,
       );
    }

    fn add_layout2(&mut self, layout: &crate::Layout) {
        add_layout(self,
           layout,
           layout.is_contiguous(),
           Constants::ConstDims2,
           Constants::ConstIsStartoffsetZero2,
           Constants::ConstIsContiguous2,
       );
    }

    fn add_layout3(&mut self, layout: &crate::Layout) {
        add_layout(self,
           layout,
           layout.is_contiguous(),
           Constants::ConstDims3,
           Constants::ConstIsStartoffsetZero3,
           Constants::ConstIsContiguous3,
       );
    }
    
    fn add_layout1_non_contiguous(&mut self, layout: &crate::Layout) {
        add_layout(self,
           layout,
           false,
           Constants::ConstDims1,
           Constants::ConstIsStartoffsetZero1,
           Constants::ConstIsContiguous1,
       );
    }

    fn add_layout2_non_contiguous(&mut self, layout: &crate::Layout) {
        add_layout(self,
           layout,
           false,
           Constants::ConstDims2,
           Constants::ConstIsStartoffsetZero2,
           Constants::ConstIsContiguous2,
       );
    }

    fn add_layout3_non_contiguous(&mut self, layout: &crate::Layout) {
        add_layout(self,
           layout,
           false,
           Constants::ConstDims3,
           Constants::ConstIsStartoffsetZero3,
           Constants::ConstIsContiguous3,
       );
    }

    
    fn get_pipeline_const<T: ToU32>(
        &mut self,
        pipeline: impl Into<PipelineIndex>,
        const_vec: Vec<T>,
    ) -> PipelineReference {
        for (index, v) in const_vec.into_iter().enumerate() {
            self.add_const(candle_wgpu_kernels::Constants::get_const(index), v);
        }
        self.get_pipeline(pipeline)
    }

    fn get_pipeline_const_inplace<T: ToU32>(
        &mut self,
        pipeline: impl Into<PipelineIndex>,
        const_vec: Vec<T>,
        inplaceable: OpIsInplaceable,
    ) -> PipelineReference {
        for (index, v) in const_vec.into_iter().enumerate() {
            self.add_const(candle_wgpu_kernels::Constants::get_const(index), v);
        }

        self.get_pipeline_inplace(pipeline, inplaceable)
    }


}