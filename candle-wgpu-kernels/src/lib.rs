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
mod generated;

pub use generated::Pipelines as Pipelines;
pub use generated::Shaders as Shaders;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType{
    F32,
    U32,
    U8
}
pub trait EntryPoint{
    fn get_entry_point(&self) -> &'static str;
}