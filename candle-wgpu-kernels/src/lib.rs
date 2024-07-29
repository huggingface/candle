pub mod binary;
pub mod cmp;
pub mod conv1d;
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
pub mod testshader;
mod generated;

pub use generated::Pipelines as Pipelines;
pub use generated::Shaders as Shaders;
pub use generated::Constants as Constants;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType{
    F32,
    U32,
    U8
}
pub trait EntryPoint{
    fn get_entry_point(&self) -> &'static str;
}


// ///Helper Type MetaArray, for constructing the MetaBuffer
// #[derive(Debug)]
// pub struct MetaArray(Vec<u32>);

// #[derive(Debug)]
// pub struct ConstArray(HashMap<String, f64>);


// pub trait KernelParameterMeta{
//     fn write_meta(&self, meta : &mut MetaArray);
// }

// pub trait KernelParameterConsts{
//     fn write_consts(&self, _consts : &mut ConstArray){}
// }
// pub trait KernelParameter : KernelParameterMeta + KernelParameterConsts{
   
// }


// impl MetaArray {
//     pub fn new(capacity: u32) -> Self {
//         MetaArray(Vec::with_capacity(capacity as usize))
//     }

//     // pub(crate) fn add_layout(&mut self, layout: &Layout) {
//     //     let shape = layout.shape().dims();
//     //     let stride = layout.stride();
//     //     self.0.push(shape.len() as u32);
//     //     self.0.push(layout.start_offset() as u32);

//     //     if layout.is_contiguous() {
//     //         self.0.push(layout.shape().elem_count() as u32);
//     //     } else {
//     //         self.0.push(0);
//     //     }

//     //     self.0.extend(shape.iter().map(|&x| x as u32));
//     //     self.0.extend(stride.iter().map(|&x| x as u32));
//     // }

//     // pub(crate) fn add_layout2(&mut self, layout: &Layout) {
//     //     let shape = layout.shape().dims();
//     //     let stride = layout.stride();
//     //     self.0.push(layout.start_offset() as u32);

//     //     if layout.is_contiguous() {
//     //         self.0.push(layout.shape().elem_count() as u32);
//     //     } else {
//     //         self.0.push(0);
//     //     }

//     //     self.0.extend(shape.iter().map(|&x| x as u32));
//     //     self.0.extend(stride.iter().map(|&x| x as u32));
//     // }

//     // pub(crate) fn add<T: ToU32>(&mut self, value: T) {
//     //     self.0.push(value.to_u32());
//     // }

//     pub fn add<T : KernelParameterMeta>(&mut self, value : T){
//         value.write_meta(self);
//     }
// }



// impl ConstArray {
//     pub fn new() -> Self {
//         ConstArray(HashMap::new())
//     }

//     // pub(crate) fn add<T: ToU32>(&mut self, value: T) {
//     //     self.0.push(value.to_u32());
//     // }

//     pub fn add<T : KernelParameterConsts>(&mut self, value : T){
//         value.write_consts(self);
//     }

//     pub fn insert<ST : Into<String>, T : ToF64>(&mut self, key : ST, value : T){
//         self.0.insert(key.into(), value.to_f64());
//     }
// }


// // impl<T : ToF64> KernelParameterConsts for Option<T>{
// //     fn write_consts(&self, consts : &mut crate::ConstArray){
// //         if let Some(value) = &self{
// //             consts.add(value.to_f64());
// //         }
// //     }
// // }
