mod generated;
pub use generated::*;

impl Constants{
    pub fn get_const(i : usize)->Constants{
        match i{
            0 => Constants::Constv0,
            1 => Constants::Constv1,
            2 => Constants::Constv2,
            3 => Constants::Constv3,
            4 => Constants::Constv4,
            5 => Constants::Constv5,
            6 => Constants::Constv6,
            7 => Constants::Constv7,
            8 => Constants::Constv8,
            9 => Constants::Constv9,
            _ => todo!()
        }
    }
}

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
