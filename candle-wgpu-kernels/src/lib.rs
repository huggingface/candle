mod generated;
pub use generated::kernels::*;
pub use generated::*;

impl Constants {
    pub fn get_const(i: usize) -> Constants {
        match i {
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
            //10 => Constants::Constv10,
            //11 => Constants::Constv11,
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "wgpu_debug_serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DType {
    F32,
    U32,
    U8,
    I64,
    F64
}
pub trait EntryPoint {
    fn get_entry_point(&self) -> &'static str;
}
