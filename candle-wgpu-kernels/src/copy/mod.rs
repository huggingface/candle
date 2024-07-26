/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{Copy,Copy2d,Copy3d,Copy2dTranspose,Copy2d2,Copy3dPadded,CopyStrided,Copy2dTranspose2,Copy3dPaddedNobatch}
impl crate::EntryPoint for Functions{
    fn get_entry_point(&self) -> &'static str{
        match self{
            Functions::Copy => "gD",Functions::Copy2d => "gE",Functions::Copy3d => "gP",Functions::Copy2dTranspose => "gN",Functions::Copy2d2 => "gI",Functions::Copy3dPadded => "gZ",Functions::CopyStrided => "gy",Functions::Copy2dTranspose2 => "gO",Functions::Copy3dPaddedNobatch => "gaa"
        }
    } 
}
pub fn load_shader(typ : crate::DType) -> &'static str {
    match typ{
        crate::DType::F32 => include_str!("generated/shader.pwgsl_generated_f32.wgsl"),
        crate::DType::U32 => include_str!("generated/shader.pwgsl_generated_u32.wgsl"),
        crate::DType::U8 => include_str!("generated/shader.pwgsl_generated_u8.wgsl"),
    }
}
    