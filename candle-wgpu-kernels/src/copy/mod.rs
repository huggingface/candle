/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{Copy3d,Copy3dPaddedNobatch,Copy2dTranspose2,Copy2d,Copy3dPadded,CopyStrided,Copy2dTranspose,Copy2d2,Copy}
impl crate::EntryPoint for Functions{
    fn get_entry_point(&self) -> &'static str{
        match self{
            Functions::Copy3d => "gg",Functions::Copy3dPaddedNobatch => "gi",Functions::Copy2dTranspose2 => "gf",Functions::Copy2d => "gc",Functions::Copy3dPadded => "gh",Functions::CopyStrided => "ga",Functions::Copy2dTranspose => "ge",Functions::Copy2d2 => "gd",Functions::Copy => "gb"
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
    