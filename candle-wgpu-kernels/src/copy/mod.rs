/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{Copy2dTranspose,CopyStrided,Copy3dPaddedNobatch,Copy3d,Copy2d,Copy,Copy2dTranspose2,Copy3dPadded,Copy2d2}
impl crate::EntryPoint for Functions{
    fn get_entry_point(&self) -> &'static str{
        match self{
            Functions::Copy2dTranspose => "ge",Functions::CopyStrided => "ga",Functions::Copy3dPaddedNobatch => "gi",Functions::Copy3d => "gg",Functions::Copy2d => "gc",Functions::Copy => "gb",Functions::Copy2dTranspose2 => "gf",Functions::Copy3dPadded => "gh",Functions::Copy2d2 => "gd"
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
    