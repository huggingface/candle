/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{ConvertF32ToU8,ConvertU32ToU8,ConvertToU32,ConvertToF32,ConvertU8ToF32}
impl crate::EntryPoint for Functions{
    fn get_entry_point(&self) -> &'static str{
        match self{
            Functions::ConvertF32ToU8 => "gb",Functions::ConvertU32ToU8 => "gd",Functions::ConvertToU32 => "ga",Functions::ConvertToF32 => "gc",Functions::ConvertU8ToF32 => "ge"
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
    