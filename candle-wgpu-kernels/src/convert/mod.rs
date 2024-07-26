/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{ConvertU8ToF32,ConvertF32ToU8,ConvertU32ToU8,ConvertToF32,ConvertToU32}
impl crate::EntryPoint for Functions{
    fn get_entry_point(&self) -> &'static str{
        match self{
            Functions::ConvertU8ToF32 => "gA",Functions::ConvertF32ToU8 => "gB",Functions::ConvertU32ToU8 => "gC",Functions::ConvertToF32 => "gy",Functions::ConvertToU32 => "gx"
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
    