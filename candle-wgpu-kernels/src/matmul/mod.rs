/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{Matmul1,Matmul7,Matmul1End,Matmul5}
impl crate::EntryPoint for Functions{
    fn get_entry_point(&self) -> &'static str{
        match self{
            Functions::Matmul1 => "ga",Functions::Matmul7 => "gd",Functions::Matmul1End => "gb",Functions::Matmul5 => "gc"
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
    