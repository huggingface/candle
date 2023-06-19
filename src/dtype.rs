#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}
