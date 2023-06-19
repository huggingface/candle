#[allow(dead_code)]
pub(crate) enum Storage {
    Cpu {
        dtype: crate::DType,
        buffer: Vec<u8>,
    },
}
