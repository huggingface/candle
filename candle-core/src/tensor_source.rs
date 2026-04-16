/// A source of raw tensor bytes that can be held without materialising a typed `Vec<T>` on the heap.
///
/// The two required methods are `data_u8` (raw, contiguous bytes) and `data_type`.
/// `element_count` is a derived convenience.
///
/// Since shape is not part of the trait callers that need it must access it on
/// the concrete type before erasing to `dyn TensorSource`.
pub trait TensorSource: Send + Sync + std::fmt::Debug {
    /// Raw, contiguous bytes of the tensor data in row-major order.
    fn data_u8(&self) -> &[u8];
    /// Element data type. Avoids fn name collision with `dtype()`.
    fn data_type(&self) -> crate::DType;
    /// Number of elements — derived by default.
    fn element_count(&self) -> usize {
        self.data_u8().len() / self.data_type().size_in_bytes()
    }
}
