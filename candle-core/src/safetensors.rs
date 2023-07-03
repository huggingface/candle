use crate::{DType, Device, Error, Result, Tensor, WithDType};
use safetensors::tensor as st;
use std::borrow::Cow;

impl From<DType> for st::Dtype {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => st::Dtype::U8,
            DType::U32 => st::Dtype::U32,
            DType::BF16 => st::Dtype::BF16,
            DType::F16 => st::Dtype::F16,
            DType::F32 => st::Dtype::F32,
            DType::F64 => st::Dtype::F64,
        }
    }
}

impl TryFrom<st::Dtype> for DType {
    type Error = Error;
    fn try_from(value: st::Dtype) -> Result<Self> {
        match value {
            st::Dtype::U8 => Ok(DType::U8),
            st::Dtype::U32 => Ok(DType::U32),
            st::Dtype::BF16 => Ok(DType::BF16),
            st::Dtype::F16 => Ok(DType::F16),
            st::Dtype::F32 => Ok(DType::F32),
            st::Dtype::F64 => Ok(DType::F64),
            dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
        }
    }
}

impl st::View for Tensor {
    fn dtype(&self) -> st::Dtype {
        self.dtype().into()
    }
    fn shape(&self) -> &[usize] {
        self.shape().dims()
    }

    fn data(&self) -> Cow<[u8]> {
        // This copies data from GPU to CPU.
        convert_back(self).unwrap()
    }

    fn data_len(&self) -> usize {
        let n: usize = self.shape().elem_count();
        let bytes_per_element = self.dtype().size_in_bytes();
        n * bytes_per_element
    }
}

impl Tensor {
    pub fn save_safetensors<P: AsRef<std::path::Path>>(
        &self,
        name: &str,
        filename: P,
    ) -> Result<()> {
        let data = [(name, self.clone())];
        Ok(st::serialize_to_file(data, &None, filename.as_ref())?)
    }
}

fn convert_<T: WithDType>(view: st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    let v = view.data();
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = v.len() / size_in_bytes;
    if (v.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, view.shape(), device)
    } else {
        let mut c = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), c.as_mut_ptr() as *mut u8, v.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, view.shape(), device)
    }
}

fn convert_back_<T: WithDType>(value: Cow<'_, [T]>) -> Cow<'_, [u8]> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    // SAFETY:
    //
    // Every T is larger than u8, so there is no issue regarding alignment.
    // This is safe only because we explicitly take the lifetime from the Cow's lifetime
    // and consume the original Cow.
    // This means that borrowed Cow, will keep their lifetime information, preventing
    // this slice from being accessed after freeing the original memory.
    let slice = unsafe {
        std::slice::from_raw_parts(value.as_ptr() as *const u8, value.len() * size_in_bytes)
    };
    Cow::Borrowed(slice)
}

pub fn convert(view: st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    match view.dtype() {
        st::Dtype::U8 => convert_::<u8>(view, device),
        st::Dtype::U32 => convert_::<u8>(view, device),
        st::Dtype::BF16 => convert_::<half::bf16>(view, device),
        st::Dtype::F16 => convert_::<half::f16>(view, device),
        st::Dtype::F32 => convert_::<f32>(view, device),
        st::Dtype::F64 => convert_::<f64>(view, device),
        dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
    }
}

pub fn convert_back(tensor: &Tensor) -> Result<Cow<[u8]>> {
    match tensor.dtype() {
        DType::U8 => Ok(convert_back_::<u8>(tensor.storage_data()?)),
        DType::U32 => Ok(convert_back_::<u32>(tensor.storage_data()?)),
        DType::F16 => Ok(convert_back_::<half::f16>(tensor.storage_data()?)),
        DType::BF16 => Ok(convert_back_::<half::bf16>(tensor.storage_data()?)),
        DType::F32 => Ok(convert_back_::<f32>(tensor.storage_data()?)),
        DType::F64 => Ok(convert_back_::<f64>(tensor.storage_data()?)),
    }
}

// If Rust allowed for self-referential struct, we could store both the Mmap buffer and the
// SafeTensor bits in the same struct and avoid having the final users calling two methods.
// We could try using the ouroboros crate or equivalent for this at some point.
// Wrap the SafeTensors main module so as to provide accessors with the candle types for errors,
// dtypes, etc
pub struct SafeTensors<'a>(st::SafeTensors<'a>);

pub struct MmapedFile(memmap2::Mmap);

impl MmapedFile {
    /// Creates a wrapper around a memory mapped file from which you can retrieve
    /// tensors using [`MmapedFile::deserialize`]
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let file = std::fs::File::open(p)?;
        let mmap = memmap2::MmapOptions::new().map(&file)?;
        Ok(Self(mmap))
    }

    pub fn deserialize(&self) -> Result<SafeTensors<'_>> {
        let st = safetensors::SafeTensors::deserialize(&self.0)?;
        Ok(SafeTensors(st))
    }
}

impl<'a> SafeTensors<'a> {
    pub fn tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        convert(self.0.tensor(name)?, device)
    }

    pub fn tensors(&self, device: &Device) -> Result<Vec<(String, Tensor)>> {
        self.0
            .tensors()
            .into_iter()
            .map(|(name, tensor_view)| {
                let tensor = convert(tensor_view, device)?;
                Ok((name, tensor))
            })
            .collect()
    }

    pub fn names(&self) -> Vec<&String> {
        self.0.names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn save_single_tensor() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        t.save_safetensors("t", "t.safetensors").unwrap();
        let bytes = std::fs::read("t.safetensors").unwrap();
        assert_eq!(bytes, b"@\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       \0\0\0\0");
        std::fs::remove_file("t.safetensors").unwrap();
    }

    #[test]
    fn save_multiple_tensors() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let map: HashMap<_, _> = [("t", t), ("u", u)].into_iter().collect();
        st::serialize_to_file(map, &None, std::path::Path::new("multi.safetensors")).unwrap();
        let bytes = std::fs::read("multi.safetensors").unwrap();
        assert_eq!(bytes, b"x\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"u\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}      \0\0\0\0\0\0\0\0");
        std::fs::remove_file("multi.safetensors").unwrap();
    }
}
