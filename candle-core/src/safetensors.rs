use crate::{Device, Error, Result, Tensor, WithDType};
use safetensors::tensor as st;

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
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), c.as_mut_ptr() as *mut u8, v.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, view.shape(), device)
    }
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

// If Rust allowed for self-referential struct, we could store both the Mmap buffer and the
// SafeTensor bits in the same struct and avoid having the final users calling two methods.
// We could try using the ouroboros crate or equivalent for this at some point.
// Wrap the SafeTensors main module so as to provide accessors with the candle types for errors,
// dtypes, etc
pub struct SafeTensors<'a>(st::SafeTensors<'a>);

pub struct MmapedFile(memmap2::Mmap);

impl MmapedFile {
    pub fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let file = std::fs::File::open(p)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
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
