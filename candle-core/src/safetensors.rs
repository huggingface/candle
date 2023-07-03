use crate::{Device, Result, Tensor};
use half::f16;
use safetensors::tensor as st;

pub fn convert(view: st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    match view.dtype() {
        st::Dtype::F16 => {
            let v = view.data();
            if (v.as_ptr() as usize) % 2 == 0 {
                // SAFETY This is safe because we just checked that this
                // was correctly aligned.
                let data: &[f16] =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f16, v.len() / 2) };
                Tensor::from_slice(data, view.shape(), device)
            } else {
                let mut c = Vec::with_capacity(v.len() / 2);
                let mut i = 0;
                while i < v.len() {
                    c.push(f16::from_le_bytes([v[i], v[i + 1]]));
                    i += 2;
                }
                Tensor::from_slice(&c, view.shape(), device)
            }
        }
        dt => todo!("Unhandled dtype {dt:?}"),
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
