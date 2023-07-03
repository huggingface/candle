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
