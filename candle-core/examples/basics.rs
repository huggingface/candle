#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{CpuStorage, Device, Layout, Shape, Tensor};
use candle_core as candle;

struct ArgSort;
impl candle::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "arg-sort"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle::Result<(CpuStorage, Shape)> {
        if layout.shape().rank() != 1 {
            candle::bail!(
                "input should have a single dimension, got {:?}",
                layout.shape()
            )
        }
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        let mut dst = (0..src.len() as u32).collect::<Vec<u32>>();
        dst.sort_by(|&i, &j| src[i as usize].total_cmp(&src[j as usize]));
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }
}

fn main() -> Result<()> {
    let a = Tensor::new(&[0.0f32, 1.0, 3.0, 2.0, -12.0, 4.0, 3.5], &Device::Cpu)?;
    let indices = a.apply_op1(ArgSort)?;
    let a_sorted = a.gather(&indices, 0)?;
    println!("{indices}");
    println!("{a_sorted}");
    Ok(())
}
