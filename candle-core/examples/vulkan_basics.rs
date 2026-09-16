#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let x = Tensor::arange(0f32, 16., &device)?;
    let x = x.reshape((4, 4))?;
    // `affine` dispatches the vulkan affine compute kernel (y = 2x + 1).
    let y = x.affine(2., 1.)?;
    // `sum` over the last axis dispatches the reduce_sum kernel.
    let sums = y.sum((1,))?;
    let y: Vec<f32> = y.reshape((16,))?.to_vec1::<f32>()?;
    let sums: Vec<f32> = sums.to_vec1::<f32>()?;
    assert_eq!(
        y,
        vec![1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25., 27., 29., 31.]
    );
    assert_eq!(sums, vec![16., 48., 80., 112.]);
    println!("vulkan_basics: affine + reduce_sum verified on {device:?}");
    Ok(())
}
