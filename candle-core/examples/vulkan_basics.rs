#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) -> Result<()> {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if (g - w).abs() > tol {
            bail!("{what}: element {i}: got {g}, want {w}");
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let device = Device::new_vulkan(gpu)?;

    // `affine` dispatches the vulkan affine compute kernel (y = 2x + 1).
    let x = Tensor::arange(0f32, 16., &device)?.reshape((4, 4))?;
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

    // `silu` dispatches the unary elementwise kernel.
    let s: Vec<f32> = x.silu()?.flatten(0, 1)?.to_vec1::<f32>()?;
    let want: Vec<f32> = (0..16)
        .map(|i| {
            let v = i as f32;
            v / (1. + (-v).exp())
        })
        .collect();
    assert_close(&s, &want, 1e-4, "silu")?;

    // `broadcast_mul` dispatches the binary elementwise kernel; the rhs
    // (4, 1) broadcast view exercises the dim-1 broadcast path.
    let rhs = Tensor::new(&[10f32, 20., 30., 40.], &device)?.reshape((4, 1))?;
    let bm: Vec<f32> = x.broadcast_mul(&rhs)?.flatten(0, 1)?.to_vec1::<f32>()?;
    let want: Vec<f32> = (0..4u32)
        .flat_map(|i| (0..4u32).map(move |j| (i * 4 + j) as f32 * (i as f32 + 1.) * 10.))
        .collect();
    assert_close(&bm, &want, 1e-4, "broadcast_mul")?;

    // 2D `matmul` (4, 8) @ (8, 4) dispatches the gemm kernel.
    let a = Tensor::arange(0f32, 32., &device)?.reshape((4, 8))?;
    let b = Tensor::arange(0f32, 32., &device)?.reshape((8, 4))?;
    let c: Vec<f32> = a.matmul(&b)?.flatten(0, 1)?.to_vec1::<f32>()?;
    let want: Vec<f32> = a
        .to_device(&Device::Cpu)?
        .matmul(&b.to_device(&Device::Cpu)?)?
        .flatten(0, 1)?
        .to_vec1::<f32>()?;
    assert_close(&c, &want, 1e-3, "matmul 2d")?;

    // 4D `matmul` with a transposed rhs, the attention pattern:
    // (1, 2, 3, 4) @ (1, 2, 4, 3)^T -> (1, 2, 3, 3).
    let a4 = Tensor::arange(0f32, 24., &device)?.reshape((1, 2, 3, 4))?;
    let b4 = Tensor::arange(0f32, 24., &device)?
        .reshape((1, 2, 3, 4))?
        .transpose(2, 3)?;
    let c4: Vec<f32> = a4.matmul(&b4)?.flatten(0, 3)?.to_vec1::<f32>()?;
    let want: Vec<f32> = a4
        .to_device(&Device::Cpu)?
        .matmul(&b4.to_device(&Device::Cpu)?)?
        .flatten(0, 3)?
        .to_vec1::<f32>()?;
    assert_close(&c4, &want, 1e-3, "matmul 4d transposed")?;

    println!("vulkan_basics: affine, reduce, elementwise, gemm verified on {device:?}");
    Ok(())
}
