use candle_core::{Device, Tensor};
fn main() -> anyhow::Result<()> {
    let device = Device::new_vulkan(0)?;
    let x = Tensor::arange(0f32, 16., &device)?.reshape((4, 4))?;
    let rhs = Tensor::new(&[10f32, 20., 30., 40.], &device)?.reshape((4, 1))?;
    let bm: Vec<f32> = x.broadcast_mul(&rhs)?.flatten(0, 1)?.to_vec1::<f32>()?;
    let want: Vec<f32> = (0..4u32)
        .flat_map(|i| (0..4u32).map(move |j| (i * 4 + j) as f32 * (i as f32 + 1.) * 10.))
        .collect();
    let ok = bm
        .iter()
        .zip(want.iter())
        .all(|(g, w)| (g - w).abs() <= 1e-4);
    if !ok {
        for (i, (g, w)) in bm.iter().zip(want.iter()).enumerate() {
            if (g - w).abs() > 1e-4 {
                eprintln!("FAIL element {i}: got {g}, want {w}");
            }
        }
        std::process::exit(1);
    }
    println!("vulkan_mul_min: OK");
    Ok(())
}
