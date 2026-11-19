//! Regression test: binary ops on the Vulkan backend must slice the
//! rhs subbuffer at the SingleBlock start offset. The decode MLP splits a
//! fused gate/up GEMM output `(1, 1, 2*inter)` into two narrow views
//! `(1, 1, inter)`; at decode shape `contiguous()` is a no-op, so `up`
//! is a strided view (strides `(2*inter, 2*inter, 1)`, start `inter`).
//! `binary_impl` used to pass the whole rhs buffer to the kernel, which
//! then read the gate half instead of the up half (silent corruption,
//! first decode token garbage).
#![cfg(feature = "vulkan")]
use candle_core::{Device, Tensor};

fn silu_cpu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn binary_op_with_strided_rhs() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_vulkan(0)?;
    let inter = 9728usize;
    let total = 2 * inter;
    let mut data = vec![0f32; total];
    for (i, v) in data.iter_mut().enumerate() {
        *v = 0.5 + ((i % 97) as f32) * 0.01;
    }
    let f = Tensor::from_vec(data, (1, 1, total), &dev)?;
    let gate = f.narrow(2, 0, inter)?;
    let up = f.narrow(2, inter, inter)?;
    println!("gate contiguous? {}", gate.is_contiguous());
    println!("up contiguous? {}", up.is_contiguous());
    let g = gate.contiguous()?;
    let u = up.contiguous()?;
    println!("g dims {:?} strides {:?}", g.dims(), g.layout().stride());
    println!("u dims {:?} strides {:?}", u.dims(), u.layout().stride());
    let s = g.silu()?;
    let s_cpu = s.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let gated = (s * &u)?;
    let gated_cpu = gated
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let g_cpu = g.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let u_cpu = u.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let ds = s_cpu
        .iter()
        .zip(g_cpu.iter())
        .map(|(a, b)| (a - silu_cpu(*b)).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let dg = gated_cpu
        .iter()
        .zip(s_cpu.iter())
        .zip(u_cpu.iter())
        .map(|((a, b), c)| (a - *b * c).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("max |silu_gpu - silu_cpu| = {ds:.6e}");
    println!("max |mul_gpu - b*c| = {dg:.6e}");
    assert!(ds < 1e-6, "silu wrong");
    assert!(dg < 1e-5, "mul wrong");
    Ok(())
}
