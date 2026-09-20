//! Temporary microbenchmark: per-dispatch GPU cost for elementwise and
//! decode-shaped gemm. Not a permanent example.
use candle_core::{Device, Tensor};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Vulkan per-dispatch microbenchmark", long_about = None)]
struct Args {
    /// The Vulkan device ordinal.
    #[arg(short = 'g', long, default_value_t = 0)]
    gpu: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let args = Args::parse();
    let device = Device::new_vulkan(args.gpu)?;
    let n = 1024usize;
    let mut x = Tensor::arange(0f32, n as f32, &device)?;
    for _ in 0..8 {
        x = x.silu()?;
    }
    let n_ops = 2000;
    let t0 = std::time::Instant::now();
    for _ in 0..n_ops {
        x = x.silu()?;
    }
    let _v: Vec<f32> = x.to_vec1::<f32>()?;
    let dt = t0.elapsed();
    tracing::debug!(
        "vulkan_bench: {n_ops} silu dispatches ({n} elems): {:.3} ms/dispatch",
        dt.as_secs_f32() * 1000.0 / n_ops as f32
    );
    let a = Tensor::arange(0f32, 1024f32, &device)?.reshape((1, 1024))?;
    let w = Tensor::arange(0f32, 1048576f32, &device)?.reshape((1024, 1024))?;
    for _ in 0..8 {
        let _ = a.matmul(&w)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n_ops {
        let _ = a.matmul(&w)?;
    }
    // force one sync: materialize the last gemm's dependency chain via a
    // small readback of a derived value
    let probe = a.matmul(&w)?;
    let _v: Vec<f32> = probe.flatten(0, 1)?.to_vec1::<f32>()?;
    let dt = t0.elapsed();
    tracing::debug!(
        "vulkan_bench: {n_ops} gemm dispatches (1x1024 @ 1024x1024): {:.3} ms/dispatch",
        dt.as_secs_f32() * 1000.0 / (n_ops + 1) as f32
    );
    // gemv_t timing: (1,1024) @ (1024x1024) transposed view
    let wt_t = Tensor::arange(0f32, 1048576f32, &device)?.reshape((1024, 1024))?;
    for _ in 0..8 {
        let _ = a.matmul(&wt_t.t()?)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n_ops {
        let _ = a.matmul(&wt_t.t()?)?;
    }
    let probe_t = a.matmul(&wt_t.t()?)?;
    let _v: Vec<f32> = probe_t.flatten(0, 1)?.to_vec1::<f32>()?;
    let dt = t0.elapsed();
    tracing::debug!(
        "vulkan_bench: {n_ops} gemv_t dispatches (1x1024 @ 1024x1024^T): {:.3} ms/dispatch",
        dt.as_secs_f32() * 1000.0 / (n_ops + 1) as f32
    );
    // gemv_t shape sweep: the Qwen3-0.6B decode GEMV shapes.
    for &(ns, ks) in &[
        (512usize, 1024usize),
        (3072usize, 1024usize),
        (1024usize, 3072usize),
    ] {
        let a_s = Tensor::arange(0f32, ks as f32, &device)?.reshape((1, ks))?;
        let w_s = Tensor::arange(0f32, (ns * ks) as f32, &device)?.reshape((ns, ks))?;
        for _ in 0..4 {
            let _ = a_s.matmul(&w_s.t()?)?;
        }
        let t0 = std::time::Instant::now();
        let m_ops = 500usize;
        for _ in 0..m_ops {
            let _ = a_s.matmul(&w_s.t()?)?;
        }
        let probe_s = a_s.matmul(&w_s.t()?)?;
        let _v: Vec<f32> = probe_s.flatten(0, 1)?.to_vec1::<f32>()?;
        let dt = t0.elapsed();
        tracing::debug!(
            "vulkan_bench: {m_ops} gemv_t dispatches (1x{ks} @ {ns}x{ks}^T): {:.3} ms/dispatch",
            dt.as_secs_f32() * 1000.0 / (m_ops + 1) as f32
        );
    }
    // gemv_t correctness: a (k,) @ w.t() with w stored (n, k) row-major.
    // n/k are deliberately not multiples of 16/32 to exercise the guards.
    // Micro test: n=4, k=4, w_stride=4, known values.
    {
        let a: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let w: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let at = Tensor::from_vec(a.clone(), (1, 4), &device)?;
        let wt = Tensor::from_vec(w.clone(), (4, 4), &device)?;
        let out = at.matmul(&wt.t()?)?;
        let got = out
            .flatten(0, 1)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let want = [10f32, 26.0, 42.0, 58.0];
        tracing::debug!(
            "vulkan_bench: gemv_t micro got: {:?} want: {:?} ok: {}",
            got,
            want,
            got.iter().zip(want).all(|(g, x)| (g - x).abs() < 1e-4)
        );
    }
    let (n, k) = (3074usize, 1000usize);
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001f32).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| (((i * 7) % 13) as f32 - 6.0) * 0.01f32)
        .collect();
    let at = Tensor::from_vec(a.clone(), (1, k), &device)?;
    let wt = Tensor::from_vec(w.clone(), (n, k), &device)?;
    let out = at.matmul(&wt.t()?)?;
    let got = out
        .flatten(0, 1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let want: Vec<f32> = (0..n)
        .map(|ni| (0..k).map(|ki| a[ki] * w[ni * k + ki]).sum())
        .collect();
    let max_err = got
        .iter()
        .zip(want.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    tracing::debug!("vulkan_bench: gemv_t max_err: {max_err:.3e}");
    let bad = (0..n)
        .filter(|ni| (got[*ni] - want[*ni]).abs() > 1e-3)
        .collect::<Vec<_>>();
    tracing::debug!(
        "vulkan_bench: gemv_t bad: {}/{} first: {:?}",
        bad.len(),
        n,
        bad.iter().take(8).cloned().collect::<Vec<_>>()
    );
    for ni in [0usize, 1, 2, 15, 16, 17, 3072, 3073] {
        tracing::debug!("  n={ni} got={:.6} want={:.6}", got[ni], want[ni]);
    }
    // bisection: regular n + irregular k, and irregular n + regular k
    for &(nb, kb) in &[
        (3072usize, 64usize),
        (32usize, 1024usize),
        (32usize, 64usize),
        (256usize, 1024usize),
        (1024usize, 1024usize),
        (3072usize, 1024usize),
    ] {
        let ab: Vec<f32> = (0..kb).map(|i| (i as f32) * 0.001).collect();
        let wb: Vec<f32> = (0..nb * kb)
            .map(|i| (((i * 7) % 13) as f32 - 6.0) * 0.01)
            .collect();
        let atb = Tensor::from_vec(ab.clone(), (1, kb), &device)?;
        let wtb = Tensor::from_vec(wb.clone(), (nb, kb), &device)?;
        let outb2 = atb.matmul(&wtb.t()?)?;
        let gotb2 = outb2
            .flatten(0, 1)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let wantb2: Vec<f32> = (0..nb)
            .map(|ni| (0..kb).map(|ki| ab[ki] * wb[ni * kb + ki]).sum())
            .collect();
        let maxe = gotb2
            .iter()
            .zip(wantb2.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        tracing::debug!("vulkan_bench: gemv_t n={nb} k={kb} max_err: {maxe:.3e}");
    }
    // invertible case: a = e0 -> out[n] = w[n, 0]
    let (n2, k2) = (32usize, 64usize);
    let a2: Vec<f32> = (0..k2).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let w2: Vec<f32> = (0..n2 * k2).map(|i| (i as f32) * 0.001).collect();
    let at2 = Tensor::from_vec(a2.clone(), (1, k2), &device)?;
    let wt2b = Tensor::from_vec(w2.clone(), (n2, k2), &device)?;
    let outb = at2.matmul(&wt2b.t()?)?;
    let gotb = outb
        .flatten(0, 1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let wantb: Vec<f32> = (0..n2).map(|ni| w2[ni * k2]).collect();
    let badb = (0..n2)
        .filter(|ni| (gotb[*ni] - wantb[*ni]).abs() > 1e-6)
        .collect::<Vec<_>>();
    tracing::debug!(
        "vulkan_bench: gemv_t delta bad: {}/{} first: {:?}",
        badb.len(),
        n2,
        badb.iter().take(8).cloned().collect::<Vec<_>>()
    );
    for ni in 0..8usize {
        tracing::debug!("  n={ni} got={:.6} want={:.6}", gotb[ni], wantb[ni]);
    }
    // non-transposed gemv sanity (same data, w stored (k, n))
    let wt2 = Tensor::from_vec(w.clone(), (n, k), &device)?;
    let out2 = at.matmul(&wt2.t()?.contiguous()?)?;
    let got2 = out2
        .flatten(0, 1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let max_err2 = got2
        .iter()
        .zip(want.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    tracing::debug!("vulkan_bench: gemv max_err: {max_err2:.3e}");
    Ok(())
}
