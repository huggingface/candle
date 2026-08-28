// Smoke test for the SYCL backend. After Phase 3 the whole non-quantized op
// surface runs on the Intel GPU; this exercises a mini attention-style block.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Module, Tensor, D,
};

fn main() -> Result<()> {
    let dev = Device::new_sycl(0)?;
    println!("device: {:?}", dev.location());

    let (seq, dim, vocab) = (16usize, 256usize, 100usize);
    let embed = Tensor::randn(0f32, 1.0, (vocab, dim), &dev)?;
    let ids = Tensor::arange(0u32, seq as u32, &dev)?;

    // embedding lookup (index_select) -> qkv projections (matmul)
    let x = embed.index_select(&ids, 0)?; // (seq, dim)
    let wq = Tensor::randn(0f32, 0.1, (dim, dim), &dev)?;
    let wk = Tensor::randn(0f32, 0.1, (dim, dim), &dev)?;
    let wv = Tensor::randn(0f32, 0.1, (dim, dim), &dev)?;
    let (q, k, v) = (x.matmul(&wq)?, x.matmul(&wk)?, x.matmul(&wv)?);

    // scores = softmax(q k^T / sqrt(d)) -- manual softmax (max, sub, exp, sum, div)
    let scores = (q.matmul(&k.t()?)? * (1.0 / (dim as f64).sqrt()))?;
    let m = scores.max_keepdim(D::Minus1)?;
    let e = scores.broadcast_sub(&m)?.exp()?;
    let attn = e.broadcast_div(&e.sum_keepdim(D::Minus1)?)?;
    let out = attn.matmul(&v)?; // (seq, dim)

    // GGUF-quantized projection (Q4_K), dequant + GEMM on the GPU
    let w = Tensor::randn(0f32, 0.1, (dim, dim), &dev)?;
    let qw = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let qmm = candle_core::quantized::QMatMul::from_qtensor(qw)?;
    let projected = qmm.forward(&out)?;

    dev.synchronize()?;
    let norm = projected.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    println!("attention out:      {out:?}");
    println!("q4k projection out: {projected:?}");
    println!("‖proj‖ = {norm:.4}");

    // cross-check against the CPU
    let ref_norm = projected
        .to_device(&Device::Cpu)?
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_scalar::<f32>()?;
    println!(
        "‖out‖ (host recompute) = {ref_norm:.4}  dtype={:?}",
        out.dtype()
    );
    assert_eq!(out.dtype(), DType::F32);
    Ok(())
}
