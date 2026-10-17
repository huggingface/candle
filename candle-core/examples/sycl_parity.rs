// SYCL vs CPU numerical parity probe for the ops a GGUF model exercises.
// Prints the max-abs-diff and cosine for each case; anything with cos < 0.9999
// is flagged `MISMATCH`.
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

fn cmp(name: &str, a: &Tensor, b: &Tensor) -> anyhow::Result<()> {
    let a = a
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?;
    let b = b
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?;
    let diff = (&a - &b)?.abs()?.max(0)?.to_scalar::<f64>()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f64>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();
    let cos = dot / (na * nb).max(1e-30);
    let flag = if cos < 0.9999 || !cos.is_finite() {
        "  <-- MISMATCH"
    } else {
        ""
    };
    println!("{name:<48} maxdiff={diff:<12.4e} cos={cos:.8}{flag}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_sycl(0)?;
    let cpu = Device::Cpu;

    // --- QMatMul: decode (m <= 8, fused mmvq) and prefill (dequant + gemm) ---
    for dt in [
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
    ] {
        for (n, k) in [(64usize, 1024usize), (64, 5120), (64, 17408), (5120, 17408)] {
            let w = Tensor::randn(0f32, 1.0, (n, k), &cpu)?;
            let qc = QTensor::quantize(&w, dt)?;
            let qs = QTensor::quantize_onto(&w, dt, &dev)?;
            let mc = QMatMul::from_qtensor(qc)?;
            let ms = QMatMul::from_qtensor(qs)?;
            // 1..13 exercise the fused mat-vec; 64 and 512 are past
            // `MMVQ_MAX_M` and so cover the dequantize-and-GEMM prefill path.
            for m in [1usize, 3, 8, 13, 64, 512] {
                let x = Tensor::randn(0f32, 1.0, (1, m, k), &cpu)?;
                let yc = mc.forward(&x)?;
                let ys = ms.forward(&x.to_device(&dev)?)?;
                cmp(&format!("qmatmul {dt:?} n={n} k={k} m={m}"), &yc, &ys)?;
                let ys16 = ms.forward(&x.to_dtype(DType::F16)?.to_device(&dev)?)?;
                cmp(&format!("qmatmul f16 {dt:?} n={n} k={k} m={m}"), &yc, &ys16)?;
            }
        }
    }

    // --- dequantize / embedding ---
    for dt in [
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::Q5K,
    ] {
        let w = Tensor::randn(0f32, 1.0, (1000, 5120), &cpu)?;
        let qc = QTensor::quantize(&w, dt)?;
        let qs = QTensor::quantize_onto(&w, dt, &dev)?;
        cmp(
            &format!("dequantize {dt:?}"),
            &qc.dequantize(&cpu)?,
            &qs.dequantize(&dev)?,
        )?;
        let ids = Tensor::new(&[[3u32, 999, 0], [512, 7, 7]], &cpu)?;
        cmp(
            &format!("embedding {dt:?}"),
            &qc.embedding(&ids)?,
            &qs.embedding(&ids.to_device(&dev)?)?,
        )?;
    }

    // --- repeat / cat with aliased inputs ---
    for (s, v) in [(1usize, 3usize), (7, 3), (1, 2)] {
        let t = Tensor::randn(0f32, 1.0, (1, s, 16, 128), &cpu)?;
        let ts = t.to_device(&dev)?;
        let rc = t.unsqueeze(2)?.repeat((1, 1, v, 1, 1))?.contiguous()?;
        let rs = ts.unsqueeze(2)?.repeat((1, 1, v, 1, 1))?.contiguous()?;
        cmp(&format!("repeat chunked s={s} v={v}"), &rc, &rs)?;
        let rc = t.unsqueeze(3)?.repeat((1, 1, 1, v, 1))?.contiguous()?;
        let rs = ts.unsqueeze(3)?.repeat((1, 1, 1, v, 1))?.contiguous()?;
        cmp(&format!("repeat interleaved s={s} v={v}"), &rc, &rs)?;
        for dtype in [DType::F16, DType::F32] {
            let t2 = t.to_dtype(dtype)?;
            let rc = t2.unsqueeze(2)?.repeat((1, 1, v, 1, 1))?.contiguous()?;
            let rs = t2
                .to_device(&dev)?
                .unsqueeze(2)?
                .repeat((1, 1, v, 1, 1))?
                .contiguous()?;
            cmp(&format!("repeat chunked {dtype:?} s={s} v={v}"), &rc, &rs)?;
        }
    }

    // --- GQA attention-style repeat_kv + matmul on strided views ---
    let k = Tensor::randn(0f32, 1.0, (1, 4, 9, 256), &cpu)?;
    let q = Tensor::randn(0f32, 1.0, (1, 24, 9, 256), &cpu)?;
    let run = |q: &Tensor, k: &Tensor| -> anyhow::Result<Tensor> {
        let (b, h, s, d) = k.dims4()?;
        let k = k
            .unsqueeze(2)?
            .expand((b, h, 6, s, d))?
            .reshape((b, h * 6, s, d))?;
        Ok(q.matmul(&k.transpose(2, 3)?)?)
    };
    cmp(
        "gqa expand+matmul",
        &run(&q, &k)?,
        &run(&q.to_device(&dev)?, &k.to_device(&dev)?)?,
    )?;

    // --- elementwise chain ---
    let a = Tensor::randn(0f32, 3.0, (1, 5, 48), &cpu)?;
    let bias = Tensor::randn(0f32, 1.0, 48, &cpu)?;
    let f = |a: &Tensor, bias: &Tensor| -> anyhow::Result<Tensor> {
        let x = a.broadcast_add(bias)?;
        Ok((x.exp()? + 1.0)?.log()?)
    };
    cmp(
        "softplus chain",
        &f(&a, &bias)?,
        &f(&a.to_device(&dev)?, &bias.to_device(&dev)?)?,
    )?;

    Ok(())
}
