//! F8E4M3 correctness on ROCm, checked bit-for-bit against the CPU backend.
//!
//! gfx1101 has no fp8 hardware: every kernel here converts to f32 in software,
//! computes, and rounds back. The point of comparing bit patterns rather than
//! approximate values is that the *encoding* is what has to match — HIP's
//! `__hip_fp8_e4m3` is OCP E4M3 like NVIDIA's, not the AMD `_fnuz` variant, and
//! a wrong typedef in `hip_shim/cuda_fp8.h` would show up here first.

use super::tests::rocm_device;
use crate::{DType, Device, Result, Tensor};
use float8::F8E4M3;

fn fp8(v: &[f32]) -> Vec<F8E4M3> {
    v.iter().copied().map(F8E4M3::from_f32).collect()
}

/// Raw 8-bit payloads, flattened. Comparing these rather than `to_vec` values
/// catches a sign-of-zero or NaN-encoding difference that `==` would hide.
fn bits(t: &Tensor) -> Result<Vec<u8>> {
    Ok(t.flatten_all()?
        .to_vec1::<F8E4M3>()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// Values spanning the format: subnormals, the exact powers of two, a value
/// needing round-to-nearest-even, and the largest finite magnitude (448).
const VALUES: &[f32] = &[
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    3.5,
    -3.5,
    7.0,
    0.001953125, // smallest positive subnormal
    -0.001953125,
    0.017578125,
    448.0, // largest finite
    -448.0,
    0.25,
    6.0,
];

/// The same tensor on the CPU and on the GPU.
fn pair(dev: &Device, v: &[f32]) -> Result<(Tensor, Tensor)> {
    let vals = fp8(v);
    Ok((
        Tensor::from_slice(&vals, v.len(), &Device::Cpu)?,
        Tensor::from_slice(&vals, v.len(), dev)?,
    ))
}

/// Round trip through the device. Fails if `storage_from_slice` /
/// `to_cpu_storage` disagree about the element type — which they would if the
/// slice still shared `u8`'s.
#[test]
fn f8e4m3_host_round_trip_is_exact() -> Result<()> {
    let dev = rocm_device!();
    let (c, g) = pair(&dev, VALUES)?;
    assert_eq!(bits(&g)?, bits(&c)?);
    assert_eq!(g.dtype(), DType::F8E4M3);
    Ok(())
}

/// `unary.cu`'s `*_fp8_e4m3` kernels, which were unreachable twice over: gated
/// behind `__CUDA_ARCH__ >= 890` in the source, and refused by `Map1` before
/// that.
///
/// Only ops the CPU backend computes the same way are compared bit-for-bit —
/// one conversion out, the maths in a wider type, one rounding back. `gelu`
/// and `silu` are excluded deliberately: the CPU backend evaluates those as a
/// chain of `F8E4M3` arithmetic, rounding to 3 mantissa bits at every step,
/// while the kernel rounds once at the end. Neither is wrong; they are just not
/// the same function.
#[test]
fn f8e4m3_unary_ops_match_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let (c, g) = pair(&dev, VALUES)?;
    // `neg` also exercises the sign of zero.
    assert_eq!(bits(&g.neg()?)?, bits(&c.neg()?)?, "neg");
    assert_eq!(bits(&g.abs()?)?, bits(&c.abs()?)?, "abs");
    assert_eq!(bits(&g.sqr()?)?, bits(&c.sqr()?)?, "sqr");
    assert_eq!(bits(&g.ceil()?)?, bits(&c.ceil()?)?, "ceil");
    assert_eq!(bits(&g.floor()?)?, bits(&c.floor()?)?, "floor");
    assert_eq!(bits(&g.round()?)?, bits(&c.round()?)?, "round");
    assert_eq!(bits(&g.relu()?)?, bits(&c.relu()?)?, "relu");
    assert_eq!(bits(&g.sign()?)?, bits(&c.sign()?)?, "sign");

    // Transcendentals, on inputs whose f32 and f64 results cannot straddle an
    // fp8 rounding boundary. Zero is excluded from `recip`; see
    // `f8e4m3_overflow_to_infinity_diverges_from_cuda`.
    let (c, g) = pair(&dev, &[1.0, -1.0, 0.5, 2.0, -2.0, 0.25])?;
    assert_eq!(bits(&g.recip()?)?, bits(&c.recip()?)?, "recip");
    let (c, g) = pair(&dev, &[0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 0.25])?;
    assert_eq!(bits(&g.exp()?)?, bits(&c.exp()?)?, "exp");
    assert_eq!(bits(&g.sin()?)?, bits(&c.sin()?)?, "sin");
    assert_eq!(bits(&g.cos()?)?, bits(&c.cos()?)?, "cos");
    assert_eq!(bits(&g.tanh()?)?, bits(&c.tanh()?)?, "tanh");
    let (c, g) = pair(&dev, &[1.0, 4.0, 0.25, 16.0, 0.5])?;
    assert_eq!(bits(&g.sqrt()?)?, bits(&c.sqrt()?)?, "sqrt");
    assert_eq!(bits(&g.log()?)?, bits(&c.log()?)?, "log");
    Ok(())
}

/// `binary.cu`'s fp8 arithmetic. These were already compiled for ROCm — they
/// sit behind the 800 gate, not 890 — but `Map2` refused the dtype, so nothing
/// had ever launched them.
#[test]
fn f8e4m3_binary_ops_match_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let lhs = &[1.0, -2.0, 0.5, 6.0, -0.25, 448.0, 0.0, 3.5];
    let rhs = &[2.0, 3.0, -0.5, 0.25, 4.0, 2.0, 1.0, -3.5];
    let (cl, gl) = pair(&dev, lhs)?;
    let (cr, gr) = pair(&dev, rhs)?;

    assert_eq!(bits(&(&gl + &gr)?)?, bits(&(&cl + &cr)?)?, "add");
    assert_eq!(bits(&(&gl - &gr)?)?, bits(&(&cl - &cr)?)?, "sub");
    assert_eq!(bits(&(&gl * &gr)?)?, bits(&(&cl * &cr)?)?, "mul");
    assert_eq!(bits(&(&gl / &gr)?)?, bits(&(&cl / &cr)?)?, "div");
    assert_eq!(
        bits(&gl.maximum(&gr)?)?,
        bits(&cl.maximum(&cr)?)?,
        "maximum"
    );
    assert_eq!(
        bits(&gl.minimum(&gr)?)?,
        bits(&cl.minimum(&cr)?)?,
        "minimum"
    );

    // 448 + 448 overflows the format. The kernel must saturate to the largest
    // finite value the way the CPU backend does, not wrap or produce NaN.
    let (c, g) = pair(&dev, &[448.0, -448.0])?;
    assert_eq!(bits(&(&g + &g)?)?, bits(&(&c + &c)?)?, "saturating add");
    Ok(())
}

/// `BINARY_OP_OUT` writes `u8` whatever the input dtype, so this also checks
/// that `Map2Any` re-tags the output rather than carrying F8E4M3 through.
#[test]
fn f8e4m3_comparisons_match_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let lhs = &[1.0, -2.0, 0.5, 0.0, -0.0, 448.0];
    let rhs = &[1.0, 3.0, -0.5, 0.0, 0.0, 2.0];
    let (cl, gl) = pair(&dev, lhs)?;
    let (cr, gr) = pair(&dev, rhs)?;

    for (name, g, c) in [
        ("eq", gl.eq(&gr)?, cl.eq(&cr)?),
        ("ne", gl.ne(&gr)?, cl.ne(&cr)?),
        ("lt", gl.lt(&gr)?, cl.lt(&cr)?),
        ("le", gl.le(&gr)?, cl.le(&cr)?),
        ("gt", gl.gt(&gr)?, cl.gt(&cr)?),
        ("ge", gl.ge(&gr)?, cl.ge(&cr)?),
    ] {
        assert_eq!(g.dtype(), DType::U8, "{name} dtype");
        assert_eq!(g.to_vec1::<u8>()?, c.to_vec1::<u8>()?, "{name}");
    }
    Ok(())
}

/// `affine.cu`'s `affine_f8_e4m3` and `unary.cu`'s `upowf`/`uelu`, all three of
/// which take a scalar and all three of which were gated at 890.
#[test]
fn f8e4m3_scalar_ops_match_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let (c, g) = pair(&dev, &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 4.0])?;

    assert_eq!(
        bits(&g.affine(2.0, 1.0)?)?,
        bits(&c.affine(2.0, 1.0)?)?,
        "affine"
    );
    // The scalars themselves are rounded to fp8 before the kernel runs, so a
    // non-representable multiplier is the interesting case.
    assert_eq!(
        bits(&g.affine(0.5, -0.25)?)?,
        bits(&c.affine(0.5, -0.25)?)?,
        "affine fractional"
    );
    assert_eq!(bits(&g.powf(2.0)?)?, bits(&c.powf(2.0)?)?, "powf");
    assert_eq!(bits(&g.elu(1.0)?)?, bits(&c.elu(1.0)?)?, "elu");
    Ok(())
}

/// `ternary.cu`'s `where_*_fp8_e4m3`, reached through `Map3`.
#[test]
fn f8e4m3_where_cond_matches_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let (ct, gt) = pair(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let (cf, gf) = pair(&dev, &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])?;
    let mask = &[1u8, 0, 1, 1, 0, 0];
    let cm = Tensor::from_slice(mask, 6, &Device::Cpu)?;
    let gm = Tensor::from_slice(mask, 6, &dev)?;

    assert_eq!(
        bits(&gm.where_cond(&gt, &gf)?)?,
        bits(&cm.where_cond(&ct, &cf)?)?
    );
    Ok(())
}

/// `indexing.cu`'s fp8 kernels: `is_*`, `gather_*`, `ia_*` and `sa_*`, all
/// gated at 890.
#[test]
fn f8e4m3_indexing_matches_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let vals = fp8(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = Tensor::from_slice(&vals, (3, 2), &Device::Cpu)?;
    let g = Tensor::from_slice(&vals, (3, 2), &dev)?;
    let ids = &[2u32, 0, 1];
    let ci = Tensor::from_slice(ids, 3, &Device::Cpu)?;
    let gi = Tensor::from_slice(ids, 3, &dev)?;

    assert_eq!(
        bits(&g.index_select(&gi, 0)?)?,
        bits(&c.index_select(&ci, 0)?)?,
        "index_select"
    );

    let gather_ids = &[0u32, 1, 1, 0, 1, 0];
    let cg = Tensor::from_slice(gather_ids, (3, 2), &Device::Cpu)?;
    let gg = Tensor::from_slice(gather_ids, (3, 2), &dev)?;
    assert_eq!(
        bits(&g.gather(&gg, 1)?)?,
        bits(&c.gather(&cg, 1)?)?,
        "gather"
    );

    // `index_add` accumulates in fp8, so it exercises the read-modify-write in
    // `IA_OP_F8` rather than a plain store.
    let src = fp8(&[1.0, 1.0, 2.0, 2.0, 4.0, 4.0]);
    let cs = Tensor::from_slice(&src, (3, 2), &Device::Cpu)?;
    let gs = Tensor::from_slice(&src, (3, 2), &dev)?;
    assert_eq!(
        bits(&g.index_add(&gi, &gs, 0)?)?,
        bits(&c.index_add(&ci, &cs, 0)?)?,
        "index_add"
    );
    assert_eq!(
        bits(&g.scatter_add(&gg, &gs, 1)?)?,
        bits(&c.scatter_add(&cg, &cs, 1)?)?,
        "scatter_add"
    );
    Ok(())
}

/// `indexing.cu` has `SA_OP_F8` but no fp8 `S_OP`, so scatter-add works and
/// scatter-set has no kernel to launch. The error must say which, not report a
/// missing driver symbol.
#[test]
fn f8e4m3_scatter_set_reports_the_missing_kernel() -> Result<()> {
    let dev = rocm_device!();
    let vals = fp8(&[1.0, 2.0, 3.0, 4.0]);
    let g = Tensor::from_slice(&vals, (2, 2), &dev)?;
    let ids = Tensor::from_slice(&[0u32, 1, 1, 0], (2, 2), &dev)?;
    let err = g.scatter(&ids, &g, 1).unwrap_err().to_string();
    assert!(err.contains("indexing.cu"), "unexpected error: {err}");
    assert!(err.contains("scatter-add"), "unexpected error: {err}");
    Ok(())
}

/// A known, deliberate divergence from the CUDA backend.
///
/// E4M3 has no infinity. NVIDIA's `__nv_fp8_e4m3(float)` uses SATFINITE and
/// clamps an infinite input to +/-448, which is also what the `float8` crate
/// (and so the CPU backend) does. HIP's conversion encodes it as NaN instead.
/// Only reachable by producing an infinity in the f32 the kernel computes in —
/// `1/0` here — since fp8's own overflow already saturates. Pinned rather than
/// worked around: papering over it means replacing HIP's fp8 type with a
/// hand-written one in the shim, for a case no real workload depends on.
#[test]
fn f8e4m3_overflow_to_infinity_diverges_from_cuda() -> Result<()> {
    let dev = rocm_device!();
    let (c, g) = pair(&dev, &[0.0])?;
    assert_eq!(bits(&g.recip()?)?, vec![0x7f], "HIP encodes inf as NaN");
    assert_eq!(bits(&c.recip()?)?, vec![0x7e], "CPU saturates inf to 448");
    // Saturation *within* the format still agrees: only an already-infinite f32
    // differs.
    let (c, g) = pair(&dev, &[448.0])?;
    assert_eq!(
        bits(&(&g * &g)?)?,
        bits(&(&c * &c)?)?,
        "448 * 448 saturates"
    );
    Ok(())
}

/// `cast.cu` ships fp8 conversions in both directions; `to_dtype` used to
/// reject the dtype outright because `DType::as_str` spells it `f8e4m3` where
/// the kernels spell it `f8_e4m3`.
#[test]
fn f8e4m3_casts_match_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let (c, g) = pair(&dev, VALUES)?;

    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let from_gpu = g.to_dtype(dtype)?;
        assert_eq!(from_gpu.dtype(), dtype);
        // Widening from fp8 is exact, so a value comparison is meaningful here.
        assert_eq!(
            from_gpu.to_dtype(DType::F64)?.to_vec1::<f64>()?,
            c.to_dtype(dtype)?.to_dtype(DType::F64)?.to_vec1::<f64>()?,
            "f8e4m3 -> {dtype:?}"
        );
        // And back, which is where the rounding happens.
        assert_eq!(
            bits(&from_gpu.to_dtype(DType::F8E4M3)?)?,
            bits(&c.to_dtype(dtype)?.to_dtype(DType::F8E4M3)?)?,
            "{dtype:?} -> f8e4m3"
        );
    }

    // Rounding into fp8 from values that are not representable.
    let raw = &[0.3f32, 1.1, -1.1, 2.4, 2.6, 500.0, -500.0, 0.0001];
    let cr = Tensor::from_slice(raw, raw.len(), &Device::Cpu)?;
    let gr = Tensor::from_slice(raw, raw.len(), &dev)?;
    assert_eq!(
        bits(&gr.to_dtype(DType::F8E4M3)?)?,
        bits(&cr.to_dtype(DType::F8E4M3)?)?,
        "f32 -> f8e4m3 rounding"
    );
    Ok(())
}

/// `reduce.cu` leaves every fp8 instantiation commented out upstream, so there
/// is no kernel to launch at any `__CUDA_ARCH__`. The error has to say that
/// rather than blame the dtype dispatch.
#[test]
fn f8e4m3_reductions_report_the_missing_kernels() -> Result<()> {
    let dev = rocm_device!();
    let (_, g) = pair(&dev, &[1.0, 2.0, 3.0, 4.0])?;
    let err = g.sum_all().unwrap_err().to_string();
    assert!(err.contains("reduce.cu"), "unexpected error: {err}");
    assert!(err.contains("F8E4M3"), "unexpected error: {err}");
    Ok(())
}

/// `to_dtype` between fp8 and a dtype `cast.cu` never pairs it with must name
/// the source, the destination and the reason.
#[test]
fn f8e4m3_unsupported_casts_name_the_pair() -> Result<()> {
    let dev = rocm_device!();
    let (_, g) = pair(&dev, &[1.0, 2.0])?;
    let err = g.to_dtype(DType::U32).unwrap_err().to_string();
    assert!(err.contains("F8E4M3"), "unexpected error: {err}");
    assert!(err.contains("U32"), "unexpected error: {err}");
    assert!(err.contains("cast.cu"), "unexpected error: {err}");
    Ok(())
}
