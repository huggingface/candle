//! Integration tests for the Vulkan compute kernels (`candle-vulkan-kernels`).
//!
//! Every test regenerates its inputs with the deterministic SplitMix64
//! `Rng` below (seed `1000 + test index`, bit-for-bit identical to the
//! numpy generator `gen_expected.py`), dispatches the kernel through
//! `VulkanDevice`, and compares the output against the hardcoded
//! expectations in `vulkan_expected.rs` (leading elements plus whole-output
//! sum-of-squares and max-abs, computed by numpy with the exact operation
//! order of each shader).
//!
//! Device selection:
//! `CANDLE_VULKAN_TEST_GPU=<n> cargo test --features vulkan --test vulkan_integration_tests`
//! (defaults to physical device 0).

#![cfg(feature = "vulkan")]

mod vulkan_expected;

use vulkan_expected::*;

use candle_core::{Device, Result};
use candle_core::VulkanDevice;
use candle_vulkan_kernels as vk;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

// --------------------------------------------------------------------- RNG
//
// SplitMix64, bit-for-bit identical to the `Rng` in gen_expected.py
// (wrapping 64-bit arithmetic matches python's explicit masking).

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// f32 in `[lo, hi)`, f32 arithmetic throughout (matches numpy).
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 32) as f32 / 4294967296.0;
        lo + (hi - lo) * u
    }

    fn f32s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(lo, hi)).collect()
    }

    fn u32_mod(&mut self, m: u32) -> u32 {
        (self.next_u64() >> 32) as u32 % m
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() & 0xFF) as u8).collect()
    }
}

// --------------------------------------------------------------- helpers

fn vulkan_dev() -> Result<Device> {
    let gpu = std::env::var("CANDLE_VULKAN_TEST_GPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Device::new_vulkan(gpu)
}

/// Executes `encode` on the device, waits for completion, and downloads
/// `out`.
fn run_and_get(
    d: &VulkanDevice,
    out: &Subbuffer<[f32]>,
    encode: impl FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
        -> std::result::Result<(), String>
        + Send
        + 'static,
) -> Result<Vec<f32>> {
    d.execute(encode)?;
    d.synchronize()?;
    d.download_f32(out)
}

/// Compares a kernel output against a generated expectation.
///
/// Tolerances absorb f32 rounding-order differences (pairwise vs serial
/// accumulation, driver FMA contraction); real kernel bugs (wrong scale,
/// nibble, stride, or zero output) are orders of magnitude outside them.
fn check(name: &str, got: &[f32], exp: &Exp, len: usize) {
    assert_eq!(got.len(), len, "{name}: output length");
    for (i, w) in exp.head.iter().enumerate() {
        let g = got[i];
        let tol = 1e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{name}: head[{i}]: got {g}, want {w}");
    }
    let sumsq: f64 = got.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let rel = (sumsq - exp.sumsq).abs() / exp.sumsq.max(1.0);
    assert!(
        rel <= 1e-3,
        "{name}: sumsq {sumsq:.6} vs expected {} (rel err {rel:.2e})",
        exp.sumsq
    );
    let maxabs = got.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));
    let rel = (maxabs - exp.maxabs).abs() / exp.maxabs.max(1.0);
    assert!(
        rel <= 1e-3,
        "{name}: maxabs {maxabs:.6} vs expected {} (rel err {rel:.2e})",
        exp.maxabs
    );
}

// ------------------------------------------------------------------ tests

#[test]
fn affine() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1000);
    let x = rng.f32s(256, -4.0, 4.0);
    let i = vd.upload_f32(&x)?;
    let o = vd.new_f32_buffer(256)?;
    let (i2, o2, k) = (i.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_affine_f32(cbb, &k, &i2, &o2, 2.5, -1.75).map_err(|e| e.to_string())
    })?;
    check("affine", &got, &AFFINE, 256);
    Ok(())
}

#[test]
fn copy() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1001);
    let src = rng.f32s(64, -8.0, 8.0);
    let isrc = vd.upload_f32(&src)?;
    let idst = vd.new_f32_buffer(40)?;
    let (s2, t2, k) = (isrc.clone(), idst.clone(), vd.kernels());
    let got = run_and_get(vd, &idst, move |cbb| {
        vk::call_copy_f32(cbb, &k, &s2, &t2, 32, &[4, 8], &[16, 1], &[10, 1], 0, 2)
            .map_err(|e| e.to_string())
    })?;
    check("copy", &got, &COPY, 40);
    Ok(())
}

fn elem_binary(op: vk::KernelName, name: &str, seed: u64, div: bool) -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(seed);
    let lhs = rng.f32s(128, -2.0, 2.0);
    let rhs = if div { rng.f32s(8, 0.5, 1.5) } else { rng.f32s(8, -1.0, 1.0) };
    let il = vd.upload_f32(&lhs)?;
    let ir = vd.upload_f32(&rhs)?;
    let o = vd.new_f32_buffer(128)?;
    let (l2, r2, o2, k) = (il.clone(), ir.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_elem_binary_f32(cbb, &k, op, &l2, &r2, &o2, &[16, 8], &[8])
            .map_err(|e| e.to_string())
    })?;
    let exp = match name {
        "ELEM_ADD" => &ELEM_ADD,
        "ELEM_SUB" => &ELEM_SUB,
        "ELEM_MUL" => &ELEM_MUL,
        _ => &ELEM_DIV,
    };
    check(name, &got, exp, 128);
    Ok(())
}

#[test]
fn elem_add() -> Result<()> {
    elem_binary(vk::KernelName::ElemAddF32, "ELEM_ADD", 1002, false)
}

#[test]
fn elem_sub() -> Result<()> {
    elem_binary(vk::KernelName::ElemSubF32, "ELEM_SUB", 1003, false)
}

#[test]
fn elem_mul() -> Result<()> {
    elem_binary(vk::KernelName::ElemMulF32, "ELEM_MUL", 1004, false)
}

#[test]
fn elem_div() -> Result<()> {
    elem_binary(vk::KernelName::ElemDivF32, "ELEM_DIV", 1005, true)
}

fn elem_unary(op: vk::KernelName, name: &str, seed: u64, lo: f32, hi: f32) -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(seed);
    let x = rng.f32s(128, lo, hi);
    let i = vd.upload_f32(&x)?;
    let o = vd.new_f32_buffer(128)?;
    let (i2, o2, k) = (i.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_elem_unary_f32(cbb, &k, op, &i2, &o2).map_err(|e| e.to_string())
    })?;
    let exp = match name {
        "ELEM_SIGMOID" => &ELEM_SIGMOID,
        "ELEM_SILU" => &ELEM_SILU,
        "ELEM_EXP" => &ELEM_EXP,
        "ELEM_SQRT" => &ELEM_SQRT,
        "ELEM_SIN" => &ELEM_SIN,
        "ELEM_COS" => &ELEM_COS,
        _ => &ELEM_NEG,
    };
    check(name, &got, exp, 128);
    Ok(())
}

#[test]
fn elem_sigmoid() -> Result<()> {
    elem_unary(vk::KernelName::ElemSigmoidF32, "ELEM_SIGMOID", 1006, -3.0, 3.0)
}

#[test]
fn elem_silu() -> Result<()> {
    elem_unary(vk::KernelName::ElemSiluF32, "ELEM_SILU", 1007, -3.0, 3.0)
}

#[test]
fn elem_exp() -> Result<()> {
    elem_unary(vk::KernelName::ElemExpF32, "ELEM_EXP", 1008, -2.0, 2.0)
}

#[test]
fn elem_sqrt() -> Result<()> {
    elem_unary(vk::KernelName::ElemSqrtF32, "ELEM_SQRT", 1009, 0.0, 9.0)
}

#[test]
fn elem_sin() -> Result<()> {
    elem_unary(vk::KernelName::ElemSinF32, "ELEM_SIN", 1010, -3.0, 3.0)
}

#[test]
fn elem_cos() -> Result<()> {
    elem_unary(vk::KernelName::ElemCosF32, "ELEM_COS", 1011, -3.0, 3.0)
}

#[test]
fn elem_neg() -> Result<()> {
    elem_unary(vk::KernelName::ElemNegF32, "ELEM_NEG", 1012, -3.0, 3.0)
}

#[test]
fn gather() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1013);
    let ids: Vec<u32> = (0..12).map(|_| rng.u32_mod(32)).collect();
    let emb = rng.f32s(32 * 16, -2.0, 2.0);
    let iids = vd.upload_u32(&ids)?;
    let iemb = vd.upload_f32(&emb)?;
    let o = vd.new_f32_buffer(12 * 16)?;
    let (i2, e2, o2, k) = (iids.clone(), iemb.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_gather_f32(cbb, &k, &i2, &e2, &o2, 12, 16).map_err(|e| e.to_string())
    })?;
    check("gather", &got, &GATHER, 12 * 16);
    Ok(())
}

#[test]
fn gemm() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1014);
    let (bsz, m, k, n) = (2usize, 32, 60, 40);
    let lhs = rng.f32s(bsz * m * k, -1.0, 1.0);
    let rhs = rng.f32s(bsz * k * n, -1.0, 1.0);
    let rhs_t = rng.f32s(bsz * n * k, -1.0, 1.0);
    let il = vd.upload_f32(&lhs)?;
    let ir = vd.upload_f32(&rhs)?;
    let irt = vd.upload_f32(&rhs_t)?;
    let o1 = vd.new_f32_buffer(bsz * m * n)?;
    let o2 = vd.new_f32_buffer(bsz * m * n)?;
    let (l2, r2, rt2, o12, o22, kern) =
        (il.clone(), ir.clone(), irt.clone(), o1.clone(), o2.clone(), vd.kernels());
    vd.execute(move |cbb| {
        vk::call_gemm_f32(cbb, &kern, &l2, &r2, &o12, bsz, m, n, k, false)
            .map_err(|e| e.to_string())?;
        vk::call_gemm_f32(cbb, &kern, &l2, &rt2, &o22, bsz, m, n, k, true).map_err(|e| e.to_string())
    })?;
    vd.synchronize()?;
    let g1 = vd.download_f32(&o1)?;
    let g2 = vd.download_f32(&o2)?;
    check("gemm_nt", &g1, &GEMM_NT, bsz * m * n);
    check("gemm_t", &g2, &GEMM_T, bsz * m * n);
    Ok(())
}

#[test]
fn gemv() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1015);
    let (k, n) = (64usize, 100);
    let a = rng.f32s(k, -1.0, 1.0);
    let w = rng.f32s(k * n, -1.0, 1.0);
    let ia = vd.upload_f32(&a)?;
    let iw = vd.upload_f32(&w)?;
    let o = vd.new_f32_buffer(n)?;
    let (a2, w2, o2, kern) = (ia.clone(), iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_gemv_f32(cbb, &kern, &a2, &w2, &o2, k, n).map_err(|e| e.to_string())
    })?;
    check("gemv", &got, &GEMV, n);
    Ok(())
}

#[test]
fn gemv_t() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1016);
    let (n, k) = (96usize, 128);
    let a = rng.f32s(k, -1.0, 1.0);
    let w = rng.f32s(n * k, -1.0, 1.0);
    let ia = vd.upload_f32(&a)?;
    let iw = vd.upload_f32(&w)?;
    let o = vd.new_f32_buffer(n)?;
    let (a2, w2, o2, kern) = (ia.clone(), iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_gemv_t_f32(cbb, &kern, &a2, &w2, &o2, n, k, k).map_err(|e| e.to_string())
    })?;
    check("gemv_t", &got, &GEMV_T, n);
    Ok(())
}

#[test]
fn q4k_qmatvec() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1017);
    let (k, n) = (512usize, 8);
    let a = rng.f32s(k, -1.0, 1.0);
    let mut wb = rng.bytes(n * 2 * 144);
    let p = Q4K_DPATCH[0];
    for b in 0..n * 2 {
        let o = b * 144;
        wb[o] = p.0;
        wb[o + 1] = p.1;
        wb[o + 2] = p.2;
        wb[o + 3] = p.3;
    }
    let ia = vd.upload_f32(&a)?;
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(n)?;
    let (a2, w2, o2, kern) = (ia.clone(), iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q4k_qmatvec_f32(cbb, &kern, &a2, &w2, &o2, k, n).map_err(|e| e.to_string())
    })?;
    check("q4k_qmatvec", &got, &Q4K_QMATVEC, n);
    Ok(())
}

#[test]
fn q4k_dequant() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1018);
    let mut wb = rng.bytes(3 * 144);
    for (b, p) in Q4K_DPATCH.iter().enumerate() {
        let o = b * 144;
        wb[o..o + 4].copy_from_slice(&[p.0, p.1, p.2, p.3]);
    }
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(3 * 256)?;
    let (w2, o2, k) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q4k_dequant_f32(cbb, &k, &w2, &o2, 3 * 256).map_err(|e| e.to_string())
    })?;
    check("q4k_dequant", &got, &Q4K_DEQUANT, 3 * 256);
    Ok(())
}

#[test]
fn q6k_dequant() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1019);
    let mut wb = rng.bytes(2 * 210);
    for (b, p) in Q6K_DPATCH.iter().enumerate() {
        let o = b * 210 + 208;
        wb[o] = p.0;
        wb[o + 1] = p.1;
    }
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(2 * 256)?;
    let (w2, o2, k) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q6k_dequant_f32(cbb, &k, &w2, &o2, 2 * 256).map_err(|e| e.to_string())
    })?;
    check("q6k_dequant", &got, &Q6K_DEQUANT, 2 * 256);
    Ok(())
}

#[test]
fn reduce_sum() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1020);
    let x = rng.f32s(32 * 64, -1.0, 1.0);
    let i = vd.upload_f32(&x)?;
    let o = vd.new_f32_buffer(32)?;
    let (i2, o2, k) = (i.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_reduce_sum_f32(cbb, &k, &i2, &o2, 32, 64).map_err(|e| e.to_string())
    })?;
    check("reduce_sum", &got, &REDUCE_SUM, 32);
    Ok(())
}

#[test]
fn reduce_max() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1021);
    let x = rng.f32s(32 * 64, -1.0, 1.0);
    let i = vd.upload_f32(&x)?;
    let o = vd.new_f32_buffer(32)?;
    let (i2, o2, k) = (i.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_reduce_max_f32(cbb, &k, &i2, &o2, 32, 64).map_err(|e| e.to_string())
    })?;
    check("reduce_max", &got, &REDUCE_MAX, 32);
    Ok(())
}

#[test]
fn rms_norm() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1022);
    let x = rng.f32s(16 * 64, -2.0, 2.0);
    let w = rng.f32s(64, 0.5, 1.5);
    let i = vd.upload_f32(&x)?;
    let iw = vd.upload_f32(&w)?;
    let o = vd.new_f32_buffer(16 * 64)?;
    let (i2, w2, o2, k) = (i.clone(), iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_rms_norm_f32(cbb, &k, &i2, &w2, &o2, 16, 64, 1e-5).map_err(|e| e.to_string())
    })?;
    check("rms_norm", &got, &RMS_NORM, 16 * 64);
    Ok(())
}

#[test]
fn softmax() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1023);
    let x = rng.f32s(24 * 56, -5.0, 5.0);
    let i = vd.upload_f32(&x)?;
    let o = vd.new_f32_buffer(24 * 56)?;
    let (i2, o2, k) = (i.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_softmax_last_dim_f32(cbb, &k, &i2, &o2, 24, 56).map_err(|e| e.to_string())
    })?;
    check("softmax", &got, &SOFTMAX, 24 * 56);
    Ok(())
}

#[test]
fn rope() -> Result<()> {
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1024);
    let (b, h, t, dh) = (2usize, 4, 8, 16);
    let x = rng.f32s(b * h * t * dh, -2.0, 2.0);
    let cos = rng.f32s(t * dh / 2, -1.0, 1.0);
    let sin = rng.f32s(t * dh / 2, -1.0, 1.0);
    let cos_u = rng.f32s(b * t * dh / 2, -1.0, 1.0);
    let sin_u = rng.f32s(b * t * dh / 2, -1.0, 1.0);
    let ix = vd.upload_f32(&x)?;
    let ic = vd.upload_f32(&cos)?;
    let is = vd.upload_f32(&sin)?;
    let icu = vd.upload_f32(&cos_u)?;
    let isu = vd.upload_f32(&sin_u)?;
    let o1 = vd.new_f32_buffer(b * h * t * dh)?;
    let o2 = vd.new_f32_buffer(b * h * t * dh)?;
    let (x2, c2, s2, cu2, su2, o12, o22, k) = (
        ix.clone(),
        ic.clone(),
        is.clone(),
        icu.clone(),
        isu.clone(),
        o1.clone(),
        o2.clone(),
        vd.kernels(),
    );
    vd.execute(move |cbb| {
        vk::call_rope_f32(cbb, &k, &x2, &c2, &s2, &o12, b, h, t, dh, false)
            .map_err(|e| e.to_string())?;
        vk::call_rope_f32(cbb, &k, &x2, &cu2, &su2, &o22, b, h, t, dh, true)
            .map_err(|e| e.to_string())
    })?;
    vd.synchronize()?;
    let g1 = vd.download_f32(&o1)?;
    let g2 = vd.download_f32(&o2)?;
    check("rope", &g1, &ROPE, b * h * t * dh);
    check("rope_u", &g2, &ROPE_U, b * h * t * dh);
    Ok(())
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32 * 0x8000_0000;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let (mut e, mut m) = (1u32, mant);
        while m < 0x200 {
            m <<= 1;
            e += 1;
        }
        f32::from_bits(sign | ((112 - e) << 23) | ((m & 0x1FF) << 13))
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

fn dequant_q4k_cpu(w: &[u8]) -> Vec<f32> {
    let nb = w.len() / 144;
    let mut out = vec![0.0f32; nb * 256];
    for b in 0..nb {
        let blk = &w[b * 144..(b + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let s = &blk[4..16];
        let mut sc = [0.0f32; 8];
        let mut mn = [0.0f32; 8];
        for j in 0..8 {
            sc[j] = if j < 4 {
                (s[j] & 63) as f32
            } else {
                ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)) as f32
            };
            mn[j] = if j < 4 {
                (s[j + 4] & 63) as f32
            } else {
                ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as f32
            };
        }
        for e in 0..256 {
            let sub = e / 64;
            let r = e % 64;
            let t = r % 32;
            let half = r / 32;
            let pair = sub * 2 + half;
            let byte = blk[16 + 32 * sub + t];
            let qval = ((byte >> (4 * half)) & 0xF) as f32;
            out[b * 256 + e] = d * sc[pair] * qval - dmin * mn[pair];
        }
    }
    out
}

#[test]
fn u8_buffer_roundtrip() -> Result<()> {
    // u8 upload/download round-trip (the Q4_K/Q6_K weights path).
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(4020);
    let wb = rng.bytes(3 * 144);
    let iw = vd.upload_u8(&wb)?;
    let rt = vd.download_u8(&iw)?;
    assert_eq!(rt, wb, "u8 round-trip mismatch");
    Ok(())
}

#[test]
fn q4k_dequant_vs_cpu() -> Result<()> {
    // Kernel vs an independent Rust port of ggml's dequantize_row_q4_K,
    // computed at test time (catches kernel/reference layout or formula
    // drift that baked expectations would hide).
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1018);
    let mut wb = rng.bytes(3 * 144);
    for (b, p) in Q4K_DPATCH.iter().enumerate() {
        let o = b * 144;
        wb[o..o + 4].copy_from_slice(&[p.0, p.1, p.2, p.3]);
    }
    let cpu = dequant_q4k_cpu(&wb);
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(3 * 256)?;
    let (w2, o2, kern) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q4k_dequant_f32(cbb, &kern, &w2, &o2, 3 * 256).map_err(|e| e.to_string())
    })?;
    for (i, (g, w)) in got.iter().zip(cpu.iter()).enumerate() {
        let tol = 1e-5 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "q4k_dequant vs cpu: [{i}] got {g}, want {w}");
    }
    Ok(())
}

#[test]
fn q4k_qmatvec_vs_cpu() -> Result<()> {
    // Q4_K GEMV vs an independent CPU dot-product reference.
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let k = 512usize;
    let n = 8usize;
    let mut rng = Rng::new(4021);
    let a: Vec<f32> = (0..k).map(|_| rng.f32(-1.0, 1.0)).collect();
    let mut wb = rng.bytes((k / 256) * n * 144);
    for (r, p) in Q4K_DPATCH.iter().enumerate() {
        let o = r * 144;
        wb[o..o + 4].copy_from_slice(&[p.0, p.1, p.2, p.3]);
    }
    let mut cpu = vec![0.0f32; n];
    for r in 0..n {
        let deq = dequant_q4k_cpu(&wb[r * 2 * 144..(r + 1) * 2 * 144]);
        let mut s = 0.0f32;
        for i in 0..k {
            s += a[i] * deq[i];
        }
        cpu[r] = s;
    }
    let ia = vd.upload_f32(&a)?;
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(n)?;
    let (a2, w2, o2, kern) = (ia.clone(), iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q4k_qmatvec_f32(cbb, &kern, &a2, &w2, &o2, k, n).map_err(|e| e.to_string())
    })?;
    for (i, (g, w)) in got.iter().zip(cpu.iter()).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "q4k_qmatvec vs cpu: [{i}] got {g}, want {w}");
    }
    Ok(())
}
// ------------------------------------------------- quantized dequant CPU
//
// Q8_0 / Q5_0 / Q5_K dequant kernels vs candle's own CPU dequantization
// (`GgmlType::to_float`, the same reference the GGUF loader uses on the
// CPU backend). The d/dmin f16 bytes are pinned to finite values: raw
// random bytes have a ~3% chance per f16 of landing on NaN/Inf.

use candle_core::quantized::k_quants::{BlockQ5K, BlockQ5_0, BlockQ8_0, GgmlType};

/// Reinterprets `w` as `T` blocks in an allocation aligned for `T`
/// (`Vec<u8>` allocations carry no alignment guarantee).
fn blocks_aligned<T: Clone>(w: &[u8]) -> Vec<T> {
    let n = w.len() / std::mem::size_of::<T>();
    let layout = std::alloc::Layout::from_size_align(
        n * std::mem::size_of::<T>(),
        std::mem::align_of::<T>(),
    )
    .unwrap();
    unsafe {
        let ptr = std::alloc::alloc(layout);
        std::ptr::copy_nonoverlapping(w.as_ptr(), ptr as *mut u8, w.len());
        let v: Vec<T> = std::slice::from_raw_parts(ptr as *const T, n).to_vec();
        std::alloc::dealloc(ptr as *mut u8, layout);
        v
    }
}

/// Pins the leading f16 field(s) of block `b` of `w` (each `block` bytes
/// long) to finite values: `d = 1.0`, and `dmin = 0.5` when the dtype has
/// one (Q5_K).
fn pin_dbytes(w: &mut [u8], b: usize, block: usize, dmin: bool) {
    w[b * block..b * block + 2].copy_from_slice(&[0x00, 0x3C]);
    if dmin {
        w[b * block + 2..b * block + 4].copy_from_slice(&[0x00, 0x38]);
    }
}

#[test]
fn q80_dequant_vs_cpu() -> Result<()> {
    // Q8_0 kernel vs candle's CPU reference (BlockQ8_0::to_float).
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1101);
    let mut wb = rng.bytes(3 * 34);
    for b in 0..3 {
        pin_dbytes(&mut wb, b, 34, false);
    }
    let blocks = blocks_aligned::<BlockQ8_0>(&wb);
    let mut cpu = vec![0.0f32; 3 * 32];
    GgmlType::to_float(&blocks, &mut cpu);
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(3 * 32)?;
    let (w2, o2, kern) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q80_dequant_f32(cbb, &kern, &w2, &o2, 3 * 32).map_err(|e| e.to_string())
    })?;
    for (i, (g, w)) in got.iter().zip(cpu.iter()).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "q80_dequant vs cpu: [{i}] got {g}, want {w}");
    }
    Ok(())
}

#[test]
fn q50_dequant_vs_cpu() -> Result<()> {
    // Q5_0 kernel vs candle's CPU reference (BlockQ5_0::to_float).
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1102);
    let mut wb = rng.bytes(3 * 22);
    for b in 0..3 {
        pin_dbytes(&mut wb, b, 22, false);
    }
    let blocks = blocks_aligned::<BlockQ5_0>(&wb);
    let mut cpu = vec![0.0f32; 3 * 32];
    GgmlType::to_float(&blocks, &mut cpu);
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(3 * 32)?;
    let (w2, o2, kern) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q50_dequant_f32(cbb, &kern, &w2, &o2, 3 * 32).map_err(|e| e.to_string())
    })?;
    for (i, (g, w)) in got.iter().zip(cpu.iter()).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "q50_dequant vs cpu: [{i}] got {g}, want {w}");
    }
    Ok(())
}

#[test]
fn q5k_dequant_vs_cpu() -> Result<()> {
    // Q5_K kernel vs candle's CPU reference (BlockQ5K::to_float).
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1103);
    let mut wb = rng.bytes(3 * 176);
    for b in 0..3 {
        pin_dbytes(&mut wb, b, 176, true);
    }
    let blocks = blocks_aligned::<BlockQ5K>(&wb);
    let mut cpu = vec![0.0f32; 3 * 256];
    GgmlType::to_float(&blocks, &mut cpu);
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(3 * 256)?;
    let (w2, o2, kern) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q5k_dequant_f32(cbb, &kern, &w2, &o2, 3 * 256).map_err(|e| e.to_string())
    })?;
    for (i, (g, w)) in got.iter().zip(cpu.iter()).enumerate() {
        let tol = 1e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "q5k_dequant vs cpu: [{i}] got {g}, want {w}");
    }
    Ok(())
}
#[test]
fn q50_debug_dump() -> Result<()> {
    // Temporary diagnostic: dump raw bytes + all got/want pairs.
    let d = vulkan_dev()?;
    let vd = d.as_vulkan_device()?;
    let mut rng = Rng::new(1102);
    let mut wb = rng.bytes(3 * 22);
    for b in 0..3 {
        pin_dbytes(&mut wb, b, 22, false);
    }
    println!(
        "wb: {}",
        wb.iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let blocks = blocks_aligned::<BlockQ5_0>(&wb);
    let mut cpu = vec![0.0f32; 3 * 32];
    GgmlType::to_float(&blocks, &mut cpu);
    let iw = vd.upload_u8(&wb)?;
    let o = vd.new_f32_buffer(3 * 32)?;
    let (w2, o2, kern) = (iw.clone(), o.clone(), vd.kernels());
    let got = run_and_get(vd, &o, move |cbb| {
        vk::call_q50_dequant_f32(cbb, &kern, &w2, &o2, 3 * 32).map_err(|e| e.to_string())
    })?;
    for (i, (g, w)) in got.iter().zip(cpu.iter()).enumerate() {
        println!("elem[{i}]: got {g} want {w}");
    }
    Ok(())
}
