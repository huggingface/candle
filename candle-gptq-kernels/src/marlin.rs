//! FFI binding to the real Marlin FP16xINT4 tensor-core GEMM kernel (vendored under
//! `kernels/marlin/`, Apache-2.0, from <https://github.com/IST-DASLab/marlin>).
//!
//! This module does NOT reimplement Marlin's GEMM: the CUDA C++ kernel is compiled as-is and
//! driven over FFI (`ffi::run_marlin_gemm`). What is reimplemented here, host-side in Rust, is the
//! offline weight *repack* (ported from upstream's `marlin/__init__.py` `Layer.pack` /
//! `_get_perms`), which converts a GPTQ checkpoint into the permuted layout the kernel expects.
//!
//! Marlin only handles a restricted (but very common) case, so [`marlin_repack_gptq`] returns
//! `Ok(None)` for checkpoints it cannot serve and the caller falls back to the generic kernels:
//! 4-bit, symmetric (per-output zero point fixed at 8), FP16 operands, group size 128 or -1
//! (per-channel), sequential `g_idx` (no act-order), `in_dim % 128 == 0`, `out_dim % 256 == 0`.

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Tensor};
use half::f16;

use crate::ffi;

const TILE: usize = 16;
const PERM_LEN: usize = 1024;
const SCALE_PERM_LEN: usize = 64;
const SCALE_PERM_SINGLE_LEN: usize = 32;
const MAX_PAR: i32 = 16;
/// 4-bit symmetric GPTQ stores `zero_point - 1`, and a symmetric checkpoint fixes the zero point
/// at `2^(4-1) = 8`, so every packed `qzeros` word is `0x7777_7777`.
const SYM_QZERO_WORD: i32 = 0x7777_7777u32 as i32;

/// Port of upstream `_get_perms()`: the weight tile permutation (`perm`, 1024 entries) and the two
/// scale-column permutations (grouped: 64 entries; per-channel: 32 entries).
fn marlin_perms() -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut perm: Vec<usize> = Vec::with_capacity(PERM_LEN);
    for i in 0..32usize {
        let mut perm1: Vec<usize> = Vec::with_capacity(8);
        let col = i / 4;
        for block in [0usize, 1] {
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1.push(16 * row + col + 8 * block);
            }
        }
        for j in 0..4usize {
            for &p in perm1.iter() {
                perm.push(p + 256 * j);
            }
        }
    }
    // `perm.reshape(-1, 8)[:, [0,2,4,6,1,3,5,7]].ravel()`
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let mut perm_i = Vec::with_capacity(PERM_LEN);
    for chunk in perm.chunks_exact(8) {
        for &idx in interleave.iter() {
            perm_i.push(chunk[idx]);
        }
    }

    let mut scale_perm: Vec<usize> = Vec::with_capacity(SCALE_PERM_LEN);
    for i in 0..8usize {
        for j in 0..8usize {
            scale_perm.push(i + 8 * j);
        }
    }
    let mut scale_perm_single: Vec<usize> = Vec::with_capacity(SCALE_PERM_SINGLE_LEN);
    for i in 0..4usize {
        for j in [0usize, 1, 8, 9, 16, 17, 24, 25] {
            scale_perm_single.push(2 * i + j);
        }
    }
    (perm_i, scale_perm, scale_perm_single)
}

/// Repack already-unpacked 4-bit levels `q` (row-major `[k, n]`, values 0..15) into Marlin's
/// `B` layout, returning a flat `[k/16, n*2]` `i32` buffer. Pure function of `q`; ported from
/// `Layer.pack` (the weight half).
fn pack_weights(q: &[i32], k: usize, n: usize) -> Vec<i32> {
    let (perm, _, _) = marlin_perms();
    // w.reshape(k/16,16,n/16,16).transpose(0,2,1,3).reshape(k/16, n*16)
    let row_len = n * TILE; // columns per `k/16` row after the tile transpose
    let mut tmp = vec![0i32; (k / TILE) * row_len];
    for ko in 0..k / TILE {
        for no in 0..n / TILE {
            for i in 0..TILE {
                for j in 0..TILE {
                    let src = (ko * TILE + i) * n + (no * TILE + j);
                    let dst = ko * row_len + no * 256 + i * TILE + j;
                    tmp[dst] = q[src];
                }
            }
        }
    }
    // res = tmp.reshape(-1, 1024)[:, perm] applied per flat 1024-block.
    let mut res = vec![0i32; tmp.len()];
    for (b, block) in tmp.chunks_exact(PERM_LEN).enumerate() {
        let base = b * PERM_LEN;
        for i in 0..PERM_LEN {
            res[base + i] = block[perm[i]];
        }
    }
    // Bit-pack 8 nibbles per i32: B[r,p] = OR_i (res[r, 8p+i] & 0xF) << 4i
    let packed_cols = n * 2; // n*16/8
    let mut out = vec![0i32; (k / TILE) * packed_cols];
    for r in 0..k / TILE {
        for p in 0..packed_cols {
            let mut acc: u32 = 0;
            for i in 0..8usize {
                let v = (res[r * row_len + 8 * p + i] as u32) & 0xF;
                acc |= v << (4 * i);
            }
            out[r * packed_cols + p] = acc as i32;
        }
    }
    out
}

/// Repack GPTQ scales `[n_groups, n]` into Marlin's permuted scale layout (same shape). `grouped`
/// selects the 64-wide group permutation vs the 32-wide per-channel one. Ported from `Layer.pack`
/// (the scale half).
fn pack_scales(scales: &[f32], n_groups: usize, n: usize, grouped: bool) -> Vec<f32> {
    let (_, scale_perm, scale_perm_single) = marlin_perms();
    let (perm, block) = if grouped {
        (&scale_perm, SCALE_PERM_LEN)
    } else {
        (&scale_perm_single, SCALE_PERM_SINGLE_LEN)
    };
    let mut out = vec![0f32; n_groups * n];
    // reshape(-1, block)[:, perm]: each contiguous `block` of columns is permuted; blocks never
    // cross a group row because `n` is a multiple of `block`.
    for g in 0..n_groups {
        for b in 0..n / block {
            let base = g * n + b * block;
            for i in 0..block {
                out[base + i] = scales[base + perm[i]];
            }
        }
    }
    out
}

/// Marlin's groupsize convention: `-1` means a single per-output-channel scale.
fn marlin_groupsize(group_size: usize, k: usize) -> i32 {
    if group_size >= k {
        -1
    } else {
        group_size as i32
    }
}

/// Repack a GPTQ-quantized weight into Marlin's `(B, s)` layout if (and only if) the checkpoint is
/// Marlin-eligible; otherwise return `Ok(None)` so the caller can fall back to the generic kernels.
///
/// * `qweight` - packed GPTQ weight, `i32`, shape `[k / 8, n]`.
/// * `qzeros`  - packed GPTQ zero points, `i32`, shape `[n_groups, n / 8]`.
/// * `scales`  - per-group scales, `f32`, shape `[n_groups, n]`.
///
/// On success returns `(B, s)` where `B` is `i32 [k/16, n*2]` and `s` is `f16 [n_groups, n]`, both
/// on the same device as `qweight`.
pub fn marlin_repack_gptq(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    if bits != 4 {
        return Ok(None);
    }
    let device = qweight.device().clone();
    let (packed_k, n) = qweight.dims2()?;
    let k = packed_k * 8;
    let (n_groups, n_packed_z) = qzeros.dims2()?;
    let (scale_groups, scale_n) = scales.dims2()?;

    // Shape eligibility (Marlin tiling + our repack assumptions).
    if k % 128 != 0
        || n % 256 != 0
        || n_packed_z * 8 != n
        || scale_n != n
        || scale_groups != n_groups
    {
        return Ok(None);
    }
    let gs = marlin_groupsize(group_size, k);
    let grouped = gs != -1;
    if grouped {
        if gs != 128 || n_groups != k / 128 {
            return Ok(None);
        }
    } else if n_groups != 1 {
        return Ok(None);
    }

    // Pull the packed tensors to host. Repack is a one-time, load-time cost.
    let qweight_host = qweight.to_dtype(DType::I32)?.to_vec2::<i32>()?;
    let qzeros_host = qzeros.to_dtype(DType::I32)?.to_vec2::<i32>()?;
    let scales_host = scales.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    // Symmetric-only: every packed zero word must be 0x77777777 (zero point == 8 everywhere).
    for row in qzeros_host.iter() {
        for &w in row.iter() {
            if w != SYM_QZERO_WORD {
                return Ok(None);
            }
        }
    }

    // Unpack GPTQ qweight -> q[k, n] in 0..15.
    let mut q = vec![0i32; k * n];
    for row in 0..k {
        let word_row = &qweight_host[row / 8];
        let shift = (row % 8) * 4;
        for col in 0..n {
            q[row * n + col] = (word_row[col] >> shift) & 0xF;
        }
    }
    let b_flat = pack_weights(&q, k, n);

    let scales_flat: Vec<f32> = scales_host.into_iter().flatten().collect();
    let s_flat = pack_scales(&scales_flat, n_groups, n, grouped);
    let s_flat_f16: Vec<f16> = s_flat.into_iter().map(f16::from_f32).collect();

    let b =
        Tensor::from_vec(b_flat, (k / TILE, n * 2), &candle::Device::Cpu)?.to_device(&device)?;
    let s =
        Tensor::from_vec(s_flat_f16, (n_groups, n), &candle::Device::Cpu)?.to_device(&device)?;
    Ok(Some((b, s)))
}

/// `a.apply_op3(b, s, MarlinGemm)`: drives the vendored Marlin kernel. `a`/`s` are `f16`, `b` is
/// `i32` (already in Marlin layout, see [`marlin_repack_gptq`]). Output is `f16 [m, n]`.
struct MarlinGemm;

impl candle::CustomOp3 for MarlinGemm {
    fn name(&self) -> &'static str {
        "marlin-gemm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for the Marlin kernel; use the dense path instead")
    }

    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        s: &CudaStorage,
        s_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if a.dtype() != DType::F16 {
            candle::bail!("marlin: activations must be f16, got {:?}", a.dtype());
        }
        if s.dtype() != DType::F16 {
            candle::bail!("marlin: scales must be f16, got {:?}", s.dtype());
        }
        let (m, k) = a_l.shape().dims2()?;
        let (k_tiles, n2) = b_l.shape().dims2()?;
        let n = n2 / 2;
        if k_tiles * TILE != k {
            candle::bail!("marlin: B rows {k_tiles} * 16 != k {k}");
        }
        let (n_groups, s_n) = s_l.shape().dims2()?;
        if s_n != n {
            candle::bail!("marlin: scales cols {s_n} != n {n}");
        }
        if n % 256 != 0 || k % 128 != 0 {
            candle::bail!("marlin: unsupported shape m={m} n={n} k={k}");
        }
        // Mirror upstream `mul`: groupsize == -1 for a single scale row, else k / n_groups.
        let groupsize: i32 = if n_groups == 1 {
            -1
        } else {
            (k / n_groups) as i32
        };

        let dev = a.device();
        let stream = dev.cuda_stream();

        let a_slice = a.as_cuda_slice::<f16>()?;
        let a_slice = match a_l.contiguous_offsets() {
            Some((o1, o2)) => a_slice.slice(o1..o2),
            None => candle::bail!("marlin: activations must be contiguous"),
        };
        let b_slice = b.as_cuda_slice::<i32>()?;
        let b_slice = match b_l.contiguous_offsets() {
            Some((o1, o2)) => b_slice.slice(o1..o2),
            None => candle::bail!("marlin: B must be contiguous"),
        };
        let s_slice = s.as_cuda_slice::<f16>()?;
        let s_slice = match s_l.contiguous_offsets() {
            Some((o1, o2)) => s_slice.slice(o1..o2),
            None => candle::bail!("marlin: scales must be contiguous"),
        };

        let mut dst = unsafe { dev.alloc::<f16>(m * n)? };
        // Marlin's Stream-K reduction needs a zero-initialized lock buffer of n/128*max_par ints.
        let mut workspace = dev.alloc_zeros::<i32>((n / 128) * MAX_PAR as usize)?;

        let ret = unsafe {
            let (a_ptr, _g0) = a_slice.device_ptr(&stream);
            let (b_ptr, _g1) = b_slice.device_ptr(&stream);
            let (s_ptr, _g2) = s_slice.device_ptr(&stream);
            let (dst_ptr, _g3) = dst.device_ptr_mut(&stream);
            let (ws_ptr, _g4) = workspace.device_ptr_mut(&stream);
            ffi::run_marlin_gemm(
                a_ptr as *const core::ffi::c_void,
                b_ptr as *const core::ffi::c_void,
                dst_ptr as *mut core::ffi::c_void,
                s_ptr as *const core::ffi::c_void,
                m as i32,
                n as i32,
                k as i32,
                ws_ptr as *mut core::ffi::c_void,
                groupsize,
                // Single-GPU CI runner: device ordinal 0. Marlin only uses `dev` to query the SM
                // count when `sms == -1`.
                0,
                MAX_PAR,
            )
        };
        if ret != 0 {
            candle::bail!(
                "marlin kernel returned error {ret} (1=problem shape, 2=unsupported kernel shape) \
                 for m={m} n={n} k={k} groupsize={groupsize}"
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the vendored Marlin FP16xINT4 GEMM: `y = a @ dequant(b, s)`, where `b`/`s` come from
/// [`marlin_repack_gptq`]. `a` is `f16 [m, k]`; the result is `f16 [m, n]`.
pub fn marlin_gemm(a: &Tensor, b: &Tensor, s: &Tensor) -> Result<Tensor> {
    a.apply_op3(b, s, MarlinGemm)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic q/scale formulas, mirrored from the numpy reference used to generate the
    // golden values below (k=128, n=256, group size -1).
    fn golden_q(k: usize, n: usize) -> Vec<i32> {
        let mut q = vec![0i32; k * n];
        for i in 0..k {
            for j in 0..n {
                q[i * n + j] = ((i * 7 + j * 3) % 16) as i32;
            }
        }
        q
    }
    fn golden_scales(n_groups: usize, n: usize) -> Vec<f32> {
        let mut s = vec![0f32; n_groups * n];
        for g in 0..n_groups {
            for j in 0..n {
                s[g * n + j] = 0.01 + 0.001 * ((g * 5 + j) % 7) as f32;
            }
        }
        s
    }

    #[test]
    fn perms_are_permutations() {
        let (perm, scale_perm, scale_perm_single) = marlin_perms();
        assert_eq!(perm.len(), PERM_LEN);
        assert_eq!(scale_perm.len(), SCALE_PERM_LEN);
        assert_eq!(scale_perm_single.len(), SCALE_PERM_SINGLE_LEN);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert!(
            sorted.iter().copied().eq(0..PERM_LEN),
            "perm not a bijection"
        );
        assert_eq!(perm.iter().sum::<usize>(), 523776);
        assert_eq!(&perm[..8], &[0, 128, 8, 136, 16, 144, 24, 152]);
        assert_eq!(
            scale_perm_single,
            vec![
                0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28,
                29, 6, 7, 14, 15, 22, 23, 30, 31
            ]
        );
    }

    // Cross-check the pure repack against values produced by the upstream-equivalent numpy port
    // (see the session's marlin_ref.py). Catches sign/shift/transpose mistakes that a self-
    // consistent round-trip alone could miss.
    #[test]
    fn pack_weights_matches_golden() {
        let (k, n) = (128usize, 256usize);
        let b = pack_weights(&golden_q(k, n), k, n);
        assert_eq!(b.len(), (k / 16) * (n * 2)); // 8 * 512
        let cols = n * 2;
        assert_eq!(b[0], 2146896000);
        assert_eq!(b[1], 2146896000);
        assert_eq!(b[511], 1860630399);
        assert_eq!(b[7 * cols], 2146896000);
        assert_eq!(b[7 * cols + 511], 1860630399);
        let sum: i64 = b.iter().map(|&x| x as i64).sum();
        assert_eq!(sum, 586128816128);
    }

    #[test]
    fn pack_scales_matches_golden() {
        let (n_groups, n) = (1usize, 256usize);
        let s = pack_scales(&golden_scales(n_groups, n), n_groups, n, false);
        assert!((s[0] - 0.01).abs() < 1e-6);
        assert!((s[255] - 0.013).abs() < 1e-6);
        let sum: f64 = s.iter().map(|&x| x as f64).sum();
        assert!((sum - 3.322).abs() < 1e-3, "scale sum {sum}");
    }

    // Round-trip: packing then unpacking with the inverse permutation must recover q exactly,
    // for both the grouped and per-channel layouts.
    fn unpack_weights(b: &[i32], k: usize, n: usize) -> Vec<i32> {
        let (perm, _, _) = marlin_perms();
        let row_len = n * TILE;
        let packed_cols = n * 2;
        // invert bit packing
        let mut res = vec![0i32; (k / TILE) * row_len];
        for r in 0..k / TILE {
            for p in 0..packed_cols {
                let word = b[r * packed_cols + p] as u32;
                for i in 0..8usize {
                    res[r * row_len + 8 * p + i] = ((word >> (4 * i)) & 0xF) as i32;
                }
            }
        }
        // invert perm: the forward direction computed `res[i] = tmp[perm[i]]`, so recovering
        // `tmp` from `res` means scattering by `perm[i]` (not by the inverse permutation, which
        // would only be correct if `perm` were self-inverse).
        let mut tmp = vec![0i32; res.len()];
        for (bk, block) in res.chunks_exact(PERM_LEN).enumerate() {
            let base = bk * PERM_LEN;
            for i in 0..PERM_LEN {
                tmp[base + perm[i]] = block[i];
            }
        }
        // invert tile transpose
        let mut q = vec![0i32; k * n];
        for ko in 0..k / TILE {
            for no in 0..n / TILE {
                for i in 0..TILE {
                    for j in 0..TILE {
                        let src = ko * row_len + no * 256 + i * TILE + j;
                        let dst = (ko * TILE + i) * n + (no * TILE + j);
                        q[dst] = tmp[src];
                    }
                }
            }
        }
        q
    }

    #[test]
    fn pack_weights_roundtrips() {
        for (k, n) in [(128usize, 256usize), (256, 256), (128, 512)] {
            let mut q = vec![0i32; k * n];
            for (idx, v) in q.iter_mut().enumerate() {
                *v = ((idx * 1103515245 + 12345) % 16) as i32;
            }
            let b = pack_weights(&q, k, n);
            let q2 = unpack_weights(&b, k, n);
            assert_eq!(q, q2, "roundtrip failed for k={k} n={n}");
        }
    }

    // End-to-end GPU test: repack a synthetic symmetric 4-bit GPTQ weight into Marlin layout, run
    // the vendored Marlin kernel over FFI, and check the output against a dense `A @ W` reference.
    // This validates the whole binding (repack + FFI + kernel) on real Ampere hardware; it is run
    // by the `candle-gptq-kernels` test step in ci_cuda.yaml, not locally.
    #[test]
    fn marlin_gemm_matches_dense() -> Result<()> {
        let device = candle::Device::new_cuda(0)?;
        // Marlin-eligible: k % 128 == 0, n % 256 == 0, group size 128.
        let (m, k, n, group_size) = (16usize, 256usize, 256usize, 128usize);
        let n_groups = k / group_size;

        // Symmetric 4-bit levels (0..15) and per-group scales.
        let mut q = vec![0i32; k * n];
        for i in 0..k {
            for j in 0..n {
                q[i * n + j] = ((i * 3 + j * 7) % 16) as i32;
            }
        }
        let mut scales = vec![0f32; n_groups * n];
        for g in 0..n_groups {
            for j in 0..n {
                scales[g * n + j] = 0.01 + 0.002 * ((g * 3 + j) % 5) as f32;
            }
        }

        // Dense reference weight W[k,n] = (q - 8) * scale[group(row), col].
        let mut w = vec![0f32; k * n];
        for i in 0..k {
            let g = i / group_size;
            for j in 0..n {
                w[i * n + j] = (q[i * n + j] - 8) as f32 * scales[g * n + j];
            }
        }
        // Activations and expected A @ W (f32 accumulation).
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let mut expected = vec![0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += a[row * k + kk] as f64 * w[kk * n + col] as f64;
                }
                expected[row * n + col] = acc as f32;
            }
        }

        // Pack a GPTQ checkpoint: qweight [k/8, n], qzeros [n_groups, n/8] (all 0x77777777 = sym).
        let mut qweight = vec![0i32; (k / 8) * n];
        for i in 0..k {
            let shift = (i % 8) * 4;
            for j in 0..n {
                qweight[(i / 8) * n + j] |= q[i * n + j] << shift;
            }
        }
        let qzeros = vec![SYM_QZERO_WORD; n_groups * (n / 8)];

        let qweight_t = Tensor::from_vec(qweight, (k / 8, n), &device)?;
        let qzeros_t = Tensor::from_vec(qzeros, (n_groups, n / 8), &device)?;
        let scales_t = Tensor::from_vec(scales, (n_groups, n), &device)?;

        let (b, s) = marlin_repack_gptq(&qweight_t, &qzeros_t, &scales_t, 4, group_size)?
            .expect("checkpoint should be Marlin-eligible");

        let a_t = Tensor::from_vec(a, (m, k), &device)?.to_dtype(DType::F16)?;
        let y = marlin_gemm(&a_t, &b, &s)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        for row in 0..m {
            for col in 0..n {
                let exp = expected[row * n + col];
                // fp16 operands + fp16 accumulation in the scale step: allow a relative tolerance
                // wide enough for fp16 noise but tight enough to catch layout/repack bugs.
                let tol = 0.05 + 0.03 * exp.abs();
                assert!(
                    (y[row][col] - exp).abs() < tol,
                    "marlin mismatch at ({row},{col}): {} vs {exp}",
                    y[row][col]
                );
            }
        }
        Ok(())
    }
}
