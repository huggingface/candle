//! amdgcn kernels for the candle `std::offload` backend, plus their host-side
//! launchers.
//!
//! This crate is compiled twice: once for the host, where the `offload!`
//! invocations live, and once for `amdgcn-amd-amdhsa` as `no_std`, where the
//! `#[offload_kernel]` bodies become the device image. Both compilations must
//! produce the *same crate name*, because the symbol `offload!` looks up is
//! mangled from it. Hence a library with no dependencies: anything it pulled in
//! would also have to build for amdgcn.
//!
//! Every kernel takes device addresses as `usize` rather than as slices or raw
//! pointers, and that is load-bearing. `offload!` infers data movement from
//! argument *types*: a `&[T]` is copied host->device before the launch and back
//! afterwards, every time, which makes device residency impossible. Scalars are
//! passed by value, so an address laundered through `usize` escapes the mapping
//! machinery and the buffer stays where `omp_target_alloc` put it.
#![allow(internal_features)]
#![allow(unused_unsafe)]
#![feature(gpu_offload, core_intrinsics, f16)]
#![cfg_attr(target_arch = "amdgpu", no_std)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu, abi_gpu_kernel))]

use core::offload::offload_kernel;

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{ds_bpermute, workgroup_id_x, workitem_id_x};

// Host-side stubs so the same source compiles for x86. Only the
// `#[offload_kernel]` signatures matter there; the bodies are never reached.
#[cfg(not(target_arch = "amdgpu"))]
fn workitem_id_x() -> u32 {
    0
}
#[cfg(not(target_arch = "amdgpu"))]
fn workgroup_id_x() -> u32 {
    0
}

// G4: the device link step does not link `core`, so a panicking path becomes an
// undefined symbol rather than a panic. A `#[panic_handler]` does not fix that --
// the kernels below still have to avoid `[]` and guard every `/` and `%`.
#[cfg(target_arch = "amdgpu")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Threads per workgroup. Fixed, because there is no `blockDim` to read back on
/// the device side (G8) -- the kernel must hardcode what the launcher used.
pub const TPB: u32 = 256;

/// Wavefront size on gfx1201, and the number of waves in a workgroup.
pub const WAVE: u32 = 32;
pub const WAVES_PER_WG: u32 = TPB / WAVE;

#[inline(always)]
fn gid() -> usize {
    (workitem_id_x() + workgroup_id_x() * TPB) as usize
}

#[inline(always)]
fn expf(x: f32) -> f32 {
    // `core::intrinsics` has suffixed sqrtf32/fmaf32 but only a *generic* exp.
    unsafe { core::intrinsics::exp(x) }
}

#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    unsafe { core::intrinsics::sqrtf32(x) }
}

#[inline(always)]
fn tanhf(x: f32) -> f32 {
    // G12: no_std reaches neither tanh nor erf, so this is built from exp.
    // Clamped because exp(2x) overflows long before tanh saturates.
    if x > 10.0 {
        1.0
    } else if x < -10.0 {
        -1.0
    } else {
        let e = expf(2.0 * x);
        (e - 1.0) / (e + 1.0)
    }
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

/// Batched GEMM with explicit strides on B, so candle's `x.matmul(&w.t())` --
/// where the right-hand side is a transposed *view* rather than a fresh buffer --
/// is handled without materialising the transpose.
///
/// A is row-major within a batch, C is row-major contiguous, and B is addressed
/// as `b[kk * brs + col * bcs]`.
#[offload_kernel]
pub fn k_gemm_f32(
    pa: usize,
    pb: usize,
    pc: usize,
    nb: usize,
    m: usize,
    n: usize,
    k: usize,
    brs: usize,
    bcs: usize,
    a_bs: usize,
    b_bs: usize,
) {
    let idx = gid();
    // G4: these guards exist so the device link succeeds, and guarding the
    // *operands* is not enough -- `m != 0 && n != 0` does not prove `m * n != 0`
    // to LLVM, because the multiply can wrap. The divisor expression itself has
    // to be tested, or rustc emits a call to `panic_const_div_by_zero` that the
    // device link cannot resolve.
    let mn = m * n;
    if n == 0 || mn == 0 || idx >= nb * mn {
        return;
    }
    let batch = idx / mn;
    let rem = idx - batch * mn;
    let row = rem / n;
    let col = rem - row * n;

    let a = (pa as *const f32).wrapping_add(batch * a_bs + row * k);
    let b = (pb as *const f32).wrapping_add(batch * b_bs + col * bcs);
    let mut acc = 0.0f32;
    let mut t = 0usize;
    while t < k {
        unsafe { acc += *a.add(t) * *b.add(t * brs) };
        t += 1;
    }
    unsafe { *(pc as *mut f32).add(batch * mn + row * n + col) = acc };
}

/// Specialisation for the shape candle's `Linear` actually produces: `brs == 1`,
/// `bcs == k`, i.e. B is the untransposed weight laid out (n, k) row-major, so
/// both operands walk contiguously along k. Four output columns per thread, so
/// each loaded A value is reused 4x.
#[offload_kernel]
pub fn k_gemm_f32_bt4(
    pa: usize,
    pb: usize,
    pc: usize,
    nb: usize,
    m: usize,
    n: usize,
    k: usize,
    a_bs: usize,
    b_bs: usize,
) {
    let idx = gid();
    let nq = n / 4;
    let mnq = m * nq;
    if nq == 0 || mnq == 0 || idx >= nb * mnq {
        return;
    }
    let batch = idx / mnq;
    let rem = idx - batch * mnq;
    let row = rem / nq;
    let colb = (rem - row * nq) * 4;

    let a = (pa as *const f32).wrapping_add(batch * a_bs + row * k);
    let b = (pb as *const f32).wrapping_add(batch * b_bs + colb * k);
    // G4 from the other direction: `[f32; 4]` would index through a
    // bounds-checked path, so the accumulators are four separate scalars.
    let (mut c0, mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut t = 0usize;
    while t < k {
        unsafe {
            let av = *a.add(t);
            let bp = b.add(t);
            c0 += av * *bp;
            c1 += av * *bp.add(k);
            c2 += av * *bp.add(2 * k);
            c3 += av * *bp.add(3 * k);
        }
        t += 1;
    }
    unsafe {
        let o = (pc as *mut f32).add(batch * m * n + row * n + colb);
        *o = c0;
        *o.add(1) = c1;
        *o.add(2) = c2;
        *o.add(3) = c3;
    }
}

/// GEMV: the `m == 1` case, which is every matmul in autoregressive decode.
///
/// Worth its own kernel because the general GEMM parallelises over output
/// elements, and with m == 1 there are only `n` of those -- a 2048x2048 decode
/// matmul gets 512 threads and leaves the GPU essentially idle. Here one *wave*
/// owns one output element and its 32 lanes split the k-loop, so the same matmul
/// gets 32x the parallelism and reads B in a coalesced pattern.
///
/// This is only writable because of the `ds_bpermute` butterfly below; the
/// natural formulation, a workgroup reducing through shared memory, has no Rust
/// spelling (G6), and the wave reductions that do exist are integer-only.
/// Requires B laid out (n, k) row-major, i.e. `brs == 1 && bcs == k`.
#[offload_kernel]
pub fn k_gemv_f32_bt(
    pa: usize,
    pb: usize,
    pc: usize,
    nb: usize,
    n: usize,
    k: usize,
    a_bs: usize,
    b_bs: usize,
) {
    let (idx, lane) = wave_row_lane();
    if n == 0 || idx >= nb * n {
        return;
    }
    let batch = idx / n;
    let col = idx - batch * n;
    let a = (pa as *const f32).wrapping_add(batch * a_bs);
    let b = (pb as *const f32).wrapping_add(batch * b_bs + col * k);
    let mut acc = 0.0f32;
    let mut t = lane;
    while t < k {
        unsafe { acc += *a.add(t) * *b.add(t) };
        t += WAVE as usize;
    }
    let s = wave_sum_f32(acc);
    if lane == 0 {
        unsafe { *(pc as *mut f32).add(batch * n + col) = s };
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------
//
// Binary and unary ops dispatch on an integer opcode inside one kernel rather
// than compiling one kernel each. The opcode is a scalar, so it lands in a
// scalar register and the branch is uniform across the wave -- a scalar compare,
// not divergence. The alternative, ~20 kernels, would mean ~20 `offload!` sites
// and ~20 entries in the device image.

pub const OP_ADD: usize = 0;
pub const OP_SUB: usize = 1;
pub const OP_MUL: usize = 2;
pub const OP_DIV: usize = 3;
pub const OP_MAX: usize = 4;
pub const OP_MIN: usize = 5;

#[offload_kernel]
pub fn k_binary_f32(pa: usize, pb: usize, pc: usize, n: usize, op: usize) {
    let i = gid();
    if i >= n {
        return;
    }
    unsafe {
        let x = *(pa as *const f32).add(i);
        let y = *(pb as *const f32).add(i);
        let r = match op {
            OP_ADD => x + y,
            OP_SUB => x - y,
            OP_MUL => x * y,
            OP_DIV => x / y,
            OP_MAX => {
                if x > y {
                    x
                } else {
                    y
                }
            }
            _ => {
                if x < y {
                    x
                } else {
                    y
                }
            }
        };
        *(pc as *mut f32).add(i) = r;
    }
}

pub const U_NEG: usize = 0;
pub const U_SQR: usize = 1;
pub const U_SQRT: usize = 2;
pub const U_EXP: usize = 3;
pub const U_LOG: usize = 4;
pub const U_SIN: usize = 5;
pub const U_COS: usize = 6;
pub const U_ABS: usize = 7;
pub const U_RECIP: usize = 8;
pub const U_RELU: usize = 9;
pub const U_SILU: usize = 10;
pub const U_GELU: usize = 11;
pub const U_TANH: usize = 12;
pub const U_SIGN: usize = 13;

#[offload_kernel]
pub fn k_unary_f32(pa: usize, pc: usize, n: usize, op: usize) {
    let i = gid();
    if i >= n {
        return;
    }
    unsafe {
        let x = *(pa as *const f32).add(i);
        let r = match op {
            U_NEG => -x,
            U_SQR => x * x,
            U_SQRT => sqrtf(x),
            U_EXP => expf(x),
            U_LOG => core::intrinsics::log(x),
            U_SIN => core::intrinsics::sin(x),
            U_COS => core::intrinsics::cos(x),
            // No `f32::abs` in no_std, but the sign bit is right there.
            U_ABS => f32::from_bits(x.to_bits() & 0x7fff_ffff),
            U_RECIP => 1.0 / x,
            U_RELU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            U_SILU => x / (1.0 + expf(-x)),
            U_GELU => {
                // The tanh approximation candle uses on CPU, rebuilt from exp.
                let x3 = x * x * x;
                0.5 * x * (1.0 + tanhf(0.797_884_6 * (x + 0.044_715 * x3)))
            }
            U_TANH => tanhf(x),
            _ => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        };
        *(pc as *mut f32).add(i) = r;
    }
}

#[offload_kernel]
pub fn k_affine_f32(pa: usize, pc: usize, n: usize, mul: f32, add: f32) {
    let i = gid();
    if i < n {
        unsafe { *(pc as *mut f32).add(i) = *(pa as *const f32).add(i) * mul + add };
    }
}

#[offload_kernel]
pub fn k_fill_f32(pc: usize, n: usize, v: f32) {
    let i = gid();
    if i < n {
        unsafe { *(pc as *mut f32).add(i) = v };
    }
}

#[offload_kernel]
pub fn k_copy_f32(pa: usize, pc: usize, n: usize) {
    let i = gid();
    if i < n {
        unsafe { *(pc as *mut f32).add(i) = *(pa as *const f32).add(i) };
    }
}

// ---------------------------------------------------------------------------
// Transformer ops: RmsNorm, softmax, rotary embedding
// ---------------------------------------------------------------------------
//
// These exist because candle-nn implements them as `CustomOp`s whose GPU path is
// gated on `feature = "cuda"`. Without them a transformer round-trips three
// tensors per layer through the host.
//
// RmsNorm and softmax both reduce along the innermost dimension, and that is
// where the language runs out. The natural shape is one workgroup per row
// cooperating through shared memory, but LDS has no accessor in Rust (G6).
// `core::arch::amdgpu` does have wave reductions -- and they are integer-only:
// `wave_reduce_add` takes u32, `wave_reduce_max`/`_min` take i32, and there is no
// f32 form at all. So the reduction is a hand-rolled XOR butterfly over
// `ds_bpermute`, which moves a u32 between lanes, bitcasting the f32 by hand.
// One wave per row, `WAVES_PER_WG` rows per workgroup.

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn wave_sum_f32(mut v: f32) -> f32 {
    let lane = workitem_id_x() & (WAVE - 1);
    let mut off = 1u32;
    while off < WAVE {
        // ds_bpermute addresses source lanes by byte offset, hence * 4.
        let other = f32::from_bits(unsafe { ds_bpermute((lane ^ off) * 4, v.to_bits()) });
        v += other;
        off <<= 1;
    }
    v
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn wave_max_f32(mut v: f32) -> f32 {
    let lane = workitem_id_x() & (WAVE - 1);
    let mut off = 1u32;
    while off < WAVE {
        let other = f32::from_bits(unsafe { ds_bpermute((lane ^ off) * 4, v.to_bits()) });
        if other > v {
            v = other;
        }
        off <<= 1;
    }
    v
}

#[cfg(not(target_arch = "amdgpu"))]
fn wave_sum_f32(v: f32) -> f32 {
    v
}
#[cfg(not(target_arch = "amdgpu"))]
fn wave_max_f32(v: f32) -> f32 {
    v
}

/// The row this thread's wave is responsible for, and its lane within the wave.
#[inline(always)]
fn wave_row_lane() -> (usize, usize) {
    let t = workitem_id_x();
    let row = (workgroup_id_x() * WAVES_PER_WG + t / WAVE) as usize;
    (row, (t & (WAVE - 1)) as usize)
}

/// `dst[i] = src[i] / sqrt(mean(row^2) + eps) * alpha[i]`, written to match
/// candle-nn's CPU kernel operation for operation so the two agree numerically.
#[offload_kernel]
pub fn k_rms_norm_f32(px: usize, pa: usize, pd: usize, rows: usize, cols: usize, eps: f32) {
    let (row, lane) = wave_row_lane();
    if cols == 0 || row >= rows {
        return;
    }
    let x = (px as *const f32).wrapping_add(row * cols);
    let mut acc = 0.0f32;
    let mut i = lane;
    while i < cols {
        let v = unsafe { *x.add(i) };
        acc += v * v;
        i += WAVE as usize;
    }
    let m = sqrtf(wave_sum_f32(acc) / cols as f32 + eps);
    let a = pa as *const f32;
    let d = (pd as *mut f32).wrapping_add(row * cols);
    let mut i = lane;
    while i < cols {
        unsafe { *d.add(i) = *x.add(i) / m * *a.add(i) };
        i += WAVE as usize;
    }
}

/// Softmax over the innermost dimension: subtract the row max, exponentiate,
/// divide by the row sum. Two wave reductions per row.
#[offload_kernel]
pub fn k_softmax_last_f32(px: usize, pd: usize, rows: usize, cols: usize) {
    let (row, lane) = wave_row_lane();
    if cols == 0 || row >= rows {
        return;
    }
    let x = (px as *const f32).wrapping_add(row * cols);
    let d = (pd as *mut f32).wrapping_add(row * cols);

    let mut mx = f32::NEG_INFINITY;
    let mut i = lane;
    while i < cols {
        let v = unsafe { *x.add(i) };
        if v > mx {
            mx = v;
        }
        i += WAVE as usize;
    }
    let mx = wave_max_f32(mx);

    let mut s = 0.0f32;
    let mut i = lane;
    while i < cols {
        let e = expf(unsafe { *x.add(i) } - mx);
        unsafe { *d.add(i) = e };
        s += e;
        i += WAVE as usize;
    }
    let s = wave_sum_f32(s);

    let mut i = lane;
    while i < cols {
        unsafe { *d.add(i) = *d.add(i) / s };
        i += WAVE as usize;
    }
}

/// Contiguous rotary embedding, the variant candle's llama uses. Source is
/// (b, h, t, d) and the two halves of each head vector rotate against each
/// other. `cs_b` is the per-batch stride into cos/sin, 0 when they are shared.
#[offload_kernel]
pub fn k_rope_f32(
    ps: usize,
    pcos: usize,
    psin: usize,
    pd: usize,
    bh: usize,
    t: usize,
    d: usize,
    h: usize,
    cs_b: usize,
) {
    let j = gid();
    let hd = d / 2;
    if hd == 0 || t == 0 || h == 0 || j >= bh * t * hd {
        return;
    }
    let i_d = j % hd;
    let rest = j / hd;
    let i_t = rest % t;
    let bh_i = rest / t;

    let i1 = bh_i * t * d + i_t * d + i_d;
    let i2 = i1 + hd;
    let i_cs = i_t * hd + i_d + (bh_i / h) * cs_b;

    unsafe {
        let s = ps as *const f32;
        let c = *(pcos as *const f32).add(i_cs);
        let sn = *(psin as *const f32).add(i_cs);
        let x1 = *s.add(i1);
        let x2 = *s.add(i2);
        let dst = pd as *mut f32;
        *dst.add(i1) = x1 * c - x2 * sn;
        *dst.add(i2) = x1 * sn + x2 * c;
    }
}

// ---------------------------------------------------------------------------
// Reductions over the innermost dimension
// ---------------------------------------------------------------------------

pub const R_SUM: usize = 0;
pub const R_MAX: usize = 1;
pub const R_MIN: usize = 2;

#[offload_kernel]
pub fn k_reduce_last_f32(pa: usize, pc: usize, rows: usize, cols: usize, op: usize) {
    let r = gid();
    if r >= rows || cols == 0 {
        return;
    }
    let src = (pa as *const f32).wrapping_add(r * cols);
    let mut acc = unsafe { *src };
    if op == R_SUM {
        acc = 0.0;
    }
    let mut j = 0usize;
    while j < cols {
        let v = unsafe { *src.add(j) };
        acc = match op {
            R_SUM => acc + v,
            R_MAX => {
                if v > acc {
                    v
                } else {
                    acc
                }
            }
            _ => {
                if v < acc {
                    v
                } else {
                    acc
                }
            }
        };
        j += 1;
    }
    unsafe { *(pc as *mut f32).add(r) = acc };
}

// ---------------------------------------------------------------------------
// dtype conversion and gather
// ---------------------------------------------------------------------------
//
// These two are where the transfer accounting pointed. `to_dtype` is the whole
// model-load path -- safetensors ship f16/bf16 and candle converts to the
// requested dtype -- and `index_select` is the embedding lookup, which without a
// kernel downloads the entire (vocab x hidden) table once per token.
//
// The dtype pair is worth noting for the language question. `f16` is a real
// primitive under `#![feature(f16)]` and `as` casts to and from it lower to
// hardware instructions. `bf16` has no primitive at all (G10), so it is handled
// as a `u16` with the conversion written out by hand: widening is just a 16-bit
// shift, and narrowing is round-to-nearest-even. That works, but nothing in the
// type system knows those u16s are floats.

pub const CVT_F16_F32: usize = 0;
pub const CVT_BF16_F32: usize = 1;
pub const CVT_F32_F16: usize = 2;
pub const CVT_F32_BF16: usize = 3;

#[offload_kernel]
pub fn k_convert(psrc: usize, pdst: usize, n: usize, op: usize) {
    let i = gid();
    if i >= n {
        return;
    }
    unsafe {
        match op {
            CVT_F16_F32 => {
                let v = *(psrc as *const f16).add(i);
                *(pdst as *mut f32).add(i) = v as f32;
            }
            CVT_BF16_F32 => {
                // bf16 is the high half of an f32, so widening is a shift.
                let v = *(psrc as *const u16).add(i);
                *(pdst as *mut f32).add(i) = f32::from_bits((v as u32) << 16);
            }
            CVT_F32_F16 => {
                let v = *(psrc as *const f32).add(i);
                *(pdst as *mut f16).add(i) = v as f16;
            }
            _ => {
                // Round-to-nearest-even, matching half::bf16::from_f32.
                let v = *(psrc as *const f32).add(i);
                let b = v.to_bits();
                let out = if v != v {
                    0x7fc0u16
                } else {
                    (((b + 0x7fff + ((b >> 16) & 1)) >> 16) & 0xffff) as u16
                };
                *(pdst as *mut u16).add(i) = out;
            }
        }
    }
}

pub const IDS_U32: usize = 0;
pub const IDS_I64: usize = 1;

/// `dst[l, i, r] = src[l, ids[i], r]`, the general form of candle's
/// `index_select` flattened to three dimensions around the selected axis.
#[offload_kernel]
pub fn k_index_select_f32(
    psrc: usize,
    pids: usize,
    pdst: usize,
    left: usize,
    n_ids: usize,
    right: usize,
    mid: usize,
    ids_kind: usize,
) {
    let j = gid();
    if right == 0 || n_ids == 0 || j >= left * n_ids * right {
        return;
    }
    let r = j % right;
    let rest = j / right;
    let i = rest % n_ids;
    let l = rest / n_ids;
    let idx = unsafe {
        if ids_kind == IDS_I64 {
            *(pids as *const i64).add(i) as usize
        } else {
            *(pids as *const u32).add(i) as usize
        }
    };
    if idx >= mid {
        return;
    }
    unsafe {
        *(pdst as *mut f32).add(j) = *(psrc as *const f32).add((l * mid + idx) * right + r);
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers, exported over the C ABI
// ---------------------------------------------------------------------------
//
// This crate is a **cdylib**, and the launchers are `extern "C"`, for a reason
// that is not stylistic.
//
// rustc performs the offload lowering during fat LTO, and fat LTO never runs for
// an rlib. So the crate whose codegen contains an `offload!` expansion has to be
// a *final linked artifact*: a binary, a cdylib or a staticlib. Anything else
// fails with "using the offload feature requires -C lto=fat", and passing
// `-C lto=fat` to an rlib silently produces no artifact at all.
//
// That rules out the obvious layering -- kernels <- candle-core <- candle-nn <-
// example -- because whichever rlib in that chain ends up holding the launcher
// codegen cannot satisfy the requirement, and which one that is depends on
// inlining decisions. (Marking the launchers `#[inline(never)]` only moves the
// failure from candle-core to this crate.)
//
// Building the kernels as a cdylib cuts the graph instead: the offload intrinsic
// exists only here, and candle-core reaches it through `extern "C"`
// declarations, so no candle crate needs an offload flag or fat LTO.

#[cfg(not(target_arch = "amdgpu"))]
#[inline]
fn blocks(n: usize) -> u32 {
    n.div_ceil(TPB as usize).max(1) as u32
}

/// One wave per row, `WAVES_PER_WG` rows per workgroup.
#[cfg(not(target_arch = "amdgpu"))]
#[inline]
fn row_blocks(rows: usize) -> u32 {
    rows.div_ceil(WAVES_PER_WG as usize).max(1) as u32
}

/// `brs`/`bcs` are B's row and column strides in elements; `a_bs`/`b_bs` are
/// per-batch strides, where 0 broadcasts that operand across the batch.
#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_gemm_f32(
    pa: usize,
    pb: usize,
    pc: usize,
    nb: usize,
    m: usize,
    n: usize,
    k: usize,
    brs: usize,
    bcs: usize,
    a_bs: usize,
    b_bs: usize,
) {
    // Decode is all matrix-vector; it needs a different decomposition entirely.
    if m == 1 && brs == 1 && bcs == k {
        let g = row_blocks(nb * n);
        core::offload::offload! {
            kernel = k_gemv_f32_bt,
            workgroup_dim = [g, 1, 1],
            thread_dim = [TPB, 1, 1],
            args = (pa, pb, pc, nb, n, k, a_bs, b_bs),
        }
    } else if brs == 1 && bcs == k && n % 4 == 0 && n >= 4 {
        let g = blocks(nb * m * (n / 4));
        core::offload::offload! {
            kernel = k_gemm_f32_bt4,
            workgroup_dim = [g, 1, 1],
            thread_dim = [TPB, 1, 1],
            args = (pa, pb, pc, nb, m, n, k, a_bs, b_bs),
        }
    } else {
        let g = blocks(nb * m * n);
        core::offload::offload! {
            kernel = k_gemm_f32,
            workgroup_dim = [g, 1, 1],
            thread_dim = [TPB, 1, 1],
            args = (pa, pb, pc, nb, m, n, k, brs, bcs, a_bs, b_bs),
        }
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_binary_f32(op: usize, pa: usize, pb: usize, pc: usize, n: usize) {
    let g = blocks(n);
    core::offload::offload! {
        kernel = k_binary_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (pa, pb, pc, n, op),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_unary_f32(op: usize, pa: usize, pc: usize, n: usize) {
    let g = blocks(n);
    core::offload::offload! {
        kernel = k_unary_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (pa, pc, n, op),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_affine_f32(pa: usize, pc: usize, n: usize, mul: f32, add: f32) {
    let g = blocks(n);
    core::offload::offload! {
        kernel = k_affine_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (pa, pc, n, mul, add),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_fill_f32(pc: usize, n: usize, v: f32) {
    let g = blocks(n);
    core::offload::offload! {
        kernel = k_fill_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (pc, n, v),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_copy_f32(pa: usize, pc: usize, n: usize) {
    let g = blocks(n);
    core::offload::offload! {
        kernel = k_copy_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (pa, pc, n),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_reduce_last_f32(
    op: usize,
    pa: usize,
    pc: usize,
    rows: usize,
    cols: usize,
) {
    let g = blocks(rows);
    core::offload::offload! {
        kernel = k_reduce_last_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (pa, pc, rows, cols, op),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_rms_norm_f32(
    px: usize,
    pa: usize,
    pd: usize,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let g = row_blocks(rows);
    core::offload::offload! {
        kernel = k_rms_norm_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (px, pa, pd, rows, cols, eps),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_softmax_last_f32(px: usize, pd: usize, rows: usize, cols: usize) {
    let g = row_blocks(rows);
    core::offload::offload! {
        kernel = k_softmax_last_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (px, pd, rows, cols),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn candle_offload_rope_f32(
    ps: usize,
    pcos: usize,
    psin: usize,
    pd: usize,
    bh: usize,
    t: usize,
    d: usize,
    h: usize,
    cs_b: usize,
) {
    let g = blocks(bh * t * (d / 2).max(1));
    core::offload::offload! {
        kernel = k_rope_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (ps, pcos, psin, pd, bh, t, d, h, cs_b),
    }
}

/// Number of devices libomptarget can see, so candle-core does not have to link
/// the offload runtime itself.
#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_num_devices() -> i32 {
    core::intrinsics::offload_get_num_devices()
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
pub extern "C" fn candle_offload_convert(psrc: usize, pdst: usize, n: usize, op: usize) {
    let g = blocks(n);
    core::offload::offload! {
        kernel = k_convert,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (psrc, pdst, n, op),
    }
}

#[cfg(not(target_arch = "amdgpu"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn candle_offload_index_select_f32(
    psrc: usize,
    pids: usize,
    pdst: usize,
    left: usize,
    n_ids: usize,
    right: usize,
    mid: usize,
    ids_kind: usize,
) {
    let g = blocks(left * n_ids * right);
    core::offload::offload! {
        kernel = k_index_select_f32,
        workgroup_dim = [g, 1, 1],
        thread_dim = [TPB, 1, 1],
        args = (psrc, pids, pdst, left, n_ids, right, mid, ids_kind),
    }
}
