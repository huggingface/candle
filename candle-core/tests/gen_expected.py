#!/usr/bin/env python3
"""Generate the hardcoded expected outputs for the Vulkan kernel integration
tests (candle-core/tests/vulkan_integration_tests.rs).

The RNG is a SplitMix64 bit-for-bit identical to the `Rng` in the Rust test
file. Each test uses seed 1000 + test_index and draws its inputs in the
documented order. Outputs are computed in float32 with the exact operation
order of each shader.

Run: python3 gen_expected.py  ->  writes vulkan_expected.rs
"""

import struct

import numpy as np

# ---------------------------------------------------------------- RNG


_MASK = 0xFFFFFFFFFFFFFFFF


class Rng:
    """SplitMix64, identical to the Rust `Rng` in the test file.

    Pure python ints with explicit 64-bit masking: bit-for-bit identical to
    Rust wrapping arithmetic, no reliance on numpy overflow semantics.
    """

    def __init__(self, seed: int):
        self.state = seed & _MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return (z ^ (z >> 31)) & _MASK

    def f32(self, lo: float, hi: float) -> np.float32:
        u = np.float32(self.next_u64() >> 32) / np.float32(4294967296.0)
        return np.float32(lo) + (np.float32(hi) - np.float32(lo)) * u

    def f32s(self, n: int, lo: float, hi: float) -> np.ndarray:
        return np.array([self.f32(lo, hi) for _ in range(n)], dtype=np.float32)

    def u32_mod(self, m: int) -> int:
        return (self.next_u64() >> 32) % m

    def bytes(self, n: int) -> np.ndarray:
        return np.array([self.next_u64() & 0xFF for _ in range(n)], dtype=np.uint8)


# ------------------------------------------------- Q4_K / Q6_K dequant
# Shader-derived references (transcribed from q4k.comp).


def dequant_q4k(w: np.ndarray) -> np.ndarray:
    nb = len(w) // 144
    out = np.empty(nb * 256, dtype=np.float32)
    e = np.arange(256)
    sub = e // 64
    r = e % 64
    t = r % 32
    half = r // 32
    pair = sub * 2 + half
    byte_idx = 32 * sub + t
    for b in range(nb):
        blk = w[b * 144:(b + 1) * 144]
        d = np.frombuffer(blk[0:2], dtype=np.float16).item()
        dmin = np.frombuffer(blk[2:4], dtype=np.float16).item()
        s = blk[4:16]
        sc = np.array(
            [
                s[0] & 63, s[1] & 63, s[2] & 63, s[3] & 63,
                (s[8] & 0xF) | ((s[0] >> 6) << 4), (s[9] & 0xF) | ((s[1] >> 6) << 4),
                (s[10] & 0xF) | ((s[2] >> 6) << 4), (s[11] & 0xF) | ((s[3] >> 6) << 4),
            ],
            dtype=np.float32,
        )
        mn = np.array(
            [
                s[4] & 63, s[5] & 63, s[6] & 63, s[7] & 63,
                (s[8] >> 4) | ((s[4] >> 6) << 4), (s[9] >> 4) | ((s[5] >> 6) << 4),
                (s[10] >> 4) | ((s[6] >> 6) << 4), (s[11] >> 4) | ((s[7] >> 6) << 4),
            ],
            dtype=np.float32,
        )
        qs = blk[16:144].astype(np.uint16)
        byte = qs[byte_idx]
        qval = (byte >> (4 * half)) & 0xF
        out[b * 256:b * 256 + 256] = d * sc[pair] * qval.astype(np.float32) - dmin * mn[pair]
    return out


def dequant_q6k(w: np.ndarray) -> np.ndarray:
    nb = len(w) // 210
    out = np.empty(nb * 256, dtype=np.float32)
    for b in range(nb):
        blk = w[b * 210:(b + 1) * 210]
        ql = blk[0:128].astype(np.int32)
        qh = blk[128:192].astype(np.int32)
        sc = blk[192:208].astype(np.int8).astype(np.float32)
        d = np.frombuffer(blk[208:210], dtype=np.float16).item()
        for idx in range(2):
            qlo = ql[64 * idx:64 * idx + 32]
            qhi = ql[64 * idx + 32:64 * idx + 64]
            qhh = qh[32 * idx:32 * idx + 32]
            sch = sc[8 * idx:8 * idx + 8]
            q1 = ((qlo & 0xF) | ((qhh & 3) << 4)).astype(np.float32) - 32.0
            q2 = ((qhi & 0xF) | (((qhh >> 2) & 3) << 4)).astype(np.float32) - 32.0
            q3 = ((qlo >> 4) | (((qhh >> 4) & 3) << 4)).astype(np.float32) - 32.0
            q4 = ((qhi >> 4) | (((qhh >> 6) & 3) << 4)).astype(np.float32) - 32.0
            s1 = np.repeat(sch[0:2], 16)
            s2 = np.repeat(sch[2:4], 16)
            s3 = np.repeat(sch[4:6], 16)
            s4 = np.repeat(sch[6:8], 16)
            base = b * 256 + idx * 128
            out[base:base + 32] = d * s1 * q1
            out[base + 32:base + 64] = d * s2 * q2
            out[base + 64:base + 96] = d * s3 * q3
            out[base + 96:base + 128] = d * s4 * q4
    return out


# CPU-reference cross-checks (transcribed from candle-core k_quants.rs).


def _get_scale_min_k4(j: int, q):
    if j < 4:
        return (int(q[j]) & 63, int(q[j + 4]) & 63)
    return (
        (int(q[j + 4]) & 0xF) | ((int(q[j - 4]) >> 6) << 4),
        (int(q[j + 4]) >> 4) | ((int(q[j]) >> 6) << 4),
    )


def dequant_q4k_cpu(w: np.ndarray) -> np.ndarray:
    nb = len(w) // 144
    out = np.empty(nb * 256, dtype=np.float32)
    for b in range(nb):
        blk = w[b * 144:(b + 1) * 144]
        d = np.frombuffer(blk[0:2], dtype=np.float16).item()
        mn = np.frombuffer(blk[2:4], dtype=np.float16).item()
        scales = blk[4:16]
        qs = blk[16:144]
        yi = 0
        is_ = 0
        for j in range(0, 256, 64):
            qq = qs[j // 2:j // 2 + 32]
            sc0, m0 = _get_scale_min_k4(is_, scales)
            d1 = d * np.float32(sc0)
            m1 = mn * np.float32(m0)
            sc1, m1_ = _get_scale_min_k4(is_ + 1, scales)
            d2 = d * np.float32(sc1)
            m2 = mn * np.float32(m1_)
            for q in qq:
                out[b * 256 + yi] = d1 * np.float32(q & 0xF) - m1
                yi += 1
            for q in qq:
                out[b * 256 + yi] = d2 * np.float32(q >> 4) - m2
                yi += 1
            is_ += 2
    return out


def dequant_q6k_cpu(w: np.ndarray) -> np.ndarray:
    nb = len(w) // 210
    out = np.empty(nb * 256, dtype=np.float32)
    for b in range(nb):
        blk = w[b * 210:(b + 1) * 210]
        ql = blk[0:128]
        qh = blk[128:192]
        sc = blk[192:208].astype(np.int8)
        d = np.frombuffer(blk[208:210], dtype=np.float16).item()
        for n in (0, 128):
            idx = n // 128
            scs = sc[8 * idx:8 * idx + 8]
            qls = ql[64 * idx:64 * idx + 64]
            qhs = qh[32 * idx:32 * idx + 32]
            for l in range(32):
                is_ = l // 16
                q1 = (int(qls[l]) & 0xF) | ((int(qhs[l]) & 3) << 4)
                q2 = (int(qls[l + 32]) & 0xF) | (((int(qhs[l]) >> 2) & 3) << 4)
                q3 = (int(qls[l]) >> 4) | (((int(qhs[l]) >> 4) & 3) << 4)
                q4 = (int(qls[l + 32]) >> 4) | (((int(qhs[l]) >> 6) & 3) << 4)
                e = b * 256 + n
                out[e + l] = d * np.float32(scs[is_]) * np.float32(q1 - 32)
                out[e + l + 32] = d * np.float32(scs[is_ + 2]) * np.float32(q2 - 32)
                out[e + l + 64] = d * np.float32(scs[is_ + 4]) * np.float32(q3 - 32)
                out[e + l + 96] = d * np.float32(scs[is_ + 6]) * np.float32(q4 - 32)
    return out


def f16_bytes(v: float) -> bytes:
    return struct.pack("<e", v)


# ------------------------------------------------------------- helpers


def head_of(x: np.ndarray, n: int = 16) -> list:
    return [float(v) for v in x[: min(n, len(x))].ravel()]


def stats(x: np.ndarray) -> tuple:
    x64 = x.astype(np.float64)
    return float((x64 * x64).sum()), float(np.abs(x64).max())


def fmt_f32(v: float) -> str:
    return f"{v:.9e}"


def fmt_arr(xs: list) -> str:
    return "[" + ", ".join(fmt_f32(v) for v in xs) + "]"


def fmt_f64(v: float) -> str:
    s = f"{v:.17g}"
    if not any(c in s for c in ".eE") and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


EMIT = []


def emit_const(name: str, x: np.ndarray):
    h, (s, m) = head_of(x), stats(x)
    EMIT.append(f"pub const {name}: Exp = exp!(&{fmt_arr(h)}, {fmt_f64(s)}, {fmt_f64(m)});")
    print(f"{name}: size={x.size} head[:4]={h[:4]} sumsq={s:.6g} maxabs={m:.6g}")


# ---------------------------------------------------------------- tests


def main():
    # 0: affine: out = in * 2.5 + (-1.75)
    rng = Rng(1000)
    x = rng.f32s(256, -4.0, 4.0)
    emit_const("AFFINE", x * np.float32(2.5) + np.float32(-1.75))

    # 1: copy: src (4,16) -> dst view (4,8) at cols 2..10 of a (4,10) buffer
    rng = Rng(1001)
    src = rng.f32s(64, -8.0, 8.0)
    dst = np.zeros(40, dtype=np.float32)
    for i in range(4):
        for j in range(8):
            dst[i * 10 + 2 + j] = src[i * 16 + j]
    emit_const("COPY", dst)

    # 2-5: elementwise binary, lhs (16,8), rhs (8,)
    for idx, op in enumerate(("add", "sub", "mul", "div")):
        rng = Rng(1002 + idx)
        lhs = rng.f32s(128, -2.0, 2.0)
        if op == "div":
            rhs = rng.f32s(8, 0.5, 1.5)
        else:
            rhs = rng.f32s(8, -1.0, 1.0)
        l, r = lhs.reshape(16, 8), rhs
        if op == "add":
            out = l + r
        elif op == "sub":
            out = l - r
        elif op == "mul":
            out = l * r
        else:
            out = l / r
        emit_const(f"ELEM_{op.upper()}", out)

    # 6-12: elementwise unary, 128 elements
    unary = (
        ("SIGMOID", 1006, -3.0, 3.0, lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("SILU", 1007, -3.0, 3.0, lambda x: x / (1.0 + np.exp(-x))),
        ("EXP", 1008, -2.0, 2.0, lambda x: np.exp(x)),
        ("SQRT", 1009, 0.0, 9.0, lambda x: np.sqrt(x)),
        ("SIN", 1010, -3.0, 3.0, lambda x: np.sin(x)),
        ("COS", 1011, -3.0, 3.0, lambda x: np.cos(x)),
        ("NEG", 1012, -3.0, 3.0, lambda x: -x),
    )
    for name, seed, lo, hi, fn in unary:
        rng = Rng(seed)
        x = rng.f32s(128, lo, hi)
        emit_const(f"ELEM_{name}", fn(x).astype(np.float32))

    # 13: gather: out[i, :] = emb[ids[i], :]
    rng = Rng(1013)
    ids = np.array([rng.u32_mod(32) for _ in range(12)], dtype=np.uint32)
    emb = rng.f32s(32 * 16, -2.0, 2.0).reshape(32, 16)
    emit_const("GATHER", emb[ids])

    # 14: gemm: bsz=2 m=32 k=60 n=40 (k, n not multiples of the 16 tile)
    rng = Rng(1014)
    bsz, m, k, n = 2, 32, 60, 40
    lhs = rng.f32s(bsz * m * k, -1.0, 1.0).reshape(bsz, m, k)
    rhs = rng.f32s(bsz * k * n, -1.0, 1.0).reshape(bsz, k, n)
    rhs_t = rng.f32s(bsz * n * k, -1.0, 1.0).reshape(bsz, n, k)
    emit_const("GEMM_NT", np.einsum("bik,bkj->bij", lhs, rhs))
    emit_const("GEMM_T", np.einsum("bik,bjk->bij", lhs, rhs_t))

    # 15: gemv: a (k,) @ w (k, n)
    rng = Rng(1015)
    k, n = 64, 100
    a = rng.f32s(k, -1.0, 1.0)
    w = rng.f32s(k * n, -1.0, 1.0).reshape(k, n)
    emit_const("GEMV", a @ w)

    # 16: gemv_t: a (k,) @ w (n, k)^T, w contiguous (n, k), w_stride = k
    rng = Rng(1016)
    n, k = 96, 128
    a = rng.f32s(k, -1.0, 1.0)
    w = rng.f32s(n * k, -1.0, 1.0).reshape(n, k)
    emit_const("GEMV_T", a @ w.T)

    # 17: q4k_qmatvec: a (512,) vs w (8, 512) Q4_K bytes, d=2.5 dmin=-1.25 (Q4K_DPATCH[0])
    rng = Rng(1017)
    k, n = 512, 8
    a = rng.f32s(k, -1.0, 1.0)
    wb = rng.bytes(n * 2 * 144)
    qb = f16_bytes(2.5) + f16_bytes(-1.25)  # Q4K_DPATCH[0], as the test applies
    for b in range(n * 2):
        wb[b * 144:b * 144 + 4] = np.frombuffer(qb, dtype=np.uint8)
    deq = dequant_q4k(wb).reshape(n, k)
    emit_const("Q4K_QMATVEC", deq @ a)
    # cross-check the two dequant implementations
    assert np.abs(dequant_q4k(wb) - dequant_q4k_cpu(wb)).max() < 1e-3, "q4k xcheck"

    # 18: q4k_dequant: 3 blocks, fixed (d, dmin) per block
    q4k_dd = [(2.5, -1.25), (-3.25, 0.5), (4.0, -2.0)]
    rng = Rng(1018)
    wb = rng.bytes(3 * 144)
    for b, (d, dmin) in enumerate(q4k_dd):
        wb[b * 144:b * 144 + 4] = np.frombuffer(f16_bytes(d) + f16_bytes(dmin), dtype=np.uint8)
    emit_const("Q4K_DEQUANT", dequant_q4k(wb))
    assert np.abs(dequant_q4k(wb) - dequant_q4k_cpu(wb)).max() < 1e-3, "q4k deq xcheck"

    # 19: q6k_dequant: 2 blocks, fixed d per block
    q6k_d = [1.75, -2.25]
    rng = Rng(1019)
    wb = rng.bytes(2 * 210)
    for b, d in enumerate(q6k_d):
        wb[b * 210 + 208:b * 210 + 210] = np.frombuffer(f16_bytes(d), dtype=np.uint8)
    emit_const("Q6K_DEQUANT", dequant_q6k(wb))
    assert np.abs(dequant_q6k(wb) - dequant_q6k_cpu(wb)).max() < 1e-3, "q6k xcheck"

    # 20: reduce_sum: (32, 64) row sums
    rng = Rng(1020)
    x = rng.f32s(32 * 64, -1.0, 1.0).reshape(32, 64)
    emit_const("REDUCE_SUM", x.sum(axis=1, dtype=np.float32))

    # 21: reduce_max: (32, 64) row maxes
    rng = Rng(1021)
    x = rng.f32s(32 * 64, -1.0, 1.0).reshape(32, 64)
    emit_const("REDUCE_MAX", x.max(axis=1))

    # 22: rms_norm: (16, 64), weight (64,), eps = 1e-5
    rng = Rng(1022)
    x = rng.f32s(16 * 64, -2.0, 2.0).reshape(16, 64)
    w = rng.f32s(64, 0.5, 1.5)
    mean_sq = x.astype(np.float32) ** 2
    mean_sq = mean_sq.mean(axis=1, keepdims=True)
    out = x * (1.0 / np.sqrt(mean_sq + np.float32(1e-5))) * w
    emit_const("RMS_NORM", out.astype(np.float32))

    # 23: softmax: (24, 56) stable row softmax
    rng = Rng(1023)
    x = rng.f32s(24 * 56, -5.0, 5.0).reshape(24, 56)
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    emit_const("SOFTMAX", (e / e.sum(axis=1, keepdims=True)).astype(np.float32))

    # 24: rope: (b,h,t,d) = (2,4,8,16); cos/sin (t, d/2) batched and (b,t,d/2) unbatched
    rng = Rng(1024)
    b, h, t, d = 2, 4, 8, 16
    half = d // 2
    x = rng.f32s(b * h * t * d, -2.0, 2.0).reshape(b, h, t, d)
    cos = rng.f32s(t * half, -1.0, 1.0).reshape(t, half)
    sin = rng.f32s(t * half, -1.0, 1.0).reshape(t, half)
    cos_u = rng.f32s(b * t * half, -1.0, 1.0).reshape(b, t, half)
    sin_u = rng.f32s(b * t * half, -1.0, 1.0).reshape(b, t, half)

    def rope_apply(x, cos, sin, unbatched: bool) -> np.ndarray:
        out = np.empty_like(x)
        for bi in range(b):
            for hi_ in range(h):
                for ti in range(t):
                    row = (bi * h + hi_) * t + ti
                    i1 = row * d + np.arange(half)
                    i2 = i1 + half
                    if unbatched:
                        i_cs = bi * t * half + ti * half + np.arange(half)
                        c = cos_u.ravel()[i_cs]
                        s = sin_u.ravel()[i_cs]
                    else:
                        i_cs = ti * half + np.arange(half)
                        c = cos.ravel()[i_cs]
                        s = sin.ravel()[i_cs]
                    out.ravel()[i1] = x.ravel()[i1] * c - x.ravel()[i2] * s
                    out.ravel()[i2] = x.ravel()[i1] * s + x.ravel()[i2] * c
        return out

    emit_const("ROPE", rope_apply(x, cos, sin, False))
    emit_const("ROPE_U", rope_apply(x, cos_u, sin_u, True))

    # ------------------------------------------------------- emit file
    q4k_patch = [
        (f16_bytes(dd[0])[0], f16_bytes(dd[0])[1], f16_bytes(dd[1])[0], f16_bytes(dd[1])[1])
        for dd in q4k_dd
    ]
    q6k_patch = [(f16_bytes(d)[0], f16_bytes(d)[1]) for d in q6k_d]
    patch_q4k = ", ".join(f"({dl}, {dh}, {ml}, {mh})" for dl, dh, ml, mh in q4k_patch)
    patch_q6k = ", ".join(f"({dl}, {dh})" for dl, dh in q6k_patch)

    body = "\n".join(EMIT)
    file = f"""// GENERATED by gen_expected.py - do not edit by hand.
//
// Hardcoded expectations for tests/vulkan_integration_tests.rs. The inputs
// are regenerated at runtime by the deterministic SplitMix64 `Rng` in that
// file (seed 1000 + test index); these constants are the float32 outputs
// computed by numpy with the exact operation order of each shader.

pub struct Exp {{
    /// Leading elements of the expected output (all, when the output has
    /// 16 or fewer elements).
    pub head: &'static [f32],
    /// Sum of squares over the whole expected output (f64 accumulation).
    pub sumsq: f64,
    /// Max absolute value over the whole expected output.
    pub maxabs: f64,
}}

macro_rules! exp {{
    ($h:expr, $s:expr, $m:expr) => {{
        Exp {{ head: $h, sumsq: $s, maxabs: $m }}
    }};
}}

// Q4_K per-block (d, dmin) f16 little-endian byte patches applied by the
// test to the random byte sequences (block b at offset 144*b).
pub const Q4K_DPATCH: &[(u8, u8, u8, u8)] = &[{patch_q4k}];
// Q6_K per-block d f16 little-endian byte patches (block b at offset 210*b).
pub const Q6K_DPATCH: &[(u8, u8)] = &[{patch_q6k}];

{body}
"""
    with open("vulkan_expected.rs", "w") as f:
        f.write(file)
    print(f"\nwrote vulkan_expected.rs ({len(file)} bytes, {len(EMIT)} expectations)")


if __name__ == "__main__":
    main()
