//! WGSL compute kernels for the Vulkan backend: the few that need what
//! naga's GLSL front end cannot express — the packed int8 dot product
//! (`dot4I8Packed`, emitted as SPIR-V `OpSDot`).  Compiled with
//! [`super::kernels::wgsl_to_spirv`]; binding and push-constant conventions
//! follow [`super::glsl`] (tensor buffers from binding 0, the parameter
//! buffer after them, which these kernels do not read).

use super::glsl::{QGEMV_MALI_COLS, QGEMV_MALI_ROWS, QGEMV_MALI_WG};

/// Byte-level access to the weight blocks (`W`, packed `u32` words) and the
/// per-format unpacking of one 32-weight sub-block into eight words of
/// packed signed bytes plus, for each 16-weight half `h`, the affine terms
/// `A[h]`, `B[h]` with `w = A[h]·q + B[h]` — the same fields and layouts as
/// `dequant_sub` in [`super::glsl`], in integers.
const UNPACK_FN: &str = r#"
fn byte_at(b: u32) -> u32 { return (W[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
// Four bytes from any byte offset (blocks are not word-aligned).
fn word_at(b: u32) -> u32 {
    let s = (b & 3u) * 8u;
    let i = b >> 2u;
    if (s == 0u) { return W[i]; }
    return (W[i] >> s) | (W[i + 1u] << (32u - s));
}
fn f16_at(b: u32) -> f32 { return unpack2x16float(byte_at(b) | (byte_at(b + 1u) << 8u)).x; }
fn i8_at(b: u32) -> i32 { return (i32(byte_at(b)) << 24u) >> 24u; }
// Bits 0..3 of `h` to bit 4 of each byte (the fifth bit of Q5_0 / Q5_1).
fn spread4(h: u32) -> u32 { return ((h & 1u) << 4u) | ((h & 2u) << 11u) | ((h & 4u) << 18u) | ((h & 8u) << 25u); }
fn k4_scale(sc: u32, is: u32) -> vec2<u32> {
    if (is < 4u) { return vec2<u32>(byte_at(sc + is) & 63u, byte_at(sc + is + 4u) & 63u); }
    return vec2<u32>((byte_at(sc + is + 4u) & 0xFu) | ((byte_at(sc + is - 4u) >> 6u) << 4u),
                     (byte_at(sc + is + 4u) >> 4u) | ((byte_at(sc + is) >> 6u) << 4u));
}
struct Sub { q: array<u32, 8>, a: vec2<f32>, b: vec2<f32> }
fn unpack_sub(qt: i32, p: u32, j: u32) -> Sub {
    var s: Sub;
    if (qt == 8 || qt == 9 || qt == 15) { // Q8_0, Q8_1, Q8_K: the bytes are the values
        var d: f32; var q0: u32;
        if (qt == 15) { d = bitcast<f32>(word_at(p)); q0 = p + 4u + 32u * j; }
        else { d = f16_at(p); q0 = p + select(2u, 4u, qt == 9); }
        for (var i = 0u; i < 8u; i++) { s.q[i] = word_at(q0 + 4u * i); }
        s.a = vec2<f32>(d); s.b = vec2<f32>(0.0);
    } else if (qt == 2 || qt == 3) { // Q4_0 / Q4_1: low nibbles are 0..15, high 16..31
        let d = f16_at(p);
        let q0 = p + select(2u, 4u, qt == 3);
        for (var i = 0u; i < 4u; i++) {
            let w = word_at(q0 + 4u * i);
            s.q[i] = w & 0x0F0F0F0Fu;
            s.q[i + 4u] = (w >> 4u) & 0x0F0F0F0Fu;
        }
        s.a = vec2<f32>(d);
        s.b = vec2<f32>(select(-8.0 * d, f16_at(p + 2u), qt == 3));
    } else if (qt == 6 || qt == 7) { // Q5_0 / Q5_1: fifth bit of value e is bit e of qh
        let d = f16_at(p);
        let qh = word_at(p + select(2u, 4u, qt == 7));
        let q0 = p + select(6u, 8u, qt == 7);
        for (var i = 0u; i < 4u; i++) {
            let w = word_at(q0 + 4u * i);
            s.q[i] = (w & 0x0F0F0F0Fu) | spread4(qh >> (4u * i));
            s.q[i + 4u] = ((w >> 4u) & 0x0F0F0F0Fu) | spread4(qh >> (16u + 4u * i));
        }
        s.a = vec2<f32>(d);
        s.b = vec2<f32>(select(-16.0 * d, f16_at(p + 2u), qt == 7));
    } else if (qt == 12 || qt == 13) { // Q4_K / Q5_K: 6-bit scale and min per 32
        let g = j >> 1u;
        let hf = j & 1u;
        let sm = k4_scale(p + 4u, 2u * g + hf);
        let d1 = f16_at(p) * f32(sm.x);
        let m1 = f16_at(p + 2u) * f32(sm.y);
        let ql = p + select(16u, 48u, qt == 13) + 32u * g;
        for (var i = 0u; i < 8u; i++) {
            var q = (word_at(ql + 4u * i) >> (4u * hf)) & 0x0F0F0F0Fu;
            if (qt == 13) { q |= ((word_at(p + 16u + 4u * i) >> (2u * g + hf)) & 0x01010101u) << 4u; }
            s.q[i] = q;
        }
        s.a = vec2<f32>(d1); s.b = vec2<f32>(-m1);
    } else if (qt == 14) { // Q6_K: int8 scale per 16, values 0..63 offset by 32
        let hh = j >> 2u;
        let qq = j & 3u;
        let ql = p + 64u * hh + 32u * (qq & 1u);
        let qh = p + 128u + 32u * hh;
        let sc = p + 192u + 8u * hh + 2u * qq;
        let d = f16_at(p + 208u);
        for (var i = 0u; i < 8u; i++) {
            let nib = (word_at(ql + 4u * i) >> (4u * (qq >> 1u))) & 0x0F0F0F0Fu;
            s.q[i] = nib | (((word_at(qh + 4u * i) >> (2u * qq)) & 0x03030303u) << 4u);
        }
        let dl = vec2<f32>(d * f32(i8_at(sc)), d * f32(i8_at(sc + 1u)));
        s.a = dl; s.b = -32.0 * dl;
    } else if (qt == 10) { // Q2_K: 4-bit scale and min per 16
        let hh = j >> 2u;
        let m = j & 3u;
        let sc = p + 8u * hh + 2u * m;
        let qs = p + 16u + 32u * hh;
        let d = f16_at(p + 80u);
        let dmin = f16_at(p + 82u);
        for (var i = 0u; i < 8u; i++) { s.q[i] = (word_at(qs + 4u * i) >> (2u * m)) & 0x03030303u; }
        let s0 = byte_at(sc);
        let s1 = byte_at(sc + 1u);
        s.a = vec2<f32>(d * f32(s0 & 0xFu), d * f32(s1 & 0xFu));
        s.b = vec2<f32>(-dmin * f32(s0 >> 4u), -dmin * f32(s1 >> 4u));
    } else { // Q3_K (11): 6-bit signed scale per 16, values 0..7 offset by 4
        let aux0 = word_at(p + 96u); let aux1 = word_at(p + 100u); let aux2 = word_at(p + 104u);
        let a0 = (aux0 & 0x0f0f0f0fu) | (((aux2 >> 0u) & 0x03030303u) << 4u);
        let a1 = (aux1 & 0x0f0f0f0fu) | (((aux2 >> 2u) & 0x03030303u) << 4u);
        let a2 = ((aux0 >> 4u) & 0x0f0f0f0fu) | (((aux2 >> 4u) & 0x03030303u) << 4u);
        let a3 = ((aux1 >> 4u) & 0x0f0f0f0fu) | (((aux2 >> 6u) & 0x03030303u) << 4u);
        let hh = j >> 2u;
        let m = j & 3u;
        let is = 8u * hh + 2u * m;
        var aw = array<u32, 4>(a0, a1, a2, a3);
        let v0 = (aw[is / 4u] >> ((is % 4u) * 8u)) & 0xFFu;
        let v1 = (aw[(is + 1u) / 4u] >> (((is + 1u) % 4u) * 8u)) & 0xFFu;
        let d = f16_at(p + 108u);
        let dl = vec2<f32>(d * f32(((i32(v0) << 24u) >> 24u) - 32), d * f32(((i32(v1) << 24u) >> 24u) - 32));
        let qs = p + 32u + 32u * hh;
        for (var i = 0u; i < 8u; i++) {
            let lo = (word_at(qs + 4u * i) >> (2u * m)) & 0x03030303u;
            let hi = (word_at(p + 4u * i) >> (4u * hh + m)) & 0x01010101u;
            s.q[i] = lo | (hi << 2u);
        }
        s.a = dl; s.b = -4.0 * dl;
    }
    return s;
}
"#;

/// [`super::glsl::k_qgemv_mali`] on int8 dot products: `C[m, n] = sum_k
/// X[m, k] · W[n, k]` with the activations already quantized per 32 values
/// ([`super::glsl::k_quant_x8`]: packed int8 `XQ`, scales `XS`, per-16 sums
/// `XSUM`).  Each weight sub-block unpacks to packed signed bytes `q` with
/// `w = A·q + B` per 16, so a sub-block's contribution for one row is
/// `XS · Σₕ (Aₕ·Σ q·xq + Bₕ·Σ xq)` — four `dot4I8Packed` per 16 values.
///
/// `hw_dot` selects the hardware instruction (`OpSDot`, needing the
/// `shaderIntegerDotProduct` feature); without it the same kernel uses a
/// shift-and-multiply polyfill, bit-identical.
///
/// Push: `N, K, qt, qk, bsz, coff, M`.  Grid as for `k_qgemv_mali`.
/// Bindings: `XQ, XS, XSUM, W, C`.
pub fn k_qgemv_mali_i8(hw_dot: bool) -> String {
    let dot = if hw_dot {
        "fn dot4(a: u32, b: u32) -> i32 { return dot4I8Packed(a, b); }"
    } else {
        "fn dot4(a: u32, b: u32) -> i32 {
    var s = 0;
    for (var i = 0u; i < 32u; i += 8u) { s += ((bitcast<i32>(a << (24u - i)) >> 24u) * (bitcast<i32>(b << (24u - i)) >> 24u)); }
    return s;
}"
    };
    format!(
        r#"
struct PC {{ N: i32, K: i32, qt: i32, qk: i32, bsz: i32, coff: i32, M: i32 }}
var<push_constant> pc: PC;
@group(0) @binding(0) var<storage, read> XQ: array<u32>;
@group(0) @binding(1) var<storage, read> XS: array<f32>;
@group(0) @binding(2) var<storage, read> XSUM: array<i32>;
@group(0) @binding(3) var<storage, read> W: array<u32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;
const WG: u32 = {wg}u;
const TEAM: u32 = {team}u;
const COLS: u32 = {cols}u;
const RB: u32 = {rows}u;
var<workgroup> sh: array<f32, {sh}>;
{dot}
{unpack}
@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let t = lid.x;
    let lane = t % TEAM;
    let n = i32((wid.x + wid.y * nwg.x) * COLS + t / TEAM);
    let m0 = i32(wid.z * RB);
    var acc = vec4<f32>(0.0);
    if (n < pc.N) {{
        let spb = pc.qk / 32;
        let nsub = pc.K / 32;
        let row = u32(n) * u32(pc.K / pc.qk) * u32(pc.bsz);
        // Rows past M re-read the last row; their results are dropped.
        let last = pc.M - 1 - m0;
        let rows = vec4<i32>(m0, m0 + min(1, last), m0 + min(2, last), m0 + min(3, last));
        for (var s = i32(lane); s < nsub; s += i32(TEAM)) {{
            let b = s / spb;
            let sub = unpack_sub(pc.qt, row + u32(b) * u32(pc.bsz), u32(s - b * spb));
            for (var k = 0; k < 4; k++) {{
                let r = rows[k];
                let xw = u32(r * (pc.K / 4) + s * 8);
                var d0 = 0;
                var d1 = 0;
                for (var i = 0u; i < 4u; i++) {{
                    d0 += dot4(sub.q[i], XQ[xw + i]);
                    d1 += dot4(sub.q[i + 4u], XQ[xw + 4u + i]);
                }}
                let hs = r * (pc.K / 16) + 2 * s;
                let dx = XS[r * nsub + s];
                acc[k] += dx * ((sub.a.x * f32(d0) + sub.b.x * f32(XSUM[hs])) + (sub.a.y * f32(d1) + sub.b.y * f32(XSUM[hs + 1])));
            }}
        }}
    }}
    sh[t] = acc.x;
    sh[WG + t] = acc.y;
    sh[2u * WG + t] = acc.z;
    sh[3u * WG + t] = acc.w;
    workgroupBarrier();
    if (lane == 0u && n < pc.N) {{
        for (var k = 0; k < i32(RB); k++) {{
            if (m0 + k < pc.M) {{
                let h = u32(k) * WG + t;
                C[pc.coff + (m0 + k) * pc.N + n] = (sh[h] + sh[h + 1u]) + (sh[h + 2u] + sh[h + 3u]);
            }}
        }}
    }}
}}
"#,
        wg = QGEMV_MALI_WG,
        team = QGEMV_MALI_WG / QGEMV_MALI_COLS,
        cols = QGEMV_MALI_COLS,
        rows = QGEMV_MALI_ROWS,
        sh = QGEMV_MALI_ROWS * QGEMV_MALI_WG,
        unpack = UNPACK_FN,
    )
}
