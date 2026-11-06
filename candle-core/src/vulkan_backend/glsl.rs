//! GLSL compute kernels for the Vulkan backend, generated as source strings
//! and compiled to SPIR-V at runtime with `naga`.
//!
//! The kernel set mirrors the OpenCL backend's `kernels.cl` op for op:
//! strided / broadcast elementwise ops, strided copies, row reductions,
//! gathers, comparisons, casts, a tiled GEMM plus GEMVs, the fused
//! softmax / RMSNorm / RoPE the attention path calls, and the quantized
//! dequantize-in-kernel GEMV / GEMM over GGUF block formats.
//!
//! Conventions
//! * Every kernel is one complete compute shader (`main`).  Its tensor
//!   arguments are `std430` storage buffers bound in order from binding 0;
//!   scalar arguments are push constants (at most 128 bytes, the guaranteed
//!   minimum); the index descriptors ([`Idx`]) are the last binding,
//!   a small parameter buffer written per launch.
//! * Element indices are `int`; tensors stay below 2^31 elements.
//! * Buffers are typed by element width: `float`/`uint`/`int` for 4-byte
//!   elements, `uvec2` for 8-byte ones, and packed `uint` words for 1- and
//!   2-byte elements.  naga's GLSL front end has no atomics and 8/16-bit
//!   storage is optional, so kernels that *write* packed elements own a
//!   whole word per invocation (read-modify-write of the word is race-free).
//! * `WG` (the work-group size of the row / reduction kernels) is chosen
//!   from the device limit at pipeline build time; the elementwise kernels
//!   use it too and spread large launches over a 2-D grid (`gid()`).
//! * The generic index descriptor `Idx` carries the output shape and up to
//!   three input stride sets so one kernel serves every broadcast /
//!   transposed / narrowed view; outputs are written contiguous row-major.
//!
//! [`Idx`]: super::kernels::Idx

/// Number of dims the index descriptor carries (must match `kernels::MAXD`).
pub const MAXD: usize = 8;

/// The common prelude: version, the `Idx` descriptor, the parameter buffer
/// (always binding `{P}`), and the helpers.
/// The common kernel header: `#version`, `WG`, the `Idx` parameter block
/// bound at `nbuf` (after the kernel's own `nbuf` buffers) and the `gid()` /
/// `grow()` / `off*()` helpers. Public for kernels defined outside candle.
pub fn prelude(wg: usize, nbuf: usize) -> String {
    format!(
        r#"#version 450
#define WG {wg}
#define MAXD {maxd}
struct Idx {{ int nd; int dims[MAXD]; int s0[MAXD]; int s1[MAXD]; int s2[MAXD]; int o0; int o1; int o2; }};
layout(std430, set = 0, binding = {p}) readonly buffer ParamsB {{ Idx ix; Idx rd; }} p;
float INF() {{ return uintBitsToFloat(0x7F800000u); }}
float NINF() {{ return uintBitsToFloat(0xFF800000u); }}
// Linear index of this invocation over a 2-D grid of 1-D work-groups.
int gid() {{ return int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * uint(WG))); }}
// Row index for one-work-group-per-row kernels (2-D grid of groups).
int grow() {{ return int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x); }}
{off0}
{off1}
{off2}
{offr}
"#,
        p = nbuf,
        maxd = MAXD,
        off0 = off_fn("off0", "ix", "s0", "o0"),
        off1 = off_fn("off1", "ix", "s1", "o1"),
        off2 = off_fn("off2", "ix", "s2", "o2"),
        offr = off_fn("offr", "rd", "s0", "0"),
    )
}

/// `int NAME(int lin)`: storage offset of linear output index `lin` under
/// stride set `S` / offset `O` of descriptor `D`.
fn off_fn(name: &str, d: &str, s: &str, o: &str) -> String {
    let o = if o == "0" {
        "0".to_string()
    } else {
        format!("p.{d}.{o}")
    };
    format!(
        "int {name}(int lin) {{ int off = {o}; for (int d = p.{d}.nd - 1; d >= 0; d--) {{ int dim = p.{d}.dims[d]; int i = lin % dim; lin /= dim; off += i * p.{d}.{s}[d]; }} return off; }}"
    )
}

fn buf(binding: usize, ty: &str, name: &str, ro: bool) -> String {
    let q = if ro { "readonly " } else { "" };
    format!(
        "layout(std430, set = 0, binding = {binding}) {q}buffer B{binding} {{ {ty} {name}[]; }};\n"
    )
}

/// IEEE f32 → f16 bits with round-to-nearest-even (including subnormals,
/// overflow to infinity and NaN propagation), independent of the driver's
/// `packHalf2x16` rounding mode.
const F32_TO_F16_RNE: &str = r#"
uint f32_to_f16_rne(float f) {
    uint x = floatBitsToUint(f);
    uint sgn = (x >> 16u) & 0x8000u;
    uint e = (x >> 23u) & 0xFFu;
    uint m = x & 0x7FFFFFu;
    if (e == 0xFFu) { return sgn | 0x7C00u | (m != 0u ? (0x200u | (m >> 13u)) : 0u); }
    int ne = int(e) - 112;
    if (ne >= 31) { return sgn | 0x7C00u; }
    if (ne <= 0) {
        if (ne < -10) { return sgn; }
        m |= 0x800000u;
        uint sh = uint(14 - ne);
        uint hm = m >> sh;
        uint rem = m & ((1u << sh) - 1u);
        uint halfway = 1u << (sh - 1u);
        if (rem > halfway || (rem == halfway && (hm & 1u) != 0u)) { hm += 1u; }
        return sgn | hm;
    }
    uint hm = m >> 13u;
    uint rem = m & 0x1FFFu;
    uint h = sgn | (uint(ne) << 10u) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u) != 0u)) { h += 1u; }
    return h;
}
"#;

/// Unary float helpers shared by the elementwise kernels.
const UNARY_FN: &str = r#"
float erf_f(float x) {
    // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
    float s = x < 0.0 ? -1.0 : 1.0;
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
float round_half_away(float v) { return sign(v) * floor(abs(v) + 0.5); }
float unary_f(float v, int op, float f0, float f1) {
    switch (op) {
        case 0: return exp(v);
        case 1: return log(v);
        case 2: return sin(v);
        case 3: return cos(v);
        case 4: return tanh(v);
        case 5: return -v;
        case 6: return 1.0 / v;
        case 7: return v * v;
        case 8: return sqrt(v);
        case 9: return 0.5 * v * (1.0 + tanh(0.79788456080286535588 * v * (1.0 + 0.044715 * v * v)));
        case 10: return 0.5 * v * (1.0 + erf_f(v * 0.70710678118654752440));
        case 11: return erf_f(v);
        case 12: return v / (1.0 + exp(-v));
        case 13: return abs(v);
        case 14: return ceil(v);
        case 15: return floor(v);
        case 16: return round_half_away(v);
        case 17: return v > 0.0 ? v : 0.0;
        case 18: return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
        case 19: return 1.0 / (1.0 + exp(-v));
        case 20: return v * f0 + f1;           // affine
        case 21: return pow(v, f0);            // powf
        case 22: return v > 0.0 ? v : f0 * (exp(v) - 1.0); // elu
        default: return v;
    }
}
"#;

/// Elementwise float unary op (also affine / powf / elu via `op`):
/// `o[i] = f(x[idx(i)])`.  Push: `n, contig, off, op, f0, f1`.
pub fn k_unary(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += UNARY_FN;
    s += r#"
layout(push_constant) uniform PC { int n; int contig; int off; int op; float f0; float f1; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    int src = pc.contig != 0 ? pc.off + i : off0(i);
    o[i] = unary_f(x[src], pc.op, pc.f0, pc.f1);
}
"#;
    s
}

const BINARY_FN: &str = r#"
float binary_f(float a, float b, int op) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return a > b ? b : a;
        case 5: return a < b ? b : a;
        default: return a;
    }
}
uint binary_u(uint x, uint y, int op) {
    switch (op) {
        case 0: return x + y;
        case 1: return x - y;
        case 2: return x * y;
        case 3: return y == 0u ? 0u : x / y;
        case 4: return x > y ? y : x;
        case 5: return x < y ? y : x;
        default: return x;
    }
}
"#;

/// Binary op over f32 (`is_u32 == 0`) or u32 buffers, strided or contiguous.
/// Push: `n, contig, oa, ob, op`.
pub fn k_binary(wg: usize, u32: bool) -> String {
    let ty = if u32 { "uint" } else { "float" };
    let mut s = prelude(wg, 3);
    s += &buf(0, ty, "a", true);
    s += &buf(1, ty, "b", true);
    s += &buf(2, ty, "o", false);
    s += BINARY_FN;
    s += "layout(push_constant) uniform PC { int n; int contig; int oa; int ob; int op; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int i = gid();\n    if (i >= pc.n) return;\n    int ia = pc.contig != 0 ? pc.oa + i : off0(i);\n    int ib = pc.contig != 0 ? pc.ob + i : off1(i);\n";
    s += if u32 {
        "    o[i] = binary_u(a[ia], b[ib], pc.op);\n"
    } else {
        "    o[i] = binary_f(a[ia], b[ib], pc.op);\n"
    };
    s += "}\n";
    s
}

/// Comparison to a packed-u8 output: one invocation per output word
/// (4 elements).  Push: `n, op`.
pub fn k_cmp(wg: usize, u32: bool) -> String {
    let ty = if u32 { "uint" } else { "float" };
    let mut s = prelude(wg, 3);
    s += &buf(0, ty, "a", true);
    s += &buf(1, ty, "b", true);
    s += &buf(2, "uint", "o", false);
    s += &format!(
        r#"
bool cmp_v({ty} x, {ty} y, int op) {{
    switch (op) {{
        case 0: return x == y;
        case 1: return x != y;
        case 2: return x <= y;
        case 3: return x >= y;
        case 4: return x < y;
        default: return x > y;
    }}
}}
layout(push_constant) uniform PC {{ int n; int op; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int w = gid();
    int nw = (pc.n + 3) / 4;
    if (w >= nw) return;
    uint word = 0u;
    for (int k = 0; k < 4; k++) {{
        int i = w * 4 + k;
        if (i < pc.n) {{
            bool r = cmp_v(a[off0(i)], b[off1(i)], pc.op);
            if (r) word |= 1u << (uint(k) * 8u);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// `where_cond`: `o[i] = c[i] ? t[i] : f[i]` with a packed-u8 or u32
/// condition over 4-byte (`uint`) or 8-byte (`uvec2`) payloads.
pub fn k_where(wg: usize, cond_u32: bool, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "c", true);
    s += &buf(1, ty, "t", true);
    s += &buf(2, ty, "f", true);
    s += &buf(3, ty, "o", false);
    let cond = if cond_u32 {
        "uint cv = c[off0(i)];"
    } else {
        "int ci = off0(i); uint cv = (c[ci >> 2] >> (uint(ci & 3) * 8u)) & 0xFFu;"
    };
    s += &format!(
        "layout(push_constant) uniform PC {{ int n; }} pc;\nlayout(local_size_x = WG) in;\nvoid main() {{\n    int i = gid();\n    if (i >= pc.n) return;\n    {cond}\n    o[i] = cv != 0u ? t[off1(i)] : f[off2(i)];\n}}\n"
    );
    s
}

/// Strided copy of 4- or 8-byte elements to a contiguous destination range.
/// Push: `n, dst_off`.
pub fn k_copy_wide(wg: usize, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 2);
    s += &buf(0, ty, "x", true);
    s += &buf(1, ty, "o", false);
    s += "layout(push_constant) uniform PC { int n; int dst_off; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int i = gid();\n    if (i >= pc.n) return;\n    o[pc.dst_off + i] = x[off0(i)];\n}\n";
    s
}

/// Strided copy of 1- or 2-byte elements into a contiguous destination
/// range `[dst_off, dst_off + n)`: one invocation per destination word,
/// keeping the bytes outside the range.  Push: `n, dst_off, w0`
/// (`w0` = first word index).
pub fn k_copy_packed(wg: usize, elem: usize) -> String {
    let (per, bits, mask) = if elem == 1 {
        (4, 8u32, "0xFFu")
    } else {
        (2, 16, "0xFFFFu")
    };
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "x", true);
    s += &buf(1, "uint", "o", false);
    s += &format!(
        r#"
uint src_elem(int e) {{ return (x[e / {per}] >> (uint(e % {per}) * {bits}u)) & {mask}; }}
layout(push_constant) uniform PC {{ int n; int dst_off; int w0; int nw; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int wi = gid();
    if (wi >= pc.nw) return;
    int w = pc.w0 + wi;
    uint word = o[w];
    for (int k = 0; k < {per}; k++) {{
        int e = w * {per} + k - pc.dst_off;
        if (e >= 0 && e < pc.n) {{
            uint v = src_elem(off0(e));
            uint sh = uint(k) * {bits}u;
            word = (word & ~({mask} << sh)) | (v << sh);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// copy2d over 4- or 8-byte elements: `d1` rows of `d2` contiguous elements.
/// Push: `d1, d2, src_s, dst_s, src_o, dst_o`.
pub fn k_copy2d_wide(wg: usize, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 2);
    s += &buf(0, ty, "x", true);
    s += &buf(1, ty, "o", false);
    s += "layout(push_constant) uniform PC { int d1; int d2; int src_s; int dst_s; int src_o; int dst_o; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int idx = gid();\n    if (idx >= pc.d1 * pc.d2) return;\n    int i = idx / pc.d2;\n    int j = idx % pc.d2;\n    o[pc.dst_o + i * pc.dst_s + j] = x[pc.src_o + i * pc.src_s + j];\n}\n";
    s
}

/// copy2d over 1- or 2-byte elements: one invocation per destination word
/// in the bounding range; bytes outside the copied rows are kept.
/// Push: `d1, d2, src_s, dst_s, src_o, dst_o, w0, nw`.
pub fn k_copy2d_packed(wg: usize, elem: usize) -> String {
    let (per, bits, mask) = if elem == 1 {
        (4, 8u32, "0xFFu")
    } else {
        (2, 16, "0xFFFFu")
    };
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "x", true);
    s += &buf(1, "uint", "o", false);
    s += &format!(
        r#"
uint src_elem(int e) {{ return (x[e / {per}] >> (uint(e % {per}) * {bits}u)) & {mask}; }}
layout(push_constant) uniform PC {{ int d1; int d2; int src_s; int dst_s; int src_o; int dst_o; int w0; int nw; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int wi = gid();
    if (wi >= pc.nw) return;
    int w = pc.w0 + wi;
    uint word = o[w];
    for (int k = 0; k < {per}; k++) {{
        int e = w * {per} + k - pc.dst_o;
        if (e < 0) continue;
        int i = e / pc.dst_s;
        int j = e - i * pc.dst_s;
        if (pc.dst_s == 0) {{ i = 0; j = e; }}
        if (i < pc.d1 && j < pc.d2) {{
            uint v = src_elem(pc.src_o + i * pc.src_s + j);
            uint sh = uint(k) * {bits}u;
            word = (word & ~({mask} << sh)) | (v << sh);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// Fill the strided elements of a 4- or 8-byte buffer with a constant.
/// Push: `n, v_lo, v_hi`.
pub fn k_fill_wide(wg: usize, elem8: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 1);
    s += &buf(0, ty, "o", false);
    s += "layout(push_constant) uniform PC { int n; uint lo; uint hi; } pc;\nlayout(local_size_x = WG) in;\nvoid main() {\n    int i = gid();\n    if (i >= pc.n) return;\n";
    s += if elem8 {
        "    o[off0(i)] = uvec2(pc.lo, pc.hi);\n"
    } else {
        "    o[off0(i)] = pc.lo;\n"
    };
    s += "}\n";
    s
}

/// Fill a contiguous range `[off, off + n)` of 1- or 2-byte elements.
/// Push: `n, off, w0, nw, v`.
pub fn k_fill_packed(wg: usize, elem: usize) -> String {
    let (per, bits, mask) = if elem == 1 {
        (4, 8u32, "0xFFu")
    } else {
        (2, 16, "0xFFFFu")
    };
    let mut s = prelude(wg, 1);
    s += &buf(0, "uint", "o", false);
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n; int off; int w0; int nw; uint v; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int wi = gid();
    if (wi >= pc.nw) return;
    int w = pc.w0 + wi;
    uint word = o[w];
    for (int k = 0; k < {per}; k++) {{
        int e = w * {per} + k - pc.off;
        if (e >= 0 && e < pc.n) {{
            uint sh = uint(k) * {bits}u;
            word = (word & ~({mask} << sh)) | ((pc.v & {mask}) << sh);
        }}
    }}
    o[w] = word;
}}
"#
    );
    s
}

/// Storage classes the cast kernel converts between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cls {
    F32,
    U32,
    I32,
    U8,
    F16,
    BF16,
    I64,
    F64,
}

impl Cls {
    fn packed(self) -> Option<(usize, u32, &'static str)> {
        match self {
            Cls::U8 => Some((4, 8, "0xFFu")),
            Cls::F16 | Cls::BF16 => Some((2, 16, "0xFFFFu")),
            _ => None,
        }
    }
    fn buf_ty(self) -> &'static str {
        match self {
            Cls::F32 => "float",
            Cls::U32 | Cls::U8 | Cls::F16 | Cls::BF16 => "uint",
            Cls::I32 => "int",
            Cls::I64 | Cls::F64 => "uvec2",
        }
    }
    /// GLSL expression reading element at storage offset `e` as a `double`
    /// -free intermediate: floats become `float`, ints become `int`/`uint`.
    fn read(self) -> String {
        match self {
            Cls::F32 => "x[e]".into(),
            Cls::U32 => "x[e]".into(),
            Cls::I32 => "x[e]".into(),
            Cls::U8 => "((x[e / 4] >> (uint(e % 4) * 8u)) & 0xFFu)".into(),
            Cls::F16 => "unpackHalf2x16((x[e / 2] >> (uint(e % 2) * 16u)) & 0xFFFFu).x".into(),
            Cls::BF16 => {
                "uintBitsToFloat(((x[e / 2] >> (uint(e % 2) * 16u)) & 0xFFFFu) << 16u)".into()
            }
            Cls::I64 => "x[e]".into(),
            Cls::F64 => "x[e]".into(),
        }
    }
}

/// i64 ids to u32 for the indexing kernels: a negative id or one that does
/// not fit in 32 bits becomes `0xFFFFFFFF`, which every bounds check
/// (against a dimension below 2³¹) rejects and reports through the fault
/// word.  Push: `n`.
pub fn k_ids_i64(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "uvec2", "x", true);
    s += &buf(1, "uint", "o", false);
    s += r#"
layout(push_constant) uniform PC { int n; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    uvec2 r = x[off0(i)];
    o[i] = r.y != 0u ? 0xFFFFFFFFu : r.x;
}
"#;
    s
}

/// `to_dtype` between two storage classes: strided input, contiguous
/// output.  Values go through `float` (for floats) or `int`/`uint` (ints);
/// an i64 becomes a float from both of its words (f32 precision), and the
/// integer casts of an i64 keep its low word (the values candle's tensors
/// hold here: indices, masks).
/// Push: `n` (+ `nw` for packed outputs).
pub fn k_cast(wg: usize, from: Cls, to: Cls) -> Option<String> {
    use Cls::*;
    // Intermediate value `v` as float (fv) or int (iv), depending on `from`.
    let (float_in, load) = match from {
        F32 | F16 | BF16 => (true, format!("float v = {};", from.read())),
        U32 | U8 => (true, format!("float v = float({});", from.read())),
        I32 => (true, format!("float v = float({});", from.read())),
        I64 => (true, String::new()),
        F64 => return None,
    };
    let _ = float_in;
    // Integer-preserving loads for int-to-int casts (no float rounding).
    let iload = match from {
        U32 => "uint iv = x[e];".to_string(),
        U8 => format!("uint iv = {};", from.read()),
        I32 => "int iv = x[e];".to_string(),
        I64 => "uvec2 r = x[e]; uint iv = r.x;".to_string(),
        _ => String::new(),
    };
    let int_in = !iload.is_empty();
    let mut s = prelude(wg, 2);
    s += &buf(0, from.buf_ty(), "x", true);
    s += &buf(1, to.buf_ty(), "o", false);
    if to == F16 {
        s += F32_TO_F16_RNE;
    }
    // The output element expression from `v` (float) or `iv` (int).
    let (value, store_ty): (String, &str) = match to {
        F32 => ("v".into(), "float"),
        U32 => (
            if int_in {
                "uint(iv)".into()
            } else {
                "(v <= 0.0 ? 0u : uint(v))".into()
            },
            "uint",
        ),
        I32 => (
            if int_in {
                "int(iv)".into()
            } else {
                "int(v)".into()
            },
            "int",
        ),
        I64 => (
            if int_in {
                if from == I32 {
                    "uvec2(uint(iv), iv < 0 ? 0xFFFFFFFFu : 0u)".into()
                } else {
                    "uvec2(uint(iv), 0u)".into()
                }
            } else {
                "uvec2(uint(int(v)), int(v) < 0 ? 0xFFFFFFFFu : 0u)".into()
            },
            "uvec2",
        ),
        U8 => (
            if int_in {
                "(uint(iv) & 0xFFu)".into()
            } else {
                "(v <= 0.0 ? 0u : (v >= 255.0 ? 255u : uint(v)))".into()
            },
            "packed",
        ),
        // Round to nearest even, as candle's f32 → f16 does. `packHalf2x16`
        // leaves the rounding mode to the implementation (RADV truncates).
        F16 => ("f32_to_f16_rne(v)".into(), "packed"),
        // Round to nearest even, as candle's f32 → bf16 does.
        BF16 => (
            "((floatBitsToUint(v) + 0x7FFFu + ((floatBitsToUint(v) >> 16u) & 1u)) >> 16u)".into(),
            "packed",
        ),
        F64 => return None,
    };
    // Integer inputs also expose `v` for the float-valued targets (an i64
    // from both of its words).
    let load_stmt = match from {
        // Exact when the value fits in 32 bits, f32 precision beyond.
        I64 => format!(
            "{iload} float v = (r.y == (int(r.x) < 0 ? 0xFFFFFFFFu : 0u)) ? float(int(r.x)) : float(int(r.y)) * 4294967296.0 + float(r.x);"
        ),
        _ if int_in => format!("{iload} float v = float(iv);"),
        _ => load,
    };
    if store_ty == "packed" {
        let (per, bits, mask) = to.packed().unwrap();
        s += &format!(
            r#"
layout(push_constant) uniform PC {{ int n; int nw; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int w = gid();
    if (w >= pc.nw) return;
    uint word = 0u;
    for (int k = 0; k < {per}; k++) {{
        int i = w * {per} + k;
        if (i < pc.n) {{
            int e = off0(i);
            {load_stmt}
            uint pv = {value};
            word |= (pv & {mask}) << (uint(k) * {bits}u);
        }}
    }}
    o[w] = word;
}}
"#
        );
    } else {
        s += &format!(
            "layout(push_constant) uniform PC {{ int n; int nw; }} pc;\nlayout(local_size_x = WG) in;\nvoid main() {{\n    int i = gid();\n    if (i >= pc.n) return;\n    int e = off0(i);\n    {load_stmt}\n    o[i] = {value};\n}}\n"
        );
    }
    Some(s)
}

const RED_FN: &str = r#"
float red_f(float a, float b, int op) {
    if (op == 0) return a + b;
    if (op == 1) return a > b ? a : b;
    return a < b ? a : b;
}
float red_init(int op) { return op == 0 ? 0.0 : (op == 1 ? NINF() : INF()); }
"#;

/// Row reduction over the contiguous last dim: one work-group per row.
/// Push: `rows, cols, off, op`.
pub fn k_reduce_last(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += RED_FN;
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; int op; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int base = pc.off + row * pc.cols;
    float acc = red_init(pc.op);
    for (int c = t; c < pc.cols; c += WG) acc = red_f(acc, x[base + c], pc.op);
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = red_f(sh[t], sh[t + s], pc.op);
        barrier();
    }
    if (t == 0) o[row] = sh[0];
}
"#;
    s
}

/// Generic reduction: one invocation per output element; `ix` maps output
/// indices to the input base (reduced dims: stride 0, dim 1), `rd`
/// enumerates the reduced sub-space.  Push: `n, count, op`.
pub fn k_reduce_generic(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += RED_FN;
    s += r#"
layout(push_constant) uniform PC { int n; int count; int op; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    int base = off0(i);
    float acc = red_init(pc.op);
    for (int j = 0; j < pc.count; j++) acc = red_f(acc, x[base + offr(j)], pc.op);
    o[i] = acc;
}
"#;
    s
}

/// ArgMax / ArgMin over the contiguous last dim (u32 output).
/// Push: `rows, cols, off, is_max`.
pub fn k_arg_last(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "uint", "o", false);
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; int is_max; } pc;
layout(local_size_x = WG) in;
shared float shv[WG];
shared uint shi[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int base = pc.off + row * pc.cols;
    bool mx = pc.is_max != 0;
    float best = mx ? NINF() : INF();
    uint bi = 0u;
    for (int c = t; c < pc.cols; c += WG) {
        float v = x[base + c];
        bool better = mx ? (v > best) : (v < best);
        if (better) { best = v; bi = uint(c); }
    }
    shv[t] = best; shi[t] = bi;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) {
            float a = shv[t]; float b = shv[t + s];
            bool take = mx ? (b > a || (b == a && shi[t + s] < shi[t])) : (b < a || (b == a && shi[t + s] < shi[t]));
            if (take) { shv[t] = b; shi[t] = shi[t + s]; }
        }
        barrier();
    }
    if (t == 0) o[row] = shi[0];
}
"#;
    s
}

// Out-of-range ids: the CPU backend returns an error, which a kernel
// cannot.  Every indexing kernel binds the device's `fault` buffer last,
// takes the calling thread's slot in it as its last push constant, and
// sets the slot (skipping the element) when an id exceeds the indexed
// dimension; the host checks and clears its slot at the next read-back or
// synchronize and reports the error there, so a bad id never reads or
// writes out of bounds and never turns into a plausible result.

/// `out[l, i, r] = src[l, ids[i], r]` over a contiguous source; u32 ids
/// (`ids_i64`: `uvec2` ids; a non-zero high word is out of range).
/// Push: `n, left, n_ids, right, dim_size, src_off, ids_off`.
pub fn k_index_select(wg: usize, elem8: bool, ids_i64: bool) -> String {
    let ty = if elem8 { "uvec2" } else { "uint" };
    let mut s = prelude(wg, 4);
    s += &buf(0, ty, "src", true);
    s += &buf(1, if ids_i64 { "uvec2" } else { "uint" }, "ids", true);
    s += &buf(2, ty, "o", false);
    s += &buf(3, "uint", "fault", false);
    let id = if ids_i64 {
        "uvec2 rr = ids[pc.ids_off + j]; uint id = rr.y != 0u ? 0xFFFFFFFFu : rr.x;"
    } else {
        "uint id = ids[pc.ids_off + j];"
    };
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n; int left; int n_ids; int right; int dim_size; int src_off; int ids_off; int fslot; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int i = gid();
    if (i >= pc.n) return;
    int r = i % pc.right;
    int t = i / pc.right;
    int j = t % pc.n_ids;
    int l = t / pc.n_ids;
    {id}
    if (id >= uint(pc.dim_size)) {{ fault[pc.fslot] = 1u; return; }}
    o[i] = src[pc.src_off + (l * pc.dim_size + int(id)) * pc.right + r];
}}
"#
    );
    s
}

/// gather along `dim`: `ix.s0` maps output index → ids offset, `ix.s1` →
/// src offset with the `dim` coordinate zeroed.  Push: `n, src_dim_stride, dim_size`.
pub fn k_gather(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "src", true);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "uint", "o", false);
    s += &buf(3, "uint", "fault", false);
    s += r#"
layout(push_constant) uniform PC { int n; int src_dim_stride; int dim_size; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    uint id = ids[off0(i)];
    if (id >= uint(pc.dim_size)) { fault[pc.fslot] = 1u; return; }
    o[i] = src[off1(i) + int(id) * pc.src_dim_stride];
}
"#;
    s
}

/// scatter (set or add) along `dim`: one invocation per position of the
/// ids space with `dim` collapsed (`ix` enumerates that space: s0 = ids,
/// s1 = src, s2 = dst with the dim stride zeroed); the invocation walks
/// `n_j` positions along `dim` in order, so duplicates resolve exactly as
/// the sequential CPU loop does (last write wins / sums in order).
/// Push: `n, n_j, ids_ds, src_ds, dst_ds, dim_size`.
pub fn k_scatter(wg: usize, add: bool) -> String {
    let ty = if add { "float" } else { "uint" };
    let mut s = prelude(wg, 4);
    s += &buf(0, ty, "dst", false);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, ty, "src", true);
    s += &buf(3, "uint", "fault", false);
    let write = if add {
        "dst[d + int(id) * pc.dst_ds] += v;"
    } else {
        "dst[d + int(id) * pc.dst_ds] = v;"
    };
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n; int n_j; int ids_ds; int src_ds; int dst_ds; int dim_size; int fslot; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int i = gid();
    if (i >= pc.n) return;
    int a = off0(i);
    int b = off1(i);
    int d = off2(i);
    for (int j = 0; j < pc.n_j; j++) {{
        uint id = ids[a + j * pc.ids_ds];
        if (id >= uint(pc.dim_size)) {{ fault[pc.fslot] = 1u; continue; }}
        {ty} v = src[b + j * pc.src_ds];
        {write}
    }}
}}
"#
    );
    s
}

/// index_add: `dst[l, ids[j], r] += src[l, j, r]`, one invocation per
/// `(l, r)` looping over `j` (no atomics, CPU order).
/// Push: `n_lr, left, n_ids, right, dim_size, src_off, ids_off`.
pub fn k_index_add(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "float", "dst", false);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "float", "src", true);
    s += &buf(3, "uint", "fault", false);
    s += r#"
layout(push_constant) uniform PC { int n_lr; int left; int n_ids; int right; int dim_size; int src_off; int ids_off; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n_lr) return;
    int r = i % pc.right;
    int l = i / pc.right;
    for (int j = 0; j < pc.n_ids; j++) {
        uint id = ids[pc.ids_off + j];
        if (id >= uint(pc.dim_size)) { fault[pc.fslot] = 1u; continue; }
        dst[(l * pc.dim_size + int(id)) * pc.right + r] += src[pc.src_off + (l * pc.n_ids + j) * pc.right + r];
    }
}
"#;
    s
}

/// Tiled GEMM: `C[bz][m][n] = sum_k A[bz][m][k] * B[bz][k][n]` with explicit
/// strides for A and B.  `tx`×`tx` invocations per group, each computing a
/// 4×4 block of a `4tx`×`4tx` tile; TK = 16.
/// Push: `M, N, K, sam, sak, sbk, sbn, oa, ob, oc, ba, bb, bc, b_kc`.
pub fn k_gemm(wg: usize, tx: usize) -> String {
    let tm = 4 * tx;
    let tk = 16;
    let nthreads = tx * tx;
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += &format!(
        r#"
#define TX {tx}
#define TM {tm}
#define TK {tk}
#define NT {nthreads}
layout(push_constant) uniform PC {{ int M; int N; int K; int sam; int sak; int sbk; int sbn; int oa; int ob; int oc; int ba; int bb; int bc; int b_kc; }} pc;
layout(local_size_x = TX, local_size_y = TX) in;
shared float As[TK * (TM + 1)];
shared float Bs[TK * (TM + 1)];
void main() {{
    int tx = int(gl_LocalInvocationID.x);
    int ty = int(gl_LocalInvocationID.y);
    int m0 = int(gl_WorkGroupID.y) * TM;
    int n0 = int(gl_WorkGroupID.x) * TM;
    int bz = int(gl_WorkGroupID.z);
    int ao = pc.oa + bz * pc.ba;
    int bo = pc.ob + bz * pc.bb;
    int co = pc.oc + bz * pc.bc;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0;
    int tid = ty * TX + tx;
    for (int k0 = 0; k0 < pc.K; k0 += TK) {{
        for (int i = tid; i < TM * TK; i += NT) {{
            int mm = i / TK;
            int kk = i % TK;
            int gm = m0 + mm;
            int gk = k0 + kk;
            As[kk * (TM + 1) + mm] = (gm < pc.M && gk < pc.K) ? A[ao + gm * pc.sam + gk * pc.sak] : 0.0;
        }}
        if (pc.b_kc != 0) {{
            for (int i = tid; i < TK * TM; i += NT) {{
                int kk = i % TK;
                int nn = i / TK;
                int gk = k0 + kk;
                int gn = n0 + nn;
                Bs[kk * (TM + 1) + nn] = (gk < pc.K && gn < pc.N) ? B[bo + gk * pc.sbk + gn * pc.sbn] : 0.0;
            }}
        }} else {{
            for (int i = tid; i < TK * TM; i += NT) {{
                int kk = i / TM;
                int nn = i % TM;
                int gk = k0 + kk;
                int gn = n0 + nn;
                Bs[kk * (TM + 1) + nn] = (gk < pc.K && gn < pc.N) ? B[bo + gk * pc.sbk + gn * pc.sbn] : 0.0;
            }}
        }}
        barrier();
        for (int kk = 0; kk < TK; kk++) {{
            float a0 = As[kk * (TM + 1) + ty * 4 + 0];
            float a1 = As[kk * (TM + 1) + ty * 4 + 1];
            float a2 = As[kk * (TM + 1) + ty * 4 + 2];
            float a3 = As[kk * (TM + 1) + ty * 4 + 3];
            float b0 = Bs[kk * (TM + 1) + tx * 4 + 0];
            float b1 = Bs[kk * (TM + 1) + tx * 4 + 1];
            float b2 = Bs[kk * (TM + 1) + tx * 4 + 2];
            float b3 = Bs[kk * (TM + 1) + tx * 4 + 3];
            acc[0] += a0 * b0; acc[1] += a0 * b1; acc[2] += a0 * b2; acc[3] += a0 * b3;
            acc[4] += a1 * b0; acc[5] += a1 * b1; acc[6] += a1 * b2; acc[7] += a1 * b3;
            acc[8] += a2 * b0; acc[9] += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }}
        barrier();
    }}
    for (int r = 0; r < 4; r++) {{
        int gm = m0 + ty * 4 + r;
        if (gm >= pc.M) continue;
        for (int c = 0; c < 4; c++) {{
            int gn = n0 + tx * 4 + c;
            if (gn < pc.N) C[co + gm * pc.N + gn] = acc[r * 4 + c];
        }}
    }}
}}
"#
    );
    s
}

pub fn k_gemm_f64(wg: usize, tx: usize) -> String {
    k_gemm(wg, tx)
        .replace("float acc[16];", "double acc[16];")
        .replace("acc[i] = 0.0;", "acc[i] = double(0.0);")
        .replace("float a", "double a")
        .replace("float b", "double b")
        .replace("= acc[r * 4 + c];", "= float(acc[r * 4 + c]);")
}

/// Small-row precise matmul: no padded GEMM rows, shared tiles or barriers.
/// Each output retains the CPU's sequential K-order FP64 FMA accumulation.
/// Push layout is shared with GEMM, including batch/broadcast strides.
pub fn k_small_matmul_f64(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += r#"
layout(push_constant) uniform PC { int M; int N; int K; int sam; int sak; int sbk; int sbn; int oa; int ob; int oc; int ba; int bb; int bc; int b_kc; } pc;
layout(local_size_x=WG) in;
void main() {
    int i=gid();
    int bz=int(gl_WorkGroupID.z);
    if(i>=pc.M*pc.N)return;
    int m=i/pc.N;
    int n=i%pc.N;
    int ao=pc.oa+bz*pc.ba+m*pc.sam;
    int bo=pc.ob+bz*pc.bb+n*pc.sbn;
    double acc=double(0.0);
    for(int k=0;k<pc.K;k++)acc=fma(double(A[ao+k*pc.sak]),double(B[bo+k*pc.sbk]),acc);
    C[pc.oc+bz*pc.bc+i]=float(acc);
}
"#;
    s
}

/// Parallel contraction for long, small-row matmuls. FP64 partials avoid
/// the FP32 reduction drift while making enough independent work for the GPU.
pub fn k_small_matmul_reduce_f64(_wg: usize) -> String {
    let mut s = prelude(64, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += r#"
layout(push_constant) uniform PC {int M;int N;int K;int sam;int sak;int sbk;int sbn;int oa;int ob;int oc;int ba;int bb;int bc;int b_kc;} pc;
layout(local_size_x=64) in;
shared double partial[64];
void main(){
 int i=grow();int t=int(gl_LocalInvocationID.x);int bz=int(gl_WorkGroupID.z);
 if(i>=pc.M*pc.N)return;
 int ao=pc.oa+bz*pc.ba+(i/pc.N)*pc.sam;
 int bo=pc.ob+bz*pc.bb+(i%pc.N)*pc.sbn;
 double acc=double(0.0);
 for(int k=t;k<pc.K;k+=64)acc=fma(double(A[ao+k*pc.sak]),double(B[bo+k*pc.sbk]),acc);
 partial[t]=acc;barrier();
 if(t==0){double sum=double(0.0);for(int j=0;j<64;j++)sum+=partial[j];C[pc.oc+bz*pc.bc+i]=float(sum);}
}
"#;
    s
}

/// GEMV for a transposed weight (B contiguous along k): one work-group per
/// `n` (2-D grid) with the batch on z.  Push: `N, K, sbn, oa, ob, oc, ba, bb, bc`.
pub fn k_gemv_nt(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += r#"
layout(push_constant) uniform PC { int N; int K; int sbn; int oa; int ob; int oc; int ba; int bb; int bc; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int n = grow();
    int bz = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    if (n >= pc.N) return;
    int a = pc.oa + bz * pc.ba;
    int w = pc.ob + bz * pc.bb + n * pc.sbn;
    float acc = 0.0;
    for (int k = t; k < pc.K; k += WG) acc += A[a + k] * B[w + k];
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    if (t == 0) C[pc.oc + bz * pc.bc + n] = sh[0];
}
"#;
    s
}

/// GEMV for a row-major B (contiguous along n): one invocation per `n`,
/// batch on the group's z.  Push: `N, K, sbk, oa, ob, oc, ba, bb, bc`.
pub fn k_gemv_nn(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "A", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "C", false);
    s += r#"
layout(push_constant) uniform PC { int N; int K; int sbk; int oa; int ob; int oc; int ba; int bb; int bc; } pc;
layout(local_size_x = WG) in;
void main() {
    int n = gid();
    int bz = int(gl_WorkGroupID.z);
    if (n >= pc.N) return;
    int a = pc.oa + bz * pc.ba;
    int b = pc.ob + bz * pc.bb;
    float acc = 0.0;
    for (int k = 0; k < pc.K; k++) acc += A[a + k] * B[b + k * pc.sbk + n];
    C[pc.oc + bz * pc.bc + n] = acc;
}
"#;
    s
}

/// softmax over the contiguous last dim, one work-group per row.
/// Push: `rows, cols, off`.
pub fn k_softmax_last(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "o", false);
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int r = pc.off + row * pc.cols;
    int w = row * pc.cols;
    float m = NINF();
    for (int c = t; c < pc.cols; c += WG) m = max(m, x[r + c]);
    sh[t] = m;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = max(sh[t], sh[t + s]);
        barrier();
    }
    m = sh[0];
    barrier();
    float sum = 0.0;
    for (int c = t; c < pc.cols; c += WG) sum += exp(x[r + c] - m);
    sh[t] = sum;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    float inv = 1.0 / sh[0];
    // Recompute exp() so each output element is written once, fully normalized.
    barrier();
    for (int c = t; c < pc.cols; c += WG) o[w + c] = exp(x[r + c] - m) * inv;
}
"#;
    s
}

/// rms_norm: `o = x / sqrt(mean(x^2) + eps) * alpha`, one work-group per row.
/// Push: `rows, cols, off, aoff, eps`.
pub fn k_rmsnorm(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "alpha", true);
    s += &buf(2, "float", "o", false);
    s += r#"
layout(push_constant) uniform PC { int rows; int cols; int off; int aoff; float eps; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int row = grow();
    int t = int(gl_LocalInvocationID.x);
    if (row >= pc.rows) return;
    int r = pc.off + row * pc.cols;
    int w = row * pc.cols;
    float ss = 0.0;
    for (int c = t; c < pc.cols; c += WG) { float v = x[r + c]; ss += v * v; }
    sh[t] = ss;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    float scale = 1.0 / sqrt(sh[0] / float(pc.cols) + pc.eps);
    for (int c = t; c < pc.cols; c += WG) o[w + c] = x[r + c] * scale * alpha[pc.aoff + c];
}
"#;
    s
}

/// Diagnostic accurate wide-row RMS: f64 accumulation and scaling, f32 storage.
pub fn k_rmsnorm_f64(wg: usize, weighted: bool) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "alpha", true);
    s += &buf(2, "float", "o", false);
    s += r#"
layout(push_constant) uniform PC {int rows;int cols;int off;int aoff;float eps;} pc;
layout(local_size_x=WG) in;
shared double sh[8];
void main(){
 int row=grow();int t=int(gl_LocalInvocationID.x);if(row>=pc.rows)return;
 int r=pc.off+row*pc.cols;int w=row*pc.cols;
 if(t<8){double ss=double(0.0);for(int c=t;c<pc.cols;c+=8){double v=double(x[r+c]);ss+=v*v;}sh[t]=ss;}
 barrier();
 if(t==0){double sum=double(0.0);for(int i=0;i<8;i++)sum+=sh[i];sh[0]=double(1.0)/sqrt(sum/double(pc.cols)+double(pc.eps));}
 barrier();double scale=sh[0];
 for(int c=t;c<pc.cols;c+=WG){
"#;
    s += if weighted {
        "o[w+c]=float(double(x[r+c])*scale*double(alpha[pc.aoff+c]));"
    } else {
        "o[w+c]=float(double(x[r+c])*scale);"
    };
    s += "\n}\n}\n";
    s
}

/// RoPE over `x [b, h, t, d]` with `cos`/`sin` `[t, d/2]` (or `[b, t, d/2]`
/// when `cs_b`); `interleaved` selects candle's `rope_i` pairing.
/// Push: `n_pairs, h, t, d, xoff, coff, soff, cs_b`.
pub fn k_rope(wg: usize, interleaved: bool) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "float", "x", true);
    s += &buf(1, "float", "cs", true);
    s += &buf(2, "float", "sn", true);
    s += &buf(3, "float", "o", false);
    let body = if interleaved {
        r#"
    int base = rest * pc.d + 2 * p;
    float x1 = x[pc.xoff + base];
    float x2 = x[pc.xoff + base + 1];
    o[base] = x1 * c - x2 * s;
    o[base + 1] = x1 * s + x2 * c;
"#
    } else {
        r#"
    int base = rest * pc.d;
    float x1 = x[pc.xoff + base + p];
    float x2 = x[pc.xoff + base + hd2 + p];
    o[base + p] = x1 * c - x2 * s;
    o[base + hd2 + p] = x2 * c + x1 * s;
"#
    };
    s += &format!(
        r#"
layout(push_constant) uniform PC {{ int n_pairs; int h; int t; int d; int xoff; int coff; int soff; int cs_b; }} pc;
layout(local_size_x = WG) in;
void main() {{
    int i = gid();
    if (i >= pc.n_pairs) return;
    int hd2 = pc.d / 2;
    int p = i % hd2;
    int rest = i / hd2;
    int tt = rest % pc.t;
    int bh = rest / pc.t;
    int b = bh / pc.h;
    int ci = (pc.cs_b != 0 ? b * pc.t * hd2 : 0) + tt * hd2 + p;
    float c = cs[pc.coff + ci];
    float s = sn[pc.soff + ci];
{body}
}}
"#
    );
    s
}

pub fn k_rope_i_f64(wg: usize) -> String {
    k_rope(wg, true)
        .replace(
            "float x1 = x[pc.xoff + base];",
            "double x1 = double(x[pc.xoff + base]);",
        )
        .replace(
            "float x2 = x[pc.xoff + base + 1];",
            "double x2 = double(x[pc.xoff + base + 1]);",
        )
        .replace(
            "float c = cs[pc.coff + ci];",
            "double c = double(cs[pc.coff + ci]);",
        )
        .replace(
            "float s = sn[pc.soff + ci];",
            "double s = double(sn[pc.soff + ci]);",
        )
        .replace(
            "o[base] = x1 * c - x2 * s;",
            "o[base] = float(x1 * c - x2 * s);",
        )
        .replace(
            "o[base + 1] = x1 * s + x2 * c;",
            "o[base + 1] = float(x1 * s + x2 * c);",
        )
}

/// Byte-addressed helpers over the packed weight buffer `W` and the
/// sub-block dequantizer writing into the private `dq[32]`.
const QUANT_FN: &str = r#"
uint byte_at(uint b) { return (W[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
int i8_at(uint b) { uint v = byte_at(b); return int(v) - ((v & 0x80u) != 0u ? 256 : 0); }
float f16_at(uint b) { return unpackHalf2x16(byte_at(b) | (byte_at(b + 1u) << 8u)).x; }
float bf16_at(uint b) { return uintBitsToFloat((byte_at(b + 1u) << 24u) | (byte_at(b) << 16u)); }
float f32_at(uint b) { return uintBitsToFloat(byte_at(b) | (byte_at(b + 1u) << 8u) | (byte_at(b + 2u) << 16u) | (byte_at(b + 3u) << 24u)); }
uint u32_at(uint b) { return byte_at(b) | (byte_at(b + 1u) << 8u) | (byte_at(b + 2u) << 16u) | (byte_at(b + 3u) << 24u); }
// IQ2_XXS sub-block decode tables (port of joshua/candle-core iq2xxs.rs).
const uint IQ2XXS_GRID_LO[256] = uint[256](
    0x08080808u, 0x0808082bu, 0x08081919u, 0x08082b08u, 0x08082b2bu, 0x08190819u,
    0x08191908u, 0x082b0808u, 0x082b082bu, 0x082b2b08u, 0x082b2b2bu, 0x19080819u,
    0x19081908u, 0x19190808u, 0x19192b08u, 0x192b0819u, 0x192b1908u, 0x2b080808u,
    0x2b08082bu, 0x2b082b2bu, 0x2b2b082bu, 0x08080819u, 0x08081908u, 0x08190808u,
    0x08191919u, 0x19080808u, 0x2b081908u, 0x2b192b08u, 0x08080808u, 0x0808082bu,
    0x082b082bu, 0x2b08082bu, 0x08080819u, 0x08081908u, 0x08190808u, 0x082b0819u,
    0x082b1908u, 0x19080808u, 0x1908082bu, 0x19082b08u, 0x192b0808u, 0x2b080819u,
    0x2b081908u, 0x2b190808u, 0x2b2b1908u, 0x08080808u, 0x0808082bu, 0x08082b08u,
    0x082b0808u, 0x1908192bu, 0x192b2b19u, 0x2b080808u, 0x2b190819u, 0x08082b19u,
    0x08190808u, 0x19080808u, 0x2b081908u, 0x2b2b1908u, 0x08080808u, 0x08081919u,
    0x08082b08u, 0x08191908u, 0x082b2b08u, 0x19080819u, 0x19081908u, 0x19190808u,
    0x1919082bu, 0x2b082b08u, 0x08081908u, 0x19080808u, 0x0808082bu, 0x08191908u,
    0x08080819u, 0x08081908u, 0x08190808u, 0x082b0819u, 0x19080808u, 0x192b0808u,
    0x2b081908u, 0x2b190808u, 0x2b191919u, 0x08080808u, 0x08082b08u, 0x082b0808u,
    0x19190808u, 0x19192b2bu, 0x2b080808u, 0x082b1908u, 0x19081919u, 0x08080808u,
    0x08082b08u, 0x082b0808u, 0x082b1919u, 0x19082b19u, 0x2b080808u, 0x08192b08u,
    0x192b082bu, 0x08080808u, 0x0819192bu, 0x08080819u, 0x08081908u, 0x08190808u,
    0x19080808u, 0x2b080819u, 0x08080808u, 0x08081919u, 0x2b2b0808u, 0x19190819u,
    0x08080808u, 0x0808082bu, 0x08082b2bu, 0x19081908u, 0x192b0819u, 0x2b080808u,
    0x2b08082bu, 0x082b2b19u, 0x19082b08u, 0x08080808u, 0x0808082bu, 0x08080819u,
    0x08081908u, 0x08190808u, 0x19080808u, 0x1919192bu, 0x08080808u, 0x19080819u,
    0x192b1908u, 0x2b190808u, 0x08082b08u, 0x082b0808u, 0x2b191908u, 0x19081908u,
    0x08080819u, 0x08081908u, 0x08190808u, 0x08192b08u, 0x082b0819u, 0x082b1908u,
    0x19080808u, 0x19082b08u, 0x1919192bu, 0x192b0808u, 0x2b080819u, 0x2b081908u,
    0x2b190808u, 0x08080808u, 0x082b0808u, 0x192b0819u, 0x2b080808u, 0x2b081919u,
    0x08080819u, 0x08190808u, 0x19082b08u, 0x1919192bu, 0x192b2b08u, 0x08080808u,
    0x08082b08u, 0x082b0808u, 0x2b080808u, 0x2b192b19u, 0x0819082bu, 0x082b1908u,
    0x08080808u, 0x08080819u, 0x08081908u, 0x08190808u, 0x19080808u, 0x19081919u,
    0x08080808u, 0x19192b08u, 0x192b0819u, 0x2b08082bu, 0x19081919u, 0x2b190808u,
    0x08080808u, 0x08082b08u, 0x08190819u, 0x08192b19u, 0x082b0808u, 0x2b080808u,
    0x2b082b08u, 0x08081908u, 0x1908082bu, 0x2b2b1908u, 0x2b190819u, 0x2b190808u,
    0x2b19082bu, 0x08082b2bu, 0x08080819u, 0x19191908u, 0x08080808u, 0x08190819u,
    0x08192b19u, 0x192b1908u, 0x19080808u, 0x08082b08u, 0x08081908u, 0x08190808u,
    0x19080808u, 0x192b2b08u, 0x08080808u, 0x19191919u, 0x08192b08u, 0x192b0808u,
    0x08080808u, 0x08081919u, 0x08190808u, 0x0819082bu, 0x2b081908u, 0x1908082bu,
    0x08080808u, 0x0808082bu, 0x08082b2bu, 0x19080819u, 0x2b08082bu, 0x08081908u,
    0x08192b08u, 0x19080808u, 0x08190819u, 0x08080819u, 0x08081908u, 0x08190808u,
    0x08191919u, 0x19080808u, 0x192b0808u, 0x08080808u, 0x1908192bu, 0x2b191908u,
    0x08082b19u, 0x19080808u, 0x192b0808u, 0x0808082bu, 0x08081908u, 0x08190819u,
    0x08081908u, 0x08190808u, 0x082b1908u, 0x19080808u, 0x2b2b0819u, 0x0819192bu,
    0x2b080808u, 0x19081919u, 0x08080808u, 0x082b082bu, 0x19081908u, 0x19190819u,
    0x2b080819u, 0x082b0808u, 0x0808082bu, 0x19190808u, 0x2b081919u, 0x08082b19u,
    0x08080808u, 0x08192b08u, 0x19190808u, 0x08081908u
);

const uint IQ2XXS_GRID_HI[256] = uint[256](
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080819u, 0x08080819u, 0x08080819u,
    0x08080819u, 0x08080819u, 0x08080819u, 0x08080819u, 0x0808082bu, 0x0808082bu,
    0x0808082bu, 0x0808082bu, 0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081919u, 0x08081919u, 0x08081919u,
    0x08081919u, 0x08081919u, 0x08081919u, 0x08081919u, 0x08081919u, 0x0808192bu,
    0x0808192bu, 0x0808192bu, 0x0808192bu, 0x0808192bu, 0x08082b08u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08082b19u, 0x08082b19u, 0x08082b2bu, 0x08082b2bu,
    0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u,
    0x08190808u, 0x08190808u, 0x08190808u, 0x08190819u, 0x08190819u, 0x08190819u,
    0x08190819u, 0x08190819u, 0x08190819u, 0x0819082bu, 0x0819082bu, 0x08191908u,
    0x08191908u, 0x08191908u, 0x08191908u, 0x08191908u, 0x08191908u, 0x08191919u,
    0x08191919u, 0x0819192bu, 0x0819192bu, 0x08192b08u, 0x08192b08u, 0x08192b08u,
    0x08192b08u, 0x08192b08u, 0x08192b19u, 0x08192b19u, 0x08192b19u, 0x08192b2bu,
    0x082b0808u, 0x082b0808u, 0x082b0808u, 0x082b0808u, 0x082b0808u, 0x082b0808u,
    0x082b0808u, 0x082b0819u, 0x082b0819u, 0x082b082bu, 0x082b082bu, 0x082b1908u,
    0x082b1908u, 0x082b1908u, 0x082b1908u, 0x082b1908u, 0x082b1919u, 0x082b1919u,
    0x082b1919u, 0x082b192bu, 0x082b2b08u, 0x082b2b08u, 0x082b2b08u, 0x082b2b2bu,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080819u, 0x19080819u, 0x19080819u, 0x19080819u, 0x19080819u,
    0x1908082bu, 0x1908082bu, 0x1908082bu, 0x1908082bu, 0x1908082bu, 0x19081908u,
    0x19081908u, 0x19081908u, 0x19081908u, 0x19081908u, 0x19081919u, 0x19081919u,
    0x1908192bu, 0x19082b08u, 0x19082b08u, 0x19082b08u, 0x19082b08u, 0x19082b08u,
    0x19082b19u, 0x19082b19u, 0x19082b19u, 0x19082b19u, 0x19082b2bu, 0x19082b2bu,
    0x19190808u, 0x19190808u, 0x19190808u, 0x19190808u, 0x19190808u, 0x19190808u,
    0x19190808u, 0x19190819u, 0x19190819u, 0x19190819u, 0x1919082bu, 0x19191908u,
    0x19191908u, 0x19191919u, 0x1919192bu, 0x1919192bu, 0x19192b08u, 0x19192b08u,
    0x19192b08u, 0x19192b08u, 0x19192b19u, 0x19192b2bu, 0x192b0808u, 0x192b0808u,
    0x192b0808u, 0x192b0808u, 0x192b0819u, 0x192b0819u, 0x192b082bu, 0x192b082bu,
    0x192b1908u, 0x192b1908u, 0x192b1919u, 0x192b1919u, 0x192b1919u, 0x192b2b08u,
    0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080819u,
    0x2b080819u, 0x2b080819u, 0x2b08082bu, 0x2b081908u, 0x2b081908u, 0x2b081908u,
    0x2b081908u, 0x2b081908u, 0x2b081908u, 0x2b081919u, 0x2b081919u, 0x2b081919u,
    0x2b08192bu, 0x2b08192bu, 0x2b08192bu, 0x2b082b08u, 0x2b082b19u, 0x2b082b2bu,
    0x2b190808u, 0x2b190808u, 0x2b190808u, 0x2b190808u, 0x2b190808u, 0x2b190819u,
    0x2b190819u, 0x2b19082bu, 0x2b191908u, 0x2b191908u, 0x2b191908u, 0x2b191919u,
    0x2b192b08u, 0x2b192b19u, 0x2b2b0808u, 0x2b2b0808u, 0x2b2b0808u, 0x2b2b0819u,
    0x2b2b082bu, 0x2b2b1908u, 0x2b2b2b08u, 0x2b2b2b19u
);

const uint KSIGNS_IQ2XS[128] = uint[128](
    0x00u, 0x81u, 0x82u, 0x03u, 0x84u, 0x05u, 0x06u, 0x87u,
    0x88u, 0x09u, 0x0au, 0x8bu, 0x0cu, 0x8du, 0x8eu, 0x0fu,
    0x90u, 0x11u, 0x12u, 0x93u, 0x14u, 0x95u, 0x96u, 0x17u,
    0x18u, 0x99u, 0x9au, 0x1bu, 0x9cu, 0x1du, 0x1eu, 0x9fu,
    0xa0u, 0x21u, 0x22u, 0xa3u, 0x24u, 0xa5u, 0xa6u, 0x27u,
    0x28u, 0xa9u, 0xaau, 0x2bu, 0xacu, 0x2du, 0x2eu, 0xafu,
    0x30u, 0xb1u, 0xb2u, 0x33u, 0xb4u, 0x35u, 0x36u, 0xb7u,
    0xb8u, 0x39u, 0x3au, 0xbbu, 0x3cu, 0xbdu, 0xbeu, 0x3fu,
    0xc0u, 0x41u, 0x42u, 0xc3u, 0x44u, 0xc5u, 0xc6u, 0x47u,
    0x48u, 0xc9u, 0xcau, 0x4bu, 0xccu, 0x4du, 0x4eu, 0xcfu,
    0x50u, 0xd1u, 0xd2u, 0x53u, 0xd4u, 0x55u, 0x56u, 0xd7u,
    0xd8u, 0x59u, 0x5au, 0xdbu, 0x5cu, 0xddu, 0xdeu, 0x5fu,
    0x60u, 0xe1u, 0xe2u, 0x63u, 0xe4u, 0x65u, 0x66u, 0xe7u,
    0xe8u, 0x69u, 0x6au, 0xebu, 0x6cu, 0xedu, 0xeeu, 0x6fu,
    0xf0u, 0x71u, 0x72u, 0xf3u, 0x74u, 0xf5u, 0xf6u, 0x77u,
    0x78u, 0xf9u, 0xfau, 0x7bu, 0xfcu, 0x7du, 0x7eu, 0xffu
);
float dq[32];
void k4_scale(uint sc, int is, out uint d1, out uint m1) {
    if (is < 4) { d1 = byte_at(sc + uint(is)) & 63u; m1 = byte_at(sc + uint(is) + 4u) & 63u; }
    else {
        d1 = (byte_at(sc + uint(is) + 4u) & 0xFu) | ((byte_at(sc + uint(is) - 4u) >> 6u) << 4u);
        m1 = (byte_at(sc + uint(is) + 4u) >> 4u) | ((byte_at(sc + uint(is)) >> 6u) << 4u);
    }
}
// Dequantize elements [32*j, 32*j + 32) of the block at byte offset `p`
// (type `qt`) into dq.  Sub-blocks are the unit of parallelism; `j` is
// always 0 for 32-element blocks.
void dequant_sub(int qt, uint p, int j) {
    if (qt == 8) { // Q8_0
        float d = f16_at(p);
        for (int i = 0; i < 32; i++) dq[i] = d * float(i8_at(p + 2u + uint(i)));
    } else if (qt == 2) { // Q4_0
        float d = f16_at(p);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 2u + uint(i));
            dq[i] = d * (float(b & 0xFu) - 8.0);
            dq[i + 16] = d * (float(b >> 4u) - 8.0);
        }
    } else if (qt == 3) { // Q4_1
        float d = f16_at(p); float m = f16_at(p + 2u);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 4u + uint(i));
            dq[i] = d * float(b & 0xFu) + m;
            dq[i + 16] = d * float(b >> 4u) + m;
        }
    } else if (qt == 6) { // Q5_0
        float d = f16_at(p);
        uint qh = u32_at(p + 2u);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 6u + uint(i));
            uint xh0 = ((qh >> uint(i)) << 4u) & 0x10u;
            uint xh1 = (qh >> uint(i + 12)) & 0x10u;
            dq[i] = d * (float((b & 0xFu) | xh0) - 16.0);
            dq[i + 16] = d * (float((b >> 4u) | xh1) - 16.0);
        }
    } else if (qt == 7) { // Q5_1
        float d = f16_at(p); float m = f16_at(p + 2u);
        uint qh = u32_at(p + 4u);
        for (int i = 0; i < 16; i++) {
            uint b = byte_at(p + 8u + uint(i));
            uint xh0 = ((qh >> uint(i)) << 4u) & 0x10u;
            uint xh1 = (qh >> uint(i + 12)) & 0x10u;
            dq[i] = d * float((b & 0xFu) | xh0) + m;
            dq[i + 16] = d * float((b >> 4u) | xh1) + m;
        }
    } else if (qt == 9) { // Q8_1
        float d = f16_at(p);
        for (int i = 0; i < 32; i++) dq[i] = d * float(i8_at(p + 4u + uint(i)));
    } else if (qt == 12 || qt == 13) { // Q4_K / Q5_K
        float d = f16_at(p); float dmin = f16_at(p + 2u);
        int g = j >> 1;
        int hf = j & 1;
        uint sc1; uint m1;
        k4_scale(p + 4u, 2 * g + hf, sc1, m1);
        float d1 = d * float(sc1); float mm1 = dmin * float(m1);
        if (qt == 12) {
            uint qs = p + 16u + 32u * uint(g);
            for (int l = 0; l < 32; l++) {
                uint q = byte_at(qs + uint(l));
                dq[l] = d1 * float(hf != 0 ? (q >> 4u) : (q & 0xFu)) - mm1;
            }
        } else {
            uint qh = p + 16u;
            uint ql = p + 48u + 32u * uint(g);
            uint u = 1u << uint(2 * g + hf);
            for (int l = 0; l < 32; l++) {
                uint q = byte_at(ql + uint(l));
                uint h = byte_at(qh + uint(l));
                dq[l] = d1 * float((hf != 0 ? (q >> 4u) : (q & 0xFu)) + ((h & u) != 0u ? 16u : 0u)) - mm1;
            }
        }
    } else if (qt == 14) { // Q6_K
        int hh = j >> 2;
        int q = j & 3;
        uint ql = p + 64u * uint(hh);
        uint qh = p + 128u + 32u * uint(hh);
        uint sc = p + 192u + 8u * uint(hh);
        float d = f16_at(p + 208u);
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            uint qll = byte_at(ql + uint(l));
            uint qlh = byte_at(ql + uint(l) + 32u);
            uint h = byte_at(qh + uint(l));
            int v;
            int s;
            if (q == 0) { v = int((qll & 0xFu) | (((h >> 0u) & 3u) << 4u)); s = i8_at(sc + uint(is)); }
            else if (q == 1) { v = int((qlh & 0xFu) | (((h >> 2u) & 3u) << 4u)); s = i8_at(sc + uint(is) + 2u); }
            else if (q == 2) { v = int((qll >> 4u) | (((h >> 4u) & 3u) << 4u)); s = i8_at(sc + uint(is) + 4u); }
            else { v = int((qlh >> 4u) | (((h >> 6u) & 3u) << 4u)); s = i8_at(sc + uint(is) + 6u); }
            dq[l] = d * float(s) * float(v - 32);
        }
    } else if (qt == 15) { // Q8_K
        float d = f32_at(p);
        for (int i = 0; i < 32; i++) dq[i] = d * float(i8_at(p + 4u + 32u * uint(j) + uint(i)));
    } else if (qt == 10) { // Q2_K
        int hh = j >> 2;
        int m = j & 3;
        uint sc = p + 8u * uint(hh) + 2u * uint(m);
        uint qs = p + 16u + 32u * uint(hh);
        float d = f16_at(p + 80u); float dmin = f16_at(p + 82u);
        uint shift = 2u * uint(m);
        uint s0 = byte_at(sc); uint s1 = byte_at(sc + 1u);
        float dl0 = d * float(s0 & 0xFu); float ml0 = dmin * float(s0 >> 4u);
        float dl1 = d * float(s1 & 0xFu); float ml1 = dmin * float(s1 >> 4u);
        for (int l = 0; l < 16; l++) {
            dq[l] = dl0 * float((byte_at(qs + uint(l)) >> shift) & 3u) - ml0;
            dq[16 + l] = dl1 * float((byte_at(qs + uint(l) + 16u) >> shift) & 3u) - ml1;
        }
    } else if (qt == 11) { // Q3_K
        uint hm = p;
        uint scp = p + 96u;
        float d = f16_at(p + 108u);
        uint aux0 = u32_at(scp); uint aux1 = u32_at(scp + 4u); uint aux2 = u32_at(scp + 8u);
        uint kmask1 = 0x03030303u; uint kmask2 = 0x0f0f0f0fu;
        uint a0 = (aux0 & kmask2) | (((aux2 >> 0u) & kmask1) << 4u);
        uint a1 = (aux1 & kmask2) | (((aux2 >> 2u) & kmask1) << 4u);
        uint a2 = ((aux0 >> 4u) & kmask2) | (((aux2 >> 4u) & kmask1) << 4u);
        uint a3 = ((aux1 >> 4u) & kmask2) | (((aux2 >> 6u) & kmask1) << 4u);
        int hh = j >> 2;
        int m = j & 3;
        int is = 8 * hh + 2 * m;
        uint w0 = is < 4 ? a0 : (is < 8 ? a1 : (is < 12 ? a2 : a3));
        uint w1 = (is + 1) < 4 ? a0 : ((is + 1) < 8 ? a1 : ((is + 1) < 12 ? a2 : a3));
        uint v0 = (w0 >> (uint(is % 4) * 8u)) & 0xFFu;
        uint v1 = (w1 >> (uint((is + 1) % 4) * 8u)) & 0xFFu;
        int s0 = int(v0) - ((v0 & 0x80u) != 0u ? 256 : 0) - 32;
        int s1 = int(v1) - ((v1 & 0x80u) != 0u ? 256 : 0) - 32;
        uint mask = 1u << uint(4 * hh + m);
        uint shift = 2u * uint(m);
        uint qs = p + 32u + 32u * uint(hh);
        float dl0 = d * float(s0); float dl1 = d * float(s1);
        for (int l = 0; l < 16; l++) {
            int q0 = int((byte_at(qs + uint(l)) >> shift) & 3u) - ((byte_at(hm + uint(l)) & mask) != 0u ? 0 : 4);
            int q1 = int((byte_at(qs + uint(l) + 16u) >> shift) & 3u) - ((byte_at(hm + uint(l) + 16u) & mask) != 0u ? 0 : 4);
            dq[l] = dl0 * float(q0);
            dq[16 + l] = dl1 * float(q1);
        }
    }
    else if (qt == 16) { // IQ2_XXS
        float d = f16_at(p);
        int base = 2 + 8 * j;
        uint lo = u32_at(p + uint(base));
        uint hi = u32_at(p + uint(base + 4));
        float db = d * (0.5f + float(hi >> 28u)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            uint code = (lo >> uint(8 * l)) & 0xFFu;
            uint signs = KSIGNS_IQ2XS[(hi >> uint(7 * l)) & 127u];
            uint glo = IQ2XXS_GRID_LO[code];
            uint ghi = IQ2XXS_GRID_HI[code];
            for (int v = 0; v < 8; v++) {
                uint b = (v < 4) ? ((glo >> (8u * uint(v))) & 0xFFu)
                                 : ((ghi >> (8u * uint(v - 4))) & 0xFFu);
                int gv = int(b) - ((b & 0x80u) != 0u ? 256 : 0);
                float sg = ((signs >> uint(v)) & 1u) != 0u ? -1.0 : 1.0;
                dq[l * 8 + v] = db * float(gv) * sg;
            }
        }
    }
}
"#;

/// Quantized GEMV/GEMM: `C[m, n] = sum_k X[m, k] * W[n, k]`; one
/// work-group per `n` (2-D grid), `m` on z; invocations stride over
/// 32-element sub-blocks.  Push: `N, K, qt, qk, bsz, xoff, coff, M`.
pub fn k_qgemv(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += QUANT_FN;
    s += r#"
layout(push_constant) uniform PC { int N; int K; int qt; int qk; int bsz; int xoff; int coff; int M; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int n = grow();
    int m = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    if (n >= pc.N || m >= pc.M) return;
    int x = pc.xoff + m * pc.K;
    int spb = pc.qk / 32;
    int nsub = pc.K / 32;
    uint row = uint(n) * uint(pc.K / pc.qk) * uint(pc.bsz);
    float acc = 0.0;
    for (int s = t; s < nsub; s += WG) {
        int b = s / spb;
        int jj = s - b * spb;
        dequant_sub(pc.qt, row + uint(b) * uint(pc.bsz), jj);
        int xb = x + s * 32;
        float d = 0.0;
        for (int i = 0; i < 32; i++) d += X[xb + i] * dq[i];
        acc += d;
    }
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    if (t == 0) C[pc.coff + m * pc.N + n] = sh[0];
}
"#;
    s
}

// Match the CPU's 4-lane Q8 / 8-lane Q4,Q2 accumulation exactly in
// prefill as well as decode. Several output rows share one workgroup.
pub fn k_qgemv_f64(wg: usize, groups: usize, lanes: usize, rows: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += QUANT_FN;
    s += &format!("\n#define LANES {lanes}\n#define GROUPS {groups}\n#define ROWS {rows}\n");
    s += r#"
layout(push_constant) uniform PC {int N;int K;int qt;int qk;int bsz;int xoff;int coff;int M;} pc;
layout(local_size_x=WG) in;
shared double partial[WG*ROWS];
void main(){
 int t=int(gl_LocalInvocationID.x);int lane=t%LANES;int local_row=t/LANES;
 int n=grow()*(WG/LANES)+local_row;int m=int(gl_WorkGroupID.z)*ROWS;
 if(m>=pc.M)return;
 double acc[ROWS];for(int r=0;r<ROWS;r++)acc[r]=double(0.0);
 if(n<pc.N){
  int xr=GROUPS>1?n/(pc.N/GROUPS):m;int x=pc.xoff+xr*pc.K;
  int nb=pc.K/pc.qk;
  for(int b=0;b<nb;b++){
   uint p=uint(n*nb+b)*uint(pc.bsz);float d;float dm;
   if(pc.qt==8){d=f16_at(p);dm=0.0;}else if(pc.qt==12){d=f16_at(p);dm=f16_at(p+2u);}else{d=f16_at(p+80u);dm=f16_at(p+82u);}
   for(int sg=0;sg<pc.qk/32;sg++){
    uint sc=0u;uint mn=0u;
    if(pc.qt==12)k4_scale(p+4u,sg,sc,mn);
    for(int i=lane;i<32;i+=LANES){
     float w;
     if(pc.qt==8){w=d*float(i8_at(p+2u+uint(i)));}
     else if(pc.qt==12){uint q=byte_at(p+16u+uint(sg/2*32+i));q=(q>>uint((sg%2)*4))&15u;w=(d*float(sc))*float(q)-dm*float(mn);}
     else{int hh=sg/4;uint sh=uint(2*(sg%4));uint scale=byte_at(p+uint(8*hh+2*(sg%4)+i/16));uint q=(byte_at(p+16u+uint(32*hh+i))>>sh)&3u;w=(d*float(scale&15u))*float(q)-dm*float(scale>>4u);}
     for(int r=0;r<ROWS;r++){
      if(m+r<pc.M)acc[r]=fma(double(X[x+r*pc.K+b*pc.qk+sg*32+i]),double(w),acc[r]);
     }
    }
    // JOSHUA_QDOT_SUBBLOCK_END
   }
  }
 }
 for(int r=0;r<ROWS;r++)partial[r*WG+t]=acc[r];barrier();
 if(lane==0&&n<pc.N){
  for(int r=0;r<ROWS;r++){
   if(m+r<pc.M){double sum=double(0.0);for(int i=0;i<LANES;i++)sum+=partial[r*WG+local_row*LANES+i];C[pc.coff+(m+r)*pc.N+n]=float(sum);}
  }
 }
}
"#;
    s
}

#[cfg(test)]
pub fn k_qgemv_compensated(wg: usize, groups: usize, lanes: usize, rows: usize) -> String {
    let s = k_qgemv_f64(wg, groups, lanes, rows)
        .replace(
            "double acc[ROWS];for(int r=0;r<ROWS;r++)acc[r]=double(0.0);",
            "float hi[ROWS];float lo[ROWS];for(int r=0;r<ROWS;r++){hi[r]=0.0;lo[r]=0.0;}",
        )
        .replace(
            "if(m+r<pc.M)acc[r]=fma(double(X[x+r*pc.K+b*pc.qk+sg*32+i]),double(w),acc[r]);",
            r#"
      if(m+r<pc.M){
       float a=X[x+r*pc.K+b*pc.qk+sg*32+i];
       float product=a*w;
       float pe=fma(a,w,-product);
       float sum=hi[r]+product;
       float bb=sum-hi[r];
       float se=(hi[r]-(sum-bb))+(product-bb);
       float correction=lo[r]+(se+pe);
       float next=sum+correction;
       lo[r]=correction-(next-sum);
       hi[r]=next;
      }
"#,
        )
        .replace(
            "partial[r*WG+t]=acc[r];",
            "partial[r*WG+t]=double(hi[r])+double(lo[r]);",
        );
    format!("// JOSHUA_NO_CONTRACTION\n{s}")
}

/// Defer the high/low renormalization to each 32-value quantized sub-block.
/// Product and addition residuals are still retained for every input term.
#[cfg(test)]
pub fn k_qgemv_compensated_block(wg: usize, groups: usize, lanes: usize, rows: usize) -> String {
    k_qgemv_compensated(wg,groups,lanes,rows)
        .replace("float correction=lo[r]+(se+pe);\n       float next=sum+correction;\n       lo[r]=correction-(next-sum);\n       hi[r]=next;", "lo[r]=lo[r]+(se+pe);\n       hi[r]=sum;")
        .replace("// JOSHUA_QDOT_SUBBLOCK_END", r#"
    for(int r=0;r<ROWS;r++){
     float next=hi[r]+lo[r];
     float bb=next-hi[r];
     lo[r]=(hi[r]-(next-bb))+(lo[r]-bb);
     hi[r]=next;
    }
"#)
}

/// Q8-only precise projection. Fixed block geometry and named accumulators
/// remove generic quantization branches, indirect private arrays, and row
/// loops from the inner dot. The CPU's four partial accumulation streams
/// and their final reduction order are unchanged.
pub fn k_q8_precise(wg: usize, groups: usize, rows: usize, comp: bool) -> String {
    let mut s = prelude(wg, 3);
    if comp {
        s += "// JOSHUA_NO_CONTRACTION\n";
    }
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += &format!("\n#define ROWS {rows}\n#define GROUPS {groups}\n");
    s += r#"
layout(push_constant) uniform PC {int N;int K;int qt;int qk;int bsz;int xoff;int coff;int M;} pc;
layout(local_size_x=WG) in;
shared double partial[WG*ROWS];
void main(){
 int t=int(gl_LocalInvocationID.x);int lane=t%4;
 int n=grow()*(WG/4)+t/4;int m=int(gl_WorkGroupID.z)*ROWS;
 if(m>=pc.M)return;
"#;
    for r in 0..rows {
        s += &if comp {
            format!("float hi{r}=0.0;float lo{r}=0.0;\n")
        } else {
            format!("double acc{r}=double(0.0);\n")
        };
    }
    s += r#"
 if(n<pc.N){
  int x=pc.xoff+(m*GROUPS+(GROUPS>1?n/(pc.N/GROUPS):0))*pc.K;
  int nb=pc.K/32;
  for(int b=0;b<nb;b++){
   uint p=uint(n*nb+b)*34u;
   float d=unpackHalf2x16(W[p>>2u]>>((p&3u)*8u)).x;
"#;
    for j in 0..8 {
        s+=&format!("{{int k=b*32+lane+{};uint qp=p+2u+uint(lane)+{}u;int qi=int(W[qp>>2u]<<((3u-(qp&3u))*8u))>>24;float w=d*float(qi);\n",j*4,j*4);
        for r in 0..rows {
            s += &format!("if(m+{r}<pc.M){{\n");
            if comp {
                s += &format!(
                    r#"
 float a=X[x+{r}*GROUPS*pc.K+k];float product=a*w;float pe=fma(a,w,-product);
 float sum=hi{r}+product;float bb=sum-hi{r};float se=(hi{r}-(sum-bb))+(product-bb);
 float correction=lo{r}+(se+pe);float next=sum+correction;
 lo{r}=correction-(next-sum);hi{r}=next;
"#
                );
            } else {
                s += &format!("acc{r}=fma(double(X[x+{r}*GROUPS*pc.K+k]),double(w),acc{r});\n");
            }
            s += "}\n";
        }
        s += "}\n";
    }
    s += "  }\n }\n";
    for r in 0..rows {
        s += &if comp {
            format!("partial[{r}*WG+t]=double(hi{r})+double(lo{r});\n")
        } else {
            format!("partial[{r}*WG+t]=acc{r};\n")
        };
    }
    s += "barrier();\nif(lane==0&&n<pc.N){\n";
    for r in 0..rows {
        s+=&format!("if(m+{r}<pc.M){{double sum=double(0.0);for(int i=0;i<4;i++)sum+=partial[{r}*WG+(t/4)*4+i];C[pc.coff+(m+{r})*pc.N+n]=float(sum);}}\n");
    }
    s += "}\n}\n";
    s
}

/// Work-group size of [`k_qgemv_mali`]: four 16-wide Mali warps.
pub const QGEMV_MALI_WG: usize = 64;
/// Output columns per [`k_qgemv_mali`] work-group (`QGEMV_MALI_WG` / 4
/// lanes per column).
pub const QGEMV_MALI_COLS: usize = 16;
/// Activation rows each [`k_qgemv_mali`] invocation carries.
pub const QGEMV_MALI_ROWS: usize = 4;

/// [`k_qgemv`] shaped for Arm Mali / Immortalis GPUs (Valhall and the 5th
/// generation, e.g. the Immortalis-G720 of the CIX P1 in the Orange Pi 6):
///
/// * **Decode once for several rows.**  Each weight sub-block is
///   dequantized once and dotted against up to four activation rows held
///   in scalar accumulators (`k_qgemv` re-decodes it per row: `M`
///   work-groups per column).
/// * **Small teams, one barrier.**  Four lanes share an output column and
///   a 64-invocation work-group (four 16-wide warps) covers 16 columns, so
///   every warp is busy on a decode-sized `N` and the partial sums meet in
///   one shared-memory step — Mali's shared memory is ordinary cached
///   memory, so `k_qgemv`'s 256-wide tree (eight barriers) is its costliest
///   part there.
/// * **No divergent row tests in the inner loop.**  Rows past `M` read a
///   valid row and are dropped at the store.
///
/// Same push constants as [`k_qgemv`]; the grid is
/// `ceil(N / QGEMV_MALI_COLS)` groups (2-D) by `ceil(M / QGEMV_MALI_ROWS)`.
pub fn k_qgemv_mali(_wg: usize) -> String {
    let mut s = prelude(QGEMV_MALI_WG, 3);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += QUANT_FN;
    s += &format!(
        r#"
#define TEAM {team}
#define COLS {cols}
#define RB {rows}
layout(push_constant) uniform PC {{ int N; int K; int qt; int qk; int bsz; int xoff; int coff; int M; }} pc;
layout(local_size_x = WG) in;
shared float sh[RB * WG];
void main() {{
    int t = int(gl_LocalInvocationID.x);
    int lane = t % TEAM;
    int n = grow() * COLS + t / TEAM;
    int m0 = int(gl_WorkGroupID.z) * RB;
    float a0 = 0.0; float a1 = 0.0; float a2 = 0.0; float a3 = 0.0;
    if (n < pc.N) {{
        int spb = pc.qk / 32;
        int nsub = pc.K / 32;
        uint row = uint(n) * uint(pc.K / pc.qk) * uint(pc.bsz);
        int last = pc.M - 1 - m0;
        int x0 = pc.xoff + m0 * pc.K;
        int x1 = pc.xoff + (m0 + min(1, last)) * pc.K;
        int x2 = pc.xoff + (m0 + min(2, last)) * pc.K;
        int x3 = pc.xoff + (m0 + min(3, last)) * pc.K;
        for (int s = lane; s < nsub; s += TEAM) {{
            int b = s / spb;
            int jj = s - b * spb;
            dequant_sub(pc.qt, row + uint(b) * uint(pc.bsz), jj);
            int o = s * 32;
            for (int i = 0; i < 32; i++) {{
                float w = dq[i];
                a0 += X[x0 + o + i] * w;
                a1 += X[x1 + o + i] * w;
                a2 += X[x2 + o + i] * w;
                a3 += X[x3 + o + i] * w;
            }}
        }}
    }}
    sh[t] = a0;
    sh[WG + t] = a1;
    sh[2 * WG + t] = a2;
    sh[3 * WG + t] = a3;
    barrier();
    if (lane == 0 && n < pc.N) {{
        for (int r = 0; r < RB; r++) {{
            if (m0 + r < pc.M) {{
                int h = r * WG + t;
                C[pc.coff + (m0 + r) * pc.N + n] = (sh[h] + sh[h + 1]) + (sh[h + 2] + sh[h + 3]);
            }}
        }}
    }}
}}
"#,
        team = QGEMV_MALI_WG / QGEMV_MALI_COLS,
        cols = QGEMV_MALI_COLS,
        rows = QGEMV_MALI_ROWS,
    );
    s
}

/// Quantize activations to int8 per 32 values the way llama.cpp's
/// `quantize_row_q8_0` does (`d = max|x| / 127`, `q = round(x / d)`, half
/// away from zero), for the int8 GEMV (`wgsl::k_qgemv_mali_i8`): one
/// invocation per 32-value block writes its 8 words of packed signed bytes
/// to `XQ`, its scale to `XS` and the sums of its two halves' `q` to
/// `XSUM`.  Push: `nblk, xoff`.  Bindings: `X, XQ, XS, XSUM`.
pub fn k_quant_x8(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "XQ", false);
    s += &buf(2, "float", "XS", false);
    s += &buf(3, "int", "XSUM", false);
    s += r#"
layout(push_constant) uniform PC { int nblk; int xoff; } pc;
layout(local_size_x = WG) in;
void main() {
    int b = gid();
    if (b >= pc.nblk) return;
    int x0 = pc.xoff + 32 * b;
    float amax = 0.0;
    for (int i = 0; i < 32; i++) amax = max(amax, abs(X[x0 + i]));
    float d = amax / 127.0;
    float id = d != 0.0 ? 1.0 / d : 0.0;
    int sum0 = 0;
    int sum1 = 0;
    for (int w = 0; w < 8; w++) {
        uint word = 0u;
        for (int e = 0; e < 4; e++) {
            float v = X[x0 + 4 * w + e] * id;
            int q = int(sign(v) * floor(abs(v) + 0.5));
            word |= (uint(q) & 0xFFu) << uint(8 * e);
            if (w < 4) sum0 += q; else sum1 += q;
        }
        XQ[8 * b + w] = word;
    }
    XS[b] = d;
    XSUM[2 * b] = sum0;
    XSUM[2 * b + 1] = sum1;
}
"#;
    s
}

/// Half / bf16 weight GEMV: `C[m, n] = sum_k X[m, k] * W[n, k]`.
/// Push: `N, K, is_bf16, xoff, coff, M`.
pub fn k_hgemv(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "uint", "W", true);
    s += &buf(2, "float", "C", false);
    s += r#"
float h_at(int e, int bf16) {
    uint v = (W[e >> 1] >> (uint(e & 1) * 16u)) & 0xFFFFu;
    return bf16 != 0 ? uintBitsToFloat(v << 16u) : unpackHalf2x16(v).x;
}
layout(push_constant) uniform PC { int N; int K; int is_bf16; int xoff; int coff; int M; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];
void main() {
    int n = grow();
    int m = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    if (n >= pc.N || m >= pc.M) return;
    int x = pc.xoff + m * pc.K;
    int row = n * pc.K;
    float acc = 0.0;
    for (int k = t; k < pc.K; k += WG) acc += X[x + k] * h_at(row + k, pc.is_bf16);
    sh[t] = acc;
    barrier();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier();
    }
    if (t == 0) C[pc.coff + m * pc.N + n] = sh[0];
}
"#;
    s
}

/// Dequantize a whole block tensor to f32: one invocation per 32-element
/// sub-block.  Push: `nsub, qt, qk, bsz`.
pub fn k_dequant(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "float", "O", false);
    s += QUANT_FN;
    s += r#"
layout(push_constant) uniform PC { int nsub; int qt; int qk; int bsz; } pc;
layout(local_size_x = WG) in;
void main() {
    int s = gid();
    if (s >= pc.nsub) return;
    int spb = pc.qk / 32;
    int b = s / spb;
    int jj = s - b * spb;
    dequant_sub(pc.qt, uint(b) * uint(pc.bsz), jj);
    int o = s * 32;
    for (int i = 0; i < 32; i++) O[o + i] = dq[i];
}
"#;
    s
}

/// Dequantize f16 / bf16 elements to f32.  Push: `n, is_bf16`.
pub fn k_dequant_half(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "float", "O", false);
    s += r#"
layout(push_constant) uniform PC { int n; int is_bf16; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    uint v = (W[i >> 1] >> (uint(i & 1) * 16u)) & 0xFFFFu;
    O[i] = pc.is_bf16 != 0 ? uintBitsToFloat(v << 16u) : unpackHalf2x16(v).x;
}
"#;
    s
}

/// Half-precision embedding gather: `O[j, :] = f32(W[ids[j], :])` for an
/// f16 / bf16 `[vocab, K]` table, one invocation per output element.
/// Push: `n, K, is_bf16, ids_off, vocab, fslot`.
pub fn k_hembed(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "float", "O", false);
    s += &buf(3, "uint", "fault", false);
    s += r#"
float h_at(int e, int bf16) {
    uint v = (W[e >> 1] >> (uint(e & 1) * 16u)) & 0xFFFFu;
    return bf16 != 0 ? uintBitsToFloat(v << 16u) : unpackHalf2x16(v).x;
}
layout(push_constant) uniform PC { int n; int K; int is_bf16; int ids_off; int vocab; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int i = gid();
    if (i >= pc.n) return;
    int j = i / pc.K;
    int c = i - j * pc.K;
    uint id = ids[pc.ids_off + j];
    if (id >= uint(pc.vocab)) { fault[pc.fslot] = 1u; return; }
    O[i] = h_at(int(id) * pc.K + c, pc.is_bf16);
}
"#;
    s
}

/// Quantized embedding gather: `O[j, :] = dequant(W[ids[j], :])`, one
/// work-group per row striding over sub-blocks.
/// Push: `n_ids, K, qt, qk, bsz, ids_off, vocab`.
pub fn k_qembed(wg: usize) -> String {
    let mut s = prelude(wg, 4);
    s += &buf(0, "uint", "W", true);
    s += &buf(1, "uint", "ids", true);
    s += &buf(2, "float", "O", false);
    s += &buf(3, "uint", "fault", false);
    s += QUANT_FN;
    s += r#"
layout(push_constant) uniform PC { int n_ids; int K; int qt; int qk; int bsz; int ids_off; int vocab; int fslot; } pc;
layout(local_size_x = WG) in;
void main() {
    int j = grow();
    int t = int(gl_LocalInvocationID.x);
    if (j >= pc.n_ids) return;
    uint id = ids[pc.ids_off + j];
    if (id >= uint(pc.vocab)) { fault[pc.fslot] = 1u; return; }
    int spb = pc.qk / 32;
    int nsub = pc.K / 32;
    uint row = id * uint(pc.K / pc.qk) * uint(pc.bsz);
    int o = j * pc.K;
    for (int s = t; s < nsub; s += WG) {
        int b = s / spb;
        int jj = s - b * spb;
        dequant_sub(pc.qt, row + uint(b) * uint(pc.bsz), jj);
        for (int i = 0; i < 32; i++) O[o + s * 32 + i] = dq[i];
    }
}
"#;
    s
}

const PRECISE_MATH: &str = r#"
double wide_exp(double x) {
    if (isnan(x)) return x;
    if (x < double(-208.0)) return double(0.0);
    if (x > double(89.0)) return double(uintBitsToFloat(0x7f800000u));
    double inv = double(uintBitsToFloat(0x3fb8aa3bu)) + double(uintBitsToFloat(0x32a57060u));
    int n = int(floor(fma(x,inv,double(0.5))));
    double r = fma(-double(n),double(uintBitsToFloat(0x3f317218u)),x);
    r = fma(-double(n),double(uintBitsToFloat(0xb102e308u)),r);
    r = fma(-double(n),double(uintBitsToFloat(0xa4ca86c4u)),r);
    double p = double(1.0)/(double(6227020800.0)*double(14.0));
    p = fma(p,r,double(1.0)/double(6227020800.0));
    p = fma(p,r,double(1.0)/double(479001600.0));
    p = fma(p,r,double(1.0)/double(39916800.0));
    p = fma(p,r,double(1.0)/double(3628800.0));
    p = fma(p,r,double(1.0)/double(362880.0));
    p = fma(p,r,double(1.0)/double(40320.0));
    p = fma(p,r,double(1.0)/double(5040.0));
    p = fma(p,r,double(1.0)/double(720.0));
    p = fma(p,r,double(1.0)/double(120.0));
    p = fma(p,r,double(1.0)/double(24.0));
    p = fma(p,r,double(1.0)/double(6.0));
    p = fma(p,r,double(1.0)/double(2.0));
    p = fma(p,r,double(1.0)/double(1.0));
    p = fma(p,r,double(1.0)/double(1.0));
    int h = n/3;
    return p*double(uintBitsToFloat(uint(h+127)<<23u))*double(uintBitsToFloat(uint(h+127)<<23u))*double(uintBitsToFloat(uint(n-2*h+127)<<23u));
}
float wide_unary(float x,int op) {
    double v = double(x);
    if (op == 0) return float(wide_exp(v));
    double e = wide_exp(-abs(v));
    if (op == 3) {
        double z = e/(double(2.0)+e);
        double zz = z*z;
        double p = double(1.0)/double(31.0);
        p = fma(p,zz,double(1.0)/double(29.0));
        p = fma(p,zz,double(1.0)/double(27.0));
        p = fma(p,zz,double(1.0)/double(25.0));
        p = fma(p,zz,double(1.0)/double(23.0));
        p = fma(p,zz,double(1.0)/double(21.0));
        p = fma(p,zz,double(1.0)/double(19.0));
        p = fma(p,zz,double(1.0)/double(17.0));
        p = fma(p,zz,double(1.0)/double(15.0));
        p = fma(p,zz,double(1.0)/double(13.0));
        p = fma(p,zz,double(1.0)/double(11.0));
        p = fma(p,zz,double(1.0)/double(9.0));
        p = fma(p,zz,double(1.0)/double(7.0));
        p = fma(p,zz,double(1.0)/double(5.0));
        p = fma(p,zz,double(1.0)/double(3.0));
        p = fma(p,zz,double(1.0)/double(1.0));
        return float(sqrt(max(v,double(0.0))+double(2.0)*z*p));
    }
    double sigmoid = v >= double(0.0) ? double(1.0)/(double(1.0)+e) : e/(double(1.0)+e);
    return float(op == 1 ? sigmoid : v*sigmoid);
}
"#;

pub fn k_precise_unary(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "float", "O", false);
    s += PRECISE_MATH;
    s += r#"
layout(push_constant) uniform PC { int n; int off; int op; } pc;
layout(local_size_x=WG) in;
void main(){int i=gid();if(i<pc.n)O[i]=wide_unary(X[pc.off+i],pc.op);}
"#;
    s
}

/// One invocation owns a complete HC=4 row. Float temporaries deliberately
/// retain all the storage-rounding boundaries of the unfused implementation.
/// Output has three contiguous sections so consumers receive borrowed views.
pub fn k_fused_mhc_serial(_wg: usize) -> String {
    let mut s = prelude(64, 3);
    s += "// JOSHUA_NO_CONTRACTION\n";
    s += &buf(0, "float", "X", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "O", false);
    s += PRECISE_MATH;
    s += r#"
layout(push_constant) uniform PC {int rows;int xoff;int boff;int iters;float eps;float s0;float s1;float s2;} pc;
layout(local_size_x=64) in;
void main(){
 int row=gid();if(row>=pc.rows)return;
 int x=pc.xoff+row*24;
 for(int c=0;c<4;c++){
  float pre=float(double(X[x+c])*double(pc.s0))+0.0;
  pre=pre+B[pc.boff+c];
  O[row*4+c]=wide_unary(pre,1)+pc.eps;
  float post=float(double(X[x+4+c])*double(pc.s1))+0.0;
  post=post+B[pc.boff+4+c];
  O[pc.rows*4+row*4+c]=wide_unary(post,1)*2.0+0.0;
 }
 float v[16];
 for(int i=0;i<16;i++){float scaled=float(double(X[x+8+i])*double(pc.s2))+0.0;v[i]=scaled+B[pc.boff+8+i];}
 for(int r=0;r<4;r++){
  float m=-uintBitsToFloat(0x7f800000u);for(int c=0;c<4;c++)m=max(m,v[r*4+c]);
  double e[4];double sum=double(0.0);
  for(int c=0;c<4;c++){e[c]=wide_exp(double(v[r*4+c])-double(m));sum+=e[c];}
  for(int c=0;c<4;c++)v[r*4+c]=float(e[c]/sum)+pc.eps;
 }
 // Initial column normalization is required even for iters=0.
 for(int iteration=0;iteration<max(pc.iters,1);iteration++){
  if(iteration>0){
   for(int r=0;r<4;r++){
    double sum=double(0.0);for(int c=0;c<4;c++)sum+=double(v[r*4+c]);
    float den=float(sum)+pc.eps;
    for(int c=0;c<4;c++)v[r*4+c]=float(double(v[r*4+c])/double(den));
   }
  }
  for(int c=0;c<4;c++){
   double sum=double(0.0);for(int r=0;r<4;r++)sum+=double(v[r*4+c]);
   float den=float(sum)+pc.eps;
   for(int r=0;r<4;r++)v[r*4+c]=float(double(v[r*4+c])/double(den));
  }
 }
 for(int i=0;i<16;i++)O[pc.rows*8+row*16+i]=v[i];
}
"#;
    s
}
/// Keep the 16 coefficients in shared storage and normalize them in parallel.
/// Ping-pong buffers preserve each unfused normalization's simultaneous reads.
pub fn k_fused_mhc(_wg: usize) -> String {
    let mut s = prelude(64, 3);
    s += "// JOSHUA_NO_CONTRACTION\n";
    s += &buf(0, "float", "X", true);
    s += &buf(1, "float", "B", true);
    s += &buf(2, "float", "O", false);
    s += PRECISE_MATH;
    s += r#"
layout(push_constant) uniform PC {int rows;int xoff;int boff;int iters;float eps;float s0;float s1;float s2;} pc;
layout(local_size_x=64) in;
shared float V[32];
shared double E[16];
void balance_hc(bool row_axis,int src,int dst){
 int t=int(gl_LocalInvocationID.x);
 if(t<16){
  int r=t/4;int c=t%4;double sum=double(0.0);
  for(int j=0;j<4;j++)sum+=double(V[src+(row_axis?r*4+j:j*4+c)]);
  float den=float(sum)+pc.eps;
  V[dst+t]=float(double(V[src+t])/double(den));
 }
 barrier();
}
void main(){
 int row=grow();int t=int(gl_LocalInvocationID.x);if(row>=pc.rows)return;
 int x=pc.xoff+row*24;
 if(t<4){
  float pre=float(double(X[x+t])*double(pc.s0))+0.0;pre=pre+B[pc.boff+t];
  O[row*4+t]=wide_unary(pre,1)+pc.eps;
  float post=float(double(X[x+4+t])*double(pc.s1))+0.0;post=post+B[pc.boff+4+t];
  O[pc.rows*4+row*4+t]=wide_unary(post,1)*2.0+0.0;
 }
 if(t<16){float scaled=float(double(X[x+8+t])*double(pc.s2))+0.0;V[t]=scaled+B[pc.boff+8+t];}
 barrier();
 if(t<16){
  int r=t/4;float m=-uintBitsToFloat(0x7f800000u);
  for(int c=0;c<4;c++)m=max(m,V[r*4+c]);
  E[t]=wide_exp(double(V[t])-double(m));
 }
 barrier();
 if(t<16){
  int r=t/4;double sum=double(0.0);for(int c=0;c<4;c++)sum+=E[r*4+c];
  V[t]=float(E[t]/sum)+pc.eps;
 }
 barrier();
 balance_hc(false,0,16);
 for(int iteration=1;iteration<max(pc.iters,1);iteration++){
  balance_hc(true,16,0);
  balance_hc(false,0,16);
 }
 if(t<16)O[pc.rows*8+row*16+t]=V[16+t];
}
"#;
    s
}
/// Fuse HC4 collapse while preserving the f32 product before the ordered
/// f64 reduction. No hc-times-expanded weighted activation is allocated.
pub fn k_mhc_collapse(wg: usize) -> String {
    let mut s = prelude(wg, 3);
    s += "// JOSHUA_NO_CONTRACTION\n";
    s += &buf(0, "float", "X", true);
    s += &buf(1, "float", "P", true);
    s += &buf(2, "float", "O", false);
    s += r#"
layout(push_constant) uniform PC {int len;int width;int xoff;int poff;} pc;
layout(local_size_x=WG) in;
void main(){
 int i=gid();if(i>=pc.len)return;
 int row=i/pc.width;int col=i%pc.width;
 double sum=double(0.0);
 for(int h=0;h<4;h++){
  float product=X[pc.xoff+(row*4+h)*pc.width+col]*P[pc.poff+row*4+h];
  sum+=double(product);
 }
 O[i]=float(sum);
}
"#;
    s
}

pub fn k_precise_rows(wg: usize) -> String {
    let mut s = prelude(wg, 2);
    s += &buf(0, "float", "X", true);
    s += &buf(1, "float", "O", false);
    s += PRECISE_MATH;
    s += r#"
layout(push_constant) uniform PC {int rows;int cols;int inner;int off;int soft;} pc;
layout(local_size_x=WG) in;
void main(){
 int row=gid();if(row>=pc.rows)return;
 int base=(row/pc.inner)*pc.cols*pc.inner+row%pc.inner;
 if(pc.soft==0){double sum=double(0.0);for(int c=0;c<pc.cols;c++)sum+=double(X[pc.off+base+c*pc.inner]);O[row]=float(sum);}
 else{
  float m=-uintBitsToFloat(0x7f800000u);for(int c=0;c<pc.cols;c++)m=max(m,X[pc.off+base+c*pc.inner]);
  double sum=double(0.0);for(int c=0;c<pc.cols;c++)sum+=wide_exp(double(X[pc.off+base+c*pc.inner])-double(m));
  for(int c=0;c<pc.cols;c++)O[base+c*pc.inner]=float(wide_exp(double(X[pc.off+base+c*pc.inner])-double(m))/sum);
 }
}
"#;
    s
}
pub fn k_unary_precise(wg: usize) -> String {
    k_unary(wg)
        .replace(
            "case 6: return 1.0 / v;",
            "case 6: return float(double(1.0)/double(v));",
        )
        .replace(
            "case 8: return sqrt(v);",
            "case 8: return float(sqrt(double(v)));",
        )
        .replace(
            "case 20: return v * f0 + f1;",
            "case 20: { float product=float(double(v)*double(f0)); return product+f1; }",
        )
}
pub fn k_binary_precise(wg: usize) -> String {
    k_binary(wg, false).replace(
        "case 3: return a / b;",
        "case 3: return float(double(a)/double(b));",
    )
}
