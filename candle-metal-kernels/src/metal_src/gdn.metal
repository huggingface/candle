#include <metal_stdlib>
using namespace metal;

// Fused gated-DeltaNet (gated linear attention) single-token decode step,
// as used by Qwen3.5 / Qwen3-Next style hybrid architectures. Batch-1,
// seq-len-1 decode of this recurrence normally costs ~9 serialized,
// dependency-chained elementwise/reduction ops per layer per token
// (decay, a key-weighted read, a delta computation, a state write, a
// query-weighted read) -- cheap in FLOPs but expensive in fixed
// per-dispatch overhead at this shape, since every op depends on the
// previous op's output and the tensors involved are small. This kernel
// fuses the whole step into one dispatch.
//
// Per output column j (over the value-head-dim axis), for a fixed
// (batch, head), and summing over i (the state-size axis):
//   g           = exp(g_log)
//   s_dec[i][j] = g * s_in[i][j]
//   kv_mem[j]   = sum_i s_dec[i][j] * k[i]
//   delta[j]    = (v[j] - kv_mem[j]) * beta
//   s_out[i][j] = s_dec[i][j] + k[i] * delta[j]
//   out[j]      = sum_i s_out[i][j] * q[i]
//
// One thread per (batch, head, j) -- no cross-thread communication, no
// threadgroup memory. `state_out` is written functionally (a fresh
// buffer; `state_in` is read-only and left untouched) rather than
// mutated in place -- callers that keep other live references to the
// input state tensor (e.g. a checkpoint taken before this step) depend
// on that not being silently invalidated. Every per-thread write to
// `state_out`/`out` is to a disjoint element, so the kernel itself has no
// internal write hazard -- but the caller must bind `state_out` and
// `out` via the write path (see `Output` in this crate's `utils.rs`),
// not the read-only path, or a subsequent dispatch's read of `state_out`
// may run without a hazard-tracking barrier under
// `HazardTrackingModeUntracked`.

struct gdn_step_args {
    uint hk; // state_size
    uint hv; // value_head_dim
    uint h;  // value_head_count (heads)
};

kernel void kernel_gdn_decode_step_f32(
        device const float * q      [[buffer(0)]],  // [b, h, hk]
        device const float * k      [[buffer(1)]],  // [b, h, hk]
        device const float * v      [[buffer(2)]],  // [b, h, hv]
        device const float * g_log  [[buffer(3)]],  // [b, h], NOT exp'ed -- the raw decay-gate log; this kernel exponentiates it internally
        device const float * beta   [[buffer(4)]],  // [b, h]
        device const float * s_in   [[buffer(5)]],  // [b, h, hk, hv]
        device float       * s_out  [[buffer(6)]],  // [b, h, hk, hv], functional -- fresh buffer, s_in untouched
        device float       * out    [[buffer(7)]],  // [b, h, hv]
        constant gdn_step_args & args [[buffer(8)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    if (j >= args.hv) {
        return;
    }
    const uint hk = args.hk;
    const uint hv = args.hv;
    const uint bh = gid.z * args.h + gid.y; // flattened (batch, head) index

    device const float * qh = q + bh * hk;
    device const float * kh = k + bh * hk;
    device const float * si = s_in  + bh * hk * hv;
    device float       * so = s_out + bh * hk * hv;

    const float gv = exp(g_log[bh]); // one exp() per (batch, head), not a separate caller-side dispatch
    const float bv = beta[bh];

    float kv_mem = 0.0f;
    for (uint i = 0; i < hk; ++i) {
        kv_mem += (gv * si[i * hv + j]) * kh[i];
    }
    const float delta = (v[bh * hv + j] - kv_mem) * bv;

    float acc = 0.0f;
    for (uint i = 0; i < hk; ++i) {
        const float s_new = gv * si[i * hv + j] + kh[i] * delta;
        so[i * hv + j] = s_new;
        acc += s_new * qh[i];
    }
    out[bh * hv + j] = acc;
}
