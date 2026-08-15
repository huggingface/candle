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
// Fused causal depthwise conv1d + silu, as used by gated-DeltaNet /
// gated-linear-attention hybrid architectures (Qwen3.5, Qwen3-Next-style
// models) ahead of their recurrent scan. A naive implementation processes
// this as a `for t in 0..seq_len { for k in 0..kernel_size { ... } }` loop
// over per-tap elementwise multiply-adds -- O(seq_len * kernel_size)
// separate dispatches at small batch sizes (decode, or a short
// multi-token forward like a speculative-decode verify step), each paying
// a full pipeline/bind/encode cycle for a handful of FLOPs. This fuses the
// whole causal convolution + activation into one dispatch for the output,
// plus one small dispatch for the next conv state.
//
// Conceptually operates on `padded = history ++ x` (concat along time),
// length `hist_len + seq_len`. For output position t (0 <= t < seq_len):
//   out[t][c] = silu( sum_k padded[t+k][c] * weight[c][k] )
// For next-state position s (0 <= s < hist_len) -- the trailing hist_len
// entries of `padded`, i.e. `padded[seq_len + s]`:
//   new_state[s][c] = padded[seq_len + s][c]
// Both are expressed directly against `history`/`x` (never materializing
// `padded` itself) via the same `idx < hist_len ? history[idx] : x[idx -
// hist_len]` branch -- one thread per (channel, output-position, batch), no
// cross-thread communication, no threadgroup memory. `new_state` is written
// functionally (a fresh buffer, `history` read-only and untouched) --
// deliberate, not an oversight: callers holding another reference to the
// input history (e.g. an externally-taken checkpoint) must be unaffected
// by this call running, same discipline as `kernel_gdn_decode_step_f32`
// above.
//
// weight is `[channels, kernel_size]` -- the caller is expected to
// canonicalize to this exact layout once (a raw GGUF/checkpoint tensor can
// arrive as either `[channels, kernel]` or `[kernel, channels]`), not
// re-derive it every dispatch.

struct gdn_conv1d_args {
    uint seq_len;
    uint hist_len;
    uint channels;
    uint kernel_size; // only read by the _output kernel; harmless unused field on the _state kernel
};

kernel void kernel_gdn_causal_conv1d_output_f32(
        device const float * x        [[buffer(0)]],  // [b, seq_len, channels]
        device const float * history  [[buffer(1)]],  // [b, hist_len, channels]
        device const float * weight   [[buffer(2)]],  // [channels, kernel_size]
        device float       * out      [[buffer(3)]],  // [b, seq_len, channels], silu already applied
        constant gdn_conv1d_args & args [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= args.channels || t >= args.seq_len) {
        return;
    }
    const uint hist_len = args.hist_len;
    const uint seq_len = args.seq_len;
    const uint channels = args.channels;
    const uint kernel_size = args.kernel_size;
    const uint b = gid.z;

    device const float * xb = x + b * seq_len * channels;
    device const float * hb = history + b * hist_len * channels;
    device const float * wc = weight + c * kernel_size;

    float acc = 0.0f;
    for (uint k = 0; k < kernel_size; ++k) {
        const uint idx = t + k; // index into conceptual padded = history ++ x
        const float val = (idx < hist_len)
            ? hb[idx * channels + c]
            : xb[(idx - hist_len) * channels + c];
        acc += val * wc[k];
    }
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    out[(b * seq_len + t) * channels + c] = acc / (1.0f + exp(-acc));
}

kernel void kernel_gdn_causal_conv1d_state_f32(
        device const float * x           [[buffer(0)]],  // [b, seq_len, channels]
        device const float * history     [[buffer(1)]],  // [b, hist_len, channels]
        device float       * new_state   [[buffer(2)]],  // [b, hist_len, channels], functional
        constant gdn_conv1d_args & args [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint s = gid.y;
    if (c >= args.channels || s >= args.hist_len) {
        return;
    }
    const uint hist_len = args.hist_len;
    const uint seq_len = args.seq_len;
    const uint channels = args.channels;
    const uint b = gid.z;

    device const float * xb = x + b * seq_len * channels;
    device const float * hb = history + b * hist_len * channels;

    const uint idx = seq_len + s; // index into conceptual padded = history ++ x
    const float val = (idx < hist_len)
        ? hb[idx * channels + c]
        : xb[(idx - hist_len) * channels + c];
    new_state[(b * hist_len + s) * channels + c] = val;
}

// Fused elementwise "gating tail" for gated-DeltaNet's preprocessing
// pipeline, two independent sub-kernels replacing what a naive
// implementation would otherwise dispatch as ~17 separate ops
// (L2-normalize q and k, scale q, a softplus-style decay gate, a sigmoid
// beta gate).
//
// kernel_gdn_l2_normalize_scale_f32: out[b][t][h][:] = scale *
// x[b][t][h][:] / sqrt(sum_d x[b][t][h][d]^2 + eps) -- eps is added to the
// sum of squares, not the mean, so this is *not* interchangeable with a
// generic RMS-norm kernel; match your own model's exact epsilon placement
// before reusing this. One thread per (b, t, h), looping over `dim` twice
// (sum-of-squares, then the normalized write) -- re-reads `x` rather than
// caching it in a local array, since `dim` is typically small (~64-256) and
// the extra read is cheap relative to a per-dispatch fixed-overhead
// avoided. Intended to be called once for q (with a query-side scale
// factor) and once for k (scale = 1.0) -- two dispatches of the same
// kernel, not two kernels.
struct gdn_l2norm_args {
    uint seq_len;
    uint heads;
    uint dim;
    float scale;
    float eps;
};

kernel void kernel_gdn_l2_normalize_scale_f32(
        device const float * x    [[buffer(0)]],  // [b, seq_len, heads, dim]
        device float       * out  [[buffer(1)]],  // [b, seq_len, heads, dim]
        constant gdn_l2norm_args & args [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint t = gid.y;
    if (h >= args.heads || t >= args.seq_len) {
        return;
    }
    const uint dim = args.dim;
    const uint b = gid.z;
    const uint row = (b * args.seq_len + t) * args.heads + h;
    device const float * xr = x + row * dim;
    device float       * outr = out + row * dim;

    float sum_sq = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        const float v = xr[d];
        sum_sq += v * v;
    }
    const float inv_norm = args.scale / sqrt(sum_sq + args.eps);
    for (uint d = 0; d < dim; ++d) {
        outr[d] = xr[d] * inv_norm;
    }
}

// kernel_gdn_decay_beta_gate_f32: fuses a softplus-style decay-gate chain
// and a sigmoid beta gate, the two per-head scalar gates this architecture
// family derives per token before its recurrent scan:
//   g[b][t][h]    = a[h] * log(exp(alpha_logits[b][t][h] + dt_bias[h]) + 1)
//   beta[b][t][h] = sigmoid(beta_logits[b][t][h])
// `dt_bias`/`a` are indexed directly by head (`[heads]`, not pre-broadcast
// to `[b, seq_len, heads]`) -- this also eliminates the caller's own
// broadcast step, not just the elementwise math around it. One thread per
// (b, t, h), no reduction, purely elementwise.
struct gdn_decay_beta_args {
    uint seq_len;
    uint heads;
};

kernel void kernel_gdn_decay_beta_gate_f32(
        device const float * alpha_logits [[buffer(0)]],  // [b, seq_len, heads]
        device const float * dt_bias      [[buffer(1)]],  // [heads]
        device const float * a            [[buffer(2)]],  // [heads]
        device const float * beta_logits  [[buffer(3)]],  // [b, seq_len, heads]
        device float       * g_out        [[buffer(4)]],  // [b, seq_len, heads]
        device float       * beta_out     [[buffer(5)]],  // [b, seq_len, heads]
        constant gdn_decay_beta_args & args [[buffer(6)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint t = gid.y;
    if (h >= args.heads || t >= args.seq_len) {
        return;
    }
    const uint b = gid.z;
    const uint idx = (b * args.seq_len + t) * args.heads + h;

    const float softplus = log(exp(alpha_logits[idx] + dt_bias[h]) + 1.0f);
    g_out[idx] = a[h] * softplus;
    beta_out[idx] = 1.0f / (1.0f + exp(-beta_logits[idx]));
}
