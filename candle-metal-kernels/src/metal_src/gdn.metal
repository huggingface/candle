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

// Column-parallel forward-substitution solve for gated-DeltaNet's chunked
// prefill scan (ratatoskr's `gated_delta_rule_chunked`, Phase 3.1a -- see
// yggdrasil/ratatoskr/DESIGN.md's "Phase 3 design, revised 2026-08-17"
// section for the full derivation and why this layout, not an in-place
// threadgroup tile, was chosen).
//
// Solves `(I - a_mat) * attn = I`, i.e. `attn = (I - a_mat)^-1`, for a
// strictly lower-triangular `a_mat` (`a_mat[i][k] == 0` for `k >= i`),
// per independent `(batch, head, chunk)` problem. `(I - A)X = I` decomposes
// by columns: column j of X satisfies
//   x_i = e_j[i] + sum_{k<i} a_mat[i][k] * x_k        (e_j[i] = 1 iff i==j)
// and every dependency for column j stays inside column j -- a_mat is
// read-only throughout, no thread ever reads another thread's output. So:
// **one thread per column**, zero threadgroup memory, zero barriers --
// matching every other kernel in this file. The `i == j` select (not a
// branch) keeps every thread running the identical loop regardless of
// column, so there is no thread divergence; for `i < j` it correctly
// yields `x_i = 0` by the same induction the design's own numerical-
// stability argument relies on (each `x_i` is a final entry of the answer,
// computed once from already-final `x_k`, `k < i` -- never a
// repeated-squaring-style intermediate that could blow up, unlike Phase
// 1a's reverted doubling attempt).
//
// **Deliberately does NOT hold `x[CHUNK]` in a local/register array.**
// The design's own implementation note assumed a 64-entry per-thread
// array would either stay register-resident or spill to thread-local
// memory as a benign, perf-only fallback -- verified false during 3.1a's
// own bring-up (differential against the scalar reference below): at
// `CHUNK=64` this compiler mis-lowers a dynamically-indexed 64-entry
// `float` array, silently producing zeros for most rows (confirmed by
// shrinking `CHUNK` to 4 in isolation, where the identical logic is
// bit-exact -- a real, size-dependent miscompilation, not a logic bug).
// Fixed by writing each row directly to `attn` as it's computed and
// reading prior rows back from `attn` itself (`out[k * CHUNK + j]`)
// instead of a local array -- safe with zero synchronization because a
// thread's own prior writes to device memory are always visible to its
// own later reads with no barrier required (ordering within one thread's
// instruction stream, not cross-thread visibility, which is the only
// thing `HazardTrackingModeUntracked` and barriers govern). This costs
// device-memory read/write traffic instead of register accesses --
// acceptable here, matching this stage's own documented scaffolding
// status (column-major writes are already mildly uncoalesced; 3.1b/c are
// expected to restructure this further, not preserve this exact form).
//
// One dispatch per `(batch, head, chunk)` grid of problems (`bhnc` =
// `b * n_heads * num_chunks`, already flattened by the caller, same
// convention as the tensor path's own `bhnc` reshape) -- `attn` is written
// functionally (a fresh buffer): the caller must bind it via the write
// path (Output) so this fork's HazardTrackingModeUntracked convention
// gives any dependent dispatch (3.1b/c's later folds) a real barrier.
#define GDN_SCAN_CHUNK 64u

struct gdn_scan_solve_args {
    uint bhnc; // number of independent (batch, head, chunk) problems
};

kernel void kernel_gdn_chunked_scan_solve_f32(
        device const float * a_mat [[buffer(0)]],  // [bhnc, CHUNK, CHUNK], strictly lower-triangular
        device float       * attn  [[buffer(1)]],  // [bhnc, CHUNK, CHUNK], functional -- (I - a_mat)^-1
        constant gdn_scan_solve_args & args [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x; // column index, 0..CHUNK
    const uint p = gid.y; // which (batch, head, chunk) problem
    if (j >= GDN_SCAN_CHUNK || p >= args.bhnc) {
        return;
    }

    device const float * a   = a_mat + p * GDN_SCAN_CHUNK * GDN_SCAN_CHUNK;
    device float       * out = attn  + p * GDN_SCAN_CHUNK * GDN_SCAN_CHUNK;

    for (uint i = 0; i < GDN_SCAN_CHUNK; ++i) {
        float acc = (i == j) ? 1.0f : 0.0f;
        for (uint k = 0; k < i; ++k) {
            acc += a[i * GDN_SCAN_CHUNK + k] * out[k * GDN_SCAN_CHUNK + j];
        }
        out[i * GDN_SCAN_CHUNK + j] = acc;
    }
}

// Phase 3.1b: folds a_mat's construction (mask/kk/triangle) into the
// solve kernel above -- this fork's first kernel using threadgroup
// memory and a barrier at all. See yggdrasil/ratatoskr/DESIGN.md's
// "Phase 3 design" section for the design consult this implements
// (2026-08-17, verified against qwen3_5_linear_attn_scan.rs's steps
// 3a-3c before being recorded).
//
// Per (batch, head, chunk) problem, for j < i (strictly lower
// triangular, matching a_mat's own definition):
//   g_cumsum[i] - g_cumsum[j] = sum_{m=j+1..i} log_g[m]   (running
//     suffix sum, computed directly -- not two independent prefix
//     sums subtracted, which is what the tensor path does; this is
//     the *more* numerically accurate grouping, not a shortcut)
//   kk[i][j]    = beta[i] * dot(k_c[i,:], k_c[j,:])
//   a_mat[i][j] = -kk[i][j] * exp(g_cumsum[i] - g_cumsum[j])
// log_g is always <= 0 (ssm_a * softplus(...), confirmed against this
// crate's own consumer -- quantized_qwen35.rs's own comment states
// this explicitly), so the running suffix sum is always <= 0 and
// exp(...) <= 1 always -- no overflow is possible by construction, for
// any j < i. This is why no separate mask-before-exp step exists here:
// the tensor path's own guard (this file's own doc, and DESIGN.md's
// correctness risk (1)) exists because ITS gc_i/gc_j broadcast-and-
// subtract materializes the *upper* triangle too before masking it
// away; this kernel's loop bound (i from j+1 upward) means the
// upper triangle and diagonal are never evaluated at all, structurally,
// not masked after the fact.
//
// **One thread per column j, same as the solve half** -- deliberately,
// not a separate assignment for build vs. solve. Thread j builds its
// own column's strictly-lower entries (i = j+1..CHUNK-1) using a
// single scalar running accumulator (`acc_g`) and a scalar dot-product
// accumulator -- **no per-thread array anywhere in this kernel**. This
// is a hard constraint, not a preference: 3.1a's first implementation
// held a 64-entry `float x[64]` per thread and was found to silently
// miscompile (produce zeros for most rows) at that exact size on this
// Metal toolchain -- confirmed size-dependent, not a logic bug, by
// shrinking to 4 in isolation, where identical logic was correct. See
// yggdrasil/ratatoskr/DESIGN.md's "standing risk for future kernel
// work in this fork" paragraph for the full writeup -- any kernel in
// this crate wanting a per-thread array above roughly 32-64 entries
// should differential-test at the real target size specifically, the
// same way this one did, not assume register-residency holds.
//
// a_mat lands in a single 16KB threadgroup tile (the only shared state
// this kernel needs -- k_c/log_g_c/beta_c stay device-resident and are
// read via uniform/cache-friendly access patterns, not staged).
// Exactly **one** threadgroup_barrier, after every thread's build loop
// completes and before any solve read -- dispatch is deliberately
// sized to exactly (CHUNK, bhnc) threads (grid_dims == group_dims-
// multiple, see the Rust wrapper), so **no thread ever needs a
// bounds-check return before the barrier** (an early return before a
// barrier is undefined behavior -- not every thread would reach it --
// so this kernel has none, unlike every other kernel in this file,
// deliberately).
//
// **This is a real landmine for anyone extending this kernel, not
// just an absence to note in passing.** Every OTHER kernel in this
// file uses `if (idx >= bound) { return; }`, including the literally
// adjacent kernel_gdn_chunked_scan_solve_f32 just above this one --
// copying that idiom into a barrier-containing kernel out of habit is
// undefined behavior, not merely redundant. The two halves of a
// combined check are not equally dangerous here: with this kernel's
// own `group_dims.height == 1`, `p` (the problem index) is
// threadgroup-uniform (every thread in a group has the same `p`), so a
// `p`-only early return would be legal; `j` (the column) varies within
// a group, so a `j`-only (or combined) early return before the barrier
// is the real hazard. If a future variant of this kernel ever needs
// non-exact dispatch dimensions, do not add the combined check back
// in -- restructure so any skipped work still falls through to the
// barrier (skip the work, never `return`).
//
// Numerics note for verification: this kernel's a_mat differs from
// the tensor path's own a_mat by ordinary f32 summation-order drift
// (running suffix sum vs. two-prefix-sum subtraction) -- expected to
// be tolerance-close, not bit-identical, in the Phase-1b sense (benign
// reassociation), not the Phase-1a sense (repeated-squaring blowup).
// The production-scale differential is the arbiter, same as every
// other phase in this design.
struct gdn_scan_build_solve_args {
    uint hk; // head_k_dim (k_c's last dimension) -- bhnc is NOT a field
             // here: the kernel never reads it (dispatch is sized to
             // exactly (CHUNK, bhnc) threads, so there is nothing for a
             // bhnc bounds check to do -- see the "no bounds-check
             // return" note above). Keeping an unused field would be
             // dead GPU-side state with no compiler warning to catch it.
};

kernel void kernel_gdn_chunked_scan_build_and_solve_f32(
        device const float * k_c     [[buffer(0)]],  // [bhnc, CHUNK, hk]
        device const float * log_g_c [[buffer(1)]],  // [bhnc, CHUNK]
        device const float * beta_c  [[buffer(2)]],  // [bhnc, CHUNK]
        device float       * attn    [[buffer(3)]],  // [bhnc, CHUNK, CHUNK], functional
        constant gdn_scan_build_solve_args & args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x; // this thread's column, 0..CHUNK -- exact dispatch sizing, no bounds check
    const uint p = gid.y; // which (batch, head, chunk) problem -- exact dispatch sizing, no bounds check

    // Fixed-size static threadgroup allocation (16KB) -- compile-time
    // constant, so no host-side setThreadgroupMemoryLength plumbing is
    // needed; every threadgroup gets its own private copy.
    threadgroup float a_tile[GDN_SCAN_CHUNK * GDN_SCAN_CHUNK];

    device const float * k_p     = k_c     + p * GDN_SCAN_CHUNK * args.hk;
    device const float * log_g_p = log_g_c + p * GDN_SCAN_CHUNK;
    device const float * beta_p  = beta_c  + p * GDN_SCAN_CHUNK;

    // Build: column j, strictly-lower entries only (i > j).
    float acc_g = 0.0f;
    for (uint i = j + 1; i < GDN_SCAN_CHUNK; ++i) {
        acc_g += log_g_p[i]; // always <= 0 -- see doc comment above
        float dot = 0.0f;
        for (uint d = 0; d < args.hk; ++d) {
            dot += k_p[i * args.hk + d] * k_p[j * args.hk + d];
        }
        a_tile[i * GDN_SCAN_CHUNK + j] = -beta_p[i] * dot * exp(acc_g);
    }

    // The one barrier: every thread falls through to here regardless
    // of its own build-loop trip count (thread CHUNK-1 does zero
    // build iterations but still reaches this point) -- required
    // before any thread reads another thread's tile writes.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Solve: identical math/shape to kernel_gdn_chunked_scan_solve_f32
    // above, reading a_mat from the threadgroup tile instead of device
    // memory. attn is still read back from the OUTPUT device buffer
    // for prior rows (no per-thread array here either, same reasoning
    // as the solve-only kernel).
    device float * out = attn + p * GDN_SCAN_CHUNK * GDN_SCAN_CHUNK;
    for (uint i = 0; i < GDN_SCAN_CHUNK; ++i) {
        float acc = (i == j) ? 1.0f : 0.0f;
        for (uint k = 0; k < i; ++k) {
            acc += a_tile[i * GDN_SCAN_CHUNK + k] * out[k * GDN_SCAN_CHUNK + j];
        }
        out[i * GDN_SCAN_CHUNK + j] = acc;
    }
}
