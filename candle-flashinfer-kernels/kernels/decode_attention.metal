// Metal port of the single-token (decode) attention kernel (see decode_attention.cu).
// One threadgroup per (batch, query head), one thread per head-dimension element, looping
// sequentially over the key/value cache with a numerically-stable streaming softmax. The
// per-key dot product is a threadgroup reduction. This mirrors the CUDA reference path on
// Apple Silicon; it is not a tensor-core / split-KV kernel.
#include <metal_stdlib>
using namespace metal;

// Must match `struct DecodeParams` in src/metal.rs (13 × i32 then 1 × f32).
struct DecodeParams {
    int hkv_group;
    int seqlen_k;
    int head_dim;
    int q_b_stride;
    int q_h_stride;
    int k_b_stride;
    int k_h_stride;
    int k_l_stride;
    int v_b_stride;
    int v_h_stride;
    int v_l_stride;
    int o_b_stride;
    int o_h_stride;
    float scale;
};

template <typename T>
inline void decode_attention_impl(
    device const T *q,
    device const T *k,
    device const T *v,
    device T *out,
    constant DecodeParams &p,
    threadgroup float *red,
    uint2 tg_pos,
    uint tid,
    uint tcount) {
    const int b = int(tg_pos.y);
    const int h = int(tg_pos.x);
    const int hkv = h / p.hkv_group;
    const uint hd = uint(p.head_dim);

    device const T *q_ptr = q + (long)b * p.q_b_stride + (long)h * p.q_h_stride;
    device const T *k_base = k + (long)b * p.k_b_stride + (long)hkv * p.k_h_stride;
    device const T *v_base = v + (long)b * p.v_b_stride + (long)hkv * p.v_h_stride;
    device T *o_ptr = out + (long)b * p.o_b_stride + (long)h * p.o_h_stride;

    if (p.seqlen_k == 0) {
        if (tid < hd) {
            o_ptr[tid] = T(0.0f);
        }
        return;
    }

    const float qd = (tid < hd) ? float(q_ptr[tid]) : 0.0f;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t < p.seqlen_k; ++t) {
        device const T *k_row = k_base + (long)t * p.k_l_stride;
        red[tid] = (tid < hd) ? qd * float(k_row[tid]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = tcount / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                red[tid] += red[tid + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float score = red[0] * p.scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float new_m = max(m, score);
        const float corr = exp(m - new_m);
        const float pscore = exp(score - new_m);
        l = l * corr + pscore;

        device const T *v_row = v_base + (long)t * p.v_l_stride;
        const float vd = (tid < hd) ? float(v_row[tid]) : 0.0f;
        acc = acc * corr + pscore * vd;
        m = new_m;
    }

    if (tid < hd) {
        o_ptr[tid] = T(acc / l);
    }
}

kernel void decode_attention_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant DecodeParams &p [[buffer(4)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]) {
    threadgroup float red[1024];
    decode_attention_impl<float>(q, k, v, out, p, red, tg_pos, tid, tcount);
}

kernel void decode_attention_f16(
    device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant DecodeParams &p [[buffer(4)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]) {
    threadgroup float red[1024];
    decode_attention_impl<half>(q, k, v, out, p, red, tg_pos, tid, tcount);
}
