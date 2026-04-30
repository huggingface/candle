/**
 * @brief Fused post-QKV attention kernel for single-token decode.
 *
 * After the fused QKV matmul produces a packed F32 [1, n_q*hd + 2*n_kv*hd]
 * tensor, we still need to:
 *   - apply per-head RMS norm to Q (q_norm)
 *   - apply per-head RMS norm to K (k_norm)
 *   - cast Q, K, V from F32 to the model's working dtype (F16/BF16)
 *   - apply RoPE rotation to Q and K
 *
 * Candle's stock op composition produces 5+ kernel launches per layer for
 * this stage (q_norm, k_norm, q.to_dtype, k.to_dtype, v.to_dtype, plus the
 * RoPE op which itself does cos.to_dtype + sin.to_dtype + 2 rope calls).
 * On a 48-layer MoE that's 240+ launches per decoded token, dominated by
 * fixed launch overhead rather than actual work — a profile of qwen3-coder
 * decode showed attention contributes ~110 µs/layer almost entirely from
 * launch overhead.
 *
 * This kernel does all of the above in ONE launch:
 *   - Block(0): handles head h0..h0+block_per_head Q heads
 *   - Block(1): K heads
 *   - Block(2): V heads (just cast, no norm, no RoPE)
 *
 * Block size = head_dim. Each thread handles one element of one head.
 * RMS norm is a block-local reduce over head_dim threads. RoPE pairs
 * threads at offsets (i, i + hd/2) via shared memory.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32

template <typename T>
__device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ __half        from_float<__half>(float v)        { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template<> __device__ __forceinline__ float         from_float<float>(float v)         { return v; }

template <typename T>
__device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float<__half>(__half v)               { return __half2float(v); }
template<> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template<> __device__ __forceinline__ float to_float<float>(float v)                 { return v; }

namespace ll_apq {

// Block-wide reduction across `nthreads` threads (each thread holds one
// partial value). Uses shared memory to handle blocks larger than one
// warp. Result is broadcast to all threads.
__device__ __forceinline__ float block_reduce_sum(float v, int nthreads, float* tmp) {
    // Warp reduce
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
    }
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) tmp[wid] = v;
    __syncthreads();
    if (wid == 0) {
        const int n_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;
        v = (lane < n_warps) ? tmp[lane] : 0.f;
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
        }
        if (lane == 0) tmp[0] = v;
    }
    __syncthreads();
    return tmp[0];
}

// One block processes ONE head (Q, K, or V kind selected by blockIdx.x).
// blockIdx.y = head index within the kind. blockDim.x = head_dim.
//
// Layout of qkv input (F32):
//   offset 0..n_q*hd                : Q heads (hd elements each)
//   offset n_q*hd..(n_q+n_kv)*hd    : K heads
//   offset (n_q+n_kv)*hd..(n_q+2*n_kv)*hd : V heads
//
// Outputs are 3 separate tensors, each laid out as [num_heads, hd].
// Same kernel as legacy, but Q is written in F32. Used by the Q4 KV
// cache decode path so the downstream score kernel doesn't need a
// to_dtype(F32) launch on Q. Saves one launch per layer per token.
//
// WithNorm template parameter: 1 = apply RMS norm to Q/K (qwen3 style),
// 0 = skip norm (qwen2 / llama / mistral). When WithNorm=0 the norm
// weight pointers are unused; pass any non-null value (or nullptr —
// they're not dereferenced).
template <typename T, int RopeStyle, int WithNorm = 1>
__global__ void attn_post_qkv_decode_qf32_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ q_norm_w,
    const float* __restrict__ k_norm_w,
    const T*     __restrict__ rope_cos,
    const T*     __restrict__ rope_sin,
    float*       __restrict__ q_out,       // [n_q, hd] FLOAT
    T*           __restrict__ k_out,
    float*       __restrict__ v_out,       // [n_kv, hd] FLOAT (used by Q4 quantize)
    int n_q, int n_kv, int hd,
    int rope_pos,
    float rms_eps,
    float q_scale
) {
    const int kind = blockIdx.x;
    const int head = blockIdx.y;
    const int tid  = threadIdx.x;

    if ((kind == 0 && head >= n_q)
     || (kind == 1 && head >= n_kv)
     || (kind == 2 && head >= n_kv)) return;

    extern __shared__ float smem[];
    float* warp_tmp = smem;
    float* shared_v = smem + ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE);

    int off;
    if (kind == 0)      off = head * hd;
    else if (kind == 1) off = n_q * hd + head * hd;
    else                off = n_q * hd + n_kv * hd + head * hd;

    float v = qkv[off + tid];

    if (kind == 2) {
        v_out[head * hd + tid] = v;     // F32 output
        return;
    }

    if constexpr (WithNorm == 1) {
        float sq = v * v;
        float ssum = block_reduce_sum(sq, hd, warp_tmp);
        float inv_rms = rsqrtf(ssum / (float)hd + rms_eps);
        const float w = (kind == 0) ? q_norm_w[tid] : k_norm_w[tid];
        v = v * inv_rms * w;
    }

    shared_v[tid] = v;
    __syncthreads();

    int half = hd / 2;
    float cur, other;
    int rope_idx;
    const int row_base = rope_pos * half;
    if constexpr (RopeStyle == 0) {
        if (tid < half) {
            cur = shared_v[tid];
            other = shared_v[tid + half];
            rope_idx = tid;
            float c = to_float<T>(rope_cos[row_base + rope_idx]);
            float s = to_float<T>(rope_sin[row_base + rope_idx]);
            v = cur * c - other * s;
        } else {
            cur = shared_v[tid];
            other = shared_v[tid - half];
            rope_idx = tid - half;
            float c = to_float<T>(rope_cos[row_base + rope_idx]);
            float s = to_float<T>(rope_sin[row_base + rope_idx]);
            v = cur * c + other * s;
        }
    } else {
        const int pair = (tid & 1) ? tid - 1 : tid + 1;
        cur = shared_v[tid];
        other = shared_v[pair];
        rope_idx = tid >> 1;
        float c = to_float<T>(rope_cos[row_base + rope_idx]);
        float s = to_float<T>(rope_sin[row_base + rope_idx]);
        if ((tid & 1) == 0) v = cur * c - other * s;
        else                v = other * s + cur * c;
    }

    if (kind == 0) {
        v *= q_scale;
        q_out[head * hd + tid] = v;     // F32 output
    } else {
        k_out[head * hd + tid] = from_float<T>(v);
    }
}

// RopeStyle: 0 = "neox" (paired halves: i and i+hd/2)
//            1 = "gpt-j" / "interleaved" (paired adjacent: 2i and 2i+1)
template <typename T, int RopeStyle>
__global__ void attn_post_qkv_decode_kernel(
    const float* __restrict__ qkv,         // [n_q*hd + 2*n_kv*hd]
    const float* __restrict__ q_norm_w,    // [hd]
    const float* __restrict__ k_norm_w,    // [hd]
    const T*     __restrict__ rope_cos,    // [max_seq, hd/2] — full table
    const T*     __restrict__ rope_sin,    // [max_seq, hd/2]
    T*           __restrict__ q_out,       // [n_q, hd]
    T*           __restrict__ k_out,       // [n_kv, hd]
    T*           __restrict__ v_out,       // [n_kv, hd]
    int n_q, int n_kv, int hd,
    int rope_pos,
    float rms_eps,
    float q_scale                           // multiplied into Q only — folds
                                            // attention's 1/sqrt(d) scale so
                                            // the explicit affine launch can be
                                            // skipped downstream.
) {
    const int kind = blockIdx.x; // 0=Q, 1=K, 2=V
    const int head = blockIdx.y;
    const int tid  = threadIdx.x;

    if ((kind == 0 && head >= n_q)
     || (kind == 1 && head >= n_kv)
     || (kind == 2 && head >= n_kv)) return;

    extern __shared__ float smem[];
    float* warp_tmp  = smem;                // [(blockDim.x+31)/32]
    float* shared_v  = smem + ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE);

    // Compute the byte offset of this head in the packed qkv buffer.
    int off;
    if (kind == 0)      off = head * hd;
    else if (kind == 1) off = n_q * hd + head * hd;
    else                off = n_q * hd + n_kv * hd + head * hd;

    float v = qkv[off + tid];

    if (kind == 2) {
        // V — just cast and write
        v_out[head * hd + tid] = from_float<T>(v);
        return;
    }

    // RMS norm: rms = sqrt(mean(v^2) + eps); v = v / rms * w
    // We need a block reduction over hd threads.
    float sq = v * v;
    float ssum = block_reduce_sum(sq, hd, warp_tmp);
    float inv_rms = rsqrtf(ssum / (float)hd + rms_eps);
    const float w = (kind == 0) ? q_norm_w[tid] : k_norm_w[tid];
    v = v * inv_rms * w;

    // RoPE: pair this thread's element with its rotation partner.
    shared_v[tid] = v;
    __syncthreads();

    int half = hd / 2;
    float cur, other;
    int rope_idx;

    const int row_base = rope_pos * half; // each row has hd/2 entries
    if constexpr (RopeStyle == 0) {
        if (tid < half) {
            cur   = shared_v[tid];
            other = shared_v[tid + half];
            rope_idx = tid;
            float c = to_float<T>(rope_cos[row_base + rope_idx]);
            float s = to_float<T>(rope_sin[row_base + rope_idx]);
            v = cur * c - other * s;
        } else {
            cur   = shared_v[tid];
            other = shared_v[tid - half];
            rope_idx = tid - half;
            float c = to_float<T>(rope_cos[row_base + rope_idx]);
            float s = to_float<T>(rope_sin[row_base + rope_idx]);
            v = cur * c + other * s;
        }
    } else {
        // interleaved: pair (2i, 2i+1)
        const int pair = (tid & 1) ? tid - 1 : tid + 1;
        cur   = shared_v[tid];
        other = shared_v[pair];
        rope_idx = tid >> 1;
        float c = to_float<T>(rope_cos[row_base + rope_idx]);
        float s = to_float<T>(rope_sin[row_base + rope_idx]);
        if ((tid & 1) == 0) v = cur * c - other * s;
        else                v = other * s + cur * c;
    }

    // Apply attention's 1/sqrt(d) scale to Q only — folds the
    // downstream `scores.affine(scale, 0.0)` launch.
    if (kind == 0) {
        v *= q_scale;
        q_out[head * hd + tid] = from_float<T>(v);
    } else {
        k_out[head * hd + tid] = from_float<T>(v);
    }
}

} // namespace ll_apq

// Host launcher. dtype: 0=f16, 1=bf16. rope_style: 0=neox, 1=interleaved.
extern "C" void attn_post_qkv_decode(
    const float* qkv,
    const float* q_norm_w,
    const float* k_norm_w,
    const void*  rope_cos,
    const void*  rope_sin,
    void*        q_out,
    void*        k_out,
    void*        v_out,
    int n_q, int n_kv, int hd,
    int rope_pos,
    float rms_eps,
    float q_scale,
    int dtype,
    int rope_style,
    cudaStream_t stream
) {
    dim3 grid(3, max(n_q, n_kv), 1);
    dim3 block(hd, 1, 1);
    int n_warps = (hd + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = (n_warps + hd) * sizeof(float);

#define LAUNCH(T, STYLE) \
    ll_apq::attn_post_qkv_decode_kernel<T, STYLE> \
        <<<grid, block, smem_bytes, stream>>>( \
            qkv, q_norm_w, k_norm_w, \
            (const T*)rope_cos, (const T*)rope_sin, \
            (T*)q_out, (T*)k_out, (T*)v_out, \
            n_q, n_kv, hd, rope_pos, rms_eps, q_scale); \

    if (dtype == 0) {
        if (rope_style == 0) { LAUNCH(__half, 0); }
        else                 { LAUNCH(__half, 1); }
    } else if (dtype == 1) {
#ifndef NO_BF16_KERNEL
        if (rope_style == 0) { LAUNCH(__nv_bfloat16, 0); }
        else                 { LAUNCH(__nv_bfloat16, 1); }
#endif
    } else {
        // dtype == 2: F32
        if (rope_style == 0) { LAUNCH(float, 0); }
        else                 { LAUNCH(float, 1); }
    }
#undef LAUNCH
}

// Same as attn_post_qkv_decode but Q is written in F32 instead of `dtype`.
// Used by the Q4 KV cache decode path so the downstream attn_score F32
// kernel doesn't need a to_dtype(F32) cast launch on Q.
extern "C" void attn_post_qkv_decode_qf32(
    const float* qkv,
    const float* q_norm_w,
    const float* k_norm_w,
    const void*  rope_cos,
    const void*  rope_sin,
    float*       q_out,         // F32
    void*        k_out,
    float*       v_out,         // F32 (used by Q4 quantize)
    int n_q, int n_kv, int hd,
    int rope_pos,
    float rms_eps,
    float q_scale,
    int dtype,
    int rope_style,
    cudaStream_t stream
) {
    dim3 grid(3, max(n_q, n_kv), 1);
    dim3 block(hd, 1, 1);
    int n_warps = (hd + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = (n_warps + hd) * sizeof(float);

#define LAUNCH_QF32(T, STYLE) \
    ll_apq::attn_post_qkv_decode_qf32_kernel<T, STYLE> \
        <<<grid, block, smem_bytes, stream>>>( \
            qkv, q_norm_w, k_norm_w, \
            (const T*)rope_cos, (const T*)rope_sin, \
            q_out, (T*)k_out, v_out, \
            n_q, n_kv, hd, rope_pos, rms_eps, q_scale);

    if (dtype == 0) {
        if (rope_style == 0) { LAUNCH_QF32(__half, 0); }
        else                 { LAUNCH_QF32(__half, 1); }
    } else if (dtype == 1) {
#ifndef NO_BF16_KERNEL
        if (rope_style == 0) { LAUNCH_QF32(__nv_bfloat16, 0); }
        else                 { LAUNCH_QF32(__nv_bfloat16, 1); }
#endif
    } else {
        // dtype == 2: F32 (Q,V already F32 in this kernel; K becomes F32 too)
        if (rope_style == 0) { LAUNCH_QF32(float, 0); }
        else                 { LAUNCH_QF32(float, 1); }
    }
#undef LAUNCH_QF32
}

// No-norm variant of attn_post_qkv_decode_qf32: just RoPE + scale +
// dtype cast, no q_norm/k_norm step. For models like qwen2/llama/mistral
// that don't apply RMS norm to Q/K. Same Q/V → F32 output as the qf32
// variant.
extern "C" void attn_post_qkv_decode_qf32_no_norm(
    const float* qkv,
    const void*  rope_cos,
    const void*  rope_sin,
    float*       q_out,
    void*        k_out,
    float*       v_out,
    int n_q, int n_kv, int hd,
    int rope_pos,
    float q_scale,
    int dtype,
    int rope_style,
    cudaStream_t stream
) {
    dim3 grid(3, max(n_q, n_kv), 1);
    dim3 block(hd, 1, 1);
    int n_warps = (hd + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = (n_warps + hd) * sizeof(float);

    // Norm weights are unused but the kernel signature still references
    // them. Pass q_out for both — guarantees a valid pointer; the
    // WithNorm=0 path never dereferences them.
    const float* dummy_norm = (const float*)q_out;
    // rms_eps ignored when WithNorm=0.

#define LAUNCH_QF32_NN(T, STYLE) \
    ll_apq::attn_post_qkv_decode_qf32_kernel<T, STYLE, 0> \
        <<<grid, block, smem_bytes, stream>>>( \
            qkv, dummy_norm, dummy_norm, \
            (const T*)rope_cos, (const T*)rope_sin, \
            q_out, (T*)k_out, v_out, \
            n_q, n_kv, hd, rope_pos, 0.0f, q_scale);

    if (dtype == 0) {
        if (rope_style == 0) { LAUNCH_QF32_NN(__half, 0); }
        else                 { LAUNCH_QF32_NN(__half, 1); }
    } else if (dtype == 1) {
#ifndef NO_BF16_KERNEL
        if (rope_style == 0) { LAUNCH_QF32_NN(__nv_bfloat16, 0); }
        else                 { LAUNCH_QF32_NN(__nv_bfloat16, 1); }
#endif
    } else {
        if (rope_style == 0) { LAUNCH_QF32_NN(float, 0); }
        else                 { LAUNCH_QF32_NN(float, 1); }
    }
#undef LAUNCH_QF32_NN
}
