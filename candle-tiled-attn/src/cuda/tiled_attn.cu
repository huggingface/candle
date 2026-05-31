#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void tiled_attn_decode_f16_kernel(
    half* __restrict__ out,
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    float softmax_scale,
    int batches,
    int heads,
    int kv_len,
    int head_dim
) {
    extern __shared__ float scratch[];

    const int tid = threadIdx.x;
    const int bh = blockIdx.x;
    if (bh >= batches * heads || kv_len <= 0) {
        return;
    }
    const int batch = bh / heads;
    const int head = bh - batch * heads;

    const int q_base = (batch * heads + head) * head_dim;
    const int kv_head_base = (batch * heads + head) * kv_len * head_dim;
    const int out_base = (batch * heads + head) * head_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = 0; pos < kv_len; ++pos) {
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            const float qv = __half2float(q[q_base + d]);
            const float kv = __half2float(k[kv_head_base + pos * head_dim + d]);
            partial += qv * kv;
        }

        scratch[tid] = partial;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            __syncthreads();
        }

        const float score = scratch[0] * softmax_scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float beta = expf(score - m_new);

        if (tid < head_dim) {
            const float vv0 = __half2float(v[kv_head_base + pos * head_dim + tid]);
            acc0 = acc0 * alpha + vv0 * beta;
        }
        const int d1 = tid + blockDim.x;
        if (d1 < head_dim) {
            const float vv1 = __half2float(v[kv_head_base + pos * head_dim + d1]);
            acc1 = acc1 * alpha + vv1 * beta;
        }

        l = l * alpha + beta;
        m = m_new;
        __syncthreads();
    }

    const float inv_l = 1.0f / l;
    if (tid < head_dim) {
        out[out_base + tid] = __float2half_rn(acc0 * inv_l);
    }
    const int d1 = tid + blockDim.x;
    if (d1 < head_dim) {
        out[out_base + d1] = __float2half_rn(acc1 * inv_l);
    }
}
