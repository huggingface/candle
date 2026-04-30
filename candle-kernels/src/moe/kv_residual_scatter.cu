/**
 * @brief Single-launch scatter for the Q4 KV residual write path.
 *
 * The Q4 KV cache holds a 32-position residual block in [n_kv,
 * head_dim, 32] F16 layout. On every decode step we write one new
 * (k or v) vector into slot `current_seq_len % 32`. The previous Rust
 * path did a clone_dtod + slice_set + memcpy_dtod (3 launches × 2
 * residuals per layer = 6/layer × 48 layers ≈ 288 extra launches/token).
 *
 * This kernel performs the scatter directly: each thread writes one
 * (h, c) F16 value at the strided offset `h*head_dim*32 + c*32 + slot`.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ll_kv_scatter {

__global__ void kv_residual_scatter_kernel(
    const __half * __restrict__ src,    // [n_kv * head_dim] F16 (one slot's data)
    __half       * __restrict__ dst,    // [n_kv, head_dim, 32] F16
    int n_kv,
    int head_dim,
    int slot                            // 0..31
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv * head_dim;
    if (idx >= total) return;
    const int h = idx / head_dim;
    const int c = idx - h * head_dim;
    const int dst_off = h * head_dim * 32 + c * 32 + slot;
    dst[dst_off] = src[idx];
}

} // namespace ll_kv_scatter

extern "C" void kv_residual_scatter_f16(
    const void * src,        // device ptr [n_kv * head_dim] F16
    void       * dst,        // device ptr [n_kv * head_dim * 32] F16
    int n_kv,
    int head_dim,
    int slot,
    cudaStream_t stream
) {
    const int total = n_kv * head_dim;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    ll_kv_scatter::kv_residual_scatter_kernel<<<grid, block, 0, stream>>>(
        (const __half *)src,
        (__half *)dst,
        n_kv,
        head_dim,
        slot
    );
}
