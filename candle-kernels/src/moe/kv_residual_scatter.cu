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

// Device-side `slot` variant — reads the slot index from a device tensor
// instead of taking it as a host int. Required for CUDA graph capture:
// the slot value can be updated from the host (via slice_set into the
// `cur_pos_dev` tensor) OUTSIDE the captured graph each token, while
// the captured kernel always reads from the same pointer.
__global__ void kv_residual_scatter_kernel_dev_slot(
    const __half  * __restrict__ src,
    __half        * __restrict__ dst,
    const int32_t * __restrict__ slot_dev,   // device ptr to current pos
    int n_kv,
    int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv * head_dim;
    if (idx >= total) return;
    const int slot = slot_dev[0] & 31;       // pos % 32 — written by host
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

// Device-slot variant — slot read from `slot_dev[0] % 32` on the GPU.
extern "C" void kv_residual_scatter_f16_dev_slot(
    const void  * src,
    void        * dst,
    const void  * slot_dev,    // device ptr to i32 (1 element)
    int n_kv,
    int head_dim,
    cudaStream_t stream
) {
    const int total = n_kv * head_dim;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    ll_kv_scatter::kv_residual_scatter_kernel_dev_slot<<<grid, block, 0, stream>>>(
        (const __half *)src,
        (__half *)dst,
        (const int32_t *)slot_dev,
        n_kv,
        head_dim
    );
}
