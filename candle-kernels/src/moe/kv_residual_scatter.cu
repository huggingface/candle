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

namespace ll_kv_scatter {

// Byte-copy scatter for the Q4_0 V append path. Copies `token_bytes`
// bytes from `src` into `dst[pos_dev[0] * token_bytes ..]`.
// One thread per output byte; grid is sized to cover token_bytes.
//
// Used to replace the host-side `memcpy_dtod` with a compile-time
// `dst_byte_offset = current_seq_len * token_bytes` — under CUDA graph
// capture the host offset freezes, so replay always overwrites the
// same slot. With this kernel the offset is computed from the device
// pointer each replay, after the host updates `pos_dev` outside the
// captured region.
__global__ void q4_v_scatter_bytes_kernel(
    const uchar4 * __restrict__ src,
    uchar4       * __restrict__ dst,        // base of v_q4 buffer
    const int32_t * __restrict__ pos_dev,   // device ptr to current token index
    int token_words                          // token_bytes / 4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= token_words) return;
    const int pos = pos_dev[0];
    const int dst_off_words = pos * token_words + idx;
    dst[dst_off_words] = src[idx];
}

} // namespace ll_kv_scatter

namespace ll_kv_scatter {

// Conditional Q4_0 flush of the K residual to k_q4_blocks. Fired every
// token under graph capture, but actually quantises only when the
// just-appended position closes a 32-token window (i.e. pos_dev[0] & 31
// == 31). Otherwise the kernel does nothing — graph capture sees the
// same launch every replay, the kernel decides at run-time whether to
// write.
//
// Residual layout:  [n_kv * head_dim * 32] F16, flat. Each 32-element
//                   slice corresponds to one (h, c) row of 32 seq
//                   positions, in the same order seen by
//                   `QCudaStorage::quantize_with_layout` on the host
//                   path. One Q4_0 output block per (h, c) row.
//
// k_blocks layout:  [seq_block, n_kv, head_dim] of 18-byte Q4_0 blocks.
//                   For the closing block (block_idx = pos_dev[0] >> 5),
//                   the destination of block (h, c) within the slab is
//                   `(block_idx * n_kv * head_dim + h * head_dim + c) * 18`.
//                   Within-block bytes 0..1 are F16 scale, 2..17 are 16
//                   packed nibble bytes.
//
// Grid: (n_kv * head_dim, 1, 1). Block: (32, 1, 1). Each warp handles
// one Q4_0 output block. Quantisation matches `quantize_q4_0_f16` in
// `quantized.cu`: signed-max amplitude reduction, d = mval / -8,
// q = clamp(round(x*id) + 8, 0, 15), packing via __shfl_down(16).
__global__ void flush_k_residual_q4_dev_pos_kernel(
    const __half * __restrict__ residual,    // [n_kv * head_dim * 32] F16
    uint8_t      * __restrict__ k_blocks,    // base of [seq_block, n_kv, head_dim] Q4_0 slab
    const int32_t * __restrict__ pos_dev,    // device pos (= current_seq_len just after append)
    int blocks_per_window,                    // n_kv * head_dim
    int max_seq_blocks                         // bounds-check for block_idx
) {
    const int pos = pos_dev[0];
    // Trigger when this append closes a 32-window. `pos` here is
    // `current_seq_len` BEFORE the append (host writes it that way via
    // update_graph_state). After the append `current_seq_len` becomes
    // pos+1; the window closes when (pos+1) % 32 == 0, i.e. pos & 31 == 31.
    if ((pos & 31) != 31) return;

    const int block_idx = pos >> 5;                      // = pos / 32
    if (block_idx < 0 || block_idx >= max_seq_blocks) return;

    const int warp = blockIdx.x;                          // 0 .. blocks_per_window - 1
    if (warp >= blocks_per_window) return;
    const int lane = threadIdx.x;                         // 0..31

    // Load residual value for this (warp, lane).
    const int res_off = warp * 32 + lane;
    const float xi = __half2float(residual[res_off]);

    // Signed-max amplitude reduction across the warp (matches
    // quantize_q4_0_f16 in quantized.cu).
    float amax = fabsf(xi);
    float mval = xi;
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        float a2 = __shfl_xor_sync(0xffffffff, amax, off);
        float v2 = __shfl_xor_sync(0xffffffff, mval, off);
        if (a2 > amax) { amax = a2; mval = v2; }
    }
    const float d  = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q_int = (int)(xi * id + 8.5f);
    q_int = q_int < 0 ? 0 : (q_int > 15 ? 15 : q_int);
    const int q_high = __shfl_down_sync(0xffffffff, q_int, 16);

    // Destination: block (block_idx, warp). Within-block byte stride is 18.
    const size_t block_byte_off =
        ((size_t)block_idx * (size_t)blocks_per_window + (size_t)warp) * 18;
    uint8_t * out = k_blocks + block_byte_off;

    if (lane == 0) {
        const __half d_h = __float2half(d);
        // Q4_0 block lays out FP16 scale at bytes 0..1, packed nibbles at 2..17.
        out[0] = ((const uint8_t *)&d_h)[0];
        out[1] = ((const uint8_t *)&d_h)[1];
    }
    if (lane < 16) {
        out[2 + lane] = (uint8_t)((q_int & 0xF) | ((q_high & 0xF) << 4));
    }
}

} // namespace ll_kv_scatter

extern "C" void flush_k_residual_q4_dev_pos(
    const void  * residual,         // [n_kv * head_dim * 32] F16
    void        * k_blocks,         // base of Q4_0 slab
    const void  * pos_dev,          // device ptr to i32 (1 element)
    int n_kv,
    int head_dim,
    int max_seq_blocks,
    cudaStream_t stream
) {
    const int blocks_per_window = n_kv * head_dim;
    dim3 grid(blocks_per_window, 1, 1);
    dim3 block(32, 1, 1);
    ll_kv_scatter::flush_k_residual_q4_dev_pos_kernel<<<grid, block, 0, stream>>>(
        (const __half *)residual,
        (uint8_t *)k_blocks,
        (const int32_t *)pos_dev,
        blocks_per_window,
        max_seq_blocks
    );
}

// Device-position byte scatter for Q4_0 V append.
//   `token_bytes` MUST be a multiple of 4 (Q4_0 block size is 18, n_kv *
//   blocks_per_head * 18: for any n_kv that's even, this is a multiple
//   of 4 in practice; assert it).
extern "C" void q4_v_scatter_bytes_dev_pos(
    const void  * src,         // [token_bytes] u8
    void        * dst,         // [max_seq * token_bytes] u8
    const void  * pos_dev,     // device ptr to i32 (1 element)
    int token_bytes,
    cudaStream_t stream
) {
    const int token_words = token_bytes / 4;
    const int block = 128;
    const int grid  = (token_words + block - 1) / block;
    ll_kv_scatter::q4_v_scatter_bytes_kernel<<<grid, block, 0, stream>>>(
        (const uchar4 *)src,
        (uchar4 *)dst,
        (const int32_t *)pos_dev,
        token_words
    );
}
