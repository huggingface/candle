/**
 * @file moe_q4k_imma_m8_build_chunks.cu
 *
 * Device-side chunk builder for the chunked IMMA M=8 variant. Replaces
 * the D2H-sync host pre-processing that regressed gemma4:26b prefill
 * by -47% (the host path serialized the CUDA stream).
 *
 * Walks expert_ids[0..size_m) sequentially, emitting one chunk metadata
 * entry every 8 same-expert pairs OR at any expert boundary. Single-
 * thread kernel — chunk count is small (~256 for typical prefill) and
 * the linear scan beats parallel-prefix overhead at this size.
 *
 * Output: chunk_pair_start[num_chunks], chunk_expert[num_chunks],
 *         num_chunks_out (single i32 scalar).
 *
 * Caller pre-allocates the metadata arrays sized to ceil(size_m/1) =
 * size_m (worst case all-different-experts).
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void moe_q4k_imma_m8_build_chunks_kernel(
    const int32_t * __restrict__ expert_ids,         // [size_m]
    int32_t * __restrict__ chunk_pair_start,         // [size_m] (capacity)
    int32_t * __restrict__ chunk_expert,             // [size_m]
    int32_t * __restrict__ num_chunks_out,           // [1]
    int size_m
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int n_chunks = 0;
    int i = 0;
    while (i < size_m) {
        const int e = expert_ids[i];
        // Find end of this expert's run.
        int j = i + 1;
        while (j < size_m && expert_ids[j] == e) ++j;
        // Emit chunks of up to 8 within [i, j).
        int k = i;
        while (k < j) {
            chunk_pair_start[n_chunks] = k;
            chunk_expert[n_chunks]     = e;
            ++n_chunks;
            k += 8;
        }
        i = j;
    }
    *num_chunks_out = n_chunks;
}

extern "C" void moe_q4k_imma_m8_build_chunks(
    const int32_t * expert_ids,
    int32_t * chunk_pair_start,
    int32_t * chunk_expert,
    int32_t * num_chunks_out,
    int size_m,
    cudaStream_t stream
) {
    if (size_m <= 0) return;
    moe_q4k_imma_m8_build_chunks_kernel<<<1, 1, 0, stream>>>(
        expert_ids, chunk_pair_start, chunk_expert, num_chunks_out, size_m
    );
}
