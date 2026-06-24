#undef __CUDA_FP8_TYPES_EXIST__
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/**
 * @brief Counts the number of tokens assigned to each expert.
 *
 * @param expert_ids     Device pointer to the sorted expert IDs [size_m].
 * @param expert_counts  Device pointer to the output counts [num_experts]
 * (must be pre-initialized to zero).
 * @param size_m         Total number of tokens.
 */
static __global__ void count_tokens_per_expert_kernel(
    const int32_t* expert_ids, 
    int32_t* expert_counts, 
    int size_m) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        int32_t expert_id = expert_ids[i];
        // expert_id is from a sorted list, so we assume it's valid
        // (i.e., 0 <= expert_id < num_experts)
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

/**
 * @brief Calculates expert offsets array on the GPU.
 *
 * @param d_expert_ids     Device pointer to sorted expert IDs [size_m].
 * @param size_m           Total number of tokens.
 * @param d_expert_offsets Device pointer for output offsets [num_experts + 1].
 * @param num_experts      Number of experts.
 * @param stream           CUDA stream.
 */
static void calculate_expert_offsets(
    const int32_t* d_expert_ids,
    int size_m,
    int32_t* d_expert_counts,
    int32_t* d_expert_offsets,
    int num_experts,
    cudaStream_t stream
) {
    // 1. Zero-initialize the counts buffer
    cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);

    // 2. Launch kernel to count tokens per expert
    int threads = 256;
    int blocks = (size_m + threads - 1) / threads;
    count_tokens_per_expert_kernel<<<blocks, threads, 0, stream>>>(
        d_expert_ids, d_expert_counts, size_m
    );

    // 3. Perform prefix sum (scan)
    // We will use inclusive_scan on [counts] and store results in [offsets + 1]
    // This is a common and efficient pattern.

    // Wrap raw pointers for Thrust
    thrust::device_ptr<const int32_t> d_counts_ptr(d_expert_counts);
    thrust::device_ptr<int32_t> d_offsets_ptr(d_expert_offsets);

    // Run inclusive scan.
    // Input:  [c0, c1, c2, ...] (size num_experts)
    // Output: [c0, c0+c1, c0+c1+c2, ...] (stored at offsets[1])
    thrust::inclusive_scan(
        thrust::cuda::par.on(stream), // Execute on the specified stream
        d_counts_ptr,                 // Input start
        d_counts_ptr + num_experts,   // Input end
        d_offsets_ptr + 1             // Output start (shifted by 1)
    );

    // 4. Set the first offset (offsets[0]) to 0
    // This completes the exclusive scan.
    cudaMemsetAsync(d_expert_offsets, 0, sizeof(int32_t), stream);
}


// This performs an EXCLUSIVE scan: [c0, c1] -> [0, c0, c0+c1]
// Assumptions: num_experts <= 1024 (fits in one block)
static __global__ void expert_prefix_sum_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_experts
) {
    // Use shared memory for fast scanning
    // Size needs to be enough for num_experts
    extern __shared__ int32_t temp_storage[];

    int tid = threadIdx.x;

    // We pad with 0 if tid >= num_experts
    int val = (tid < num_experts) ? counts[tid] : 0;
    temp_storage[tid] = val;
    
    __syncthreads();

    // Hillis-Steele Parallel Scan (Inclusive in shared mem)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp_val = 0;
        if (tid >= offset) {
            temp_val = temp_storage[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp_storage[tid] += temp_val;
        }
        __syncthreads();
    }

    // The result at temp_storage[i] is the inclusive sum of counts[0..i]
    // We want offsets[i] = inclusive_sum[i-1]
    // We want offsets[0] = 0
    
    if (tid < num_experts) {
        // Shift right: Offset[i+1] gets the inclusive sum up to i
        offsets[tid + 1] = temp_storage[tid];
        
        // Handle the first element separately
        if (tid == 0) {
            offsets[0] = 0;
        }
    }
}

static void calculate_expert_offsets_light(
    const int32_t* d_expert_ids,
    int size_m,
    int32_t* d_expert_counts,
    int32_t* d_expert_offsets,
    int num_experts,
    cudaStream_t stream
) {
    cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);

    int threads = 256;
    int blocks = (size_m + threads - 1) / threads;
    count_tokens_per_expert_kernel<<<blocks, threads, 0, stream>>>(
        d_expert_ids, d_expert_counts, size_m
    );

    // We launch exactly one block with 'num_experts' threads (or next power of 2)
    // We need shared memory size = threads * sizeof(int32_t)
    int scan_threads = num_experts;
    
    // Round up scan_threads to next power of 2 if needed, 
    // or just use a fixed size like 1024 if num_experts is small enough.
    if (scan_threads < 32) scan_threads = 32;
    else if (scan_threads > 1024) {
        // Error: This custom kernel only supports up to 1024 experts
        // Handle error or assert here
    }

    size_t smem_size = scan_threads * sizeof(int32_t);

    expert_prefix_sum_kernel<<<1, scan_threads, smem_size, stream>>>(
        d_expert_counts, 
        d_expert_offsets, 
        num_experts
    );
}

namespace vllm_rs {

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

inline __device__ void from_float(half& dst, float src) {
  dst = static_cast<half>(float_to_half(src));
}

inline __device__ void from_float(__nv_bfloat16& dst, float src) {
  dst = __float2bfloat16(src);
}

}