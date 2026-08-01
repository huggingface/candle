// Block reductions for `ssa::Instr::ReduceLocal`, emitted ahead of the kernel
// whenever it contains one. The HIP counterpart of `ug-cuda`'s `reduce.cu`.
//
// The shuffles are HIP's unsuffixed `__shfl_xor`: HIP does declare
// `__shfl_xor_sync`, but its mask is a 64-bit lane mask and it static_asserts
// when handed CUDA's 32-bit 0xffffffff. The unsuffixed form has whole-wavefront
// semantics, which is what a full mask means anyway.
//
// The explicit width of 32 is what keeps the two-stage block reduction correct
// on wave64 (CDNA) as well as wave32 (RDNA): HIP honours it as a sub-wavefront
// shuffle, so a "warp" is 32 lanes on both and the `smem[32]` staging array is
// indexed consistently with it.

template <typename T>
static __device__ __forceinline__ T warp_reduce_sum(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask, 32);
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T warp_reduce_max(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = max(x, __shfl_xor(x, mask, 32));
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T warp_reduce_min(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = min(x, __shfl_xor(x, mask, 32));
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T block_reduce_sum(T x) {
    const int block_size = blockDim.x;
    x = warp_reduce_sum(x);
    if (block_size > 32) {
        __shared__ T smem[32];
        unsigned int warp_id = threadIdx.x / 32;
        unsigned int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            smem[warp_id] = x;
        }
        __syncthreads();
        x = smem[lane_id];
        x = warp_reduce_sum(x);
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T block_reduce_max(T x) {
    const int block_size = blockDim.x;
    x = warp_reduce_max(x);
    if (block_size > 32) {
        __shared__ T smem[32];
        unsigned int warp_id = threadIdx.x / 32;
        unsigned int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            smem[warp_id] = x;
        }
        __syncthreads();
        x = smem[lane_id];
        x = warp_reduce_max(x);
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T block_reduce_min(T x) {
    const int block_size = blockDim.x;
    x = warp_reduce_min(x);
    if (block_size > 32) {
        __shared__ T smem[32];
        unsigned int warp_id = threadIdx.x / 32;
        unsigned int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            smem[warp_id] = x;
        }
        __syncthreads();
        x = smem[lane_id];
        x = warp_reduce_min(x);
    }
    return x;
}
