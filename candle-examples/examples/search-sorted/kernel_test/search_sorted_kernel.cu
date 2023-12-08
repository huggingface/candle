#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
template <typename input_t>
__device__ int lower_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val)
{
    while (start < end)
    {
        const int64_t mid = start + ((end - start) >> 1);
        const input_t mid_val = data_ss[mid];
        if (!(mid_val >= val))
        {
            start = mid + 1;
        }
        else
        {
            end = mid;
        }
    }
    return start;
}

template <typename input_t>
__device__ int upper_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val)
{
    while (start < end)
    {
        const int mid = start + ((end - start) >> 1);
        const input_t mid_val = data_ss[mid];
        if (!(mid_val > val))
        {
            start = mid + 1;
        }
        else
        {
            end = mid;
        }
    }
    return start;
}

template <typename input_t, typename output_t>
__global__ void searchsorted_cuda_kernel(
    output_t *data_out,
    const input_t *data_in,
    const input_t *data_bd,
    const u_int32_t idim_in,
    const u_int32_t idim_bd,
    const u_int32_t numel_in,
    const bool right,
    const bool is_1d_boundaries,
    const bool is_1d_values)
{
    // Define boundaries for sorted seq and values
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel_in; tid += blockDim.x * gridDim.x)
    {
        // If boundaries tensor is 1d, we always search the entire boundary tensor
        int start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
        int end_bd = start_bd + idim_bd;
        int val_idx = is_1d_values ? tid % idim_in : tid;

        // printf("Thread id: %d, start_bd: %d, end_bd: %d val_idx: %d, val: %f, idim_in: %d idim_bd: %d\n", tid, start_bd, end_bd, val_idx, data_in[val_idx], idim_in, idim_bd);
        int pos = !right ? lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[val_idx]) - start_bd : upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[val_idx]) - start_bd;

        data_out[tid] = pos;

        // data_out[tid] = *reinterpret_cast<output_t *>(&pos);
        // printf("tid %d, data_out[%d]: %d, pos: %d\n", tid, tid, data_out[tid], pos);

        // printf("Thread id: %d, start_bd: %d, end_bd: %d val_idx: %d, val: %f, pos: %d\n", tid, start_bd, end_bd, val_idx, data_in[val_idx], pos);
    }
}
// extern "C" __global__ void search_sorted_f32(
//     int64_t *data_out,
//     const float *data_in,
//     const float *data_bd,
//     const u_int32_t idim_in,
//     const u_int32_t idim_bd,
//     const u_int32_t numel_in,
//     const bool right,
//     const bool is_1d_boundaries,
//     const bool is_1d_values)
// {
//     searchsorted_cuda_kernel(data_out, data_in, data_bd, idim_in, idim_bd, numel_in, right, is_1d_boundaries, is_1d_values);
// }
// extern "C" __global__ void search_sorted_u32(
//     int64_t *data_out,
//     const u_int32_t *data_in,
//     const u_int32_t *data_bd,
//     const u_int32_t idim_in,
//     const u_int32_t idim_bd,
//     const u_int32_t numel_in,
//     const bool right,
//     const bool is_1d_boundaries,
//     const bool is_1d_values)
// {
//     searchsorted_cuda_kernel(data_out, data_in, data_bd, idim_in, idim_bd, numel_in, right, is_1d_boundaries, is_1d_values);
// }
// #define SEARCH_SORTED_KERNEL(TYPE, TYPE_STR)                                                                                     \
//     extern "C" __global__ void search_sorted_##TYPE_STR(                                                                         \
//         int64_t *data_out,                                                                                                       \
//         const TYPE *data_in,                                                                                                     \
//         const TYPE *data_bd,                                                                                                     \
//         const u_int32_t idim_in,                                                                                                 \
//         const u_int32_t idim_bd,                                                                                                 \
//         const u_int32_t numel_in,                                                                                                \
//         const bool right,                                                                                                        \
//         const bool is_1d_boundaries,                                                                                             \
//         const bool is_1d_values)                                                                                                 \
//     {                                                                                                                            \
//         if (threadIdx.x == 0)                                                                                                    \
//             printf("data_in: %f %f %f\n", data_in[0], data_in[1], data_in[2]);                                                   \
//         if (threadIdx.x == 0)                                                                                                    \
//             printf("data_bd: %f %f %f\n", data_bd[0], data_bd[1], data_bd[2]);                                                   \
//         searchsorted_cuda_kernel(data_out, data_in, data_bd, idim_in, idim_bd, numel_in, right, is_1d_boundaries, is_1d_values); \
//         int tid = threadIdx.x;                                                                                                   \
//         if (tid < numel_in)                                                                                                      \
//             printf("tid: %d data_out: %d", tid, data_out[tid]);                                                                  \
//     }

// SEARCH_SORTED_KERNEL(u_int8_t, u8);
// SEARCH_SORTED_KERNEL(u_int32_t, u32);
// SEARCH_SORTED_KERNEL(float, f32);
// SEARCH_SORTED_KERNEL(double, f64);
// SEARCH_SORTED_KERNEL(int64_t, i64);
// SEARCH_SORTED_KERNEL(__half, f16);
// SEARCH_SORTED_KERNEL(__nv_bfloat16, bf16);
