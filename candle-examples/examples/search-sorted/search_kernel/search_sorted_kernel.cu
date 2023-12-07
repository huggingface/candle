
template <typename input_t>
__device__ int lower_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val)
{
    // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
    // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
    // const int64_t orig_start = start;
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
    // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
    // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
    // const int orig_start = start;
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
    // if (threadIdx.x >= 3)
    //     printf("Thread id: %d, start: %d", threadIdx.x, start);
    return start;
}

template <typename input_t, typename output_t>
__global__ void searchsorted_cuda_kernel(
    output_t *data_out,
    const input_t *data_in,
    const input_t *data_bd,
    int idim_in,
    int idim_bd,
    int numel_in,
    bool right,
    bool is_1d_boundaries,
    bool is_1d_values)
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
    }
}

// template <>
// __global__ void searchsorted_cuda_kernel<float, int>(
//     int *data_out,
//     const float *data_in,
//     const float *data_bd,
//     int idim_in,
//     int idim_bd,
//     int numel_in,
//     bool right,
//     bool is_1d_boundaries,
//     bool is_1d_values);
