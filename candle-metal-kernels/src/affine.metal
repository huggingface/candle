#include <metal_stdlib>
using namespace metal;

METAL_FUNC bool is_contiguous(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    size_t acc = 1;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        if (acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

kernel void affine(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,

    device float *inp [[buffer(4)]],
    device float *out [[buffer(5)]],

    constant float &mul,
    constant float &add,
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]]
) {
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size;
    const size_t start = thread_index * length;
    const size_t stop = min(start + length, dim);
    if (is_contiguous(num_dims, dims, strides)) {
        for (size_t i = start; i < stop; i++) {
            float x = inp ? inp[i] : out[i];
            out[i] = x * mul + add;
        }
    } else {
        for (size_t i = start; i < stop; i++) {
            uint strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = inp ? inp[strided_i] : out[strided_i];
            out[strided_i] = x * mul + add;
        }
    }
}
