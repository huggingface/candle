#include "cuda_utils.cuh"

extern "C" __global__ void affine_f32( 
    const size_t numel, 
    const size_t num_dims,
    const size_t *info,
    const float *x,
    float *y,
    const float mul,
    const float add
) { 
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    // This is likely to be very very slow, we should either optimize the contiguous case
    // as a separate kernel, proceed by block, improve the stride computations (and probably
    // do all of these).
    unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
    y[strided_i] = x[i] * mul + add;
}

extern "C" __global__ void affine_f64( 
    const size_t numel, 
    const size_t num_dims,
    const size_t *info,
    const double *x,
    double *y,
    const double mul,
    const double add
) { 
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
    y[strided_i] = x[i] * mul + add;
}
