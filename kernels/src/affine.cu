extern "C" __global__ void affine_f32( 
    const size_t numel, 
    const float *x,
    float *y,
    const float mul,
    const float add
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    y[i] = x[i] * mul + add;
}

extern "C" __global__ void affine_f64( 
    const size_t numel, 
    const double *x,
    double *y,
    const double mul,
    const double add
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 
    y[i] = x[i] * mul + add;
}
