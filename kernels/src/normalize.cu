template<typename T>
__device__ void normalize(
    const size_t numel, 
    T *lhs, 
    const size_t size, 
    const T epsilon
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 

    const size_t offset = i * size;

    float sum = 0.0;
    for (int i=0; i < size; i ++){
	sum += lhs[offset + i];
    }
    sum /= size;
    for (int i=0; i < size; i ++){
	lhs[offset + i] -= sum;
    }

    float var = 0.0;
    for (int i=0; i < size; i ++){
	var += lhs[offset + i] * lhs[offset + i];
    }
    var /= size;
    var += epsilon;
    const float std = sqrt(var);
    for (int i=0; i < size; i ++){
	lhs[offset + i] /= std;
    }
} 

extern "C" __global__ void normalize_16( const size_t numel, __half *lhs, const size_t size, const __half epsilon) { normalize (numel, lhs, size, epsilon);}
extern "C" __global__ void normalize_f32( const size_t numel, float *lhs, const size_t size, const float epsilon) { normalize (numel, lhs, size, epsilon);}
extern "C" __global__ void normalize_f64( const size_t numel, double *lhs, const size_t size, const double epsilon) { normalize (numel, lhs, size, epsilon);}
