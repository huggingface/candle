#include <metal_stdlib>
using namespace metal;

// Utils
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

#define MAX(x, y) ((x) > (y) ? (x) : (y))

template<typename T>
constexpr int work_per_thread() {
    constexpr int wpt = 8 / sizeof(T);
    return MAX(1, wpt);
}

// Kernels
template <typename T, int W = work_per_thread<T>()>
[[kernel]] void affine_kernel(
    constant size_t &dim,
    constant float &mul,
    constant float &add,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    tid *= W;
    if (W > 1 && tid + W > dim) {
        for (int i = 0; tid + i < dim; ++i) {
            float result = fma(float(input[tid + i]), mul, add);
            output[tid + i] = static_cast<T>(result);
        }
    } else {
        for (int i = 0; i < W; ++i) {
            float result = fma(float(input[tid + i]), mul, add);
            output[tid + i] = static_cast<T>(result);
        }
    }
}

template <typename T>
[[kernel]] void affine_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant float &add,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    float result = fma(float(input[idx]), mul, add);
    output[tid] = static_cast<T>(result);
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_affine(tname, t)                                           \
    init_kernel("affine_" #tname, affine_kernel, t)                     \
    init_kernel("affine_" #tname "_strided", affine_kernel_strided, t)

#define POWF(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    output[id] = TYPENAME(pow(input[id], TYPENAME(mul))); \
} \
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    output[id] = TYPENAME(pow(input[get_strided_index(id, num_dims, dims, strides)], TYPENAME(mul))); \
}

#define ELU(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    const TYPENAME x = input[id]; \
    output[id] = TYPENAME((x > 0)?x: mul * (exp(x) - 1)); \
} \
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    const TYPENAME x = input[get_strided_index(id, num_dims, dims, strides)]; \
    output[id] = TYPENAME((x > 0)?x: mul * (exp(x) - 1)); \
} \


init_affine(u8, uint8_t);
init_affine(u32, uint32_t);
init_affine(i64, int64_t);
init_affine(f32, float);
init_affine(f16, half);

POWF(powf_f32, float)
POWF(powf_f16, half)
ELU(elu_f32, float)
ELU(elu_f16, half)


#if defined(__HAVE_BFLOAT__)
init_affine(bf16, bfloat);
POWF(powf_bf16, bfloat);
ELU(elu_bf16, bfloat);
#endif
