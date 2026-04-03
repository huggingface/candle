#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Table showing which features are supported on which compute capability
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

// FIXME: the minimum compute capabilities are just guesses since the table is not specific enough

// Vectorized Memory Access Traits for maximum bandwidth utilization.
// VecType<T> must occupy exactly VecConfig<T>::size * sizeof(T) bytes because
// reduce.cu reinterprets the loaded vector as a contiguous T array.
template<typename T> struct VecType { typedef T Type; };
template<> struct VecType<float> { typedef float4 Type; };
template<> struct VecType<double> { typedef double2 Type; };
template<> struct VecType<__half> { typedef float4 Type; };
template<> struct VecType<__nv_bfloat16> { typedef float4 Type; };
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890) || ALLOW_LEGACY_FP8
template<> struct VecType<__nv_fp8_e4m3> { typedef int4 Type; };
#endif

// Vectorization configuration constants
template<typename T> struct VecConfig {
    static constexpr int size = 1;      // Elements per vector
    static constexpr bool supported = false;
};

// 128-bit vectorization (float4 is 4x32 bits = 16 bytes)
template<> struct VecConfig<__half> {
    static constexpr int size = 8;
    static constexpr bool supported = true;
};

template<> struct VecConfig<__nv_bfloat16> {
    static constexpr int size = 8;
    static constexpr bool supported = true;
};

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890) || ALLOW_LEGACY_FP8
// 128-bit int4 = 4x32 bits = 16 bytes. Holds 16x 8-bit fp8 values
template<> struct VecConfig<__nv_fp8_e4m3> {
    static constexpr int size = 16;
    static constexpr bool supported = true;
};
#endif

template<> struct VecConfig<float> {
    static constexpr int size = 4;
    static constexpr bool supported = true;
};

template<> struct VecConfig<double> {
    static constexpr int size = 2; // double2 is 128-bit
    static constexpr bool supported = true;
};

template <typename T>
inline constexpr bool vec_layout_matches_v =
    !VecConfig<T>::supported ||
    sizeof(typename VecType<T>::Type) == VecConfig<T>::size * sizeof(T);

static_assert(vec_layout_matches_v<float>);
static_assert(vec_layout_matches_v<double>);
static_assert(vec_layout_matches_v<__half>);
static_assert(vec_layout_matches_v<__nv_bfloat16>);

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890) || ALLOW_LEGACY_FP8
static_assert(vec_layout_matches_v<__nv_fp8_e4m3>);
#endif

#if (__CUDACC_VER_MAJOR__ < 12 || __CUDACC_VER_MINOR__ < 2) && __CUDA_ARCH__ < 750
__device__ __forceinline__ __half __hmax_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}
__device__ __forceinline__ __half __hmin_nan(__half a, __half b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}
#endif
#if (__CUDACC_VER_MAJOR__ < 12 || __CUDACC_VER_MINOR__ < 2) && __CUDA_ARCH__ < 800
__device__ __forceinline__ __nv_bfloat16 __hmax_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}
__device__ __forceinline__ __nv_bfloat16 __hmin_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}
#endif

#if __CUDA_ARCH__ < 600
// Copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700 && (HAS_F16_ARITHMETIC || ALLOW_LEGACY_FP16)
__device__ __forceinline__ __half atomicAdd(__half* address, __half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t)address & 2;

    do {
        assumed = old;
        unsigned int hsum = unaligned ? (old >> 16) : (old & 0xffff);
        __half cur = __ushort_as_half(hsum);

#if HAS_F16_ARITHMETIC
        __half sum = __hadd(cur, val);
#else
        __half sum = __float2half(__half2float(cur) + __half2float(val));
#endif

        hsum = __half_as_ushort(sum);
        old = atomicCAS(
            address_as_ui,
            assumed,
            unaligned ? ((old & 0xffff) | (hsum << 16))
                      : ((old & 0xffff0000) | hsum)
        );
    } while (assumed != old);

    return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
}
#endif

// Polyfill: atomicAdd for bfloat16
// Native atomicAdd(__nv_bfloat16*, __nv_bfloat16) is only available on SM 8.0+ (Ampere).
// For older architectures with ALLOW_LEGACY_BF16, we emulate it using 32-bit CAS operations.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 && ALLOW_LEGACY_BF16
__device__ __forceinline__ __nv_bfloat16 atomicAdd(__nv_bfloat16* address, __nv_bfloat16 val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t)address & 2;

    do {
        assumed = old;
        unsigned int hsum = unaligned ? (old >> 16) : (old & 0xffff);
        __nv_bfloat16 cur = __ushort_as_bfloat16(hsum);

#if HAS_F16_ARITHMETIC
        __half cur_f16 = __float2half(__bfloat162float(cur));
        __half val_f16 = __float2half(__bfloat162float(val));
        __half sum_f16 = __hadd(cur_f16, val_f16);
        __nv_bfloat16 sum = __float2bfloat16(__half2float(sum_f16));
#else
        __nv_bfloat16 sum = __float2bfloat16(__bfloat162float(cur) + __bfloat162float(val));
#endif

        hsum = __bfloat16_as_ushort(sum);
        old = atomicCAS(
            address_as_ui,
            assumed,
            unaligned ? ((old & 0xffff) | (hsum << 16))
                      : ((old & 0xffff0000) | hsum)
        );
    } while (assumed != old);

    return __ushort_as_bfloat16(unaligned ? (old >> 16) : (old & 0xffff));
}
#endif

__device__ __forceinline__ __half atomicMaxf(__half* address, __half val) {
#if __CUDA_ARCH__ < 700
    // On older GPUs we do not have access to atomicCAS for shorts, so we have to do some trickery.
    // Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
    unsigned int *address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t) address & 2;
    do {
        assumed = old;
        unsigned int hmax;
        hmax = unaligned ? (old >> 16) : (old & 0xffff);
        hmax = __half_as_ushort(__hmax_nan(val, __ushort_as_half(hmax))); 
        old = atomicCAS(address_as_ui, assumed,
            unaligned ? (old & 0xffff) | (hmax << 16) : (old & 0xffff0000) | hmax
        );

    } while (assumed != old);
    return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
#else
    // Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmax_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
#endif
}

// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));        
    } else {
        return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMaxf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMin((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMax((long long int *)addr, __double_as_longlong(value)));
    }
}


__device__ __forceinline__ __half atomicMinf(__half* address, __half val) {
#if __CUDA_ARCH__ < 700
    // On older GPUs we do not have access to atomicCAS for shorts, so we have to do some trickery.
    // Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
    unsigned int *address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    bool unaligned = (size_t) address & 2;
    do {
        assumed = old;
        unsigned int hmin;
        hmin = unaligned ? (old >> 16) : (old & 0xffff);
        hmin = __half_as_ushort(__hmin_nan(val, __ushort_as_half(hmin))); 
        old = atomicCAS(address_as_ui, assumed,
            unaligned ? (old & 0xffff) | (hmin << 16) : (old & 0xffff0000) | hmin
        );

    } while (assumed != old);
    return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
#else
    // Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmin_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
#endif
}

// atomicMin is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMinf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMax((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMin((long long int *)addr, __double_as_longlong(value)));
    }
}
