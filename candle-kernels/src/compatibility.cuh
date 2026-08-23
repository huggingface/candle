#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_fp8.h"

// Table showing which features are supported on which compute capability
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

// FIXME: the minimum compute capabilities are just guesses since the table is not specific enough

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

#if __CUDA_ARCH__ < 700
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
// The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
// Solution adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh#L96-L119
//__device__ __half atomicAdd(__half *address, __half val) {
   //  unsigned int *address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
   //  unsigned int old = *address_as_ui;
   //  unsigned int assumed;
   //  bool unaligned = (size_t) address & 2;
   //  do {
   //      assumed = old;
   //      unsigned int hsum;
   //      hsum = unaligned ? (old >> 16) : (old & 0xffff);
   //      hsum = __half_as_ushort(__ushort_as_half(hsum) + val); 
   //      old = atomicCAS(address_as_ui, assumed,
   //          unaligned ? (old & 0xffff) | (hsum << 16) : (old & 0xffff0000) | hsum
   //      );

   // } while (assumed != old);
   // return __ushort_as_half(unaligned ? (old >> 16) : (old & 0xffff));
//}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
__device__ __forceinline__ __half atomicAdd(__half* address, __half val) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    const bool upper = ((size_t)address & 2) != 0;

    do {
        assumed = old;

        const unsigned short old_bits =
            upper ? (unsigned short)(old >> 16)
                  : (unsigned short)(old & 0xffff);

        const __half current = __ushort_as_half(old_bits);

        const __half sum = __float2half(
            __half2float(current) + __half2float(val)
        );

        const unsigned int sum_bits =
            (unsigned int)__half_as_ushort(sum);

        const unsigned int updated =
            upper
                ? ((old & 0x0000ffffu) | (sum_bits << 16))
                : ((old & 0xffff0000u) | sum_bits);

        old = atomicCAS(address_as_ui, assumed, updated);
    } while (old != assumed);

    const unsigned short result_bits =
        upper ? (unsigned short)(old >> 16)
              : (unsigned short)(old & 0xffff);

    return __ushort_as_half(result_bits);
}
#endif


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 && defined(CANDLE_CUDA_BF16_FALLBACK)
__device__ __forceinline__ __nv_bfloat16 atomicAdd(
    __nv_bfloat16* address,
    __nv_bfloat16 val
) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));

    unsigned int old = *address_as_ui;
    unsigned int assumed;
    const bool upper = ((size_t)address & 2) != 0;

    do {
        assumed = old;

        const unsigned short old_bits =
            upper
                ? (unsigned short)(old >> 16)
                : (unsigned short)(old & 0xffff);

        const __nv_bfloat16 current =
            __ushort_as_bfloat16(old_bits);

        const __nv_bfloat16 sum =
            __float2bfloat16(
                __bfloat162float(current) +
                __bfloat162float(val)
            );

        const unsigned int sum_bits =
            (unsigned int)__bfloat16_as_ushort(sum);

        const unsigned int updated =
            upper
                ? ((old & 0x0000ffffu) | (sum_bits << 16))
                : ((old & 0xffff0000u) | sum_bits);

        old = atomicCAS(address_as_ui, assumed, updated);
    } while (old != assumed);

    const unsigned short result_bits =
        upper
            ? (unsigned short)(old >> 16)
            : (unsigned short)(old & 0xffff);

    return __ushort_as_bfloat16(result_bits);
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
