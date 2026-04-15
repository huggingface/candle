// Self-contained common header for MMQ GGUF kernels.
// Replaces llama.cpp's common.cuh + ggml.h + ggml-common.h for standalone compilation.
#pragma once

#include <cstdint>
#include <cstdio>
#include <climits>

#include "cuda_fp16.h"
#include "cuda_bf16.h"

// ============================================================
// Basic macros
// ============================================================

#define WARP_SIZE 32
#define MATRIX_ROW_PADDING 512
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_MAX_DEVICES 16

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

// ============================================================
// ggml_type enum (matching llama.cpp values)
// ============================================================

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
};

// ============================================================
// Quantization constants
// ============================================================

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))

#define QK_MXFP4 32
#define QR_MXFP4 2
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
#define QR_NVFP4 2
#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))

#define QR2_K 4
#define QI2_K (QK_K / (4 * QR2_K))

#define QR3_K 4
#define QI3_K (QK_K / (4 * QR3_K))

#define QR4_K 2
#define QI4_K (QK_K / (4 * QR4_K))

#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))

#define QR6_K 2
#define QI6_K (QK_K / (4 * QR6_K))

// IQ constants (needed for template compilation even if not instantiated)
#define QR2_XXS 4
#define QI2_XXS (QK_K / (4 * QR2_XXS))
#define QR2_XS  4
#define QI2_XS  (QK_K / (4 * QR2_XS))
#define QR2_S   4
#define QI2_S   (QK_K / (4 * QR2_S))
#define QR3_XXS 4
#define QI3_XXS (QK_K / (4 * QR3_XXS))
#define QR3_S   4
#define QI3_S   (QK_K / (4 * QR3_S))
#define QR1_S   8
#define QI1_S   (QK_K / (4 * QR1_S))
#define QR1_M   8
#define QI1_M   (QK_K / (4 * QR1_M))
#define QK4_NL  32
#define QR4_NL  2
#define QI4_NL  (QK4_NL / (4 * QR4_NL))
#define QR4_XS  2
#define QI4_XS  (QK_K / (4 * QR4_XS))
#define QR3_XS  4
#define QI3_XS  (QK_K / (4 * QR3_XS))

// ============================================================
// Block type definitions (CUDA half/half2)
// ============================================================

typedef struct { half d; uint8_t qs[QK4_0 / 2]; } block_q4_0;
typedef struct { half2 dm; uint8_t qs[QK4_1 / 2]; } block_q4_1;
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;
typedef struct { uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; uint8_t qs[QK_NVFP4/2]; } block_nvfp4;
typedef struct { half d; uint8_t qh[4]; uint8_t qs[QK5_0 / 2]; } block_q5_0;
typedef struct { half2 dm; uint8_t qh[4]; uint8_t qs[QK5_1 / 2]; } block_q5_1;
typedef struct { half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { half2 ds; int8_t qs[QK8_1]; } block_q8_1;

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    half2 dm;
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    half d;
} block_q3_K;

typedef struct {
    half2 dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    half2 dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half d;
} block_q6_K;

// IQ block types (needed for template compilation)
typedef struct { half d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
typedef struct { half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;
typedef struct { half d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
#define IQ3S_N_SCALE QK_K/64
typedef struct { half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[IQ3S_N_SCALE]; } block_iq3_s;
typedef struct { half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
typedef struct { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; } block_iq1_m;
typedef struct { half d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;
typedef struct { half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;

// ============================================================
// Architecture detection
// ============================================================

#define GGML_CUDA_CC_PASCAL       600
#define GGML_CUDA_CC_DP4A         610
#define GGML_CUDA_CC_VOLTA        700
#define GGML_CUDA_CC_TURING       750
#define GGML_CUDA_CC_AMPERE       800
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_BLACKWELL    1200
#define GGML_CUDA_CC_DGX_SPARK    1210
#define GGML_CUDA_CC_RUBIN        1300

#define GGML_CUDA_CC_OFFSET_AMD      0x1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc) (cc < GGML_CUDA_CC_OFFSET_MTHREADS)
#define GGML_CUDA_CC_IS_AMD(cc)    (cc >= GGML_CUDA_CC_OFFSET_AMD)

// AMD CC constants (needed for compile-time checks even though we target NVIDIA)
#define GGML_CUDA_CC_CDNA1   (GGML_CUDA_CC_OFFSET_AMD + 0x908)
#define GGML_CUDA_CC_RDNA1   (GGML_CUDA_CC_OFFSET_AMD + 0x1010)
#define GGML_CUDA_CC_RDNA2   (GGML_CUDA_CC_OFFSET_AMD + 0x1030)
#define GGML_CUDA_CC_RDNA3   (GGML_CUDA_CC_OFFSET_AMD + 0x1100)
#define GGML_CUDA_CC_RDNA3_5 (GGML_CUDA_CC_OFFSET_AMD + 0x1150)
#define GGML_CUDA_CC_RDNA4   (GGML_CUDA_CC_OFFSET_AMD + 0x1200)
#define GGML_CUDA_CC_CDNA3   (GGML_CUDA_CC_OFFSET_AMD + 0x942)

#define GGML_CUDA_CC_IS_RDNA(cc)    (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc)   (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) (cc >= GGML_CUDA_CC_RDNA3 && cc < GGML_CUDA_CC_RDNA3_5)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
#define GGML_CUDA_CC_IS_RDNA4(cc)   (cc >= GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_CDNA(cc)    (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_CDNA3(cc)   (cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_RDNA1)

// Compile-time architecture detection
#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) { return false; }

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
    if (cur == 0) return -1;
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// FP16 availability
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif

// MMA (tensor core) availability
#if __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#define BLACKWELL_MMA_AVAILABLE
#endif

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif

#if defined(TURING_MMA_AVAILABLE)
#define LDMATRIX_TRANS_AVAILABLE
#endif

// Host-side architecture query functions
static bool fp16_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA;
}

static bool amd_mfma_available(const int /*cc*/) { return false; } // NVIDIA only
static bool amd_wmma_available(const int /*cc*/) { return false; } // NVIDIA only

static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static bool blackwell_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL &&
           ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_RUBIN;
}

// ============================================================
// Device helpers
// ============================================================

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 32; // NVIDIA only
}

// NO_DEVICE_CODE: called from unused template paths to satisfy compiler
[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {
    printf("%s:%d: ERROR: CUDA kernel %s has no device code for arch %d. Compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
    __trap();
    GGML_UNUSED(no_device_code);
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE
#endif

#ifdef __CUDA_ARCH__
#define GGML_ABORT(msg) do { printf("GGML_ABORT: %s\n", msg); __trap(); } while(0)
#define GGML_ASSERT(x)  do { if (!(x)) { printf("GGML_ASSERT failed: %s\n", #x); __trap(); } } while(0)
#else
#define GGML_ABORT(msg) do { fprintf(stderr, "GGML_ABORT: %s\n", msg); abort(); } while(0)
#define GGML_ASSERT(x)  do { if (!(x)) { fprintf(stderr, "GGML_ASSERT failed: %s\n", #x); abort(); } } while(0)
#endif

// dp4a intrinsic
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// Warp reductions
template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_max(int x) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_max_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = max(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
#endif
}

// CUDA_SET_SHARED_MEMORY_LIMIT
#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
    do { \
        static bool raised[GGML_CUDA_MAX_DEVICES] = {false}; \
        int dev; cudaGetDevice(&dev); \
        if (!raised[dev]) { \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes); \
            raised[dev] = true; \
        } \
    } while(0)

// ============================================================
// ggml_cuda_type_traits
// ============================================================

template <ggml_type type>
struct ggml_cuda_type_traits;

template<> struct ggml_cuda_type_traits<GGML_TYPE_F16>     { static constexpr int qk = 1;     static constexpr int qr = 1; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q4_0>    { static constexpr int qk = QK4_0;  static constexpr int qr = QR4_0;  static constexpr int qi = QI4_0; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q4_1>    { static constexpr int qk = QK4_1;  static constexpr int qr = QR4_1;  static constexpr int qi = QI4_1; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q5_0>    { static constexpr int qk = QK5_0;  static constexpr int qr = QR5_0;  static constexpr int qi = QI5_0; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q5_1>    { static constexpr int qk = QK5_1;  static constexpr int qr = QR5_1;  static constexpr int qi = QI5_1; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q8_0>    { static constexpr int qk = QK8_0;  static constexpr int qr = QR8_0;  static constexpr int qi = QI8_0; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q2_K>    { static constexpr int qk = QK_K;   static constexpr int qr = QR2_K;  static constexpr int qi = QI2_K; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q3_K>    { static constexpr int qk = QK_K;   static constexpr int qr = QR3_K;  static constexpr int qi = QI3_K; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q4_K>    { static constexpr int qk = QK_K;   static constexpr int qr = QR4_K;  static constexpr int qi = QI4_K; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q5_K>    { static constexpr int qk = QK_K;   static constexpr int qr = QR5_K;  static constexpr int qi = QI5_K; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_Q6_K>    { static constexpr int qk = QK_K;   static constexpr int qr = QR6_K;  static constexpr int qi = QI6_K; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_MXFP4>   { static constexpr int qk = QK_MXFP4; static constexpr int qr = QR_MXFP4; static constexpr int qi = QI_MXFP4; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_NVFP4>   { static constexpr int qk = QK_NVFP4; static constexpr int qr = QR_NVFP4; static constexpr int qi = QI_NVFP4; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> { static constexpr int qk = QK_K;   static constexpr int qr = QR2_XXS; static constexpr int qi = QI2_XXS; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS>  { static constexpr int qk = QK_K;   static constexpr int qr = QR2_XS;  static constexpr int qi = QI2_XS; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S>   { static constexpr int qk = QK_K;   static constexpr int qr = QR2_S;   static constexpr int qi = QI2_S; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> { static constexpr int qk = QK_K;   static constexpr int qr = QR3_XXS; static constexpr int qi = QI3_XXS; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S>   { static constexpr int qk = QK_K;   static constexpr int qr = QR3_S;   static constexpr int qi = QI3_S; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S>   { static constexpr int qk = QK_K;   static constexpr int qr = QR1_S;   static constexpr int qi = QI1_S; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL>  { static constexpr int qk = QK4_NL; static constexpr int qr = QR4_NL;  static constexpr int qi = QI4_NL; };
template<> struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS>  { static constexpr int qk = QK_K;   static constexpr int qr = QR4_XS;  static constexpr int qi = QI4_XS; };

// ============================================================
// Additional macros and helpers
// ============================================================

template<typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif
}

// Device memcpy helper for register<->SRAM copies
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "Alignment misuse in ggml_cuda_memcpy_1");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// E8M0/UE4M3 float conversion helpers (for MXFP4/NVFP4)
static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) { bits = 0x00400000; } else { bits = (uint32_t) x << 23; }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || (x == 0x7F && x != 0xFF)) { return 0.0f; }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) { raw = ldexpf((float) man, -9); } else { raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7); }
    return static_cast<float>(raw / 2);
}

// IQ/MXFP4 lookup table stubs (needed for compilation even though we only instantiate standard quant types)
// These are device constants from ggml-common.h. We provide minimal stubs.
// The functions referencing them are only called for IQ/MXFP4 types which we never instantiate.
static const __device__ int8_t  kvalues_mxfp4[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
static const __device__ int8_t  kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const __device__ uint64_t iq2xxs_grid[256] = {0};
static const __device__ uint64_t iq2xs_grid[512] = {0};
static const __device__ uint64_t iq2s_grid[1024] = {0};
static const __device__ uint32_t iq3xxs_grid[256] = {0};
static const __device__ uint32_t iq3s_grid[512] = {0};
static const __device__ uint32_t iq1s_grid_gpu[512] = {0};
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
typedef union { half f16; uint16_t u16; } iq1m_scale_t;

// ============================================================
// ggml_cuda_unroll helper (used by some kernels)
// ============================================================

template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};
