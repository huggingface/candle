#pragma once

#include "flash.h"

#define PREC_SWITCH(prec_type, ELEMENT, ...)             \
    do {                                                  \
        if (prec_type == 1) {                             \
            using ELEMENT = cutlass::half_t;              \
            __VA_ARGS__();                                \
        } else if (prec_type == 2) {                      \
            using ELEMENT = cutlass::bfloat16_t;          \
            __VA_ARGS__();                                \
        } else {                                          \
            throw std::runtime_error(                     \
                "Unsupported precision type for FlashAttention-4: " \
                + std::to_string(prec_type));             \
        }                                                 \
    } while (0)

#define HEADDIM_SWITCH(d, KHEADSIZE, ...)                 \
    do {                                                  \
        if (d == 64) {                                    \
            constexpr int KHEADSIZE = 64;                 \
            __VA_ARGS__();                                \
        } else if (d == 128) {                            \
            constexpr int KHEADSIZE = 128;                \
            __VA_ARGS__();                                \
        } else if (d == 256) {                            \
            constexpr int KHEADSIZE = 256;                \
            __VA_ARGS__();                                \
        } else if (d == 512) {                            \
            constexpr int KHEADSIZE = 512;                \
            __VA_ARGS__();                                \
        } else {                                          \
            throw std::runtime_error(                     \
                "Unsupported head dimension for FlashAttention-4: " \
                + std::to_string(d));                     \
        }                                                 \
    } while (0)

#define QUERYHEAD_SWITCH(ratio, KBLOCKH, ...)             \
    do {                                                  \
        if (ratio == 2) {                                 \
            constexpr int KBLOCKH = 2;                    \
            __VA_ARGS__();                                \
        } else if (ratio == 4) {                          \
            constexpr int KBLOCKH = 4;                    \
            __VA_ARGS__();                                \
        } else if (ratio == 8) {                          \
            constexpr int KBLOCKH = 8;                    \
            __VA_ARGS__();                                \
        } else if (ratio == 16) {                         \
            constexpr int KBLOCKH = 16;                   \
            __VA_ARGS__();                                \
        } else if (ratio == 32) {                         \
            constexpr int KBLOCKH = 32;                   \
            __VA_ARGS__();                                \
        } else {                                          \
            throw std::runtime_error(                     \
                "Unsupported GQA ratio for FlashAttention-4: " \
                + std::to_string(ratio));                 \
        }                                                 \
    } while (0)

template<typename Element, int kHeadSize>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template<typename Element, int kHeadSize, int kBlockH>
void run_mha_fwd_gqa_(Flash_fwd_params &params, cudaStream_t stream);

template<typename Element, int kHeadSize>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream);
