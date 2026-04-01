// PolarQuant fused centroid-lookup matmul kernel (SIMD-optimized).
//
// Computes: output[batch][row] = norms[row] * Σ_i x_rot[batch][i] * centroids[unpack(indices, row*d+i)]
//
// Uses SIMD groups for parallel reduction, matching GGML's mul_vec_q_n pattern.
// Each SIMD group (32 threads) handles N_DST=4 output rows.
// Each thread accumulates a partial dot product over d/32 elements, then simd_sum reduces.

#include <metal_stdlib>
using namespace metal;

#define PQ_N_DST 4         // rows per SIMD group
#define PQ_N_SIMDGROUP 2   // SIMD groups per threadgroup
#define PQ_SIMD_WIDTH 32

inline uint pq_unpack_4bit(device const uint8_t* packed, uint pos) {
    uint byte_pos = pos / 2;
    uint nibble = pos % 2;
    return (packed[byte_pos] >> (nibble * 4)) & 0xF;
}

kernel void polarquant_mv_f32_4bit(
    device const float*   x_rot     [[buffer(0)]],
    device const uint8_t* indices   [[buffer(1)]],
    device const float*   norms     [[buffer(2)]],
    device const float*   centroids [[buffer(3)]],
    device       float*   output    [[buffer(4)]],
    constant     uint&    d         [[buffer(5)]],
    constant     uint&    n_out     [[buffer(6)]],
    constant     uint&    batch     [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    uint   tiisg [[thread_index_in_simdgroup]],
    uint   sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row_group = tgpig.x;
    const uint batch_id  = tgpig.y;

    const uint first_row = (row_group * PQ_N_SIMDGROUP + sgitg) * PQ_N_DST;
    if (first_row >= n_out) return;

    device const float* xr = x_rot + batch_id * d;

    // Each thread accumulates partial sums for PQ_N_DST rows
    float sumf[PQ_N_DST] = {0.f};

    // Stride: each thread handles elements tiisg, tiisg+32, tiisg+64, ...
    for (uint i = tiisg; i < d; i += PQ_SIMD_WIDTH) {
        float xval = xr[i];
        for (uint row_off = 0; row_off < PQ_N_DST && (first_row + row_off) < n_out; row_off++) {
            uint row = first_row + row_off;
            uint idx = pq_unpack_4bit(indices, row * d + i);
            sumf[row_off] += xval * centroids[idx];
        }
    }

    // SIMD reduction
    for (uint row_off = 0; row_off < PQ_N_DST; row_off++) {
        float total = simd_sum(sumf[row_off]);
        if (tiisg == 0 && (first_row + row_off) < n_out) {
            output[batch_id * n_out + first_row + row_off] = total * norms[first_row + row_off];
        }
    }
}

// Generic bit-width version (for non-4-bit)
inline uint pq_unpack_generic(device const uint8_t* packed, uint pos, uint bit_width) {
    uint indices_per_byte = 8 / bit_width;
    uint byte_pos = pos / indices_per_byte;
    uint bit_offset = (pos % indices_per_byte) * bit_width;
    uint8_t mask = (1u << bit_width) - 1;
    return (packed[byte_pos] >> bit_offset) & mask;
}

template<uint BIT_WIDTH>
kernel void polarquant_mv_f32_generic(
    device const float*   x_rot     [[buffer(0)]],
    device const uint8_t* indices   [[buffer(1)]],
    device const float*   norms     [[buffer(2)]],
    device const float*   centroids [[buffer(3)]],
    device       float*   output    [[buffer(4)]],
    constant     uint&    d         [[buffer(5)]],
    constant     uint&    n_out     [[buffer(6)]],
    constant     uint&    batch     [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    uint   tiisg [[thread_index_in_simdgroup]],
    uint   sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row_group = tgpig.x;
    const uint batch_id  = tgpig.y;

    const uint first_row = (row_group * PQ_N_SIMDGROUP + sgitg) * PQ_N_DST;
    if (first_row >= n_out) return;

    device const float* xr = x_rot + batch_id * d;

    float sumf[PQ_N_DST] = {0.f};

    for (uint i = tiisg; i < d; i += PQ_SIMD_WIDTH) {
        float xval = xr[i];
        for (uint row_off = 0; row_off < PQ_N_DST && (first_row + row_off) < n_out; row_off++) {
            uint row = first_row + row_off;
            uint idx = pq_unpack_generic(indices, row * d + i, BIT_WIDTH);
            sumf[row_off] += xval * centroids[idx];
        }
    }

    for (uint row_off = 0; row_off < PQ_N_DST; row_off++) {
        float total = simd_sum(sumf[row_off]);
        if (tiisg == 0 && (first_row + row_off) < n_out) {
            output[batch_id * n_out + first_row + row_off] = total * norms[first_row + row_off];
        }
    }
}

// Instantiate for each bit width
template [[host_name("polarquant_mv_f32_1bit")]]
kernel void polarquant_mv_f32_generic<1>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint);

template [[host_name("polarquant_mv_f32_2bit")]]
kernel void polarquant_mv_f32_generic<2>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint);

template [[host_name("polarquant_mv_f32_3bit")]]
kernel void polarquant_mv_f32_generic<3>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint);

template [[host_name("polarquant_mv_f32_5bit")]]
kernel void polarquant_mv_f32_generic<5>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint);

template [[host_name("polarquant_mv_f32_6bit")]]
kernel void polarquant_mv_f32_generic<6>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint);

template [[host_name("polarquant_mv_f32_7bit")]]
kernel void polarquant_mv_f32_generic<7>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint3, uint, uint);
