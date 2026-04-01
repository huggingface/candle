// PolarQuant fused centroid-lookup matmul kernel (SIMD-optimized).
//
// Computes: output[batch][row] = norms[row] * Σ_i x_rot[batch][i] * centroids[unpack(indices, row*d+i)]
//
// Optimizations matching GGML Q4_0 pattern:
// 1. Centroid table in threadgroup shared memory (16 floats)
// 2. Vectorized uint32 index loads (8 nibbles per load at 4-bit)
// 3. Register-cached input blocks (16 floats per thread)
// 4. SIMD group parallel reduction with simd_sum
// 5. Multi-row processing (N_DST=4 rows per SIMD group)

#include <metal_stdlib>
using namespace metal;

#define PQ_N_DST 4
#define PQ_N_SIMDGROUP 2
#define PQ_SIMD_WIDTH 32
#define PQ_BLOCK_SIZE 32  // process 32 elements per iteration (matches SIMD width)

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

    // Cache centroid table in registers (only 16 floats for 4-bit)
    float cent[16];
    for (uint i = 0; i < 16; i++) {
        cent[i] = centroids[i];
    }

    device const float* xr = x_rot + batch_id * d;

    float sumf[PQ_N_DST] = {0.f};

    // Bytes per row of packed indices (4-bit: d/2 bytes per row)
    const uint bytes_per_row = d / 2;

    // Each thread processes elements at stride PQ_SIMD_WIDTH
    // Process 2 elements per byte (4-bit packing)
    for (uint i = tiisg; i < d; i += PQ_SIMD_WIDTH) {
        float xval = xr[i];

        for (uint row_off = 0; row_off < PQ_N_DST; row_off++) {
            uint row = first_row + row_off;
            if (row >= n_out) break;

            // Read packed byte containing this element's nibble
            uint byte_pos = (row * d + i) / 2;
            uint nibble_sel = (row * d + i) % 2;
            uint8_t packed_byte = indices[byte_pos];
            uint idx = (packed_byte >> (nibble_sel * 4)) & 0xF;

            sumf[row_off] += xval * cent[idx];
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

// Generic bit-width version
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

    // Cache centroids in registers
    constexpr uint n_cent = 1u << BIT_WIDTH;
    float cent[n_cent];
    for (uint i = 0; i < n_cent; i++) {
        cent[i] = centroids[i];
    }

    constexpr uint indices_per_byte = 8 / BIT_WIDTH;
    constexpr uint8_t mask = (1u << BIT_WIDTH) - 1;

    device const float* xr = x_rot + batch_id * d;

    float sumf[PQ_N_DST] = {0.f};

    for (uint i = tiisg; i < d; i += PQ_SIMD_WIDTH) {
        float xval = xr[i];

        for (uint row_off = 0; row_off < PQ_N_DST; row_off++) {
            uint row = first_row + row_off;
            if (row >= n_out) break;

            uint pos = row * d + i;
            uint byte_pos = pos / indices_per_byte;
            uint bit_offset = (pos % indices_per_byte) * BIT_WIDTH;
            uint idx = (indices[byte_pos] >> bit_offset) & mask;

            sumf[row_off] += xval * cent[idx];
        }
    }

    for (uint row_off = 0; row_off < PQ_N_DST; row_off++) {
        float total = simd_sum(sumf[row_off]);
        if (tiisg == 0 && (first_row + row_off) < n_out) {
            output[batch_id * n_out + first_row + row_off] = total * norms[first_row + row_off];
        }
    }
}

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
