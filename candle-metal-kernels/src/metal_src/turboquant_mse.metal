// TurboQuant MSE fused centroid-lookup matmul kernel (SIMD-optimized).
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

#define PQ_N_DST 8
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

    // Determine how many valid rows we have
    const uint nr = min((uint)PQ_N_DST, n_out - first_row);

    // Each thread strides by SIMD_WIDTH, processes 2 elements per packed byte.
    // We step by 2 to process both nibbles of a byte together.
    const uint half_d = d / 2;
    for (uint pair = tiisg; pair < half_d; pair += PQ_SIMD_WIDTH) {
        uint i0 = pair * 2;
        uint i1 = i0 + 1;

        // Pre-cache input pair in registers
        float xval0 = xr[i0];
        float xval1 = xr[i1];

        for (uint row_off = 0; row_off < nr; row_off++) {
            uint row = first_row + row_off;
            // Read one byte = two 4-bit indices
            uint byte_pos = row * half_d + pair;
            uint8_t packed_byte = indices[byte_pos];
            uint idx0 = packed_byte & 0xF;
            uint idx1 = (packed_byte >> 4) & 0xF;

            sumf[row_off] += xval0 * cent[idx0] + xval1 * cent[idx1];
        }
    }

    // Handle odd d (if d is not divisible by 2)
    if ((d & 1) && tiisg == 0) {
        uint i = d - 1;
        float xval = xr[i];
        for (uint row_off = 0; row_off < nr; row_off++) {
            uint row = first_row + row_off;
            uint byte_pos = (row * d + i) / 2;
            uint nibble_sel = (row * d + i) % 2;
            uint idx = (indices[byte_pos] >> (nibble_sel * 4)) & 0xF;
            sumf[row_off] += xval * cent[idx];
        }
    }

    // SIMD reduction
    for (uint row_off = 0; row_off < nr; row_off++) {
        float total = simd_sum(sumf[row_off]);
        if (tiisg == 0) {
            output[batch_id * n_out + first_row + row_off] = total * norms[first_row + row_off];
        }
    }
}

// Fused Hadamard rotation + centroid dot product kernel.
// Eliminates the separate x @ Π^T matmul by doing the Walsh-Hadamard
// transform in threadgroup shared memory before the centroid dot.
//
// Each threadgroup handles one batch element. All SIMD groups cooperate
// on the Hadamard transform, then each processes N_DST output rows.
#define PQ_FUSED_THREADS 256  // threads per threadgroup for Hadamard

// Generic fused Hadamard kernel templated on bit width.
template<uint BIT_WIDTH>
kernel void polarquant_mv_fused_hadamard_generic(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* indices   [[buffer(1)]],
    device const float*   norms     [[buffer(2)]],
    device const float*   centroids [[buffer(3)]],
    device       float*   output    [[buffer(4)]],
    device const float*   signs     [[buffer(5)]],
    constant     uint&    d         [[buffer(6)]],
    constant     uint&    n_out     [[buffer(7)]],
    constant     uint&    batch     [[buffer(8)]],
    constant     float&   scale     [[buffer(9)]],
    threadgroup  float*   shared_x  [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    uint   tid   [[thread_index_in_threadgroup]],
    uint   tiisg [[thread_index_in_simdgroup]],
    uint   sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row_group = tgpig.x;
    const uint batch_id  = tgpig.y;

    // Step 1: Cooperatively load input into shared memory with sign flips
    device const float* xb = x + batch_id * d;
    for (uint i = tid; i < d; i += PQ_FUSED_THREADS) {
        shared_x[i] = xb[i] * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: In-place Walsh-Hadamard transform in shared memory
    for (uint h = 1; h < d; h *= 2) {
        for (uint i = tid; i < d / 2; i += PQ_FUSED_THREADS) {
            uint grp = i / h;
            uint pos = i % h;
            uint j = grp * h * 2 + pos;
            float a = shared_x[j];
            float b = shared_x[j + h];
            shared_x[j]     = a + b;
            shared_x[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Apply scale (1/√d)
    for (uint i = tid; i < d; i += PQ_FUSED_THREADS) {
        shared_x[i] *= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Centroid dot product (generic bit-width)
    const uint n_sg = PQ_FUSED_THREADS / PQ_SIMD_WIDTH;
    const uint first_row = (row_group * n_sg + sgitg) * PQ_N_DST;
    if (first_row >= n_out) return;

    constexpr uint n_cent = 1u << BIT_WIDTH;
    float cent[n_cent];
    for (uint i = 0; i < n_cent; i++) {
        cent[i] = centroids[i];
    }

    constexpr uint indices_per_byte = 8 / BIT_WIDTH;
    constexpr uint8_t mask = (1u << BIT_WIDTH) - 1;

    float sumf[PQ_N_DST] = {0.f};
    const uint nr = min((uint)PQ_N_DST, n_out - first_row);

    for (uint i = tiisg; i < d; i += PQ_SIMD_WIDTH) {
        float xval = shared_x[i];

        for (uint row_off = 0; row_off < nr; row_off++) {
            uint row = first_row + row_off;
            uint pos = row * d + i;
            uint byte_pos = pos / indices_per_byte;
            uint bit_offset = (pos % indices_per_byte) * BIT_WIDTH;
            uint idx = (indices[byte_pos] >> bit_offset) & mask;
            sumf[row_off] += xval * cent[idx];
        }
    }

    for (uint row_off = 0; row_off < nr; row_off++) {
        float total = simd_sum(sumf[row_off]);
        if (tiisg == 0 && (first_row + row_off) < n_out) {
            output[batch_id * n_out + first_row + row_off] = total * norms[first_row + row_off];
        }
    }
}

// Instantiate fused Hadamard kernel for each bit width
template [[host_name("polarquant_mv_fused_hadamard_1bit")]]
kernel void polarquant_mv_fused_hadamard_generic<1>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

template [[host_name("polarquant_mv_fused_hadamard_2bit")]]
kernel void polarquant_mv_fused_hadamard_generic<2>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

template [[host_name("polarquant_mv_fused_hadamard_3bit")]]
kernel void polarquant_mv_fused_hadamard_generic<3>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

template [[host_name("polarquant_mv_fused_hadamard_4bit")]]
kernel void polarquant_mv_fused_hadamard_generic<4>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

template [[host_name("polarquant_mv_fused_hadamard_5bit")]]
kernel void polarquant_mv_fused_hadamard_generic<5>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

template [[host_name("polarquant_mv_fused_hadamard_6bit")]]
kernel void polarquant_mv_fused_hadamard_generic<6>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

template [[host_name("polarquant_mv_fused_hadamard_7bit")]]
kernel void polarquant_mv_fused_hadamard_generic<7>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*, device const float*,
    constant uint&, constant uint&, constant uint&, constant float&,
    threadgroup float*, uint3, uint, uint, uint);

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
