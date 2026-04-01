// PolarQuant fused centroid-lookup matmul kernel.
//
// Computes output = x_rot @ W_rot^T where W_rot is built on-the-fly from
// bit-packed centroid indices + norms + a centroid table.
//
// Each thread computes one element of the output:
//   output[batch_idx][out_idx] = norms[out_idx] * Σ_i x_rot[batch_idx][i] * centroids[unpack(indices, out_idx*d+i)]
//
// Arguments:
//   x_rot:    [batch, d] f32 — pre-rotated input
//   indices:  [n_out * d / indices_per_byte] u8 — bit-packed centroid indices
//   norms:    [n_out] f32 — per-row weight norms
//   centroids: [n_centroids] f32 — Lloyd-Max centroid values
//   output:   [batch, n_out] f32 — result

#include <metal_stdlib>
using namespace metal;

// Unpack a b-bit index from packed byte storage
inline uint unpack_index(
    device const uint8_t* packed,
    uint pos,
    uint bit_width,
    uint indices_per_byte,
    uint8_t mask
) {
    uint byte_pos = pos / indices_per_byte;
    uint bit_offset = (pos % indices_per_byte) * bit_width;
    return (packed[byte_pos] >> bit_offset) & mask;
}

// Main kernel: each thread computes output[batch_id][out_id]
template<uint BIT_WIDTH>
kernel void polarquant_centroid_dot(
    device const float* x_rot       [[buffer(0)]],
    device const uint8_t* indices    [[buffer(1)]],
    device const float* norms        [[buffer(2)]],
    device const float* centroids    [[buffer(3)]],
    device float* output             [[buffer(4)]],
    constant uint& d                 [[buffer(5)]],
    constant uint& n_out             [[buffer(6)]],
    constant uint& batch             [[buffer(7)]],
    uint2 tid                        [[thread_position_in_grid]]
) {
    uint batch_id = tid.y;
    uint out_id = tid.x;

    if (batch_id >= batch || out_id >= n_out) return;

    constexpr uint indices_per_byte = 8 / BIT_WIDTH;
    constexpr uint8_t mask = (1u << BIT_WIDTH) - 1;

    float norm = norms[out_id];
    uint idx_start = out_id * d;
    device const float* xr = x_rot + batch_id * d;

    float dot = 0.0f;
    for (uint i = 0; i < d; i++) {
        uint centroid_idx = unpack_index(indices, idx_start + i, BIT_WIDTH, indices_per_byte, mask);
        dot += xr[i] * centroids[centroid_idx];
    }

    output[batch_id * n_out + out_id] = dot * norm;
}

// Instantiate for common bit widths
template [[host_name("polarquant_centroid_dot_1bit")]]
kernel void polarquant_centroid_dot<1>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);

template [[host_name("polarquant_centroid_dot_2bit")]]
kernel void polarquant_centroid_dot<2>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);

template [[host_name("polarquant_centroid_dot_3bit")]]
kernel void polarquant_centroid_dot<3>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);

template [[host_name("polarquant_centroid_dot_4bit")]]
kernel void polarquant_centroid_dot<4>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);

template [[host_name("polarquant_centroid_dot_5bit")]]
kernel void polarquant_centroid_dot<5>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);

template [[host_name("polarquant_centroid_dot_6bit")]]
kernel void polarquant_centroid_dot<6>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);

template [[host_name("polarquant_centroid_dot_7bit")]]
kernel void polarquant_centroid_dot<7>(
    device const float*, device const uint8_t*, device const float*,
    device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint2);
