#include <metal_stdlib>
using namespace metal;

template <typename T> T erf(T in) {
    // constants
    constexpr const float a1 =  0.254829592;
    constexpr const float a2 = -0.284496736;
    constexpr const float a3 =  1.421413741;
    constexpr const float a4 = -1.453152027;
    constexpr const float a5 =  1.061405429;
    constexpr const float p  =  0.3275911;

    float x = static_cast<float>(in);

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    float t = 1.0/(1.0 + p*x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return T(sign*y);
}

// Fused GELU Kernel: x = 0.5 * x * (1 + erf(x / sqrt(2)))
// Using precise erf instead of fast::erf which was missing/invalid
template<typename T>
kernel void fused_gelu(
    constant size_t &elem_count [[buffer(0)]],
    device const T *input [[buffer(1)]],
    device T *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= elem_count) {
        return;
    }
    
    T x = input[id];
    T sqrt2_inv = static_cast<T>(0.70710678118654752440); // 1/sqrt(2)
    T erf_val = erf(x * sqrt2_inv);
    output[id] = static_cast<T>(0.5) * x * (static_cast<T>(1.0) + erf_val);
}

// Explicit instantiations for GELU
#define INSTANTIATE_GELU(T, NAME) \
    template [[host_name(NAME)]] \
    kernel void fused_gelu<T>( \
        constant size_t &elem_count [[buffer(0)]], \
        device const T *input [[buffer(1)]], \
        device T *output [[buffer(2)]], \
        uint id [[thread_position_in_grid]])

INSTANTIATE_GELU(float, "fused_gelu_f32");
INSTANTIATE_GELU(half, "fused_gelu_f16");
INSTANTIATE_GELU(bfloat, "fused_gelu_bf16");


// Fused LayerNorm Kernel
// Supports generic dimension normalization (usually last dim)
// Computes mean, variance, and normalization in one go if possible, 
// or naive implementation for general cases.
//
// For SAM3, we mostly care about [B, C, H, W] normalizing over C (dim=1) or last dim.
// Let's implement a standard "normalize last dimension" kernel first as it's most common 
// (e.g. for Transformer tokens [B, N, C]).
// For [B, C, H, W] normalizing C, that's "channel last" conceptually if we view H,W as batch.

// Naive implementation for standard LayerNorm (normalize over contiguous last dim)
// Assumes one threadgroup per row, or generic grid. 
// For simplicity and to cover general cases, we can start with a robust implementation.

template<typename T>
kernel void fused_layernorm(
    device const T *input [[buffer(0)]],
    device T *output [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device const T *bias [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant uint &norm_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],  // x: element in row, y: row index
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpd [[threads_per_threadgroup]]
) {
    // This simple kernel works if we dispatch 1 thread per element, 
    // but that means we redundantly compute mean/var or need atomic reductions.
    // Better approach: One threadgroup per row (instance).
    
    // Let's implement reduction within threadgroup for mean/var.
    
    // However, simpler robust baseline: 
    // If we rely on the implementation in reduce.metal pattern, it's complex.
    // Let's optimize for the specific SAM3 use case:
    // SAM3 uses LayerNorm2d (normalize C in B,C,H,W) and LayerNorm (normalize last dim).
}

// Optimized LayerNorm for "last dimension" (contiguous)
// Each threadgroup handles one or more rows.
// Assumes norm_size is small enough to be efficient (e.g. 768, 1024, 256).

template<typename T, int BLOCK_SIZE>
kernel void fused_layernorm_row(
    device const T *input [[buffer(0)]],
    device T *output [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device const T *bias [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant uint &norm_size [[buffer(5)]], // Number of elements in last dim
    uint gid [[threadgroup_position_in_grid]], // Row index
    uint lid [[thread_index_in_threadgroup]]
) {
    // Pointer to this row's data
    device const T *row_input = input + gid * norm_size;
    device T *row_output = output + gid * norm_size;
    
    // 1. Compute Mean
    float sum_val = 0.0f;
    for (uint i = lid; i < norm_size; i += BLOCK_SIZE) {
        sum_val += float(row_input[i]);
    }
    
    // Block-wide reduction for mean
    threadgroup float shared_sum[BLOCK_SIZE];
    shared_sum[lid] = sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared mem
    for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared_sum[0] / float(norm_size);
    
    // 2. Compute Variance
    float sum_sq_diff = 0.0f;
    for (uint i = lid; i < norm_size; i += BLOCK_SIZE) {
        float diff = float(row_input[i]) - mean;
        sum_sq_diff += diff * diff;
    }
    
    shared_sum[lid] = sum_sq_diff; // Reuse shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float variance = shared_sum[0] / float(norm_size);
    float inv_std = rsqrt(variance + eps);
    
    // 3. Normalize and Write Output
    for (uint i = lid; i < norm_size; i += BLOCK_SIZE) {
        float val = float(row_input[i]);
        float normalized = (val - mean) * inv_std;
        
        float w = float(weight[i]);
        float b = float(bias[i]); // Assuming bias exists, if not need separate kernel or check
        
        row_output[i] = T(normalized * w + b);
    }
}

// Instantiations
#define INSTANTIATE_LN(T, NAME) \
    template [[host_name(NAME)]] \
    kernel void fused_layernorm_row<T, 256>( \
        device const T *input [[buffer(0)]], \
        device T *output [[buffer(1)]], \
        device const T *weight [[buffer(2)]], \
        device const T *bias [[buffer(3)]], \
        constant float &eps [[buffer(4)]], \
        constant uint &norm_size [[buffer(5)]], \
        uint gid [[threadgroup_position_in_grid]], \
        uint lid [[thread_index_in_threadgroup]])

INSTANTIATE_LN(float, "fused_layernorm_f32");
INSTANTIATE_LN(half, "fused_layernorm_f16");
// Output: [Batch, L, L] where L = h*w
// term_h: [Batch, w, h, h]
// term_w: [Batch, h, w, w]
// Batch represents bs*heads
template<typename T>
kernel void gen_rel_pos_mask(
    device const T *term_h [[buffer(0)]],
    device const T *term_w [[buffer(1)]],
    device T *output [[buffer(2)]],
    constant uint &h_size [[buffer(3)]],
    constant uint &w_size [[buffer(4)]],
    constant size_t &elem_count [[buffer(5)]], // Total elements to process
    uint id [[thread_position_in_grid]])
{
    if (id >= elem_count) {
        return;
    }
    
    // Index arithmetic
    // id = b * (L*L) + row * L + col
    // L = h * w
    uint h = h_size;
    uint w = w_size;
    uint L = h * w;
    uint LL = L * L;
    
    uint b = id / LL;
    uint rem_b = id % LL;
    
    uint row = rem_b / L;
    uint col = rem_b % L;
    
    // row = y1 * w + x1
    uint y1 = row / w;
    uint x1 = row % w;
    
    // col = y2 * w + x2
    uint y2 = col / w;
    uint x2 = col % w;
    
    // term_h index: [b, x1, y1, y2]
    // layout: b * (w * h * h) + x1 * (h * h) + y1 * h + y2
    uint idx_h = b * (w * h * h) + x1 * (h * h) + y1 * h + y2;
    
    // term_w index: [b, y1, x1, x2]
    // layout: b * (h * w * w) + y1 * (w * w) + x1 * w + x2
    uint idx_w = b * (h * w * w) + y1 * (w * w) + x1 * w + x2;
    
    output[id] = term_h[idx_h] + term_w[idx_w];
}

#define INSTANTIATE_REL_POS(T, NAME) \
    template [[host_name(NAME)]] \
    kernel void gen_rel_pos_mask<T>( \
        device const T *term_h [[buffer(0)]], \
        device const T *term_w [[buffer(1)]], \
        device T *output [[buffer(2)]], \
        constant uint &h_size [[buffer(3)]], \
        constant uint &w_size [[buffer(4)]], \
        constant size_t &elem_count [[buffer(5)]], \
        uint id [[thread_position_in_grid]])

INSTANTIATE_REL_POS(float, "gen_rel_pos_mask_f32");
INSTANTIATE_REL_POS(half, "gen_rel_pos_mask_f16");
INSTANTIATE_REL_POS(bfloat, "gen_rel_pos_mask_bf16");
