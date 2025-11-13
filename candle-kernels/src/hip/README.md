# Candle HIP Kernels

**TEAM-491 Note:** Most primitive operations have been moved to `rocm-rs`.

## What's in rocm-rs (use these):
- ✅ Binary operations (add, sub, mul, div) - `rocm_rs::rocarray::elementwise_*`
- ✅ Reduction operations (sum, max, min) - `rocm_rs::rocarray::reduce_*`
- ✅ Fill operations - `rocm_rs::rocarray::fill_value`
- ✅ Transpose - `rocm_rs::rocarray::transpose`
- ✅ Sorting - `rocm_rs::hip::memory_ext::MemoryExt::sort()`
- ✅ Matrix operations - `rocm_rs::rocblas::gemm`
- ✅ Convolution - `rocm_rs::miopen::convolution`
- ✅ Activation (ReLU, Tanh, Sigmoid, ELU) - `rocm_rs::miopen::activation`
- ✅ **Cast operations** - `rocm_rs::rocarray::kernels.hip` (cast_f32_f16, etc.)
- ✅ **Ternary operations** - `rocm_rs::rocarray::kernels.hip` (where_u8_f32, etc.)
- ✅ **Affine operations** - `rocm_rs::rocarray::kernels.hip` (affine_f32, etc.)
- ✅ **Unary operations** - `rocm_rs::rocarray::kernels.hip` (ugelu_f32, usilu_f32, uexp_f32, etc.)

## What's Candle-specific (implement here):
- ⏭️ **Quantization** (`quantized.cu` - 158KB) - GGUF quantization formats
  - INT8/INT4 quantization schemes
  - Dequantization operations
  - Candle-specific quantization formats

## Directory Structure:
```
candle-kernels/src/hip/
└── README.md (this file)
└── (quantized.hip - to be added when needed)
```

## Integration:
Candle's ROCm backend should:
1. Use `rocm-rs` for all primitive operations
2. Only implement quantization kernels here (Candle-specific)
3. Load quantization kernels via `rocm_rs::hip::Module`

## Example Usage:
```rust
use rocm_rs::rocarray::kernels;

// Binary operation
kernels::elementwise_add_async(&a, &b, &result, len, &stream)?;

// Unary operation (GELU)
kernels::unary_gelu_f32(&input, &output, len)?;

// Cast operation
kernels::cast_f32_f16(&input_f32, &output_f16, len)?;

// Affine operation
kernels::affine_f32(&input, &output, mul, add, len)?;

// Ternary operation (where)
kernels::where_u8_f32(&condition, &true_val, &false_val, &output, len)?;
```

See `rocm-rs/src/rocarray/kernels.hip` for full list of available operations.
