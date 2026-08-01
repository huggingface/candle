// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
use candle::quantized::QTensor;
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "rocm")]
mod rocm;

#[cfg(feature = "cuda")]
pub use cuda::moe_gemm;

/// The dense (f16/bf16) fused expert GEMM has no ROCm implementation.
///
/// It is `moe_gemm_wmma` in `candle-kernels/src/moe/moe_wmma.cu`, written
/// against `nvcuda::wmma` and `<mma.h>`, and compiled into the `libmoe.a`
/// static library — neither the intrinsics nor that build step exists on the
/// ROCm side. The quantized [`moe_gemm_gguf`] does run there.
#[cfg(not(feature = "cuda"))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!(
        "moe_gemm is only implemented for the cuda backend: its kernel is \
         `nvcuda::wmma` and has no ROCm equivalent. Quantized MoE weights go \
         through moe_gemm_gguf, which does support ROCm."
    )
}

/// Dispatches on the input's device, not on which feature is enabled.
///
/// Cargo features are additive, so `cuda` and `rocm` can both be on -- one
/// dependent asking for each is enough. Selecting the backend at compile time
/// would then drop the ROCm path silently and leave a ROCm tensor to fail
/// inside the CUDA one. `candle-core` resolves the same question by matching
/// the storage at runtime with both arms compiled in.
///
/// ROCm reaches the same routed mat-vecs through `indexed_moe_forward`; see
/// [`rocm`] for why, and for what `is_prefill` and `dtype` mean there (nothing).
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[allow(clippy::too_many_arguments)]
// `is_prefill` and `dtype` are consumed by the CUDA arm alone, so a build
// without it has nothing to bind them to.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    match input.device() {
        #[cfg(feature = "cuda")]
        candle::Device::Cuda(_) => cuda::moe_gemm_gguf(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        #[cfg(feature = "rocm")]
        candle::Device::Rocm(_) => rocm::moe_gemm_gguf(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
        ),
        dev => candle::bail!("moe_gemm_gguf is not implemented for {dev:?}"),
    }
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle::DType,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf is only implemented for the cuda and rocm backends")
}
