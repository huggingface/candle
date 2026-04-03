// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
#[cfg(feature = "cuda")]
use candle::cuda_backend::kernels::ffi;
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType
            + candle::cuda_backend::cudarc::driver::DeviceRepr
            + candle::cuda_backend::cudarc::driver::ValidAsZeroBits,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle::bail!("moe_gemm only accepts f16/bf16 inputs")
            }
        };

        let (input_storage, input_layout) = input.storage_and_layout();
        let input = match &*input_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };
        let input = match input_layout.contiguous_offsets() {
            Some((o1, o2)) => input.slice(o1..o2),
            None => candle::bail!("input must be contiguous for moe_gemm"),
        };

        let (weights_storage, weights_layout) = weights.storage_and_layout();
        let weights = match &*weights_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let weights = match weights_layout.contiguous_offsets() {
            Some((o1, o2)) => weights.slice(o1..o2),
            None => candle::bail!("weight must be contiguous for moe_gemm"),
        };

        let (sorted_token_ids_storage, sorted_token_ids_layout) =
            sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let sorted_token_ids = match sorted_token_ids_layout.contiguous_offsets() {
            Some((o1, o2)) => sorted_token_ids.slice(o1..o2),
            None => candle::bail!("sorted_token_ids must be contiguous for moe_gemm"),
        };

        let (experts_ids_storage, experts_ids_layout) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };
        let experts_ids = match experts_ids_layout.contiguous_offsets() {
            Some((o1, o2)) => experts_ids.slice(o1..o2),
            None => candle::bail!("experts_ids must be contiguous for moe_gemm"),
        };

        let topk_weights_s_l = topk_weights.as_ref().map(|t| t.storage_and_layout());
        let topk_weights =
            if let Some((topk_weights_storage, topk_weights_layout)) = topk_weights_s_l.as_ref() {
                let topk_weights = match &**topk_weights_storage {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("topk_weights must be a cuda tensor"),
                };
                Some(match topk_weights_layout.contiguous_offsets() {
                    Some((o1, o2)) => topk_weights.slice(o1..o2),
                    None => candle::bail!("topk_weights must be contiguous for moe_gemm"),
                })
            } else {
                None
            };

        let output = dev.alloc_zeros::<T>(size_m * size_n)?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let stream = dev.cuda_stream();
        let cu_stream = stream.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            let (topk_weights_ptr, _topk_weights_guard) =
                if let Some(topk_weights) = topk_weights.as_ref() {
                    let (ptr, guard) = topk_weights.device_ptr(&stream);
                    (ptr as *const f32, Some(guard))
                } else {
                    (std::ptr::null(), None)
                };
            let (input_ptr, _input_guard) = input.device_ptr(&stream);
            let (weights_ptr, _weights_guard) = weights.device_ptr(&stream);
            let (sorted_token_ids_ptr, _sorted_token_ids_guard) =
                sorted_token_ids.device_ptr(&stream);
            let (experts_ids_ptr, _experts_ids_guard) = experts_ids.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr(&stream);
            let (expert_counts_ptr, _expert_counts_guard) = expert_counts.device_ptr(&stream);
            let (expert_offsets_ptr, _expert_offsets_guard) = expert_offsets.device_ptr(&stream);
            ffi::dispatch_moe_gemm(
                input_ptr as *const c_void,   // [size_m, size_k]
                weights_ptr as *const c_void, // [num_experts, size_n, size_k]
                sorted_token_ids_ptr as *const i32,
                experts_ids_ptr as *const i32,
                topk_weights_ptr,
                output_ptr as *mut c_void,      // [size_m, size_n]
                expert_counts_ptr as *mut i32,  // pre-allocated buffer [num_experts]
                expert_offsets_ptr as *mut i32, // pre-allocated buffer [num_experts + 1]
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                is_prefill,
                cu_stream,
            );
        }

        use candle::op::BackpropOp;
        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle::bail!("moe_gemm only accepts f16/bf16 inputs")
        }
    }
}

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
    candle::bail!("moe_gemm is only implemented for the cuda backend")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
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
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use half::{bf16, f16};

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = input.device().as_cuda_device()?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_s_l = topk_weights.as_ref().map(|t| t.storage_and_layout());
        let topk_weights =
            if let Some((topk_weights_storage, topk_weights_layout)) = topk_weights_s_l.as_ref() {
                let topk_weights = match &**topk_weights_storage {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("topk_weights must be a cuda tensor"),
                };
                Some(match topk_weights_layout.contiguous_offsets() {
                    Some((o1, o2)) => topk_weights.slice(o1..o2),
                    None => candle::bail!("topk_weights must be contiguous for moe_gemm_gguf"),
                })
            } else {
                None
            };

        let (sorted_token_ids_storage, sorted_token_ids_layout) =
            sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let sorted_token_ids = match sorted_token_ids_layout.contiguous_offsets() {
            Some((o1, o2)) => sorted_token_ids.slice(o1..o2),
            None => candle::bail!("sorted_token_ids must be contiguous for moe_gemm_gguf"),
        };

        let (experts_ids_storage, experts_ids_layout) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };
        let experts_ids = match experts_ids_layout.contiguous_offsets() {
            Some((o1, o2)) => experts_ids.slice(o1..o2),
            None => candle::bail!("experts_ids must be contiguous for moe_gemm_gguf"),
        };

        let output = dev.alloc_zeros::<f32>(size_m * size_n)?;
        let stream = dev.cuda_stream();
        let cu_stream = stream.cu_stream() as i64;
        use candle::op::BackpropOp;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        unsafe {
            let (topk_weights_ptr, _topk_weights_guard) =
                if let Some(topk_weights) = topk_weights.as_ref() {
                    let (ptr, guard) = topk_weights.device_ptr(&stream);
                    (ptr as *const f32, Some(guard))
                } else {
                    (std::ptr::null(), None)
                };
            let (sorted_token_ids_ptr, _sorted_token_ids_guard) =
                sorted_token_ids.device_ptr(&stream);
            let (experts_ids_ptr, _experts_ids_guard) = experts_ids.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr(&stream);
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input_storage, input_layout) = input.storage_and_layout();
                match &*input_storage {
                    candle::Storage::Cuda(c) => {
                        if dtype == DType::F16 {
                            let c = c.as_cuda_slice::<f16>()?;
                            let c = match input_layout.contiguous_offsets() {
                                Some((o1, o2)) => c.slice(o1..o2),
                                None => {
                                    candle::bail!("input must be contiguous for moe_gemm_gguf")
                                }
                            };
                            let (ptr, guard) = c.device_ptr(&stream);
                            ffi::moe_gemm_gguf_prefill(
                                ptr as *const c_void, // [size_m or size_m/topk, size_k]
                                weight_ptr,           // [num_experts, size_n, size_k]
                                sorted_token_ids_ptr as *const i32,
                                experts_ids_ptr as *const i32,
                                topk_weights_ptr,
                                output_ptr as *mut c_void, // [size_m, size_n]
                                num_experts as i32,
                                topk as i32,
                                size_m as i32,
                                size_n as i32,
                                size_k as i32,
                                0,
                                gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                                cu_stream,
                            );
                            let _input_guard = guard;
                        } else {
                            let c = c.as_cuda_slice::<bf16>()?;
                            let c = match input_layout.contiguous_offsets() {
                                Some((o1, o2)) => c.slice(o1..o2),
                                None => {
                                    candle::bail!("input must be contiguous for moe_gemm_gguf")
                                }
                            };
                            let (ptr, guard) = c.device_ptr(&stream);
                            ffi::moe_gemm_gguf_prefill(
                                ptr as *const c_void, // [size_m or size_m/topk, size_k]
                                weight_ptr,           // [num_experts, size_n, size_k]
                                sorted_token_ids_ptr as *const i32,
                                experts_ids_ptr as *const i32,
                                topk_weights_ptr,
                                output_ptr as *mut c_void, // [size_m, size_n]
                                num_experts as i32,
                                topk as i32,
                                size_m as i32,
                                size_n as i32,
                                size_k as i32,
                                1,
                                gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                                cu_stream,
                            );
                            let _input_guard = guard;
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                }
            } else {
                let (input_storage, input_layout) = input.storage_and_layout();
                let input = match &*input_storage {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input must be a cuda tensor"),
                };
                let input = match input_layout.contiguous_offsets() {
                    Some((o1, o2)) => input.slice(o1..o2),
                    None => candle::bail!("input must be contiguous for moe_gemm_gguf"),
                };
                let (input_ptr, _input_guard) = input.device_ptr(&stream);

                ffi::moe_gemm_gguf(
                    input_ptr as *const f32,     // [size_m or size_m/topk, size_k]
                    weight_ptr as *const c_void, // [num_experts, size_n, size_k]
                    sorted_token_ids_ptr as *const i32,
                    experts_ids_ptr as *const i32,
                    topk_weights_ptr,
                    output_ptr as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    cu_stream,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_fwd(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            candle::bail!("moe_gemm_gguf only accepts f32 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
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
    candle::bail!("moe_gemm_gguf is only implemented for the cuda backend")
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::moe_gemm;
    use candle::{DType, Device, Tensor};

    #[test]
    fn moe_gemm_accepts_contiguous_offset_views_cuda() -> candle::Result<()> {
        let device = Device::new_cuda(0)?;
        let topk = 1usize;
        let size_m = 4usize;
        let size_n = 16usize;
        let size_k = 32usize;
        let num_experts = 2usize;

        let input_full = Tensor::arange(0f32, ((size_m + 1) * size_k) as f32, &device)?
            .to_dtype(DType::F16)?
            .reshape((size_m + 1, size_k))?;
        let input = input_full.narrow(0, 1, size_m)?;
        let input_ref = input.force_contiguous()?;

        let weights_full =
            Tensor::arange(0f32, ((num_experts + 1) * size_n * size_k) as f32, &device)?
                .to_dtype(DType::F16)?
                .reshape((num_experts + 1, size_n, size_k))?;
        let weights = weights_full.narrow(0, 1, num_experts)?;
        let weights_ref = weights.force_contiguous()?;

        let sorted_ids_full =
            Tensor::from_vec((0u32..=(size_m as u32)).collect(), size_m + 1, &device)?;
        let sorted_ids = sorted_ids_full.narrow(0, 1, size_m)?;
        let sorted_ids_ref = sorted_ids.force_contiguous()?;

        let experts_ids_full = Tensor::from_vec(vec![0u32, 0, 1, 0, 1], size_m + 1, &device)?;
        let experts_ids = experts_ids_full.narrow(0, 1, size_m)?;
        let experts_ids_ref = experts_ids.force_contiguous()?;

        let out = moe_gemm(
            &input,
            &weights,
            &None,
            &sorted_ids,
            &experts_ids,
            topk,
            true,
        )?;
        let out_ref = moe_gemm(
            &input_ref,
            &weights_ref,
            &None,
            &sorted_ids_ref,
            &experts_ids_ref,
            topk,
            true,
        )?;

        assert_eq!(out.to_vec2::<half::f16>()?, out_ref.to_vec2::<half::f16>()?);
        Ok(())
    }
}
