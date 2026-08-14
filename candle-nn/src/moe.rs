// Mixture-of-Experts (MoE) GEMM operations on CUDA via dynamic PTX runtime.
//
// Uses WMMA grouped kernels for prefill and vectorized MMVQ for decode.
// All kernels are dynamically loaded through cudarc without any link-time dependency on libmoe.a or cudart.

#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle::cuda_backend::cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
    use candle::cuda_backend::WrapErr;
    use candle::quantized::GgmlDType;
    use candle::{CudaDevice, CudaStorage, DType, Storage};
    use half::{bf16, f16};

    const CEILDIV: fn(usize, usize) -> usize = |x, y| (x + y - 1) / y;

    fn calculate_expert_offsets(
        dev: &CudaDevice,
        expert_ids: &CudaSlice<u32>,
        size_m: usize,
        num_experts: usize,
    ) -> Result<CudaSlice<i32>> {
        let expert_counts = dev.alloc_zeros::<i32>(num_experts)?;
        let expert_offsets = dev.alloc_zeros::<i32>(num_experts + 1)?;

        // 1. Count tokens per expert
        let threads = 256;
        let blocks = CEILDIV(size_m, threads);
        let count_func = dev.get_or_load_func("count_tokens_per_expert_kernel", &candle::cuda_backend::kernels::MOE)?;
        let count_cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = count_func.builder();
        builder.arg(expert_ids);
        builder.arg(&expert_counts);
        let size_m_i = size_m as i32; builder.arg(&size_m_i);
        unsafe { builder.launch(count_cfg) }.w()?;

        // 2. Prefix sum to get expert offsets (supports up to 65536 experts via chunked scan)
        let scan_threads = (num_experts.next_power_of_two()).clamp(32, 1024);
        let smem_size = (scan_threads * std::mem::size_of::<i32>()) as u32;
        let scan_func = dev.get_or_load_func("expert_prefix_sum_kernel", &candle::cuda_backend::kernels::MOE)?;
        let scan_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (scan_threads as u32, 1, 1),
            shared_mem_bytes: smem_size,
        };
        let mut builder = scan_func.builder();
        builder.arg(&expert_counts);
        builder.arg(&expert_offsets);
        let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
        unsafe { builder.launch(scan_cfg) }.w()?;

        Ok(expert_offsets)
    }

    pub fn moe_gemm_cuda(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        fn cuda_fwd<T: candle::cuda_backend::CudaDType + DeviceRepr>(
            input: &Tensor,
            weights: &Tensor,
            topk_weights: &Option<Tensor>,
            sorted_token_ids: &Tensor,
            expert_ids: &Tensor,
            topk: usize,
            is_prefill: bool,
            is_bf16: bool,
        ) -> Result<Tensor> {
            let (mut size_m, size_k1) = input.dims2()?;
            if topk_weights.is_none() {
                size_m *= topk;
            }
            let (num_experts, size_n, size_k) = weights.dims3()?;
            if size_k != size_k1 {
                candle::bail!(
                    "input size_k ({size_k1}) and weight size_k ({size_k}) mismatch!"
                );
            }
            let dev = input.device().as_cuda_device()?;
                        
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let (weights_storage, _) = weights.storage_and_layout();
            let weights_slice = match &*weights_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("weight must be a cuda tensor"),
            };

            let (sorted_ids_storage, _) = sorted_token_ids.storage_and_layout();
            let sorted_ids_slice = match &*sorted_ids_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
            };

            let (expert_ids_storage, _) = expert_ids.storage_and_layout();
            let expert_ids_slice = match &*expert_ids_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("expert_ids must be a cuda tensor"),
            };

            let expert_offsets = calculate_expert_offsets(dev, expert_ids_slice, size_m, num_experts)?;

            let output_slice = unsafe { dev.alloc::<T>(size_m * size_n) }?;

            let kernel_name = match (is_bf16, is_prefill) {
                (false, true) => "moe_gemm_wmma_f16_prefill",
                (false, false) => "moe_gemm_wmma_f16_decode",
                (true, true) => "moe_gemm_wmma_bf16_prefill",
                (true, false) => "moe_gemm_wmma_bf16_decode",
            };

            let func = dev.get_or_load_func(kernel_name, &candle::cuda_backend::kernels::MOE)?;

            let grid_n = CEILDIV(size_n, 32);
            let grid = (num_experts as u32, grid_n as u32, 1);
            let block = (128, 1, 1);

            let a_sh_bytes = (32 * 16 * std::mem::size_of::<T>() + 15) & !15;
            let b_sh_bytes = (32 * 16 * std::mem::size_of::<T>() + 15) & !15;
            let c_sh_bytes = (32 * 32 * std::mem::size_of::<f32>() + 15) & !15;
            let smem_bytes = (a_sh_bytes + b_sh_bytes + c_sh_bytes) as u32;

            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem_bytes,
            };

            let topk_w_guard = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
            let topk_w_slice = match &topk_w_guard {
                Some((tw_storage, _)) => match &**tw_storage {
                    Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                    _ => candle::bail!("topk_weights must be a cuda tensor"),
                },
                None => None,
            };

            let mut builder = func.builder();
            builder.arg(input_slice);
            builder.arg(weights_slice);
            builder.arg(sorted_ids_slice);
            builder.arg(&expert_offsets);
            if let Some(tw) = topk_w_slice {
                builder.arg(tw);
            } else {
                builder.arg(&0u64);
            }
            builder.arg(&output_slice);
            let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
            let topk_i = topk as i32; builder.arg(&topk_i);
            let size_m_i = size_m as i32; builder.arg(&size_m_i);
            let size_n_i = size_n as i32; builder.arg(&size_n_i);
            let size_k_i = size_k as i32; builder.arg(&size_k_i);

            unsafe { builder.launch(cfg) }.w()?;

            let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());
            Ok(Tensor::from_storage(
                Storage::Cuda(output),
                (size_m, size_n),
                candle::op::BackpropOp::none(),
                false,
            ))
        }

        match input.dtype() {
            DType::F16 => cuda_fwd::<f16>(
                input, weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, false,
            ),
            DType::BF16 => cuda_fwd::<bf16>(
                input, weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, true,
            ),
            dtype => candle::bail!("moe_gemm only accepts f16/bf16 inputs, got {dtype:?}"),
        }
    }

    pub fn moe_gemm_gguf_cuda(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        if size_k != size_k1 {
            candle::bail!(
                "input size_k ({size_k}) and weight size_k ({size_k1}) mismatch!"
            );
        }
        if size_k % 8 != 0 {
            candle::bail!("size_k must be divisible by 8, got {size_k}");
        }

        let dev = input.device().as_cuda_device()?;
                        let weight_ptr = weights.device_ptr()?;

        let (sorted_ids_storage, _) = sorted_token_ids.storage_and_layout();
        let sorted_ids_slice = match &*sorted_ids_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (expert_ids_storage, _) = expert_ids.storage_and_layout();
        let expert_ids_slice = match &*expert_ids_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("expert_ids must be a cuda tensor"),
        };

        let topk_w_guard = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
        let topk_w_slice = match &topk_w_guard {
            Some((tw_storage, _)) => match &**tw_storage {
                Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            },
            None => None,
        };

        let output_slice = unsafe { dev.alloc::<f32>(size_m * size_n) }?;

        let quant_name = match weights.dtype() {
            GgmlDType::Q8_0 => "q8_0",
            GgmlDType::Q4K => "q4_k",
            GgmlDType::Q2K => "q2_k",
            GgmlDType::Q3K => "q3_k",
            GgmlDType::Q5K => "q5_k",
            GgmlDType::Q6K => "q6_k",
            d => candle::bail!("moe_gemm_gguf does not support weight dtype {d:?}"),
        };

        if is_prefill {
            let expert_offsets = calculate_expert_offsets(dev, expert_ids_slice, size_m, num_experts)?;
            let input_act = input.to_dtype(dtype)?;
            let (act_storage, _) = input_act.storage_and_layout();

            let type_str = if dtype == DType::F16 { "f16" } else { "bf16" };
            let kernel_name = format!("moe_gemm_gguf_prefill_{type_str}_{quant_name}");
            let func = dev.get_or_load_func(&kernel_name, &candle::cuda_backend::kernels::MOE)?;

            let grid = (num_experts as u32, CEILDIV(size_n, 32) as u32, 1);
            let wrap_size = if matches!(weights.dtype(), GgmlDType::Q8_0 | GgmlDType::Q4K) { 32 } else { 64 };
            let block = (wrap_size as u32, 4, 1);

            let block_size_bytes = weights.dtype().type_size();
            let qk = weights.dtype().block_size();
            let a_sh_bytes = (32 * qk * 2 + 15) & !15;
            let b_sh_bytes = (32 * qk * 2 + 15) & !15;
            let b_quant_sh_bytes = (32 * block_size_bytes + 15) & !15;
            let c_sh_bytes = (32 * 32 * std::mem::size_of::<f32>() + 15) & !15;
            let smem_bytes = (a_sh_bytes + b_sh_bytes + b_quant_sh_bytes + c_sh_bytes) as u32;

            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem_bytes,
            };

            let weight_dev_ptr = weight_ptr as u64;

            let mut builder = func.builder();
            match &*act_storage {
                Storage::Cuda(c) => {
                    if dtype == DType::F16 {
                        builder.arg(c.as_cuda_slice::<f16>()?);
                    } else {
                        builder.arg(c.as_cuda_slice::<bf16>()?);
                    }
                }
                _ => candle::bail!("input must be a cuda tensor"),
            }
            builder.arg(&weight_dev_ptr);
            builder.arg(sorted_ids_slice);
            builder.arg(&expert_offsets);
            if let Some(tw) = topk_w_slice {
                builder.arg(tw);
            } else {
                builder.arg(&0u64);
            }
            builder.arg(&output_slice);
            let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
            let topk_i = topk as i32; builder.arg(&topk_i);
            let size_m_i = size_m as i32; builder.arg(&size_m_i);
            let size_n_i = size_n as i32; builder.arg(&size_n_i);
            let size_k_i = size_k as i32; builder.arg(&size_k_i);

            unsafe { builder.launch(cfg) }.w()?;
        } else {
            // Decode path: Quantize input to Q8_1
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let matrix_row_padding = 512;
            let k_padded = (size_k + matrix_row_padding - 1) / matrix_row_padding * matrix_row_padding;
            let m_quant = if topk_weights.is_some() { size_m } else { size_m / topk };
            let q8_1_size = m_quant * (k_padded / 32 * std::mem::size_of::<candle::quantized::k_quants::BlockQ8_1>());
            let mut y_q8_1 = dev.alloc_zeros::<u8>(q8_1_size)?;

            let quant_func = dev.get_or_load_func("quantize_q8_1", &candle::cuda_backend::kernels::QUANTIZED)?;
            let num_blocks = (k_padded + 255) / 256;
            let quant_cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, m_quant as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut quant_builder = quant_func.builder();
            quant_builder.arg(input_slice);
            quant_builder.arg(&mut y_q8_1);
            let size_k_i = size_k as i32; quant_builder.arg(&size_k_i);
            let k_padded_i = k_padded as i32; quant_builder.arg(&k_padded_i);
            unsafe { quant_builder.launch(quant_cfg) }.w()?;

            let kernel_name = format!("moe_gemm_gguf_{quant_name}");
            let func = dev.get_or_load_func(&kernel_name, &candle::cuda_backend::kernels::MOE)?;

            let n_warps = 4;
            let grid_dim = (CEILDIV(size_n, n_warps) as u32, size_m as u32, 1);
            let block_dim = (32, n_warps as u32, 1);
            let block_size_bytes = weights.dtype().type_size();
            let qk = weights.dtype().block_size();
            let shared_bytes = (size_k / qk * block_size_bytes * n_warps + 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim,
                block_dim,
                shared_mem_bytes: shared_bytes,
            };

            let weight_dev_ptr = weight_ptr as u64;

            let mut builder = func.builder();
            builder.arg(&weight_dev_ptr);
            builder.arg(&y_q8_1);
            builder.arg(sorted_ids_slice);
            builder.arg(expert_ids_slice);
            if let Some(tw) = topk_w_slice {
                builder.arg(tw);
            } else {
                builder.arg(&0u64);
            }
            builder.arg(&output_slice);
            let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
            let topk_i = topk as i32; builder.arg(&topk_i);
            let size_m_i = size_m as i32; builder.arg(&size_m_i);
            let size_n_i = size_n as i32; builder.arg(&size_n_i);
            let size_k_i = size_k as i32; builder.arg(&size_k_i);
            let k_padded_i = k_padded as i32; builder.arg(&k_padded_i);

            unsafe { builder.launch(cfg) }.w()?;
        }

        let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());
        Ok(Tensor::from_storage(
            Storage::Cuda(output),
            (size_m, size_n),
            candle::op::BackpropOp::none(),
            false,
        ))
    }
}

#[allow(unused_variables)]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return cuda::moe_gemm_cuda(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                topk,
                is_prefill,
            );
        }
    }
    candle::bail!("moe_gemm (dense MoE) is only supported on CUDA")
}

#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return cuda::moe_gemm_gguf_cuda(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                topk,
                is_prefill,
                dtype,
            );
        }
    }
    candle::bail!("moe_gemm_gguf is only supported on CUDA")
}
