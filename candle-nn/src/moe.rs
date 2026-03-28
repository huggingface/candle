// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
// Rewritten to use cudarc's driver API for dynamic loading (PTX-based).
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda {
    use candle::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
    use candle::cuda_backend::kernels;
    use candle::cuda_backend::{CudaDevice, WrapErr};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{builder_arg as barg, DType, Result, Tensor};
    use half::{bf16, f16};

    const MATRIX_ROW_PADDING: usize = 512;
    const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
    const WARP_SIZE: usize = 32;

    fn pad(p: usize, q: usize) -> usize {
        p.div_ceil(q) * q
    }

    fn ceil_div(a: usize, b: usize) -> usize {
        (a + b - 1) / b
    }

    /// GGUF block type sizes and quantization block sizes.
    /// Returns (qk, sizeof_block_q_t) for a given gguf_dtype index.
    fn gguf_block_info(gguf_dtype: usize) -> Result<(usize, usize)> {
        // These must match the C++ struct sizes from gguf.cuh
        match gguf_dtype {
            0 => Ok((32, 34)),   // Q8_0: qk=32, sizeof(block_q8_0)=34
            1 => Ok((256, 144)), // Q4_K: qk=256, sizeof(block_q4_K)=144
            2 => Ok((256, 84)),  // Q2_K: qk=256, sizeof(block_q2_K)=84
            3 => Ok((256, 110)), // Q3_K: qk=256, sizeof(block_q3_K)=110
            4 => Ok((256, 176)), // Q5_K: qk=256, sizeof(block_q5_K)=176
            5 => Ok((256, 210)), // Q6_K: qk=256, sizeof(block_q6_K)=210
            _ => candle::bail!("unsupported gguf dtype {gguf_dtype}"),
        }
    }

    /// Launches count_tokens_per_expert_kernel and expert_prefix_sum_kernel
    /// to compute expert offsets, replacing the C++ `calculate_expert_offsets()` function.
    fn calculate_expert_offsets(
        sorted_expert_ids: &CudaSlice<u32>,
        expert_counts: &CudaSlice<u32>,
        expert_offsets: &CudaSlice<u32>,
        num_experts: usize,
        size_m: usize,
        dev: &CudaDevice,
    ) -> Result<()> {
        // expert_counts is already zeroed from alloc_zeros

        // Step 1: count_tokens_per_expert_kernel
        let count_func =
            dev.get_or_load_func("count_tokens_per_expert_kernel", &kernels::MOE_UTILS)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(size_m, 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = count_func.builder();
        builder.arg(sorted_expert_ids);
        builder.arg(expert_counts);
        barg!(builder, size_m as i32);
        unsafe { builder.launch(cfg) }.w()?;

        // Step 2: expert_prefix_sum_kernel (exclusive scan: counts -> offsets)
        // Requires num_experts <= 1024 (single-block Hillis-Steele scan)
        let scan_func = dev.get_or_load_func("expert_prefix_sum_kernel", &kernels::MOE_UTILS)?;
        let block_size = num_experts.next_power_of_two().max(32);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * std::mem::size_of::<i32>()) as u32,
        };
        let mut builder = scan_func.builder();
        builder.arg(expert_counts);
        builder.arg(expert_offsets);
        barg!(builder, num_experts as i32);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(())
    }

    /// WMMA-based MoE GEMM for fp16/bf16 weights (non-quantized).
    pub fn moe_gemm_cuda(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        fn inner<
            T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
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

            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let (weights_storage, _) = weights.storage_and_layout();
            let weights_slice = match &*weights_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("weight must be a cuda tensor"),
            };

            let (sorted_token_ids_storage, _) = sorted_token_ids.storage_and_layout();
            let sorted_token_ids_slice = match &*sorted_token_ids_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
            };

            let (experts_ids_storage, _) = experts_ids.storage_and_layout();
            let experts_ids_slice = match &*experts_ids_storage {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("experts_ids must be a cuda tensor"),
            };

            let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;

            // Calculate expert offsets from expert_ids
            let expert_counts = dev.alloc_zeros::<u32>(num_experts)?;
            let expert_offsets = dev.alloc_zeros::<u32>(num_experts + 1)?;
            calculate_expert_offsets(
                experts_ids_slice,
                &expert_counts,
                &expert_offsets,
                num_experts,
                size_m,
                dev,
            )?;

            // Select kernel variant
            let kernel_name = match (input.dtype(), is_prefill) {
                (DType::F16, true) => "moe_gemm_grouped_f16_prefill",
                (DType::F16, false) => "moe_gemm_grouped_f16_decode",
                (DType::BF16, true) => "moe_gemm_grouped_bf16_prefill",
                (DType::BF16, false) => "moe_gemm_grouped_bf16_decode",
                _ => candle::bail!("moe_gemm_wmma only accepts f16/bf16 inputs"),
            };
            let func = dev.get_or_load_func(kernel_name, &kernels::MOE_WMMA)?;

            // Grid: (num_experts, ceil(size_n/N_BLK))
            // Block: 128 threads (4 warps, 1D layout)
            const M_BLK: usize = 32;
            const N_BLK: usize = 32;
            const K_BLK: usize = 16;
            let cfg = LaunchConfig {
                grid_dim: (num_experts as u32, ceil_div(size_n, N_BLK) as u32, 1),
                block_dim: (128, 1, 1),
                // Shared memory: A_sh[M_BLK, K_BLK] + B_sh[N_BLK, K_BLK] + pad + C_sh[M_BLK, N_BLK]
                shared_mem_bytes: {
                    let a_sh = M_BLK * K_BLK * 2; // half/bf16 = 2 bytes
                    let b_sh = N_BLK * K_BLK * 2;
                    let ab = a_sh + b_sh;
                    let pad = (16 - (ab % 16)) % 16;
                    let c_sh = M_BLK * N_BLK * 4; // float = 4 bytes
                    (ab + pad + c_sh) as u32
                },
            };

            // Extract topk_weights storage before builder so borrow outlives builder
            let tw_storage = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
            let tw_slice: Option<&CudaSlice<f32>> = if let Some((ref store, _)) = tw_storage {
                match &**store {
                    candle::Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                    _ => candle::bail!("topk_weights must be a cuda tensor"),
                }
            } else {
                None
            };

            let mut builder = func.builder();
            builder.arg(input_slice);
            builder.arg(weights_slice);
            builder.arg(sorted_token_ids_slice);
            builder.arg(&expert_offsets); // kernel uses expert_offsets (computed above)

            if let Some(tw_s) = tw_slice {
                builder.arg(tw_s);
            } else {
                builder.arg(&0usize); // null pointer
            }

            builder.arg(&output);
            barg!(
                builder,
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32
            );
            unsafe { builder.launch(cfg) }.w()?;

            let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
            let output = Tensor::from_storage(
                candle::Storage::Cuda(output),
                (size_m, size_n),
                candle::op::BackpropOp::none(),
                false,
            );
            Ok(output)
        }

        match input.dtype() {
            DType::F16 => inner::<f16>(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                experts_ids,
                topk,
                is_prefill,
            ),
            DType::BF16 => inner::<bf16>(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                experts_ids,
                topk,
                is_prefill,
            ),
            _ => candle::bail!("moe_gemm only accepts f16/bf16 inputs"),
        }
    }

    /// GGUF-quantized MoE GEMM (decode path: dot-product based, f32 input).
    /// The decode kernel uses expert_ids directly per token (no expert_offsets).
    #[allow(clippy::too_many_arguments)]
    fn moe_gemm_gguf_decode(
        input: &Tensor,
        weight_ptr: u64,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &CudaSlice<u32>,
        experts_ids: &CudaSlice<u32>,
        num_experts: usize,
        topk: usize,
        size_m: usize,
        size_n: usize,
        size_k: usize,
        gguf_dtype: usize,
        dev: &CudaDevice,
    ) -> Result<CudaSlice<f32>> {
        let (qk_val, block_type_size) = gguf_block_info(gguf_dtype)?;

        // Step 1: Quantize input f32 -> q8_1
        let (input_storage, _) = input.storage_and_layout();
        let input_slice = match &*input_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let qk8_1_block_size = GgmlDType::Q8_1.block_size();
        let qk8_1_type_size = GgmlDType::Q8_1.type_size();
        let k_padded = pad(size_k, MATRIX_ROW_PADDING);
        let num_blocks_per_row = k_padded / qk8_1_block_size;
        let dst_row_size_bytes = num_blocks_per_row * qk8_1_type_size;
        let total_rows = if topk_weights.is_some() {
            size_m
        } else {
            size_m / topk
        };
        let y_size_in_bytes = total_rows * dst_row_size_bytes;
        let y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };

        // Launch quantize_q8_1 kernel
        let quantize_func = dev.get_or_load_func("quantize_q8_1", &kernels::QUANTIZED)?;
        let num_q_blocks = ceil_div(k_padded, CUDA_QUANTIZE_BLOCK_SIZE);
        let q_cfg = LaunchConfig {
            grid_dim: (num_q_blocks as u32, total_rows as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = quantize_func.builder();
        builder.arg(input_slice);
        builder.arg(&y_q8_1);
        barg!(builder, size_k as i32, k_padded as i32);
        unsafe { builder.launch(q_cfg) }.w()?;

        // Step 2: Launch MOE GGUF decode kernel
        // Grid: (ceil(size_n / nwarps), size_m)
        // Block: (WARP_SIZE, nwarps)
        let kernel_name = match gguf_dtype {
            0 => "moe_gemm_gguf_q8_0",
            1 => "moe_gemm_gguf_q4k",
            2 => "moe_gemm_gguf_q2k",
            3 => "moe_gemm_gguf_q3k",
            4 => "moe_gemm_gguf_q5k",
            5 => "moe_gemm_gguf_q6k",
            _ => candle::bail!("unsupported gguf dtype {gguf_dtype}"),
        };
        let func = dev.get_or_load_func(kernel_name, &kernels::MOE_GGUF)?;

        let nwarps: usize = 4;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(size_n, nwarps) as u32, size_m as u32, 1),
            block_dim: (WARP_SIZE as u32, nwarps as u32, 1),
            // shared: (size_k / qk) * sizeof(block_q_t) * nwarps + 1024
            shared_mem_bytes: ((size_k / qk_val) * block_type_size * nwarps + 1024) as u32,
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n)? };

        // Extract topk_weights storage before builder
        let tw_storage = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
        let tw_slice: Option<&CudaSlice<f32>> = if let Some((ref store, _)) = tw_storage {
            match &**store {
                candle::Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            }
        } else {
            None
        };

        let mut builder = func.builder();
        barg!(builder, weight_ptr); // all_weights (void*)
        builder.arg(&y_q8_1); // all_inputs (q8_1)
        builder.arg(sorted_token_ids); // sorted_token_ids
        builder.arg(experts_ids); // expert_ids (NOT expert_offsets!)

        if let Some(tw_s) = tw_slice {
            builder.arg(tw_s); // topk_weights
        } else {
            builder.arg(&0usize); // null pointer
        }

        builder.arg(&output); // all_outputs
        barg!(
            builder,
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            k_padded as i32
        );
        unsafe { builder.launch(cfg) }.w()?;

        Ok(output)
    }

    /// GGUF-quantized MoE GEMM (prefill path: WMMA-based, fp16/bf16 input).
    /// The prefill kernel uses expert_offsets (calculated from expert_ids).
    #[allow(clippy::too_many_arguments)]
    fn moe_gemm_gguf_prefill(
        input: &Tensor,
        weight_ptr: u64,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &CudaSlice<u32>,
        experts_ids: &CudaSlice<u32>,
        num_experts: usize,
        topk: usize,
        size_m: usize,
        size_n: usize,
        size_k: usize,
        input_dtype: DType,
        gguf_dtype: usize,
        dev: &CudaDevice,
    ) -> Result<CudaSlice<f32>> {
        let (qk_val, block_type_size) = gguf_block_info(gguf_dtype)?;

        // Calculate expert offsets from expert_ids
        let expert_counts = dev.alloc_zeros::<u32>(num_experts)?;
        let expert_offsets = dev.alloc_zeros::<u32>(num_experts + 1)?;
        calculate_expert_offsets(
            experts_ids,
            &expert_counts,
            &expert_offsets,
            num_experts,
            size_m,
            dev,
        )?;

        // Select kernel name based on input dtype and GGUF type
        let dtype_prefix = match input_dtype {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => candle::bail!("moe_gemm_gguf_prefill only accepts f16/bf16"),
        };
        let gguf_suffix = match gguf_dtype {
            0 => "q8_0",
            1 => "q4k",
            2 => "q2k",
            3 => "q3k",
            4 => "q5k",
            5 => "q6k",
            _ => candle::bail!("unsupported gguf dtype {gguf_dtype}"),
        };
        let kernel_name = format!("moe_gemm_gguf_prefill_{}_{}", dtype_prefix, gguf_suffix);
        let func = dev.get_or_load_func(&kernel_name, &kernels::MOE_WMMA_GGUF)?;

        const M_BLK: usize = 32;
        const N_BLK: usize = 32;

        // Grid: (num_experts, ceil(size_n/N_BLK))
        // Block: (wrap_size, WARPS_PER_BLOCK=4, 1) where wrap_size is 32 for Q8_0/Q4K, 64 for Q2K/Q3K/Q5K/Q6K
        let wrap_size: usize = match gguf_dtype {
            0 | 1 => 32,
            _ => 64, // Q2K, Q3K, Q5K, Q6K need 64-thread warps
        };

        // Shared memory calculation matches original host code:
        //   A_sh: M_BLK * qk * 2 (half/bf16)
        //   B_sh: N_BLK * qk * 2 (half/bf16)
        //   B_quant_sh: N_BLK * block_size_bytes
        //   C_sh: M_BLK * N_BLK * 4 (float), with alignment padding
        let a_sh_bytes = M_BLK * qk_val * 2;
        let b_sh_bytes = N_BLK * qk_val * 2;
        let b_quant_sh_bytes = N_BLK * block_type_size;
        let smem_no_c = a_sh_bytes + b_sh_bytes + b_quant_sh_bytes;
        let c_sh_offset = smem_no_c % 4; // alignof(float) = 4
        let padding = if c_sh_offset != 0 { 4 - c_sh_offset } else { 0 };
        let c_sh_bytes = M_BLK * N_BLK * 4;
        let smem_bytes = smem_no_c + padding + c_sh_bytes;

        let cfg = LaunchConfig {
            grid_dim: (num_experts as u32, ceil_div(size_n, N_BLK) as u32, 1),
            block_dim: (wrap_size as u32, 4, 1), // (wrap_size, WARPS_PER_BLOCK)
            shared_mem_bytes: smem_bytes as u32,
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n)? };

        // Cast input to the appropriate type
        let input_cast = input.to_dtype(input_dtype)?;
        let (input_s, _) = input_cast.storage_and_layout();

        // Extract topk_weights storage before builder
        let tw_storage = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
        let tw_slice: Option<&CudaSlice<f32>> = if let Some((ref store, _)) = tw_storage {
            match &**store {
                candle::Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            }
        } else {
            None
        };

        let mut builder = func.builder();
        match input_dtype {
            DType::F16 => {
                let slice = match &*input_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f16>()?,
                    _ => candle::bail!("input must be cuda"),
                };
                builder.arg(slice);
            }
            DType::BF16 => {
                let slice = match &*input_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                    _ => candle::bail!("input must be cuda"),
                };
                builder.arg(slice);
            }
            _ => candle::bail!("unsupported dtype"),
        }

        barg!(builder, weight_ptr); // weights (uint8_t*)
        builder.arg(sorted_token_ids); // sorted_token_ids
        builder.arg(&expert_offsets); // expert_offsets (computed above)

        if let Some(tw_s) = tw_slice {
            builder.arg(tw_s); // topk_weights
        } else {
            builder.arg(&0usize); // null pointer
        }

        builder.arg(&output); // output
        barg!(
            builder,
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            gguf_dtype as i32
        );
        unsafe { builder.launch(cfg) }.w()?;

        Ok(output)
    }

    /// Main entry point for GGUF-quantized MoE GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemm_gguf_cuda(
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

        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle::bail!("moe_gemm_gguf only accepts q2k, q3k, q4k, q5k, q6k or q8_0 weights!")
            }
        };

        let weight_ptr = weights.device_ptr()? as u64;

        let (sorted_token_ids_storage, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids_slice = match &*sorted_token_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let (experts_ids_storage, _) = experts_ids.storage_and_layout();
        let experts_ids_slice = match &*experts_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        assert!(size_k % 8 == 0, "size_k must be divisible by 8");

        let output = if is_prefill {
            moe_gemm_gguf_prefill(
                input,
                weight_ptr,
                topk_weights,
                sorted_token_ids_slice,
                experts_ids_slice,
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
                dtype,
                gguf_dtype,
                dev,
            )?
        } else {
            moe_gemm_gguf_decode(
                input,
                weight_ptr,
                topk_weights,
                sorted_token_ids_slice,
                experts_ids_slice,
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
                gguf_dtype,
                dev,
            )?
        };

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            candle::op::BackpropOp::none(),
            false,
        );

        Ok(output)
    }
}

// ============================================================================
// Public API
// ============================================================================

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
    cuda::moe_gemm_cuda(
        input,
        weights,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        topk,
        is_prefill,
    )
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
    cuda::moe_gemm_gguf_cuda(
        input,
        weights,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        topk,
        is_prefill,
        dtype,
    )
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
