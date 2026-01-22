use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, MetalDevice, MetalStorage, Result, Shape, D};
use candle_metal_kernels::metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device.allocate_zeros(size)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        let blit = self.device.blit_command_encoder()?;
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }

    /// Indexed MoE forward - wrapper for fwd_with_ids matching CUDA signature
    ///
    /// This is the main entry point for MoE inference on Metal.
    /// Uses the fused gather + quantized matmul kernel (kernel_mul_mm_id).
    #[allow(clippy::too_many_arguments)]
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &crate::Layout,
        ids: &MetalStorage,
        ids_l: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        self.fwd_with_ids(self_shape, input, input_l, ids, ids_l)
    }

    /// Indexed quantized matrix multiplication for MoE (Mixture of Experts)
    ///
    /// This performs a fused gather + quantized matmul operation:
    /// For each row i: output[i] = input[i] @ dequantize(self.weights[ids[i]]).T
    ///
    /// The dequantization happens fused inside the kernel, not as a separate pass.
    /// This is much more efficient than dequantizing all expert weights upfront.
    ///
    /// # Arguments
    /// * `self_shape` - Shape of the quantized weights [num_experts, n, k]
    /// * `storage` - Input activations storage
    /// * `layout` - Input layout
    /// * `ids` - Expert indices tensor (i32)
    /// * `ids_shape` - Shape of ids tensor
    ///
    /// # Returns
    /// Output storage and shape [batch, seq, n]
    #[allow(clippy::too_many_arguments)]
    pub fn fwd_with_ids(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        ids: &MetalStorage,
        ids_layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        if !ids_layout.is_contiguous() {
            crate::bail!("ids tensor is not contiguous {ids_layout:?}")
        }

        let src_shape = layout.shape();
        let ids_shape = ids_layout.shape();

        // self_shape is [num_experts, n, k] (weights are transposed, so n is output dim)
        if self_shape.rank() < 3 {
            crate::bail!(
                "weight tensor must have at least 3 dimensions for MoE, got {:?}",
                self_shape
            )
        }
        let dims = self_shape.dims();
        let num_experts = dims[dims.len() - 3];
        let n = dims[dims.len() - 2]; // output features
        let k = dims[dims.len() - 1]; // hidden dim

        // src_shape should be [batch, seq, k] or [seq, k]
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let last_k = src_shape.dim(D::Minus1)?;
        if last_k != k {
            crate::bail!(
                "input tensor last dim {} doesn't match weight k dim {}",
                last_k,
                k
            )
        }

        // Index dimensions from ids tensor
        // ids shape: [num_tokens, num_experts_per_tok] or [num_tokens]
        // nei0 = number of experts per token
        // nei1 = number of tokens
        let nei0 = if ids_shape.rank() >= 2 {
            ids_shape.dim(D::Minus1)?
        } else {
            1
        };
        let nei1 = ids_shape.elem_count() / nei0;
        let nbi1 = nei0 * std::mem::size_of::<i32>(); // stride in bytes for ids

        // Determine input layout mode:
        // - Broadcast mode: input shape [..., 1, k] -> all expert slots read same row
        // - Normal mode: input shape [..., nei0, k] -> each expert slot reads its own row
        let input_expert_dim = if src_shape.rank() >= 2 {
            src_shape.dim(D::Minus2)?
        } else {
            1
        };
        let broadcast_input = input_expert_dim == 1;

        // Compute num_tokens (total number of independent tokens)
        let num_tokens = if src_shape.rank() >= 3 {
            src_shape.dims()[..src_shape.rank() - 2].iter().product()
        } else if src_shape.rank() == 2 {
            src_shape.dims()[0]
        } else {
            1
        };

        // Build output shape: [num_tokens, num_experts_per_tok, n]
        // The kernel produces one output row for each (token, expert_slot) pair
        let dst_shape = Shape::from(vec![num_tokens, nei0, n]);

        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_id")?;
        let encoder = device.command_encoder()?;

        // Prepare shapes and strides for the kernel
        // Weight shape: [num_experts, n, k] -> pad to 4D: [1, num_experts, n, k]
        let src0_shape_4d = vec![1, num_experts, n, k];
        let src0_stride_4d: Vec<usize> = {
            let type_ratio =
                self.dtype.type_size() as f32 / self.dtype.block_size() as f32;
            vec![
                (num_experts * n * k) as f32 * type_ratio,
                (n * k) as f32 * type_ratio,
                k as f32 * type_ratio,
                type_ratio,
            ]
            .iter()
            .map(|x| *x as usize)
            .collect()
        };

        // Input shape for kernel: [1, num_tokens, expert_dim, k]
        // The kernel accesses: src1 + id.y * nb12 + id.x * nb11
        // where id.y = token_idx, id.x = expert_slot
        //
        // Two modes:
        // 1. Broadcast mode (input_expert_dim == 1): nb11 = 0, all expert slots read same row
        // 2. Normal mode (input_expert_dim == nei0): nb11 = k*sizeof(float), each reads own row
        let src1_shape_4d = vec![1usize, num_tokens, input_expert_dim, k];
        let src1_stride_4d: Vec<usize> = if broadcast_input {
            vec![
                num_tokens * k * DType::F32.size_in_bytes(), // nb13: outer batch stride
                k * DType::F32.size_in_bytes(),              // nb12: token stride
                0,                                            // nb11: expert slot stride = 0 (BROADCAST!)
                DType::F32.size_in_bytes(),                  // nb10: element stride
            ]
        } else {
            vec![
                num_tokens * nei0 * k * DType::F32.size_in_bytes(), // nb13: outer batch stride
                nei0 * k * DType::F32.size_in_bytes(),              // nb12: token stride
                k * DType::F32.size_in_bytes(),                     // nb11: expert slot stride
                DType::F32.size_in_bytes(),                         // nb10: element stride
            ]
        };

        // Output shape: [1, num_tokens, num_experts_per_tok, n]
        // The kernel writes: dst + token_idx * nb12/4 + expert_slot * nb11/4
        // We want output[token, expert_slot, :] to be at different locations
        let dst_shape_4d = vec![1usize, num_tokens, nei0, n];
        let dst_stride_4d: Vec<usize> = vec![
            num_tokens * nei0 * n * DType::F32.size_in_bytes(), // nb13
            nei0 * n * DType::F32.size_in_bytes(),              // nb12: token stride
            n * DType::F32.size_in_bytes(),                     // nb11: expert slot stride
            DType::F32.size_in_bytes(),                         // nb10: element stride
        ];

        candle_metal_kernels::call_quantized_matmul_mm_id_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            (nei0, nei1),
            nbi1,
            &src0_shape_4d,
            &src0_stride_4d,
            &self.buffer,
            &src1_shape_4d,
            &src1_stride_4d,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &dst_shape_4d,
            &dst_stride_4d,
            0,
            &dst,
            ids.buffer(),
        )
        .map_err(MetalError::from)?;

        let dst_storage =
            crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for candle_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => candle_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => candle_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => candle_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => candle_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => candle_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => candle_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => candle_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::F16,
        }
    }
}
