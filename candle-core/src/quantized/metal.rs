use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, Layout, MetalDevice, MetalStorage, Result, Shape, D};
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
        let buffer = device
            .new_buffer_builder()
            .with_zeros(size)
            .with_label("qstorage_zeros")
            .build()?;
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

    /// Converts an element-stride array to bytes for this storage's
    /// (possibly sub-byte-per-element, block-quantized) dtype. Shared by
    /// `fwd` and `indexed_moe_forward` so the two matmul paths can't
    /// silently diverge on how quantized weight strides are computed.
    fn quantized_byte_strides(&self, stride: &[usize]) -> Vec<usize> {
        stride
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect()
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.buffer.length())
            .with_label("qstorage_dequantize_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
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

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&out)
            .with_label("qstorage_dequantized")
            .build()?;
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
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantized")
            .build()?;
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
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix")
            .build()?;
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

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix_onto")
            .build()?;
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

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_onto")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<MetalStorage> {
        use crate::MetalError;

        if ids.dtype() != DType::U32 {
            crate::bail!("quantized embedding expects u32 ids, got {:?}", ids.dtype())
        }
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids_len = ids_l.shape().elem_count();
        let device = self.device.clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(ids_len * hidden, DType::F32)
            .with_label("qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_quantized_get_rows(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            hidden,
            hidden * self.dtype.type_size() / self.dtype.block_size(),
            ids_len,
            &self.buffer,
            ids.buffer(),
            ids_l.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(
            dst,
            device.clone(),
            ids_len * hidden,
            DType::F32,
        ))
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
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
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
        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
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
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = self.quantized_byte_strides(src0_l.stride());

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

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Indexed/routed matmul for MoE expert dispatch: `self` holds all
    /// experts' weights stacked as `[num_experts, n, k]`; `ids` is the
    /// routing table `[batch, topk]` (one row of expert indices per token);
    /// `input` is `[batch, topk_or_1, k]` (the `topk_or_1` broadcasts a
    /// single per-token embedding across all its selected experts when 1,
    /// or supplies a distinct value per selected expert when `topk`).
    /// Returns `[batch, topk, n]`, one row per (token, selected-expert)
    /// pair -- this does not apply top-k routing weights or sum across
    /// experts; see candle_metal_kernels::call_quantized_matmul_mm_id.
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &crate::Layout,
        ids: &MetalStorage,
        ids_l: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !input_l.is_contiguous() {
            crate::bail!("indexed_moe_forward input is not contiguous {input_l:?}")
        }
        if !ids_l.is_contiguous() {
            crate::bail!("indexed_moe_forward ids is not contiguous {ids_l:?}")
        }
        if input.dtype() != DType::F32 {
            crate::bail!(
                "indexed_moe_forward input must be F32, got {:?}",
                input.dtype()
            )
        }
        // The kernel unconditionally reads `ids` as raw int32_t regardless of
        // the Rust-side dtype tag; U32 is bit-compatible for the small
        // non-negative expert indices this carries, but anything else would
        // silently misread the buffer at the byte-stride computed below.
        if ids.dtype() != DType::U32 {
            crate::bail!("indexed_moe_forward ids must be U32, got {:?}", ids.dtype())
        }

        let (_num_experts, n, k) = self_shape.dims3()?;
        let in_shape = input_l.shape();
        let (batch, in_dim1, in_k) = in_shape.dims3()?;
        if in_k != k {
            crate::bail!(
                "indexed_moe_forward input {:?} incompatible with weight {:?}",
                in_shape,
                self_shape
            )
        }
        let idx_shape = ids_l.shape();
        let (idx_batch, topk) = idx_shape.dims2()?;
        if idx_batch != batch {
            crate::bail!("indexed_moe_forward batch mismatch: input {batch} vs ids {idx_batch}")
        }
        if in_dim1 != 1 && in_dim1 != topk {
            crate::bail!("indexed_moe_forward input dim1 ({in_dim1}) must be 1 or topk ({topk})")
        }

        let device = input.device().clone();
        let dst_shape = Shape::from((batch, topk, n));
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("indexed_moe_forward")
            .build()?;
        let encoder = device.command_encoder()?;

        let src0_l = crate::Layout::contiguous(self_shape.dims());
        let src0_stride = self.quantized_byte_strides(src0_l.stride());

        let input_stride = input_l
            .stride()
            .iter()
            .map(|x| x * DType::F32.size_in_bytes())
            .collect::<Vec<_>>();
        let ids_stride = ids_l
            .stride()
            .iter()
            .map(|x| x * ids.dtype().size_in_bytes())
            .collect::<Vec<_>>();

        // Decode (batch == 1) and small multi-token batches (batch <= 8,
        // e.g. a speculative-decode verify step) route to the matrix-
        // *vector* kernel. `mv_id`'s Metal dispatch grid depth is
        // `nei0 * nei1` (top-k * n_tokens) -- proportional to real work --
        // while `mm_id`'s is `ne02` (total expert count), effectively
        // fixed regardless of batch size; at these small batch sizes
        // `mm_id` pays that near-fixed dispatch tax against very little
        // real work, `mv_id` doesn't. Measured (production 256-expert/
        // top-8/2048-in/512-out shape): `mv_id` beats `mm_id` by 10-20%
        // throughout batch 1-16, roughly ties at 16, and loses clearly by
        // 32+ -- `8` is chosen with a safety margin below the measured
        // crossover, matching the real speculative-decode verify-window
        // range this targets, not the full range `mv_id` still wins.
        // Batch-general correctness (not just batch == 1) confirmed via a
        // bit-exact self-differential (`mv_id` at batch N vs. N
        // independent batch-1 calls) plus a tolerance-based `mm_id`
        // cross-check, both at this same production shape (see this
        // crate's test suite). `mv_id_eligible`
        // (candle-metal-kernels/src/kernels/quantized.rs) is the single
        // source of truth for which dtypes qualify and the minimum
        // contraction-dim (k) each needs -- not duplicated here, so it
        // can't drift from the wrapper's own per-dtype tuning table. Every
        // other dtype, every too-small-k shape, and every batch > 8 call
        // keeps using the matrix-*matrix* kernel -- always correct, just
        // not what this extension targets.
        let use_mv = batch <= 8 && candle_metal_kernels::mv_id_eligible(self.dtype.into(), k);

        // call_quantized_matmul_mv_id, call_quantized_matmul_mm_id, and
        // call_quantized_matmul_mm_id_chunked all take the same leading
        // 19-argument shape (chunked takes one more, `max_nei1`, trailing);
        // this macro keeps every dispatch site (which must stay in exact
        // argument-for-argument sync on that shared prefix) structurally
        // unable to desync, rather than independent ~20-line call
        // expressions a future edit could update one of and forget the
        // others. Variadic in its trailing args (review finding, 2026-07-27:
        // an earlier version only covered the exact-19-arg case, so adding
        // chunking meant hand-writing a second, unprotected ~20-line call
        // site for mm_id instead of extending this macro).
        macro_rules! dispatch_indexed_moe {
            ($f:path $(, $extra:expr)*) => {
                $f(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    self.dtype.into(),
                    self_shape.dims(),
                    &src0_stride,
                    &self.buffer,
                    0, // this storage always owns its buffer outright, so its offset is always 0
                    in_shape.dims(),
                    &input_stride,
                    input.buffer(),
                    input_l.start_offset() * DType::F32.size_in_bytes(),
                    idx_shape.dims(),
                    &ids_stride,
                    ids.buffer(),
                    ids_l.start_offset() * ids.dtype().size_in_bytes(),
                    dst_shape.dims(),
                    0,
                    &dst,
                    $($extra),*
                )
                .map_err(MetalError::from)?
            };
        }

        if use_mv {
            dispatch_indexed_moe!(candle_metal_kernels::call_quantized_matmul_mv_id);
        } else {
            // mm_id, unlike mv_id, has a hard per-dispatch ceiling: its
            // rowids scratch is sized off the *whole-batch* token count in
            // threadgroup memory, so a large enough prefill batch overflows
            // the device's fixed budget regardless of tuning -- found live
            // via a real ~2594-token prompt.
            let max_nei1 = candle_metal_kernels::mm_id_max_nei1(device.device(), topk as i64);
            // Per-expert row counts let the kernel skip the routing-table
            // scan for a token tile that holds none of its expert's rows,
            // rather than paying the full scan to discover that -- measured
            // 3.61s of a 9.42s prefill. `call_quantized_matmul_mm_id_chunked`
            // computes these itself, per chunk rather than once for the
            // whole batch, since a chunk-local count is what its chunk-local
            // guard actually needs (see that function's own doc comment).
            dispatch_indexed_moe!(
                candle_metal_kernels::call_quantized_matmul_mm_id_chunked,
                max_nei1
            );
        }

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.buffer.length())
            .with_label("qstorage_data_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device
        .new_buffer_builder()
        .with_data(data)
        .with_label("qstorage_load_quantized")
        .build()?;
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
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::BF16,
        }
    }
}
