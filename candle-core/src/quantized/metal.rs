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
        // `call_quantized_matmul_mv_t`'s own dispatch grid already supports
        // multiple rows in one dispatch (`height: ne11`, `ne11 = m`) -- for
        // the block-quantized dtypes below, every row is read via
        // `src1 + r1*ne10 + ...` inside the kernel itself (`ne10` is the
        // contraction dim, a stride that's correct for any contiguous
        // multi-row `src1` on its own, independent of the wrapper's own
        // `nb11` field), so one batched call is exactly equivalent to `m`
        // independent single-row calls -- confirmed against the actual
        // kernel bodies (`metal_src/quantized.metal`) AND against a real
        // bit-exact differential test per dtype
        // (`candle-core/tests/quantized_tests.rs`'s
        // `qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact`),
        // not assumed from source reading alone. F16, BF16, F32 index via
        // `src1 + r1*nb11 + ...` (their own distinct kernel template), and
        // the wrapper always passes `nb11 = 0` -- every row would silently
        // read row 0 for those three dtypes at `m > 1`, so they keep the
        // old, always-correct per-row loop. **`Q2K` and `Q5_1` also stay on
        // the loop, despite superficially matching the `ne10`-indexed
        // template**: the differential test suite above found real, silent
        // per-dtype corruption at `batch > 1` for both -- `Q2K` scattered
        // zeros within rows at `batch == 2` (this dtype already carries a
        // known Metal-kernel quirk elsewhere in this file, see the "Fixing
        // a bug in Metal for GGML" comment in `candle-metal-kernels`' own
        // tuning table), `Q5_1` an entirely-zeroed row 0 at `batch == 4`
        // with rows 1-3 correct (a distinct failure shape, root cause not
        // chased further). Root cause not chased for either -- this list is
        // deliberately built from "verified correct by a real differential
        // test," not "looks safe by reading the kernel source," after these
        // two proved that shortcut wrong twice. `Q8_1` is also absent: its
        // own mat-vec Metal kernel (`kernel_mul_mv_q8_1_f32`) doesn't exist
        // anywhere in the vendored `.metal` source at all -- a pre-existing
        // gap this change neither causes nor fixes (the identical
        // kernel-load failure already happens at `batch == 1` through the
        // unchanged loop path).
        let batched_dispatch_safe = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q8_0
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        );
        if batched_dispatch_safe {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, m, n, k),
                storage.buffer(),
                layout.start_offset() * storage.dtype().size_in_bytes(),
                &self.buffer,
                0,
                &dst,
            )
            .map_err(MetalError::from)?;
        } else {
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

        // Decode (batch == 1) and small multi-token batches (batch <= 8,
        // e.g. a speculative-decode verify step) route to `fwd_mv`'s
        // matrix-vector dispatch. `mv_t`'s own Metal dispatch grid is
        // `height: ne11` (the row/batch count) -- proportional to real
        // work -- while `mm_t`'s `width: ne11/32` stays at a fixed 1 for
        // any batch from 1 to 32, so `mm_t` pays nearly the same dispatch
        // tax at batch 2 as it would at batch 32: a near-fixed cost poorly
        // amortized at small batch. `8` is a conservative margin below an
        // unmeasured-past-that-point crossover, not a proven-optimal
        // number. `fwd_mv` requires a rank-2 weight and a rank-<=3 src --
        // routing a rank-3-weight or rank-4-src call into it would trade a
        // previously-working `mm_t` path for a hard error, so those stay
        // on `mm_t` regardless of batch size.
        //
        // `Q5_1` is excluded from this `m > 1` routing entirely (not just
        // from `fwd_mv`'s own internal batched-dispatch dtype list): a
        // real differential test found `fwd_mv`'s pre-existing per-row
        // *loop* -- unmodified by this change, previously only reachable
        // at `m > 1` via a rank-3, leading-batch-dim-`b`-greater-than-1
        // input, never via this rank-2 path -- silently zeroes row 0's
        // output for this dtype specifically at `m == 4`, rows 1..m
        // unaffected. Root cause not found (plausibly a Metal hazard-
        // tracking/encoder-ordering issue specific to this dtype's own
        // kernel tuning, across repeated same-buffer dispatches); `Q5_1`
        // keeps its exact prior behavior (`mm_t` for any `m > 1`) rather
        // than risk exposing the same latent bug through this new code
        // path.
        //
        // `Q2K` is excluded the same way, for the same reason, found the
        // same way: the pre-existing `qmm_batch` test (`quantized_tests.rs`,
        // stacks real batches up to `m == 12`) failed once `m > 1` could
        // reach `fwd_mv`'s loop at all for this dtype -- beyond the `m <= 8`
        // range this fix's own differential test suite swept, where the
        // loop looked correct. Root cause not chased (plausibly the same
        // hazard-tracking family of issue as `Q5_1`, just with a larger
        // `m` threshold before it manifests); excluded rather than
        // partially trusted.
        let m_gt_1_safe_for_fwd_mv = !matches!(self.dtype, GgmlDType::Q5_1 | GgmlDType::Q2K);
        if src_shape.dim(D::Minus2)? == 1
            || (src_shape.dim(D::Minus2)? <= 8
                && self_shape.rank() == 2
                && src_shape.rank() <= 3
                && m_gt_1_safe_for_fwd_mv)
        {
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
