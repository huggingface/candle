//! Heterogeneous multi-LoRA batching CUDA kernels (S-LoRA / Punica style BGMV).
//!
//! This crate provides the fused-kernel fast path for
//! [`candle_nn::lora::LoraLinear::forward_with_adapters`]: a single decode-step
//! batch where every row selects its own LoRA adapter is applied as two
//! gather-GEMV passes ("shrink" `in -> r`, then "expand" `r -> out`) that read
//! the stacked adapter matrices in place through a per-row slot index, instead
//! of materializing the per-row gather that the pure-tensor reference path
//! builds before its batched matmuls.
//!
//! [`bgmv_delta`] is numerically equivalent to that reference path and returns
//! the same `delta` tensor; candle-nn selects it behind the `lora-cuda`
//! feature when the input is on a CUDA device and the sequence length is 1.

mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Tensor};
use core::ffi::c_void;
use half::{bf16, f16};

fn dtype_code(dt: DType) -> Result<u32> {
    match dt {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        dt => candle::bail!("lora kernels only support f32/f16/bf16 inputs, got {dt:?}"),
    }
}

/// `tmp[i, :] = x[i, :] · A_stack[slot(i)]`, gathering `A_stack` by `slot`.
struct LoraShrink;

impl LoraShrink {
    fn fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        a: &CudaStorage,
        a_l: &Layout,
        slots: &CudaStorage,
        slots_l: &Layout,
        dtype: u32,
    ) -> Result<(CudaStorage, Shape)> {
        if !x_l.is_contiguous() || !a_l.is_contiguous() || !slots_l.is_contiguous() {
            candle::bail!("lora bgmv shrink expects contiguous inputs")
        }
        let (batch, in_dim) = x_l.shape().dims2()?;
        let (_n_slots, a_in, r) = a_l.shape().dims3()?;
        if a_in != in_dim {
            candle::bail!("lora bgmv shrink: x in_dim {in_dim} != A_stack in_dim {a_in}")
        }
        if slots_l.shape().dims1()? != batch {
            candle::bail!("lora bgmv shrink: expected {batch} slots")
        }
        let dev = x.device();
        let x = x.as_cuda_slice::<T>()?.slice(x_l.start_offset()..);
        let a = a.as_cuda_slice::<T>()?.slice(a_l.start_offset()..);
        let slots = slots
            .as_cuda_slice::<u32>()?
            .slice(slots_l.start_offset()..);
        let out_shape = Shape::from((batch, r));
        let mut dst = unsafe { dev.alloc::<T>(batch * r)? };
        let stream = dev.cuda_stream();
        unsafe {
            let (x_ptr, _g) = x.device_ptr(&stream);
            let (a_ptr, _g) = a.device_ptr(&stream);
            let (slots_ptr, _g) = slots.device_ptr(&stream);
            let (dst_ptr, _g) = dst.device_ptr_mut(&stream);
            ffi::lora_bgmv_shrink(
                x_ptr as *const c_void,
                a_ptr as *const c_void,
                slots_ptr as *const u32,
                dst_ptr as *mut c_void,
                batch as u32,
                in_dim as u32,
                r as u32,
                dtype,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok((CudaStorage::wrap_cuda_slice(dst, dev.clone()), out_shape))
    }
}

impl candle::CustomOp3 for LoraShrink {
    fn name(&self) -> &'static str {
        "lora-bgmv-shrink"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for lora-bgmv kernels")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        a: &CudaStorage,
        a_l: &Layout,
        slots: &CudaStorage,
        slots_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F32 => self.fwd_t::<f32>(x, x_l, a, a_l, slots, slots_l, 0),
            DType::F16 => self.fwd_t::<f16>(x, x_l, a, a_l, slots, slots_l, 1),
            DType::BF16 => self.fwd_t::<bf16>(x, x_l, a, a_l, slots, slots_l, 2),
            dt => candle::bail!("lora-bgmv-shrink only supports f32/f16/bf16, got {dt:?}"),
        }
    }
}

/// `delta[i, :] = tmp[i, :] · B_stack[slot(i)]`, gathering `B_stack` by `slot`.
struct LoraExpand;

impl LoraExpand {
    fn fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        tmp: &CudaStorage,
        tmp_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        slots: &CudaStorage,
        slots_l: &Layout,
        dtype: u32,
    ) -> Result<(CudaStorage, Shape)> {
        if !tmp_l.is_contiguous() || !b_l.is_contiguous() || !slots_l.is_contiguous() {
            candle::bail!("lora bgmv expand expects contiguous inputs")
        }
        let (batch, r) = tmp_l.shape().dims2()?;
        let (_n_slots, b_r, out_dim) = b_l.shape().dims3()?;
        if b_r != r {
            candle::bail!("lora bgmv expand: tmp rank {r} != B_stack rank {b_r}")
        }
        if slots_l.shape().dims1()? != batch {
            candle::bail!("lora bgmv expand: expected {batch} slots")
        }
        let dev = tmp.device();
        let tmp = tmp.as_cuda_slice::<T>()?.slice(tmp_l.start_offset()..);
        let b = b.as_cuda_slice::<T>()?.slice(b_l.start_offset()..);
        let slots = slots
            .as_cuda_slice::<u32>()?
            .slice(slots_l.start_offset()..);
        let out_shape = Shape::from((batch, out_dim));
        let mut dst = unsafe { dev.alloc::<T>(batch * out_dim)? };
        let stream = dev.cuda_stream();
        unsafe {
            let (tmp_ptr, _g) = tmp.device_ptr(&stream);
            let (b_ptr, _g) = b.device_ptr(&stream);
            let (slots_ptr, _g) = slots.device_ptr(&stream);
            let (dst_ptr, _g) = dst.device_ptr_mut(&stream);
            ffi::lora_bgmv_expand(
                tmp_ptr as *const c_void,
                b_ptr as *const c_void,
                slots_ptr as *const u32,
                dst_ptr as *mut c_void,
                batch as u32,
                r as u32,
                out_dim as u32,
                dtype,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok((CudaStorage::wrap_cuda_slice(dst, dev.clone()), out_shape))
    }
}

impl candle::CustomOp3 for LoraExpand {
    fn name(&self) -> &'static str {
        "lora-bgmv-expand"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for lora-bgmv kernels")
    }

    fn cuda_fwd(
        &self,
        tmp: &CudaStorage,
        tmp_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        slots: &CudaStorage,
        slots_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match tmp.dtype() {
            DType::F32 => self.fwd_t::<f32>(tmp, tmp_l, b, b_l, slots, slots_l, 0),
            DType::F16 => self.fwd_t::<f16>(tmp, tmp_l, b, b_l, slots, slots_l, 1),
            DType::BF16 => self.fwd_t::<bf16>(tmp, tmp_l, b, b_l, slots, slots_l, 2),
            dt => candle::bail!("lora-bgmv-expand only supports f32/f16/bf16, got {dt:?}"),
        }
    }
}

/// Batched heterogeneous LoRA delta for a decode step (one token per row).
///
/// Computes `delta[i] = x[i] · A_stack[slot(i)] · B_stack[slot(i)]` for every
/// row of the batch, gathering the adapter matrices in place by `row_slots`.
///
/// * `x`         — `[batch, in_dim]`, one of f32/f16/bf16.
/// * `a_stack`   — `[n_slots, in_dim, r]`, `A^T` per slot, rank-padded with
///   zeros; slot 0 is the all-zero (no-adapter) slot.
/// * `b_stack`   — `[n_slots, r, out_dim]`, `B^T · (alpha/r)` per slot, rank-padded.
/// * `row_slots` — `[batch]`, `u32`, the slot index selected by each row.
///
/// The result matches candle-nn's pure-tensor `batched_lora_delta`.
pub fn bgmv_delta(
    x: &Tensor,
    a_stack: &Tensor,
    b_stack: &Tensor,
    row_slots: &Tensor,
) -> Result<Tensor> {
    dtype_code(x.dtype())?;
    let x = x.contiguous()?;
    let a_stack = a_stack.contiguous()?;
    let b_stack = b_stack.contiguous()?;
    let row_slots = row_slots.contiguous()?;
    let tmp = x.apply_op3(&a_stack, &row_slots, LoraShrink)?;
    tmp.apply_op3(&b_stack, &row_slots, LoraExpand)
}
