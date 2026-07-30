//! Rotary Embeddings
//!
use candle::{CpuStorage, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// The `(buffer, layout)` pairs every rope kernel reads, in kernel argument order.
#[cfg(feature = "rocm")]
type RocmRopeInputs<'a, T> = [(
    &'a candle::rocm_backend::SendSyncDeviceMemory<T>,
    &'a Layout,
); 3];

/// Shared launcher for the three ROCm rope kernels (`rope`, `rope_i`, `rope_thd`).
///
/// All three take `src, cos, sin, dst` followed by a variable number of `uint32_t`
/// shape parameters (see `ROPE_OP` in `candle-kernels/src/reduce.cu`), so only
/// `params` differs between them.
#[cfg(feature = "rocm")]
fn rocm_rope_launch<T: Copy + Send + Sync + 'static>(
    name: &str,
    inputs: RocmRopeInputs<T>,
    el: usize,
    params: &[u32],
    dev: &candle::RocmDevice,
) -> Result<candle::rocm_backend::SendSyncDeviceMemory<T>> {
    use candle::rocm_backend::{kernel_name, rocm_rs};

    let func = dev.get_or_load_func(
        &kernel_name::<T>(name),
        &candle::rocm_backend::kernels::REDUCE,
    )?;
    // SAFETY: Set later by running the kernel.
    let dst = dev.alloc::<T>(el)?;

    // The rope kernels are not grid-stride: every one of the `el / 2` pairs needs
    // its own thread, so the grid has to cover them all.
    const BLOCK_SIZE: u32 = 256;
    let n_threads = (el / 2) as u32;
    let grid = rocm_rs::hip::Dim3::from(n_threads.div_ceil(BLOCK_SIZE));
    let block = rocm_rs::hip::Dim3::from(BLOCK_SIZE);

    let mut ptrs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(4);
    for ((mem, layout), what) in inputs.iter().zip(["src", "cos", "sin"]) {
        let offset = match layout.contiguous_offsets() {
            None => candle::bail!("{what} input has to be contiguous"),
            Some((o1, _o2)) => o1,
        };
        // SAFETY: `offset` is an in-bounds element index of a contiguous layout.
        ptrs.push(unsafe { mem.ptr_at(offset) });
    }
    ptrs.push(dst.0.as_ptr());

    // Kernel params are passed by address, so `ptrs`/`params` must outlive the launch.
    let mut args: Vec<*mut std::ffi::c_void> = Vec::with_capacity(ptrs.len() + params.len());
    for p in ptrs.iter() {
        args.push(p as *const *mut std::ffi::c_void as *mut std::ffi::c_void);
    }
    for p in params {
        args.push(p as *const u32 as *mut std::ffi::c_void);
    }
    func.launch(grid, block, 0, Some(dev.stream()), &mut args)
        .map_err(|e| candle::Error::Msg(format!("Kernel launch failed: {}", e)))?;
    Ok(dst)
}

/// Interleaved variant of rotary embeddings.
/// The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
/// The resulting y0 and y1 are also interleaved with:
///   y0 = x0*cos - x1*sin
///   y1 = x0*sin + x1*cos
#[derive(Debug, Clone)]
struct RotaryEmbI;

impl candle::CustomOp3 for RotaryEmbI {
    fn name(&self) -> &'static str {
        "rotary-emb-int"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            if t == 1 {
                for bh_i in 0..b * h {
                    let off = bh_i * d;
                    for i_over_2 in 0..d / 2 {
                        let i = off + 2 * i_over_2;
                        let rope_i = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_over_2 + b_i * d / 2
                        } else {
                            i_over_2
                        };
                        dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                        dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                    }
                }
            } else {
                src.par_chunks(t * d)
                    .zip(dst.par_chunks_mut(t * d))
                    .enumerate()
                    .for_each(|(bh_i, (src, dst))| {
                        for i_over_2 in 0..t * d / 2 {
                            let i = 2 * i_over_2;
                            let rope_i = if unbatched_rope {
                                let b_i = bh_i / h;
                                i_over_2 + b_i * t * d / 2
                            } else {
                                i_over_2
                            };
                            dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                            dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                        }
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_i"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, (b * h) as u32, (t * d) as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope_i");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope-i {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_i_f32",
            candle::DType::F16 => "rope_i_f16",
            candle::DType::BF16 => "rope_i_bf16",
            dtype => candle::bail!("rope-i is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, src.dtype())
            .with_label("rope_i")
            .build()?;
        candle_metal_kernels::call_rope_i(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b * h,
            t * d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s1: &candle::RocmStorage,
        l1: &Layout,
        s2: &candle::RocmStorage,
        l2: &Layout,
        s3: &candle::RocmStorage,
        l3: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        use candle::rocm_backend::{RocmStorageSlice as S, SendSyncDeviceMemory};
        use candle::RocmDevice;

        fn inner<T: Copy + Send + Sync + 'static>(
            src: &SendSyncDeviceMemory<T>,
            l_src: &Layout,
            cos: &SendSyncDeviceMemory<T>,
            l_cos: &Layout,
            sin: &SendSyncDeviceMemory<T>,
            l_sin: &Layout,
            dev: &RocmDevice,
        ) -> Result<SendSyncDeviceMemory<T>> {
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let params = [(b * h) as u32, (t * d) as u32, stride_b];
            let inputs = [(src, l_src), (cos, l_cos), (sin, l_sin)];
            rocm_rope_launch("rope_i", inputs, el, &params, dev)
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (S::BF16(s1), S::BF16(s2), S::BF16(s3)) => S::BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F16(s1), S::F16(s2), S::F16(s3)) => S::F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F32(s1), S::F32(s2), S::F32(s3)) => S::F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F64(s1), S::F64(s2), S::F64(s3)) => S::F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::rocm_backend::RocmStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }
}

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle::bail!("inconsistent batch size in rope {b_sz} {cs:?}",)
            }
            Ok((t, d))
        }
        _ => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

pub fn rope_i(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbI)
}

pub fn rope_i_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    let rope = rope.flatten_from(D::Minus2)?;
    Ok(rope)
}

/// Contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmb;

impl candle::CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            if t == 1 {
                for bh_i in 0..b * h {
                    let off = bh_i * d;
                    for i_d in 0..d / 2 {
                        let i1 = off + i_d;
                        let i2 = i1 + d / 2;
                        let i_cs = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_d + b_i * d / 2
                        } else {
                            i_d
                        };
                        dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                        dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                    }
                }
            } else {
                src.par_chunks(t * d)
                    .zip(dst.par_chunks_mut(t * d))
                    .enumerate()
                    .for_each(|(bh_i, (src, dst))| {
                        for i_t in 0..t {
                            for i_d in 0..d / 2 {
                                let i1 = i_t * d + i_d;
                                let i2 = i1 + d / 2;
                                let i_cs = i_t * (d / 2) + i_d;
                                let i_cs = if unbatched_rope {
                                    let b_i = bh_i / h;
                                    i_cs + b_i * t * d / 2
                                } else {
                                    i_cs
                                };
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, (b * h) as u32, (t * d) as u32, d as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_f32",
            candle::DType::F16 => "rope_f16",
            candle::DType::BF16 => "rope_bf16",
            dtype => candle::bail!("rope is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, src.dtype())
            .with_label("rope")
            .build()?;
        candle_metal_kernels::call_rope(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b * h,
            t * d,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s1: &candle::RocmStorage,
        l1: &Layout,
        s2: &candle::RocmStorage,
        l2: &Layout,
        s3: &candle::RocmStorage,
        l3: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        use candle::rocm_backend::{RocmStorageSlice as S, SendSyncDeviceMemory};
        use candle::RocmDevice;

        fn inner<T: Copy + Send + Sync + 'static>(
            src: &SendSyncDeviceMemory<T>,
            l_src: &Layout,
            cos: &SendSyncDeviceMemory<T>,
            l_cos: &Layout,
            sin: &SendSyncDeviceMemory<T>,
            l_sin: &Layout,
            dev: &RocmDevice,
        ) -> Result<SendSyncDeviceMemory<T>> {
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let params = [(b * h) as u32, (t * d) as u32, d as u32, stride_b];
            let inputs = [(src, l_src), (cos, l_cos), (sin, l_sin)];
            rocm_rope_launch("rope", inputs, el, &params, dev)
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (S::BF16(s1), S::BF16(s2), S::BF16(s3)) => S::BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F16(s1), S::F16(s2), S::F16(s3)) => S::F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F32(s1), S::F32(s2), S::F32(s3)) => S::F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F64(s1), S::F64(s2), S::F64(s3)) => S::F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::rocm_backend::RocmStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }
}

pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmb)
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

pub fn rope_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _h, seq_len, _n_embd) = x.dims4()?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// T (seqlen)/H (num-heads)/D (head-dim) contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmbThd;

impl candle::CustomOp3 for RotaryEmbThd {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * h * d)
                .zip(dst.par_chunks_mut(t * h * d))
                .enumerate()
                .for_each(|(b_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            for i_h in 0..h {
                                let i1 = i_t * h * d + i_h * d + i_d;
                                let i2 = i1 + d / 2;
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, t, h, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_thd"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, b as u32, t as u32, h as u32, d as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope_thd");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_thd_f32",
            candle::DType::F16 => "rope_thd_f16",
            candle::DType::BF16 => "rope_thd_bf16",
            dtype => candle::bail!("rope_thd is not implemented for {dtype:?}"),
        };
        let (b, t, h, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, src.dtype())
            .with_label("rope_thd")
            .build()?;
        candle_metal_kernels::call_rope_thd(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b,
            t,
            h,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s1: &candle::RocmStorage,
        l1: &Layout,
        s2: &candle::RocmStorage,
        l2: &Layout,
        s3: &candle::RocmStorage,
        l3: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        use candle::rocm_backend::{RocmStorageSlice as S, SendSyncDeviceMemory};
        use candle::RocmDevice;

        fn inner<T: Copy + Send + Sync + 'static>(
            src: &SendSyncDeviceMemory<T>,
            l_src: &Layout,
            cos: &SendSyncDeviceMemory<T>,
            l_cos: &Layout,
            sin: &SendSyncDeviceMemory<T>,
            l_sin: &Layout,
            dev: &RocmDevice,
        ) -> Result<SendSyncDeviceMemory<T>> {
            let (b, t, h, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let params = [b as u32, t as u32, h as u32, d as u32, stride_b];
            let inputs = [(src, l_src), (cos, l_cos), (sin, l_sin)];
            rocm_rope_launch("rope_thd", inputs, el, &params, dev)
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (S::BF16(s1), S::BF16(s2), S::BF16(s3)) => S::BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F16(s1), S::F16(s2), S::F16(s3)) => S::F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F32(s1), S::F32(s2), S::F32(s3)) => S::F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (S::F64(s1), S::F64(s2), S::F64(s3)) => S::F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::rocm_backend::RocmStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }
}

pub fn rope_thd(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len, _n_head, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbThd)
}

#[cfg(all(test, feature = "rocm"))]
mod rocm_tests {
    use candle::{DType, Device, Result, Tensor};

    fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
        let lhs = lhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let rhs = rhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        (lhs - rhs)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()
    }

    /// The device tests in `tests/ops.rs` only exercise f32, so cover the other three
    /// dtypes the ROCm dispatch claims to support against the CPU reference.
    #[test]
    fn rope_dtype_coverage() -> Result<()> {
        let dev = Device::new_rocm(0)?;
        let (b, t, h, d) = (2, 4, 3, 8);
        let cpu = Device::Cpu;
        let src = Tensor::rand(0f32, 1f32, (b, t, h, d), &cpu)?;
        let cos = Tensor::rand(0f32, 1f32, (t, d / 2), &cpu)?;
        let sin = Tensor::rand(0f32, 1f32, (t, d / 2), &cpu)?;
        for (dtype, tol) in [
            (DType::BF16, 5e-2f32),
            (DType::F16, 5e-3),
            (DType::F32, 1e-5),
            (DType::F64, 1e-5),
        ] {
            let (src, cos, sin) = (
                src.to_dtype(dtype)?,
                cos.to_dtype(dtype)?,
                sin.to_dtype(dtype)?,
            );
            let (gsrc, gcos, gsin) = (
                src.to_device(&dev)?,
                cos.to_device(&dev)?,
                sin.to_device(&dev)?,
            );
            for (name, f) in [
                (
                    "rope",
                    super::rope as fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
                ),
                ("rope_i", super::rope_i),
                ("rope_thd", super::rope_thd),
            ] {
                let diff = max_abs_diff(&f(&gsrc, &gcos, &gsin)?, &f(&src, &cos, &sin)?)?;
                assert!(diff <= tol, "{name} {dtype:?}: max abs diff {diff} > {tol}");
            }
        }
        Ok(())
    }
}
