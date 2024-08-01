use candle::{CpuStorage, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

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
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .for_each(|(src, dst)| {
                    for i_over_2 in 0..t * d / 2 {
                        let i = 2 * i_over_2;
                        dst[i] = src[i] * cos[i_over_2] - src[i + 1] * sin[i_over_2];
                        dst[i + 1] = src[i] * sin[i_over_2] + src[i + 1] * cos[i_over_2];
                    }
                });
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
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
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
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_i"), kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el) }.w()?;
            let params = (&src, &cos, &sin, &dst, (b * h) as u32, (t * d) as u32);
            // SAFETY: ffi.
            unsafe { func.launch(cfg, params) }.w()?;
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
        let command_buffer = device.command_buffer()?;
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
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope-i")?;
        candle_metal_kernels::call_rope_i(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            b * h,
            t * d,
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
}

pub fn rope_i(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = cos.dims2()?;
    let (sin_seq_len, sin_n_embd) = cos.dims2()?;
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
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .for_each(|(src, dst)| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i1 = i_t * d + i_d;
                            let i2 = i1 + d / 2;
                            let i_cs = i_t * (d / 2) + i_d;
                            dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                            dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                        }
                    }
                });
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
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
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
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope"), kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el) }.w()?;
            let params = (
                &src,
                &cos,
                &sin,
                &dst,
                (b * h) as u32,
                (t * d) as u32,
                d as u32,
            );
            // SAFETY: ffi.
            unsafe { func.launch(cfg, params) }.w()?;
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
        let command_buffer = device.command_buffer()?;
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
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope-i")?;
        candle_metal_kernels::call_rope(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            b * h,
            t * d,
            d,
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
}

pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = cos.dims2()?;
    let (sin_seq_len, sin_n_embd) = sin.dims2()?;
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
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * h * d)
                .zip(dst.par_chunks_mut(t * h * d))
                .for_each(|(src, dst)| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i_cs = i_t * (d / 2) + i_d;
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
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
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
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_thd"), kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el) }.w()?;
            let params = (
                &src, &cos, &sin, &dst, b as u32, t as u32, h as u32, d as u32,
            );
            // SAFETY: ffi.
            unsafe { func.launch(cfg, params) }.w()?;
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
        let command_buffer = device.command_buffer()?;
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
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope-thd")?;
        candle_metal_kernels::call_rope_thd(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            b,
            t,
            h,
            d,
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
}

pub fn rope_thd(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, seq_len, _n_head, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = cos.dims2()?;
    let (sin_seq_len, sin_n_embd) = sin.dims2()?;
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
