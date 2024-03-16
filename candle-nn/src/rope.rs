use std::iter::zip;

use candle::{
    backend::BackendStorage, CudaDevice, CudaStorage, DType, Device, IndexOp, Module, Result,
    Storage, Tensor, WithDType, D,
};

#[cfg(feature = "cuda")]
use candle::cuda_backend::{
    cudarc::driver::{
        CudaFunction, CudaStream, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
    },
    kernel_name, kernels, CudaDType,
};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    cos_unsqz: Tensor,
    sin_unsqz: Tensor,
    head_size: usize,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
    ) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self {
            head_size: head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            cos_unsqz: cos.unsqueeze(1)?.unsqueeze(1)?,
            sin_unsqz: sin.unsqueeze(1)?.unsqueeze(1)?,
        })
    }

    fn run_kernel<T: CudaDType + WithDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        input: &Tensor,
        inp_storage: &CudaStorage,
        cos_storage: &CudaStorage,
        sin_storage: &CudaStorage,
        pos_storage: &CudaStorage,
    ) -> Result<Tensor> {
        use candle::{cuda_backend::WrapErr, from_storage_no_op};

        let (s, b, h, d) = input.dims4()?;
        let stride_s = input.stride()[0];
        let stride_b = input.stride()[1];
        let stride_h = input.stride()[2];
        let stride_d = input.stride()[3];

        let d2 = self.cos_unsqz.dims()[3];

        let output = unsafe { dev.alloc::<T>(s * b * h * d) }.w()?; // this will be the same as input

        let out = from_storage_no_op(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
            (s, b, h, d),
            false,
        );

        let o_stride_s = out.stride()[0];
        let o_stride_b = out.stride()[1];
        let o_stride_h = out.stride()[2];
        let o_stride_d = out.stride()[3];

        let func = dev.get_or_load_func(
            &kernel_name::<T>("rotary_embedding_kernel"),
            kernels::FUSED_ROPE,
        )?;

        const WARP_SIZE: u32 = 32;

        let cfg = LaunchConfig {
            grid_dim: (s as u32, b as u32, 1),
            block_dim: (WARP_SIZE, if h < 16 { 4 } else { 8 }, 1),
            shared_mem_bytes: 0,
        };

        {
            let bdg = out.storage_and_layout();
            let out_storage = match &*bdg.0 {
                Storage::Cuda(storage) => storage,
                _ => {
                    unreachable!();
                }
            };

            let strides_s = [stride_s as i64, o_stride_s as i64];
            let strides_b = [stride_b as i64, o_stride_b as i64];
            let strides_h = [stride_h as i64, o_stride_h as i64];
            let strides_d = [stride_d as i64, o_stride_d as i64];

            let strides_s = Tensor::new(strides_s.as_slice(), input.device())?;
            let bdg = strides_s.storage_and_layout();
            let strides_s_storage = match &*bdg.0 {
                Storage::Cuda(storage) => storage,
                _ => {
                    unreachable!();
                }
            };

            let strides_b = Tensor::new(strides_b.as_slice(), input.device())?;
            let bdg = strides_b.storage_and_layout();
            let strides_b_storage = match &*bdg.0 {
                Storage::Cuda(storage) => storage,
                _ => {
                    unreachable!();
                }
            };

            let strides_h: Tensor = Tensor::new(strides_h.as_slice(), input.device())?;
            let bdg = strides_h.storage_and_layout();
            let strides_h_storage = match &*bdg.0 {
                Storage::Cuda(storage) => storage,
                _ => {
                    unreachable!();
                }
            };

            let strides_d = Tensor::new(strides_d.as_slice(), input.device())?;
            let bdg = strides_d.storage_and_layout();
            let strides_d_storage = match &*bdg.0 {
                Storage::Cuda(storage) => storage,
                _ => {
                    unreachable!();
                }
            };

            let params = (
                h as i32,
                d as i32,
                d2 as i32,
                strides_s_storage.as_cuda_slice::<i64>()?,
                strides_b_storage.as_cuda_slice::<i64>()?,
                strides_h_storage.as_cuda_slice::<i64>()?,
                strides_d_storage.as_cuda_slice::<i64>()?,
                inp_storage.as_cuda_slice::<T>()?,
                cos_storage.as_cuda_slice::<f32>()?,
                sin_storage.as_cuda_slice::<f32>()?,
                out_storage.as_cuda_slice::<T>()?,
                pos_storage.as_cuda_slice::<i64>()?,
            );
            unsafe { func.launch(cfg, params) }.w()?;
        }

        dbg!(input.mean_all()?);
        dbg!(out.mean_all()?);
        dbg!(input.shape());
        dbg!(out.shape());

        // shape: (seqlen, bs, heads, head_dim)
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn execute_dtype<T: CudaDType + WithDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        positions: &[usize],
        q_storage: &CudaStorage,
        k_storage: &CudaStorage,
        q: &Tensor,
        k: &Tensor,
        cos_storage: &CudaStorage,
        sin_storage: &CudaStorage,
    ) -> Result<(Tensor, Tensor)> {
        use candle::{cuda_backend::WrapErr, from_storage_no_op};

        let positions = positions.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let positions = Tensor::new(positions.as_slice(), q.device())?;
        let bdg = positions.storage_and_layout();
        let pos_storage = match &*bdg.0 {
            Storage::Cuda(storage) => storage,
            _ => {
                unreachable!();
            }
        };

        Ok((
            self.run_kernel::<T>(dev, q, q_storage, cos_storage, sin_storage, pos_storage)?,
            self.run_kernel::<T>(dev, k, k_storage, cos_storage, sin_storage, pos_storage)?,
        ))

        /*use candle::cuda_backend::WrapErr;

        let num_tokens = q.elem_count() / q.dim(D::Minus1)?;
        let rot_dim = self.cache.dim(1)?;
        let num_heads = q.dim(D::Minus1)? / self.head_size;
        let num_kv_heads = k.dim(D::Minus1)? / self.head_size;
        let q_stride = q.stride()[q.stride().len() - 2];
        let k_stride = k.stride()[k.stride().len() - 2];

        dbg!(num_heads);
        dbg!(num_kv_heads);
        dbg!(q_stride);
        dbg!(k_stride);

        let func = dev.get_or_load_func(
            &kernel_name::<T>("rotary_embedding_kernel_neox"),
            kernels::FUSED_ROPE,
        )?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (512.min((num_heads * rot_dim / 2) as u32), 1, 1),
            shared_mem_bytes: 0,
        };

        let positions = positions.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let positions = Tensor::new(positions.as_slice(), q.device())?;
        let bdg = positions.storage_and_layout();
        let pos_storage = match  &*bdg.0{
            Storage::Cuda(storage) => {
                storage
            }
            _ => {
                unreachable!();
            }
        };
        let params = (
            pos_storage.as_cuda_slice::<i64>()?,
            q_storage.as_cuda_slice::<T>()?,
            k_storage.as_cuda_slice::<T>()?,
            cache_storage.as_cuda_slice::<f32>()?,
            rot_dim,
            q_stride,
            k_stride,
            num_heads,
            num_kv_heads,
            self.head_size,
        );
        unsafe { func.launch(cfg, params) }.w()?;

        Ok(())*/
    }

    #[cfg(feature = "cuda")]
    fn fused_rope(
        &self,
        dev: &CudaDevice,
        positions: &[usize],
        q: &Tensor,
        k: &Tensor,
        is_neox: bool,
    ) -> Result<(Tensor, Tensor)> {
        match (
            &*q.storage_and_layout().0,
            &*k.storage_and_layout().0,
            &*self.cos_unsqz.storage_and_layout().0,
            &*self.sin_unsqz.storage_and_layout().0,
        ) {
            (
                Storage::Cuda(q_storage),
                Storage::Cuda(k_storage),
                Storage::Cuda(cos_storage),
                Storage::Cuda(sin_storage),
            ) => {
                return match (q.dtype(), k.dtype()) {
                    (DType::BF16, DType::BF16) => self.execute_dtype::<half::bf16>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cos_storage,
                        sin_storage,
                    ),
                    (DType::F16, DType::F16) => self.execute_dtype::<half::f16>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cos_storage,
                        sin_storage,
                    ),
                    (DType::F32, DType::F32) => self.execute_dtype::<f32>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cos_storage,
                        sin_storage,
                    ),
                    (DType::F64, DType::F64) => self.execute_dtype::<f64>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cos_storage,
                        sin_storage,
                    ),
                    _ => candle::bail!("DType mismatch in fused RotaryEmbedding"),
                }
            }
            _ => unreachable!(),
        }
    }

    /// This may modify the tensors in place!
    pub fn forward(
        &self,
        positions: &[usize],
        q: &mut Tensor,
        k: &mut Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match (q.device(), k.device()) {
            #[cfg(feature = "cuda")]
            (Device::Cuda(dev), Device::Cuda(_)) => {
                // input is (bs, seqlen, num_head, head_dim)
                // want (seqlen, bs, num_head, head_dim)
                let in_q = q.permute((1, 0, 2, 3))?;
                let in_k = k.permute((1, 0, 2, 3))?;
                let (new_q, new_k) = self.fused_rope(dev, positions, &in_q, &in_k, is_neox)?;
                // output is (seqlen, bs, num_head, head_dim)
                // want (bs, seqlen, num_head, head_dim)
                let new_q = new_q.permute((1, 0, 2, 3))?;
                let new_k = new_k.permute((1, 0, 2, 3))?;
                *q = new_q;
                *k = new_k;
            }

            _ => {
                *q = self.apply_rotary_emb(&*q, positions)?;
                *k = self.apply_rotary_emb(&*k, positions)?;
            }
        };
        Ok(())
    }

    fn apply_rotary_emb(&self, x: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
        let mut ropes = Vec::new();
        let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
        for (b, seqlen_offset) in zip(0..b_sz, seqlen_offsets) {
            let cos =
                self.cos
                    .narrow(0, *seqlen_offset, seq_len)?
                    .reshape((seq_len, n_embd / 2, 1))?;
            let sin =
                self.sin
                    .narrow(0, *seqlen_offset, seq_len)?
                    .reshape((seq_len, n_embd / 2, 1))?;
            let cos = cos.broadcast_as((1, 1, seq_len, n_embd / 2, 1))?;
            let sin = sin.broadcast_as((1, 1, seq_len, n_embd / 2, 1))?;
            // This mimics the llama.cpp behavior.
            // https://github.com/ggerganov/llama.cpp/blob/1f0bccb27929e261744c979bc75114955da49e98/ggml.c#L12104-L12105
            // The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
            // The resulting y0 and y1 are also interleaved with:
            //   y0 = x0*cos - x1*sin
            //   y1 = x0*sin + x1*cos
            let x_b = x.i(b)?.unsqueeze(0)?;
            let x0 = x_b.narrow(D::Minus1, 0, 1)?;
            let x1 = x_b.narrow(D::Minus1, 1, 1)?;
            let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
            let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
            let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
            let rope = rope.flatten_from(D::Minus2)?;
            ropes.push(rope);
        }
        Tensor::cat(&ropes, 0)
    }
}
