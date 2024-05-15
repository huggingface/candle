use std::iter::zip;

#[allow(unused_imports)]
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
#[allow(dead_code)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    head_size: usize,
    cache: Tensor,
    is_gpt_neox: bool,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let theta_len = theta.len();
        let theta = Tensor::from_vec(theta, (1, theta_len), device)?.to_dtype(DType::F32)?;
        let idx_theta = Tensor::arange(0, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?
            .matmul(&theta)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self {
            head_size: head_dim,
            cos: if is_gpt_neox {
                Tensor::cat(
                    &[cos.clone().to_dtype(dtype)?, cos.clone().to_dtype(dtype)?],
                    D::Minus1,
                )?
            } else {
                cos.clone().to_dtype(dtype)?
            },
            sin: if is_gpt_neox {
                Tensor::cat(
                    &[sin.clone().to_dtype(dtype)?, sin.clone().to_dtype(dtype)?],
                    D::Minus1,
                )?
            } else {
                sin.clone().to_dtype(dtype)?
            },
            cache: Tensor::cat(&[cos.clone(), sin.clone()], D::Minus1)?
                .contiguous()?
                .to_dtype(dtype)?,
            is_gpt_neox,
        })
    }

    pub fn new_partial(
        base: f32,
        head_dim: usize,
        rot_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let theta: Vec<_> = (0..rot_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / rot_dim as f32))
            .collect();
        let theta_len = theta.len();
        let theta = Tensor::from_vec(theta, (1, theta_len), device)?.to_dtype(DType::F32)?;
        let idx_theta = Tensor::arange(0, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?
            .matmul(&theta)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self {
            head_size: head_dim,
            cos: if is_gpt_neox {
                Tensor::cat(
                    &[cos.clone().to_dtype(dtype)?, cos.clone().to_dtype(dtype)?],
                    D::Minus1,
                )?
            } else {
                cos.clone().to_dtype(dtype)?
            },
            sin: if is_gpt_neox {
                Tensor::cat(
                    &[sin.clone().to_dtype(dtype)?, sin.clone().to_dtype(dtype)?],
                    D::Minus1,
                )?
            } else {
                sin.clone().to_dtype(dtype)?
            },
            cache: Tensor::cat(&[cos.clone(), sin.clone()], D::Minus1)?
                .contiguous()?
                .to_dtype(dtype)?,
            is_gpt_neox,
        })
    }

    #[cfg(feature = "cuda")]
    fn execute_dtype<T: CudaDType + WithDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        q_storage: &CudaStorage,
        k_storage: &CudaStorage,
        q: &Tensor,
        k: &Tensor,
        cache_storage: &CudaStorage,
        pos_storage: &CudaStorage,
    ) -> Result<()> {
        use candle::cuda_backend::WrapErr;

        let num_tokens = q.dim(0)?;
        let rot_dim = self.cache.dim(1)?;
        let num_heads = q.dim(1)?;
        let num_kv_heads = k.dim(1)?;
        let q_stride = q.stride()[0];
        let k_stride = k.stride()[0];

        let func = dev.get_or_load_func(
            &if self.is_gpt_neox {
                kernel_name::<T>("rotary_embedding_kernel_neox")
            } else {
                kernel_name::<T>("rotary_embedding_kernel")
            },
            kernels::FUSED_ROPE,
        )?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (512.min((num_heads * rot_dim / 2) as u32), 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (
            pos_storage.as_cuda_slice::<i64>()?,
            q_storage.as_cuda_slice::<T>()?,
            k_storage.as_cuda_slice::<T>()?,
            cache_storage.as_cuda_slice::<T>()?,
            rot_dim as i32,
            q_stride as i64,
            k_stride as i64,
            num_heads as i32,
            num_kv_heads as i32,
            self.head_size as i32,
        );
        unsafe { func.launch(cfg, params) }.w()?;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn fused_rope(
        &self,
        dev: &CudaDevice,
        positions: &Tensor,
        q: &Tensor,
        k: &Tensor,
    ) -> Result<()> {
        let cache_type = self.cache.dtype();
        match (
            &*q.storage_and_layout().0,
            &*k.storage_and_layout().0,
            &*self.cache.storage_and_layout().0,
            &*positions.storage_and_layout().0,
        ) {
            (
                Storage::Cuda(q_storage),
                Storage::Cuda(k_storage),
                Storage::Cuda(cache_storage),
                Storage::Cuda(pos_storage),
            ) => {
                return match (q.dtype(), k.dtype(), cache_type) {
                    (DType::BF16, DType::BF16, DType::BF16) => self.execute_dtype::<half::bf16>(
                        &dev,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                        pos_storage,
                    ),
                    (DType::F16, DType::F16, DType::F16) => self.execute_dtype::<half::f16>(
                        &dev,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                        pos_storage,
                    ),
                    (DType::F32, DType::F32, DType::F32) => self.execute_dtype::<f32>(
                        &dev,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                        pos_storage,
                    ),
                    (DType::F64, DType::F64, DType::F64) => self.execute_dtype::<f64>(
                        &dev,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                        pos_storage,
                    ),
                    _ => candle::bail!(
                        "DType mismatch in fused RotaryEmbedding q={:?}, k={:?}, cache={:?}",
                        q.dtype(),
                        k.dtype(),
                        cache_type
                    ),
                }
            }
            _ => unreachable!(),
        };
    }

    /// This may modify the tensors in place!
    #[allow(unused_variables)]
    pub fn forward(
        &self,
        positions: &[usize],
        positions_kernel: &Tensor,
        q: &mut Tensor,
        k: &mut Tensor,
        b_sz: usize,
    ) -> Result<()> {
        match (q.device(), k.device()) {
            #[cfg(feature = "cuda")]
            (Device::Cuda(dev), Device::Cuda(_)) => {
                self.fused_rope(dev, positions_kernel, &*q, &*k)?;
            }

            _ => {
                *q = self.apply_rotary_emb(&*q, positions, b_sz)?;
                *k = self.apply_rotary_emb(&*k, positions, b_sz)?;
            }
        };
        Ok(())
    }

    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        b_sz: usize,
    ) -> Result<Tensor> {
        let (b_sz_seq_len, h, n_embd) = x.dims3()?;
        let x = x
            .reshape((b_sz, b_sz_seq_len / b_sz, h, n_embd))?
            .transpose(1, 2)?;

        fn rotate_half(xs: &Tensor) -> Result<Tensor> {
            let last_dim = xs.dim(D::Minus1)?;
            let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
            let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
            Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
        }
        let (b_sz, n_head, seq_len, _n_embd) = x.dims4()?;
        if self.is_gpt_neox {
            let mut embeds = Vec::new();
            for (b, seqlen_offset) in zip(0..b_sz, seqlen_offsets) {
                let cos = self.cos.narrow(0, *seqlen_offset, seq_len)?;
                let sin = self.sin.narrow(0, *seqlen_offset, seq_len)?;
                let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
                let sin = sin.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
                let x_b = x.i(b)?.unsqueeze(0)?;
                let embed = (x_b.broadcast_mul(&cos)? + rotate_half(&x_b)?.broadcast_mul(&sin)?)?;
                embeds.push(embed);
            }
            Tensor::cat(&embeds, 0)
        } else {
            let mut ropes = Vec::new();
            let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
            for (b, seqlen_offset) in zip(0..b_sz, seqlen_offsets) {
                let cos = self.cos.narrow(0, *seqlen_offset, seq_len)?.reshape((
                    seq_len,
                    n_embd / 2,
                    1,
                ))?;
                let sin = self.sin.narrow(0, *seqlen_offset, seq_len)?.reshape((
                    seq_len,
                    n_embd / 2,
                    1,
                ))?;
                let cos = cos.broadcast_as((1, 1, seq_len, n_embd / 2, 1))?;
                let sin = sin.broadcast_as((1, 1, seq_len, n_embd / 2, 1))?;
                // This mimics the llama.cpp behavior.
                // https://github.com/ggerganov/llama.cpp/blob/1f0bccb27929e261744c979bc75114955da49e98/ggml.c#L12104-L12105
                // The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
                // The resulting y0 and y1 are also interleaved with:
                //   y0 = x0*cos - x1*sin
                //   y1 = x0*sin + x1*cos
                let x_b = x.i(b)?.unsqueeze(0)?;
                let x_b = x_b.reshape((1, n_head, seq_len, n_embd / 2, 2))?;
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
}
