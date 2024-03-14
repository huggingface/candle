use std::iter::zip;

use candle::{
    backend::BackendStorage, CudaDevice, CudaStorage, DType, Device, IndexOp, Module, Result,
    Storage, Tensor, WithDType, D,
};

#[cfg(feature = "cuda")]
use candle::cuda_backend::{
    cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig},
    kernel_name, kernels, CudaDType,
};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cache: Tensor,
    cos: Tensor,
    sin: Tensor,
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
        let idx_theta = Tensor::arange(0., max_position_embeddings as f64, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings as usize, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self {
            head_size: head_dim,
            cache: Tensor::cat(&[cos.clone(), sin.clone()], D::Minus1)?,
            cos,
            sin,
        })
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
        cache_storage: &CudaStorage,
    ) -> Result<()> {
        use candle::cuda_backend::WrapErr;

        let num_tokens = q.elem_count() / q.dim(D::Minus1)?;
        let rot_dim = self.cache.dim(1)?;
        let num_heads = q.dim(D::Minus1)? / self.head_size;
        let num_kv_heads = k.dim(D::Minus1)? / self.head_size;
        let q_stride = q.stride()[q.stride().len() - 2];
        let k_stride = k.stride()[k.stride().len() - 2];

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

        let params = (
            positions.as_ptr() as u64,
            q_storage.as_cuda_slice::<T>()?,
            k_storage.as_cuda_slice::<T>()?,
            cache_storage.as_cuda_slice::<T>()?,
            rot_dim,
            q_stride,
            k_stride,
            num_heads,
            num_kv_heads,
            self.head_size,
        );
        unsafe { func.launch(cfg, params) }.w()?;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn fused_rope(
        &self,
        dev: &CudaDevice,
        positions: &[usize],
        q: &Tensor,
        k: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match (
            &*q.storage_and_layout().0,
            &*k.storage_and_layout().0,
            &*self.cache.storage_and_layout().0,
        ) {
            (Storage::Cuda(q_storage), Storage::Cuda(k_storage), Storage::Cuda(cache_storage)) => {
                return match (cache_storage.dtype(), q.dtype(), k.dtype()) {
                    (DType::BF16, DType::BF16, DType::BF16) => self.execute_dtype::<half::bf16>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                    ),
                    (DType::F16, DType::F16, DType::F16) => self.execute_dtype::<half::f16>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                    ),
                    (DType::F32, DType::F32, DType::F32) => self.execute_dtype::<f32>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
                    ),
                    (DType::F64, DType::F64, DType::F64) => self.execute_dtype::<f64>(
                        &dev,
                        positions,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        cache_storage,
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
                self.fused_rope(dev, positions, &*q, &*k, is_neox);
            }

            _ => {
                *q = self.apply_rotary_emb(&*q, positions)?;
                *k = self.apply_rotary_emb(&*k, positions)?;
            }
        };
        *q = q.contiguous()?;
        *k = k.contiguous()?;
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
