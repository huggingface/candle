use candle::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig},
        kernel_name, kernels, CudaDType,
    },
    CudaDevice, CudaStorage, DType, Device, Module, Result, Storage, Tensor, WithDType, D,
};

pub struct RotaryEmbedding {
    cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(base: f32, head_dim: usize, max_position_embeddings: usize) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, max_position_embeddings, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings as usize, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self {
            cache: Tensor::cat(&[cos, sin], D::Minus1)?,
        })
    }

    fn execute_dtype<T: CudaDType + WithDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        positions: &[usize],
        q_storage: &CudaStorage,
        k_storage: &CudaStorage,
        q: Tensor,
        k: Tensor,
        head_size: usize,
    ) -> Result<()> {
        let num_tokens = q.elem_count() / q.dim(D::Minus1)?;
        let rot_dim = self.cache.dim(1)?;
        let num_heads = q.dim(D::Minus1)? / head_size;
        let num_kv_heads = k.dim(D::Minus1)? / head_size;
        let q_stride = q.stride()[q.stride().len() - 2];
        let k_stride = k.stride()[k.stride().len() - 2];

        let func = dev.get_or_load_func(
            &kernel_name::<T>("rotary_embedding_kernel"),
            kernels::FUSED_ROPE,
        )?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens, 1, 1),
            block_dim: (512.min(num_heads * rot_dim / 2), 1, 1),
            shared_mem_bytes: 0,
        };

        let positions = positions.iter().map(|x| x as i64).collect::<Vec<_>>();

        let params = (
            positions.as_ptr(),
            q_storage.as_cuda_slice::<T>()?,
            k_storage.as_cuda_slice::<T>()?,
            cache_storage.as_cuda_slice::<T>()?,
            rot_dim,
            q_stride,
            k_stride,
            num_heads,
            num_kv_heads,
            head_size,
        );
        unsafe { func.launch(cfg, params) }.w()?;

        Ok(())
    }

    fn fused_rope(
        &self,
        dev: CudaDevice,
        positions: &[usize],
        q: Tensor,
        k: Tensor,
        head_size: usize,
        is_neox: bool,
    ) -> Result<()> {
        match (
            &*q.storage_and_layout().0,
            &*k.storage_and_layout().0,
            &*self.cache.storage_and_layout().0,
        ) {
            (Storage::Cuda(q_storage), Storage::Cuda(k_storage), Storage::Cuda(cache_storage)) => {
                match (cache_storage.dtype(), q.dtype(), k.dtype()) {
                    (DType::BF16, DType::BF16, DType::BF16) => self.execute_dtype::<half::bf16>(
                        &dev, positions, q_storage, k_storage, q, k, head_size,
                    ),
                    (DType::F16, DType::F16, DType::F16) => self.execute_dtype::<half::f16>(
                        &dev, positions, q_storage, k_storage, q, k, head_size,
                    ),
                    (DType::F32, DType::F32, DType::F32) => self.execute_dtype::<f32>(
                        &dev, positions, q_storage, k_storage, q, k, head_size,
                    ),
                    (DType::F64, DType::F64, DType::F64) => self.execute_dtype::<f64>(
                        &dev, positions, q_storage, k_storage, q, k, head_size,
                    ),
                    _ => candle::bail!("DType mismatch in fused RotaryEmbedding"),
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub fn apply_forward(
        &self,
        positions: &[usize],
        q: Tensor,
        k: Tensor,
        head_size: usize,
        is_neox: bool,
    ) -> Result<()> {
        match (q.device(), k.device()) {
            (Device::Cuda(dev), Device::Cuda(_)) => {
                return self.fused_rope(dev, positions, q, k, head_size, is_neox)
            }
            _ => candle::bail!("Expected a CUDA device."),
        };
    }
}
