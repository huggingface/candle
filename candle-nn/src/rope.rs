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
    pub fn new(cache: Tensor) -> Result<Self> {
        if !cache.device().is_cuda() {
            candle::bail!("Expected CUDA cache.");
        }
        Ok(Self { cache })
    }

    fn execute_dtype<T: CudaDType + WithDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        pos_storage: &CudaStorage,
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

        let params = (
            pos_storage.as_cuda_slice::<i64>()?,
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
        positions: Tensor,
        q: Tensor,
        k: Tensor,
        head_size: usize,
        is_neox: bool,
    ) -> Result<()> {
        match (
            &*positions.storage_and_layout().0,
            &*q.storage_and_layout().0,
            &*k.storage_and_layout().0,
            s & *self.cache.storage_and_layout().0,
        ) {
            (
                Storage::Cuda(pos_storage),
                Storage::Cuda(q_storage),
                Storage::Cuda(k_storage),
                Storage::Cuda(cache_storage),
            ) => {
                match (
                    cache_storage.dtype(),
                    positions.dtype(),
                    q.dtype(),
                    k.dtype(),
                ) {
                    (DType::BF16, DType::I64, DType::BF16, DType::BF16) => self
                        .execute_dtype::<half::bf16>(
                            &dev,
                            pos_storage,
                            q_storage,
                            k_storage,
                            q,
                            k,
                            head_size,
                        ),
                    (DType::F16, DType::I64, DType::F16, DType::F16) => self
                        .execute_dtype::<half::f16>(
                            &dev,
                            pos_storage,
                            q_storage,
                            k_storage,
                            q,
                            k,
                            head_size,
                        ),
                    (DType::F32, DType::I64, DType::F32, DType::F32) => self.execute_dtype::<f32>(
                        &dev,
                        pos_storage,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        head_size,
                    ),
                    (DType::F64, DType::I64, DType::F64, DType::F64) => self.execute_dtype::<f64>(
                        &dev,
                        pos_storage,
                        q_storage,
                        k_storage,
                        q,
                        k,
                        head_size,
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
        positions: Tensor,
        q: Tensor,
        k: Tensor,
        head_size: usize,
        is_neox: bool,
    ) -> Result<()> {
        match (positions.device(), q.device(), k.device()) {
            (Device::Cuda(dev), Device::Cuda(_), Device::Cuda(_)) => {
                return self.fused_rope(dev, positions, q, k, head_size, is_neox)
            }
            _ => candle::bail!("Expected a CUDA device."),
        };
    }
}
