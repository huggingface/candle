use candle::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Tensor};
use half::f16;

#[rustfmt::skip]
mod kernels {
    include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));
}

struct TiledAttnDecode {
    softmax_scale: f32,
}

impl CustomOp3 for TiledAttnDecode {
    fn name(&self) -> &'static str {
        "tiled-attn-decode"
    }

    fn cpu_fwd(
        &self,
        _q: &CpuStorage,
        _q_l: &Layout,
        _k: &CpuStorage,
        _k_l: &Layout,
        _v: &CpuStorage,
        _v_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("tiled_attn_decode requires CUDA tensors");
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
        k: &CudaStorage,
        k_l: &Layout,
        v: &CudaStorage,
        v_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !q_l.is_contiguous() || !k_l.is_contiguous() || !v_l.is_contiguous() {
            candle::bail!("tiled_attn_decode expects contiguous tensors");
        }

        let (b, h, q_len, d) = q_l.shape().dims4()?;
        let (_, _, k_len, _) = k_l.shape().dims4()?;
        let out_elems = b * h * q_len * d;
        let dev = q.device().clone();

        let q = q.as_cuda_slice::<f16>()?.slice(q_l.start_offset()..);
        let k = k.as_cuda_slice::<f16>()?.slice(k_l.start_offset()..);
        let v = v.as_cuda_slice::<f16>()?.slice(v_l.start_offset()..);
        let out = unsafe { dev.alloc::<f16>(out_elems) }?;

        let func = dev.get_or_load_custom_func(
            "tiled_attn_decode_f16_kernel",
            "tiled_attn",
            kernels::TILED_ATTN,
        )?;
        let cfg = LaunchConfig {
            grid_dim: ((b * h) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };

        let mut builder = func.builder();
        builder.arg(&out);
        builder.arg(&q);
        builder.arg(&k);
        builder.arg(&v);
        candle::builder_arg!(
            builder,
            self.softmax_scale,
            b as i32,
            h as i32,
            k_len as i32,
            d as i32
        );
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            CudaStorage::wrap_cuda_slice(out, dev),
            Shape::from((b, h, q_len, d)),
        ))
    }
}

pub fn tiled_attn_decode(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    q.apply_op3_no_bwd(k, v, &TiledAttnDecode { softmax_scale })
}
