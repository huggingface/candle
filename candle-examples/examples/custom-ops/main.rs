#![allow(dead_code)]
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod cuda_kernels;

use clap::Parser;

use candle::backend::BackendStorage;
use candle::cpu_backend;
use candle::{CpuStorage, CustomOp1, DType, Device, Error, Layout, Result, Shape, Tensor};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

struct LayerNorm;

impl CustomOp1 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (dim1, dim2) = l.shape().dims2()?;
        let s = s.as_slice::<f32>()?;
        let src = match l.contiguous_offsets() {
            None => Err(Error::Wrapped("input has to be contiguous".into()))?,
            Some((o1, o2)) => &s[o1..o2],
        };
        let mut dst = Vec::with_capacity(dim1 * dim2);
        for idx1 in 0..dim1 {
            let src = &src[idx1 * dim2..(idx1 + 1) * dim2];
            let variance = src.iter().map(|x| x * x).sum::<f32>();
            let s_variance = 1f32 / (variance / dim2 as f32 + 1e-5).sqrt();
            dst.extend(src.iter().map(|x| x * s_variance))
        }
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::{cudarc, WrapErr};
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let (d1, d2) = l.shape().dims2()?;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        let dev = s.device().clone();
        let s = s.as_cuda_slice::<f32>()?;
        let s = match l.contiguous_offsets() {
            None => Err(Error::Wrapped("input has to be contiguous".into()))?,
            Some((o1, o2)) => s.slice(o1..o2),
        };
        let elem_count = l.shape().elem_count();
        let dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
        let func = dev.get_or_load_func("rms_f32", cuda_kernels::LAYERNORM_KERNELS)?;
        let params = (&dst, &s, 1e-5f32, d1, d2);
        let cfg = LaunchConfig {
            grid_dim: (d1, 1, 1),
            block_dim: (d2, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { func.launch(cfg, params) }.w()?;

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, l.shape().clone()))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let t = Tensor::arange(0f32, 14f32, &device)?.reshape((2, 7))?;
    println!("{t}");
    let t = t.custom_op1(LayerNorm)?;
    println!("{t}");
    Ok(())
}
