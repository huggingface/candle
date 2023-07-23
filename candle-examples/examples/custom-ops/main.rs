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
        let s = s.as_slice::<f32>()?;
        let _s = match l.contiguous_offsets() {
            None => Err(Error::Wrapped("input has to be contiguous".into()))?,
            Some((o1, o2)) => &s[o1..o2],
        };
        todo!()
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::{cudarc, WrapErr};
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let dev = s.device().clone();
        let s = s.as_cuda_slice::<f32>()?;
        let s = match l.contiguous_offsets() {
            None => Err(Error::Wrapped("input has to be contiguous".into()))?,
            Some((o1, o2)) => s, // TODO: slice with o1 and o2
        };
        let s = s.try_clone().w()?;
        let elem_count = l.shape().elem_count();
        let dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
        let func = dev.get_or_load_func("fill_u8", cuda_kernels::LAYERNORM_KERNELS)?;
        let params = (&dst, elem_count);
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
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
