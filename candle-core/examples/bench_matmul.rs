#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{DType, Device, Tensor};

const N: usize = 1_000;
const M: usize = 1_000;
const K: usize = 1_000;

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let lhs = Tensor::randn((N, K), DType::F32, &dev, 1., 0.)?;
    let rhs = Tensor::randn((K, M), DType::F32, &dev, 1., 0.)?;
    println!("generated tensors, starting the matmul loop");
    for idx in 0..100 {
        let start_time = std::time::Instant::now();
        let _prod = lhs.matmul(&rhs)?;
        println!("{idx} {:?}", start_time.elapsed());
    }
    Ok(())
}
