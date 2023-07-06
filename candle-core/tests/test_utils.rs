#![allow(dead_code)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle::{Result, Tensor};

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?)
        }
    };
}

pub fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}
