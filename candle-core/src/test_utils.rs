use crate::{BackendStorage, Result, Tensor};

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            use candle_core::backend::BackendDevice;
            $fn_name::<candle_core::CpuStorage>(&candle_core::cpu_backend::CpuDevice {})
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            use candle_core::backend::BackendDevice;
            $fn_name::<candle_core::CudaStorage>(&candle_core::CudaDevice::new(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            use candle_core::backend::BackendDevice;
            $fn_name::<candle_core::MetalStorage>(&candle_core::MetalDevice::new(0)?)
        }
    };
}

#[macro_export]
macro_rules! test_quantized_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            use candle_core::backend::BackendDevice;
            $fn_name::<candle_core::CpuStorage, candle_core::quantized::QCpuStorage>(
                &candle_core::cpu_backend::CpuDevice {},
            )
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            use candle_core::backend::BackendDevice;
            $fn_name::<candle_core::CudaStorage, candle_core::quantized::QCudaStorage>(
                &candle_core::CudaDevice::new(0)?,
            )
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            use candle_core::backend::BackendDevice;
            $fn_name::<candle_core::MetalStorage, candle_core::quantized::metal::QMetalStorage>(
                &candle_core::MetalDevice::new(0)?,
            )
        }
    };
}

pub fn assert_tensor_eq<B: BackendStorage>(t1: &Tensor<B>, t2: &Tensor<B>) -> Result<()> {
    assert_eq!(t1.shape(), t2.shape());
    // Default U8 may not be large enough to hold the sum (`t.sum_all` defaults to the dtype of `t`)
    let eq_tensor = t1.eq(t2)?.to_dtype(crate::DType::U32)?;
    let all_equal = eq_tensor.sum_all()?;
    assert_eq!(all_equal.to_scalar::<u32>()?, eq_tensor.elem_count() as u32);
    Ok(())
}

pub fn to_vec0_round<B: BackendStorage>(t: &Tensor<B>, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

pub fn to_vec1_round<B: BackendStorage>(t: &Tensor<B>, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

pub fn to_vec2_round<B: BackendStorage>(t: &Tensor<B>, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

pub fn to_vec3_round<B: BackendStorage>(t: &Tensor<B>, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
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
