#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{Device, Result, Tensor};

#[test]
fn kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);
    let data = cache.current_data()?;
    assert!(data.is_none());
    let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    cache.append(&t)?;
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3.]);
    let t = Tensor::new(&[4f32], &Device::Cpu)?;
    cache.append(&t)?;
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4.]);
    let t = Tensor::new(&[0f32, 5., 6., 7.], &Device::Cpu)?;
    cache.append(&t)?;
    let data = cache.current_data()?.unwrap();
    assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 0., 5., 6., 7.]);
    Ok(())
}
