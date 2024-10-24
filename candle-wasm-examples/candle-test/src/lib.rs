use candle::{Result, Tensor};

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident, $test_wgpu: ident) => {
        #[test]
        async fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu).await
        }

        #[cfg(feature = "cuda")]
        #[tewasm_bindgen_testst]
        async fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?).await
        }

        #[cfg(feature = "metal")]
        #[test]
        async fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?).await
        }

        #[cfg(feature = "wgpu")]
        #[test]
        async fn $test_wgpu() -> Result<()> {
            let device = Device::new_wgpu(0).await?;
            $fn_name(&device).await
        }
    };
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        async fn $test_cpu() -> Result<()> {
            $fn_name(&Device::Cpu).await
        }

        #[cfg(feature = "cuda")]
        #[test]
        async fn $test_cuda() -> Result<()> {
            $fn_name(&Device::new_cuda(0)?).await
        }

        #[cfg(feature = "metal")]
        #[test]
        async fn $test_metal() -> Result<()> {
            $fn_name(&Device::new_metal(0)?).await
        }
    };
}

pub async fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0_async::<f32>().await?;
    Ok(f32::round(t * b) / b)
}

pub async fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1_async::<f32>().await?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

pub async fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2_async::<f32>().await?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

pub async fn to_vec3_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3_async::<f32>().await?;
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


pub trait ToVecRound{
    fn to_vec0_round(&self, digits: i32) -> impl std::future::Future<Output = Result<f32>>;
    fn to_vec1_round(&self, digits: i32) -> impl std::future::Future<Output = Result<Vec<f32>>>;
    fn to_vec2_round(&self, digits: i32) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>>>;
    fn to_vec3_round(&self, digits: i32) ->  impl std::future::Future<Output = Result<Vec<Vec<Vec<f32>>>>>;
}

impl ToVecRound for Tensor{
    async fn to_vec0_round(&self, digits: i32) -> Result<f32> {
        to_vec0_round(self, digits).await
    }

    async fn to_vec1_round(&self, digits: i32) -> Result<Vec<f32>> {
        to_vec1_round(self, digits).await
    }

    async fn to_vec2_round(&self, digits: i32) -> Result<Vec<Vec<f32>>> {
        to_vec2_round(self, digits).await
    }

    async fn to_vec3_round(&self, digits: i32) ->  Result<Vec<Vec<Vec<f32>>>> {
        to_vec3_round(self, digits).await
    }
}