use candle::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Preprocessor {
   pub patch_size: u32,
   pub merge_size: u32,
   pub max_edge: u32,
   pub mean: [f32; 3],
   pub std: [f32; 3],
}

impl Preprocessor{
    pub fn new(
        patch_size: u32,
        merge_size: u32,
        max_edge: u32,
        mean: [f32; 3],
        std: [f32; 3]
    ) -> Self{
        Self{ 
            patch_size,
            merge_size,
            max_edge,
            mean,
            std
        }
    }

    /// Normalize image tensor using mean and std values.
    pub fn preprocess(&self, img: &Tensor) -> Result<Tensor> {
        let img = if img.dims().len() == 3 {
            img.unsqueeze(0)?
        } else {
            img.clone()
        };
        
        // Create mean and std tensors
        let mean = Tensor::new(self.mean.as_slice(), img.device())?
            .reshape((1, 3, 1, 1))?;
        let std = Tensor::new(self.std.as_slice(), img.device())?
            .reshape((1, 3, 1, 1))?;

        // Normalize: (img - mean) / std
        let normalized = (img.broadcast_sub(&mean)?)
            .broadcast_div(&std)?;

        Ok(normalized)
    }
}
