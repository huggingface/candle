use candle::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::idefics3::model::{Idefics3Config, Idefics3VisionEmbeddings};
use hf_hub::{api::sync::Api, Repo, RepoType};

use crate::processing::Idefics3Processor;
mod processing;

fn main() {
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::new(
        "akshayballal/colSmol-256M-merged".to_string(),
        RepoType::Model,
    ));
    let config_file = repo.get("config.json").unwrap();
    let model_file = repo.get("model.safetensors").unwrap();
    let config: Idefics3Config =
        serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();

    let processor = Idefics3Processor::from_pretrained("akshayballal/colSmol-256M-merged").unwrap();

    let device = candle_examples::device(false).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file], candle::DType::F16, &device).unwrap()
    };
    let embedding_model = Idefics3VisionEmbeddings::load(config.vision_config, vb).unwrap();

    let image = image::open("/home/akshay/projects/EmbedAnything/test.jpg").unwrap();
    let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
        processor.preprocess(&[image], &Device::Cpu).unwrap();

    embedding_model
        .forward(
            pixel_values
                .to_device(&device)
                .unwrap()
                .get(0)
                .unwrap()
                .to_dtype(candle::DType::F16)
                .unwrap(),
            pixel_attention_mask
                .unwrap()
                .to_device(&device)
                .unwrap()
                .get(0)
                .unwrap(),
            &device,
        )
        .unwrap();
}

fn unfold_1d(x: &candle::Tensor, size: usize, step: usize) -> candle::Result<candle::Tensor> {
    let len = x.dim(0)? as usize;
    let num = (len - size) / step + 1;

    // Build index data: for each window i, indices [i*step .. i*step + size)
    let mut idx_data = Vec::with_capacity(num * size);
    for i in 0..num {
        let base = i * step;
        for j in 0..size {
            idx_data.push((base + j) as i64);
        }
    }
    let idx_shape = [num, size];
    let idx = candle::Tensor::from_vec(idx_data, &idx_shape, x.device())?;

    let x = x.repeat(&[num, 1])?;
    let out = x.gather(&idx, 1)?;
    Ok(out)
}

fn unfold(x: &candle::Tensor, size: usize, step: usize, dim: usize) -> candle::Result<candle::Tensor> {
    let x_shape = x.shape().dims().to_vec();
    let len = x_shape[dim] as usize;
    let num = (len - size) / step + 1;

    // Build index data for the unfolding dimension
    let mut idx_data = Vec::with_capacity(num * size);
    for i in 0..num {
        let base = i * step;
        for j in 0..size {
            idx_data.push((base + j) as i64);
        }
    }

    // Create permutation to move the unfolding dimension to the end
    let mut perm: Vec<usize> = (0..x_shape.len()).filter(|&i| i != dim).collect();
    perm.push(dim);
    let perm_clone = perm.clone();
    let x = x.permute(perm)?;

    // Create inverse permutation to restore original order
    let mut inv_perm = vec![0; x_shape.len()];
    for (i, &p) in perm_clone.iter().enumerate() {
        inv_perm[p] = i;
    }
    inv_perm.push(x_shape.len());

    // Create index tensor with proper broadcasting
    // Build the target shape: [num, other_dims..., size]
    let mut idx_shape = vec![num];
    for i in 0..x_shape.len() {
        if i != dim {
            idx_shape.push(x_shape[i] as usize);
        }
    }
    idx_shape.push(size);
    
    // Create index tensor and reshape for broadcasting
    let idx = candle::Tensor::from_vec(idx_data, &[num, size], x.device())?;
    
    // Build reshape dimensions: [num, 1s for other dims, size]
    let mut reshape_dims = vec![num];
    for i in 0..x_shape.len() {
        if i != dim {
            reshape_dims.push(1);
        }
    }
    reshape_dims.push(size);
    
    let reshape_dims: &[usize] = &reshape_dims;
    let idx = idx.reshape(reshape_dims)?.broadcast_as(&idx_shape[..])?.contiguous()?;

    // Repeat x for each window
    let mut repeat_dims = vec![1; x_shape.len()];
    repeat_dims[0] = num;
    let x = x.unsqueeze(0)?.repeat(repeat_dims)?;

    println!("x shape: {}", x);
    println!("idx shape: {}", idx);
    
    // Gather and permute back to original order
    let x_i = x.gather(&idx, x_shape.len())?;
    let x_i = x_i.permute(inv_perm)?;
    Ok(x_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unfold_1d() {
        let x = candle::Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
            &Device::Cpu,
        )
        .unwrap();
        let out = unfold_1d(&x, 3, 2).unwrap();
        println!("{}", out);
        let expected = candle::Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0],
            &[4, 3],
            &Device::Cpu
        ).unwrap();
        assert!(out.broadcast_sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);
    }

    #[test]
    fn test_unfold() {
        // Test 2D case (dim=0)
        // let x = candle::Tensor::from_vec(
        //     vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        //     &[3, 4],
        //     &Device::Cpu,
        // )
        // .unwrap();
        // let out = unfold(&x, 2, 1, 0).unwrap();
        // println!("2D test - input shape: {:?}, output shape: {:?}", x.shape(), out.shape());
        // println!("2D test - output: {}", out);

        // Test 3D case (dim=1)
        let x = candle::Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            &[2, 3, 4],
            &Device::Cpu,
        )
        .unwrap();
        let out = unfold(&x, 2, 1, 1).unwrap();
        println!("3D test - input shape: {:?}, output shape: {:?}", x.shape(), out.shape());
        println!("3D test - output: {}", out);
    }
}
