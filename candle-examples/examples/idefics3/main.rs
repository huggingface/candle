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
    println!("x_repeat: {}", x);
    println!("idx: {}", idx);
    let out = x.gather(&idx, 1)?;
    Ok(out)
}

// Helper: apply unfold_1d to each row= of a 2D tensor and stack results
fn unfold_1d_batched(x: &candle::Tensor, size: usize, step: usize) -> candle::Result<candle::Tensor> {
    let (batch, len) = {
        let shape = x.shape().dims();
        (shape[0], shape[1])
    };
    let mut outs = Vec::with_capacity(batch);
    for i in 0..batch {
        let row = x.get(i)?;
        let unfolded = unfold_1d(&row, size, step)?;
        outs.push(unfolded);
    }
    candle::Tensor::stack(&outs, 0)
}

fn unfold(x: &candle::Tensor, size: usize, step: usize, dim: usize) -> candle::Result<candle::Tensor> {
    
    // if dim == 0 {
    //     return unfold_1d(x, size, step);
    // }

    let x_shape = x.shape().dims().to_vec();
    let len = x_shape[dim] as usize;
    let num = (len - size) / step + 1;

    println!("num: {}", num);

    // Build index data: for each window i, indices [i*step .. i*step + size)
    let mut idx_data = Vec::with_capacity(num * size);
    for i in 0..num {
        let base = i * step;
        for j in 0..size {
            idx_data.push((base + j) as i64);
        }
    }

    
    let idx_shape = [num, size];
    let idx = candle::Tensor::from_vec(idx_data, &idx_shape, x.device())?.broadcast_as(&[2,2,4]).unwrap();

    let x = x.unsqueeze(0).unwrap().repeat(&[num, 1,1]).unwrap();
    println!("x_repeat: {}", x);
    println!("idx: {}", idx);

    let x_i = x.gather(&idx, 1).unwrap();

    println!("x_gather: {}", x_i);
    Ok(x)
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
        // Test 1D case (dim=0)
        // let x = candle::Tensor::from_vec(
        //     vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        //     &[10],
        //     &Device::Cpu,
        // )
        // .unwrap();
        // let out = unfold(&x, 3, 2, 0).unwrap();
        // let expected = candle::Tensor::from_vec(
        //     vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0],
        //     &[4, 3],
        //     &Device::Cpu
        // ).unwrap();
        // assert!(out.broadcast_sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);

        // Test 2D case (dim=1)
        let x = candle::Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 4],
            &Device::Cpu,
        )
        .unwrap();
        println!("x: {}", x);
        let out = unfold(&x, 2, 1, 0).unwrap();
        // Expected shape: [3, 3, 2] where:
        // - First window: [[1,2], [5,6], [9,10]]
        // - Second window: [[2,3], [6,7], [10,11]]
        // - Third window: [[3,4], [7,8], [11,12]]
        // let expected = candle::Tensor::from_vec(
        //     vec![
        //         1.0f32, 2.0, 3.0, 5.0, 6.0, 7.0,  // First window
        //         8.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Second window
        //         10.0,11.0, 12.0, // Third window
        //     ],
        //     &[2, 4, 2],
        //     &Device::Cpu,
        // ).unwrap();
        // println!("out: {}", out);
        // println!("expected: {}", expected);
        // println!("out values: {:?}", out.to_vec3::<f32>().unwrap());
        // println!("expected values: {:?}", expected.to_vec3::<f32>().unwrap());
        // assert!(out.broadcast_sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);
    }
}
