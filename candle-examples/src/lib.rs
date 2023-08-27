pub mod coco_classes;
pub mod imagenet;
pub mod object_detection;

use candle::{Device, Result, Tensor};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
    width: usize,
    height: usize,
) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    Tensor::from_vec(data, (width, height, 3), &Device::Cpu)?.permute((2, 0, 1))
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, width, height).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, width, height) = img.dims3()?;
    if channel != 3 {
        candle::bail!("save_image expects an input of shape (3, width, height)")
    }
    let img = img.transpose(0, 1)?.t()?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // NOTE: Waiting on https://github.com/rust-lang/mdBook/pull/1856
    #[rustfmt::skip]
    #[tokio::test]
    async fn book_hub_1() {
// ANCHOR: book_hub_1
use candle::Device;
use hf_hub::api::tokio::Api;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());

let weights_filename = repo.get("model.safetensors").await.unwrap();

let weights = candle::safetensors::load(weights_filename, &Device::Cpu).unwrap();
// ANCHOR_END: book_hub_1
        assert_eq!(weights.len(), 206);
    }

    #[rustfmt::skip]
    #[test]
    fn book_hub_2() {
// ANCHOR: book_hub_2
use candle::Device;
use hf_hub::api::sync::Api;
use memmap2::Mmap;
use std::fs;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());
let weights_filename = repo.get("model.safetensors").unwrap();

let file = fs::File::open(weights_filename).unwrap();
let mmap = unsafe { Mmap::map(&file).unwrap() };
let weights = candle::safetensors::load_buffer(&mmap[..], &Device::Cpu).unwrap();
// ANCHOR_END: book_hub_2
        assert_eq!(weights.len(), 206);
    }

    #[rustfmt::skip]
    #[test]
    fn book_hub_3() {
// ANCHOR: book_hub_3
use candle::{DType, Device, Tensor};
use hf_hub::api::sync::Api;
use memmap2::Mmap;
use safetensors::slice::IndexOp;
use safetensors::SafeTensors;
use std::fs;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());
let weights_filename = repo.get("model.safetensors").unwrap();

let file = fs::File::open(weights_filename).unwrap();
let mmap = unsafe { Mmap::map(&file).unwrap() };

// Use safetensors directly
let tensors = SafeTensors::deserialize(&mmap[..]).unwrap();
let view = tensors
    .tensor("bert.encoder.layer.0.attention.self.query.weight")
    .unwrap();

// We're going to load shard with rank 1, within a world_size of 4
// We're going to split along dimension 0 doing VIEW[start..stop, :]
let rank = 1;
let world_size = 4;
let dim = 0;
let dtype = view.dtype();
let mut tp_shape = view.shape().to_vec();
let size = tp_shape[0];

if size % world_size != 0 {
    panic!("The dimension is not divisble by `world_size`");
}
let block_size = size / world_size;
let start = rank * block_size;
let stop = (rank + 1) * block_size;

// Everything is expressed in tensor dimension
// bytes offsets is handled automatically for safetensors.

let iterator = view.slice(start..stop).unwrap();

tp_shape[dim] = block_size;

// Convert safetensors Dtype to candle DType
let dtype: DType = dtype.try_into().unwrap();

// TODO: Implement from_buffer_iterator so we can skip the extra CPU alloc.
let raw: Vec<u8> = iterator.into_iter().flatten().cloned().collect();
let tp_tensor = Tensor::from_raw_buffer(&raw, dtype, &tp_shape, &Device::Cpu).unwrap();
// ANCHOR_END: book_hub_3
        assert_eq!(view.shape(), &[768, 768]);
        assert_eq!(tp_tensor.dims(), &[192, 768]);
    }
}
