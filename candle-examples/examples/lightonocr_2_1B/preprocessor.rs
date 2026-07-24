use candle::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use anyhow::Result;

const PATCH_SIZE: u32 = 14;
const MERGE_SIZE: u32 = 2;
const TILE_SIZE: u32 = PATCH_SIZE * MERGE_SIZE;

const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct PreprocessedImage {
    pub pixel_values: Tensor,
    pub ph: usize,
    pub pw: usize,
}

pub fn preprocess(img: &DynamicImage, max_edge: u32, device: &Device, dtype: DType) -> Result<PreprocessedImage> {
    let (orig_w, orig_h) = img.dimensions();
    let scale = max_edge as f32 / orig_w.max(orig_h) as f32;
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;
    let img = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);

    let pad_w = new_w.div_ceil(TILE_SIZE) * TILE_SIZE;
    let pad_h = new_h.div_ceil(TILE_SIZE) * TILE_SIZE;
    let mut padded = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(pad_w, pad_h);
    for y in 0..new_h {
        for x in 0..new_w {
            let px = img.get_pixel(x, y);
            padded.put_pixel(x, y, Rgb([px[0], px[1], px[2]]));
        }
    }

    let h = pad_h as usize;
    let w = pad_w as usize;
    let mut data = vec![0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let px = padded.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                data[c * h * w + y * w + x] = val;
            }
        }
    }

    let pixel_values = Tensor::from_vec(data, (3, h, w), device)?
        .unsqueeze(0)?
        .to_dtype(dtype)?;

    let ph = h / PATCH_SIZE as usize;
    let pw = w / PATCH_SIZE as usize;

    Ok(PreprocessedImage { pixel_values, ph, pw })
}
