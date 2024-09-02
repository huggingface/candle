use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_wasm_example_sam as sam;
use wasm_bindgen::prelude::*;

struct Embeddings {
    original_width: u32,
    original_height: u32,
    width: u32,
    height: u32,
    data: Tensor,
}

#[wasm_bindgen]
pub struct Model {
    sam: sam::Sam,
    embeddings: Option<Embeddings>,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Vec<u8>, use_tiny: bool) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        let dev = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, dev)?;
        let sam = if use_tiny {
            sam::Sam::new_tiny(vb)? // tiny vit_t
        } else {
            sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)? // sam_vit_b
        };
        Ok(Self {
            sam,
            embeddings: None,
        })
    }

    pub fn set_image_embeddings(&mut self, image_data: Vec<u8>) -> Result<(), JsError> {
        sam::console_log!("image data: {}", image_data.len());
        let image_data = std::io::Cursor::new(image_data);
        let image = image::ImageReader::new(image_data)
            .with_guessed_format()?
            .decode()
            .map_err(candle::Error::wrap)?;
        let (original_height, original_width) = (image.height(), image.width());
        let (height, width) = (original_height, original_width);
        let resize_longest = sam::IMAGE_SIZE as u32;
        let (height, width) = if height < width {
            let h = (resize_longest * height) / width;
            (h, resize_longest)
        } else {
            let w = (resize_longest * width) / height;
            (resize_longest, w)
        };
        let image_t = {
            let img = image.resize_exact(width, height, image::imageops::FilterType::CatmullRom);
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &Device::Cpu,
            )?
            .permute((2, 0, 1))?
        };
        let data = self.sam.embeddings(&image_t)?;
        self.embeddings = Some(Embeddings {
            original_width,
            original_height,
            width,
            height,
            data,
        });
        Ok(())
    }

    pub fn mask_for_point(&self, input: JsValue) -> Result<JsValue, JsError> {
        let input: PointsInput =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        let transformed_points = input.points;

        for &(x, y, _bool) in &transformed_points {
            if !(0.0..=1.0).contains(&x) {
                return Err(JsError::new(&format!(
                    "x has to be between 0 and 1, got {}",
                    x
                )));
            }
            if !(0.0..=1.0).contains(&y) {
                return Err(JsError::new(&format!(
                    "y has to be between 0 and 1, got {}",
                    y
                )));
            }
        }
        let embeddings = match &self.embeddings {
            None => Err(JsError::new("image embeddings have not been set"))?,
            Some(embeddings) => embeddings,
        };
        let (mask, iou_predictions) = self.sam.forward_for_embeddings(
            &embeddings.data,
            embeddings.height as usize,
            embeddings.width as usize,
            &transformed_points,
            false,
        )?;
        let iou = iou_predictions.flatten(0, 1)?.to_vec1::<f32>()?[0];
        let mask_shape = mask.dims().to_vec();
        let mask_data = mask.ge(0f32)?.flatten_all()?.to_vec1::<u8>()?;
        let mask = Mask {
            iou,
            mask_shape,
            mask_data,
        };
        let image = Image {
            original_width: embeddings.original_width,
            original_height: embeddings.original_height,
            width: embeddings.width,
            height: embeddings.height,
        };
        Ok(serde_wasm_bindgen::to_value(&MaskImage { mask, image })?)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Mask {
    iou: f32,
    mask_shape: Vec<usize>,
    mask_data: Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct Image {
    original_width: u32,
    original_height: u32,
    width: u32,
    height: u32,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct MaskImage {
    mask: Mask,
    image: Image,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PointsInput {
    points: Vec<(f64, f64, bool)>,
}

fn main() {
    console_error_panic_hook::set_once();
}
