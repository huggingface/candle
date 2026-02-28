use candle_wasm_example_yolo::coco_classes;
use candle_wasm_example_yolo::model::Bbox;
use candle_wasm_example_yolo::worker::Model as M;
use candle_wasm_example_yolo::worker::ModelPose as P;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: M,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub async fn new(data: Vec<u8>, model_size: &str, use_wgpu : bool) -> Result<Model, JsError> {
        let inner = M::load_(data, model_size, use_wgpu).await?;
        Ok(Self { inner })
    }

    #[wasm_bindgen]
    pub async fn run(
        &self,
        image: Vec<u8>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<String, JsError> {
        let bboxes = self.inner.run(image, conf_threshold, iou_threshold).await?;
        let mut detections: Vec<(String, Bbox)> = vec![];

        for (class_index, bboxes_for_class) in bboxes.into_iter().enumerate() {
            for b in bboxes_for_class.into_iter() {
                detections.push((coco_classes::NAMES[class_index].to_string(), b));
            }
        }
        let json = serde_json::to_string(&detections)?;
        Ok(json)
    }
}

#[wasm_bindgen]
pub struct ModelPose {
    inner: P,
}

#[wasm_bindgen]
impl ModelPose {
    #[wasm_bindgen(constructor)]
    pub async fn new(data: Vec<u8>, model_size: &str, use_wgpu : bool) -> Result<ModelPose, JsError> {
        let inner = P::load_(data, model_size, use_wgpu).await?;
        Ok(Self { inner })
    }

    #[wasm_bindgen]
    pub async fn run(
        &self,
        image: Vec<u8>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<String, JsError> {
        let bboxes = self.inner.run(image, conf_threshold, iou_threshold).await?;
        let json = serde_json::to_string(&bboxes)?;
        Ok(json)
    }
}

fn main() {}
