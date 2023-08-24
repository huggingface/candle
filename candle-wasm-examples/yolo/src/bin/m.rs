use candle_wasm_example_yolo::coco_classes;
use candle_wasm_example_yolo::model::Bbox;
use candle_wasm_example_yolo::worker::Model as M;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: M,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<u8>, model_size: &str) -> Result<Model, JsError> {
        let inner = M::load_(&data, model_size)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen]
    pub fn run(&self, image: Vec<u8>) -> Result<String, JsError> {
        let bboxes = self.inner.run(image)?;
        let mut detections: Vec<(String, Bbox)> = vec![];

        for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
            for b in bboxes_for_class.iter() {
                detections.push((
                    coco_classes::NAMES[class_index].to_string(),
                    Bbox {
                        xmin: b.xmin,
                        ymin: b.ymin,
                        xmax: b.xmax,
                        ymax: b.ymax,
                        confidence: b.confidence,
                    },
                ));
            }
        }
        let json = serde_json::to_string(&detections)?;
        Ok(json)
    }
}

fn main() {}
