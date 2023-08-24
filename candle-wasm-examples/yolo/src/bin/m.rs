use candle_wasm_example_yolo::worker::Model as M;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: M,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<u8>) -> Result<Model, JsError> {
        let inner = M::load_(&data)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen]
    pub fn run(&self, image: Vec<u8>) -> Result<String, JsError> {
        let boxes = self.inner.run(image)?;
        let json = serde_json::to_string(&boxes)?;
        Ok(json)
    }
}

fn main() {}
