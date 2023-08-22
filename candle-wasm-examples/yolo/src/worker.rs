use crate::model::{report, Bbox, Multiples, YoloV8};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use yew_agent::{HandlerId, Public, WorkerLink};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transfered via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub weights: Vec<u8>,
}

struct Model {
    model: YoloV8,
}

impl Model {
    fn run(
        &self,
        _link: &WorkerLink<Worker>,
        _id: HandlerId,
        image_data: Vec<u8>,
    ) -> Result<Vec<Vec<Bbox>>> {
        console_log!("image data: {}", image_data.len());
        let image_data = std::io::Cursor::new(image_data);
        let original_image = image::io::Reader::new(image_data)
            .with_guessed_format()?
            .decode()
            .map_err(candle::Error::wrap)?;
        let image = {
            let data = original_image
                .resize_exact(640, 640, image::imageops::FilterType::Triangle)
                .to_rgb8()
                .into_raw();
            Tensor::from_vec(data, (640, 640, 3), &Device::Cpu)?.permute((2, 0, 1))?
        };
        let image = (image.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = self.model.forward(&image)?.squeeze(0)?;
        console_log!("generated predictions {predictions:?}");
        let bboxes = report(&predictions, original_image, 640, 640)?;
        Ok(bboxes)
    }
}

impl Model {
    fn load(md: ModelData) -> Result<Self> {
        let dev = &Device::Cpu;
        let weights = safetensors::tensor::SafeTensors::deserialize(&md.weights)?;
        let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, dev);
        let model = YoloV8::load(vb, Multiples::s(), 80)?;
        Ok(Self { model })
    }
}

pub struct Worker {
    link: WorkerLink<Self>,
    model: Option<Model>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    Run(Vec<u8>),
}

#[derive(Serialize, Deserialize)]
pub enum WorkerOutput {
    ProcessingDone(std::result::Result<Vec<Vec<Bbox>>, String>),
    WeightsLoaded,
}

impl yew_agent::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = std::result::Result<WorkerOutput, String>;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Self { link, model: None }
    }

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        let output = match msg {
            WorkerInput::ModelData(md) => match Model::load(md) {
                Ok(model) => {
                    self.model = Some(model);
                    Ok(WorkerOutput::WeightsLoaded)
                }
                Err(err) => Err(format!("model creation error {err:?}")),
            },
            WorkerInput::Run(image_data) => match &mut self.model {
                None => Err("model has not been set yet".to_string()),
                Some(model) => {
                    let result = model
                        .run(&self.link, id, image_data)
                        .map_err(|e| e.to_string());
                    Ok(WorkerOutput::ProcessingDone(result))
                }
            },
        };
        self.link.respond(id, output);
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}
