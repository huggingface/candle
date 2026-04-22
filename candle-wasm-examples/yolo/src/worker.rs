use crate::model::{report_detect, report_pose, Bbox, Multiples, YoloV8, YoloV8Pose};
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
// on the main thread and transferred via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub weights: Vec<u8>,
    pub model_size: String,
}

#[derive(Serialize, Deserialize)]
pub struct RunData {
    pub image_data: Vec<u8>,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
}

pub struct Model {
    model: YoloV8,
}

impl Model {
    pub fn run(
        &self,
        image_data: Vec<u8>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Vec<Bbox>>> {
        console_log!("image data: {}", image_data.len());
        let image_data = std::io::Cursor::new(image_data);
        let original_image = image::ImageReader::new(image_data)
            .with_guessed_format()?
            .decode()
            .map_err(candle::Error::wrap)?;
        let (width, height) = {
            let w = original_image.width() as usize;
            let h = original_image.height() as usize;
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };
        let image_t = {
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &Device::Cpu,
            )?
            .permute((2, 0, 1))?
        };
        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = self.model.forward(&image_t)?.squeeze(0)?;
        console_log!("generated predictions {predictions:?}");
        let bboxes = report_detect(
            &predictions,
            original_image,
            width,
            height,
            conf_threshold,
            iou_threshold,
        )?;
        Ok(bboxes)
    }

    pub fn load_(weights: Vec<u8>, model_size: &str) -> Result<Self> {
        let multiples = match model_size {
            "n" => Multiples::n(),
            "s" => Multiples::s(),
            "m" => Multiples::m(),
            "l" => Multiples::l(),
            "x" => Multiples::x(),
            _ => Err(candle::Error::Msg(
                "invalid model size: must be n, s, m, l or x".to_string(),
            ))?,
        };
        let dev = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, dev)?;
        let model = YoloV8::load(vb, multiples, 80)?;
        Ok(Self { model })
    }

    pub fn load(md: ModelData) -> Result<Self> {
        Self::load_(md.weights, &md.model_size.to_string())
    }
}

pub struct ModelPose {
    model: YoloV8Pose,
}

impl ModelPose {
    pub fn run(
        &self,
        image_data: Vec<u8>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Bbox>> {
        console_log!("image data: {}", image_data.len());
        let image_data = std::io::Cursor::new(image_data);
        let original_image = image::ImageReader::new(image_data)
            .with_guessed_format()?
            .decode()
            .map_err(candle::Error::wrap)?;
        let (width, height) = {
            let w = original_image.width() as usize;
            let h = original_image.height() as usize;
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };
        let image_t = {
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &Device::Cpu,
            )?
            .permute((2, 0, 1))?
        };
        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = self.model.forward(&image_t)?.squeeze(0)?;
        console_log!("generated predictions {predictions:?}");
        let bboxes = report_pose(
            &predictions,
            original_image,
            width,
            height,
            conf_threshold,
            iou_threshold,
        )?;
        Ok(bboxes)
    }

    pub fn load_(weights: Vec<u8>, model_size: &str) -> Result<Self> {
        let multiples = match model_size {
            "n" => Multiples::n(),
            "s" => Multiples::s(),
            "m" => Multiples::m(),
            "l" => Multiples::l(),
            "x" => Multiples::x(),
            _ => Err(candle::Error::Msg(
                "invalid model size: must be n, s, m, l or x".to_string(),
            ))?,
        };
        let dev = &Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, dev)?;
        let model = YoloV8Pose::load(vb, multiples, 1, (17, 3))?;
        Ok(Self { model })
    }

    pub fn load(md: ModelData) -> Result<Self> {
        Self::load_(md.weights, &md.model_size.to_string())
    }
}

pub struct Worker {
    link: WorkerLink<Self>,
    model: Option<Model>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    RunData(RunData),
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
            WorkerInput::RunData(rd) => match &mut self.model {
                None => Err("model has not been set yet".to_string()),
                Some(model) => {
                    let result = model
                        .run(rd.image_data, rd.conf_threshold, rd.iou_threshold)
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
