use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::blip;
use candle_transformers::models::quantized_blip;
use candle_wasm_example_blip::console_log;
use candle_wasm_example_blip::token_output_stream::TokenOutputStream;
use js_sys::Date;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

enum SelectedModel {
    M(blip::BlipForConditionalGeneration),
    Q(quantized_blip::BlipForConditionalGeneration),
}

impl SelectedModel {
    fn text_decoder_forward(&mut self, xs: &Tensor, img_xs: &Tensor) -> Result<Tensor, JsError> {
        match self {
            Self::M(m) => m
                .text_decoder()
                .forward(xs, img_xs)
                .map_err(|e| JsError::new(&e.to_string())),
            Self::Q(m) => m
                .text_decoder()
                .forward(xs, img_xs)
                .map_err(|e| JsError::new(&e.to_string())),
        }
    }
    fn reset_kv_cache(&mut self) {
        match self {
            Self::M(m) => m.reset_kv_cache(),
            Self::Q(m) => m.reset_kv_cache(),
        }
    }
}
#[wasm_bindgen]
pub struct Model {
    model: SelectedModel,
    tokenizer: TokenOutputStream,
}
const SEP_TOKEN_ID: u32 = 102;

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let tokenizer = TokenOutputStream::new(tokenizer);

        let config: blip::Config = serde_json::from_slice(&config)?;
        let device = Device::Cpu;

        let start = Date::now();
        let model: SelectedModel = if quantized {
            let vb = quantized_blip::VarBuilder::from_gguf_buffer(&weights, &device)?;
            let model = quantized_blip::BlipForConditionalGeneration::new(&config, vb)?;
            SelectedModel::Q(model)
        } else {
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
            let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
            SelectedModel::M(model)
        };

        console_log!("model loaded in {:?}s", (Date::now() - start) / 1000.);
        Ok(Self { model, tokenizer })
    }
    #[wasm_bindgen]
    pub fn generate_caption_from_image(&mut self, image: Vec<u8>) -> Result<String, JsError> {
        self.model.reset_kv_cache();

        let device = Device::Cpu;
        console_log!("loading image as tensor");
        let start = Date::now();
        let image: Tensor = self.load_image(image)?.to_device(&device)?;
        console_log!("image loaded in {:?}s", (Date::now() - start) / 1000.);
        let start = Date::now();
        let image_embeds: Tensor = match &mut self.model {
            SelectedModel::M(m) => image.unsqueeze(0)?.apply(m.vision_model())?,
            SelectedModel::Q(m) => image.unsqueeze(0)?.apply(m.vision_model())?,
        };
        console_log!("image embedded in {:?}s", (Date::now() - start) / 1000.);
        let mut logits_processor = LogitsProcessor::new(299792458, None, None);
        let mut token_ids = vec![30522u32];
        let mut text: String = "".to_string();

        let start = Date::now();
        for index in 0..1000 {
            let context_size = if index > 0 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
            let logits = self.model.text_decoder_forward(&input_ids, &image_embeds)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            if token == SEP_TOKEN_ID {
                break;
            }
            token_ids.push(token);
            if let Some(t) = self.tokenizer.next_token(token)? {
                text.push_str(&t);
            }
        }
        if let Some(rest) = self
            .tokenizer
            .decode_rest()
            .map_err(|m| JsError::new(&m.to_string()))?
        {
            text.push_str(&rest);
        }
        console_log!("caption generated in {:?}s", (Date::now() - start) / 1000.);
        Ok(text)
    }
}

impl Model {
    fn load_image(&self, image: Vec<u8>) -> Result<Tensor, JsError> {
        let device = &Device::Cpu;
        let img = image::ImageReader::new(std::io::Cursor::new(image))
            .with_guessed_format()?
            .decode()
            .map_err(|e| JsError::new(&e.to_string()))?
            .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (384, 384, 3), device)?.permute((2, 0, 1))?;
        let mean =
            Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], device)?.reshape((3, 1, 1))?;
        let std =
            Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], device)?.reshape((3, 1, 1))?;
        (data.to_dtype(candle::DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
