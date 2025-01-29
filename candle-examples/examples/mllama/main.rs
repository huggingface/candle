// TODO: we need to impelement these layers
// 1. MllamaPrecomputedPositionEmbedding
// 2. MllamaPrecomputedAspectRatioEmbedding
// 3. MllamaVisionSdpaAttention -> we have a struct named SdpaAttention, could we use it here too?
// 4. MllamaTextCrossSdpaAttention
// 5. MllamaTextSelfSdpaAttention

use anyhow::Ok;
use anyhow::{bail, Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::{init, Embedding, Module, VarBuilder};
use candle_transformers::models::llava::config::HFPreProcessorConfig;
use config::MllamaConfig;
use hf_hub::{api::sync::Api, Repo, RepoType};
use image_processor::{process_image, ImageProcessor};
use serde::{Deserialize, Serialize};

#[path = "config.rs"]
mod config;

#[path = "vision_model.rs"]
mod MllamaVisionModel;

fn main() -> Result<()> {
    let api = Api::new()?;
    let revision = String::from("main");
    let api = api.repo(Repo::with_revision(
        String::from("meta-llama/Llama-3.2-11B-Vision"),
        RepoType::Model,
        revision,
    ));
    // Load config
    let config_filename = api.get("config.json")?;
    let config: MllamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    println!("config: {:?}", config);

    // Load Image Processor
    let preprocessor_config_filename = api.get("preprocessor_config.json")?;
    let preprocessor_config: HFPreProcessorConfig =
        serde_json::from_slice(&std::fs::read(preprocessor_config_filename)?)?;
    let image_processor = ImageProcessor::from_hf_preprocessor_config(&preprocessor_config);

    let img = image::ImageReader::open(&"path")?.decode()?;

    // Load weights
    // let dtype = DType::F16;
    // let device = candle_examples::device(true)?;
    // let filenames = candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;
    // let var = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    // let init_ws = init::DEFAULT_KAIMING_NORMAL;
    // let s = (9, 5248000);
    // let name = "vision_model.gated_positional_embedding.tile_embedding.weight";
    // let weights = var.get_with_hints((9, 5248000), name, init_ws)?;
    // println!("Tensor: {:?}", weights);
    Ok(())
}
