use candle::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::idefics3::model::{
    ColIdefics3Model, Idefic3VisionConfig, Idefics3Config, Idefics3Model, Idefics3VisionEmbeddings,
    Idefics3VisionTransformer,
};

use hf_hub::{api::sync::Api, Repo, RepoType};

use crate::processing::Idefics3Processor;
mod processing;

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

fn main() {
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::new(
        "akshayballal/colSmol-256M-merged".to_string(),
        RepoType::Model,
    ));
    let config_file = repo.get("config.json").unwrap();
    let model_file = repo.get("model.safetensors").unwrap();
    let config: Idefics3Config =
        serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();

    let processor = Idefics3Processor::from_pretrained("akshayballal/colSmol-256M-merged").unwrap();

    let device = candle_examples::device(false).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file], candle::DType::F16, &device).unwrap()
    };
    // let embedding_model = Idefics3VisionEmbeddings::load(&config.vision_config, vb).unwrap();
    let transformer = Idefics3Model::load(&config, false, vb.pp("model")).unwrap();

    
    let image = image::open("/home/akshay/projects/EmbedAnything/test.jpg").unwrap();

    let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
        processor.preprocess(&[image.clone(), image.clone()], &Device::Cpu).unwrap();


    let hidden_states = transformer
        .forward(
            &input_ids.to_device(&device).unwrap(),
            &Some(
                pixel_values
                    .to_device(&device)
                    .unwrap()
                    .to_dtype(candle::DType::F16)
                    .unwrap(),
            ),
            &Some(pixel_attention_mask.unwrap().to_device(&device).unwrap()),
        )
        .unwrap();
}
