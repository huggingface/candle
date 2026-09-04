use anyhow::Result;
use hf_hub::{split_id, HFClientSync};

pub fn download_model(model_and_revision: &str) -> Result<()> {
    let (model_id, revision) = match model_and_revision.split_once(":") {
        Some((model_id, revision)) => (model_id, revision),
        None => (model_and_revision, "main"),
    };
    let (config_filename, tokenizer_filename, weights_filename) = {
        let (owner, name) = split_id(model_id);
        let repo = HFClientSync::new()?.model(owner, name);
        let get = |filename: &str| -> Result<String> {
            let download = repo.download_file().filename(filename).revision(revision);
            let path = match download.clone().local_files_only(true).send() {
                Err(hf_hub::HFError::LocalEntryNotFound { .. }) => download.send()?,
                cached => cached?,
            };
            Ok(path.to_string_lossy().to_string())
        };
        (
            get("config.json")?,
            get("tokenizer.json")?,
            get("model.safetensors")?,
        )
    };
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_CONFIG={config_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_TOKENIZER={tokenizer_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_WEIGHTS={weights_filename}");

    Ok(())
}
