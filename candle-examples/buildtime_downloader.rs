use anyhow::Result;
use hf_hub::HFClientSync;

pub fn download_model(model_and_revision: &str) -> Result<()> {
    let (model_id, revision) = match model_and_revision.split_once(":") {
        Some((model_id, revision)) => (model_id, revision),
        None => (model_and_revision, "main"),
    };
    let (config_filename, tokenizer_filename, weights_filename) = {
        let client = HFClientSync::new()?;
        let (owner, name) = hf_hub::split_id(model_id);
        let repo = client.model(owner, name);
        let config = repo
            .download_file()
            .filename("config.json")
            .revision(revision)
            .send()?
            .to_string_lossy()
            .to_string();
        let tokenizer = repo
            .download_file()
            .filename("tokenizer.json")
            .revision(revision)
            .send()?
            .to_string_lossy()
            .to_string();
        let weights = repo
            .download_file()
            .filename("model.safetensors")
            .revision(revision)
            .send()?
            .to_string_lossy()
            .to_string();
        (config, tokenizer, weights)
    };
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_CONFIG={config_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_TOKENIZER={tokenizer_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_WEIGHTS={weights_filename}");

    Ok(())
}
