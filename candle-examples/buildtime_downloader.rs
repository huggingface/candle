use anyhow::Result;
use hf_hub::HFClientSync;

pub fn download_model(model_and_revision: &str) -> Result<()> {
    let (model_id, revision) = match model_and_revision.split_once(":") {
        Some((model_id, revision)) => (model_id, revision),
        None => (model_and_revision, "main"),
    };
    let (config_filename, tokenizer_filename, weights_filename) = {
        let client = HFClientSync::new()?;
        let (owner, name) = model_id.split_once('/').unwrap_or(("", model_id));
        let repo = client.model(owner, name);
        let dl = |file: &str| -> Result<String> {
            Ok(repo
                .download_file()
                .filename(file)
                .revision(revision.to_string())
                .send()?
                .to_string_lossy()
                .to_string())
        };
        (
            dl("config.json")?,
            dl("tokenizer.json")?,
            dl("model.safetensors")?,
        )
    };
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_CONFIG={config_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_TOKENIZER={tokenizer_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_WEIGHTS={weights_filename}");

    Ok(())
}
