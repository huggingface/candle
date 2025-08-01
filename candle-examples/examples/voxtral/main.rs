use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
use model::VoxtralModel;

mod audio_utils;
mod download;
mod model;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/candle_demo/
    #[arg(long)]
    input: Option<String>,

    #[arg(long, default_value = "mistralai/Voxtral-Mini-3B-2507")]
    model_id: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let use_cpu = args.cpu;

    let model_id = args.model_id.unwrap();

    // Create model - equivalent to loading the model and processor in Python
    let mut model =
        VoxtralModel::new(&model_id, use_cpu).context("Failed to load Voxtral model")?;

    println!("Model loaded successfully on device: {:?}", model.device());

    let api = Api::new()?;
    let dataset = api.dataset("Narsil/candle-examples".to_string());

    let audio_file = if let Some(input) = args.input {
        if let Some(sample) = input.strip_prefix("sample:") {
            dataset.get(&format!("samples_{sample}.wav"))?
        } else {
            std::path::PathBuf::from(input)
        }
    } else {
        println!("No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/candle_demo/blob/main/samples_jfk.wav");
        dataset.get("samples_jfk.wav")?
    };

    let (audio_data, sample_rate) =
        candle_examples::audio::pcm_decode(audio_file).context("Failed to decode audio file")?;

    // Transcribe audio with token output
    let result = model
        .transcribe_audio(&audio_data, sample_rate)
        .context("Failed to transcribe audio with tokens")?;

    println!("\n===================================================\n");
    println!("{}", result.text);

    Ok(())
}
