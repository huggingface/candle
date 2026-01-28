use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

use candle::DType;
use candle_nn::VarBuilder;

use candle_transformers::models::parakeet::{
    from_config_value, Beam, Decoding, DecodingConfig, ParakeetModel,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Hugging Face model id.
    #[arg(long, default_value = "mlx-community/parakeet-tdt-0.6b-v3")]
    model_id: String,

    /// Input audio file path.
    #[arg(long)]
    input: PathBuf,

    /// Chunk duration in seconds for long audio.
    #[arg(long)]
    chunk_duration: Option<f64>,

    /// Overlap duration in seconds when chunking.
    #[arg(long, default_value_t = 15.0)]
    overlap_duration: f64,

    /// Beam size for TDT decoding (enables beam search when set).
    #[arg(long)]
    beam_size: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    let api = Api::new()?;
    let repo = api.repo(Repo::new(args.model_id.clone(), RepoType::Model));

    let config_path = repo.get("config.json").context("missing config.json")?;
    let config_file = File::open(&config_path)?;
    let config: serde_json::Value = serde_json::from_reader(config_file)?;

    let safetensors_files = match repo.get("model.safetensors.index.json") {
        Ok(_) => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        Err(_) => vec![repo.get("model.safetensors")?],
    };

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, DType::F32, &device)? };
    let mut model = from_config_value(config, vb)?;

    let mut decoding_config = DecodingConfig::default();
    if let Some(beam_size) = args.beam_size {
        decoding_config.decoding = Decoding::Beam(Beam {
            beam_size,
            ..Beam::default()
        });
    }

    let result = match &mut model {
        ParakeetModel::Tdt(m) => m.transcribe(
            &args.input,
            &decoding_config,
            args.chunk_duration,
            args.overlap_duration,
            None,
        )?,
        ParakeetModel::Rnnt(m) => m.transcribe(
            &args.input,
            &decoding_config,
            args.chunk_duration,
            args.overlap_duration,
            None,
        )?,
        ParakeetModel::Ctc(m) => m.transcribe(
            &args.input,
            &decoding_config,
            args.chunk_duration,
            args.overlap_duration,
            None,
        )?,
        ParakeetModel::TdtCtc(m) => m.base.transcribe(
            &args.input,
            &decoding_config,
            args.chunk_duration,
            args.overlap_duration,
            None,
        )?,
    };

    println!("{}", result.text);
    Ok(())
}
