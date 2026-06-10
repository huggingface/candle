use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

use candle::{DType, IndexOp};
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

    /// Print debug info about decoded tokens.
    #[arg(long, default_value_t = false)]
    debug: bool,
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
    decoding_config.debug_decode = args.debug;
    if let Some(beam_size) = args.beam_size {
        decoding_config.decoding = Decoding::Beam(Beam {
            beam_size,
            ..Beam::default()
        });
    }

    if args.debug {
        match &mut model {
            ParakeetModel::Tdt(m) => {
                let audio = candle_transformers::models::parakeet::load_audio(
                    &args.input,
                    m.preprocessor_config.sample_rate,
                )?;
                let audio_secs = audio.len() as f64 / m.preprocessor_config.sample_rate as f64;
                eprintln!(
                    "debug: audio_len={} samples secs={:.2} sr={}",
                    audio.len(),
                    audio_secs,
                    m.preprocessor_config.sample_rate
                );
                let mel = candle_transformers::models::parakeet::get_logmel(
                    &audio,
                    &m.preprocessor_config,
                    &m.device,
                )?;
                let (mb, mt, mf) = mel.dims3()?;
                eprintln!("debug: mel dims=({mb},{mt},{mf})");
                let (features, lengths) = m.encoder.forward(&mel, None)?;
                let (fb, ft, ff) = features.dims3()?;
                let lengths = lengths.to_vec1::<i64>()?;
                eprintln!(
                    "debug: features dims=({fb},{ft},{ff}) lengths={:?}",
                    lengths
                );

                let vocab = &m.vocabulary;
                let blank_id = vocab.len();
                let (decoder_out, _) = m.decoder.forward(None, None)?;
                let enc_step = features.narrow(1, 0, 1)?;
                let joint_out = m.joint.forward(&enc_step, &decoder_out)?;
                let vocab_size = vocab.len() + 1;
                let token_logits = joint_out.i((0, 0, 0, 0..vocab_size))?;
                let duration_logits = joint_out.i((0, 0, 0, vocab_size..))?;
                let token_vals = token_logits.to_vec1::<f32>()?;
                let mut idxs: Vec<usize> = (0..token_vals.len()).collect();
                idxs.sort_by(|&a, &b| {
                    token_vals[b]
                        .partial_cmp(&token_vals[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                eprintln!("debug: top token logits (step0)");
                for (i, id) in idxs.iter().take(10).enumerate() {
                    let text = if *id == blank_id {
                        "<blank>".to_string()
                    } else {
                        vocab
                            .get(*id)
                            .cloned()
                            .unwrap_or_else(|| "<oob>".to_string())
                    };
                    eprintln!(
                        "debug: tok[{i}] id={} text={:?} logit={:.4}",
                        id, text, token_vals[*id]
                    );
                }
                let dur_vals = duration_logits.to_vec1::<f32>()?;
                eprintln!("debug: duration logits {:?}", dur_vals);
            }
            ParakeetModel::Rnnt(m) => {
                let audio = candle_transformers::models::parakeet::load_audio(
                    &args.input,
                    m.preprocessor_config.sample_rate,
                )?;
                let audio_secs = audio.len() as f64 / m.preprocessor_config.sample_rate as f64;
                eprintln!(
                    "debug: audio_len={} samples secs={:.2} sr={}",
                    audio.len(),
                    audio_secs,
                    m.preprocessor_config.sample_rate
                );
                let mel = candle_transformers::models::parakeet::get_logmel(
                    &audio,
                    &m.preprocessor_config,
                    &m.device,
                )?;
                let (mb, mt, mf) = mel.dims3()?;
                eprintln!("debug: mel dims=({mb},{mt},{mf})");
                let (features, lengths) = m.encoder.forward(&mel, None)?;
                let (fb, ft, ff) = features.dims3()?;
                let lengths = lengths.to_vec1::<i64>()?;
                eprintln!(
                    "debug: features dims=({fb},{ft},{ff}) lengths={:?}",
                    lengths
                );
            }
            ParakeetModel::Ctc(m) => {
                let audio = candle_transformers::models::parakeet::load_audio(
                    &args.input,
                    m.preprocessor_config.sample_rate,
                )?;
                let audio_secs = audio.len() as f64 / m.preprocessor_config.sample_rate as f64;
                eprintln!(
                    "debug: audio_len={} samples secs={:.2} sr={}",
                    audio.len(),
                    audio_secs,
                    m.preprocessor_config.sample_rate
                );
                let mel = candle_transformers::models::parakeet::get_logmel(
                    &audio,
                    &m.preprocessor_config,
                    &m.device,
                )?;
                let (mb, mt, mf) = mel.dims3()?;
                eprintln!("debug: mel dims=({mb},{mt},{mf})");
                let (features, lengths) = m.encoder.forward(&mel, None)?;
                let (fb, ft, ff) = features.dims3()?;
                let lengths = lengths.to_vec1::<i64>()?;
                eprintln!(
                    "debug: features dims=({fb},{ft},{ff}) lengths={:?}",
                    lengths
                );
            }
            ParakeetModel::TdtCtc(m) => {
                let audio = candle_transformers::models::parakeet::load_audio(
                    &args.input,
                    m.base.preprocessor_config.sample_rate,
                )?;
                let audio_secs = audio.len() as f64 / m.base.preprocessor_config.sample_rate as f64;
                eprintln!(
                    "debug: audio_len={} samples secs={:.2} sr={}",
                    audio.len(),
                    audio_secs,
                    m.base.preprocessor_config.sample_rate
                );
                let mel = candle_transformers::models::parakeet::get_logmel(
                    &audio,
                    &m.base.preprocessor_config,
                    &m.base.device,
                )?;
                let (mb, mt, mf) = mel.dims3()?;
                eprintln!("debug: mel dims=({mb},{mt},{mf})");
                let (features, lengths) = m.base.encoder.forward(&mel, None)?;
                let (fb, ft, ff) = features.dims3()?;
                let lengths = lengths.to_vec1::<i64>()?;
                eprintln!(
                    "debug: features dims=({fb},{ft},{ff}) lengths={:?}",
                    lengths
                );
            }
        }
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

    if args.debug {
        let tokens = result.tokens();
        eprintln!(
            "debug: text_len={} tokens={} sentences={}",
            result.text.len(),
            tokens.len(),
            result.sentences.len()
        );
        for (i, tok) in tokens.iter().take(20).enumerate() {
            eprintln!(
                "debug: tok[{i}] id={} text={:?} start={:.3} dur={:.3} conf={:.3}",
                tok.id, tok.text, tok.start, tok.duration, tok.confidence
            );
        }
        if tokens.len() > 20 {
            eprintln!("debug: ... ({} more tokens)", tokens.len() - 20);
        }
    }

    println!("{}", result.text);
    Ok(())
}
