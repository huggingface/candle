#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod audio;
mod midi;

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Result};
use candle::DType;
use candle_nn::VarBuilder;
use candle_transformers::generation::Sampling;
use candle_transformers::models::muscriptor::{
    tokenizer, Config, GenerateOptions, Model, SAMPLE_RATE, SEGMENT_DURATION,
};
use clap::Parser;
use hf_hub::api::sync::Api;

const FRAME_RATE: usize = 100;
const MODEL_SIZES: [&str; 3] = ["small", "medium", "large"];

#[derive(Parser, Debug)]
#[command(author, version, about = "muscriptor — audio-to-MIDI transcription", long_about = None)]
struct Args {
    /// Input audio file (wav, mp3, flac, ...).
    audio_file: PathBuf,

    /// Output MIDI file path. Default: <audio_file>.mid
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Model size ('small', 'medium', 'large') or a local safetensors path.
    #[arg(long, short, default_value = "medium")]
    model: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Transformer dtype: f32, f16 or bf16. Defaults to f16 on Metal
    /// (decoding is bandwidth-bound), f32 elsewhere. The conditioning
    /// pipeline always runs in f32.
    #[arg(long)]
    dtype: Option<String>,

    /// Use temperature sampling instead of greedy decoding.
    #[arg(long)]
    sampling: bool,

    /// Sampling temperature (only with --sampling).
    #[arg(long, short, default_value_t = 1.0)]
    temperature: f64,

    /// Top-p (nucleus) sampling (only with --sampling; takes precedence over
    /// --top-k, matching the reference).
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling (only with --sampling).
    #[arg(long)]
    top_k: Option<usize>,

    /// Seed for --sampling.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Comma-separated instrument names to condition on (e.g. "acoustic_piano,drums").
    #[arg(long)]
    instruments: Option<String>,

    /// Number of 5-second chunks transcribed per batch. Default: 4 on GPU, 1 on CPU.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Maximum generated tokens per chunk.
    #[arg(long, default_value_t = 2000)]
    max_gen_len: usize,

    /// Print decoded note events as they stream.
    #[arg(long)]
    notes: bool,
}

fn resolve_model(model: &str) -> Result<(PathBuf, Config)> {
    if MODEL_SIZES.contains(&model) {
        let api = Api::new()?;
        let repo = api.model(format!("MuScriptor/muscriptor-{model}"));
        let config: Config =
            serde_json::from_reader(std::fs::File::open(repo.get("config.json")?)?)?;
        return Ok((repo.get("model.safetensors")?, config));
    }
    let weights = PathBuf::from(model);
    if !weights.is_file() {
        bail!("--model must be 'small', 'medium', 'large' or a safetensors file path")
    }
    let config_path = weights.parent().map(|p| p.join("config.json"));
    if let Some(config_path) = config_path.filter(|p| p.is_file()) {
        let config = serde_json::from_reader(std::fs::File::open(config_path)?)?;
        return Ok((weights, config));
    }
    // No config.json next to the weights: identify a known variant from the
    // embedding shape (which determines dim and card) and the layer count.
    let st = unsafe { candle::safetensors::MmapedSafetensors::new(&weights)? };
    let mut dim = None;
    let mut num_layers = 0;
    for (name, view) in st.tensors() {
        if name == "emb.0.weight" || name == "emb.weight" {
            let shape = view.shape();
            dim = Some((shape[1], shape[0] - 1));
        }
        if name.starts_with("transformer.layers.") {
            if let Some(n) = name
                .strip_prefix("transformer.layers.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
            {
                num_layers = num_layers.max(n + 1);
            }
        }
    }
    let Some((dim, card)) = dim else {
        bail!("no token embedding found in {}", weights.display())
    };
    for config in [Config::small(), Config::medium(), Config::large()] {
        if config.dim == dim && config.card == card && config.num_layers == num_layers {
            return Ok((weights, config));
        }
    }
    bail!(
        "cannot infer the model architecture (dim={dim}, card={card}, layers={num_layers}); \
         put a config.json next to the weights"
    )
}

fn print_event(ev: &tokenizer::NoteEvent) {
    match ev {
        tokenizer::NoteEvent::Start(s) => println!(
            "NoteStartEvent(pitch={}, start_time={:.2}, index={}, instrument={})",
            s.pitch, s.start_time, s.index, s.instrument
        ),
        tokenizer::NoteEvent::End {
            end_time,
            start_index,
        } => println!("NoteEndEvent(end_time={end_time:.2}, start_event_index={start_index})",),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f32") | Some("float32") => DType::F32,
        Some("f16") | Some("float16") => DType::F16,
        Some("bf16") | Some("bfloat16") => DType::BF16,
        Some(other) => bail!("unsupported dtype {other}"),
        None if device.is_metal() => DType::F16,
        None => DType::F32,
    };

    let start = std::time::Instant::now();
    let (weights, config) = resolve_model(&args.model)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device)? };
    let vb_f32 = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)? };
    let mut model = Model::new(&config, vb, vb_f32)?;
    eprintln!(
        "[muscriptor] loaded {} ({:?}, {:?}): {:.2}s",
        weights.display(),
        dtype,
        device.location(),
        start.elapsed().as_secs_f64()
    );

    let start = std::time::Instant::now();
    let (pcm, sample_rate) = audio::pcm_decode_mono(&args.audio_file)?;
    let pcm = if sample_rate as usize != SAMPLE_RATE {
        audio::resample(&pcm, sample_rate as usize, SAMPLE_RATE)
    } else {
        pcm
    };
    eprintln!(
        "[muscriptor] load audio: {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let instrument_ids = match &args.instruments {
        None => None,
        Some(names) => {
            let names: Vec<&str> = names.split(',').filter(|s| !s.is_empty()).collect();
            Some(tokenizer::instrument_class_ids(&names).map_err(anyhow::Error::msg)?)
        }
    };

    // Split into fixed 5-second segments, zero-padding the last one.
    let segment_samples = (SEGMENT_DURATION * SAMPLE_RATE as f64) as usize;
    let num_chunks = pcm.len().div_ceil(segment_samples).max(1);
    let chunks: Vec<Vec<f32>> = (0..num_chunks)
        .map(|i| {
            let start = i * segment_samples;
            let mut chunk = pcm[start..(start + segment_samples).min(pcm.len())].to_vec();
            chunk.resize(segment_samples, 0.);
            chunk
        })
        .collect();
    eprintln!(
        "[muscriptor] audio: {:.1}s → {num_chunks} chunk(s) of {SEGMENT_DURATION}s",
        pcm.len() as f64 / SAMPLE_RATE as f64
    );

    let batch_size = args
        .batch_size
        .unwrap_or(if device.is_cpu() { 1 } else { 4 })
        .max(1);
    let opts = GenerateOptions {
        max_gen_len: args.max_gen_len,
        sampling: match (args.sampling, args.temperature) {
            (false, _) => None,
            (true, t) if t <= 0. => None,
            // As in the reference: top-p wins over top-k, and a
            // non-positive value disables the filter.
            (true, temperature) => Some(
                match (
                    args.top_p.filter(|&p| p > 0.),
                    args.top_k.filter(|&k| k > 0),
                ) {
                    (Some(p), _) => Sampling::TopP { p, temperature },
                    (None, Some(k)) => Sampling::TopK { k, temperature },
                    (None, None) => Sampling::All { temperature },
                },
            ),
        },
        seed: args.seed,
    };

    let seek_time = |chunk_idx: usize| chunk_idx as f64 * SEGMENT_DURATION;
    let mut decoder = tokenizer::TokenDecoder::new(FRAME_RATE);
    let mut all_events: Vec<tokenizer::NoteEvent> = Vec::new();
    let emit = |events: &mut Vec<tokenizer::NoteEvent>, all: &mut Vec<tokenizer::NoteEvent>| {
        if args.notes {
            for ev in events.iter() {
                print_event(ev);
            }
        }
        all.append(events);
    };
    let start_chunk = |decoder: &mut tokenizer::TokenDecoder,
                       out: &mut Vec<tokenizer::NoteEvent>,
                       chunk_idx: usize| {
        let next = (chunk_idx + 1 < num_chunks).then(|| seek_time(chunk_idx + 1));
        decoder.start_chunk(seek_time(chunk_idx), next, out);
    };

    let gen_start = std::time::Instant::now();
    for batch_start in (0..num_chunks).step_by(batch_size) {
        let batch = &chunks[batch_start..(batch_start + batch_size).min(num_chunks)];
        let n = batch.len();
        let prefix = model.build_prefix(batch, instrument_ids.as_deref())?;

        // The model emits one token per chunk per step, but the decoder
        // consumes whole chunks in order: the first chunk streams live, the
        // others buffer until every earlier chunk has hit EOS.
        let mut buffers: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut done = vec![false; n];
        let mut active = 0usize;
        let mut events = Vec::new();
        start_chunk(&mut decoder, &mut events, batch_start);
        emit(&mut events, &mut all_events);

        model.generate(&prefix, &opts, |tokens| {
            let mut events = Vec::new();
            for (j, &tok) in tokens.iter().enumerate() {
                if done[j] {
                    continue;
                }
                if tok == tokenizer::EOS_ID {
                    done[j] = true;
                } else if j == active {
                    decoder.push(tok, &mut events);
                } else {
                    buffers[j].push(tok);
                }
            }
            while active < n && done[active] {
                active += 1;
                if active < n {
                    start_chunk(&mut decoder, &mut events, batch_start + active);
                    for tok in buffers[active].drain(..) {
                        decoder.push(tok, &mut events);
                    }
                }
            }
            emit(&mut events, &mut all_events);
            Ok(())
        })?;

        // Chunks that never emitted EOS within max_gen_len.
        for j in active..n {
            if !done[j] {
                eprintln!(
                    "[muscriptor] warning: chunk {} (seek={:.1}s) did not emit EOS within {} tokens",
                    batch_start + j,
                    seek_time(batch_start + j),
                    args.max_gen_len
                );
            }
            if j != active {
                let mut events = Vec::new();
                start_chunk(&mut decoder, &mut events, batch_start + j);
                for tok in buffers[j].drain(..) {
                    decoder.push(tok, &mut events);
                }
                emit(&mut events, &mut all_events);
            }
        }
        eprintln!(
            "[muscriptor] chunks {}/{}: {:.2}s",
            batch_start + n,
            num_chunks,
            gen_start.elapsed().as_secs_f64()
        );
    }
    let mut events = Vec::new();
    decoder.finish(&mut events);
    emit(&mut events, &mut all_events);
    eprintln!(
        "[muscriptor] generate total: {:.2}s",
        gen_start.elapsed().as_secs_f64()
    );

    let notes = tokenizer::events_to_notes(&all_events);
    let midi_bytes = midi::notes_to_midi_bytes(&notes);
    let output = args
        .output
        .unwrap_or_else(|| args.audio_file.with_extension("mid"));
    let mut file = std::fs::File::create(&output)?;
    file.write_all(&midi_bytes)?;
    eprintln!(
        "[muscriptor] wrote {} note(s) to {}",
        notes.len(),
        output.display()
    );
    Ok(())
}
