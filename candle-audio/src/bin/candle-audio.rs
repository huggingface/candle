use anyhow::{Context, Result};
use candle::Device;
use candle_audio::{
    models::registry::{ModelRegistry, ModelSpec},
    *,
};
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "live-io")]
#[path = "audio_cli/live.rs"]
mod live_cli;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    about = "Native Candle speech recognition, synthesis and realtime audio",
    version
)]
struct Args {
    #[arg(long, value_enum, default_value = "cpu", global = true)]
    device: Backend,
    #[arg(long, default_value_t = 0, global = true)]
    gpu_index: usize,
    #[arg(long, value_enum, default_value = "f32", global = true)]
    dtype: Precision,
    /// JSON object with adapter-specific loader options (local paths and budgets).
    #[arg(long, global = true)]
    model_options: Option<PathBuf>,
    /// Precision of an optional --tts-model readback model, independent of --dtype.
    #[arg(long, value_enum, default_value = "f32", global = true)]
    tts_dtype: Precision,
    /// Loader options for the optional readback synthesizer only.
    #[arg(long, global = true)]
    tts_model_options: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    Bf16,
}
struct LoadSettings {
    dtype: candle::DType,
    options: serde_json::Value,
}
impl LoadSettings {
    fn new(args: &Args) -> Result<Self> {
        Self::from_options(args.dtype, args.model_options.as_deref())
    }
    fn readback(args: &Args) -> Result<Self> {
        Self::from_options(args.tts_dtype, args.tts_model_options.as_deref())
    }
    fn from_options(dtype: Precision, path: Option<&Path>) -> Result<Self> {
        let options = match path {
            None => serde_json::json!({}),
            Some(path) => {
                let mut bytes = Vec::new();
                File::open(path)?.take(65537).read_to_end(&mut bytes)?;
                anyhow::ensure!(bytes.len() <= 65536, "model-options exceeds65536bytes");
                let value: serde_json::Value = serde_json::from_slice(&bytes)?;
                anyhow::ensure!(value.is_object(), "model-options must be a JSON object");
                value
            }
        };
        Ok(Self {
            dtype: match dtype {
                Precision::F32 => candle::DType::F32,
                Precision::Bf16 => candle::DType::BF16,
            },
            options,
        })
    }
    fn apply(&self, mut spec: ModelSpec) -> ModelSpec {
        let mut options = self.options.clone();
        if let Some(object) = spec.options.as_object() {
            for (key, value) in object {
                options[key] = value.clone();
            }
        }
        spec.options = options;
        spec.dtype = self.dtype;
        spec
    }
    fn spec(&self, directory: PathBuf) -> ModelSpec {
        self.apply(ModelSpec::new(directory))
    }
}
#[derive(Clone, ValueEnum)]
enum Backend {
    Cpu,
    Vulkan,
    Cuda,
    Metal,
}
#[derive(Clone, Copy, ValueEnum)]
enum Task {
    Asr,
    Tts,
    Realtime,
    Codec,
    Understanding,
    AudioGeneration,
}
impl Task {
    fn audio_task(self) -> AudioTask {
        match self {
            Self::Asr => AudioTask::Recognition,
            Self::Tts => AudioTask::Synthesis,
            Self::Realtime => AudioTask::Realtime,
            Self::Codec => AudioTask::Codec,
            Self::Understanding => AudioTask::Understanding,
            Self::AudioGeneration => AudioTask::AudioGeneration,
        }
    }
}
#[derive(Subcommand)]
enum Command {
    /// List registered native model families and their tasks without loading weights.
    Models,
    /// List native capture/playback devices without opening streams.
    #[cfg(feature = "live-io")]
    Devices,
    /// Manually controlled native microphone and playback (Enter starts/stops a turn).
    #[cfg(feature = "live-io")]
    Live {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input_device: Option<String>,
        #[arg(long)]
        output_device: Option<String>,
        #[arg(long)]
        tts_model: Option<PathBuf>,
        /// Text context for realtime adapters that require an audio question.
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 20)]
        turn_seconds: u32,
    },

    /// Load a checkpoint and report the selected adapter's declared capabilities.
    Capabilities {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_enum)]
        task: Task,
        #[arg(long, default_value_t = 16000)]
        sample_rate: u32,
        #[arg(long, default_value_t = 1)]
        channels: u16,
    },
    /// Encode/decode native codec codebooks as lossless multiscale JSON.
    Codec {
        #[command(subcommand)]
        command: CodecCommand,
    },
    /// Transcribe WAV audio using bounded windows and automatic context rollover.
    #[command(alias = "asr")]
    Transcribe {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        translate: bool,
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Maximum source duration per turn, further clamped to model context.
        #[arg(long, default_value_t = 30)]
        turn_seconds: u32,
    },
    /// Synthesize PCM16 WAV. A completed file is published atomically.
    #[command(alias = "tts")]
    Synthesize {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        /// JSON ConditionedSynthesisRequest references and speaker embedding.
        /// The text/language/voice flags supply its synthesis request.
        #[arg(long)]
        conditioning: Option<PathBuf>,
    },
    /// Generate music or other audio from a text prompt.
    GenerateAudio {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 250)]
        max_steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 3.)]
        guidance_scale: f64,
        /// Omit for greedy decoding.
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long)]
        audio_reference: Option<PathBuf>,
    },
    /// Ask a native audio understanding model about one or more WAV clips.
    Understand {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, required=true, num_args=1..)]
        input: Vec<PathBuf>,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 512)]
        max_steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Stream WAV chunks through any registered realtime model or speech pipeline.
    Realtime {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        /// Optional synthesizer to read back recognized segments.
        #[arg(long)]
        tts_model: Option<PathBuf>,
        /// Text context for realtime adapters that require an audio question.
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        max_tokens: Option<usize>,
        #[arg(long, default_value_t = 100)]
        chunk_ms: u32,
        #[arg(long, default_value_t = 5000)]
        window_ms: u32,
        #[arg(long, default_value_t = 30)]
        turn_seconds: u32,
        /// Pace file input by its audio clock; this does not capture a microphone.
        #[arg(long)]
        pace: bool,
        /// Keep committed conversation history across automatic file turns.
        /// The model will reject input when its retained context becomes full.
        #[arg(long)]
        preserve_conversation: bool,
    },
}
#[derive(Subcommand)]
enum CodecCommand {
    /// Encode up to 30 seconds of WAV into JSON codebooks, preserving ragged scales.
    Encode {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        num_codebooks: Option<usize>,
    },
    /// Decode bounded JSON codebooks into an atomically published PCM16 WAV.
    Decode {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}
fn convert_audio_reference(audio: AudioBuffer, target: AudioFormat) -> Result<AudioBuffer> {
    let source = audio.format();
    let policy = if source.channels() == target.channels() {
        ChannelPolicy::Preserve
    } else if target.channels() == 1 {
        ChannelPolicy::Average
    } else if source.channels() == 1 && target.channels() == 2 {
        ChannelPolicy::Map(vec![0, 0])
    } else {
        anyhow::bail!(
            "reference channel layout requires an explicit channel map through the native API"
        )
    };
    Ok(AudioConverter::convert_to(&audio, target, policy)?)
}
fn read_conditioning(path: &Path) -> Result<ConditionedSynthesisRequest> {
    let file = File::open(path)?;
    anyhow::ensure!(
        file.metadata()?.len() <= MAX_CODEC_JSON_BYTES as u64,
        "conditioning JSON exceeds 32 MiB"
    );
    let mut bytes = Vec::new();
    file.take(MAX_CODEC_JSON_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= MAX_CODEC_JSON_BYTES,
        "conditioning JSON exceeds 32 MiB"
    );
    let request: ConditionedSynthesisRequest = serde_json::from_slice(&bytes)?;
    request.validate()?;
    Ok(request)
}
const MAX_CODEC_JSON_BYTES: usize = 32 * 1024 * 1024;
const MAX_CODEC_INPUT_SAMPLES: usize = 16_000_000;
fn read_codec_audio(path: &Path) -> Result<AudioBuffer> {
    let mut source = WavSource::open(path)?;
    let frames = source.reader.duration() as usize;
    let maximum = source.format.sample_rate() as usize * 30;
    anyhow::ensure!(
        frames > 0 && frames <= maximum,
        "codec input must contain more than zero and at most 30 seconds"
    );
    anyhow::ensure!(
        frames
            .checked_mul(source.format.channels() as usize)
            .is_some_and(|n| n <= MAX_CODEC_INPUT_SAMPLES),
        "codec source exceeds the 16 million sample memory budget"
    );
    let audio = source.read(frames)?;
    anyhow::ensure!(audio.frames() == frames, "codec input WAV is incomplete");
    Ok(audio)
}
fn read_codes(path: &Path) -> Result<MultiScaleAudioCodes> {
    let file = File::open(path)?;
    anyhow::ensure!(
        file.metadata()?.len() <= MAX_CODEC_JSON_BYTES as u64,
        "codec JSON exceeds 32 MiB"
    );
    let mut bytes = Vec::new();
    file.take(MAX_CODEC_JSON_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= MAX_CODEC_JSON_BYTES,
        "codec JSON exceeds 32 MiB"
    );
    let codes: MultiScaleAudioCodes = serde_json::from_slice(&bytes)?;
    let format = codes.format();
    format.validate()?;
    anyhow::ensure!(
        !codes.codebooks().is_empty() && codes.codebooks().iter().all(|book| !book.is_empty()),
        "codec JSON contains no complete code sequence"
    );
    let max_seconds = 30. + (1. / format.frame_rate).min(1.);
    anyhow::ensure!(
        codes
            .codebooks()
            .iter()
            .zip(codes.codebook_strides())
            .all(
                |(book, stride)| book.len() as f64 * (*stride as f64) / format.frame_rate
                    <= max_seconds
            ),
        "codec JSON exceeds 30 seconds plus one padding frame"
    );
    anyhow::ensure!(
        codes
            .original_audio_frames()
            .is_none_or(|n| n as f64 / format.audio_format.sample_rate() as f64 <= 30.),
        "original codec audio exceeds 30 seconds"
    );
    Ok(codes)
}
fn encode_codec(
    codec: &mut dyn AudioCodec,
    audio: &AudioBuffer,
    cancel: &CancellationToken,
) -> Result<MultiScaleAudioCodes> {
    cancel.check()?;
    let format = codec.format();
    format.validate()?;
    let channels = audio.format().channels();
    let target = format.audio_format;
    let policy = if channels == target.channels() {
        ChannelPolicy::Preserve
    } else if target.channels() == 1 {
        ChannelPolicy::Average
    } else if channels == 1 {
        ChannelPolicy::Map(vec![0; target.channels() as usize])
    } else {
        anyhow::bail!(
            "cannot implicitly map {channels} source channels to {} codec channels",
            target.channels()
        )
    };
    anyhow::ensure!(
        1024. * target.sample_rate() as f64 / audio.format().sample_rate() as f64
            * target.channels() as f64
            <= MAX_CODEC_INPUT_SAMPLES as f64,
        "codec resampling block exceeds the memory budget"
    );
    let converted = AudioConverter::convert_to(audio, target, policy)?;
    anyhow::ensure!(
        converted.frames() <= target.sample_rate() as usize * 30,
        "converted codec input exceeds 30 seconds"
    );
    cancel.check()?;
    let codes = codec.encode_multiscale(&converted, cancel)?;
    cancel.check()?;
    Ok(codes)
}
fn publish_codes(
    path: &Path,
    codes: &MultiScaleAudioCodes,
    cancel: &CancellationToken,
) -> Result<()> {
    validate_output_path(path)?;
    cancel.check()?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = tempfile::Builder::new()
        .prefix(".candle-audio-codes-")
        .tempfile_in(parent)?;
    {
        let mut writer = BufWriter::new(temporary.reopen()?);
        serde_json::to_writer(&mut writer, codes)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    cancel.check()?;
    temporary.as_file().sync_all()?;
    temporary.persist_noclobber(path).with_context(|| {
        format!(
            "publishing codec codes {} without overwriting",
            path.display()
        )
    })?;
    Ok(())
}
fn check_language(language: Option<&str>) -> Result<()> {
    anyhow::ensure!(
        !language.is_some_and(|v| v.trim().is_empty()),
        "language must not be empty"
    );
    Ok(())
}
fn check_budget(max_tokens: Option<usize>, turn_seconds: u32) -> Result<()> {
    anyhow::ensure!(
        max_tokens.is_none_or(|n| (1..=32768).contains(&n)),
        "max-tokens must be in 1..32768"
    );
    anyhow::ensure!(
        (1..=3600).contains(&turn_seconds),
        "turn-seconds must be in 1..3600"
    );
    Ok(())
}
fn validate_output_path(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => anyhow::bail!(
            "output {} already exists: choose a new path; existing files are never overwritten",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("checking output {}", path.display()))
        }
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    anyhow::ensure!(
        parent.is_dir(),
        "output directory {} does not exist",
        parent.display()
    );
    anyhow::ensure!(path.file_name().is_some(), "output must name a file");
    Ok(())
}
impl Args {
    fn validate(&self) -> Result<()> {
        let prompt = match &self.command {
            Command::Realtime { prompt, .. } => prompt.as_deref(),
            #[cfg(feature = "live-io")]
            Command::Live { prompt, .. } => prompt.as_deref(),
            _ => None,
        };
        if let Some(prompt) = prompt {
            anyhow::ensure!(
                !prompt.trim().is_empty() && prompt.len() <= 16384,
                "prompt must contain1..16384bytes"
            );
        }
        match &self.command {
            Command::Models => {}
            #[cfg(feature = "live-io")]
            Command::Devices => {}
            #[cfg(feature = "live-io")]
            Command::Live {
                model,
                tts_model,
                turn_seconds,
                ..
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                anyhow::ensure!(
                    tts_model
                        .as_ref()
                        .is_none_or(|directory| directory.is_dir()),
                    "TTS model directory does not exist"
                );
                anyhow::ensure!(
                    (1..=30).contains(turn_seconds),
                    "turn-seconds must be in1..30"
                );
            }

            Command::Codec { command } => match command {
                CodecCommand::Encode {
                    model,
                    input,
                    output,
                    num_codebooks,
                } => {
                    anyhow::ensure!(model.is_dir(), "model directory does not exist");
                    anyhow::ensure!(
                        num_codebooks.is_none_or(|n| (1..=64).contains(&n)),
                        "num-codebooks must be in 1..64"
                    );
                    validate_output_path(output)?;
                    read_codec_audio(input)?;
                }
                CodecCommand::Decode {
                    model,
                    input,
                    output,
                } => {
                    anyhow::ensure!(model.is_dir(), "model directory does not exist");
                    validate_output_path(output)?;
                    read_codes(input)?;
                }
            },
            Command::GenerateAudio {
                model,
                prompt,
                output,
                max_steps,
                seed,
                guidance_scale,
                temperature,
                top_k,
                top_p,
                audio_reference,
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                let request = AudioGenerationRequest {
                    prompt: prompt.clone(),
                    seed: *seed,
                    max_steps: *max_steps,
                    guidance_scale: *guidance_scale,
                    temperature: *temperature,
                    top_k: *top_k,
                    top_p: *top_p,
                    audio_reference: None,
                };
                request.validate()?;
                validate_output_path(output)?;
                if let Some(path) = audio_reference {
                    read_codec_audio(path)?;
                }
            }
            Command::Understand {
                model,
                input,
                prompt,
                max_steps,
                seed,
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                anyhow::ensure!(
                    !input.is_empty() && input.len() <= 16,
                    "understanding accepts 1..16 clips"
                );
                let mut audio = Vec::new();
                let mut samples = 0usize;
                for path in input {
                    let clip = read_codec_audio(path)?;
                    samples = samples
                        .checked_add(clip.samples().len())
                        .context("input sample count overflow")?;
                    anyhow::ensure!(
                        samples <= MAX_CODEC_INPUT_SAMPLES,
                        "understanding clips exceed total sample budget"
                    );
                    audio.push(clip);
                }
                AudioUnderstandingRequest {
                    audio,
                    prompt: prompt.clone(),
                    max_steps: *max_steps,
                    seed: *seed,
                }
                .validate()?;
            }
            Command::Capabilities {
                model,
                sample_rate,
                channels,
                ..
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                AudioFormat::new(*sample_rate, *channels)?;
            }
            Command::Transcribe {
                model,
                input,
                language,
                max_tokens,
                turn_seconds,
                ..
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                check_language(language.as_deref())?;
                check_budget(*max_tokens, *turn_seconds)?;
                WavSource::open(input)?;
            }
            Command::Synthesize {
                model,
                text,
                output,
                description,
                voice,
                language,
                max_steps,
                conditioning,
                ..
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                if let Some(path) = conditioning {
                    read_conditioning(path)?.validate()?;
                }
                anyhow::ensure!(
                    !text.trim().is_empty() && text.len() <= 65536,
                    "text must contain 1..65536 bytes"
                );
                anyhow::ensure!(
                    *max_steps > 0 && *max_steps <= 32768,
                    "max-steps must be in 1..32768"
                );
                anyhow::ensure!(
                    description.as_ref().is_none_or(|v| v.len() <= 65536),
                    "description exceeds 65536 bytes"
                );
                anyhow::ensure!(
                    voice.as_ref().is_none_or(|v| !v.trim().is_empty()),
                    "voice must not be empty"
                );
                check_language(language.as_deref())?;
                validate_output_path(output)?;
            }
            Command::Realtime {
                model,
                input,
                output,
                tts_model,
                language,
                max_tokens,
                chunk_ms,
                window_ms,
                turn_seconds,
                ..
            } => {
                anyhow::ensure!(model.is_dir(), "model directory does not exist");
                anyhow::ensure!(
                    tts_model.as_ref().is_none_or(|v| v.is_dir()),
                    "TTS model directory does not exist"
                );
                check_language(language.as_deref())?;
                check_budget(*max_tokens, *turn_seconds)?;
                anyhow::ensure!(
                    (1..=30000).contains(chunk_ms),
                    "chunk-ms must be in 1..30000"
                );
                anyhow::ensure!(
                    (1..=30000).contains(window_ms),
                    "window-ms must be in 1..30000"
                );
                WavSource::open(input)?;
                if let Some(path) = output {
                    validate_output_path(path)?;
                }
            }
        }
        Ok(())
    }
}
struct WavSource {
    reader: hound::WavReader<BufReader<File>>,
    format: AudioFormat,
}
impl WavSource {
    fn open(path: &Path) -> Result<Self> {
        let reader =
            hound::WavReader::open(path).with_context(|| format!("reading {}", path.display()))?;
        let spec = reader.spec();
        let format = AudioFormat::new(spec.sample_rate, spec.channels)?;
        anyhow::ensure!(
            match spec.sample_format {
                hound::SampleFormat::Float => spec.bits_per_sample == 32,
                hound::SampleFormat::Int => (1..=32).contains(&spec.bits_per_sample),
            },
            "unsupported WAV sample representation"
        );
        Ok(Self { reader, format })
    }
    fn read(&mut self, frames: usize) -> Result<AudioBuffer> {
        let count = frames
            .checked_mul(self.format.channels() as usize)
            .context("WAV chunk size overflow")?;
        anyhow::ensure!(
            count <= MAX_CODEC_INPUT_SAMPLES,
            "WAV read exceeds16million samples; use smaller streaming chunks"
        );
        let spec = self.reader.spec();
        let pcm = match spec.sample_format {
            hound::SampleFormat::Float => self
                .reader
                .samples::<f32>()
                .take(count)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                let scale = 2f64.powi(spec.bits_per_sample as i32 - 1);
                self.reader
                    .samples::<i32>()
                    .take(count)
                    .map(|v| v.map(|x| (x as f64 / scale) as f32))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
        };
        Ok(AudioBuffer::new(self.format, pcm)?)
    }
}
struct PendingWav {
    temporary: tempfile::NamedTempFile,
    destination: PathBuf,
}
struct Output {
    wav: Option<hound::WavWriter<BufWriter<File>>>,
    pending: Option<PendingWav>,
    format: Option<AudioFormat>,
    started: Instant,
    audio_frames: u64,
}
impl Output {
    fn new(path: Option<&Path>, format: Option<AudioFormat>) -> Result<Self> {
        let (wav, pending) = match (path, format) {
            (Some(path), Some(format)) => {
                validate_output_path(path)?;
                let parent = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let temporary = tempfile::Builder::new()
                    .prefix(".candle-audio-")
                    .suffix(".wav.part")
                    .tempfile_in(parent)?;
                let writer = hound::WavWriter::new(
                    BufWriter::new(temporary.reopen()?),
                    hound::WavSpec {
                        channels: format.channels(),
                        sample_rate: format.sample_rate(),
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )?;
                (
                    Some(writer),
                    Some(PendingWav {
                        temporary,
                        destination: path.to_owned(),
                    }),
                )
            }
            (Some(_), None) => anyhow::bail!("selected model does not produce audio"),
            _ => (None, None),
        };
        Ok(Self {
            wav,
            pending,
            format,
            started: Instant::now(),
            audio_frames: 0,
        })
    }
    fn finish(mut self) -> Result<()> {
        if let Some(writer) = self.wav.take() {
            writer.finalize()?;
        }
        if let Some(pending) = self.pending.take() {
            pending.temporary.as_file().sync_all()?;
            pending
                .temporary
                .persist_noclobber(&pending.destination)
                .with_context(|| {
                    format!(
                        "publishing {} without overwriting another file",
                        pending.destination.display()
                    )
                })?;
        }
        Ok(())
    }
}
impl EventSink for Output {
    fn emit(&mut self, event: AudioEvent) -> candle_audio::Result<()> {
        let elapsed = self.started.elapsed().as_secs_f64();
        let json = match &event {
            AudioEvent::Audio { turn, audio } | AudioEvent::AudioChunk { turn, audio, .. } => {
                if self.format.is_some_and(|format| format != audio.format()) {
                    return Err(Error::InvalidInput(
                        "model changed WAV output format".into(),
                    ));
                }
                let start_frame = match &event {
                    AudioEvent::AudioChunk { start_frame, .. } => *start_frame,
                    _ => self.audio_frames,
                };
                self.audio_frames += audio.frames() as u64;
                if let Some(writer) = self.wav.as_mut() {
                    for &sample in audio.samples() {
                        writer
                            .write_sample((sample.clamp(-1., 1.) * 32767.).round() as i16)
                            .map_err(|e| Error::InvalidInput(e.to_string()))?;
                    }
                }
                serde_json::json!({"type":"audio","turn":turn,"format":audio.format(),"frames":audio.frames(),"start_frame":start_frame,"elapsed_seconds":elapsed})
            }
            _ => {
                let mut json = serde_json::to_value(event)?;
                json["elapsed_seconds"] = elapsed.into();
                json
            }
        };
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        serde_json::to_writer(&mut stdout, &json)?;
        writeln!(stdout)?;
        stdout.flush()?;
        Ok(())
    }
}
#[derive(Clone, Default)]
struct Interrupts {
    stop: CancellationToken,
    active: Arc<Mutex<Option<CancellationToken>>>,
}
impl Interrupts {
    fn install() -> Result<Self> {
        let interrupts = Self::default();
        let handler = interrupts.clone();
        ctrlc::set_handler(move || handler.cancel()).context("installing Ctrl+C cancellation")?;
        Ok(interrupts)
    }
    fn cancel(&self) {
        self.stop.cancel();
        if let Some(token) = self
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            token.cancel();
        }
    }
    fn activate(&self, token: CancellationToken) {
        // Publish under the same lock used by the signal handler before testing
        // the global flag, so a signal racing a turn rotation cannot miss it.
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        *active = Some(token);
        if self.stop.is_cancelled() {
            if let Some(token) = active.as_ref() {
                token.cancel();
            }
        }
    }
    fn check(&self) -> Result<()> {
        self.stop.check()?;
        Ok(())
    }
}
struct Timeline<'a, S> {
    sink: &'a mut S,
    input_offset_seconds: f64,
    output_offset_frames: u64,
    turn_output_frames: u64,
}
impl<S: EventSink> EventSink for Timeline<'_, S> {
    fn emit(&mut self, mut event: AudioEvent) -> candle_audio::Result<()> {
        match &mut event {
            AudioEvent::Transcript { transcript, .. }
            | AudioEvent::TranscriptSegment { transcript, .. } => {
                transcript.start += self.input_offset_seconds;
                transcript.end += self.input_offset_seconds;
            }
            AudioEvent::Audio { audio, .. } => self.turn_output_frames += audio.frames() as u64,
            AudioEvent::AudioChunk {
                audio, start_frame, ..
            } => {
                *start_frame = start_frame
                    .checked_add(self.output_offset_frames)
                    .ok_or_else(|| Error::LimitExceeded("audio clock overflow".into()))?;
                self.turn_output_frames += audio.frames() as u64;
            }
            _ => {}
        };
        self.sink.emit(event)
    }
}
#[derive(Clone)]
struct StreamSettings {
    prompt: Option<String>,
    chunk_ms: u32,
    turn_seconds: u32,
    pace: bool,
    preserve_conversation: bool,
}
fn complete_turn<M: RealtimeModel, S: EventSink>(
    session: &mut RealtimeSession<M>,
    sink: &mut Timeline<'_, S>,
    interrupts: &Interrupts,
    total_frames: u64,
    rate: usize,
) -> Result<()> {
    interrupts.check()?;
    interrupts.activate(session.cancellation_token());
    session.commit(sink)?;
    while session.output_pending() {
        interrupts.check()?;
        interrupts.activate(session.cancellation_token());
        if session.poll_output(sink)? == PollStatus::Idle {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    if let Some(reason) = session.last_finish_reason() {
        if matches!(reason, FinishReason::Length | FinishReason::Deadline) {
            eprintln!(
                "{}",
                serde_json::json!({"type":"turn_warning","finish_reason":reason,"input_end_seconds":total_frames as f64/rate as f64})
            );
        }
    }
    sink.input_offset_seconds = total_frames as f64 / rate as f64;
    sink.output_offset_frames += sink.turn_output_frames;
    sink.turn_output_frames = 0;
    interrupts.activate(session.cancellation_token());
    Ok(())
}
fn stream_file<M: RealtimeModel, S: EventSink>(
    model: M,
    source: &mut WavSource,
    output: &mut S,
    settings: StreamSettings,
    interrupts: &Interrupts,
) -> Result<()> {
    anyhow::ensure!(
        (1..=30000).contains(&settings.chunk_ms),
        "chunk-ms must be in 1..30000"
    );
    anyhow::ensure!(
        (1..=3600).contains(&settings.turn_seconds),
        "turn-seconds must be in 1..3600"
    );
    let rate = source.format.sample_rate() as usize;
    let channels = source.format.channels() as usize;
    let frames = (rate * settings.chunk_ms as usize / 1000)
        .max(1)
        .min(MAX_CODEC_INPUT_SAMPLES / channels);
    let mut session =
        RealtimeSession::new(model, rate * 30, rate * settings.turn_seconds as usize)?;
    let mut timeline = Timeline {
        sink: output,
        input_offset_seconds: 0.,
        output_offset_frames: 0,
        turn_output_frames: 0,
    };
    let start = Instant::now();
    let mut read = 0u64;
    loop {
        interrupts.check()?;
        let pcm = source.read(frames)?;
        if pcm.is_empty() {
            break;
        }
        let mut offset = 0;
        while offset < pcm.frames() {
            interrupts.check()?;
            if session.remaining_input_frames() == 0 {
                complete_turn(&mut session, &mut timeline, interrupts, read, rate)?;
                if !settings.preserve_conversation {
                    session.clear_context()?;
                    interrupts.activate(session.cancellation_token());
                }
            }
            if session.state() == SessionState::Ready {
                if let Some(prompt) = settings.prompt.as_deref() {
                    interrupts.activate(session.cancellation_token());
                    session.push_text(prompt, &mut timeline)?;
                }
            }
            let count = (pcm.frames() - offset).min(session.remaining_input_frames());
            if settings.pace {
                let due = Duration::from_secs_f64(read as f64 / rate as f64);
                while let Some(delay) = due.checked_sub(start.elapsed()) {
                    interrupts.check()?;
                    std::thread::sleep(delay.min(Duration::from_millis(20)));
                }
            }
            let part = AudioBuffer::new(
                source.format,
                pcm.samples()[offset * channels..(offset + count) * channels].to_vec(),
            )?;
            interrupts.activate(session.cancellation_token());
            session.push_audio(&part, &mut timeline)?;
            // Step independent output without waiting for a committed turn or
            // starving the next input block behind an unbounded output drain.
            for _ in 0..8 {
                if !session.output_pending() {
                    break;
                }
                interrupts.check()?;
                if matches!(
                    session.poll_output(&mut timeline)?,
                    PollStatus::Idle | PollStatus::Complete
                ) {
                    break;
                }
            }
            offset += count;
            read += count as u64;
        }
    }
    if session.state() == SessionState::Active {
        complete_turn(&mut session, &mut timeline, interrupts, read, rate)?;
    }
    interrupts.check()?;
    session.finish(&mut timeline)?;
    Ok(())
}
fn spec(
    directory: PathBuf,
    input: AudioFormat,
    language: Option<String>,
    max_tokens: Option<usize>,
    window_ms: u32,
) -> ModelSpec {
    let mut options = serde_json::json!({"transcription":{"language":language,"translate":false},"window_ms":window_ms});
    if let Some(maximum) = max_tokens {
        options["max_tokens"] = maximum.into();
        options["max_steps"] = maximum.into();
    }
    ModelSpec::new(directory)
        .with_input(input)
        .with_options(options)
}
fn load_realtime_input(
    registry: &ModelRegistry,
    spec: &ModelSpec,
    device: &Device,
    readback: bool,
    window_ms: u32,
) -> Result<Box<dyn RealtimeModel + Send>> {
    if !readback {
        return Ok(registry.realtime(spec, device)?);
    }
    let options = spec
        .options
        .get("transcription")
        .cloned()
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    let recognizer = registry.asr(spec, device)?;
    let window = (recognizer.input_format().sample_rate() as usize * window_ms as usize / 1000)
        .max(1)
        .min(recognizer.max_input_frames());
    Ok(Box::new(AsrStream::new(
        recognizer,
        spec.input_format
            .context("readback source format is required")?,
        window,
        options,
    )?))
}
fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .try_init();
    let args = Args::parse();
    args.validate()?;
    let settings = LoadSettings::new(&args)?;
    let tts_settings = LoadSettings::readback(&args)?;
    let registry = ModelRegistry::standard();
    if matches!(args.command, Command::Models) {
        let families = registry
            .families()
            .map(|(family, tasks)| serde_json::json!({"family":family,"tasks":tasks}))
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&families)?);
        return Ok(());
    }
    #[cfg(feature = "live-io")]
    if matches!(args.command, Command::Devices) {
        use candle_audio::live::{enumerate_devices, Direction};
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"input":enumerate_devices(Direction::Input)?,"output":enumerate_devices(Direction::Output)?})
            )?
        );
        return Ok(());
    }
    let interrupts = Interrupts::install()?;
    let device = match args.device {
        Backend::Cpu => Device::Cpu,
        Backend::Vulkan => Device::new_vulkan(args.gpu_index)?,
        Backend::Cuda => Device::new_cuda(args.gpu_index)?,
        Backend::Metal => Device::new_metal(args.gpu_index)?,
    };
    match args.command {
        Command::Models => unreachable!(),
        #[cfg(feature = "live-io")]
        Command::Devices => unreachable!(),
        #[cfg(feature = "live-io")]
        Command::Live {
            model,
            input_device,
            output_device,
            tts_model,
            prompt,
            turn_seconds,
        } => {
            use candle_audio::live::{resolve_format, DeviceSelector, Direction};
            let input = input_device.map(DeviceSelector::Id).unwrap_or_default();
            let output = output_device.map(DeviceSelector::Id).unwrap_or_default();
            let source = resolve_format(&input, Direction::Input, None)?;
            anyhow::ensure!(tts_model.is_none()||prompt.is_none(),"readback uses the recognition task; use understand/realtime without --tts-model for question prompts");
            let model = load_realtime_input(
                &registry,
                &settings.apply(ModelSpec::new(model).with_input(source)),
                &device,
                tts_model.is_some(),
                30000,
            )?;
            let model: Box<dyn RealtimeModel + Send> = if let Some(directory) = tts_model {
                anyhow::ensure!(
                    model.capabilities().text_output,
                    "readback requires recognized text"
                );
                let tts = registry.tts(&tts_settings.spec(directory), &device)?;
                Box::new(SpeechPipeline::new(
                    model,
                    tts,
                    |transcript: &Transcript| {
                        Ok((!transcript.text.is_empty())
                            .then(|| SynthesisRequest::new(&transcript.text)))
                    },
                ))
            } else {
                model
            };
            eprintln!("{}", serde_json::to_string(&model.descriptor())?);
            interrupts.activate(interrupts.stop.clone());
            live_cli::run(
                model,
                input,
                output,
                turn_seconds as f64,
                prompt,
                interrupts.stop.clone(),
            )?;
        }

        Command::Codec { command } => match command {
            CodecCommand::Encode {
                model,
                input,
                output,
                num_codebooks,
            } => {
                let audio = read_codec_audio(&input)?;
                let mut options = serde_json::json!({});
                if let Some(n) = num_codebooks {
                    options["num_codebooks"] = n.into();
                }
                let mut codec = registry.codec(
                    &settings.apply(ModelSpec::new(model).with_options(options)),
                    &device,
                )?;
                interrupts.check()?;
                interrupts.activate(interrupts.stop.clone());
                let codes = encode_codec(codec.as_mut(), &audio, &interrupts.stop)?;
                publish_codes(&output, &codes, &interrupts.stop)?;
                println!(
                    "{}",
                    serde_json::json!({"type":"codec_encoded","format":codes.format(),"codebook_lengths":codes.codebooks().iter().map(Vec::len).collect::<Vec<_>>(),"codebook_strides":codes.codebook_strides(),"output":output})
                );
            }
            CodecCommand::Decode {
                model,
                input,
                output,
            } => {
                let codes = read_codes(&input)?;
                let options = serde_json::json!({"num_codebooks":codes.format().num_codebooks});
                let mut codec = registry.codec(
                    &settings.apply(ModelSpec::new(model).with_options(options)),
                    &device,
                )?;
                interrupts.check()?;
                interrupts.activate(interrupts.stop.clone());
                let audio = codec.decode_multiscale(&codes, &interrupts.stop)?;
                interrupts.check()?;
                let mut sink = Output::new(Some(&output), Some(audio.format()))?;
                sink.emit(AudioEvent::Audio { turn: 0, audio })?;
                sink.finish()?;
            }
        },
        Command::GenerateAudio {
            model,
            prompt,
            output,
            max_steps,
            seed,
            guidance_scale,
            temperature,
            top_k,
            top_p,
            audio_reference,
        } => {
            let mut model = registry.generator(&settings.spec(model), &device)?;
            let descriptor = model.descriptor();
            let format = descriptor
                .output_formats
                .first()
                .copied()
                .context("generator does not advertise an output format")?;
            let request = AudioGenerationRequest {
                prompt,
                seed,
                max_steps,
                guidance_scale,
                temperature,
                top_k,
                top_p,
                audio_reference: if let Some(path) = audio_reference {
                    anyhow::ensure!(
                        descriptor.accepts_audio_reference,
                        "model does not support audio references"
                    );
                    let target = descriptor
                        .input_formats
                        .first()
                        .copied()
                        .context("generator does not advertise a reference format")?;
                    Some(convert_audio_reference(read_codec_audio(&path)?, target)?)
                } else {
                    None
                },
            };
            let mut sink = Output::new(Some(&output), Some(format))?;
            interrupts.activate(interrupts.stop.clone());
            interrupts.check()?;
            model.generate(&request, &interrupts.stop, &mut |audio| {
                sink.emit(AudioEvent::Audio { turn: 0, audio })
            })?;
            interrupts.check()?;
            anyhow::ensure!(sink.audio_frames > 0, "model returned no audio");
            let frames = sink.audio_frames;
            let finish_reason = model.finish_reason();
            sink.finish()?;
            println!(
                "{}",
                serde_json::json!({"type":"audio_generation_completed","finish_reason":finish_reason,"frames":frames,"output":output})
            );
        }
        Command::Understand {
            model,
            input,
            prompt,
            max_steps,
            seed,
        } => {
            let audio = input
                .iter()
                .map(|path| read_codec_audio(path))
                .collect::<Result<Vec<_>>>()?;
            let mut model = registry.understanding(&settings.spec(model), &device)?;
            let mut sink = Output::new(None, None)?;
            interrupts.activate(interrupts.stop.clone());
            interrupts.check()?;
            let result = model.understand(
                &AudioUnderstandingRequest {
                    audio,
                    prompt,
                    max_steps,
                    seed,
                },
                &interrupts.stop,
                &mut sink,
            )?;
            interrupts.check()?;
            println!(
                "{}",
                serde_json::json!({"type":"understanding_completed","result":result})
            );
        }
        Command::Capabilities {
            model,
            task,
            sample_rate,
            channels,
        } => {
            let model = registry.load(
                &settings.apply(
                    ModelSpec::new(model).with_input(AudioFormat::new(sample_rate, channels)?),
                ),
                task.audio_task(),
                &device,
            )?;
            println!("{}", serde_json::to_string_pretty(&model.descriptor())?);
        }
        Command::Transcribe {
            model,
            input,
            language,
            translate,
            max_tokens,
            turn_seconds,
        } => {
            let mut source = WavSource::open(&input)?;
            let spec = settings.apply(spec(
                model,
                source.format,
                language.clone(),
                max_tokens,
                30000,
            ));
            let model = registry.asr(&spec, &device)?;
            interrupts.check()?;
            let window = model.max_input_frames();
            let stream = AsrStream::new(
                model,
                source.format,
                window,
                TranscriptionOptions {
                    language,
                    translate,
                },
            )?;
            let mut output = Output::new(None, None)?;
            stream_file(
                stream,
                &mut source,
                &mut output,
                StreamSettings {
                    prompt: None,
                    chunk_ms: 100,
                    turn_seconds,
                    pace: false,
                    preserve_conversation: false,
                },
                &interrupts,
            )?;
            output.finish()?;
        }
        Command::Synthesize {
            model,
            text,
            output,
            description,
            voice,
            language,
            seed,
            max_steps,
            conditioning,
        } => {
            let mut model = registry.tts(&settings.spec(model), &device)?;
            interrupts.check()?;
            let request = SynthesisRequest {
                text,
                description,
                voice,
                language,
                seed,
                max_steps,
            };
            let mut sink = Output::new(Some(&output), Some(model.output_format()))?;
            interrupts.activate(interrupts.stop.clone());
            if let Some(path) = conditioning {
                let mut conditioned = read_conditioning(&path)?;
                conditioned.request = request;
                model.synthesize_conditioned(&conditioned, &interrupts.stop, &mut sink)?;
            } else {
                model.synthesize(&request, &interrupts.stop, &mut sink)?;
            }
            interrupts.check()?;
            anyhow::ensure!(sink.audio_frames > 0, "model returned no audio");
            let finish_reason = model.finish_reason();
            let frames = sink.audio_frames;
            sink.finish()?;
            println!(
                "{}",
                serde_json::json!({"type":"synthesis_completed","finish_reason":finish_reason,
                "truncated":finish_reason==FinishReason::Length,"frames":frames,"output":output})
            );
            if finish_reason == FinishReason::Length {
                eprintln!("generation reached its step budget; the published WAV contains partial speech (finish_reason=length)");
            }
        }
        Command::Realtime {
            model,
            input,
            output,
            tts_model,
            prompt,
            language,
            max_tokens,
            chunk_ms,
            window_ms,
            turn_seconds,
            pace,
            preserve_conversation,
        } => {
            let mut source = WavSource::open(&input)?;
            anyhow::ensure!(tts_model.is_none()||prompt.is_none(),"readback uses the recognition task; use understand/realtime without --tts-model for question prompts");
            let model = load_realtime_input(
                &registry,
                &settings.apply(spec(model, source.format, language, max_tokens, window_ms)),
                &device,
                tts_model.is_some(),
                window_ms,
            )?;
            let model: Box<dyn RealtimeModel + Send> = if let Some(directory) = tts_model {
                anyhow::ensure!(
                    model.capabilities().text_output,
                    "readback needs a realtime recognizer with text output"
                );
                let tts = registry.tts(&tts_settings.spec(directory), &device)?;
                Box::new(SpeechPipeline::with_source_audio_policy(
                    model,
                    tts,
                    |transcript: &Transcript| {
                        Ok((!transcript.text.is_empty())
                            .then(|| SynthesisRequest::new(&transcript.text)))
                    },
                    SourceAudioPolicy::Discard,
                )?)
            } else {
                model
            };
            interrupts.check()?;
            let caps = model.capabilities();
            anyhow::ensure!(
                caps.input_format == Some(source.format),
                "selected realtime model does not accept this audio input"
            );
            eprintln!("{}", serde_json::to_string(&model.descriptor())?);
            let mut sink = Output::new(output.as_deref(), caps.output_format)?;
            stream_file(
                model,
                &mut source,
                &mut sink,
                StreamSettings {
                    prompt,
                    chunk_ms,
                    turn_seconds,
                    pace,
                    preserve_conversation,
                },
                &interrupts,
            )?;
            sink.finish()?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "candle-audio-cli-output-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn existing_output_and_hardlink_preserve_source() {
        let fixture = Fixture::new();
        let source = fixture.0.join("source.wav");
        let alias = fixture.0.join("alias.wav");
        let original = b"recording that must survive";
        std::fs::write(&source, original).unwrap();
        std::fs::hard_link(&source, &alias).unwrap();
        for path in [&source, &alias] {
            let error = Output::new(Some(path), Some(AudioFormat::mono(24_000).unwrap()))
                .err()
                .expect("existing output must be rejected");
            assert!(error.to_string().contains("choose a new path"));
            assert_eq!(std::fs::read(&source).unwrap(), original);
            assert_eq!(std::fs::read(&alias).unwrap(), original);
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlink_output_preserves_source() {
        let fixture = Fixture::new();
        let source = fixture.0.join("source.wav");
        let alias = fixture.0.join("alias.wav");
        let original = b"recording that must survive";
        std::fs::write(&source, original).unwrap();
        std::os::unix::fs::symlink(&source, &alias).unwrap();
        assert!(Output::new(Some(&alias), Some(AudioFormat::mono(24_000).unwrap())).is_err());
        assert_eq!(std::fs::read(&source).unwrap(), original);
        assert!(std::fs::symlink_metadata(alias).unwrap().is_symlink());
    }

    #[test]
    fn fresh_output_is_a_valid_wav() {
        let fixture = Fixture::new();
        let path = fixture.0.join("new.wav");
        let format = AudioFormat::mono(24_000).unwrap();
        let mut output = Output::new(Some(&path), Some(format)).unwrap();
        output
            .emit(AudioEvent::Audio {
                turn: 0,
                audio: AudioBuffer::new(format, vec![0., 0.5, -0.5]).unwrap(),
            })
            .unwrap();
        output.finish().unwrap();
        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().sample_rate, 24_000);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.duration(), 3);
        assert_eq!(
            reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
            [0, 16384, -16384]
        );
    }
    #[test]
    fn failed_or_cancelled_output_is_not_published() {
        let fixture = Fixture::new();
        let path = fixture.0.join("failed.wav");
        {
            let mut output =
                Output::new(Some(&path), Some(AudioFormat::mono(24000).unwrap())).unwrap();
            output
                .emit(AudioEvent::AudioChunk {
                    turn: 0,
                    audio: AudioBuffer::new(AudioFormat::mono(24000).unwrap(), vec![0.2; 32])
                        .unwrap(),
                    start_frame: 0,
                })
                .unwrap();
            assert!(!path.exists(), "WAV must remain private until finalized");
            assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 1);
            // Model/sink failures and cancellation unwind this same output owner.
        }
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 0);
    }
    #[test]
    fn atomic_publication_never_overwrites_a_racing_file() {
        let fixture = Fixture::new();
        let path = fixture.0.join("racing.wav");
        let output = Output::new(Some(&path), Some(AudioFormat::mono(24000).unwrap())).unwrap();
        std::fs::write(&path, b"someone else's completed file").unwrap();
        assert!(output.finish().is_err());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"someone else's completed file"
        );
        assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 1);
    }
    #[test]
    fn ctrl_c_cancels_active_and_subsequently_rotated_tokens() {
        let interrupts = Interrupts::default();
        let first = CancellationToken::default();
        interrupts.activate(first.clone());
        interrupts.cancel();
        assert!(first.is_cancelled());
        assert!(interrupts.check().is_err());
        let next = CancellationToken::default();
        interrupts.activate(next.clone());
        assert!(
            next.is_cancelled(),
            "signal between turns must also cancel a fresh token"
        );
    }
    #[test]
    fn readback_loads_recognition_instead_of_treating_text_deltas_as_transcripts() {
        use candle_audio::models::registry::{LoadedModel, ModelLoader};
        struct Recognizer;
        impl AsrModel for Recognizer {
            fn input_format(&self) -> AudioFormat {
                AudioFormat::mono(16000).unwrap()
            }
            fn max_input_frames(&self) -> usize {
                16000
            }
            fn transcribe(
                &mut self,
                audio: &AudioBuffer,
                _: &TranscriptionOptions,
                _: &CancellationToken,
            ) -> candle_audio::Result<Transcript> {
                Ok(Transcript {
                    text: "recognized".into(),
                    start: 0.,
                    end: audio.duration(),
                })
            }
            fn reset(&mut self) -> candle_audio::Result<()> {
                Ok(())
            }
        }
        struct Loader(Arc<Mutex<Vec<AudioTask>>>);
        impl ModelLoader for Loader {
            fn family(&self) -> &str {
                "hybrid"
            }
            fn tasks(&self) -> &[AudioTask] {
                &[AudioTask::Recognition, AudioTask::Realtime]
            }
            fn load(
                &self,
                _: &ModelSpec,
                task: AudioTask,
                _: &Device,
            ) -> candle_audio::Result<LoadedModel> {
                self.0.lock().unwrap().push(task);
                assert_eq!(task, AudioTask::Recognition);
                Ok(LoadedModel::Asr(Box::new(Recognizer)))
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ModelRegistry::new();
        registry.register(Loader(seen.clone())).unwrap();
        let source = AudioFormat::mono(16000).unwrap();
        let spec = ModelSpec::new("unused")
            .with_family("hybrid")
            .with_input(source);
        let mut model = load_realtime_input(&registry, &spec, &Device::Cpu, true, 1000).unwrap();
        let mut texts = Vec::new();
        model
            .push_audio(
                &AudioBuffer::new(source, vec![0.; 16000]).unwrap(),
                &CancellationToken::default(),
                &mut |event| {
                    if let AudioEvent::Transcript { transcript, .. } = event {
                        texts.push(transcript.text);
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(*seen.lock().unwrap(), [AudioTask::Recognition]);
        assert_eq!(texts, ["recognized"]);
    }
    #[test]
    fn readback_precision_and_loader_options_are_independent_of_primary_model() {
        let fixture = Fixture::new();
        let primary = fixture.0.join("primary.json");
        let readback = fixture.0.join("readback.json");
        std::fs::write(&primary, r#"{"max_context":2048}"#).unwrap();
        std::fs::write(
            &readback,
            r#"{"speaker_embedding_file":"speaker.safetensors"}"#,
        )
        .unwrap();
        let args = Args::try_parse_from([
            "candle-audio",
            "--dtype",
            "bf16",
            "--model-options",
            primary.to_str().unwrap(),
            "--tts-model-options",
            readback.to_str().unwrap(),
            "models",
        ])
        .unwrap();
        let primary = LoadSettings::new(&args).unwrap().spec("primary".into());
        let readback = LoadSettings::readback(&args).unwrap().spec("tts".into());
        assert_eq!(primary.dtype, candle::DType::BF16);
        assert_eq!(readback.dtype, candle::DType::F32);
        assert_eq!(primary.options["max_context"], 2048);
        assert!(readback.options.get("max_context").is_none());
        assert_eq!(
            readback.options["speaker_embedding_file"],
            "speaker.safetensors"
        );
    }
    #[test]
    fn invalid_request_is_rejected_before_any_output_file() {
        let fixture = Fixture::new();
        let model = fixture.0.join("model");
        std::fs::create_dir(&model).unwrap();
        let path = fixture.0.join("invalid.wav");
        let args = Args {
            device: Backend::Cpu,
            gpu_index: 0,
            dtype: Precision::F32,
            model_options: None,
            tts_dtype: Precision::F32,
            tts_model_options: None,
            command: Command::Synthesize {
                model,
                text: " ".into(),
                output: path.clone(),
                description: None,
                voice: None,
                language: None,
                seed: 0,
                max_steps: 512,
                conditioning: None,
            },
        };
        assert!(args.validate().is_err());
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 1);
    }
    struct SmallContext {
        frames: usize,
        turns: Arc<Mutex<Vec<usize>>>,
        pending: bool,
        clears: Arc<AtomicU64>,
    }
    impl RealtimeModel for SmallContext {
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities {
                input_format: Some(AudioFormat::mono(10).unwrap()),
                output_format: None,
                max_input_frames_per_turn: Some(4),
                text_input: false,
                text_output: true,
                streaming: StreamingMode::Native,
                output_streaming: None,
            }
        }
        fn push_audio(
            &mut self,
            a: &AudioBuffer,
            c: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> candle_audio::Result<()> {
            c.check()?;
            self.frames += a.frames();
            Ok(())
        }
        fn commit(
            &mut self,
            c: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> candle_audio::Result<()> {
            c.check()?;
            self.pending = true;
            Ok(())
        }
        fn output_pending(&self) -> bool {
            self.pending
        }
        fn poll_output(
            &mut self,
            c: &CancellationToken,
            s: &mut dyn EventSink,
        ) -> candle_audio::Result<PollStatus> {
            c.check()?;
            s.emit(AudioEvent::Transcript {
                turn: 99,
                transcript: Transcript {
                    text: self.frames.to_string(),
                    start: 0.,
                    end: self.frames as f64 / 10.,
                },
            })?;
            self.pending = false;
            Ok(PollStatus::Complete)
        }
        fn reset_conversation(&mut self) -> candle_audio::Result<()> {
            self.clears.fetch_add(1, Ordering::SeqCst);
            self.reset()
        }
        fn reset(&mut self) -> candle_audio::Result<()> {
            if self.frames > 0 {
                self.turns.lock().unwrap().push(self.frames);
            }
            self.frames = 0;
            self.pending = false;
            Ok(())
        }
    }
    #[test]
    fn file_chunks_split_across_context_turns_without_losing_frames_or_timestamps() {
        let fixture = Fixture::new();
        let input = fixture.0.join("input.wav");
        let mut wav = hound::WavWriter::create(
            &input,
            hound::WavSpec {
                channels: 1,
                sample_rate: 10,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..11 {
            wav.write_sample(1i16).unwrap();
        }
        wav.finalize().unwrap();
        for preserve_conversation in [false, true] {
            let turns = Arc::new(Mutex::new(Vec::new()));
            let clears = Arc::new(AtomicU64::new(0));
            let model = SmallContext {
                frames: 0,
                turns: turns.clone(),
                pending: false,
                clears: clears.clone(),
            };
            let mut events = Vec::new();
            stream_file(
                model,
                &mut WavSource::open(&input).unwrap(),
                &mut |e| {
                    events.push(e);
                    Ok(())
                },
                StreamSettings {
                    prompt: None,
                    chunk_ms: 1000,
                    turn_seconds: 30,
                    pace: false,
                    preserve_conversation,
                },
                &Interrupts::default(),
            )
            .unwrap();
            assert_eq!(*turns.lock().unwrap(), vec![4, 4, 3]);
            assert_eq!(
                clears.load(Ordering::SeqCst),
                if preserve_conversation { 1 } else { 3 },
                "only default file rollover clears completed context; final close always clears it"
            );
            let transcripts = events
                .iter()
                .filter_map(|e| {
                    if let AudioEvent::Transcript { turn, transcript } = e {
                        Some((*turn, transcript.start, transcript.end))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(transcripts.len(), 3);
            for ((turn, start, end), (expected_turn, expected_start, expected_end)) in transcripts
                .into_iter()
                .zip([(0, 0., 0.4), (1, 0.4, 0.8), (2, 0.8, 1.1)])
            {
                assert_eq!(turn, expected_turn);
                assert!((start - expected_start).abs() < 1e-9);
                assert!((end - expected_end).abs() < 1e-9);
            }
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, AudioEvent::TurnCompleted { .. }))
                    .count(),
                3
            );
        }
    }
    struct RaggedCodec;
    impl AudioCodec for RaggedCodec {
        fn format(&self) -> CodecFormat {
            CodecFormat {
                audio_format: AudioFormat::mono(10).unwrap(),
                frame_rate: 5.,
                num_codebooks: 2,
                cardinality: 8,
            }
        }
        fn encode(
            &mut self,
            _: &AudioBuffer,
            _: &CancellationToken,
        ) -> candle_audio::Result<AudioCodes> {
            Err(Error::Unsupported(
                "test codec uses ragged codebooks".into(),
            ))
        }
        fn decode(
            &mut self,
            _: &AudioCodes,
            _: &CancellationToken,
        ) -> candle_audio::Result<AudioBuffer> {
            Err(Error::Unsupported(
                "test codec uses ragged codebooks".into(),
            ))
        }
        fn reset(&mut self) -> candle_audio::Result<()> {
            Ok(())
        }
        fn encode_multiscale(
            &mut self,
            audio: &AudioBuffer,
            cancel: &CancellationToken,
        ) -> candle_audio::Result<MultiScaleAudioCodes> {
            cancel.check()?;
            assert_eq!(audio.format(), self.format().audio_format);
            assert_eq!(audio.frames(), 8);
            assert!(audio.samples().iter().all(|v| (*v - 0.4).abs() < 1e-6));
            MultiScaleAudioCodes::new(
                self.format(),
                vec![1, 2],
                vec![vec![1, 2, 3, 4], vec![5, 6]],
            )?
            .with_original_audio_frames(8)
        }
        fn decode_multiscale(
            &mut self,
            codes: &MultiScaleAudioCodes,
            cancel: &CancellationToken,
        ) -> candle_audio::Result<AudioBuffer> {
            cancel.check()?;
            assert_eq!(codes.codebook_strides(), &[1, 2]);
            assert_eq!(codes.codebooks(), &[vec![1, 2, 3, 4], vec![5, 6]]);
            AudioBuffer::new(
                self.format().audio_format,
                vec![0.25; codes.original_audio_frames().unwrap()],
            )
        }
    }
    #[test]
    fn codec_json_preserves_ragged_scales_and_exact_original_length() {
        let fixture = Fixture::new();
        let path = fixture.0.join("codes.json");
        let mut codec = RaggedCodec;
        let audio =
            AudioBuffer::new(AudioFormat::new(10, 2).unwrap(), [0.2, 0.6].repeat(8)).unwrap();
        let codes = encode_codec(&mut codec, &audio, &CancellationToken::default()).unwrap();
        publish_codes(&path, &codes, &CancellationToken::default()).unwrap();
        let reloaded = read_codes(&path).unwrap();
        assert_eq!(reloaded, codes);
        let decoded = codec
            .decode_multiscale(&reloaded, &CancellationToken::default())
            .unwrap();
        assert_eq!(decoded.frames(), 8);
        assert_eq!(decoded.format(), AudioFormat::mono(10).unwrap());
        assert!(publish_codes(&path, &codes, &CancellationToken::default()).is_err());
        assert_eq!(read_codes(&path).unwrap(), codes);
    }
    #[test]
    fn codec_preflight_rejects_oversized_wav_and_json() {
        let fixture = Fixture::new();
        let wav = fixture.0.join("long.wav");
        let mut out = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 10,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..301 {
            out.write_sample(0i16).unwrap();
        }
        out.finalize().unwrap();
        assert!(read_codec_audio(&wav).is_err());
        let json = fixture.0.join("too-large.json");
        File::create(&json)
            .unwrap()
            .set_len(MAX_CODEC_JSON_BYTES as u64 + 1)
            .unwrap();
        assert!(read_codes(&json).is_err());
    }
}
