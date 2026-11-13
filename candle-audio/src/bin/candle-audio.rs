use anyhow::{Context, Result};
use candle::Device;
use candle_audio::{models::load, *};
use clap::{Parser, Subcommand, ValueEnum};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
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
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, ValueEnum)]
enum Backend {
    Cpu,
    Vulkan,
    Cuda,
    Metal,
}
#[derive(Subcommand)]
enum Command {
    /// Transcribe WAV audio in bounded recognition windows; emits JSON lines.
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
    },
    /// Synthesize speech into a PCM16 WAV file.
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
    },
    /// Stream WAV chunks through an ASR or native realtime codec model.
    Realtime {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        /// Optional synthesizer to read back recognized speech.
        #[arg(long)]
        tts_model: Option<PathBuf>,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = 100)]
        chunk_ms: u32,
        #[arg(long, default_value_t = 5000)]
        window_ms: u32,
        /// Pace file input by its audio clock. This does not capture a microphone.
        #[arg(long)]
        pace: bool,
    },
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
        Ok(Self { reader, format })
    }
    fn read(&mut self, frames: usize) -> Result<AudioBuffer> {
        let count = frames
            .checked_mul(self.format.channels() as usize)
            .context("WAV chunk size overflow")?;
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

struct Output {
    wav: Option<hound::WavWriter<std::io::BufWriter<File>>>,
    started: Instant,
    audio_frames: u64,
}
impl Output {
    fn new(path: Option<&Path>, format: Option<AudioFormat>) -> Result<Self> {
        let wav = match (path, format) {
            (Some(path), Some(format)) => {
                // Atomic exclusive creation also rejects symlinks and hardlinks
                // to the input, preventing accidental destruction of recordings.
                let file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .with_context(|| {
                        format!(
                            "creating output {}: choose a new path; existing files are never overwritten",
                            path.display()
                        )
                    })?;
                Some(hound::WavWriter::new(
                    BufWriter::new(file),
                    hound::WavSpec {
                        channels: format.channels(),
                        sample_rate: format.sample_rate(),
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )?)
            }
            (Some(_), None) => anyhow::bail!("selected model does not produce audio"),
            _ => None,
        };
        Ok(Self {
            wav,
            started: Instant::now(),
            audio_frames: 0,
        })
    }
    fn finish(mut self) -> Result<()> {
        if let Some(wav) = self.wav.take() {
            wav.finalize()?;
        }
        Ok(())
    }
}
impl EventSink for Output {
    fn emit(&mut self, event: AudioEvent) -> candle_audio::Result<()> {
        let elapsed = self.started.elapsed().as_secs_f64();
        let json = if let AudioEvent::Audio { turn, audio } = &event {
            self.audio_frames += audio.frames() as u64;
            if let Some(wav) = self.wav.as_mut() {
                for &sample in audio.samples() {
                    wav.write_sample((sample.clamp(-1., 1.) * 32767.).round() as i16)
                        .map_err(|e| Error::InvalidInput(e.to_string()))?;
                }
            }
            serde_json::json!({"type":"audio","turn":turn,"format":audio.format(),"frames":audio.frames(),"elapsed_seconds":elapsed})
        } else {
            let mut json = serde_json::to_value(event)?;
            json["elapsed_seconds"] = elapsed.into();
            json
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

fn stream_file<M: RealtimeModel>(
    model: M,
    source: &mut WavSource,
    output: &mut Output,
    chunk_ms: u32,
    pace: bool,
) -> Result<()> {
    anyhow::ensure!(
        (1..=30_000).contains(&chunk_ms),
        "chunk-ms must be in 1..30000"
    );
    let rate = source.format.sample_rate() as usize;
    let frames = (rate * chunk_ms as usize / 1000).max(1);
    let mut session = RealtimeSession::new(model, rate * 30, rate * 3600)?;
    let start = Instant::now();
    let mut read = 0u64;
    loop {
        let pcm = source.read(frames)?;
        if pcm.is_empty() {
            break;
        }
        if pace {
            let due = Duration::from_secs_f64(read as f64 / rate as f64);
            if let Some(delay) = due.checked_sub(start.elapsed()) {
                std::thread::sleep(delay);
            }
        }
        read += pcm.frames() as u64;
        session.push_audio(&pcm, output)?;
    }
    session.finish(output)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device {
        Backend::Cpu => Device::Cpu,
        Backend::Vulkan => Device::new_vulkan(args.gpu_index)?,
        Backend::Cuda => Device::new_cuda(args.gpu_index)?,
        Backend::Metal => Device::new_metal(args.gpu_index)?,
    };
    match args.command {
        Command::Transcribe {
            model,
            input,
            language,
            translate,
        } => {
            let mut source = WavSource::open(&input)?;
            let model = load::asr(&model, &device)?;
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
            stream_file(stream, &mut source, &mut output, 100, false)?;
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
        } => {
            let mut model = load::tts(&model, &device)?;
            let mut sink = Output::new(Some(&output), Some(model.output_format()))?;
            let request = SynthesisRequest {
                text,
                description,
                voice,
                language,
                seed,
                max_steps,
            };
            model.synthesize(&request, &CancellationToken::default(), &mut sink)?;
            anyhow::ensure!(sink.audio_frames > 0, "model returned no audio");
            sink.finish()?;
        }
        Command::Realtime {
            model,
            input,
            output,
            tts_model,
            language,
            chunk_ms,
            window_ms,
            pace,
        } => {
            let mut source = WavSource::open(&input)?;
            let kind = load::model_type(&model)?;
            let model:Box<dyn RealtimeModel>=match kind.as_str() {
                "whisper"=>{
                    let asr=load::asr(&model,&device)?;
                    let window=(asr.input_format().sample_rate() as u64*window_ms as u64/1000) as usize;
                    let stream=AsrStream::new(asr,source.format,window,TranscriptionOptions{language,translate:false})?;
                    if let Some(tts)=tts_model {
                        let tts=load::tts(&tts,&device)?;
                        Box::new(SpeechPipeline::new(stream,tts,|transcript:&Transcript| {
                            Ok((!transcript.text.is_empty()).then(||SynthesisRequest::new(&transcript.text)))
                        }))
                    } else {Box::new(stream)}
                }
                "mimi"=>{anyhow::ensure!(tts_model.is_none(),"Mimi is a codec, not an ASR model"); load::codec(&model,&device,source.format)?}
                other=>anyhow::bail!("unsupported realtime model {other:?}; available: whisper (windowed), mimi (native codec)"),
            };
            let caps = model.capabilities();
            eprintln!("{}", serde_json::to_string(&caps)?);
            let mut sink = Output::new(output.as_deref(), caps.output_format)?;
            stream_file(model, &mut source, &mut sink, chunk_ms, pace)?;
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
}
