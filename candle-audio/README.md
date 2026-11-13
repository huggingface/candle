# Candle audio

Native Rust APIs and a local CLI for speech recognition, synthesis, and realtime
models. Models implement `AsrModel`, `TtsModel`, or `RealtimeModel`; PCM conversion,
turn handling, cancellation, backpressure, and application code are shared.
There is no HTTP/WebSocket server or required microphone dependency.

| Adapter | Task | Streaming behavior |
| --- | --- | --- |
| Whisper, including GGUF weights | Transcription and translation | Bounded, non-overlapping recognition windows |
| Parler-TTS | Descriptively conditioned synthesis | Buffered generation and DAC decoding; then output chunks |
| Mimi v0.1 | Neural codec reconstruction | Native causal audio input/output |
| Your `RealtimeModel` implementation | Live ASR, duplex speech, or other audio models | Defined by the model's capabilities |

These adapters run Candle tensor operations on the selected device. The generic
interfaces do not make unsupported architectures loadable: LFM3 audio, for
example, still needs its own native model/codec implementation and adapter.

## CLI

Build from the workspace with `cargo build -p candle-audio --release --features cli`.
Add `vulkan`, `cuda`, or `metal` to enable the corresponding device. CPU is the
default. Checkpoints are local directories; no model or executable code is
downloaded by this crate. Each directory contains `config.json` and
`model.safetensors` (or its shard index and shards). Whisper and Parler also need
`tokenizer.json`. A Whisper directory can instead contain `model.gguf`.

```sh
cargo run -p candle-audio --release --features cli,vulkan -- --device vulkan \
  asr --model /models/whisper-tiny --input speech.wav --language en

cargo run -p candle-audio --release --features cli,vulkan -- --device vulkan \
  tts --model /models/parler-tts-mini-v1 --text "Hello from Candle." \
  --description "A clear female voice in a quiet recording." --output speech.wav

# Incremental recognition, with 100 ms file input paced by its audio clock.
cargo run -p candle-audio --release --features cli,vulkan -- --device vulkan \
  realtime --model /models/whisper-tiny --input speech.wav \
  --window-ms 5000 --chunk-ms 100 --pace

# Native causal codec reconstruction. This is not a conversational model.
cargo run -p candle-audio --release --features cli,vulkan -- --device vulkan \
  realtime --model /models/mimi --input speech.wav --output reconstructed.wav
```

`realtime --tts-model /models/parler-tts-mini-v1` reads back recognized segments.
Throughput depends on the model, backend, and hardware; use release builds for
inference. Streaming capabilities describe when a model can emit output.
Applications that want conversational replies can supply a response function to
`SpeechPipeline`, or implement `RealtimeModel` for a native speech model.
Whisper window boundaries can split words: this adapter does not advertise
causal live transcription or word-level timestamps. Parler's `--description`
controls the voice; unsupported named voices/languages are rejected.

Input supports PCM integer and float WAV files, 1–64 channels, and
sample rates up to 384 kHz. Source channels are averaged and a stateful sinc
resampler converts to each model's rate. Realtime WAV input is read in chunks,
not collected into an unbounded recording. Output WAV is PCM16, clamped to its
representable range. Output paths must not already exist; this also protects
source recordings from accidental overwrite. Standard output contains JSON-line
events; audio events report frame counts instead of embedding samples. Realtime capabilities go to
standard error. Library sinks receive the actual f32 samples.

## Native API

For an application outside this workspace, add path dependencies on this
checkout (the `candle` alias is used by the example below):

```toml
[dependencies]
candle-audio = { path = "/path/to/candle/candle-audio" }
candle = { package = "candle-core", path = "/path/to/candle/candle-core" }
```

```rust,no_run
use candle_audio::{models::load, AsrStream, AudioBuffer, AudioFormat,
    RealtimeSession, TranscriptionOptions};
use candle::Device;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let model = load::asr(std::path::Path::new("/models/whisper-tiny"), &Device::Cpu)?;
let source = AudioFormat::new(48_000, 2)?;
let recognizer = AsrStream::new(model, source, 16_000 * 5,
    TranscriptionOptions::default())?;
let mut session = RealtimeSession::new(recognizer, 48_000, 48_000 * 120)?;
# fn assert_send<T: Send>(_: &T) {}
# assert_send(&session);
let cancel = session.cancellation_token(); // safe to move into another thread
let mut sink = |event| { println!("{event:?}"); Ok(()) };
session.push_audio(&AudioBuffer::new(source, vec![0.; 4_800 * 2])?, &mut sink)?;
session.finish(&mut sink)?; // drains resampler and the short final ASR window
# Ok(())
# }
```

`TtsStream` drives a synthesizer through the same session using `push_text`.
`SpeechPipeline` composes recognition, a caller-provided text response function,
and synthesis. No vendor dispatch is part of the session implementation. A new
native streaming model implements `capabilities`, `push_audio`/`push_text`,
`commit`, and `reset`, emitting typed audio/text/transcript events to an
`EventSink`. Model events have their turn IDs assigned by the session. Only the
session emits completion/cancellation events.

Capabilities distinguish recognition/input streaming from audio-output
streaming, including a buffered synthesizer composed with a streaming
recognizer. A model can declare a maximum source-frame count per turn; sessions
enforce the stricter of that context limit and the application's configured
limit. Mimi reserves room to flush its final codec frame within its rotary
position context. Commit a turn and start a new one before reaching that limit.

Sinks are synchronous and fallible: slow consumers apply backpressure, and a
sink error stops generation. Session frame limits reject oversized input before
mutating model state. Model/sink errors fault a session until explicit `reset`.
Cancellation is cooperative at model steps, not a preemption of an in-flight
GPU kernel. A cancelled token stays cancelled; the next turn gets a fresh token.
Cancel/reset discards pending input. `commit` and `finish` flush it. A closed
session rejects further input. Audio callbacks are not suitable for an OS audio
realtime-priority thread; drive inference on a worker thread.

Local loaders retain `Send` on their model trait objects, so applications can
load a model first and then move its session onto an inference worker. Custom
models can use the traits without a `Send` requirement when they stay local.

## Validation

`cargo test -p candle-audio` runs deterministic PCM, anti-aliasing, chunk-boundary,
tail-drain, session, backpressure, cancellation, and composition tests without
model downloads. The bundled Whisper mel filter assets are copied unchanged
from Candle's existing Whisper example.

Real checkpoint tests are opt-in, fail if their fixtures are missing, and use
the native adapters:

```sh
CANDLE_AUDIO_TEST_MODELS=/models CANDLE_AUDIO_TEST_WAV=/data/jfk.wav \
  CANDLE_AUDIO_TEST_DEVICE=vulkan \
  cargo test -p candle-audio --features cli,vulkan --test pretrained -- --ignored --test-threads=1
```

The models directory must contain `whisper-tiny`, `whisper-tiny-q80`,
`parler-tts-mini-v1`, and `mimi`. The Q8 directory uses the tiny config/tokenizer
and `model-tiny-q80.gguf` from `lmz/candle-whisper`, renamed to the local loader's
standard filenames (`config.json`, `tokenizer.json`, `model.gguf`).
The WAV fixture is the standard 11-second JFK sample from whisper.cpp. See the
test source for exact assertions and runtime evidence in `bench_results/audio-runtime`
when working in the development checkout.
Set `CANDLE_AUDIO_TEST_OUTPUT_DIR` to keep generated TTS WAV files. The TTS
checkpoint test drives the public CLI and verifies its event frame counts and
WAV output before checking native cancellation.

Vulkan-specific audio regressions run with `cargo test -p candle-audio
--features vulkan --test backend_audio -- --ignored`. They check ID-view casts
used by audio codebooks and DAC's broadcast Snake activation against CPU output.
