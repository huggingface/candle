# Candle audio

Native Rust APIs and a local CLI for speech recognition, synthesis, and realtime
models. Models implement `AsrModel`, `TtsModel`, or `RealtimeModel`; PCM conversion,
turn handling, cancellation, backpressure, and application code are shared.
There is no HTTP/WebSocket server or required microphone dependency.

| Registered family | Native tasks | Streaming behavior |
| --- | --- | --- |
| `whisper` (safetensors/GGUF) | Transcription, translation | Bounded, non-overlapping recognition windows |
| `parler_tts` | Descriptively conditioned synthesis | Buffered generation and DAC decoding |
| `mimi` | Codec encode/decode and reconstruction | Causal PCM input/output |
| `encodec`, `dac`, `snac` | Codec encode/decode and reconstruction | Buffered; SNAC preserves ragged codebook scales |
| `lfm3_audio` | Recognition, synthesis, spoken conversation | Native encoder, backbone, MaskGIT and incremental LiquidCodec |
| `csm` | Context-conditioned speech synthesis | Incremental codebook frames and Mimi decoding |
| `metavoice`, `quantized_metavoice` | Speech synthesis | First stage, refinement and EnCodec decoding |
| `voxtral`, `gemma4` | Recognition and audio-conditioned text | Buffered audio, incremental text generation |
| `musicgen` | Text-conditioned audio/music generation | Buffered codec decoding |

Registration identifies implemented adapters. It does not certify every checkpoint,
precision or backend. See `AUDIO-IMPLEMENTATION-STATUS.json` in the workspace root
for actual validation and limitations. CSM pretrained validation requires access to its gated weights. Other large
checkpoint validation is recorded separately; component fixtures alone do not
establish task quality.
The standard loaders default to F32 activation/preprocessing; Voxtral and Gemma4 also accept
`--dtype bf16` to retain its official BF16 checkpoint precision, with supported
quantized checkpoint weights handled inside the corresponding model implementation.

These adapters run Candle tensor operations on the selected device. Adding a new
architecture still requires native model operations and a loader. Applications can
register a custom `ModelLoader` without editing sessions, workers or transports.

## CLI

Build from the workspace with `cargo build -p candle-audio --release --features cli`.
Add `vulkan`, `cuda`, or `metal` to enable the corresponding device. CPU is the
default. Checkpoints are local directories; no model or executable code is
downloaded by this crate. Each directory contains `config.json` and
`model.safetensors` (or its shard index and shards). Text-conditioned models also need their tokenizer and processor metadata. Whisper
can instead contain `model.gguf`. LFM3 requires `config.json`, `tokenizer.json`,
`model.gguf` (language backbone) and `mmproj.gguf` (audio projector/codec); the
loader accepts the official LiquidAI GGUF pair. MetaVoice uses its native
`first_stage.meta.json`, first-stage weights, second-stage weights, EnCodec and
speaker embedding files. Use `--model-options options.json` for adapter-specific loader options (for example
MetaVoice stage paths or context budgets). Inspect each adapter's loader for exact artifact names;
model directories and shard files must remain unchanged while mmap weights are live.

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

```sh
# List task support without loading weights; inspect actual checkpoint capabilities.
cargo run -p candle-audio --release --features cli -- models
cargo run -p candle-audio --release --features cli -- capabilities \
  --model /models/lfm3-audio --task realtime

# Multilingual ASR/TTS; LFM3 voice conversation is officially English-only.
cargo run -p candle-audio --release --features cli -- tts \
  --model /models/lfm3-audio --text "Hola, ¿cómo estás?" \
  --language es --voice female --output hola.wav

# Codec codes retain format, normalization, original length and ragged scales.
cargo run -p candle-audio --release --features cli -- codec encode \
  --model /models/snac-24khz --input speech.wav --output codes.json
cargo run -p candle-audio --release --features cli -- codec decode \
  --model /models/snac-24khz --input codes.json --output restored.wav

cargo run -p candle-audio --release --features cli -- generate-audio \
  --model /models/musicgen-small --prompt "A mellow piano melody" \
  --max-steps 250 --output music.wav
cargo run -p candle-audio --release --features cli -- understand \
  --model /models/voxtral --input speech.wav \
  --prompt "Summarize the spoken request."
```

`tts --conditioning request.json` accepts the serialized
`ConditionedSynthesisRequest` reference/speaker fields. Its embedded `request`
is replaced by the CLI's text, language, voice and generation flags. Unsupported
conditioning returns an explicit error. Reference audio and speaker identity are
model-specific; a reference does not automatically enable voice cloning.

`realtime --tts-model /models/parler-tts-mini-v1` reads back recognized segments.
Readback uses its own `--tts-dtype` (default F32) and `--tts-model-options` JSON;
primary `--dtype bf16` or encoder options are not inherited by the synthesizer.
Throughput depends on the model, backend, and hardware; use release builds for
inference. Streaming capabilities describe when a model can emit output.
Applications that want conversational replies can supply a response function to
`SpeechPipeline`, or implement `RealtimeModel` for a native speech model.
Whisper window boundaries can split words: this adapter does not advertise
causal live transcription or word-level timestamps. Parler's `--description`
controls the voice; unsupported named voices/languages are rejected.

Parler defaults to greedy decoding (`temperature: 0`), where the request seed has
no effect. Enable sampling with `--model-options` JSON such as
`{"temperature": 1.0, "top_k": 50, "top_p": 0.95}`; readback uses
`--tts-model-options`. Top-k must be positive, top-p must lie in `(0, 1]`, and both
filters require a positive temperature. Each synthesis creates a fresh sampler
from its request seed. Resetting the model retains the configured sampling options.

Parler's official greedy reference also fails to reach EOS for some prompts.
For natural speech, start with sampled generation (`temperature: 1`, `top_k: 50`)
and allow an adequate step budget. A `Length` completion is partial speech, even
when a playable WAV was produced.

Input supports PCM integer and float WAV files, 1–64 channels, and
sample rates up to 384 kHz. ASR averages source channels and uses a stateful sinc
resampler. Codec/generation APIs preserve the model's channel layout; applications
can select `ChannelPolicy::Average`, `Preserve`, `Select` or `Map` explicitly. Conversion ratios are limited to `1/256..=256`; each conversion call returns at
most 8 Mi interleaved samples (32 MiB). Process larger audio through smaller
streaming chunks. Resampler scratch blocks are bounded independently of the
whole recording. Realtime WAV input is read in chunks,
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

`AudioEvent::TurnCompleted` includes `finish_reason`, so worker and transport
consumers can distinguish EOS from a step budget. Rust patterns that previously
used `{ turn }` must use `{ turn, .. }` or bind the new field; event constructors
must supply it. Older serialized JSON events without the field still deserialize
with `FinishReason::Completed`.

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
`cancel_turn` rolls back an unfinished response according to the adapter contract.
`commit`/`end_turn` complete a turn and preserve supported conversation state;
`reset_conversation` clears it. `clear_context` clears between committed turns
without advancing the turn ID. File CLI input clears context at automatic rollover
unless `--preserve-conversation` is selected. A closed session rejects further input. Audio callbacks are not suitable for an OS audio
realtime-priority thread; drive inference on a worker thread.

`SessionLimits` bounds text per message and turn, audio chunks and turn frames,
output audio, event count and optional deadline. `RealtimeWorker` separates bounded
command and event queues on an owned inference thread. `poll_output` lets stepwise
models generate while no new input arrives. Buffered adapters remain buffered;
a worker cannot turn a blocking architecture into a full-duplex model. Cancelled
generations invalidate queued events. Output contains audio clocks, transcription
metadata and explicit completion reasons (`EndOfSequence`, `Length`, `Deadline`,
etc.). The CLI reports truncated speech instead of silently claiming EOS.

`AudioCodec`, `AudioGenerationModel` and `AudioUnderstandingModel` expose codes,
music/audio generation and audio-conditioned text without treating them as TTS.
`ModelRegistry::register` accepts application loaders; `ModelSpec` supplies local
checkpoint paths, source format and model-specific options.

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

Pinned model fixtures can be fetched explicitly with the data-only utility:

```sh
python3 .devops/fetch-audio-test-models.py --destination /models \
  --family lfm3-audio
CANDLE_AUDIO_LFM3=/models/lfm3-audio CANDLE_AUDIO_TEST_DEVICE=cpu \
  cargo test -p candle-audio --release --features cli --test lfm3 \
  -- --ignored --nocapture --test-threads=1
```

The manifest records immutable revisions, lengths and SHA256 digests. The manual
`audio-checkpoints.yml` workflow runs selected real-checkpoint tests without
silently skipping missing fixtures. Fast CI runs deterministic native tests with
`cli` enabled. Physical microphone, playback and real-browser validation require
the actual devices; simulated audio/DOM tests do not certify those paths.

The Axum/web integration is an optional local example under
`bench_results/axum-audio-example/`, ignored by Git and excluded from the workspace.
The library and CLI do not depend on that example or on a network service.

## Optional native capture and playback

`live-io` adds CPAL and bounded lock-free device queues. It is separate from `cli`
and disabled by default. On Linux, compilation needs ALSA development headers
(for example `libasound2-dev` and `pkg-config`). Device enumeration queries metadata
without opening a stream:

```sh
cargo run -p candle-audio --release --features cli,live-io -- devices
cargo run -p candle-audio --release --features cli,live-io,vulkan -- \
  --device vulkan live --model /models/lfm3-audio --turn-seconds 20
```

Use `--input-device ID` / `--output-device ID` with IDs from `devices`. Enter starts
capture; Enter stops and commits the turn. `c` cancels the current response, `r`
clears conversation context, and `q` exits. `--tts-model` adds speech readback to
recognizers. The CLI converts hardware formats outside callbacks and reports
queue overruns, underruns and backend errors. It rejects a turn with lost capture
samples. Streaming models can emit while input arrives; LFM3's interactive mode
waits for manual commit. No VAD or automatic acoustic echo cancellation is implied.

The reusable `live::{LiveCapture,LivePlayback}` APIs open paused streams; applications
must call `start` explicitly. Data callbacks only copy whole frames to/from bounded
queues; inference, allocation, waiting and resampling run elsewhere. Cancellation
clears pending application audio, but cannot retract frames already submitted to
hardware. CPAL component tests use queues and PCM fixtures; physical device capture
and playback remain a separate validation step.

MusicGen `AudioGenerationRequest.audio_reference` starts a continuation. Convert
reference PCM explicitly to the generator's advertised mono/stereo32k format.
`max_steps` counts new codec frames; the budget includes the reference plus new
frames and is capped at30seconds. Output includes the codec reconstruction of the
prompt followed by new audio, as in the reference implementation; the initial
segment is not a byte-for-byte copy of the source PCM. The CLI's `generate-audio
--audio-reference input.wav` exposes the same contract.

For audio-understanding realtime adapters, `realtime --prompt "Summarize the
spoken request"` and `live --prompt "Transcribe the audio"` provide their required
text context. The prompt is supplied at the start of each turn. `--tts-model`
selects the primary model's recognition task and reads back transcript segments;
it does not assume that arbitrary text deltas are transcripts. Use `understand`
for questions about several audio clips.

Migration: `AudioEvent::TurnCompleted` now carries `finish_reason`, also through
`RealtimeWorker`. Rust matches that only need the ID can use `{ turn, .. }`.
Legacy JSON events without the field deserialize as `Completed`. `TtsStream`
advertises its own text-only wrapper capabilities; it does not expose a model's
reference-audio or independently polled output interfaces.
