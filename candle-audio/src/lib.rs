//! Model-independent, native ASR, speech synthesis, and realtime audio.
//!
//! Models implement the traits in this crate. The runtime handles validated PCM,
//! streaming sample-rate conversion, bounded input, cancellation, and turn state.
//! No network service, microphone, or speaker is required by the library.
#![doc = include_str!("../README.md")]

pub mod models;
mod pcm;
mod session;

pub use pcm::{AudioBuffer, AudioConverter, AudioFormat};
pub use session::{AsrStream, RealtimeSession, SessionState, SpeechPipeline, TtsStream};

use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid audio input: {0}")]
    InvalidInput(String),
    #[error("invalid session state: {0}")]
    InvalidState(String),
    #[error("unsupported audio operation: {0}")]
    Unsupported(String),
    #[error("audio limit exceeded: {0}")]
    LimitExceeded(String),
    #[error("audio operation cancelled")]
    Cancelled,
    #[error(transparent)]
    Model(#[from] candle::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Clone this handle into another thread to interrupt a running model call.
/// A cancelled token stays cancelled; create a new token for another request.
#[derive(Clone, Default, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TranscriptionOptions {
    /// ISO language code. `None` lets an adapter use its default or detection.
    pub language: Option<String>,
    pub translate: bool,
}

/// Times are relative to the audio passed to the model, in seconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    pub voice: Option<String>,
    pub language: Option<String>,
    /// Model-specific descriptive conditioning, when supported.
    pub description: Option<String>,
    pub seed: u64,
    pub max_steps: usize,
}

impl SynthesisRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: None,
            language: None,
            description: None,
            seed: 0,
            max_steps: 512,
        }
    }
}

/// Buffered output is honest about models that need a full utterance before
/// emitting audio. Native streaming models may emit during generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingMode {
    Buffered,
    Windowed,
    Native,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RealtimeCapabilities {
    pub input_format: Option<AudioFormat>,
    pub output_format: Option<AudioFormat>,
    /// An input-context limit in source frames, including the model's flush
    /// allowance. `None` means the session's configured limit is sufficient.
    pub max_input_frames_per_turn: Option<usize>,
    pub text_input: bool,
    pub text_output: bool,
    pub streaming: StreamingMode,
    /// Audio output can be buffered even when recognition is native streaming.
    pub output_streaming: Option<StreamingMode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AudioEvent {
    /// A complete segment; `turn` is assigned by the session.
    Transcript {
        turn: u64,
        transcript: Transcript,
    },
    Text {
        turn: u64,
        text: String,
    },
    Audio {
        turn: u64,
        audio: AudioBuffer,
    },
    TurnCompleted {
        turn: u64,
    },
    Cancelled {
        turn: u64,
    },
}

/// A fallible sink applies backpressure synchronously. Returning an error stops
/// generation; implementations must not collect an unbounded output queue.
pub trait EventSink {
    fn emit(&mut self, event: AudioEvent) -> Result<()>;
}
impl<F: FnMut(AudioEvent) -> Result<()>> EventSink for F {
    fn emit(&mut self, event: AudioEvent) -> Result<()> {
        self(event)
    }
}

pub trait AsrModel {
    fn input_format(&self) -> AudioFormat;
    fn max_input_frames(&self) -> usize;
    fn transcribe(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<Transcript>;
    fn reset(&mut self) -> Result<()>;
}

/// A synthesizer emits `AudioEvent::Audio` chunks. For native interleaved text
/// and audio generation, implement `RealtimeModel` instead.
pub trait TtsModel {
    fn output_format(&self) -> AudioFormat;
    fn streaming_mode(&self) -> StreamingMode;
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
}

/// Native streaming, duplex, codec, or composed speech models share this
/// interface. Models own their KV/codec state, while a session owns turn IDs.
pub trait RealtimeModel {
    fn capabilities(&self) -> RealtimeCapabilities;
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()>;
    fn push_text(
        &mut self,
        _text: &str,
        _cancel: &CancellationToken,
        _sink: &mut dyn EventSink,
    ) -> Result<()> {
        Err(Error::Unsupported(
            "model does not accept text input".into(),
        ))
    }
    /// Flush all resampler/model tails and emit the remaining output. The
    /// session calls `reset` before accepting another turn. Direct trait callers
    /// must do the same. This must not discard a short final chunk.
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
}

impl<T: AsrModel + ?Sized> AsrModel for Box<T> {
    fn input_format(&self) -> AudioFormat {
        (**self).input_format()
    }
    fn max_input_frames(&self) -> usize {
        (**self).max_input_frames()
    }
    fn transcribe(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<Transcript> {
        (**self).transcribe(audio, options, cancel)
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
}
impl<T: TtsModel + ?Sized> TtsModel for Box<T> {
    fn output_format(&self) -> AudioFormat {
        (**self).output_format()
    }
    fn streaming_mode(&self) -> StreamingMode {
        (**self).streaming_mode()
    }
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        (**self).synthesize(request, cancel, sink)
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
}
impl<T: RealtimeModel + ?Sized> RealtimeModel for Box<T> {
    fn capabilities(&self) -> RealtimeCapabilities {
        (**self).capabilities()
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        (**self).push_audio(audio, cancel, sink)
    }
    fn push_text(
        &mut self,
        text: &str,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        (**self).push_text(text, cancel, sink)
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        (**self).commit(cancel, sink)
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
}
