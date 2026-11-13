//! Audio-conditioned text generation, including questions and summaries.
use crate::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AudioUnderstandingRequest {
    /// Ordered clips. Each may have its own sample rate and channel layout.
    pub audio: Vec<AudioBuffer>,
    pub prompt: String,
    pub max_steps: usize,
    pub seed: u64,
}
impl AudioUnderstandingRequest {
    pub fn new(audio: Vec<AudioBuffer>, prompt: impl Into<String>) -> Self {
        Self {
            audio,
            prompt: prompt.into(),
            max_steps: 512,
            seed: 0,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.audio.is_empty() || self.audio.iter().any(AudioBuffer::is_empty) {
            return Err(Error::InvalidInput(
                "audio understanding requires nonempty audio clips".into(),
            ));
        }
        if self.prompt.trim().is_empty()
            || self.prompt.len() > 16384
            || self.max_steps == 0
            || self.max_steps > 4096
        {
            return Err(Error::InvalidInput(
                "audio understanding requires a prompt up to 16384 bytes and max_steps in 1..=4096"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AudioUnderstandingResult {
    pub text: String,
    pub finish_reason: FinishReason,
}

/// Audio-conditioned text output. Implementations emit Text events and return
/// the complete text and completion reason. No speech synthesis is implied.
pub trait AudioUnderstandingModel {
    fn understand(
        &mut self,
        request: &AudioUnderstandingRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AudioUnderstandingResult>;
    fn descriptor(&self) -> ModelCapabilities;
    fn reset(&mut self) -> Result<()>;
    /// End a successful turn, preserving any committed conversation context.
    fn end_turn(&mut self) -> Result<()> {
        Ok(())
    }
    /// Roll back an unfinished request. Stateless implementations reset.
    fn cancel_turn(&mut self) -> Result<()> {
        self.reset()
    }
}
impl<M: AudioUnderstandingModel + ?Sized> AudioUnderstandingModel for Box<M> {
    fn understand(
        &mut self,
        request: &AudioUnderstandingRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AudioUnderstandingResult> {
        (**self).understand(request, cancel, sink)
    }
    fn descriptor(&self) -> ModelCapabilities {
        (**self).descriptor()
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
    fn end_turn(&mut self) -> Result<()> {
        (**self).end_turn()
    }
    fn cancel_turn(&mut self) -> Result<()> {
        (**self).cancel_turn()
    }
}

/// Buffers a single PCM stream and its text context until commit. It generates
/// text incrementally during commit; input buffering is bounded before mutation.
pub struct UnderstandingStream<M> {
    model: M,
    source: AudioFormat,
    max_frames: usize,
    samples: Vec<f32>,
    text: String,
    max_steps: usize,
    seed: u64,
    finish: FinishReason,
}
impl<M: AudioUnderstandingModel> UnderstandingStream<M> {
    pub fn new(
        model: M,
        source: AudioFormat,
        max_frames: usize,
        max_steps: usize,
        seed: u64,
    ) -> Result<Self> {
        if max_frames == 0 || !(1..=4096).contains(&max_steps) {
            return Err(Error::InvalidInput(
                "invalid audio understanding stream budgets".into(),
            ));
        }
        Ok(Self {
            model,
            source,
            max_frames,
            samples: Vec::new(),
            text: String::new(),
            max_steps,
            seed,
            finish: FinishReason::Completed,
        })
    }
    fn clear_turn(&mut self) {
        self.samples.clear();
        self.text.clear();
        self.finish = FinishReason::Completed;
    }
}
impl<M: AudioUnderstandingModel> RealtimeModel for UnderstandingStream<M> {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: Some(self.source),
            output_format: None,
            max_input_frames_per_turn: Some(self.max_frames),
            text_input: true,
            text_output: true,
            streaming: StreamingMode::Buffered,
            output_streaming: None,
        }
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut descriptor = self.model.descriptor();
        descriptor.tasks = vec![AudioTask::Understanding, AudioTask::Realtime];
        descriptor.input_formats = vec![self.source];
        descriptor.max_input_frames = Some(self.max_frames);
        descriptor.output_formats.clear();
        descriptor
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        if audio.format() != self.source {
            return Err(Error::InvalidInput(
                "audio understanding source format changed".into(),
            ));
        }
        let total = self
            .samples
            .len()
            .checked_add(audio.samples().len())
            .filter(|n| *n / self.source.channels() as usize <= self.max_frames)
            .ok_or_else(|| {
                Error::LimitExceeded("audio understanding turn exceeds frame budget".into())
            })?;
        self.samples.reserve(total - self.samples.len());
        self.samples.extend_from_slice(audio.samples());
        Ok(())
    }
    fn push_text(
        &mut self,
        text: &str,
        cancel: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        if text.trim().is_empty()
            || self
                .text
                .len()
                .checked_add(text.len())
                .is_none_or(|n| n > 16384)
        {
            return Err(Error::LimitExceeded(
                "audio understanding prompt is empty or exceeds 16384 bytes".into(),
            ));
        }
        self.text.push_str(text);
        Ok(())
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        let request = AudioUnderstandingRequest {
            audio: vec![AudioBuffer::new(self.source, self.samples.clone())?],
            prompt: self.text.clone(),
            max_steps: self.max_steps,
            seed: self.seed,
        };
        request.validate()?;
        self.finish = self.model.understand(&request, cancel, sink)?.finish_reason;
        Ok(())
    }
    fn finish_reason(&self) -> FinishReason {
        self.finish
    }
    fn end_turn(&mut self) -> Result<()> {
        self.model.end_turn()?;
        self.clear_turn();
        Ok(())
    }
    fn cancel_turn(&mut self) -> Result<()> {
        self.model.cancel_turn()?;
        self.clear_turn();
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset()?;
        self.clear_turn();
        Ok(())
    }
}
