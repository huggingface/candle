use crate::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    Ready,
    Active,
    Faulted,
    Closed,
}

/// Synchronous, transport-independent session with bounded input and fallible
/// output. Call `reset` after a model/sink error. Clone the cancellation handle
/// into another thread to interrupt generation without sharing mutable state.
pub struct RealtimeSession<M: RealtimeModel> {
    model: M,
    state: SessionState,
    turn: u64,
    cancel: CancellationToken,
    input_frames: usize,
    max_chunk_frames: usize,
    max_turn_frames: usize,
}
impl<M: RealtimeModel> RealtimeSession<M> {
    pub fn new(model: M, max_chunk_frames: usize, max_turn_frames: usize) -> Result<Self> {
        if max_chunk_frames == 0 || max_turn_frames == 0 {
            return Err(Error::InvalidInput(
                "session frame limits must be nonzero".into(),
            ));
        }
        let max_turn_frames = max_turn_frames.min(
            model
                .capabilities()
                .max_input_frames_per_turn
                .unwrap_or(usize::MAX),
        );
        if max_turn_frames == 0 {
            return Err(Error::InvalidInput(
                "model advertises an empty input context".into(),
            ));
        }
        Ok(Self {
            model,
            state: SessionState::Ready,
            turn: 0,
            cancel: CancellationToken::default(),
            input_frames: 0,
            max_chunk_frames,
            max_turn_frames,
        })
    }
    pub fn capabilities(&self) -> RealtimeCapabilities {
        self.model.capabilities()
    }
    pub fn state(&self) -> SessionState {
        self.state
    }
    pub fn turn(&self) -> u64 {
        self.turn
    }
    pub fn max_turn_frames(&self) -> usize {
        self.max_turn_frames
    }
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
    fn check_state(&self) -> Result<()> {
        match self.state {
            SessionState::Ready | SessionState::Active => Ok(()),
            state => Err(Error::InvalidState(format!("session is {state:?}"))),
        }
    }
    fn next_turn(&mut self) {
        self.turn += 1;
        self.input_frames = 0;
        self.cancel = CancellationToken::default();
        self.state = SessionState::Ready;
    }
    fn run(
        &mut self,
        sink: &mut dyn EventSink,
        action: impl FnOnce(&mut M, &CancellationToken, &mut dyn EventSink) -> Result<()>,
    ) -> Result<()> {
        self.check_state()?;
        self.state = SessionState::Active;
        let turn = self.turn;
        let format = self.model.capabilities().output_format;
        let token = &self.cancel;
        let result = token
            .check()
            .and_then(|()| {
                action(&mut self.model, token, &mut |event| {
                    token.check()?;
                    let event = match event {
                        AudioEvent::Transcript { transcript, .. } => {
                            if !transcript.start.is_finite()
                                || !transcript.end.is_finite()
                                || transcript.start < 0.
                                || transcript.end < transcript.start
                            {
                                return Err(Error::InvalidInput(
                                    "invalid model transcript timestamps".into(),
                                ));
                            }
                            AudioEvent::Transcript { turn, transcript }
                        }
                        AudioEvent::Text { text, .. } => AudioEvent::Text { turn, text },
                        AudioEvent::Audio { audio, .. } => {
                            if Some(audio.format()) != format {
                                return Err(Error::InvalidInput(
                                    "model output format disagrees with capabilities".into(),
                                ));
                            }
                            AudioEvent::Audio { turn, audio }
                        }
                        _ => {
                            return Err(Error::InvalidState(
                                "models cannot emit lifecycle events".into(),
                            ))
                        }
                    };
                    sink.emit(event)
                })
            })
            .and_then(|()| token.check());
        if matches!(result, Err(Error::Cancelled)) {
            self.state = SessionState::Faulted;
            self.model.reset()?;
            self.next_turn();
            if let Err(error) = sink.emit(AudioEvent::Cancelled { turn }) {
                self.state = SessionState::Faulted;
                return Err(error);
            }
        } else if result.is_err() {
            self.state = SessionState::Faulted;
        }
        result
    }
    pub fn push_audio(&mut self, audio: &AudioBuffer, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if Some(audio.format()) != self.model.capabilities().input_format {
            return Err(Error::InvalidInput(
                "audio format does not match session input".into(),
            ));
        }
        let total = self
            .input_frames
            .checked_add(audio.frames())
            .ok_or_else(|| Error::LimitExceeded("turn duration overflow".into()))?;
        if audio.frames() > self.max_chunk_frames || total > self.max_turn_frames {
            return Err(Error::LimitExceeded(
                "chunk or turn exceeds frame limit".into(),
            ));
        }
        self.run(sink, |m, c, s| m.push_audio(audio, c, s))?;
        self.input_frames = total;
        Ok(())
    }
    pub fn push_text(&mut self, text: &str, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if !self.model.capabilities().text_input {
            return Err(Error::Unsupported(
                "model does not accept text input".into(),
            ));
        }
        self.run(sink, |m, c, s| m.push_text(text, c, s))
    }
    pub fn commit(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if self.state != SessionState::Active {
            return Err(Error::InvalidState("cannot commit an empty turn".into()));
        }
        self.run(sink, |m, c, s| m.commit(c, s))?;
        let turn = self.turn;
        self.state = SessionState::Faulted;
        self.model.reset()?;
        sink.emit(AudioEvent::TurnCompleted { turn })?;
        self.next_turn();
        Ok(())
    }
    pub fn cancel(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        let turn = self.turn;
        self.cancel.cancel();
        self.state = SessionState::Faulted;
        self.model.reset()?;
        self.next_turn();
        let result = sink.emit(AudioEvent::Cancelled { turn });
        if result.is_err() {
            self.state = SessionState::Faulted;
        }
        result
    }
    pub fn reset(&mut self) -> Result<()> {
        if self.state == SessionState::Closed {
            return Err(Error::InvalidState("session is closed".into()));
        }
        self.cancel.cancel();
        self.state = SessionState::Faulted;
        self.model.reset()?;
        self.next_turn();
        Ok(())
    }
    pub fn finish(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if self.state == SessionState::Active {
            self.commit(sink)?;
        }
        self.state = SessionState::Faulted;
        self.model.reset()?;
        self.state = SessionState::Closed;
        Ok(())
    }
    pub fn into_model(self) -> M {
        self.model
    }
}

/// Bounded, non-overlapping recognition windows for offline ASR models. Emits
/// final segments. A causal live recognizer implements `RealtimeModel` directly.
pub struct AsrStream<M: AsrModel> {
    model: M,
    input_format: AudioFormat,
    converter: AudioConverter,
    pending: VecDeque<f32>,
    window_frames: usize,
    processed_frames: u64,
    options: TranscriptionOptions,
}
impl<M: AsrModel> AsrStream<M> {
    pub fn new(
        model: M,
        input_format: AudioFormat,
        window_frames: usize,
        options: TranscriptionOptions,
    ) -> Result<Self> {
        if model.input_format().channels() != 1
            || window_frames == 0
            || window_frames > model.max_input_frames()
        {
            return Err(Error::InvalidInput(
                "ASR window must fit the mono model's input limit".into(),
            ));
        }
        let converter = AudioConverter::new(input_format, model.input_format().sample_rate())?;
        Ok(Self {
            model,
            input_format,
            converter,
            pending: VecDeque::new(),
            window_frames,
            processed_frames: 0,
            options,
        })
    }
    fn decode(
        &mut self,
        count: usize,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        let format = self.model.input_format();
        let pcm = AudioBuffer::new(format, self.pending.drain(..count).collect())?;
        let mut transcript = self.model.transcribe(&pcm, &self.options, cancel)?;
        cancel.check()?;
        let offset = self.processed_frames as f64 / format.sample_rate() as f64;
        transcript.start += offset;
        transcript.end += offset;
        self.processed_frames += count as u64;
        sink.emit(AudioEvent::Transcript {
            turn: 0,
            transcript,
        })
    }
    fn accept(
        &mut self,
        pcm: AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        for part in pcm.samples().chunks(self.window_frames) {
            self.pending.extend(part);
            while self.pending.len() >= self.window_frames {
                self.decode(self.window_frames, cancel, sink)?;
            }
        }
        Ok(())
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn into_model(self) -> M {
        self.model
    }
}
impl<M: AsrModel> RealtimeModel for AsrStream<M> {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: Some(self.input_format),
            output_format: None,
            max_input_frames_per_turn: None,
            text_input: false,
            text_output: true,
            streaming: StreamingMode::Windowed,
            output_streaming: None,
        }
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        let pcm = self.converter.push(audio)?;
        self.accept(pcm, cancel, sink)
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        cancel.check()?;
        let tail = self.converter.finish()?;
        self.accept(tail, cancel, sink)?;
        if !self.pending.is_empty() {
            self.decode(self.pending.len(), cancel, sink)?;
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset()?;
        self.pending.clear();
        self.converter.reset();
        self.processed_frames = 0;
        Ok(())
    }
}

/// Compose arbitrary speech models with an application-supplied response
/// function (a translator, language model, read-back function, etc.).
pub struct SpeechPipeline<A, T, F> {
    asr: A,
    tts: T,
    respond: F,
}
impl<A, T, F> SpeechPipeline<A, T, F> {
    pub fn new(asr: A, tts: T, respond: F) -> Self {
        Self { asr, tts, respond }
    }
}
impl<A: RealtimeModel, T: TtsModel, F: FnMut(&Transcript) -> Result<Option<SynthesisRequest>>>
    RealtimeModel for SpeechPipeline<A, T, F>
{
    fn capabilities(&self) -> RealtimeCapabilities {
        let mut caps = self.asr.capabilities();
        // The pipeline accepts recognition audio and emits response text; it
        // does not forward a recognizer's optional text-input interface.
        caps.text_input = false;
        caps.text_output = true;
        caps.output_format = Some(self.tts.output_format());
        caps.output_streaming = Some(self.tts.streaming_mode());
        caps
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        let (tts, respond) = (&mut self.tts, &mut self.respond);
        self.asr.push_audio(audio, cancel, &mut |e| {
            forward_speech(e, tts, respond, cancel, sink)
        })
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        let (tts, respond) = (&mut self.tts, &mut self.respond);
        self.asr.commit(cancel, &mut |e| {
            forward_speech(e, tts, respond, cancel, sink)
        })
    }
    fn reset(&mut self) -> Result<()> {
        self.asr.reset()?;
        self.tts.reset()
    }
}
fn forward_speech<T: TtsModel, F: FnMut(&Transcript) -> Result<Option<SynthesisRequest>>>(
    event: AudioEvent,
    tts: &mut T,
    respond: &mut F,
    cancel: &CancellationToken,
    sink: &mut dyn EventSink,
) -> Result<()> {
    let request = if let AudioEvent::Transcript { transcript, .. } = &event {
        respond(transcript)?
    } else {
        None
    };
    sink.emit(event)?;
    if let Some(request) = request {
        cancel.check()?;
        sink.emit(AudioEvent::Text {
            turn: 0,
            text: request.text.clone(),
        })?;
        tts.synthesize(&request, cancel, sink)?;
    }
    Ok(())
}

/// Text-driven speech sessions. Each text push is a complete utterance, and may
/// emit audio immediately. `template.text` is replaced by that utterance; other
/// synthesis options are reused. Native streaming behavior is model-dependent.
pub struct TtsStream<M> {
    model: M,
    template: SynthesisRequest,
    max_text_bytes: usize,
}
impl<M: TtsModel> TtsStream<M> {
    pub fn new(model: M, template: SynthesisRequest, max_text_bytes: usize) -> Result<Self> {
        if max_text_bytes == 0 {
            return Err(Error::InvalidInput("text limit must be nonzero".into()));
        }
        Ok(Self {
            model,
            template,
            max_text_bytes,
        })
    }
}
impl<M: TtsModel> RealtimeModel for TtsStream<M> {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: None,
            output_format: Some(self.model.output_format()),
            max_input_frames_per_turn: None,
            text_input: true,
            text_output: true,
            streaming: self.model.streaming_mode(),
            output_streaming: Some(self.model.streaming_mode()),
        }
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        _: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        Err(Error::Unsupported("TTS session accepts text input".into()))
    }
    fn push_text(
        &mut self,
        text: &str,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        if text.trim().is_empty() || text.len() > self.max_text_bytes {
            return Err(Error::LimitExceeded(
                "empty text or utterance exceeds text limit".into(),
            ));
        }
        let mut request = self.template.clone();
        request.text = text.to_owned();
        sink.emit(AudioEvent::Text {
            turn: 0,
            text: text.to_owned(),
        })?;
        self.model.synthesize(&request, cancel, sink)
    }
    fn commit(&mut self, cancel: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        cancel.check()
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset()
    }
}
