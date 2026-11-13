use crate::*;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[path = "session_events.rs"]
mod session_events;
use session_events::EventMetadata;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    Ready,
    Active,
    Faulted,
    Closed,
}

/// All budgets apply across a complete turn, including many small input calls.
#[derive(Clone, Debug)]
pub struct SessionLimits {
    pub max_chunk_frames: usize,
    pub max_turn_frames: usize,
    pub max_input_messages_per_turn: usize,
    pub max_text_bytes_per_message: usize,
    pub max_text_bytes_per_turn: usize,
    pub max_output_frames_per_turn: usize,
    pub max_output_text_bytes_per_turn: usize,
    pub max_events_per_turn: usize,
    pub turn_timeout: Option<Duration>,
}
impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_chunk_frames: 48_000 * 30,
            max_turn_frames: 48_000 * 3600,
            max_input_messages_per_turn: 100_000,
            max_text_bytes_per_message: 65_536,
            max_text_bytes_per_turn: 262_144,
            max_output_frames_per_turn: 48_000 * 600,
            max_output_text_bytes_per_turn: 1_048_576,
            max_events_per_turn: 100_000,
            turn_timeout: Some(Duration::from_secs(300)),
        }
    }
}
impl SessionLimits {
    fn validate(&self) -> Result<()> {
        if [
            self.max_chunk_frames,
            self.max_turn_frames,
            self.max_input_messages_per_turn,
            self.max_text_bytes_per_message,
            self.max_text_bytes_per_turn,
            self.max_output_frames_per_turn,
            self.max_output_text_bytes_per_turn,
            self.max_events_per_turn,
        ]
        .contains(&0)
            || self.turn_timeout.is_some_and(|timeout| timeout.is_zero())
        {
            return Err(Error::InvalidInput(
                "session budgets must be nonzero".into(),
            ));
        }
        Ok(())
    }
}

/// Synchronous bounded session. Use `RealtimeWorker` to own it on an inference
/// thread. Stepwise models can emit independently through `poll_output`.
pub struct RealtimeSession<M: RealtimeModel> {
    model: M,
    state: SessionState,
    turn: u64,
    cancel: CancellationToken,
    input_frames: usize,
    input_messages: usize,
    text_bytes: usize,
    output_frames: usize,
    output_text_bytes: usize,
    output_events: usize,
    limits: SessionLimits,
    started: bool,
    committing: bool,
    last_finish_reason: Option<FinishReason>,
    structured_events: bool,
    event_metadata: EventMetadata,
}
impl<M: RealtimeModel> RealtimeSession<M> {
    pub fn new(model: M, max_chunk_frames: usize, max_turn_frames: usize) -> Result<Self> {
        Self::with_limits(
            model,
            SessionLimits {
                max_chunk_frames,
                max_turn_frames,
                ..Default::default()
            },
        )
    }
    pub fn with_limits(model: M, mut limits: SessionLimits) -> Result<Self> {
        limits.validate()?;
        limits.max_turn_frames = limits.max_turn_frames.min(
            model
                .capabilities()
                .max_input_frames_per_turn
                .unwrap_or(usize::MAX),
        );
        if let Some(maximum) = model.descriptor().max_text_bytes {
            limits.max_text_bytes_per_message = limits.max_text_bytes_per_message.min(maximum);
            limits.max_text_bytes_per_turn = limits.max_text_bytes_per_turn.min(maximum);
        }
        limits.validate()?;
        Ok(Self {
            model,
            state: SessionState::Ready,
            turn: 0,
            cancel: CancellationToken::default(),
            input_frames: 0,
            input_messages: 0,
            text_bytes: 0,
            output_frames: 0,
            output_text_bytes: 0,
            output_events: 0,
            limits,
            started: false,
            committing: false,
            last_finish_reason: None,
            structured_events: false,
            event_metadata: EventMetadata::default(),
        })
    }
    /// Emit identified final transcript segments and clocked audio chunks for
    /// legacy model events. Existing structured events retain their metadata.
    /// Select this before the first input of a turn; legacy output is the default.
    pub fn with_structured_events(mut self) -> Result<Self> {
        if self.state != SessionState::Ready || self.started {
            return Err(Error::InvalidState(
                "event metadata must be selected between turns".into(),
            ));
        }
        self.structured_events = true;
        Ok(self)
    }
    pub fn capabilities(&self) -> RealtimeCapabilities {
        self.model.capabilities()
    }
    pub fn descriptor(&self) -> ModelCapabilities {
        self.model.descriptor()
    }
    pub fn state(&self) -> SessionState {
        self.state
    }
    pub fn turn(&self) -> u64 {
        self.turn
    }
    pub fn max_turn_frames(&self) -> usize {
        self.limits.max_turn_frames
    }
    pub fn limits(&self) -> &SessionLimits {
        &self.limits
    }
    pub fn remaining_input_frames(&self) -> usize {
        self.limits.max_turn_frames - self.input_frames
    }
    pub fn last_finish_reason(&self) -> Option<FinishReason> {
        self.last_finish_reason
    }
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
    pub fn output_pending(&self) -> bool {
        self.model.output_pending()
    }
    fn check_state(&self) -> Result<()> {
        match self.state {
            SessionState::Ready | SessionState::Active => Ok(()),
            state => Err(Error::InvalidState(format!("session is {state:?}"))),
        }
    }
    fn next_turn(&mut self) {
        self.turn = self.turn.saturating_add(1);
        self.input_frames = 0;
        self.input_messages = 0;
        self.text_bytes = 0;
        self.output_frames = 0;
        self.output_text_bytes = 0;
        self.output_events = 0;
        self.event_metadata = EventMetadata::default();
        self.cancel = CancellationToken::default();
        self.started = false;
        self.committing = false;
        self.state = SessionState::Ready;
    }
    fn run(
        &mut self,
        sink: &mut dyn EventSink,
        action: impl FnOnce(&mut M, &CancellationToken, &mut dyn EventSink) -> Result<()>,
    ) -> Result<()> {
        self.check_state()?;
        if !self.started {
            self.started = true;
            self.cancel.set_deadline(
                self.limits
                    .turn_timeout
                    .and_then(|duration| Instant::now().checked_add(duration)),
            );
        }
        self.state = SessionState::Active;
        let turn = self.turn;
        let format = self.model.capabilities().output_format;
        let token = &self.cancel;
        let limits = &self.limits;
        let structured = self.structured_events;
        let metadata_state = &mut self.event_metadata;
        let (frames, text_bytes, events) = (
            &mut self.output_frames,
            &mut self.output_text_bytes,
            &mut self.output_events,
        );
        let result = token
            .check()
            .and_then(|()| {
                action(&mut self.model, token, &mut |event| {
                    token.check()?;
                    if *events >= limits.max_events_per_turn {
                        return Err(Error::LimitExceeded(
                            "model output exceeded turn event budget".into(),
                        ));
                    }
                    let (event, nframes, ntext) = match event {
                        AudioEvent::Transcript { transcript, .. } => {
                            validate_transcript(&transcript)?;
                            let n = transcript.text.len();
                            (AudioEvent::Transcript { turn, transcript }, 0, n)
                        }
                        AudioEvent::TranscriptSegment {
                            transcript,
                            status,
                            metadata,
                            ..
                        } => {
                            validate_transcript(&transcript)?;
                            if metadata
                                .confidence
                                .is_some_and(|v| !v.is_finite() || !(0. ..=1.).contains(&v))
                            {
                                return Err(Error::InvalidInput(
                                    "invalid transcript confidence".into(),
                                ));
                            }
                            let n = transcript
                                .text
                                .len()
                                .saturating_add(metadata.segment_id.len())
                                .saturating_add(metadata.speaker.as_ref().map_or(0, String::len));
                            (
                                AudioEvent::TranscriptSegment {
                                    turn,
                                    transcript,
                                    status,
                                    metadata,
                                },
                                0,
                                n,
                            )
                        }
                        AudioEvent::TranscriptionMetadata {
                            finish_reason,
                            language,
                            ..
                        } => {
                            let n = language.as_ref().map_or(0, String::len);
                            (
                                AudioEvent::TranscriptionMetadata {
                                    turn,
                                    finish_reason,
                                    language,
                                },
                                0,
                                n,
                            )
                        }
                        AudioEvent::Text { text, .. } => {
                            let n = text.len();
                            (AudioEvent::Text { turn, text }, 0, n)
                        }
                        AudioEvent::Audio { audio, .. } => {
                            if Some(audio.format()) != format {
                                return Err(Error::InvalidInput(
                                    "model output format disagrees with capabilities".into(),
                                ));
                            }
                            let n = audio.frames();
                            (AudioEvent::Audio { turn, audio }, n, 0)
                        }
                        AudioEvent::AudioChunk {
                            audio, start_frame, ..
                        } => {
                            if Some(audio.format()) != format {
                                return Err(Error::InvalidInput(
                                    "model output format disagrees with capabilities".into(),
                                ));
                            }
                            let n = audio.frames();
                            start_frame.checked_add(n as u64).ok_or_else(|| {
                                Error::InvalidInput("audio chunk clock overflow".into())
                            })?;
                            (
                                AudioEvent::AudioChunk {
                                    turn,
                                    audio,
                                    start_frame,
                                },
                                n,
                                0,
                            )
                        }
                        _ => {
                            return Err(Error::InvalidState(
                                "models cannot emit lifecycle events".into(),
                            ))
                        }
                    };
                    let new_frames = frames.checked_add(nframes).ok_or_else(|| {
                        Error::LimitExceeded("output frame budget overflow".into())
                    })?;
                    let new_text = text_bytes.checked_add(ntext).ok_or_else(|| {
                        Error::LimitExceeded("output text budget overflow".into())
                    })?;
                    if new_frames > limits.max_output_frames_per_turn
                        || new_text > limits.max_output_text_bytes_per_turn
                        || *events >= limits.max_events_per_turn
                    {
                        return Err(Error::LimitExceeded(
                            "model output exceeded turn budget".into(),
                        ));
                    }
                    let (event, extra_text_bytes) = metadata_state.normalize(event, structured)?;
                    let new_text = new_text.checked_add(extra_text_bytes).ok_or_else(|| {
                        Error::LimitExceeded("output metadata budget overflow".into())
                    })?;
                    if new_text > limits.max_output_text_bytes_per_turn {
                        return Err(Error::LimitExceeded(
                            "model output metadata exceeded turn budget".into(),
                        ));
                    }
                    *frames = new_frames;
                    *text_bytes = new_text;
                    *events += 1;
                    sink.emit(event)
                })
            })
            .and_then(|()| token.check());
        if matches!(result, Err(Error::Cancelled)) {
            self.state = SessionState::Faulted;
            self.model.cancel_turn()?;
            self.last_finish_reason = Some(FinishReason::Cancelled);
            self.next_turn();
            if let Err(error) = sink.emit(AudioEvent::Cancelled { turn }) {
                self.state = SessionState::Faulted;
                return Err(error);
            }
        } else if result.is_err() {
            self.state = SessionState::Faulted;
            if matches!(result, Err(Error::DeadlineExceeded)) {
                self.last_finish_reason = Some(FinishReason::Deadline);
            }
        }
        result
    }
    pub fn push_audio(&mut self, audio: &AudioBuffer, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if self.committing {
            return Err(Error::InvalidState("input turn has been committed".into()));
        }
        if Some(audio.format()) != self.model.capabilities().input_format {
            return Err(Error::InvalidInput(
                "audio format does not match session input".into(),
            ));
        }
        if self.input_messages >= self.limits.max_input_messages_per_turn {
            return Err(Error::LimitExceeded(
                "input message count exceeds turn budget".into(),
            ));
        }
        let total = self
            .input_frames
            .checked_add(audio.frames())
            .ok_or_else(|| Error::LimitExceeded("turn duration overflow".into()))?;
        if audio.frames() > self.limits.max_chunk_frames || total > self.limits.max_turn_frames {
            return Err(Error::LimitExceeded(
                "chunk or turn exceeds frame limit".into(),
            ));
        }
        self.run(sink, |model, cancel, sink| {
            model.push_audio(audio, cancel, sink)
        })?;
        self.input_frames = total;
        self.input_messages += 1;
        Ok(())
    }
    pub fn push_text(&mut self, text: &str, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if self.committing {
            return Err(Error::InvalidState("input turn has been committed".into()));
        }
        if !self.model.capabilities().text_input {
            return Err(Error::Unsupported(
                "model does not accept text input".into(),
            ));
        }
        if self.input_messages >= self.limits.max_input_messages_per_turn {
            return Err(Error::LimitExceeded(
                "input message count exceeds turn budget".into(),
            ));
        }
        let total = self
            .text_bytes
            .checked_add(text.len())
            .ok_or_else(|| Error::LimitExceeded("text budget overflow".into()))?;
        if text.len() > self.limits.max_text_bytes_per_message
            || total > self.limits.max_text_bytes_per_turn
        {
            return Err(Error::LimitExceeded(
                "text exceeds message or turn budget".into(),
            ));
        }
        self.run(sink, |model, cancel, sink| {
            model.push_text(text, cancel, sink)
        })?;
        self.text_bytes = total;
        self.input_messages += 1;
        Ok(())
    }
    fn complete_turn(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        let turn = self.turn;
        self.state = SessionState::Faulted;
        let finish_reason = self.model.finish_reason();
        self.last_finish_reason = Some(finish_reason);
        self.model.end_turn()?;
        sink.emit(AudioEvent::TurnCompleted {
            turn,
            finish_reason,
        })?;
        self.next_turn();
        Ok(())
    }
    pub fn commit(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if self.state != SessionState::Active || self.committing {
            return Err(Error::InvalidState(
                "cannot commit an empty or already committed turn".into(),
            ));
        }
        self.run(sink, |model, cancel, sink| model.commit(cancel, sink))?;
        self.committing = true;
        if !self.model.output_pending() {
            self.complete_turn(sink)?;
        }
        Ok(())
    }
    pub fn poll_output(&mut self, sink: &mut dyn EventSink) -> Result<PollStatus> {
        self.check_state()?;
        if !self.model.output_pending() {
            if self.committing {
                self.complete_turn(sink)?;
                return Ok(PollStatus::Complete);
            }
            return Ok(PollStatus::Idle);
        }
        let mut status = PollStatus::Idle;
        self.run(sink, |model, cancel, sink| {
            status = model.poll_output(cancel, sink)?;
            Ok(())
        })?;
        if self.committing && (status == PollStatus::Complete || !self.model.output_pending()) {
            self.complete_turn(sink)?;
        }
        Ok(status)
    }
    pub fn cancel(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        let turn = self.turn;
        self.cancel.cancel();
        self.state = SessionState::Faulted;
        self.model.cancel_turn()?;
        self.last_finish_reason = Some(FinishReason::Cancelled);
        self.next_turn();
        let result = sink.emit(AudioEvent::Cancelled { turn });
        if result.is_err() {
            self.state = SessionState::Faulted;
        }
        result
    }
    /// Explicitly discard the entire conversation, unlike a successful commit.
    pub fn reset(&mut self) -> Result<()> {
        if self.state == SessionState::Closed {
            return Err(Error::InvalidState("session is closed".into()));
        }
        self.cancel.cancel();
        self.state = SessionState::Faulted;
        self.model.reset_conversation()?;
        self.next_turn();
        Ok(())
    }
    pub fn reset_conversation(&mut self) -> Result<()> {
        self.reset()
    }
    /// Clear committed model context between turns without consuming another
    /// turn ID. Active/faulted sessions must use reset_conversation instead.
    pub fn clear_context(&mut self) -> Result<()> {
        if self.state != SessionState::Ready {
            return Err(Error::InvalidState(
                "context can only be cleared between turns".into(),
            ));
        }
        self.cancel.cancel();
        self.state = SessionState::Faulted;
        self.model.reset_conversation()?;
        self.cancel = CancellationToken::default();
        self.state = SessionState::Ready;
        Ok(())
    }
    pub fn finish(&mut self, sink: &mut dyn EventSink) -> Result<()> {
        self.check_state()?;
        if self.state == SessionState::Active && !self.committing {
            self.commit(sink)?;
        }
        while self.committing {
            if self.poll_output(sink)? == PollStatus::Idle {
                std::thread::sleep(Duration::from_millis(1));
            } else {
                std::thread::yield_now();
            }
        }
        self.state = SessionState::Faulted;
        self.model.reset_conversation()?;
        self.state = SessionState::Closed;
        Ok(())
    }
    pub fn into_model(self) -> M {
        self.model
    }
}
fn validate_transcript(transcript: &Transcript) -> Result<()> {
    if !transcript.start.is_finite()
        || !transcript.end.is_finite()
        || transcript.start < 0.
        || transcript.end < transcript.start
    {
        return Err(Error::InvalidInput(
            "invalid model transcript timestamps".into(),
        ));
    }
    Ok(())
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
    last_finish_reason: FinishReason,
    options: TranscriptionOptions,
}
impl<M: AsrModel> AsrStream<M> {
    pub fn new(
        model: M,
        input_format: AudioFormat,
        window_frames: usize,
        options: TranscriptionOptions,
    ) -> Result<Self> {
        let policy = if model.input_format().channels() == 1 {
            ChannelPolicy::Average
        } else {
            ChannelPolicy::Preserve
        };
        Self::with_channel_policy(model, input_format, window_frames, options, policy)
    }
    pub fn with_channel_policy(
        model: M,
        input_format: AudioFormat,
        window_frames: usize,
        options: TranscriptionOptions,
        policy: ChannelPolicy,
    ) -> Result<Self> {
        if window_frames == 0 || window_frames > model.max_input_frames() {
            return Err(Error::InvalidInput(
                "ASR window must fit the model input limit".into(),
            ));
        }
        let converter = AudioConverter::with_format(input_format, model.input_format(), policy)?;
        Ok(Self {
            model,
            input_format,
            converter,
            pending: VecDeque::new(),
            window_frames,
            processed_frames: 0,
            last_finish_reason: FinishReason::Completed,
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
        let pcm = AudioBuffer::new(
            format,
            self.pending
                .drain(..count * format.channels() as usize)
                .collect(),
        )?;
        let detailed = self
            .model
            .transcribe_detailed(&pcm, &self.options, cancel)?;
        if self.last_finish_reason != FinishReason::Length {
            self.last_finish_reason = detailed.finish_reason;
        }
        let mut transcript = detailed.transcript;
        cancel.check()?;
        let offset = self.processed_frames as f64 / format.sample_rate() as f64;
        transcript.start += offset;
        transcript.end += offset;
        self.processed_frames += count as u64;
        sink.emit(AudioEvent::Transcript {
            turn: 0,
            transcript,
        })?;
        if detailed.finish_reason != FinishReason::Completed || detailed.language.is_some() {
            sink.emit(AudioEvent::TranscriptionMetadata {
                turn: 0,
                finish_reason: detailed.finish_reason,
                language: detailed.language,
            })?;
        }
        Ok(())
    }
    fn accept(
        &mut self,
        pcm: AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        let window_samples = self.window_frames * self.model.input_format().channels() as usize;
        for part in pcm.samples().chunks(window_samples) {
            self.pending.extend(part);
            while self.pending.len() >= window_samples {
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
    fn finish_reason(&self) -> FinishReason {
        self.last_finish_reason
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut descriptor = self.model.descriptor();
        descriptor.tasks = vec![AudioTask::Recognition, AudioTask::Realtime];
        descriptor.input_formats = vec![self.input_format];
        descriptor.max_input_frames = None;
        descriptor.input_streaming = Some(StreamingMode::Windowed);
        descriptor.output_streaming = None;
        descriptor
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
            self.decode(
                self.pending.len() / self.model.input_format().channels() as usize,
                cancel,
                sink,
            )?;
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset()?;
        self.pending.clear();
        self.converter.reset();
        self.processed_frames = 0;
        self.last_finish_reason = FinishReason::Completed;
        Ok(())
    }
}

/// Compose arbitrary speech models with an application-supplied response
/// function (a translator, language model, read-back function, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceAudioPolicy {
    Discard,
    Reject,
}

pub struct SpeechPipeline<A, T, F> {
    asr: A,
    tts: T,
    respond: F,
    tts_finish: Option<FinishReason>,
}
impl<A, T, F> SpeechPipeline<A, T, F> {
    /// Recognition-side audio is discarded; only synthesized responses are output.
    pub fn new(asr: A, tts: T, respond: F) -> Self {
        Self {
            asr,
            tts,
            respond,
            tts_finish: None,
        }
    }
    pub fn with_source_audio_policy(
        asr: A,
        tts: T,
        respond: F,
        policy: SourceAudioPolicy,
    ) -> Result<Self>
    where
        A: RealtimeModel,
        F: FnMut(&Transcript) -> Result<Option<SynthesisRequest>>,
    {
        if policy == SourceAudioPolicy::Reject && asr.capabilities().output_format.is_some() {
            return Err(Error::Unsupported(
                "speech pipeline recognizer produces audio; select Discard explicitly".into(),
            ));
        }
        Ok(Self::new(asr, tts, respond))
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
        let (tts, respond, finish) = (&mut self.tts, &mut self.respond, &mut self.tts_finish);
        self.asr.push_audio(audio, cancel, &mut |e| {
            forward_speech(e, tts, respond, finish, cancel, sink)
        })
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        let (tts, respond, finish) = (&mut self.tts, &mut self.respond, &mut self.tts_finish);
        self.asr.commit(cancel, &mut |e| {
            forward_speech(e, tts, respond, finish, cancel, sink)
        })
    }
    fn end_turn(&mut self) -> Result<()> {
        self.asr.end_turn()?;
        self.tts_finish = None;
        self.tts.reset()
    }
    fn cancel_turn(&mut self) -> Result<()> {
        self.asr.cancel_turn()?;
        self.tts_finish = None;
        self.tts.reset()
    }
    fn reset_conversation(&mut self) -> Result<()> {
        self.asr.reset_conversation()?;
        self.tts_finish = None;
        self.tts.reset()
    }
    fn output_pending(&self) -> bool {
        self.asr.output_pending()
    }
    fn poll_output(
        &mut self,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<PollStatus> {
        let (tts, respond, finish) = (&mut self.tts, &mut self.respond, &mut self.tts_finish);
        self.asr.poll_output(cancel, &mut |event| {
            forward_speech(event, tts, respond, finish, cancel, sink)
        })
    }
    fn finish_reason(&self) -> FinishReason {
        let asr = self.asr.finish_reason();
        if asr == FinishReason::Length || self.tts_finish == Some(FinishReason::Length) {
            FinishReason::Length
        } else {
            self.tts_finish.unwrap_or(asr)
        }
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut descriptor = self.asr.descriptor();
        let tts = self.tts.descriptor();
        descriptor.tasks = vec![
            AudioTask::Recognition,
            AudioTask::Synthesis,
            AudioTask::Realtime,
        ];
        descriptor.output_formats = vec![self.tts.output_format()];
        descriptor.output_streaming = Some(self.tts.streaming_mode());
        descriptor.voices = tts.voices;
        descriptor
            .languages
            .retain(|language| tts.languages.contains(language));
        descriptor
    }
    fn reset(&mut self) -> Result<()> {
        self.asr.reset()?;
        self.tts_finish = None;
        self.tts.reset()
    }
}
fn forward_speech<T: TtsModel, F: FnMut(&Transcript) -> Result<Option<SynthesisRequest>>>(
    event: AudioEvent,
    tts: &mut T,
    respond: &mut F,
    finish: &mut Option<FinishReason>,
    cancel: &CancellationToken,
    sink: &mut dyn EventSink,
) -> Result<()> {
    // Recognition-side audio is not the synthesized response. Never forward it
    // under the synthesizer's output format/capabilities.
    if matches!(
        event,
        AudioEvent::Audio { .. } | AudioEvent::AudioChunk { .. }
    ) {
        return Ok(());
    }
    let request = match &event {
        AudioEvent::Transcript { transcript, .. }
        | AudioEvent::TranscriptSegment {
            transcript,
            status: SegmentStatus::Final,
            ..
        } => respond(transcript)?,
        _ => None,
    };
    sink.emit(event)?;
    if let Some(request) = request {
        cancel.check()?;
        sink.emit(AudioEvent::Text {
            turn: 0,
            text: request.text.clone(),
        })?;
        tts.synthesize(&request, cancel, sink)?;
        if *finish != Some(FinishReason::Length) {
            *finish = Some(tts.finish_reason());
        }
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
    last_finish: FinishReason,
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
            last_finish: FinishReason::Completed,
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
    fn descriptor(&self) -> ModelCapabilities {
        let mut descriptor = self.model.descriptor();
        descriptor.tasks = vec![AudioTask::Synthesis, AudioTask::Realtime];
        // This wrapper accepts plain text requests and performs synthesis during
        // push_text. It has neither conditioned input nor independently pollable
        // output, and its default end_turn resets the underlying synthesizer.
        descriptor.input_formats.clear();
        descriptor.input_streaming = None;
        descriptor.output_streaming = Some(self.model.streaming_mode());
        descriptor.max_input_frames = None;
        descriptor.preserves_conversation = false;
        descriptor.independent_output = false;
        descriptor.accepts_audio_reference = false;
        descriptor.max_text_bytes = Some(
            descriptor
                .max_text_bytes
                .unwrap_or(usize::MAX)
                .min(self.max_text_bytes),
        );
        descriptor
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
        self.model.synthesize(&request, cancel, sink)?;
        if self.last_finish != FinishReason::Length {
            self.last_finish = self.model.finish_reason();
        }
        Ok(())
    }
    fn commit(&mut self, cancel: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        cancel.check()
    }
    fn finish_reason(&self) -> FinishReason {
        self.last_finish
    }
    fn reset(&mut self) -> Result<()> {
        self.last_finish = FinishReason::Completed;
        self.model.reset()
    }
}
