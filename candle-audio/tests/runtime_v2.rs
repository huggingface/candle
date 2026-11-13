use candle_audio::*;
use std::time::{Duration, Instant};

fn format() -> AudioFormat {
    AudioFormat::mono(16000).unwrap()
}
fn pcm(n: usize) -> AudioBuffer {
    AudioBuffer::new(format(), vec![0.25; n]).unwrap()
}
fn caps() -> RealtimeCapabilities {
    RealtimeCapabilities {
        input_format: Some(format()),
        output_format: Some(format()),
        max_input_frames_per_turn: None,
        text_input: true,
        text_output: true,
        streaming: StreamingMode::Native,
        output_streaming: Some(StreamingMode::Native),
    }
}
#[derive(Default)]
struct Chat {
    history: Vec<String>,
    draft: Vec<String>,
}
impl RealtimeModel for Chat {
    fn capabilities(&self) -> RealtimeCapabilities {
        caps()
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        c: &CancellationToken,
        s: &mut dyn EventSink,
    ) -> Result<()> {
        c.check()?;
        s.emit(AudioEvent::Audio {
            turn: 0,
            audio: audio.clone(),
        })
    }
    fn push_text(
        &mut self,
        text: &str,
        c: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        c.check()?;
        self.draft.push(text.into());
        Ok(())
    }
    fn commit(&mut self, c: &CancellationToken, s: &mut dyn EventSink) -> Result<()> {
        c.check()?;
        self.history.append(&mut self.draft);
        s.emit(AudioEvent::Text {
            turn: 0,
            text: self.history.join("|"),
        })
    }
    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.draft.clear();
        Ok(())
    }
    fn end_turn(&mut self) -> Result<()> {
        self.draft.clear();
        Ok(())
    }
    fn cancel_turn(&mut self) -> Result<()> {
        self.draft.clear();
        Ok(())
    }
}
#[test]
fn committed_turns_keep_conversation_explicit_reset_and_cancel_have_distinct_semantics() {
    let mut s = RealtimeSession::new(Box::new(Chat::default()), 100, 1000).unwrap();
    let mut texts = Vec::new();
    let mut sink = |e| {
        if let AudioEvent::Text { text, .. } = e {
            texts.push(text);
        }
        Ok(())
    };
    for text in ["hello", "world"] {
        s.push_text(text, &mut sink).unwrap();
        s.commit(&mut sink).unwrap();
    }
    s.push_text("discarded", &mut sink).unwrap();
    s.cancel(&mut sink).unwrap();
    s.push_text("again", &mut sink).unwrap();
    s.commit(&mut sink).unwrap();
    s.reset_conversation().unwrap();
    s.push_text("fresh", &mut sink).unwrap();
    s.commit(&mut sink).unwrap();
    assert_eq!(
        texts,
        ["hello", "hello|world", "hello|world|again", "fresh"]
    );
}
#[test]
fn many_small_text_inputs_are_limited_before_mutating_model() {
    let mut s = RealtimeSession::with_limits(
        Chat::default(),
        SessionLimits {
            max_text_bytes_per_message: 4,
            max_text_bytes_per_turn: 6,
            ..Default::default()
        },
    )
    .unwrap();
    for text in ["ab", "cd", "ef"] {
        s.push_text(text, &mut |_| Ok(())).unwrap();
    }
    assert!(matches!(
        s.push_text("gh", &mut |_| Ok(())),
        Err(Error::LimitExceeded(_))
    ));
    assert_eq!(s.into_model().draft, ["ab", "cd", "ef"]);
}
#[test]
fn output_frame_and_event_budgets_fault_without_delivering_oversized_event() {
    let mut s = RealtimeSession::with_limits(
        Chat::default(),
        SessionLimits {
            max_output_frames_per_turn: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let mut emitted = 0;
    assert!(matches!(
        s.push_audio(&pcm(6), &mut |_| {
            emitted += 1;
            Ok(())
        }),
        Err(Error::LimitExceeded(_))
    ));
    assert_eq!(emitted, 0);
    assert_eq!(s.state(), SessionState::Faulted);
    s.reset().unwrap();
    s.push_audio(&pcm(5), &mut |_| Ok(())).unwrap();
    assert!(s.push_audio(&pcm(1), &mut |_| Ok(())).is_err());
}
struct Slow;
impl RealtimeModel for Slow {
    fn capabilities(&self) -> RealtimeCapabilities {
        caps()
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        c: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        for _ in 0..1000 {
            c.check()?;
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
    fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
#[test]
fn deadline_is_checked_inside_cooperative_model_steps() {
    let mut s = RealtimeSession::with_limits(
        Slow,
        SessionLimits {
            turn_timeout: Some(Duration::from_millis(10)),
            ..Default::default()
        },
    )
    .unwrap();
    let start = Instant::now();
    assert!(matches!(
        s.push_audio(&pcm(1), &mut |_| Ok(())),
        Err(Error::DeadlineExceeded)
    ));
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(s.last_finish_reason(), Some(FinishReason::Deadline));
}
#[test]
fn channel_preservation_selection_and_irregular_stereo_resampling() {
    let input = AudioBuffer::new(
        AudioFormat::new(48000, 2).unwrap(),
        (0..2054).map(|i| (i as f32 * 0.031).sin()).collect(),
    )
    .unwrap();
    let output = AudioFormat::new(24000, 2).unwrap();
    let expected = AudioConverter::convert_to(&input, output, ChannelPolicy::Preserve).unwrap();
    let mut converter =
        AudioConverter::with_format(input.format(), output, ChannelPolicy::Preserve).unwrap();
    let mut got = Vec::new();
    for samples in input.samples().chunks(274) {
        got.extend(
            converter
                .push(&AudioBuffer::new(input.format(), samples.to_vec()).unwrap())
                .unwrap()
                .into_samples(),
        );
    }
    got.extend(converter.finish().unwrap().into_samples());
    assert_eq!(got, expected.samples());
    assert_eq!(expected.frames(), 514);
    let pair = AudioBuffer::new(
        AudioFormat::new(16000, 2).unwrap(),
        vec![0.1, 0.9, 0.2, 0.8],
    )
    .unwrap();
    let swapped =
        AudioConverter::convert_to(&pair, pair.format(), ChannelPolicy::Map(vec![1, 0])).unwrap();
    assert_eq!(swapped.samples(), [0.9, 0.1, 0.8, 0.2]);
    let selected = AudioConverter::convert_to(&pair, format(), ChannelPolicy::Select(1)).unwrap();
    assert_eq!(selected.samples(), [0.9, 0.8]);
    assert!(AudioConverter::with_format(pair.format(), format(), ChannelPolicy::Preserve).is_err());
}
#[test]
fn codec_codes_validate_layout_padding_metadata_and_ragged_rates() {
    let f = CodecFormat {
        audio_format: format(),
        frame_rate: 100.,
        num_codebooks: 2,
        cardinality: 16,
    };
    assert!(AudioCodes::new(f.clone(), vec![0, 1, 2]).is_err());
    assert!(AudioCodes::new(f.clone(), vec![0, 16]).is_err());
    let codes = AudioCodes::new(f.clone(), vec![1, 2, 3, 4])
        .unwrap()
        .with_original_audio_frames(319)
        .unwrap();
    assert_eq!(codes.frames(), 2);
    assert_eq!(codes.original_audio_frames(), Some(319));
    assert!(codes.clone().with_original_audio_frames(321).is_err());
    let round: AudioCodes = serde_json::from_str(&serde_json::to_string(&codes).unwrap()).unwrap();
    assert_eq!(round, codes);
    let ragged = MultiScaleAudioCodes::new(f, vec![1, 2], vec![vec![1, 2, 3, 4], vec![5, 6]])
        .unwrap()
        .with_original_audio_frames(630)
        .unwrap();
    assert_eq!(ragged.codebooks()[1], [5, 6]);
    assert!(ragged.clone().into_uniform().is_err());
    let round: MultiScaleAudioCodes =
        serde_json::from_str(&serde_json::to_string(&ragged).unwrap()).unwrap();
    assert_eq!(round, ragged);
}
struct Tone;
impl TtsModel for Tone {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::mono(24000).unwrap()
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Native
    }
    fn synthesize(
        &mut self,
        _: &SynthesisRequest,
        _: &CancellationToken,
        s: &mut dyn EventSink,
    ) -> Result<()> {
        s.emit(AudioEvent::Audio {
            turn: 0,
            audio: AudioBuffer::new(self.output_format(), vec![0.; 24]).unwrap(),
        })
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
#[test]
fn composition_discards_recognition_audio_and_can_reject_it_up_front() {
    assert!(SpeechPipeline::with_source_audio_policy(
        Chat::default(),
        Tone,
        |_: &Transcript| Ok(None::<SynthesisRequest>),
        SourceAudioPolicy::Reject
    )
    .is_err());
    let pipeline = SpeechPipeline::new(Chat::default(), Tone, |_: &Transcript| {
        Ok(None::<SynthesisRequest>)
    });
    let mut s = RealtimeSession::new(pipeline, 100, 100).unwrap();
    let mut events = Vec::new();
    s.push_audio(&pcm(10), &mut |e| {
        events.push(e);
        Ok(())
    })
    .unwrap();
    assert!(events.is_empty());
}
struct Stepwise {
    pending: usize,
    committed: bool,
}
impl RealtimeModel for Stepwise {
    fn capabilities(&self) -> RealtimeCapabilities {
        caps()
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        c: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        c.check()?;
        self.pending += 2;
        Ok(())
    }
    fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        self.committed = true;
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.pending = 0;
        Ok(())
    }
    fn output_pending(&self) -> bool {
        self.pending > 0
    }
    fn poll_output(&mut self, c: &CancellationToken, s: &mut dyn EventSink) -> Result<PollStatus> {
        c.check()?;
        if self.pending > 0 {
            self.pending -= 1;
            s.emit(AudioEvent::Audio {
                turn: 0,
                audio: pcm(1),
            })?;
        }
        Ok(if self.pending == 0 {
            PollStatus::Complete
        } else {
            PollStatus::Pending
        })
    }
}
#[test]
fn worker_polls_output_between_continuous_input_and_completes_after_drain() {
    let mut worker = RealtimeWorker::spawn(
        Stepwise {
            pending: 0,
            committed: false,
        },
        SessionLimits::default(),
        WorkerOptions {
            command_capacity: 8,
            event_capacity: 16,
            ..Default::default()
        },
    )
    .unwrap();
    worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
    worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
    worker.try_send(WorkerCommand::Commit).unwrap();
    let mut frames = 0;
    let mut completed = false;
    for _ in 0..20 {
        match worker.recv_timeout(Duration::from_secs(1)).unwrap() {
            WorkerEvent::Event(AudioEvent::Audio { audio, .. }) => frames += audio.frames(),
            WorkerEvent::Event(AudioEvent::TurnCompleted { .. }) => {
                completed = true;
                break;
            }
            WorkerEvent::Error { error, .. } => panic!("{error}"),
            _ => {}
        }
    }
    assert!(completed);
    assert_eq!(frames, 4);
    worker.close().unwrap();
}
struct Endless;
impl RealtimeModel for Endless {
    fn capabilities(&self) -> RealtimeCapabilities {
        caps()
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        c: &CancellationToken,
        s: &mut dyn EventSink,
    ) -> Result<()> {
        loop {
            c.check()?;
            s.emit(AudioEvent::Audio {
                turn: 0,
                audio: pcm(1),
            })?;
        }
    }
    fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
#[test]
fn worker_drop_unblocks_full_output_queue_and_cooperatively_cancels_generation() {
    let worker = RealtimeWorker::spawn(
        Endless,
        SessionLimits::default(),
        WorkerOptions {
            event_capacity: 1,
            ..Default::default()
        },
    )
    .unwrap();
    worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let start = Instant::now();
    drop(worker);
    assert!(start.elapsed() < Duration::from_secs(1));
}
#[test]
fn cancelled_worker_generation_does_not_replay_buffered_audio() {
    let worker = RealtimeWorker::spawn(
        Endless,
        SessionLimits::default(),
        WorkerOptions {
            event_capacity: 2,
            ..Default::default()
        },
    )
    .unwrap();
    worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    worker.cancel_current();
    for _ in 0..5 {
        match worker.recv_timeout(Duration::from_secs(1)).unwrap() {
            WorkerEvent::Event(AudioEvent::Audio { .. }) => {
                panic!("stale audio escaped cancellation")
            }
            WorkerEvent::Error {
                error: Error::Cancelled,
                ..
            }
            | WorkerEvent::Event(AudioEvent::Cancelled { .. }) => return,
            _ => {}
        }
    }
    panic!("no cancellation notification");
}

#[test]
fn idle_worker_cancel_accepts_fresh_input_without_losing_or_duplicating_turns() {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    };
    struct IdleGate {
        armed: AtomicBool,
        entered: mpsc::SyncSender<()>,
        released: Mutex<mpsc::Receiver<()>>,
        cancels: Arc<AtomicUsize>,
        resets: Arc<AtomicUsize>,
    }
    impl RealtimeModel for IdleGate {
        fn capabilities(&self) -> RealtimeCapabilities {
            caps()
        }
        fn output_pending(&self) -> bool {
            if self.armed.swap(false, Ordering::AcqRel) {
                let _ = self.entered.send(());
                // Bound the hook so a failing assertion cannot leave Drop joining
                // an inference thread blocked forever on this test gate.
                let _ = self
                    .released
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(2));
            }
            false
        }
        fn push_audio(
            &mut self,
            _: &AudioBuffer,
            cancel: &CancellationToken,
            sink: &mut dyn EventSink,
        ) -> Result<()> {
            cancel.check()?;
            sink.emit(AudioEvent::Text {
                turn: 0,
                text: "audio".into(),
            })
        }
        fn push_text(
            &mut self,
            text: &str,
            cancel: &CancellationToken,
            sink: &mut dyn EventSink,
        ) -> Result<()> {
            cancel.check()?;
            sink.emit(AudioEvent::Text {
                turn: 0,
                text: text.into(),
            })
        }
        fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
            Ok(())
        }
        fn cancel_turn(&mut self) -> Result<()> {
            self.cancels.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            self.resets.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
    // Every case cancels between the worker's wait decision and recv_timeout.
    // The original implementation snapshots generation before that wait and
    // rejects these fresh-generation commands. An incomplete fix also loses the
    // direct inputs because their session token is still the cancelled one.
    for case in ["text", "audio", "cancel_then_text", "reset_then_text"] {
        let (entered, entered_rx) = mpsc::sync_channel(1);
        let (release, release_rx) = mpsc::sync_channel(1);
        let cancels = Arc::new(AtomicUsize::new(0));
        let resets = Arc::new(AtomicUsize::new(0));
        let mut worker = RealtimeWorker::spawn(
            IdleGate {
                armed: AtomicBool::new(true),
                entered,
                released: Mutex::new(release_rx),
                cancels: cancels.clone(),
                resets: resets.clone(),
            },
            SessionLimits::default(),
            WorkerOptions::default(),
        )
        .unwrap();
        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("worker did not reach idle gate");
        worker.cancel_current();
        let mut pending = Vec::new();
        if case == "cancel_then_text" {
            pending.push(worker.try_send(WorkerCommand::Cancel).unwrap());
        }
        if case == "reset_then_text" {
            pending.push(worker.try_send(WorkerCommand::ResetConversation).unwrap());
        }
        pending.push(
            worker
                .try_send(if case == "audio" {
                    WorkerCommand::Audio(pcm(1))
                } else {
                    WorkerCommand::Text("fresh".into())
                })
                .unwrap(),
        );
        release.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        let mut text = None;
        let mut cancelled = Vec::new();
        while !pending.is_empty() && Instant::now() < until {
            match worker.recv_timeout(Duration::from_millis(50)) {
                Ok(WorkerEvent::CommandCompleted { id }) => {
                    pending.retain(|pending| *pending != id)
                }
                Ok(WorkerEvent::Event(AudioEvent::Text {
                    turn,
                    text: message,
                })) => text = Some((turn, message)),
                Ok(WorkerEvent::Event(AudioEvent::Cancelled { turn })) => cancelled.push(turn),
                Ok(WorkerEvent::Error { id, error }) => {
                    panic!("{case}: fresh command {id} failed: {error}")
                }
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(error) => panic!("{case}: worker disconnected: {error}"),
            }
        }
        // Closing intentionally cancels the fresh active turn as cleanup; count
        // only rollback/reset work performed while accepting the new command.
        let performed_cancels = cancels.load(Ordering::Relaxed);
        let performed_resets = resets.load(Ordering::Relaxed);
        worker.close().unwrap();
        assert!(pending.is_empty(), "{case}: commands did not finish");
        assert_eq!(
            text,
            Some((1, if case == "audio" { "audio" } else { "fresh" }.into())),
            "{case}"
        );
        let reset = case == "reset_then_text";
        assert_eq!(
            performed_cancels,
            usize::from(!reset),
            "{case}: duplicate rollback"
        );
        assert_eq!(
            performed_resets,
            usize::from(reset),
            "{case}: duplicate reset"
        );
        assert_eq!(
            cancelled,
            if reset { vec![] } else { vec![0] },
            "{case}: duplicate cancelled turn"
        );
    }
}

#[test]
fn tts_stream_describes_only_its_text_input_and_synchronous_lifecycle() {
    struct StatefulTts;
    impl TtsModel for StatefulTts {
        fn output_format(&self) -> AudioFormat {
            format()
        }
        fn streaming_mode(&self) -> StreamingMode {
            StreamingMode::Native
        }
        fn synthesize(
            &mut self,
            _: &SynthesisRequest,
            _: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> Result<()> {
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            Ok(())
        }
        fn descriptor(&self) -> ModelCapabilities {
            ModelCapabilities {
                input_formats: vec![format()],
                output_formats: vec![format()],
                max_input_frames: Some(123),
                max_text_bytes: Some(100),
                preserves_conversation: true,
                independent_output: true,
                accepts_audio_reference: true,
                ..Default::default()
            }
        }
    }
    let stream = TtsStream::new(StatefulTts, SynthesisRequest::new(""), 50).unwrap();
    let descriptor = stream.descriptor();
    assert!(descriptor.input_formats.is_empty());
    assert_eq!(descriptor.max_input_frames, None);
    assert!(!descriptor.preserves_conversation);
    assert!(!descriptor.independent_output);
    assert!(!descriptor.accepts_audio_reference);
    assert_eq!(descriptor.max_text_bytes, Some(50));
    assert_eq!(descriptor.output_formats, [format()]);
}

#[test]
fn worker_completion_reports_eos_and_length_before_model_turn_reset() {
    struct Completion {
        finish: FinishReason,
        current: FinishReason,
    }
    impl RealtimeModel for Completion {
        fn capabilities(&self) -> RealtimeCapabilities {
            caps()
        }
        fn push_audio(
            &mut self,
            _: &AudioBuffer,
            cancel: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> Result<()> {
            cancel.check()
        }
        fn commit(&mut self, cancel: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
            cancel.check()?;
            self.current = self.finish;
            Ok(())
        }
        fn finish_reason(&self) -> FinishReason {
            self.current
        }
        fn reset(&mut self) -> Result<()> {
            self.current = FinishReason::Completed;
            Ok(())
        }
    }
    for finish in [FinishReason::EndOfSequence, FinishReason::Length] {
        let mut worker = RealtimeWorker::spawn(
            Completion {
                finish,
                current: FinishReason::Completed,
            },
            SessionLimits::default(),
            WorkerOptions::default(),
        )
        .unwrap();
        worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
        worker.try_send(WorkerCommand::Commit).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        let mut completed = None;
        while Instant::now() < until && completed.is_none() {
            match worker.recv_timeout(Duration::from_millis(50)) {
                Ok(WorkerEvent::Event(AudioEvent::TurnCompleted {
                    turn,
                    finish_reason,
                })) => completed = Some((turn, finish_reason)),
                Ok(WorkerEvent::Error { error, .. }) => panic!("worker failed: {error}"),
                Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(error) => panic!("worker disconnected: {error}"),
            }
        }
        worker.close().unwrap();
        assert_eq!(completed, Some((0, finish)));
    }
}

#[test]
fn old_completion_json_defaults_completed_and_new_json_preserves_reason() {
    let old: AudioEvent =
        serde_json::from_value(serde_json::json!({"type":"turn_completed","turn":3})).unwrap();
    assert!(matches!(
        old,
        AudioEvent::TurnCompleted {
            turn: 3,
            finish_reason: FinishReason::Completed
        }
    ));
    let new = AudioEvent::TurnCompleted {
        turn: 4,
        finish_reason: FinishReason::Length,
    };
    let json = serde_json::to_value(&new).unwrap();
    assert_eq!(json["finish_reason"], "length");
    assert!(matches!(
        serde_json::from_value::<AudioEvent>(json).unwrap(),
        AudioEvent::TurnCompleted {
            turn: 4,
            finish_reason: FinishReason::Length
        }
    ));
}

struct WorkerRollbackGate {
    block_audio: bool,
    fail_rollback: bool,
    gate_next_rollback: bool,
    started: std::sync::mpsc::SyncSender<()>,
    rollback_started: std::sync::mpsc::SyncSender<()>,
    release: std::sync::mpsc::Receiver<()>,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl RealtimeModel for WorkerRollbackGate {
    fn capabilities(&self) -> RealtimeCapabilities {
        caps()
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        cancel: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        if self.block_audio {
            let _ = self.started.try_send(());
            loop {
                cancel.check()?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        cancel.check()
    }
    fn push_text(
        &mut self,
        text: &str,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        sink.emit(AudioEvent::Text {
            turn: 0,
            text: text.into(),
        })
    }
    fn commit(&mut self, cancel: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        cancel.check()
    }
    fn cancel_turn(&mut self) -> Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if std::mem::replace(&mut self.gate_next_rollback, false) {
            let _ = self.rollback_started.try_send(());
            let _ = self.release.recv_timeout(Duration::from_secs(2));
        }
        if self.fail_rollback {
            return Err(Error::InvalidState("rollback failed fixture".into()));
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
fn collect_worker_command(worker: &RealtimeWorker, id: u64, events: &mut Vec<AudioEvent>) {
    let until = Instant::now() + Duration::from_secs(2);
    while Instant::now() < until {
        match worker.recv_timeout(Duration::from_millis(50)) {
            Ok(WorkerEvent::CommandCompleted { id: done }) if done == id => return,
            Ok(WorkerEvent::Event(event)) => events.push(event),
            Ok(WorkerEvent::Error { error, .. }) => panic!("worker command failed: {error}"),
            Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(error) => panic!("worker disconnected: {error}"),
        }
    }
    panic!("worker command timed out");
}

#[test]
fn explicit_cancel_acknowledges_completed_interrupt_without_double_rollback() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };
    for active in [false, true] {
        let (started, started_rx) = mpsc::sync_channel(1);
        let (rollback_started, rollback_rx) = mpsc::sync_channel(1);
        let (release, release_rx) = mpsc::sync_channel(1);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut worker = RealtimeWorker::spawn(
            WorkerRollbackGate {
                block_audio: active,
                fail_rollback: false,
                gate_next_rollback: true,
                started,
                rollback_started,
                release: release_rx,
                calls: calls.clone(),
            },
            SessionLimits::default(),
            WorkerOptions::default(),
        )
        .unwrap();
        if active {
            worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
            started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        worker.cancel_current();
        rollback_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("automatic rollback did not start");
        let id = worker.try_send(WorkerCommand::Cancel).unwrap();
        release.send(()).unwrap();
        let mut events = Vec::new();
        collect_worker_command(&worker, id, &mut events);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "active={active}: explicit acknowledgement rolled back twice"
        );
        let id = worker
            .try_send(WorkerCommand::Text("fresh".into()))
            .unwrap();
        collect_worker_command(&worker, id, &mut events);
        assert!(events
            .iter()
            .any(|e| matches!(e,AudioEvent::Text {turn:1,text} if text=="fresh")));
        // A real new input/commit ends idempotence even when that turn is now Ready.
        let id = worker.try_send(WorkerCommand::Commit).unwrap();
        collect_worker_command(&worker, id, &mut events);
        let id = worker.try_send(WorkerCommand::Cancel).unwrap();
        collect_worker_command(&worker, id, &mut events);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        // Another interrupt epoch also ends idempotence while idle.
        worker.cancel_current();
        let until = Instant::now() + Duration::from_secs(2);
        let mut epoch_cancelled = false;
        while Instant::now() < until && !epoch_cancelled {
            match worker.recv_timeout(Duration::from_millis(50)) {
                Ok(WorkerEvent::Event(event)) => {
                    epoch_cancelled = matches!(event, AudioEvent::Cancelled { turn: 3 });
                    events.push(event);
                }
                Ok(WorkerEvent::Error { error, .. }) => {
                    panic!("unexpected interrupt error: {error}")
                }
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(error) => panic!("worker disconnected: {error}"),
            }
        }
        assert!(epoch_cancelled);
        let id = worker.try_send(WorkerCommand::Cancel).unwrap();
        collect_worker_command(&worker, id, &mut events);
        let count = calls.load(Ordering::Relaxed);
        worker.close().unwrap();
        assert_eq!(count, 3);
        let cancelled = events
            .iter()
            .filter_map(|event| {
                if let AudioEvent::Cancelled { turn } = event {
                    Some(*turn)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(cancelled, [0, 2, 3]);
    }
}

#[test]
fn rollback_failure_remains_visible_after_cancelling_an_old_worker_generation() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };
    let (started, started_rx) = mpsc::sync_channel(1);
    let (rollback_started, rollback_rx) = mpsc::sync_channel(1);
    let (release, release_rx) = mpsc::sync_channel(1);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut worker = RealtimeWorker::spawn(
        WorkerRollbackGate {
            block_audio: true,
            fail_rollback: true,
            gate_next_rollback: true,
            started,
            rollback_started,
            release: release_rx,
            calls: calls.clone(),
        },
        SessionLimits::default(),
        WorkerOptions::default(),
    )
    .unwrap();
    worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    worker.cancel_current();
    rollback_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    release.send(()).unwrap();
    let until = Instant::now() + Duration::from_secs(2);
    let mut failure = None;
    while Instant::now() < until && failure.is_none() {
        match worker.recv_timeout(Duration::from_millis(50)) {
            Ok(WorkerEvent::Error { error, .. }) => failure = Some(error.to_string()),
            Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(error) => panic!("worker disconnected: {error}"),
        }
    }
    assert!(failure.is_some_and(|error| error.contains("rollback failed fixture")));
    let id = worker.try_send(WorkerCommand::Cancel).unwrap();
    loop {
        match worker.recv_timeout(Duration::from_secs(2)).unwrap() {
            WorkerEvent::Error {
                id: failed,
                error: Error::InvalidState(_),
            } if failed == id => break,
            WorkerEvent::CommandCompleted { id: done } if done == id => {
                panic!("failed rollback was acknowledged as success")
            }
            _ => {}
        }
    }
    worker.close().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

struct StereoAsr {
    received: Vec<AudioBuffer>,
}
impl AsrModel for StereoAsr {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::new(16000, 2).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        8
    }
    fn transcribe(
        &mut self,
        audio: &AudioBuffer,
        _: &TranscriptionOptions,
        _: &CancellationToken,
    ) -> Result<Transcript> {
        self.received.push(audio.clone());
        Ok(Transcript {
            text: "segment".into(),
            start: 0.,
            end: audio.duration(),
        })
    }
    fn transcribe_detailed(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<TranscriptionResult> {
        Ok(TranscriptionResult {
            transcript: self.transcribe(audio, options, cancel)?,
            finish_reason: FinishReason::Length,
            language: Some("es".into()),
        })
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
#[test]
fn stereo_asr_windows_count_frames_and_preserve_detailed_completion_metadata() {
    let input = AudioBuffer::new(
        AudioFormat::new(16000, 2).unwrap(),
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    )
    .unwrap();
    let stream = AsrStream::new(
        StereoAsr {
            received: Vec::new(),
        },
        input.format(),
        3,
        TranscriptionOptions::default(),
    )
    .unwrap();
    let mut s = RealtimeSession::new(stream, 10, 10).unwrap();
    let mut metadata = 0;
    let mut sink = |event| {
        if let AudioEvent::TranscriptionMetadata {
            finish_reason: FinishReason::Length,
            language: Some(language),
            ..
        } = event
        {
            assert_eq!(language, "es");
            metadata += 1;
        }
        Ok(())
    };
    s.push_audio(&input, &mut sink).unwrap();
    s.commit(&mut sink).unwrap();
    assert_eq!(s.last_finish_reason(), Some(FinishReason::Length));
    assert_eq!(metadata, 2);
    let asr = s.into_model().into_model();
    assert_eq!(
        asr.received
            .iter()
            .map(AudioBuffer::frames)
            .collect::<Vec<_>>(),
        [3, 2]
    );
    assert_eq!(
        asr.received
            .iter()
            .flat_map(|audio| audio.samples().iter().copied())
            .collect::<Vec<_>>(),
        input.samples()
    );
}
#[test]
fn unsupported_voice_conditioning_is_rejected_instead_of_silently_ignored() {
    let mut model: Box<dyn TtsModel> = Box::new(Tone);
    let mut request = ConditionedSynthesisRequest::new(SynthesisRequest::new("hello"));
    request.speaker_embedding = Some(vec![f32::NAN]);
    assert!(matches!(
        model.synthesize_conditioned(&request, &CancellationToken::default(), &mut |_| Ok(())),
        Err(Error::InvalidInput(_))
    ));
    request.speaker_embedding = Some(vec![0.; 4]);
    assert!(matches!(
        model.synthesize_conditioned(&request, &CancellationToken::default(), &mut |_| Ok(())),
        Err(Error::Unsupported(_))
    ));
    request.speaker_embedding = None;
    assert!(model
        .synthesize_conditioned(&request, &CancellationToken::default(), &mut |_| Ok(()))
        .is_ok());
}
struct FailedPoll {
    pending: bool,
}
impl RealtimeModel for FailedPoll {
    fn capabilities(&self) -> RealtimeCapabilities {
        caps()
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        _: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        self.pending = true;
        Ok(())
    }
    fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.pending = false;
        Ok(())
    }
    fn output_pending(&self) -> bool {
        self.pending
    }
    fn poll_output(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<PollStatus> {
        Err(Error::InvalidInput("decode failed".into()))
    }
}
#[test]
fn faulted_worker_does_not_busy_poll_or_flood_errors_and_can_reset() {
    let worker = RealtimeWorker::spawn(
        FailedPoll { pending: false },
        SessionLimits::default(),
        WorkerOptions::default(),
    )
    .unwrap();
    worker.try_send(WorkerCommand::Audio(pcm(1))).unwrap();
    assert!(matches!(
        worker.recv_timeout(Duration::from_secs(1)).unwrap(),
        WorkerEvent::Error {
            error: Error::InvalidInput(_),
            ..
        }
    ));
    assert!(matches!(
        worker.recv_timeout(Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    worker.try_send(WorkerCommand::ResetConversation).unwrap();
    assert!(matches!(
        worker.recv_timeout(Duration::from_secs(1)).unwrap(),
        WorkerEvent::CommandCompleted { .. }
    ));
}
#[test]
fn empty_inputs_still_consume_message_budget_before_context_mutation() {
    let mut s = RealtimeSession::with_limits(
        Chat::default(),
        SessionLimits {
            max_input_messages_per_turn: 2,
            ..Default::default()
        },
    )
    .unwrap();
    s.push_text("", &mut |_| Ok(())).unwrap();
    s.push_audio(&pcm(0), &mut |_| Ok(())).unwrap();
    assert!(matches!(
        s.push_text("", &mut |_| Ok(())),
        Err(Error::LimitExceeded(_))
    ));
    assert_eq!(s.into_model().draft.len(), 1);
}

#[test]
fn clearing_context_between_file_turns_keeps_ids_contiguous_and_rotates_cancel_handle() {
    let mut session = RealtimeSession::new(Chat::default(), 100, 1000).unwrap();
    let mut events = vec![];
    let mut sink = |e| {
        events.push(e);
        Ok(())
    };
    session.push_text("old", &mut sink).unwrap();
    assert!(session.clear_context().is_err());
    session.commit(&mut sink).unwrap();
    let turn = session.turn();
    let previous = session.cancellation_token();
    session.clear_context().unwrap();
    assert_eq!(session.turn(), turn);
    previous.cancel();
    assert!(!session.cancellation_token().is_cancelled());
    session.push_text("fresh", &mut sink).unwrap();
    session.commit(&mut sink).unwrap();
    let texts = events
        .into_iter()
        .filter_map(|e| {
            if let AudioEvent::Text { turn, text } = e {
                Some((turn, text))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(texts, vec![(0, "old".into()), (1, "fresh".into())]);
}

#[test]
fn worker_cancel_during_generation_preserves_committed_history_and_accepts_next_turn() {
    use std::sync::mpsc;
    struct Conversation {
        history: Vec<String>,
        draft: String,
        started: mpsc::SyncSender<()>,
    }
    impl RealtimeModel for Conversation {
        fn capabilities(&self) -> RealtimeCapabilities {
            caps()
        }
        fn push_audio(
            &mut self,
            _: &AudioBuffer,
            _: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> Result<()> {
            Ok(())
        }
        fn push_text(
            &mut self,
            text: &str,
            cancel: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> Result<()> {
            cancel.check()?;
            self.draft = text.into();
            Ok(())
        }
        fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
            if self.draft == "blocked" {
                self.started.send(()).unwrap();
                loop {
                    cancel.check()?;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            sink.emit(AudioEvent::Text {
                turn: 0,
                text: self
                    .history
                    .iter()
                    .chain(std::iter::once(&self.draft))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("|"),
            })
        }
        fn end_turn(&mut self) -> Result<()> {
            self.history.push(std::mem::take(&mut self.draft));
            Ok(())
        }
        fn cancel_turn(&mut self) -> Result<()> {
            self.draft.clear();
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            self.history.clear();
            self.draft.clear();
            Ok(())
        }
    }
    fn completed(worker: &RealtimeWorker, id: u64) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut text = Vec::new();
        loop {
            assert!(Instant::now() < deadline, "worker command did not finish");
            match worker.recv_timeout(Duration::from_millis(100)) {
                Ok(WorkerEvent::CommandCompleted { id: done }) if done == id => return text,
                Ok(WorkerEvent::Event(AudioEvent::Text { text: delta, .. })) => text.push(delta),
                Ok(WorkerEvent::Error {
                    error: Error::Cancelled,
                    ..
                }) => {}
                Ok(WorkerEvent::Error { error, .. }) => panic!("unexpected worker error: {error}"),
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(error) => panic!("worker disconnected: {error}"),
            }
        }
    }
    let (started, rx) = mpsc::sync_channel(1);
    let mut worker = RealtimeWorker::spawn(
        Conversation {
            history: vec![],
            draft: String::new(),
            started,
        },
        SessionLimits::default(),
        WorkerOptions::default(),
    )
    .unwrap();
    let id = worker.try_send(WorkerCommand::Text("kept".into())).unwrap();
    completed(&worker, id);
    let id = worker.try_send(WorkerCommand::Commit).unwrap();
    assert_eq!(completed(&worker, id), ["kept"]);
    let id = worker
        .try_send(WorkerCommand::Text("blocked".into()))
        .unwrap();
    completed(&worker, id);
    worker.try_send(WorkerCommand::Commit).unwrap();
    rx.recv_timeout(Duration::from_secs(5)).unwrap();
    worker.cancel_current();
    let id = worker.try_send(WorkerCommand::Cancel).unwrap();
    completed(&worker, id);
    let id = worker.try_send(WorkerCommand::Text("next".into())).unwrap();
    completed(&worker, id);
    let id = worker.try_send(WorkerCommand::Commit).unwrap();
    assert_eq!(completed(&worker, id), ["kept|next"]);
    worker.close().unwrap();
}
