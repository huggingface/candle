use candle_audio::*;

struct Events(Vec<AudioEvent>);
impl RealtimeModel for Events {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: None,
            output_format: Some(AudioFormat::mono(24000).unwrap()),
            max_input_frames_per_turn: None,
            text_input: true,
            text_output: true,
            streaming: StreamingMode::Native,
            output_streaming: Some(StreamingMode::Native),
        }
    }
    fn push_text(
        &mut self,
        _: &str,
        _: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        for event in self.0.clone() {
            sink.emit(event)?;
        }
        Ok(())
    }
    fn push_audio(
        &mut self,
        _: &AudioBuffer,
        _: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        unreachable!()
    }
    fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
fn transcript(text: &str) -> Transcript {
    Transcript {
        text: text.into(),
        start: 0.,
        end: 0.5,
    }
}
fn segment(text: &str, status: SegmentStatus, revision: u64) -> AudioEvent {
    AudioEvent::TranscriptSegment {
        turn: 99,
        transcript: transcript(text),
        status,
        metadata: SegmentMetadata {
            segment_id: "phrase".into(),
            revision,
            ..Default::default()
        },
    }
}
fn audio(start: Option<u64>, n: usize) -> AudioEvent {
    let audio = AudioBuffer::new(AudioFormat::mono(24000).unwrap(), vec![0.25; n]).unwrap();
    match start {
        Some(start_frame) => AudioEvent::AudioChunk {
            turn: 99,
            audio,
            start_frame,
        },
        None => AudioEvent::Audio { turn: 99, audio },
    }
}
#[test]
fn legacy_events_gain_unique_segments_and_turn_relative_audio_clocks() {
    let events = vec![
        AudioEvent::Transcript {
            turn: 99,
            transcript: transcript("hello"),
        },
        audio(None, 3),
        audio(None, 2),
    ];
    let mut s = RealtimeSession::new(Events(events), 10, 10)
        .unwrap()
        .with_structured_events()
        .unwrap();
    let mut all = Vec::new();
    for _ in 0..2 {
        let mut sink = |event| {
            all.push(event);
            Ok(())
        };
        s.push_text("go", &mut sink).unwrap();
        s.commit(&mut sink).unwrap();
    }
    let mut ids = Vec::new();
    let mut clocks = Vec::new();
    for e in all {
        match e {
            AudioEvent::TranscriptSegment {
                turn,
                metadata,
                status,
                ..
            } => {
                assert_eq!(status, SegmentStatus::Final);
                assert_eq!(metadata.revision, 0);
                ids.push((turn, metadata.segment_id));
            }
            AudioEvent::AudioChunk {
                turn, start_frame, ..
            } => clocks.push((turn, start_frame)),
            AudioEvent::TurnCompleted { .. } => {}
            e => panic!("unstructured event {e:?}"),
        }
    }
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0].1, ids[1].1);
    assert_eq!(clocks, vec![(0, 0), (0, 3), (1, 0), (1, 3)]);
}
#[test]
fn transcript_updates_replace_one_identity_and_reject_stale_or_unknown_revisions() {
    let updates = vec![
        segment("hel", SegmentStatus::Partial, 0),
        segment("hello", SegmentStatus::Final, 1),
        segment("Hello!", SegmentStatus::Revision, 2),
    ];
    let mut s = RealtimeSession::new(Events(updates), 10, 10).unwrap();
    let mut current = None;
    s.push_text("go", &mut |e| {
        if let AudioEvent::TranscriptSegment {
            turn,
            transcript,
            metadata,
            ..
        } = e
        {
            assert_eq!(turn, 0);
            assert_eq!(metadata.segment_id, "phrase");
            current = Some(transcript.text);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(current.as_deref(), Some("Hello!"));
    for invalid in [
        vec![segment("x", SegmentStatus::Revision, 0)],
        vec![segment("x", SegmentStatus::Partial, 1)],
        vec![
            segment("x", SegmentStatus::Partial, 0),
            segment("y", SegmentStatus::Final, 0),
        ],
        vec![
            segment("x", SegmentStatus::Final, 0),
            segment("y", SegmentStatus::Partial, 1),
        ],
    ] {
        let mut s = RealtimeSession::new(Events(invalid), 10, 10).unwrap();
        assert!(matches!(
            s.push_text("go", &mut |_| Ok(())),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(s.state(), SessionState::Faulted);
    }
}
#[test]
fn clock_overlap_is_rejected_but_explicit_gaps_are_preserved() {
    let mut s = RealtimeSession::new(Events(vec![audio(Some(10), 3), audio(None, 2)]), 10, 10)
        .unwrap()
        .with_structured_events()
        .unwrap();
    let mut clocks = vec![];
    s.push_text("go", &mut |e| {
        if let AudioEvent::AudioChunk { start_frame, .. } = e {
            clocks.push(start_frame);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(clocks, vec![10, 13]);
    let mut s =
        RealtimeSession::new(Events(vec![audio(Some(10), 3), audio(Some(12), 1)]), 10, 10).unwrap();
    assert!(matches!(
        s.push_text("go", &mut |_| Ok(())),
        Err(Error::InvalidInput(_))
    ));
}
#[test]
fn structured_mode_is_opt_in_and_metadata_counts_toward_budget() {
    let events = vec![AudioEvent::Transcript {
        turn: 99,
        transcript: transcript("x"),
    }];
    let mut s = RealtimeSession::new(Events(events.clone()), 10, 10).unwrap();
    s.push_text("go", &mut |e| {
        assert!(matches!(e, AudioEvent::Transcript { turn: 0, .. }));
        Ok(())
    })
    .unwrap();
    assert!(s.with_structured_events().is_err());
    let mut s = RealtimeSession::with_limits(
        Events(events),
        SessionLimits {
            max_output_text_bytes_per_turn: 1,
            ..Default::default()
        },
    )
    .unwrap()
    .with_structured_events()
    .unwrap();
    let mut emitted = false;
    assert!(matches!(
        s.push_text("go", &mut |_| {
            emitted = true;
            Ok(())
        }),
        Err(Error::LimitExceeded(_))
    ));
    assert!(!emitted);
}

#[test]
fn worker_transports_structured_events_and_resets_clock_after_cancellation() {
    use std::time::{Duration, Instant};
    let worker = RealtimeWorker::spawn_structured(
        Events(vec![
            AudioEvent::Transcript {
                turn: 99,
                transcript: transcript("hello"),
            },
            audio(None, 3),
        ]),
        SessionLimits::default(),
        WorkerOptions::default(),
    )
    .unwrap();
    for turn in 0..2 {
        let text = worker.try_send(WorkerCommand::Text("go".into())).unwrap();
        let mut saw_segment = false;
        let mut saw_audio = false;
        let until = Instant::now() + Duration::from_secs(3);
        loop {
            assert!(Instant::now() < until);
            match worker.recv_timeout(Duration::from_secs(1)).unwrap() {
                WorkerEvent::Event(AudioEvent::TranscriptSegment {
                    turn: actual,
                    metadata,
                    ..
                }) => {
                    assert_eq!(actual, turn);
                    assert_eq!(metadata.revision, 0);
                    saw_segment = true;
                }
                WorkerEvent::Event(AudioEvent::AudioChunk {
                    turn: actual,
                    start_frame,
                    ..
                }) => {
                    assert_eq!(actual, turn);
                    assert_eq!(start_frame, 0);
                    saw_audio = true;
                }
                WorkerEvent::CommandCompleted { id } if id == text => break,
                e => panic!("unexpected worker event {e:?}"),
            }
        }
        assert!(saw_segment && saw_audio);
        let cancelled = worker.try_send(WorkerCommand::Cancel).unwrap();
        let mut ack = false;
        loop {
            match worker.recv_timeout(Duration::from_secs(1)).unwrap() {
                WorkerEvent::Event(AudioEvent::Cancelled { turn: actual }) => {
                    assert_eq!(actual, turn);
                    ack = true;
                }
                WorkerEvent::CommandCompleted { id } if id == cancelled => break,
                e => panic!("unexpected cancellation event {e:?}"),
            }
        }
        assert!(ack);
    }
}
