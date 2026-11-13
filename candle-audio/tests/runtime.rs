use candle_audio::*;

fn mono(rate: u32, samples: Vec<f32>) -> AudioBuffer {
    AudioBuffer::new(AudioFormat::mono(rate).unwrap(), samples).unwrap()
}

#[test]
fn invalid_pcm_and_deserialized_formats_are_rejected() {
    assert!(AudioFormat::new(0, 1).is_err());
    assert!(AudioFormat::new(16000, 0).is_err());
    assert!(AudioBuffer::new(AudioFormat::new(16000, 2).unwrap(), vec![1.]).is_err());
    assert!(AudioBuffer::new(AudioFormat::mono(16000).unwrap(), vec![f32::NAN]).is_err());
    assert!(serde_json::from_str::<AudioBuffer>(
        r#"{"format":{"sample_rate":0,"channels":1},"samples":[]}"#
    )
    .is_err());
    assert!(serde_json::from_str::<AudioBuffer>(
        r#"{"format":{"sample_rate":16000,"channels":2},"samples":[1]}"#
    )
    .is_err());
}

#[test]
fn all_channels_contribute_to_downmix() {
    let pcm =
        AudioBuffer::new(AudioFormat::new(16000, 2).unwrap(), vec![1., -1., 0., 0.5]).unwrap();
    assert_eq!(
        AudioConverter::convert(&pcm, 16000).unwrap().samples(),
        &[0., 0.25]
    );
    let full = AudioBuffer::new(AudioFormat::new(16000, 2).unwrap(), vec![f32::MAX; 2]).unwrap();
    assert_eq!(
        AudioConverter::convert(&full, 16000).unwrap().samples(),
        &[f32::MAX]
    );
}

#[test]
fn resampling_is_independent_of_input_chunk_boundaries_and_drains_tail() {
    for (input_rate, output_rate, length) in [
        (48000, 16000, 48007),
        (44100, 24000, 11263),
        (16000, 24000, 71),
        (24000, 8000, 1),
    ] {
        let pcm = mono(
            input_rate,
            (0..length).map(|i| (i as f32 * 0.031).sin()).collect(),
        );
        let expected = AudioConverter::convert(&pcm, output_rate).unwrap();
        assert_eq!(
            expected.frames(),
            ((length as u64 * output_rate as u64 + input_rate as u64 / 2) / input_rate as u64)
                as usize
        );
        let mut converter = AudioConverter::new(pcm.format(), output_rate).unwrap();
        let mut samples = Vec::new();
        for part in pcm.samples().chunks(137) {
            samples.extend(
                converter
                    .push(&mono(input_rate, part.to_vec()))
                    .unwrap()
                    .into_samples(),
            );
        }
        samples.extend(converter.finish().unwrap().into_samples());
        assert_eq!(samples, expected.samples());
        assert!(converter.finish().is_err());
        assert!(converter.push(&mono(input_rate, vec![1.])).is_err());
        converter.reset();
        let mut restarted = converter.push(&pcm).unwrap().into_samples();
        restarted.extend(converter.finish().unwrap().into_samples());
        assert_eq!(restarted, expected.samples());
    }
}

#[test]
fn downsampling_filters_frequencies_above_target_nyquist() {
    let pcm = mono(
        48000,
        (0..48000)
            .map(|i| (std::f32::consts::TAU * 12000. * i as f32 / 48000.).sin())
            .collect(),
    );
    let result = AudioConverter::convert(&pcm, 16000).unwrap();
    let middle = &result.samples()[256..result.frames() - 256];
    let rms = (middle.iter().map(|x| x * x).sum::<f32>() / middle.len() as f32).sqrt();
    assert!(rms < 0.02, "out-of-band tone leaked with RMS {rms}");
}

#[derive(Default)]
struct CountingAsr {
    calls: Vec<Vec<f32>>,
}
impl AsrModel for CountingAsr {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::mono(16000).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        16000
    }
    fn transcribe(
        &mut self,
        pcm: &AudioBuffer,
        _: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<Transcript> {
        cancel.check()?;
        self.calls.push(pcm.samples().to_vec());
        Ok(Transcript {
            text: format!("{} frames", pcm.frames()),
            start: 0.,
            end: pcm.duration(),
        })
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
fn stream(window: usize) -> AsrStream<CountingAsr> {
    AsrStream::new(
        CountingAsr::default(),
        AudioFormat::mono(16000).unwrap(),
        window,
        TranscriptionOptions::default(),
    )
    .unwrap()
}

#[test]
fn irregular_chunks_preserve_samples_timestamps_and_final_short_segment() {
    let mut session = RealtimeSession::new(stream(1000), 2000, 5000).unwrap();
    let input: Vec<f32> = (0..2501).map(|i| i as f32 / 2501.).collect();
    let mut events = Vec::new();
    let mut sink = |e| {
        events.push(e);
        Ok(())
    };
    for part in input.chunks(733) {
        session
            .push_audio(&mono(16000, part.to_vec()), &mut sink)
            .unwrap();
    }
    session.finish(&mut sink).unwrap();
    assert_eq!(session.state(), SessionState::Closed);
    let model = session.into_model().into_model();
    assert_eq!(
        model.calls.iter().flatten().copied().collect::<Vec<_>>(),
        input
    );
    assert_eq!(
        model.calls.iter().map(Vec::len).collect::<Vec<_>>(),
        [1000, 1000, 501]
    );
    let transcripts: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let AudioEvent::Transcript { transcript, .. } = e {
                Some(transcript)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(transcripts[1].start, 1000. / 16000.);
    assert_eq!(transcripts[2].end, 2501. / 16000.);
    assert!(matches!(
        events.last(),
        Some(AudioEvent::TurnCompleted { turn: 0 })
    ));
}

#[test]
fn limits_reject_input_without_consuming_budget_and_reset_prevents_turn_leaks() {
    let mut session = RealtimeSession::new(stream(1000), 1000, 1500).unwrap();
    let mut events = Vec::new();
    let mut sink = |e| {
        events.push(e);
        Ok(())
    };
    assert!(session
        .push_audio(&mono(16000, vec![0.; 1001]), &mut sink)
        .is_err());
    assert_eq!(session.state(), SessionState::Ready);
    session
        .push_audio(&mono(16000, vec![0.1; 1000]), &mut sink)
        .unwrap();
    assert!(session
        .push_audio(&mono(16000, vec![0.; 501]), &mut sink)
        .is_err());
    session
        .push_audio(&mono(16000, vec![0.2; 500]), &mut sink)
        .unwrap();
    session.commit(&mut sink).unwrap();
    session
        .push_audio(&mono(16000, vec![0.3; 1]), &mut sink)
        .unwrap();
    session.commit(&mut sink).unwrap();
    let latest = events
        .iter()
        .rev()
        .find_map(|e| match e {
            AudioEvent::Transcript { turn, transcript } => Some((*turn, transcript)),
            _ => None,
        })
        .unwrap();
    assert_eq!(latest.0, 1);
    assert_eq!(latest.1.start, 0.);
    assert_eq!(latest.1.text, "1 frames");
}

#[test]
fn cancellation_discards_pending_audio_and_old_handles_cannot_cancel_new_turns() {
    let mut session = RealtimeSession::new(stream(1000), 1000, 5000).unwrap();
    let token = session.cancellation_token();
    let mut events = Vec::new();
    let mut sink = |e| {
        events.push(e);
        Ok(())
    };
    session
        .push_audio(&mono(16000, vec![1.; 500]), &mut sink)
        .unwrap();
    token.cancel();
    assert!(matches!(session.commit(&mut sink), Err(Error::Cancelled)));
    assert_eq!(session.state(), SessionState::Ready);
    token.cancel();
    session
        .push_audio(&mono(16000, vec![0.5; 5]), &mut sink)
        .unwrap();
    session.commit(&mut sink).unwrap();
    assert!(matches!(events[0], AudioEvent::Cancelled { turn: 0 }));
    let model = session.into_model().into_model();
    assert_eq!(model.calls, vec![vec![0.5; 5]]);
}

#[test]
fn sink_failure_faults_session_and_requires_explicit_reset() {
    let mut session = RealtimeSession::new(stream(10), 1000, 5000).unwrap();
    let mut broken = |_| Err(Error::InvalidState("sink closed".into()));
    assert!(session
        .push_audio(&mono(16000, vec![1.; 10]), &mut broken)
        .is_err());
    assert_eq!(session.state(), SessionState::Faulted);
    assert!(session.commit(&mut |_| Ok(())).is_err());
    session.reset().unwrap();
    assert_eq!(session.state(), SessionState::Ready);
    session
        .push_audio(&mono(16000, vec![0.; 1]), &mut |_| Ok(()))
        .unwrap();
    session.finish(&mut |_| Ok(())).unwrap();
    assert!(session.reset().is_err());
}

struct ToneTts;
impl TtsModel for ToneTts {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::mono(24000).unwrap()
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Native
    }
    fn synthesize(
        &mut self,
        r: &SynthesisRequest,
        c: &CancellationToken,
        s: &mut dyn EventSink,
    ) -> Result<()> {
        c.check()?;
        assert!(!r.text.is_empty());
        s.emit(AudioEvent::Audio {
            turn: 99,
            audio: mono(24000, vec![0.25; 240]),
        })
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn recognition_and_synthesis_compose_without_model_specific_runtime_dispatch() {
    let pipeline = SpeechPipeline::new(stream(1000), ToneTts, |t: &Transcript| {
        Ok(Some(SynthesisRequest::new(format!("heard {}", t.text))))
    });
    let mut session = RealtimeSession::new(pipeline, 1000, 5000).unwrap();
    let mut events = Vec::new();
    let mut sink = |e| {
        events.push(e);
        Ok(())
    };
    session
        .push_audio(&mono(16000, vec![0.; 12]), &mut sink)
        .unwrap();
    session.finish(&mut sink).unwrap();
    assert!(matches!(&events[1], AudioEvent::Text{turn:0,text} if text=="heard 12 frames"));
    assert!(matches!(&events[2], AudioEvent::Audio{turn:0,audio} if audio.frames()==240));
    assert!(matches!(events[3], AudioEvent::TurnCompleted { turn: 0 }));
}

#[test]
fn speech_pipeline_advertises_its_own_text_capabilities() {
    struct TextEnabledRecognizer;
    impl RealtimeModel for TextEnabledRecognizer {
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities {
                input_format: Some(AudioFormat::mono(16000).unwrap()),
                output_format: None,
                max_input_frames_per_turn: None,
                text_input: true,
                text_output: false,
                streaming: StreamingMode::Native,
                output_streaming: None,
            }
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
            _: &str,
            _: &CancellationToken,
            _: &mut dyn EventSink,
        ) -> Result<()> {
            panic!("the pipeline must not advertise or forward text input")
        }
        fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            Ok(())
        }
    }
    let pipeline = SpeechPipeline::new(TextEnabledRecognizer, ToneTts, |_: &Transcript| Ok(None));
    let mut session = RealtimeSession::new(pipeline, 1000, 5000).unwrap();
    assert!(!session.capabilities().text_input);
    assert!(session.capabilities().text_output);
    assert!(matches!(
        session.push_text("hello", &mut |_| Ok(())),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(session.state(), SessionState::Ready);
}

#[test]
fn text_sessions_use_the_same_lifecycle_and_do_not_replay_previous_utterances() {
    let model = TtsStream::new(
        ToneTts,
        SynthesisRequest::new("replaced template text"),
        100,
    )
    .unwrap();
    let mut session = RealtimeSession::new(model, 1000, 1000).unwrap();
    assert!(session.capabilities().input_format.is_none());
    assert!(session
        .push_audio(&mono(16000, vec![1.]), &mut |_| Ok(()))
        .is_err());
    let mut events = Vec::new();
    let mut sink = |e| {
        events.push(e);
        Ok(())
    };
    session.push_text("hello", &mut sink).unwrap();
    session.commit(&mut sink).unwrap();
    session.push_text("goodbye", &mut sink).unwrap();
    session.finish(&mut sink).unwrap();
    assert!(matches!(&events[0],AudioEvent::Text{turn:0,text} if text=="hello"));
    assert!(matches!(&events[3],AudioEvent::Text{turn:1,text} if text=="goodbye"));
    assert!(matches!(
        events.last(),
        Some(AudioEvent::TurnCompleted { turn: 1 })
    ));
}

struct BoundedNative;
impl RealtimeModel for BoundedNative {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: Some(AudioFormat::mono(16000).unwrap()),
            output_format: Some(AudioFormat::mono(16000).unwrap()),
            max_input_frames_per_turn: Some(8),
            text_input: false,
            text_output: false,
            streaming: StreamingMode::Native,
            output_streaming: Some(StreamingMode::Native),
        }
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        sink.emit(AudioEvent::Audio {
            turn: 0,
            audio: audio.clone(),
        })
    }
    fn commit(&mut self, cancel: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        cancel.check()
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn native_model_context_limits_override_larger_session_limits_before_accepting_input() {
    let mut session = RealtimeSession::new(BoundedNative, 100, 100).unwrap();
    assert_eq!(session.max_turn_frames(), 8);
    let mut output = Vec::new();
    let mut sink = |event| {
        if let AudioEvent::Audio { audio, .. } = event {
            output.extend(audio.into_samples());
        }
        Ok(())
    };
    session
        .push_audio(&mono(16000, vec![1.; 5]), &mut sink)
        .unwrap();
    assert!(matches!(
        session.push_audio(&mono(16000, vec![9.; 4]), &mut sink),
        Err(Error::LimitExceeded(_))
    ));
    assert_eq!(session.state(), SessionState::Active);
    session
        .push_audio(&mono(16000, vec![2.; 3]), &mut sink)
        .unwrap();
    session.finish(&mut sink).unwrap();
    assert_eq!(output, vec![1., 1., 1., 1., 1., 2., 2., 2.]);
}
