use candle::{DType, Device, Tensor};
use candle_audio::*;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{dac, parler_tts, t5},
};

struct BudgetTts {
    finish: FinishReason,
}
impl TtsModel for BudgetTts {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::mono(24000).unwrap()
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Native
    }
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        let required_steps = if request.text == "short" { 1 } else { 2 };
        self.finish = if request.max_steps >= required_steps {
            FinishReason::EndOfSequence
        } else {
            FinishReason::Length
        };
        sink.emit(AudioEvent::Audio {
            turn: 0,
            audio: AudioBuffer::new(self.output_format(), vec![0.; 4])?,
        })
    }
    fn finish_reason(&self) -> FinishReason {
        self.finish
    }
    fn reset(&mut self) -> Result<()> {
        self.finish = FinishReason::Completed;
        Ok(())
    }
}

#[test]
fn boxed_tts_and_sessions_preserve_eos_and_length() {
    for (steps, expected) in [(1, FinishReason::Length), (2, FinishReason::EndOfSequence)] {
        let mut request = SynthesisRequest::new("hello");
        request.max_steps = steps;
        let mut model: Box<dyn TtsModel> = Box::new(BudgetTts {
            finish: FinishReason::Completed,
        });
        model
            .synthesize_conditioned(
                &ConditionedSynthesisRequest::new(request.clone()),
                &CancellationToken::default(),
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(model.finish_reason(), expected);

        let stream = TtsStream::new(model, request, 100).unwrap();
        let mut session = RealtimeSession::new(stream, 100, 100).unwrap();
        for _ in 0..2 {
            session.push_text("hello", &mut |_| Ok(())).unwrap();
            session.commit(&mut |_| Ok(())).unwrap();
            // end_turn resets the fake model, so the session must capture the
            // reason before resetting it and preserve it for the caller.
            assert_eq!(session.last_finish_reason(), Some(expected));
        }
    }
}

#[test]
fn tts_stream_keeps_earlier_truncation_until_turn_reset() {
    let mut request = SynthesisRequest::new("hello");
    request.max_steps = 1;
    let stream = TtsStream::new(
        BudgetTts {
            finish: FinishReason::Completed,
        },
        request,
        100,
    )
    .unwrap();
    let mut session = RealtimeSession::new(stream, 100, 100).unwrap();
    session.push_text("hello", &mut |_| Ok(())).unwrap();
    session.push_text("short", &mut |_| Ok(())).unwrap();
    session.commit(&mut |_| Ok(())).unwrap();
    assert_eq!(session.last_finish_reason(), Some(FinishReason::Length));
    session.push_text("short", &mut |_| Ok(())).unwrap();
    session.commit(&mut |_| Ok(())).unwrap();
    assert_eq!(
        session.last_finish_reason(),
        Some(FinishReason::EndOfSequence)
    );
}

struct WindowAsr(usize);
impl AsrModel for WindowAsr {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::mono(16000).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        4
    }
    fn transcribe(
        &mut self,
        audio: &AudioBuffer,
        _: &TranscriptionOptions,
        _: &CancellationToken,
    ) -> Result<Transcript> {
        Ok(Transcript {
            text: "hello".into(),
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
        self.0 += 1;
        Ok(TranscriptionResult {
            transcript: self.transcribe(audio, options, cancel)?,
            finish_reason: if self.0 == 1 {
                FinishReason::Length
            } else {
                FinishReason::EndOfSequence
            },
            language: None,
        })
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn asr_stream_keeps_earlier_window_truncation_until_turn_reset() {
    let format = AudioFormat::mono(16000).unwrap();
    let stream = AsrStream::new(WindowAsr(0), format, 4, TranscriptionOptions::default()).unwrap();
    let mut session = RealtimeSession::new(stream, 100, 100).unwrap();
    let mut reasons = Vec::new();
    let mut sink = |event| {
        if let AudioEvent::TranscriptionMetadata { finish_reason, .. } = event {
            reasons.push(finish_reason);
        }
        Ok(())
    };
    session
        .push_audio(&AudioBuffer::new(format, vec![0.; 8]).unwrap(), &mut sink)
        .unwrap();
    session.commit(&mut sink).unwrap();
    assert_eq!(session.last_finish_reason(), Some(FinishReason::Length));
    // Per-window metadata still reflects each window's own result.
    assert_eq!(
        reasons,
        vec![FinishReason::Length, FinishReason::EndOfSequence]
    );
    session
        .push_audio(&AudioBuffer::new(format, vec![0.; 4]).unwrap(), &mut |_| {
            Ok(())
        })
        .unwrap();
    session.commit(&mut |_| Ok(())).unwrap();
    assert_eq!(
        session.last_finish_reason(),
        Some(FinishReason::EndOfSequence)
    );
}

struct CompletionAsr(FinishReason);
impl RealtimeModel for CompletionAsr {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: Some(AudioFormat::mono(16000).unwrap()),
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
        _: &AudioBuffer,
        _: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        sink.emit(AudioEvent::Transcript {
            turn: 0,
            transcript: Transcript {
                text: "hello".into(),
                start: 0.,
                end: 1.,
            },
        })
    }
    fn commit(&mut self, _: &CancellationToken, _: &mut dyn EventSink) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn finish_reason(&self) -> FinishReason {
        self.0
    }
}

#[test]
fn speech_pipeline_reports_truncation_from_either_model_and_asr_without_tts() {
    for (asr, steps, expected) in [
        (FinishReason::Length, Some(2), FinishReason::Length),
        (FinishReason::EndOfSequence, Some(1), FinishReason::Length),
        (
            FinishReason::Completed,
            Some(2),
            FinishReason::EndOfSequence,
        ),
        (FinishReason::Length, None, FinishReason::Length),
        (
            FinishReason::EndOfSequence,
            None,
            FinishReason::EndOfSequence,
        ),
    ] {
        let mut pipeline = SpeechPipeline::new(
            CompletionAsr(asr),
            BudgetTts {
                finish: FinishReason::Completed,
            },
            move |_: &Transcript| {
                Ok(steps.map(|steps| {
                    let mut request = SynthesisRequest::new("hello");
                    request.max_steps = steps;
                    request
                }))
            },
        );
        pipeline
            .push_audio(
                &AudioBuffer::new(AudioFormat::mono(16000).unwrap(), vec![0.]).unwrap(),
                &CancellationToken::default(),
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(pipeline.finish_reason(), expected);
        pipeline.end_turn().unwrap();
        assert_eq!(pipeline.finish_reason(), asr);
    }
}

#[test]
fn speech_pipeline_keeps_earlier_truncation_until_turn_reset() {
    let mut calls = 0;
    let mut pipeline = SpeechPipeline::new(
        CompletionAsr(FinishReason::EndOfSequence),
        BudgetTts {
            finish: FinishReason::Completed,
        },
        move |_: &Transcript| {
            calls += 1;
            let mut request = SynthesisRequest::new("hello");
            request.max_steps = if calls == 1 { 1 } else { 2 };
            Ok(Some(request))
        },
    );
    let input = AudioBuffer::new(AudioFormat::mono(16000).unwrap(), vec![0.]).unwrap();
    for _ in 0..2 {
        pipeline
            .push_audio(&input, &CancellationToken::default(), &mut |_| Ok(()))
            .unwrap();
        assert_eq!(pipeline.finish_reason(), FinishReason::Length);
    }
    pipeline.end_turn().unwrap();
    pipeline
        .push_audio(&input, &CancellationToken::default(), &mut |_| Ok(()))
        .unwrap();
    assert_eq!(pipeline.finish_reason(), FinishReason::EndOfSequence);
}

fn tiny_parler() -> candle::Result<parler_tts::Model> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let decoder = parler_tts::DecoderConfig {
        vocab_size: 2,
        max_position_embeddings: 8,
        num_hidden_layers: 0,
        ffn_dim: 2,
        num_attention_heads: 1,
        num_key_value_heads: None,
        num_cross_attention_key_value_heads: None,
        activation_function: candle_nn::Activation::Relu,
        hidden_size: 2,
        scale_embedding: false,
        num_codebooks: 2,
        pad_token_id: 0,
        bos_token_id: 2,
        eos_token_id: 0,
        tie_word_embeddings: false,
        rope_embeddings: false,
        rope_theta: 10000.,
    };
    let text_encoder = t5::Config {
        vocab_size: 2,
        d_model: 2,
        d_kv: 2,
        d_ff: 2,
        num_layers: 0,
        num_heads: 1,
        ..Default::default()
    };
    let audio_encoder = dac::Config {
        num_codebooks: 2,
        codebook_size: 2,
        latent_dim: 2,
        model_bitrate: 0,
        frame_rate: 0,
        sampling_rate: 24000,
    };
    // The vocoder is never executed in these generation control-flow tests.
    // A tiny architecture avoids allocating checkpoint-sized unused weights.
    Ok(parler_tts::Model {
        embed_prompts: candle_nn::embedding(2, 2, vb.pp("prompt"))?,
        enc_to_dec_proj: None,
        decoder: parler_tts::Decoder::new(&decoder, vb.pp("decoder"))?,
        text_encoder: t5::T5EncoderModel::load(vb.pp("text"), &text_encoder)?,
        decoder_start_token_id: 2,
        pad_token_id: 0,
        audio_encoder: dac::Model::with_architecture(
            &audio_encoder,
            2,
            &[2],
            2,
            &[2],
            2,
            vb.pp("audio"),
        )?,
    })
}

#[test]
fn parler_reports_eos_even_on_last_step_and_same_shaped_length_output() -> candle::Result<()> {
    let mut model = tiny_parler()?;
    let tokens = Tensor::new(&[[1u32]], &Device::Cpu)?;
    // Greedy zero logits select token zero. Codebook 0 reaches EOS at step 0,
    // codebook 1 at step 1 because of the real staggered codebook schedule.
    for (steps, expected) in [
        (1, parler_tts::GenerationFinishReason::Length),
        (2, parler_tts::GenerationFinishReason::EndOfSequence),
        (3, parler_tts::GenerationFinishReason::EndOfSequence),
        (0, parler_tts::GenerationFinishReason::Length),
        (1, parler_tts::GenerationFinishReason::Length),
    ] {
        let result = model.generate_with_callback_detailed(
            &tokens,
            &tokens,
            LogitsProcessor::new(0, Some(0.), None),
            steps,
            |_| Ok(()),
        )?;
        assert_eq!(result.finish_reason, expected, "max_steps={steps}");
        assert_eq!(result.audio_codes.dims(), &[2, 0]);
    }
    Ok(())
}

#[test]
fn parler_legacy_generation_and_callback_errors_remain_compatible() -> candle::Result<()> {
    let mut model = tiny_parler()?;
    let tokens = Tensor::new(&[[1u32]], &Device::Cpu)?;
    let codes = model.generate(&tokens, &tokens, LogitsProcessor::new(0, Some(0.), None), 2)?;
    assert_eq!(codes.dims(), &[2, 0]);
    let result = model.generate_with_callback_detailed(
        &tokens,
        &tokens,
        LogitsProcessor::new(0, Some(0.), None),
        2,
        |step| {
            if step == 1 {
                candle::bail!("cancelled at second step");
            }
            Ok(())
        },
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("cancelled at second step"));
    Ok(())
}
