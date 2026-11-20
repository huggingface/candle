use candle::{DType, Device, Tensor};
use candle_audio::{models::gemma4, *};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma4 as native;
use std::collections::HashMap;

struct ShapeBackend(HashMap<String, Vec<usize>>);
impl candle_nn::var_builder::SimpleBackend for ShapeBackend {
    fn get(
        &self,
        requested: candle::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        let expected = self
            .0
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("official checkpoint is missing {name}")))?;
        if requested.dims() != expected {
            candle::bail!(
                "official shape for {name} is {expected:?}, requested {:?}",
                requested.dims()
            );
        }
        // Scalar broadcast views validate all real checkpoint names/shapes
        // without allocating model-sized weight buffers or running inference.
        Tensor::zeros((), dtype, dev)?.broadcast_as(requested)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        let shape = self
            .0
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing {name}")))?;
        self.get(
            candle::Shape::from(shape.clone()),
            name,
            Default::default(),
            dtype,
            dev,
        )
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

#[test]
fn official_audio_text_checkpoint_weight_names_and_shapes_load_without_vision() -> candle::Result<()>
{
    let checkpoints: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/understanding_weight_shapes.json")).unwrap();
    for checkpoint in checkpoints.as_array().unwrap() {
        let shapes = checkpoint["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, tensor)| (name.clone(), shape(&tensor["shape"])))
            .collect();
        let vb = VarBuilder::from_backend(Box::new(ShapeBackend(shapes)), DType::BF16, Device::Cpu);
        match checkpoint["family"].as_str().unwrap() {
            "gemma-4-e2b-it" => {
                let mut cfg: native::config::Gemma4Config =
                    serde_json::from_value(checkpoint["config"].clone()).unwrap();
                gemma4::validate_config(&cfg).unwrap();
                cfg.text_config.max_position_embeddings = 64;
                native::AudioTextModel::new(&cfg, vb)?;
            }
            "voxtral-mini-3b-2507" => {
                let cfg: candle_transformers::models::voxtral::VoxtralConfig =
                    serde_json::from_value(checkpoint["config"].clone()).unwrap();
                candle_audio::models::voxtral::validate_config(&cfg).unwrap();
                candle_transformers::models::voxtral::VoxtralForConditionalGeneration::new(
                    &cfg, vb,
                )?;
            }
            other => panic!("unexpected checkpoint {other}"),
        }
    }
    Ok(())
}

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/gemma4_reference.json")).unwrap()
}
fn values(value: &serde_json::Value) -> Vec<f32> {
    serde_json::from_value(value.clone()).unwrap()
}
fn shape(value: &serde_json::Value) -> Vec<usize> {
    serde_json::from_value(value.clone()).unwrap()
}
fn weights(value: &serde_json::Value) -> candle::Result<VarBuilder<'static>> {
    let tensors: HashMap<_, _> = value
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, item)| {
            Ok((
                name.clone(),
                Tensor::from_vec(values(&item["values"]), shape(&item["shape"]), &Device::Cpu)?,
            ))
        })
        .collect::<candle::Result<_>>()?;
    Ok(VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu))
}
fn close(actual: &Tensor, expected: &serde_json::Value, tolerance: f32) -> candle::Result<()> {
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = values(expected);
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert!(
        error < tolerance,
        "maximum absolute error {error} exceeds {tolerance}"
    );
    Ok(())
}

#[test]
fn gemma4_text_matches_pytorch_with_per_layer_inputs_shared_kv_and_cached_windows(
) -> candle::Result<()> {
    let f = fixture();
    let config: native::Gemma4TextConfig =
        serde_json::from_value(f["text_config"].clone()).unwrap();
    let mut model = native::text::TextModel::new(&config, weights(&f["text_weights"])?)?;
    for case in f["text_cases"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let logits = model.forward(
            &Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?,
            case["offset"].as_u64().unwrap() as usize,
        )?;
        close(&logits, &case["expected"], 2e-5)?;
    }
    model.clear_kv_cache();
    let first = &f["text_cases"][0];
    let ids: Vec<u32> = serde_json::from_value(first["ids"].clone()).unwrap();
    close(
        &model.forward(&Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?, 0)?,
        &first["expected"],
        2e-5,
    )?;
    Ok(())
}

#[test]
fn gemma4_audio_encoder_matches_pytorch_clipping_norms_relative_attention_and_padding(
) -> candle::Result<()> {
    let f = fixture();
    let config: native::Gemma4AudioConfig =
        serde_json::from_value(f["audio_config"].clone()).unwrap();
    let model = native::AudioModel::new(&config, weights(&f["audio_weights"])?)?;
    let mel = Tensor::from_vec(values(&f["audio_input"]), (1, 9, 4), &Device::Cpu)?;
    let mask = Tensor::from_vec(values(&f["audio_padding_mask"]), (1, 9), &Device::Cpu)?;
    let (features, mask) = model.forward(&mel, &mask)?;
    assert_eq!(features.dims(), shape(&f["audio_shape"]));
    close(&features, &f["audio_expected"], 3e-5)?;
    close(&mask, &f["audio_output_padding_mask"], 1e-7)?;
    Ok(())
}

#[test]
fn gemma4_pcm_features_match_primary_numpy_reference_at_padding_boundaries() -> Result<()> {
    for case in fixture()["feature_cases"].as_array().unwrap() {
        let audio = AudioBuffer::new(AudioFormat::mono(16000)?, values(&case["samples"]))?;
        let features =
            gemma4::extract_features(&audio, &Device::Cpu, &CancellationToken::default())?;
        let expected_shape = shape(&case["shape"]);
        assert_eq!(
            features.mel.dims(),
            &[1, expected_shape[0], expected_shape[1]]
        );
        close(&features.mel, &case["expected"], 2e-5)?;
        close(&features.padding_mask, &case["padding_mask"], 1e-7)?;
    }
    Ok(())
}

#[test]
fn gemma4_bf16_text_and_audio_match_primary_pytorch_reference() -> candle::Result<()> {
    let f = fixture();
    let cfg: native::Gemma4TextConfig = serde_json::from_value(f["text_config"].clone()).unwrap();
    let mut text =
        native::text::TextModel::new(&cfg, weights(&f["text_weights"])?.to_dtype(DType::BF16))?;
    for case in f["text_cases"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let logits = text.forward(
            &Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?,
            case["offset"].as_u64().unwrap() as usize,
        )?;
        let actual = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = values(&case["bf16_expected"]);
        eprintln!(
            "BF16 text offset={} max_error={} max_reference={}",
            case["offset"],
            actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max),
            expected.iter().map(|a| a.abs()).fold(0., f32::max)
        );
        close(&logits, &case["bf16_expected"], 0.015)?;
    }
    let cfg: native::Gemma4AudioConfig = serde_json::from_value(f["audio_config"].clone()).unwrap();
    let audio = native::AudioModel::new(&cfg, weights(&f["audio_weights"])?.to_dtype(DType::BF16))?;
    let mel = Tensor::from_vec(values(&f["audio_input"]), (1, 9, 4), &Device::Cpu)?
        .to_dtype(DType::BF16)?;
    let mask = Tensor::from_vec(values(&f["audio_padding_mask"]), (1, 9), &Device::Cpu)?;
    let (features, _) = audio.forward(&mel, &mask)?;
    close(&features, &f["audio_bf16_expected"], 0.015)?;
    Ok(())
}

struct EchoUnderstanding;

#[test]
fn voxtral_frontend_matches_primary_centered_stft_without_extra_tail() -> Result<()> {
    let f = fixture();
    let case = &f["voxtral_features"];
    let filters = candle_audio::models::whisper::Whisper::default_mel_filters(128)?;
    let features = candle_transformers::models::voxtral::extract_features(
        &values(&case["samples"]),
        &filters,
        &Device::Cpu,
    )?;
    assert_eq!(features.dims(), &[1, 128, 3000]);
    let actual = features.to_vec3::<f32>()?;
    for point in case["points"].as_array().unwrap() {
        let band = point["band"].as_u64().unwrap() as usize;
        let frame = point["frame"].as_u64().unwrap() as usize;
        let expected = point["value"].as_f64().unwrap() as f32;
        assert!(
            (actual[0][band][frame] - expected).abs() < 2e-4,
            "band={band} frame={frame}: {} vs {expected}",
            actual[0][band][frame]
        );
    }
    let two_chunks = candle_transformers::models::voxtral::extract_features(
        &vec![0.; 480001],
        &filters,
        &Device::Cpu,
    )?;
    assert_eq!(two_chunks.dims(), &[2, 128, 3000]);
    assert!(two_chunks
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|v| (*v + 1.5).abs() < 1e-7));
    Ok(())
}
impl AudioUnderstandingModel for EchoUnderstanding {
    fn understand(
        &mut self,
        request: &AudioUnderstandingRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AudioUnderstandingResult> {
        request.validate()?;
        cancel.check()?;
        assert_eq!(request.audio[0].frames(), 4);
        sink.emit(AudioEvent::Text {
            turn: 0,
            text: request.prompt.clone(),
        })?;
        Ok(AudioUnderstandingResult {
            text: request.prompt.clone(),
            finish_reason: FinishReason::Length,
        })
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            tasks: vec![AudioTask::Understanding],
            max_text_bytes: Some(16384),
            ..Default::default()
        }
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn buffered_understanding_preserves_context_prompt_budgets_format_and_completion() -> Result<()> {
    let format = AudioFormat::mono(16000)?;
    let stream = UnderstandingStream::new(
        Box::new(EchoUnderstanding) as Box<dyn AudioUnderstandingModel>,
        format,
        4,
        8,
        0,
    )?;
    assert!(stream.descriptor().output_formats.is_empty());
    let mut session = RealtimeSession::new(stream, 8, 16)?;
    let mut events = Vec::new();
    let mut sink = |event| {
        events.push(event);
        Ok(())
    };
    session.push_text("What is ", &mut sink)?;
    session.push_text("said?", &mut sink)?;
    session.push_audio(&AudioBuffer::new(format, vec![0.; 4])?, &mut sink)?;
    assert!(session
        .push_audio(&AudioBuffer::new(format, vec![0.])?, &mut sink)
        .is_err());
    session.commit(&mut sink)?;
    assert_eq!(session.last_finish_reason(), Some(FinishReason::Length));
    assert!(events
        .iter()
        .any(|event| matches!(event,AudioEvent::Text{text,..} if text=="What is said?")));
    assert!(!events.iter().any(|event| matches!(
        event,
        AudioEvent::Audio { .. } | AudioEvent::AudioChunk { .. }
    )));
    Ok(())
}

#[test]
fn understanding_rejects_empty_audio_prompt_and_unsupported_processor_config() -> Result<()> {
    assert!(AudioUnderstandingRequest::new(vec![], "summarize")
        .validate()
        .is_err());
    let audio = AudioBuffer::new(AudioFormat::mono(16000)?, vec![0.; 320])?;
    assert!(AudioUnderstandingRequest::new(vec![audio.clone()], "")
        .validate()
        .is_err());
    let cancel = CancellationToken::default();
    cancel.cancel();
    assert!(matches!(
        gemma4::extract_features(&audio, &Device::Cpu, &cancel),
        Err(Error::Cancelled)
    ));
    assert!(gemma4::validate_processor(
        &serde_json::json!({"feature_extractor":{"feature_size":80}})
    )
    .is_err());
    Ok(())
}

#[test]
fn voxtral_cpu_bf16_encoder_projector_decoder_and_retained_cache_run_natively() -> candle::Result<()>
{
    use candle_transformers::models::voxtral as v;
    let cfg = v::VoxtralConfig {
        audio_config: v::VoxtralEncoderConfig {
            hidden_size: 4,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 2,
            max_source_positions: 4,
            ..Default::default()
        },
        text_config: v::VoxtralLlamaConfig {
            hidden_size: 4,
            intermediate_size: 8,
            vocab_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(2),
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.,
            max_position_embeddings: 32,
            tie_word_embeddings: false,
        },
        audio_token_id: 6,
        projector_hidden_act: "gelu".into(),
    };
    for dtype in [DType::F32, DType::BF16] {
        let dev = &Device::Cpu;
        let model = v::VoxtralForConditionalGeneration::new(&cfg, VarBuilder::zeros(dtype, dev))?;
        let features = Tensor::zeros((1, 128, 8), DType::F32, dev)?;
        let encoded = model.get_audio_embeds(&features)?;
        assert_eq!(encoded.dims(), &[1, 4]);
        assert_eq!(encoded.dtype(), dtype);
        let mut cache = v::VoxtralCache::new(true, dtype, &cfg.text_config, dev)?;
        let input = Tensor::new(&[[1u32, 6, 3]], dev)?;
        let first = model.forward_with_audio_embeds(&input, &encoded, &mut cache, 0)?;
        assert_eq!(first.dims(), &[1, 8]);
        let snapshot = cache.clone();
        let token = Tensor::new(&[[4u32]], dev)?;
        let next = model.forward(&token, None, &mut cache, 3)?;
        let mut restored = snapshot;
        let replay = model.forward(&token, None, &mut restored, 3)?;
        assert_eq!(next.to_vec2::<f32>()?, replay.to_vec2::<f32>()?);
        cache.reset_audio();
        // A second audio-prefill must retain the language cache and accept the
        // next position rather than mismatching its causal-mask dimensions.
        model.forward_with_audio_embeds(&input, &encoded, &mut cache, 4)?;
        cache.reset();
        let reset = model.forward_with_audio_embeds(&input, &encoded, &mut cache, 0)?;
        assert_eq!(first.to_vec2::<f32>()?, reset.to_vec2::<f32>()?);
    }
    Ok(())
}

#[cfg(feature = "cli")]
fn checkpoint_quality<M: AsrModel + AudioUnderstandingModel>(
    model: &mut M,
    family: &str,
    context: impl Fn(&M) -> usize,
) -> Result<()> {
    fn audio(file: &str) -> AudioBuffer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(file);
        let mut reader = hound::WavReader::open(path).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.sample_rate, spec.channels, spec.bits_per_sample),
            (16000, 1, 16)
        );
        AudioBuffer::new(
            AudioFormat::mono(16000).unwrap(),
            reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.)
                .collect(),
        )
        .unwrap()
    }
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .replace('+', "plus")
            .chars()
            .filter(|c| c.is_alphanumeric())
            .map(|c| match c {
                'á' => 'a',
                'é' => 'e',
                'í' => 'i',
                'ó' => 'o',
                'ú' => 'u',
                c => c,
            })
            .collect::<String>()
            .replace("two", "2")
    }
    let english = audio("question-en.wav");
    let spanish = audio("question-es.wav");
    for (pcm, expected) in [(&english, "2plus2"), (&spanish, "holacomoestas")] {
        let started = std::time::Instant::now();
        let result = model.transcribe_detailed(
            pcm,
            &TranscriptionOptions::default(),
            &CancellationToken::default(),
        )?;
        eprintln!(
            "{family} native ASR {:.3}s: {:?} {:?}",
            started.elapsed().as_secs_f64(),
            result.transcript.text,
            result.finish_reason
        );
        assert!(normalize(&result.transcript.text).contains(expected));
        assert_eq!(result.finish_reason, FinishReason::EndOfSequence);
        assert_eq!(context(model), 0);
    }
    let mut request = AudioUnderstandingRequest::new(
        vec![english],
        "Answer the question spoken in the audio. Give only the answer.",
    );
    request.max_steps = 64;
    let first = model.understand(&request, &CancellationToken::default(), &mut |_| Ok(()))?;
    eprintln!(
        "{family} first audio response: {:?} {:?}",
        first.text, first.finish_reason
    );
    assert!(normalize(&first.text).contains("four") || normalize(&first.text).contains('4'));
    assert_eq!(first.finish_reason, FinishReason::EndOfSequence);
    AudioUnderstandingModel::end_turn(model)?;
    let committed = context(model);
    assert!(committed > 0);
    let mut followup=AudioUnderstandingRequest::new(vec![spanish],"Ignore the greeting in this clip. What was the numeric answer you gave to the previous audio question? Give only the answer.");
    followup.max_steps = 64;
    let cancel = CancellationToken::default();
    cancel.cancel();
    assert!(matches!(
        model.understand(&followup, &cancel, &mut |_| Ok(())),
        Err(Error::Cancelled)
    ));
    assert_eq!(context(model), committed);
    let second = model.understand(&followup, &CancellationToken::default(), &mut |_| Ok(()))?;
    eprintln!(
        "{family} retained-context response: {:?} {:?}",
        second.text, second.finish_reason
    );
    assert!(normalize(&second.text).contains("four") || normalize(&second.text).contains('4'));
    assert_eq!(second.finish_reason, FinishReason::EndOfSequence);
    assert!(context(model) > committed);
    AudioUnderstandingModel::cancel_turn(model)?;
    assert_eq!(context(model), committed);
    AudioUnderstandingModel::reset(model)?;
    assert_eq!(context(model), 0);
    Ok(())
}

#[test]
#[cfg(feature = "cli")]
#[ignore = "requires pinned public Voxtral Mini checkpoint (~9.4GB); run serially with CANDLE_AUDIO_VOXTRAL"]
fn voxtral_native_bf16_recognition_and_audio_understanding() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_VOXTRAL").expect("set CANDLE_AUDIO_VOXTRAL"),
    );
    let mut model = candle_audio::models::voxtral::Voxtral::load(
        &directory,
        &Device::Cpu,
        &serde_json::json!({"dtype":"bf16","max_context":2048,"max_steps":64}),
        AudioFormat::mono(16000)?,
    )?;
    checkpoint_quality(&mut model, "Voxtral", |m| m.context_tokens())
}

#[test]
#[cfg(feature = "cli")]
#[ignore = "requires pinned public Gemma4 E2B checkpoint (~10.3GB); run serially with CANDLE_AUDIO_GEMMA4"]
fn gemma4_native_bf16_recognition_and_audio_understanding() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_GEMMA4").expect("set CANDLE_AUDIO_GEMMA4"),
    );
    let mut model = gemma4::Gemma4::load(
        &directory,
        &Device::Cpu,
        &serde_json::json!({"dtype":"bf16","max_context":2048,"max_steps":64}),
        AudioFormat::mono(16000)?,
    )?;
    checkpoint_quality(&mut model, "Gemma4", |m| m.context_tokens())
}
