//! Explicit native-checkpoint smoke/contract tests. These never download weights.
//! Missing checkpoints fail when an ignored test is selected; audio quality still
//! needs a separate listening/transcription evaluation on the pinned checkpoint.
use candle::Device;
use candle_audio::{
    models::{csm::Csm, metavoice::MetaVoice},
    *,
};
use std::path::PathBuf;

fn device() -> Device {
    match std::env::var("CANDLE_AUDIO_TEST_DEVICE").as_deref() {
        Ok("vulkan") => Device::new_vulkan(0).expect("Vulkan device"),
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(other) => panic!("unsupported test device {other}"),
    }
}
fn checkpoint(variable: &str) -> PathBuf {
    let directory = PathBuf::from(
        std::env::var(variable)
            .unwrap_or_else(|_| panic!("set {variable} to the local checkpoint directory")),
    );
    assert!(
        directory.is_dir(),
        "checkpoint directory does not exist: {}",
        directory.display()
    );
    directory
}
fn collect(event: AudioEvent, pcm: &mut Vec<f32>) -> Result<()> {
    if let AudioEvent::Audio { audio, .. } = event {
        assert_eq!(audio.format(), AudioFormat::mono(24000)?);
        pcm.extend_from_slice(audio.samples());
    }
    Ok(())
}
fn assert_audio(pcm: &[f32]) {
    assert!(
        pcm.len() > 2400,
        "model produced too little speech: {} frames",
        pcm.len()
    );
    assert!(pcm.iter().all(|sample| sample.is_finite()));
    let rms = (pcm
        .iter()
        .map(|sample| (*sample as f64).powi(2))
        .sum::<f64>()
        / pcm.len() as f64)
        .sqrt();
    assert!(rms > 1e-5, "model produced silence (RMS={rms})");
}

#[cfg(feature = "cli")]
fn reference_audio() -> Result<AudioBuffer> {
    let mut wav = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/question-en.wav"),
    )
    .map_err(|error| Error::InvalidInput(error.to_string()))?;
    let spec = wav.spec();
    assert_eq!(
        (spec.bits_per_sample, spec.channels, spec.sample_rate),
        (16, 1, 16000)
    );
    AudioBuffer::new(
        AudioFormat::mono(16000)?,
        wav.samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.)
            .collect(),
    )
}

#[test]
#[cfg(feature = "cli")]
#[ignore = "requires CANDLE_AUDIO_METAVOICE_GGUF directory containing public speaker_encoder.safetensors"]
fn metavoice_native_speaker_embedding_parity() -> Result<()> {
    use candle_transformers::models::metavoice::speaker_encoder;
    let directory = checkpoint("CANDLE_AUDIO_METAVOICE_GGUF");
    let device = device();
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[directory.join("speaker_encoder.safetensors")],
            candle::DType::F32,
            &device,
        )?
    };
    let config = speaker_encoder::Config::cfg();
    let filters = speaker_encoder::mel_filters(&config)?;
    let model = speaker_encoder::Model::new(config, vb)?;
    let audio = reference_audio()?;
    let actual = model
        .embed_utterance(audio.samples(), &filters, 1.3, 0.75, &device)?
        .to_vec1::<f32>()?;
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../candle-transformers/tests/fixtures/metavoice_speaker_encoder.json"
    ))?;
    let expected: Vec<f32> = serde_json::from_value(fixture["embedding"].clone())?;
    assert_eq!(actual.len(), 256);
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0f32, f32::max);
    let cosine = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| a * e)
        .sum::<f32>();
    eprintln!("Native MetaVoice speaker embedding vs independent Librosa/PyTorch: max_abs={max}, cosine={cosine}");
    assert!(
        max < 3e-5,
        "speaker encoder embedding differs from reference: max_abs={max}"
    );
    assert!(cosine > 0.99999);
    let mut checks = 0;
    let cancelled =
        model.embed_utterance_with_callback(audio.samples(), &filters, 1.3, 0.75, &device, || {
            checks += 1;
            if checks == 300 {
                return Err(candle::Error::Msg(
                    "cancelled native speaker fixture".into(),
                ));
            }
            Ok(())
        });
    assert!(cancelled.is_err());
    assert_eq!(checks, 300);
    Ok(())
}

#[test]
#[cfg(feature = "cli")]
#[ignore = "requires CANDLE_AUDIO_METAVOICE_GGUF including native speaker encoder and EnCodec weights"]
fn metavoice_native_audio_reference_gguf() -> Result<()> {
    let mut model = MetaVoice::load(
        &checkpoint("CANDLE_AUDIO_METAVOICE_GGUF"),
        &device(),
        &serde_json::json!({"quantized":true}),
    )?;
    let deadline = CancellationToken::with_timeout(std::time::Duration::from_millis(25));
    let mut cancelled_request = SynthesisRequest::new("What is two plus two?");
    cancelled_request.max_steps = 512;
    assert!(matches!(
        model.synthesize(&cancelled_request, &deadline, &mut |_| panic!(
            "deadline response emitted audio"
        )),
        Err(Error::DeadlineExceeded)
    ));
    let references = vec![AudioReference {
        audio: reference_audio()?,
        transcript: None,
        speaker_id: None,
    }];
    let embedding = model.reference_embedding(&references, &CancellationToken::default())?;
    assert_eq!(embedding.len(), 256);
    assert!((embedding.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5);
    let mut request =
        ConditionedSynthesisRequest::new(SynthesisRequest::new("What is two plus two?"));
    request.request.max_steps = 512;
    request.references = references;
    let mut pcm = Vec::new();
    model.synthesize_conditioned(&request, &CancellationToken::default(), &mut |event| {
        collect(event, &mut pcm)
    })?;
    assert_audio(&pcm);
    assert!(matches!(
        model.finish_reason(),
        FinishReason::EndOfSequence | FinishReason::Length
    ));
    artifact(
        "metavoice-reference-gguf",
        &request.request,
        model.finish_reason(),
        &pcm,
    )?;
    let generated = AudioReference {
        audio: AudioBuffer::new(AudioFormat::mono(24000)?, pcm)?,
        transcript: None,
        speaker_id: None,
    };
    let generated_embedding =
        model.reference_embedding(&[generated], &CancellationToken::default())?;
    let cosine = embedding
        .iter()
        .zip(&generated_embedding)
        .map(|(a, b)| a * b)
        .sum::<f32>();
    eprintln!("Native MetaVoice reference/generated voice embedding cosine={cosine} (reported metric, no subjective quality threshold)");
    if let Ok(directory) = std::env::var("CANDLE_AUDIO_TEST_OUTPUT_DIR") {
        std::fs::write(
            PathBuf::from(directory).join("metavoice-reference-embedding.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"reference":embedding,"generated":generated_embedding,"cosine":cosine}),
            )?,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cli")]
fn artifact(
    name: &str,
    request: &SynthesisRequest,
    finish: FinishReason,
    pcm: &[f32],
) -> Result<()> {
    let Ok(directory) = std::env::var("CANDLE_AUDIO_TEST_OUTPUT_DIR") else {
        return Ok(());
    };
    let directory = PathBuf::from(directory);
    std::fs::create_dir_all(&directory)?;
    let file = directory.join(format!("{name}.wav"));
    let mut wav = hound::WavWriter::create(
        &file,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .map_err(|error| Error::InvalidInput(error.to_string()))?;
    for sample in pcm {
        wav.write_sample((sample.clamp(-1., 1.) * 32767.).round() as i16)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
    }
    wav.finalize()
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
    let metadata = serde_json::json!({"native_candle":true,"request":request,"finish_reason":finish,"sample_rate":24000,"channels":1,"frames":pcm.len(),"duration_seconds":pcm.len() as f64/24000.,"rms":(pcm.iter().map(|v|(*v as f64).powi(2)).sum::<f64>()/pcm.len() as f64).sqrt(),"checkpoint_manifest":"candle-audio/tests/fixtures/csm-metavoice-checkpoints.json"});
    std::fs::write(
        directory.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    eprintln!(
        "Native audio artifact: {} ({finish:?}, {} frames)",
        file.display(),
        pcm.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CANDLE_AUDIO_CSM local CSM 1B + Llama tokenizer + Mimi32 checkpoint"]
fn csm_native_streaming_context_reset_and_cancellation() -> Result<()> {
    let mut model = Csm::load(
        &checkpoint("CANDLE_AUDIO_CSM"),
        &device(),
        &serde_json::json!({}),
    )?;
    let mut request = SynthesisRequest::new("Hello, how are you today?");
    request.max_steps = 64;
    let conditioned = ConditionedSynthesisRequest::new(request);
    let cancel = CancellationToken::default();
    let mut pcm = Vec::new();
    model.start_conditioned(&conditioned, &cancel)?;
    assert!(model.output_pending());
    while model.output_pending() {
        RealtimeModel::poll_output(&mut model, &cancel, &mut |event| collect(event, &mut pcm))?;
    }
    assert_audio(&pcm);
    assert!(matches!(
        RealtimeModel::finish_reason(&model),
        FinishReason::Length | FinishReason::EndOfSequence
    ));
    let committed = model.context_positions();
    assert!(committed > 0);
    model.end_turn()?;
    assert_eq!(
        model.context_positions(),
        committed,
        "end_turn must retain CSM history"
    );
    model.start_conditioned(&conditioned, &cancel)?;
    RealtimeModel::poll_output(&mut model, &cancel, &mut |_| Ok(()))?;
    assert!(model.context_positions() > committed);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        RealtimeModel::poll_output(&mut model, &cancelled, &mut |_| Ok(())),
        Err(Error::Cancelled)
    ));
    assert_eq!(
        model.context_positions(),
        committed,
        "cancelled response must roll back its KV cache"
    );
    assert!(!model.output_pending());
    let mut next = SynthesisRequest::new("It is a lovely day.");
    next.max_steps = 32;
    model.synthesize(&next, &cancel, &mut |_| Ok(()))?;
    assert!(model.context_positions() > committed);
    model.reset_conversation()?;
    assert_eq!(model.context_positions(), 0);
    Ok(())
}

#[test]
#[cfg(feature = "cli")]
#[ignore = "requires CANDLE_AUDIO_CSM and the versioned transcribed English WAV fixture"]
fn csm_native_transcribed_speaker_reference() -> Result<()> {
    let mut model = Csm::load(
        &checkpoint("CANDLE_AUDIO_CSM"),
        &device(),
        &serde_json::json!({}),
    )?;
    let mut wav = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/question-en.wav"),
    )
    .map_err(|error| Error::InvalidInput(error.to_string()))?;
    let spec = wav.spec();
    assert_eq!((spec.bits_per_sample, spec.channels), (16, 1));
    let audio = AudioBuffer::new(
        AudioFormat::mono(spec.sample_rate)?,
        wav.samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.)
            .collect(),
    )?;
    let mut request =
        ConditionedSynthesisRequest::new(SynthesisRequest::new("The answer is four."));
    request.request.max_steps = 64;
    request.speaker_id = Some(1);
    request.references.push(AudioReference {
        audio,
        transcript: Some("What is two plus two?".into()),
        speaker_id: Some(1),
    });
    let mut pcm = Vec::new();
    model.synthesize_conditioned(&request, &CancellationToken::default(), &mut |event| {
        collect(event, &mut pcm)
    })?;
    assert_audio(&pcm);
    Ok(())
}

fn metavoice(quantized: bool, variable: &str) -> Result<()> {
    let mut model = MetaVoice::load(
        &checkpoint(variable),
        &device(),
        &serde_json::json!({"quantized":quantized}),
    )?;
    let mut request = SynthesisRequest::new("What is two plus two?");
    request.max_steps = 512;
    let mut pcm = Vec::new();
    model.synthesize(&request, &CancellationToken::default(), &mut |event| {
        collect(event, &mut pcm)
    })?;
    assert_audio(&pcm);
    #[cfg(feature = "cli")]
    artifact(
        if quantized {
            "metavoice-gguf"
        } else {
            "metavoice-safetensors"
        },
        &request,
        model.finish_reason(),
        &pcm,
    )?;
    assert!(matches!(
        model.finish_reason(),
        FinishReason::Length | FinishReason::EndOfSequence
    ));
    assert_eq!(model.streaming_mode(), StreamingMode::Buffered);
    let cancel = CancellationToken::default();
    cancel.cancel();
    assert!(matches!(
        model.synthesize(&request, &cancel, &mut |_| panic!(
            "cancelled synthesis emitted audio"
        )),
        Err(Error::Cancelled)
    ));
    model.reset()?;
    let mut conditioned = ConditionedSynthesisRequest::new(request);
    conditioned.speaker_embedding = Some(vec![0.; 255]);
    assert!(matches!(
        model.synthesize_conditioned(&conditioned, &CancellationToken::default(), &mut |_| Ok(())),
        Err(Error::InvalidInput(_))
    ));
    Ok(())
}

#[test]
#[ignore = "requires CANDLE_AUDIO_METAVOICE local v0.1 safetensors, speaker embedding and EnCodec"]
fn metavoice_native_two_stage_safetensors() -> Result<()> {
    metavoice(false, "CANDLE_AUDIO_METAVOICE")
}

#[test]
#[ignore = "requires CANDLE_AUDIO_METAVOICE_GGUF local GGUF first stage, safetensors refinement, speaker and EnCodec"]
fn metavoice_native_two_stage_gguf() -> Result<()> {
    metavoice(true, "CANDLE_AUDIO_METAVOICE_GGUF")
}
