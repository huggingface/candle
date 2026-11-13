use candle_audio::*;
#[test]
#[ignore = "requires only the EnCodec prefix of pinned MusicGen-small weights; CPU only"]
fn native_musicgen_codec_blocks_match_full_decode_and_cancel() -> candle::Result<()> {
    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::{encodec, musicgen};
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_MUSICGEN_MODEL").expect("set CANDLE_AUDIO_MUSICGEN_MODEL"),
    );
    let cfg: musicgen::GenConfig =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[directory.join("model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )?
    };
    // Do not instantiate/map the T5 or language decoder into tensor storage.
    let codec = encodec::Model::new(&cfg.encodec, vb.pp("audio_encoder"))?;
    let frames = 75;
    let books = cfg.encodec.num_quantizers();
    let codes = Tensor::from_vec(
        (0..frames * books)
            .map(|i| ((i * 17 + i / 11) % cfg.encodec.codebook_size) as u32)
            .collect::<Vec<_>>(),
        (1, books, frames),
        &Device::Cpu,
    )?;
    let full = codec.decode(&codes)?;
    let difference =
        |a: &Tensor, b: &Tensor| (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>();
    for chunk in [16, 32] {
        let bounded = codec.decode_chunked_with_callback(&codes, chunk, || Ok(()))?;
        assert_eq!(bounded.dims(), full.dims());
        let error = difference(&bounded, &full)?;
        eprintln!("MusicGen real EnCodec frames={frames} chunk={chunk} max_error={error}");
        assert!(error < 3e-5);
    }
    let mut calls = 0;
    let cancelled = codec.decode_with_callback(&codes, || {
        calls += 1;
        if calls == 20 {
            candle::bail!("cancel real EnCodec CNN");
        }
        Ok(())
    });
    assert!(cancelled
        .unwrap_err()
        .to_string()
        .contains("cancel real EnCodec CNN"));
    assert_eq!(calls, 20);
    assert!(difference(&codec.decode_with_callback(&codes, || Ok(()))?, &full)? < 3e-5);
    Ok(())
}
#[test]
fn audio_generation_request_validates_sampling_and_budgets() {
    let mut request = AudioGenerationRequest::new("Warm piano and soft drums");
    assert_eq!(request.max_steps, 250);
    assert!(request.validate().is_ok());
    for scale in [0., f64::NAN, f64::INFINITY] {
        request.guidance_scale = scale;
        assert!(request.validate().is_err());
    }
    request.guidance_scale = 3.;
    request.max_steps = 0;
    assert!(request.validate().is_err());
    request.max_steps = 16;
    request.top_k = Some(0);
    assert!(request.validate().is_err());
    request.top_k = Some(10);
    request.top_p = Some(0.);
    assert!(request.validate().is_err());
    request.top_p = Some(0.9);
    request.temperature = Some(f64::NAN);
    assert!(request.validate().is_err());
    request.temperature = None;
    assert!(request.validate().is_ok());
}
#[test]
#[ignore = "requires pinned official MusicGen-small checkpoint; CPU only"]
fn native_musicgen_text_conditioning_reset_and_audio_generation() {
    use candle::Device;
    use candle_audio::models::musicgen::MusicGen;
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_MUSICGEN_MODEL").expect("set CANDLE_AUDIO_MUSICGEN_MODEL"),
    );
    let started = std::time::Instant::now();
    let mut model = MusicGen::load(
        &directory,
        &Device::Cpu,
        &serde_json::json!({"max_steps":32}),
    )
    .unwrap();
    eprintln!("MusicGen loaded in {:.3}s", started.elapsed().as_secs_f64());
    assert_eq!(model.frame_rate(), 50.);
    assert_eq!(model.descriptor().tasks, vec![AudioTask::AudioGeneration]);
    assert!(model.descriptor().accepts_audio_reference);
    let mut request = AudioGenerationRequest::new("Solo piano playing gentle notes");
    request.max_steps = 16;
    request.temperature = None;
    request.guidance_scale = 1.;
    let cancel = CancellationToken::default();
    let mut output = Vec::new();
    model
        .generate(&request, &cancel, &mut |pcm| {
            output.push(pcm);
            Ok(())
        })
        .unwrap();
    assert_eq!(output.len(), 1);
    let piano = output.pop().unwrap();
    assert_eq!(piano.format(), AudioFormat::mono(32000).unwrap());
    assert_eq!(piano.frames(), 16 * 640);
    assert!(piano.samples().iter().all(|v| v.is_finite()));
    let rms = (piano
        .samples()
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        / piano.frames() as f64)
        .sqrt();
    assert!(rms > 1e-5, "audio RMS={rms}");
    eprintln!(
        "MusicGen first 0.32s PCM callback at {:.3}s, RMS {rms:.6}",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(model.finish_reason(), FinishReason::Length);
    model.reset().unwrap();
    model
        .generate(&request, &cancel, &mut |pcm| {
            output.push(pcm);
            Ok(())
        })
        .unwrap();
    let repeated = output.pop().unwrap();
    let difference = piano
        .samples()
        .iter()
        .zip(repeated.samples())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(difference < 1e-5, "reset waveform difference={difference}");
    request.prompt = "Loud distorted electric guitar with heavy rock drums".into();
    model
        .generate(&request, &cancel, &mut |pcm| {
            output.push(pcm);
            Ok(())
        })
        .unwrap();
    let guitar = output.pop().unwrap();
    assert!(
        piano
            .samples()
            .iter()
            .zip(guitar.samples())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "different prompts produced identical music"
    );
    let token = CancellationToken::default();
    token.cancel();
    assert!(matches!(
        model.generate(&request, &token, &mut |_| panic!("cancelled output")),
        Err(Error::Cancelled)
    ));
    assert_eq!(model.finish_reason(), FinishReason::Cancelled);
    request.audio_reference =
        Some(AudioBuffer::new(AudioFormat::mono(16000).unwrap(), vec![0.; 640]).unwrap());
    assert!(matches!(
        model.generate(&request, &cancel, &mut |_| panic!(
            "invalid reference output"
        )),
        Err(Error::InvalidInput(_))
    ));
    eprintln!("MusicGen native checkpoint: 16 codec frames / 10240 PCM frames per generation, RMS {rms:.6}, prompt dependence/reset/cancel validated in {:.3}s",started.elapsed().as_secs_f64());
}

#[test]
#[ignore = "requires pinned official MusicGen-small checkpoint; CPU only"]
fn native_musicgen_audio_prompt_continuation_reset_and_budget() {
    use candle::Device;
    use candle_audio::models::musicgen::MusicGen;
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_MUSICGEN_MODEL").expect("set CANDLE_AUDIO_MUSICGEN_MODEL"),
    );
    let started = std::time::Instant::now();
    let mut model = MusicGen::load(
        &directory,
        &Device::Cpu,
        &serde_json::json!({"max_steps":16}),
    )
    .unwrap();
    assert!(model.descriptor().accepts_audio_reference);
    assert_eq!(
        model.descriptor().input_formats,
        vec![AudioFormat::mono(32000).unwrap()]
    );
    let format = AudioFormat::mono(32000).unwrap();
    let reference = |frequency: f32| {
        AudioBuffer::new(
            format,
            (0..2553)
                .map(|i| (i as f32 * frequency * std::f32::consts::TAU / 32000.).sin() * 0.12)
                .collect(),
        )
        .unwrap()
    };
    let mut request = AudioGenerationRequest::new("Soft sustained piano chords");
    request.max_steps = 4;
    request.temperature = None;
    request.guidance_scale = 1.;
    request.audio_reference = Some(reference(220.));
    let cancel = CancellationToken::default();
    let mut outputs = Vec::new();
    let mut excessive = request.clone();
    excessive.max_steps = 13;
    assert!(matches!(
        model.generate(&excessive, &cancel, &mut |_| panic!("over-budget output")),
        Err(Error::LimitExceeded(_))
    ));
    let mut empty = request.clone();
    empty.audio_reference = Some(AudioBuffer::new(format, vec![]).unwrap());
    assert!(matches!(
        model.generate(&empty, &cancel, &mut |_| panic!("empty reference output")),
        Err(Error::InvalidInput(_))
    ));
    model
        .generate(&request, &cancel, &mut |audio| {
            outputs.push(audio);
            Ok(())
        })
        .unwrap();
    let first = outputs.pop().unwrap();
    assert_eq!(first.format(), format);
    assert_eq!(first.frames(), 8 * 640);
    assert!(first.samples().iter().all(|v| v.is_finite()));
    assert!(
        first.samples()[..2553]
            .iter()
            .zip(request.audio_reference.as_ref().unwrap().samples())
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "prompt output must be a codec reconstruction"
    );
    model.reset().unwrap();
    model
        .generate(&request, &cancel, &mut |audio| {
            outputs.push(audio);
            Ok(())
        })
        .unwrap();
    let repeat = outputs.pop().unwrap();
    let difference = first
        .samples()
        .iter()
        .zip(repeat.samples())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        difference < 1e-5,
        "reset continuation waveform difference={difference}"
    );
    request.audio_reference = Some(reference(660.));
    model
        .generate(&request, &cancel, &mut |audio| {
            outputs.push(audio);
            Ok(())
        })
        .unwrap();
    let changed = outputs.pop().unwrap();
    assert!(
        first.samples()[4 * 640..]
            .iter()
            .zip(&changed.samples()[4 * 640..])
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "audio prompt must affect continuation"
    );
    let rms = (first
        .samples()
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        / first.frames() as f64)
        .sqrt();
    assert!(rms > 1e-5);
    eprintln!("MusicGen continuation: 2553 reference frames ->4 prefix code frames +4 new ->5120 decoded PCM frames; RMS {rms:.6}, reset diff {difference:.9}, {:.3}s",started.elapsed().as_secs_f64());
}
