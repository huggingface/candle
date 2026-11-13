#![cfg(feature = "cli")]
//! Explicit native-checkpoint task tests; no downloads, external inference or silent skips.
use candle::Device;
use candle_audio::{models::lfm3::Lfm3, *};
use std::{path::PathBuf, time::Instant};
fn wav(path: &str) -> AudioBuffer {
    let mut r = hound::WavReader::open(path).unwrap();
    let s = r.spec();
    assert_eq!(
        (s.channels, s.sample_rate, s.bits_per_sample),
        (1, 16000, 16)
    );
    AudioBuffer::new(
        AudioFormat::mono(16000).unwrap(),
        r.samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.)
            .collect(),
    )
    .unwrap()
}
fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .replace("two", "2")
}
#[test]
#[ignore = "requires native LFM3 GGUF/tokenizer directory and English+Spanish WAV fixtures"]
fn lfm3_native_recognition_synthesis_and_conversation() {
    let dir = PathBuf::from(std::env::var("CANDLE_AUDIO_LFM3").expect("set CANDLE_AUDIO_LFM3"));
    let english = wav(
        &std::env::var("CANDLE_AUDIO_LFM3_EN_WAV").unwrap_or_else(|_| {
            format!(
                "{}/tests/fixtures/question-en.wav",
                env!("CARGO_MANIFEST_DIR")
            )
        }),
    );
    let spanish = wav(
        &std::env::var("CANDLE_AUDIO_LFM3_ES_WAV").unwrap_or_else(|_| {
            format!(
                "{}/tests/fixtures/question-es.wav",
                env!("CARGO_MANIFEST_DIR")
            )
        }),
    );
    let device = match std::env::var("CANDLE_AUDIO_TEST_DEVICE").as_deref() {
        Ok("vulkan") => Device::new_vulkan(0).unwrap(),
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(other) => panic!("unsupported test device {other}"),
    };
    let start = Instant::now();
    let model = Lfm3::load(
        &dir,
        &device,
        AudioFormat::mono(16000).unwrap(),
        &serde_json::json!({"max_steps":256}),
    )
    .unwrap();
    eprintln!(
        "Native LFM3 loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let mut asr = model.fork(AudioFormat::mono(16000).unwrap()).unwrap();
    let en = asr
        .transcribe_detailed(
            &english,
            &TranscriptionOptions {
                language: Some("en".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    eprintln!(
        "Native English ASR: {:?} {:?}",
        en.transcript.text, en.finish_reason
    );
    let es = asr
        .transcribe_detailed(
            &spanish,
            &TranscriptionOptions {
                language: Some("es".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    eprintln!(
        "Native Spanish ASR: {:?} {:?}",
        es.transcript.text, es.finish_reason
    );
    let mut tts = model.fork(AudioFormat::mono(16000).unwrap()).unwrap();
    let mut pcm = vec![];
    let synthesis_start = Instant::now();
    let mut first_audio = None;
    let request = SynthesisRequest {
        text: "Hola, ¿cómo estás?".into(),
        language: Some("es".into()),
        voice: Some("female".into()),
        max_steps: 128,
        ..SynthesisRequest::new("")
    };
    tts.synthesize(&request, &CancellationToken::default(), &mut |event| {
        if let AudioEvent::Audio { audio, .. } = event {
            assert_eq!(audio.format().sample_rate(), 24000);
            first_audio.get_or_insert_with(|| synthesis_start.elapsed());
            pcm.extend_from_slice(audio.samples());
        }
        Ok(())
    })
    .unwrap();
    eprintln!(
        "Native Spanish TTS: {} PCM frames; first audio {:.3}s; total {:.3}s",
        pcm.len(),
        first_audio.unwrap().as_secs_f64(),
        synthesis_start.elapsed().as_secs_f64()
    );
    if let Ok(directory) = std::env::var("CANDLE_AUDIO_TEST_OUTPUT_DIR") {
        std::fs::create_dir_all(&directory).unwrap();
        let path = PathBuf::from(directory).join("lfm3-native-spanish.wav");
        let mut wav = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for &sample in &pcm {
            wav.write_sample((sample.clamp(-1., 1.) * 32767.).round() as i16)
                .unwrap();
        }
        wav.finalize().unwrap();
    }
    assert!(pcm.len() > 2400);
    assert!(pcm.iter().all(|s| s.is_finite()));
    let output = AudioConverter::convert(
        &AudioBuffer::new(AudioFormat::mono(24000).unwrap(), pcm).unwrap(),
        16000,
    )
    .unwrap();
    let roundtrip = asr
        .transcribe(
            &output,
            &TranscriptionOptions {
                language: Some("es".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    eprintln!("Native Spanish TTS roundtrip: {:?}", roundtrip.text);
    let mut realtime = model.fork(AudioFormat::mono(16000).unwrap()).unwrap();
    let cancel = CancellationToken::default();
    let mut text = String::new();
    let mut frames = 0;
    let mut sink = |event| {
        match event {
            AudioEvent::Text { text: delta, .. } => text.push_str(&delta),
            AudioEvent::Audio { audio, .. } => frames += audio.frames(),
            _ => {}
        }
        Ok(())
    };
    for chunk in english.samples().chunks(1600) {
        realtime
            .push_audio(
                &AudioBuffer::new(english.format(), chunk.to_vec()).unwrap(),
                &cancel,
                &mut sink,
            )
            .unwrap();
    }
    realtime.commit(&cancel, &mut sink).unwrap();
    let before = realtime.context_tokens();
    realtime.end_turn().unwrap();
    assert_eq!(realtime.context_tokens(), before);
    realtime
        .push_text("Repeat your previous answer.", &cancel, &mut |_| Ok(()))
        .unwrap();
    assert!(realtime.context_tokens() > before);
    realtime.cancel_turn().unwrap();
    assert_eq!(realtime.context_tokens(), before);
    // Cancel after an actual codec chunk, not just before generation starts.
    let active = CancellationToken::default();
    realtime
        .push_text(
            "Tell me a long story about the ocean.",
            &active,
            &mut |_| Ok(()),
        )
        .unwrap();
    let mut cancelled_frames = 0;
    let result = realtime.commit(&active, &mut |event| {
        if let AudioEvent::Audio { audio, .. } = event {
            cancelled_frames += audio.frames();
            active.cancel();
        }
        Ok(())
    });
    assert!(
        matches!(result, Err(Error::Cancelled)),
        "active generation must observe cancellation"
    );
    assert!(
        cancelled_frames > 0,
        "must cancel after real vocoder output"
    );
    realtime.cancel_turn().unwrap();
    assert_eq!(
        realtime.context_tokens(),
        before,
        "cancelled generated audio must roll context back"
    );
    let recovery = CancellationToken::default();
    realtime
        .push_text("Hello.", &recovery, &mut |_| Ok(()))
        .unwrap();
    realtime.cancel_turn().unwrap();
    assert_eq!(realtime.context_tokens(), before);
    eprintln!("Active native generation cancellation after {cancelled_frames} audio frames; next-input recovery passed");
    realtime.reset_conversation().unwrap();
    assert_eq!(realtime.context_tokens(), 0);
    eprintln!(
        "Native realtime: text={text:?}, audio={frames}frames; history/rollback/reset passed"
    );
    assert!(
        normalize(&en.transcript.text).contains("whatis2plus2"),
        "English recognition failed: {}",
        en.transcript.text
    );
    assert_eq!(
        normalize(&es.transcript.text),
        normalize("Hola, ¿cómo estás?")
    );
    assert_eq!(normalize(&roundtrip.text), normalize("Hola, ¿cómo estás?"));
    assert!(
        text.to_lowercase().contains("four"),
        "Realtime answer: {text}"
    );
    assert!(frames > 2400);
}
