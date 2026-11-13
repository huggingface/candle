#![cfg(feature = "cli")]
//! Real inference checks. No downloads and no silent fixture/device skips.
use candle::Device;
use candle_audio::{models::load, *};
use std::{path::PathBuf, time::Instant};

fn fixtures() -> (PathBuf, AudioBuffer, Device) {
    static LOGS: std::sync::Once = std::sync::Once::new();
    LOGS.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "warn".into()),
            )
            .with_writer(std::io::stderr)
            .try_init();
    });

    let root = PathBuf::from(
        std::env::var("CANDLE_AUDIO_TEST_MODELS").expect("set CANDLE_AUDIO_TEST_MODELS"),
    );
    let path = std::env::var("CANDLE_AUDIO_TEST_WAV")
        .expect("set CANDLE_AUDIO_TEST_WAV to whisper.cpp's JFK WAV");
    let mut wav = hound::WavReader::open(path).unwrap();
    let spec = wav.spec();
    let samples = wav
        .samples::<i16>()
        .map(|v| v.unwrap() as f32 / 32768.)
        .collect();
    let audio = AudioBuffer::new(
        AudioFormat::new(spec.sample_rate, spec.channels).unwrap(),
        samples,
    )
    .unwrap();
    let device = match std::env::var("CANDLE_AUDIO_TEST_DEVICE").as_deref() {
        Ok("vulkan") => Device::new_vulkan(0).expect("Vulkan device must initialize"),
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(other) => panic!("unsupported test device {other}"),
    };
    (root, audio, device)
}
fn wer(reference: &str, hypothesis: &str) -> f64 {
    let words = |s: &str| {
        s.split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            })
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
    };
    let (a, b) = (words(reference), words(hypothesis));
    let mut row: Vec<_> = (0..=b.len()).collect();
    for (i, word) in a.iter().enumerate() {
        let mut next = vec![i + 1];
        for (j, other) in b.iter().enumerate() {
            next.push(
                (row[j + 1] + 1)
                    .min(next[j] + 1)
                    .min(row[j] + usize::from(word != other)),
            );
        }
        row = next;
    }
    row[b.len()] as f64 / a.len() as f64
}

#[test]
#[ignore = "requires Whisper checkpoint and JFK WAV"]
fn whisper_offline_and_chunked_transcription_agree() {
    let (root, audio, device) = fixtures();
    let mut model = load::asr(&root.join("whisper-tiny"), &device).unwrap();
    let mono = AudioConverter::convert(&audio, model.input_format().sample_rate()).unwrap();
    let start = Instant::now();
    let offline = model
        .transcribe(
            &mono,
            &TranscriptionOptions {
                language: Some("en".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    let reference="And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
    let error = wer(reference, &offline.text);
    eprintln!(
        "Whisper {device:?}: {:.3}s, WER={error:.3}, text={:?}",
        start.elapsed().as_secs_f64(),
        offline.text
    );
    assert!(error <= 0.15, "JFK WER too high: {error}: {}", offline.text);
    let window = model.max_input_frames();
    let stream = AsrStream::new(
        model,
        audio.format(),
        window,
        TranscriptionOptions {
            language: Some("en".into()),
            translate: false,
        },
    )
    .unwrap();
    let mut session =
        RealtimeSession::new(stream, audio.format().sample_rate() as usize, 1_000_000).unwrap();
    let mut text = String::new();
    let mut sink = |e| {
        if let AudioEvent::Transcript { transcript, .. } = e {
            text.push_str(&transcript.text);
        }
        Ok(())
    };
    let chunk = 137 * audio.format().channels() as usize;
    for samples in audio.samples().chunks(chunk) {
        session
            .push_audio(
                &AudioBuffer::new(audio.format(), samples.to_vec()).unwrap(),
                &mut sink,
            )
            .unwrap();
    }
    session.finish(&mut sink).unwrap();
    assert_eq!(text, offline.text);
}

#[test]
#[ignore = "requires quantized Whisper checkpoint and JFK WAV"]
fn quantized_whisper_uses_the_same_asr_interface() {
    let (root, audio, device) = fixtures();
    let mut model = load::asr(&root.join("whisper-tiny-q80"), &device).unwrap();
    let audio = AudioConverter::convert(&audio, model.input_format().sample_rate()).unwrap();
    let result = model
        .transcribe(
            &audio,
            &TranscriptionOptions {
                language: Some("en".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    let reference = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
    let error = wer(reference, &result.text);
    eprintln!(
        "Quantized Whisper {device:?}: WER={error:.3}, text={:?}",
        result.text
    );
    assert!(error <= 0.15, "quantized JFK WER too high: {error}");
}

#[test]
#[ignore = "requires Parler checkpoint"]
fn parler_synthesizes_finite_non_silent_audio_and_accepts_cancellation() {
    let (root, _, device) = fixtures();
    let mut request = SynthesisRequest::new("What is two plus two?");
    request.max_steps = 256;
    let backend = std::env::var("CANDLE_AUDIO_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    let output_dir = std::env::var_os("CANDLE_AUDIO_TEST_OUTPUT_DIR").map(PathBuf::from);
    let path = output_dir
        .as_ref()
        .unwrap_or(&std::env::temp_dir())
        .join(format!(
            "candle-audio-parler-{backend}-{}.wav",
            std::process::id()
        ));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let start = Instant::now();
    // Exercise the public CLI, local model loader, Rust TtsModel implementation,
    // event stream and WAV writer with one inference, without a second resident
    // copy of the large model. Other model tests must run serially on small GPUs.
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_candle-audio"))
        .args(["--device", &backend, "tts", "--model"])
        .arg(root.join("parler-tts-mini-v1"))
        .args(["--text", &request.text, "--max-steps", "256", "--output"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "TTS CLI failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let events: Vec<serde_json::Value> = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let chunks = events.iter().filter(|e| e["type"] == "audio").count();
    let reported_frames: u64 = events
        .iter()
        .filter(|e| e["type"] == "audio")
        .filter_map(|e| e["frames"].as_u64())
        .sum();
    let completed = events
        .iter()
        .find(|e| e["type"] == "synthesis_completed")
        .expect("CLI must expose completion metadata");
    assert_eq!(completed["frames"].as_u64(), Some(reported_frames));
    assert!(matches!(
        completed["finish_reason"].as_str(),
        Some("end_of_sequence" | "length")
    ));
    let mut wav = hound::WavReader::open(&path).unwrap();
    assert_eq!(wav.spec().sample_rate, 44100);
    assert_eq!(wav.spec().channels, 1);
    let samples: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.)
        .collect();
    assert_eq!(reported_frames, samples.len() as u64);
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    eprintln!(
        "Parler {device:?}: {:.3}s, {} frames, {chunks} chunks, rms={rms}",
        start.elapsed().as_secs_f64(),
        samples.len()
    );
    assert!(samples.len() > 4410 && samples.iter().all(|s| s.is_finite()) && rms > 0.001);
    assert!(chunks > 1);
    eprintln!("TTS WAV: {}", path.display());
    if output_dir.is_none() {
        std::fs::remove_file(path).unwrap();
    }
    let mut model = load::tts(&root.join("parler-tts-mini-v1"), &device).unwrap();
    let token = CancellationToken::default();
    token.cancel();
    let mut emitted = false;
    assert!(matches!(
        model.synthesize(&request, &token, &mut |_| {
            emitted = true;
            Ok(())
        }),
        Err(Error::Cancelled)
    ));
    assert!(!emitted);
}

#[test]
#[ignore = "requires Mimi checkpoint and JFK WAV"]
fn mimi_emits_before_commit_flushes_exact_duration_and_resets_between_turns() {
    let (root, audio, device) = fixtures();
    let frames = audio.format().sample_rate() as usize + 137;
    let audio = AudioBuffer::new(
        audio.format(),
        audio.samples()[..frames * audio.format().channels() as usize].to_vec(),
    )
    .unwrap();
    let model = load::codec(&root.join("mimi"), &device, audio.format()).unwrap();
    assert_eq!(model.capabilities().streaming, StreamingMode::Native);
    let mut session = RealtimeSession::new(model, frames, frames * 2).unwrap();
    let mut turns = Vec::new();
    for _ in 0..2 {
        let mut pcm = Vec::new();
        let mut supplied = 0;
        let mut first_emission = None;
        for samples in audio
            .samples()
            .chunks(137 * audio.format().channels() as usize)
        {
            supplied += samples.len() / audio.format().channels() as usize;
            session
                .push_audio(
                    &AudioBuffer::new(audio.format(), samples.to_vec()).unwrap(),
                    &mut |e| {
                        if let AudioEvent::Audio { audio, .. } = e {
                            pcm.extend(audio.into_samples());
                        }
                        Ok(())
                    },
                )
                .unwrap();
            if !pcm.is_empty() && first_emission.is_none() {
                first_emission = Some(supplied);
            }
        }
        assert!(
            !pcm.is_empty(),
            "native output must arrive before input commit"
        );
        assert!(
            first_emission.unwrap() < frames / 2,
            "native codec must emit while most of the input is still unsent"
        );
        session
            .commit(&mut |e| {
                if let AudioEvent::Audio { audio, .. } = e {
                    pcm.extend(audio.into_samples());
                }
                Ok(())
            })
            .unwrap();
        let expected = (frames as u64 * 24000 + audio.format().sample_rate() as u64 / 2)
            / audio.format().sample_rate() as u64;
        assert_eq!(pcm.len(), expected as usize);
        assert!(pcm.iter().all(|x| x.is_finite()));
        assert!(pcm.iter().any(|x| x.abs() > 0.001));
        turns.push(pcm);
    }
    let max_error = turns[0]
        .iter()
        .zip(&turns[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "Mimi {device:?}: {} frames/turn, reset max error={max_error}",
        turns[0].len()
    );
    assert!(max_error < 1e-3, "codec state leaked between turns");
}

#[test]
#[ignore = "requires native Whisper and Parler checkpoints; real speech-pipeline validation"]
fn whisper_response_parler_pipeline_produces_recognizable_speech_and_recovers_after_cancel() {
    let (root, _, device) = fixtures();
    let mut wav = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/question-en.wav"
    ))
    .unwrap();
    let audio = AudioBuffer::new(
        AudioFormat::mono(16000).unwrap(),
        wav.samples::<i16>()
            .map(|v| v.unwrap() as f32 / 32768.)
            .collect(),
    )
    .unwrap();
    let recognizer = load::asr(&root.join("whisper-tiny"), &device).unwrap();
    let stream = AsrStream::new(
        recognizer,
        audio.format(),
        audio.frames(),
        TranscriptionOptions {
            language: Some("en".into()),
            translate: false,
        },
    )
    .unwrap();
    // Parler's published generation defaults sample rather than use argmax;
    // this phrase also reaches Length in the official greedy reference.
    let synthesizer = candle_audio::models::registry::ModelRegistry::standard()
        .tts(
            &candle_audio::models::registry::ModelSpec::new(root.join("parler-tts-mini-v1"))
                .with_options(serde_json::json!({"temperature":1.0,"top_k":50})),
            &device,
        )
        .unwrap();
    let pipeline = SpeechPipeline::new(stream, synthesizer, |transcript: &Transcript| {
        eprintln!(
            "Native pipeline recognized {:?}; starting response synthesis",
            transcript.text
        );
        assert!(
            transcript.text.to_lowercase().contains("plus"),
            "unexpected recognition {}",
            transcript.text
        );
        let mut request = SynthesisRequest::new("The answer is four.");
        request.max_steps = 512;
        Ok(Some(request))
    });
    let mut session = RealtimeSession::new(pipeline, audio.frames(), audio.frames() * 2).unwrap();
    let mut samples = Vec::new();
    let mut texts = Vec::new();
    let mut completed = 0;
    let mut sink = |event| {
        match event {
            AudioEvent::Audio { audio, .. } => {
                assert_eq!(audio.format().sample_rate(), 44100);
                samples.extend_from_slice(audio.samples());
            }
            AudioEvent::Text { text, .. } => {
                eprintln!("Native pipeline response text {text:?}");
                texts.push(text);
            }
            AudioEvent::TurnCompleted { .. } => completed += 1,
            _ => {}
        }
        Ok(())
    };
    session.push_audio(&audio, &mut sink).unwrap();
    session.commit(&mut sink).unwrap();
    assert_eq!(completed, 1);
    assert_eq!(texts, ["The answer is four."]);
    assert_eq!(
        session.last_finish_reason(),
        Some(FinishReason::EndOfSequence)
    );
    assert!(samples.len() > 4410 && samples.iter().all(|v| v.is_finite()));
    eprintln!(
        "Native pipeline synthesized {} mono44.1k frames, finish={:?}",
        samples.len(),
        session.last_finish_reason()
    );
    if let Ok(directory) = std::env::var("CANDLE_AUDIO_TEST_OUTPUT_DIR") {
        std::fs::create_dir_all(&directory).unwrap();
        let path = PathBuf::from(directory).join("whisper-parler-pipeline.wav");
        let mut writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for &sample in &samples {
            writer
                .write_sample((sample.clamp(-1., 1.) * 32767.).round() as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
    }
    let output = AudioConverter::convert(
        &AudioBuffer::new(AudioFormat::mono(44100).unwrap(), samples).unwrap(),
        16000,
    )
    .unwrap();
    let mut verifier = load::asr(&root.join("whisper-tiny"), &device).unwrap();
    let transcript = verifier
        .transcribe(
            &output,
            &TranscriptionOptions {
                language: Some("en".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    drop(verifier);
    let cancel = session.cancellation_token();
    let mut cancelled_frames = 0;
    let result = session.push_audio(&audio, &mut |event| {
        if let AudioEvent::Audio { audio, .. } = event {
            cancelled_frames += audio.frames();
            cancel.cancel();
        }
        Ok(())
    });
    assert!(matches!(result, Err(Error::Cancelled)));
    assert!(cancelled_frames > 0);
    session.cancel(&mut |_| Ok(())).unwrap();
    assert_eq!(session.state(), SessionState::Ready);
    assert!(!session.cancellation_token().is_cancelled());
    // A short new input proves that the recognizer/converter can accept a new turn.
    session
        .push_audio(
            &AudioBuffer::new(audio.format(), audio.samples()[..160].to_vec()).unwrap(),
            &mut |_| Ok(()),
        )
        .unwrap();
    session.cancel(&mut |_| Ok(())).unwrap();
    // Tiny ASR may spell the English homophone "four" as "for". Require the
    // complete response and final voiced token; the omitted-token greedy case
    // ("The answer is...") is rejected. LFM3 independently verified this audio
    // as "The answer is four." during checkpoint validation.
    let heard = transcript
        .text
        .trim_end_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    assert!(
        matches!(
            heard.as_str(),
            "the answer is four" | "the answer is 4" | "the answer is for"
        ),
        "independent pipeline output recognition: {}",
        transcript.text
    );
    eprintln!("Native Whisper→response→Parler: output={:?}; cancellation after{cancelled_frames}frames and fresh input recovery passed",transcript.text);
}
