use candle::{DType, Device, Shape, Tensor};
use candle_audio::{models::codecs::*, *};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};
use candle_transformers::models::{dac, encodec, snac};

struct SmallWeights;
impl SimpleBackend for SmallWeights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> candle::Result<Tensor> {
        let value = if name.ends_with("alpha")
            || name.ends_with("original0")
            || (shape.dims().len() == 1 && name.ends_with("weight"))
        {
            1.
        } else if name.ends_with("bias") {
            0.
        } else {
            0.02
        };
        Tensor::full(value, shape, device)?.to_dtype(dtype)
    }
    fn get_unchecked(&self, _: &str, _: DType, _: &Device) -> candle::Result<Tensor> {
        candle::bail!("test backend needs explicit shapes")
    }
    fn contains_tensor(&self, name: &str) -> bool {
        name.ends_with("weight")
    }
}
fn weights() -> VarBuilder<'static> {
    VarBuilder::from_backend(Box::new(SmallWeights), DType::F32, Device::Cpu)
}
fn input(format: AudioFormat, n: usize) -> AudioBuffer {
    AudioBuffer::new(
        format,
        (0..n * format.channels() as usize)
            .map(|i| (i as f32 * 0.03).sin() * 0.2)
            .collect(),
    )
    .unwrap()
}
fn encodec_config() -> encodec::Config {
    encodec::Config {
        sampling_rate: 8000,
        target_bandwidths: vec![8.],
        audio_channels: 2,
        hidden_size: 4,
        num_filters: 2,
        upsampling_ratios: vec![2],
        kernel_size: 3,
        last_kernel_size: 3,
        residual_kernel_size: 3,
        codebook_size: 4,
        codebook_dim: Some(4),
        num_lstm_layers: 1,
        norm_type: encodec::NormType::TimeGroupNorm,
        ..Default::default()
    }
}
#[test]
fn encodec_preserves_stereo_length_and_normalization_segments() {
    let mut cfg = encodec_config();
    cfg.normalize = true;
    cfg.chunk_length_s = Some(0.001);
    cfg.overlap = Some(0.5);
    let model = encodec::Model::new(&cfg, weights()).unwrap();
    let mut codec = EncodecCodec::new(model, cfg, Device::Cpu).unwrap();
    let source = input(codec.format().audio_format, 20);
    let codes = codec
        .encode(&source, &CancellationToken::default())
        .unwrap();
    assert_eq!(codes.segments().len(), 5);
    assert!(codes
        .segments()
        .iter()
        .all(|chunk| chunk.scale > 0. && chunk.scale < 1.));
    let json = serde_json::to_string(&codes).unwrap();
    let restored: AudioCodes = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, codes);
    let expanded = MultiScaleAudioCodes::from_uniform(codes.clone())
        .into_uniform()
        .unwrap();
    assert_eq!(expanded, codes);
    let decoded = codec.decode(&codes, &CancellationToken::default()).unwrap();
    assert_eq!(decoded.format(), source.format());
    assert_eq!(decoded.frames(), source.frames());
    assert!(decoded.samples().iter().all(|v| v.is_finite()));
    assert!(codec
        .encode(
            &input(AudioFormat::mono(8000).unwrap(), 3),
            &CancellationToken::default()
        )
        .is_err());
}
#[test]
fn dac_native_encoder_quantizer_and_tanh_decoder_roundtrip_exact_length() {
    let cfg = dac::Config {
        num_codebooks: 2,
        model_bitrate: 0,
        codebook_size: 4,
        latent_dim: 4,
        frame_rate: 4000,
        sampling_rate: 8000,
    };
    let model = dac::Model::with_architecture(&cfg, 2, &[2], 4, &[2], 2, weights()).unwrap();
    let mut codec = DacCodec::new(model, &cfg, Device::Cpu).unwrap();
    let source = input(codec.format().audio_format, 17);
    let codes = codec
        .encode(&source, &CancellationToken::default())
        .unwrap();
    assert_eq!(codes.frames(), 9);
    assert_eq!(codes.original_audio_frames(), Some(17));
    assert!(codes.codes().iter().all(|code| *code < 4));
    let decoded = codec.decode(&codes, &CancellationToken::default()).unwrap();
    assert_eq!(decoded.frames(), 17);
    assert!(decoded
        .samples()
        .iter()
        .all(|value| value.is_finite() && value.abs() <= 1.));
    let token = CancellationToken::default();
    token.cancel();
    assert!(matches!(
        codec.encode(&source, &token),
        Err(Error::Cancelled)
    ));
}
#[test]
fn snac_attention_and_multirate_codebooks_work_without_padding_invented_codes() {
    let cfg = snac::Config {
        sampling_rate: 8000,
        encoder_dim: 32,
        encoder_rates: vec![2],
        decoder_dim: 64,
        decoder_rates: vec![2],
        attn_window_size: Some(2),
        codebook_size: 4,
        codebook_dim: 2,
        vq_strides: vec![2, 1],
        noise: false,
        depthwise: false,
    };
    let model = snac::Model::new(&cfg, weights()).unwrap();
    let mut codec = SnacCodec::new(model, Device::Cpu).unwrap();
    let source = input(codec.format().audio_format, 17);
    let codes = codec
        .encode_multiscale(&source, &CancellationToken::default())
        .unwrap();
    assert_eq!(codes.codebooks()[0].len() * 2, codes.codebooks()[1].len());
    assert_eq!(codes.original_audio_frames(), Some(17));
    let decoded = codec
        .decode_multiscale(&codes, &CancellationToken::default())
        .unwrap();
    assert_eq!(decoded.frames(), 17);
    assert!(decoded
        .samples()
        .iter()
        .all(|value| value.is_finite() && value.abs() <= 1.));
    let uniform = codec
        .encode(&source, &CancellationToken::default())
        .unwrap();
    let from_uniform = codec
        .decode(&uniform, &CancellationToken::default())
        .unwrap();
    assert_eq!(decoded, from_uniform);
    let mut corrupt = uniform.codes().to_vec();
    corrupt[2] = if corrupt[0] == 0 { 1 } else { 0 };
    let corrupt = AudioCodes::new(codec.format(), corrupt).unwrap();
    assert!(codec
        .decode(&corrupt, &CancellationToken::default())
        .is_err());
}
#[test]
fn codec_config_validation_rejects_invalid_rates_before_loading_weights() {
    let mut cfg = encodec_config();
    cfg.upsampling_ratios = vec![0];
    assert!(encodec::Model::new(&cfg, weights()).is_err());
    let cfg:encodec::Config=serde_json::from_str(r#"{"sampling_rate":48000,"audio_channels":2,"norm_type":"time_group_norm","pad_mode":"reflect","chunk_length_s":1.0,"overlap":0.01}"#).unwrap();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.chunk_length_s, Some(1.));
}
#[test]
#[ignore = "requires cached native Mimi checkpoint; uses CPU only"]
fn pretrained_mimi_encode_decode_codes_retains_duration_and_resets() {
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_MIMI_MODEL").expect("set CANDLE_AUDIO_MIMI_MODEL"),
    );
    let mut codec = load_mimi(&directory, &Device::Cpu, &serde_json::json!({})).unwrap();
    let source = input(codec.format().audio_format, 2407);
    let mut previous: Option<(AudioCodes, Vec<f32>)> = None;
    for _ in 0..2 {
        let codes = codec
            .encode(&source, &CancellationToken::default())
            .unwrap();
        let decoded = codec.decode(&codes, &CancellationToken::default()).unwrap();
        assert_eq!(decoded.frames(), source.frames());
        assert!(decoded.samples().iter().all(|value| value.is_finite()));
        if let Some((previous_codes, previous_samples)) = &previous {
            assert_eq!(&codes, previous_codes);
            // Parallel CPU reductions may differ by a few float32 ULPs.
            // Reset must preserve the discrete codes and decoded waveform.
            let difference = decoded
                .samples()
                .iter()
                .zip(previous_samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(difference < 1e-6, "reset waveform difference={difference}");
        }
        previous = Some((codes, decoded.into_samples()));
    }
}

#[test]
#[ignore = "requires cached Parler checkpoint containing DAC weights; uses CPU only"]
fn pretrained_dac_encoder_and_decoder_roundtrip_from_parler_prefix() {
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_PARLER_MODEL").expect("set CANDLE_AUDIO_PARLER_MODEL"),
    );
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    let config: dac::Config = serde_json::from_value(json["audio_encoder"].clone()).unwrap();
    // Map the shared checkpoint, then request only its DAC prefix. This does not
    // instantiate Parler's text encoders or autoregressive language model.
    let weights = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[directory.join("model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    };
    let model = dac::Model::new(&config, weights.pp("audio_encoder.model")).unwrap();
    let mut codec = DacCodec::new(model, &config, Device::Cpu).unwrap();
    let source = input(codec.format().audio_format, 1027);
    let codes = codec
        .encode(&source, &CancellationToken::default())
        .unwrap();
    assert_eq!(codes.frames(), 3);
    assert_eq!(codes.original_audio_frames(), Some(source.frames()));
    assert!(codes.codes().iter().all(|code| *code < 1024));
    let decoded = codec.decode(&codes, &CancellationToken::default()).unwrap();
    assert_eq!(decoded.format(), source.format());
    assert_eq!(decoded.frames(), source.frames());
    assert!(decoded
        .samples()
        .iter()
        .all(|value| value.is_finite() && value.abs() <= 1.));
    assert!(decoded.samples().iter().any(|value| value.abs() > 1e-4));
    codec.reset().unwrap();
    let repeated = codec
        .encode(&source, &CancellationToken::default())
        .unwrap();
    assert_eq!(repeated, codes);
}

#[cfg(feature = "cli")]
fn pretrained_question_roundtrip(mut codec: Box<dyn AudioCodec + Send>) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/question-en.wav");
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 16000, 16)
    );
    let source = AudioBuffer::new(
        AudioFormat::mono(16000).unwrap(),
        reader
            .samples::<i16>()
            .map(|value| value.unwrap() as f32 / 32768.)
            .collect(),
    )
    .unwrap();
    let source =
        AudioConverter::convert(&source, codec.format().audio_format.sample_rate()).unwrap();
    let token = CancellationToken::default();
    let rms = |samples: &[f32]| {
        (samples.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
    };
    let started = std::time::Instant::now();
    let codes = codec.encode_multiscale(&source, &token).unwrap();
    assert_eq!(codes.original_audio_frames(), Some(source.frames()));
    assert!(
        codes
            .codebooks()
            .iter()
            .all(|book| !book.is_empty()
                && book.iter().all(|code| *code < codec.format().cardinality))
    );
    let decoded = codec.decode_multiscale(&codes, &token).unwrap();
    assert_eq!(decoded.format(), source.format());
    assert_eq!(decoded.frames(), source.frames());
    assert!(decoded.samples().iter().all(|value| value.is_finite()));
    let output_rms = rms(decoded.samples());
    let ratio = output_rms / rms(source.samples());
    assert!((0.05..5.).contains(&ratio), "decoded RMS ratio={ratio}");
    codec.reset().unwrap();
    let repeated = codec.encode_multiscale(&source, &token).unwrap();
    assert_eq!(repeated, codes);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        codec.decode_multiscale(&codes, &cancelled),
        Err(Error::Cancelled)
    ));
    let after_cancel = codec.decode_multiscale(&codes, &token).unwrap();
    assert_eq!(after_cancel.frames(), source.frames());
    assert!(after_cancel.samples().iter().all(|value| value.is_finite()));
    eprintln!(
        "{}: {} frames, codebook lengths {:?}, RMS {:.6}, RMS ratio {:.6}, {:.3}s",
        codec.descriptor().model_id,
        decoded.frames(),
        codes.codebooks().iter().map(Vec::len).collect::<Vec<_>>(),
        output_rms,
        ratio,
        started.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "cli")]
#[test]
#[ignore = "requires cached official EnCodec checkpoint; uses CPU only"]
fn pretrained_encodec_question_roundtrip_and_reset() {
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_ENCODEC_MODEL").expect("set CANDLE_AUDIO_ENCODEC_MODEL"),
    );
    pretrained_question_roundtrip(
        load_encodec(&directory, &Device::Cpu, &serde_json::json!({})).unwrap(),
    );
}

#[cfg(feature = "cli")]
#[test]
#[ignore = "requires cached official SNAC checkpoint; uses CPU only"]
fn pretrained_snac_question_roundtrip_and_reset() {
    let directory = std::path::PathBuf::from(
        std::env::var("CANDLE_AUDIO_SNAC_MODEL").expect("set CANDLE_AUDIO_SNAC_MODEL"),
    );
    pretrained_question_roundtrip(
        load_snac(&directory, &Device::Cpu, &serde_json::json!({})).unwrap(),
    );
}
