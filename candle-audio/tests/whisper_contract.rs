//! Contract regressions use tiny zero-weight networks and a deliberately shared
//! multilingual tokenizer, matching the .en tokenizer property without downloads.
use candle::{DType, Device};
use candle_audio::{
    models::whisper::{Whisper, WhisperModel},
    *,
};
use candle_transformers::models::whisper as w;
use tokenizers::{models::wordlevel::WordLevel, Tokenizer};
fn model(multilingual: bool, emit_word: bool) -> Whisper {
    let vocab = if multilingual { 51865 } else { 51864 };
    let eot = (vocab - 1) as u32;
    let tokens = [
        (w::SOT_TOKEN, 50257u32),
        (w::EOT_TOKEN, eot),
        (w::NO_TIMESTAMPS_TOKEN, 50362),
        (w::TRANSCRIBE_TOKEN, 50358),
        (w::TRANSLATE_TOKEN, 50359),
        ("<|en|>", 50258),
        ("<|es|>", 50300),
        ("hello", 7),
        ("[UNK]", 0),
    ];
    let tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                tokens
                    .into_iter()
                    .map(|(s, id)| (s.to_owned(), id))
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    let config = w::Config {
        num_mel_bins: 80,
        max_source_positions: 2,
        d_model: 4,
        encoder_attention_heads: 1,
        encoder_layers: 0,
        vocab_size: vocab,
        max_target_positions: 8,
        decoder_attention_heads: 1,
        decoder_layers: 0,
        suppress_tokens: (0..vocab as u32)
            .filter(|i| *i != if emit_word { 7 } else { eot })
            .collect(),
    };
    let vb = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
    let inner = w::model::Whisper::load(&vb, config).unwrap();
    Whisper::new(
        WhisperModel::Normal(inner),
        tokenizer,
        Whisper::default_mel_filters(80).unwrap(),
        Device::Cpu,
    )
    .unwrap()
}
fn audio() -> AudioBuffer {
    AudioBuffer::new(AudioFormat::mono(16000).unwrap(), vec![0.; 480]).unwrap()
}
#[test]
fn english_only_weights_do_not_infer_multilingual_support_from_tokenizer() {
    let mut model = model(false, false);
    assert!(matches!(
        model.transcribe(
            &audio(),
            &TranscriptionOptions {
                language: Some("es".into()),
                translate: false
            },
            &CancellationToken::default()
        ),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        model.transcribe(
            &audio(),
            &TranscriptionOptions {
                language: Some("en".into()),
                translate: true
            },
            &CancellationToken::default()
        ),
        Err(Error::Unsupported(_))
    ));
    let result = model
        .transcribe_detailed(
            &audio(),
            &TranscriptionOptions::default(),
            &CancellationToken::default(),
        )
        .unwrap();
    assert_eq!(result.language.as_deref(), Some("en"));
    assert_eq!(result.finish_reason, FinishReason::EndOfSequence);
    assert_eq!(model.descriptor().languages, vec!["en"]);
}
#[test]
fn multilingual_weights_detect_language_and_report_token_budget_exhaustion() {
    let mut detected = model(true, false);
    let result = detected
        .transcribe_detailed(
            &audio(),
            &TranscriptionOptions::default(),
            &CancellationToken::default(),
        )
        .unwrap();
    // Tied language logits choose the first model token ID, independently
    // of randomized tokenizer vocabulary iteration order.
    assert_eq!(result.language.as_deref(), Some("en"));
    assert_eq!(result.finish_reason, FinishReason::EndOfSequence);
    let mut limited = model(true, true);
    limited.set_max_tokens(1).unwrap();
    let result = limited
        .transcribe_detailed(
            &audio(),
            &TranscriptionOptions {
                language: Some("es".into()),
                translate: false,
            },
            &CancellationToken::default(),
        )
        .unwrap();
    assert_eq!(result.finish_reason, FinishReason::Length);
    assert_eq!(result.transcript.text, "hello");
}
