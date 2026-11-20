use candle::{Device, Tensor};
use candle_audio::{
    models::{
        parler::ParlerSamplingOptions,
        registry::{ModelRegistry, ModelSpec},
    },
    AudioTask, Error,
};

#[test]
fn default_greedy_is_seed_independent_and_sampling_restarts_per_request() {
    let logits = Tensor::new(&[0.1f32, 1., -0.1], &Device::Cpu).unwrap();
    let defaults = ParlerSamplingOptions::from_model_options(&serde_json::Value::Null).unwrap();
    for seed in [0, 1, u64::MAX] {
        assert_eq!(
            defaults
                .logits_processor(seed)
                .unwrap()
                .sample(&logits)
                .unwrap(),
            1
        );
    }
    let logits = Tensor::zeros(32, candle::DType::F32, &Device::Cpu).unwrap();
    for (top_k, top_p) in [
        (None, None),
        (Some(8), None),
        (None, Some(0.7)),
        (Some(8), Some(0.9)),
    ] {
        let options = ParlerSamplingOptions {
            temperature: 1.,
            top_k,
            top_p,
        };
        let draw = |seed| {
            let mut sampler = options.logits_processor(seed).unwrap();
            (0..64)
                .map(|_| sampler.sample(&logits).unwrap())
                .collect::<Vec<_>>()
        };
        let first = draw(42);
        let mut previous = options.logits_processor(7).unwrap();
        for _ in 0..100 {
            previous.sample(&logits).unwrap();
        }
        assert_eq!(
            first,
            draw(42),
            "earlier RNG state must not leak into another request"
        );
        assert_ne!(first, draw(43), "seed must affect nondegenerate sampling");
    }
}

#[test]
fn invalid_sampling_is_rejected_before_checkpoint_io_for_both_registry_tasks() {
    let invalid = [
        serde_json::json!({"temperature":-1.}),
        serde_json::json!({"temperature":"warm"}),
        serde_json::json!({"temperature":1.,"top_k":0}),
        serde_json::json!({"temperature":1.,"top_k":1.5}),
        serde_json::json!({"temperature":1.,"top_p":0.}),
        serde_json::json!({"temperature":1.,"top_p":1.01}),
        serde_json::json!({"top_k":8}),
        serde_json::json!({"temperature":1.,"topK":8}),
    ];
    let registry = ModelRegistry::standard();
    for options in invalid {
        for task in [AudioTask::Synthesis, AudioTask::Realtime] {
            let mut spec = ModelSpec::new("/nonexistent/parler-sampling-checkpoint")
                .with_options(options.clone());
            spec.family = Some("parler_tts".into());
            assert!(matches!(registry.load(&spec,task,&Device::Cpu),Err(Error::InvalidInput(_))), "{task:?} should validate options before touching missing checkpoint files: {options}");
        }
    }
    for temperature in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(ParlerSamplingOptions {
            temperature,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
    for p in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(ParlerSamplingOptions {
            temperature: 1.,
            top_p: Some(p),
            top_k: None
        }
        .validate()
        .is_err());
    }
}

#[test]
fn valid_model_options_preserve_requested_filters() {
    assert!(
        ParlerSamplingOptions::from_model_options(&serde_json::json!([1.0, null, null])).is_err()
    );
    let options = ParlerSamplingOptions::from_model_options(
        &serde_json::json!({"temperature":0.8,"top_k":50,"top_p":0.95}),
    )
    .unwrap();
    assert_eq!(
        options,
        ParlerSamplingOptions {
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95)
        }
    );
    assert_eq!(
        ParlerSamplingOptions::from_model_options(&serde_json::json!({})).unwrap(),
        ParlerSamplingOptions::default()
    );
}
