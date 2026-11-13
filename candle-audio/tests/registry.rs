use candle::Device;
use candle_audio::{models::registry::*, *};
struct Custom;
struct Recognizer;
impl AsrModel for Recognizer {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::mono(16000).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        16000
    }
    fn transcribe(
        &mut self,
        a: &AudioBuffer,
        _: &TranscriptionOptions,
        c: &CancellationToken,
    ) -> Result<Transcript> {
        c.check()?;
        Ok(Transcript {
            text: format!("{} frames", a.frames()),
            start: 0.,
            end: a.duration(),
        })
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
impl ModelLoader for Custom {
    fn family(&self) -> &str {
        "external-test-family"
    }
    fn tasks(&self) -> &[AudioTask] {
        &[AudioTask::Recognition]
    }
    fn load(&self, _: &ModelSpec, _: AudioTask, _: &Device) -> Result<LoadedModel> {
        Ok(LoadedModel::Asr(Box::new(Recognizer)))
    }
}
#[test]
fn application_registers_model_without_runtime_or_transport_dispatch_changes() {
    let mut registry = ModelRegistry::new();
    registry.register(Custom).unwrap();
    assert!(registry.register(Custom).is_err());
    let mut spec = ModelSpec::new("not-a-real-directory");
    spec.family = Some("external-test-family".into());
    let mut model = registry.asr(&spec, &Device::Cpu).unwrap();
    let result = model
        .transcribe(
            &AudioBuffer::new(model.input_format(), vec![0.; 17]).unwrap(),
            &TranscriptionOptions::default(),
            &CancellationToken::default(),
        )
        .unwrap();
    assert_eq!(result.text, "17 frames");
    assert!(registry.tts(&spec, &Device::Cpu).is_err());
    spec.family = Some("unregistered".into());
    assert!(registry.asr(&spec, &Device::Cpu).is_err());
}
#[test]
fn built_in_registry_advertises_lfm3_native_tasks() {
    let registry = ModelRegistry::standard();
    let entries = registry.families().collect::<Vec<_>>();
    let tasks = entries
        .iter()
        .find(|(name, _)| *name == "lfm3_audio")
        .unwrap()
        .1;
    assert!(
        tasks.contains(&AudioTask::Recognition)
            && tasks.contains(&AudioTask::Synthesis)
            && tasks.contains(&AudioTask::Realtime)
    );
}

#[test]
fn loader_constraints_are_available_without_a_checkpoint_and_match_precision_rejection() {
    let registry = ModelRegistry::standard();
    for (family, tasks) in registry.families() {
        let constraints = registry.constraints(family).unwrap();
        assert!(constraints
            .compiled_backends
            .iter()
            .any(|name| name == "cpu"));
        assert_eq!(
            constraints
                .activation_dtypes
                .iter()
                .any(|name| name == "bf16"),
            ["voxtral", "gemma4"].contains(&family)
        );
        let spec = ModelSpec::new("missing-checkpoint")
            .with_family(family)
            .with_dtype(candle::DType::F16);
        // Precision errors must precede attempts to open model artifacts.
        assert!(matches!(
            registry.load(&spec, tasks[0], &Device::Cpu),
            Err(Error::Unsupported(_))
        ));
    }
    let mut custom = ModelRegistry::new();
    custom.register(Custom).unwrap();
    let constraints = custom.constraints("external-test-family").unwrap();
    assert!(constraints.activation_dtypes.is_empty());
    assert!(constraints.compiled_backends.is_empty());
    assert!(custom.constraints("unknown").is_err());
    let legacy: ModelCapabilities = serde_json::from_str(r#"{"model_id":"legacy"}"#).unwrap();
    assert_eq!(legacy.model_id, "legacy");
    assert_eq!(legacy.max_context_tokens, None);
    assert!(legacy.input_streaming.is_none() && legacy.output_streaming.is_none());
}

#[test]
fn malformed_builtin_options_are_rejected_before_weights_or_json_indexing() {
    let registry = ModelRegistry::standard();
    for family in ["quantized_metavoice", "lfm3_audio", "gemma4", "voxtral"] {
        let task = if family == "quantized_metavoice" {
            AudioTask::Synthesis
        } else {
            AudioTask::Recognition
        };
        let spec = ModelSpec::new("missing-checkpoint")
            .with_family(family)
            .with_options(serde_json::json!(["invalid"]));
        assert!(matches!(
            registry.load(&spec, task, &Device::Cpu),
            Err(Error::InvalidInput(_))
        ));
    }
}

#[test]
fn dac_submodel_option_types_and_paths_are_rejected_before_checkpoint_io() {
    let registry = ModelRegistry::standard();
    for options in [
        serde_json::json!({"config_key":7}),
        serde_json::json!({"config_key":null}),
        serde_json::json!({"config_key":""}),
        serde_json::json!({"config_key":"audio_encoder..model"}),
        serde_json::json!({"weight_prefix":["audio_encoder"]}),
        serde_json::json!({"weight_prefix":false}),
        serde_json::json!({"weight_prefix":"../weights"}),
        serde_json::json!({"weight_prefix":"audio_encoder."}),
    ] {
        let spec = ModelSpec::new("missing-dac-checkpoint")
            .with_family("dac")
            .with_options(options);
        assert!(matches!(
            registry.codec(&spec, &Device::Cpu),
            Err(Error::InvalidInput(_))
        ));
    }
}

#[test]
#[cfg(feature = "cli")]
fn dac_config_selection_is_checked_before_mapping_weights() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("config.json"),
        r#"{"audio_encoder":false}"#,
    )
    .unwrap();
    for key in ["missing", "audio_encoder"] {
        let spec = ModelSpec::new(directory.path())
            .with_family("dac")
            .with_options(
                serde_json::json!({"config_key":key,"weight_prefix":"audio_encoder.model"}),
            );
        assert!(matches!(
            ModelRegistry::standard().codec(&spec, &Device::Cpu),
            Err(Error::InvalidInput(_))
        ));
    }
}
