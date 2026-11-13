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
