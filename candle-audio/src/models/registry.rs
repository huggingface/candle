//! Extensible native model factories. Applications may register checkpoint
//! families without changing session, transport, or CLI implementations.
use crate::*;
use candle::{DType, Device};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub directory: PathBuf,
    /// Override configuration metadata for a locally constructed checkpoint.
    pub family: Option<String>,
    pub input_format: Option<AudioFormat>,
    pub dtype: DType,
    pub options: serde_json::Value,
}
impl ModelSpec {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            family: None,
            input_format: None,
            dtype: DType::F32,
            options: serde_json::Value::Null,
        }
    }
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }
    pub fn with_input(mut self, format: AudioFormat) -> Self {
        self.input_format = Some(format);
        self
    }
    pub fn with_options(mut self, options: serde_json::Value) -> Self {
        self.options = options;
        self
    }
    pub fn family(&self) -> Result<String> {
        self.family
            .clone()
            .map(Ok)
            .unwrap_or_else(|| super::load::model_type(&self.directory))
    }
}

pub enum LoadedModel {
    Asr(Box<dyn AsrModel + Send>),
    Tts(Box<dyn TtsModel + Send>),
    Realtime(Box<dyn RealtimeModel + Send>),
    Codec(Box<dyn AudioCodec + Send>),
    Generator(Box<dyn AudioGenerationModel + Send>),
    Understanding(Box<dyn AudioUnderstandingModel + Send>),
}
impl LoadedModel {
    pub fn descriptor(&self) -> ModelCapabilities {
        match self {
            Self::Asr(m) => m.descriptor(),
            Self::Tts(m) => m.descriptor(),
            Self::Realtime(m) => m.descriptor(),
            Self::Codec(m) => m.descriptor(),
            Self::Generator(m) => m.descriptor(),
            Self::Understanding(m) => m.descriptor(),
        }
    }
}

pub trait ModelLoader: Send + Sync {
    fn family(&self) -> &str;
    fn tasks(&self) -> &[AudioTask];
    /// Declared loader restrictions, available before loading checkpoint weights.
    /// Empty lists mean unspecified, not unrestricted support.
    fn constraints(&self) -> LoaderConstraints {
        LoaderConstraints::default()
    }
    fn load(&self, spec: &ModelSpec, task: AudioTask, device: &Device) -> Result<LoadedModel>;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LoaderConstraints {
    pub activation_dtypes: Vec<String>,
    /// Backends enabled in this build. This is not a checkpoint validation matrix
    /// or a promise that every model operation supports every dtype on them.
    pub compiled_backends: Vec<String>,
}
#[derive(Clone, Default)]
pub struct ModelRegistry {
    loaders: BTreeMap<String, Arc<dyn ModelLoader>>,
}
impl ModelRegistry {
    /// An empty registry, useful for applications that deliberately restrict models.
    pub fn new() -> Self {
        Self::default()
    }
    pub fn standard() -> Self {
        let mut registry = Self::new();
        for family in [
            "whisper",
            "parler_tts",
            "mimi",
            "lfm3_audio",
            "encodec",
            "dac",
            "snac",
            "csm",
            "metavoice",
            "quantized_metavoice",
            "musicgen",
            "voxtral",
            "gemma4",
        ] {
            registry
                .register(BuiltinLoader(family))
                .expect("distinct built-in families");
        }
        registry
    }
    pub fn register(&mut self, loader: impl ModelLoader + 'static) -> Result<()> {
        let family = loader.family().to_owned();
        if family.is_empty() || self.loaders.contains_key(&family) {
            return Err(Error::InvalidInput(format!(
                "model family {family:?} is empty or already registered"
            )));
        }
        self.loaders.insert(family, Arc::new(loader));
        Ok(())
    }
    pub fn families(&self) -> impl Iterator<Item = (&str, &[AudioTask])> {
        self.loaders
            .iter()
            .map(|(name, loader)| (name.as_str(), loader.tasks()))
    }
    pub fn constraints(&self, family: &str) -> Result<LoaderConstraints> {
        self.loaders
            .get(family)
            .map(|loader| loader.constraints())
            .ok_or_else(|| {
                Error::Unsupported(format!("unregistered native model family {family:?}"))
            })
    }
    pub fn load(&self, spec: &ModelSpec, task: AudioTask, device: &Device) -> Result<LoadedModel> {
        let family = spec.family()?;
        let loader = self.loaders.get(&family).ok_or_else(|| {
            Error::Unsupported(format!(
                "unregistered native model family {family:?}; registered families: {}",
                self.loaders.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;
        if !loader.tasks().contains(&task) {
            return Err(Error::Unsupported(format!(
                "model {family:?} does not implement task {task:?}"
            )));
        }
        loader.load(spec, task, device)
    }
    pub fn asr(&self, spec: &ModelSpec, device: &Device) -> Result<Box<dyn AsrModel + Send>> {
        match self.load(spec, AudioTask::Recognition, device)? {
            LoadedModel::Asr(m) => Ok(m),
            _ => Err(Error::InvalidState(
                "ASR factory returned a different model task".into(),
            )),
        }
    }
    pub fn tts(&self, spec: &ModelSpec, device: &Device) -> Result<Box<dyn TtsModel + Send>> {
        match self.load(spec, AudioTask::Synthesis, device)? {
            LoadedModel::Tts(m) => Ok(m),
            _ => Err(Error::InvalidState(
                "TTS factory returned a different model task".into(),
            )),
        }
    }
    pub fn realtime(
        &self,
        spec: &ModelSpec,
        device: &Device,
    ) -> Result<Box<dyn RealtimeModel + Send>> {
        match self.load(spec, AudioTask::Realtime, device)? {
            LoadedModel::Realtime(m) => Ok(m),
            _ => Err(Error::InvalidState(
                "realtime factory returned a different model task".into(),
            )),
        }
    }
    pub fn generator(
        &self,
        spec: &ModelSpec,
        device: &Device,
    ) -> Result<Box<dyn AudioGenerationModel + Send>> {
        match self.load(spec, AudioTask::AudioGeneration, device)? {
            LoadedModel::Generator(model) => Ok(model),
            _ => Err(Error::InvalidState(
                "audio-generation factory returned wrong model task".into(),
            )),
        }
    }
    pub fn understanding(
        &self,
        spec: &ModelSpec,
        device: &Device,
    ) -> Result<Box<dyn AudioUnderstandingModel + Send>> {
        match self.load(spec, AudioTask::Understanding, device)? {
            LoadedModel::Understanding(model) => Ok(model),
            _ => Err(Error::InvalidState(
                "understanding factory returned wrong model task".into(),
            )),
        }
    }
    pub fn codec(&self, spec: &ModelSpec, device: &Device) -> Result<Box<dyn AudioCodec + Send>> {
        match self.load(spec, AudioTask::Codec, device)? {
            LoadedModel::Codec(m) => Ok(m),
            _ => Err(Error::InvalidState(
                "codec factory returned a different model task".into(),
            )),
        }
    }
}
struct BuiltinLoader(&'static str);
impl ModelLoader for BuiltinLoader {
    fn family(&self) -> &str {
        self.0
    }
    fn constraints(&self) -> LoaderConstraints {
        let mut compiled_backends = vec!["cpu".to_owned()];
        if cfg!(feature = "vulkan") {
            compiled_backends.push("vulkan".into());
        }
        if cfg!(feature = "cuda") {
            compiled_backends.push("cuda".into());
        }
        if cfg!(feature = "metal") {
            compiled_backends.push("metal".into());
        }
        LoaderConstraints {
            activation_dtypes: if ["voxtral", "gemma4"].contains(&self.0) {
                vec!["f32".into(), "bf16".into()]
            } else {
                vec!["f32".into()]
            },
            compiled_backends,
        }
    }
    fn tasks(&self) -> &[AudioTask] {
        match self.0 {
            "whisper" => &[AudioTask::Recognition, AudioTask::Realtime],
            "parler_tts" => &[AudioTask::Synthesis, AudioTask::Realtime],
            "csm" => &[AudioTask::Synthesis, AudioTask::Realtime],
            "musicgen" => &[AudioTask::AudioGeneration],
            "voxtral" | "gemma4" => &[
                AudioTask::Recognition,
                AudioTask::Understanding,
                AudioTask::Realtime,
            ],
            "metavoice" | "quantized_metavoice" => &[AudioTask::Synthesis],
            "lfm3_audio" => &[
                AudioTask::Recognition,
                AudioTask::Synthesis,
                AudioTask::Realtime,
            ],
            "mimi" | "encodec" | "snac" | "dac" => &[AudioTask::Codec, AudioTask::Realtime],
            _ => &[],
        }
    }
    fn load(&self, spec: &ModelSpec, task: AudioTask, device: &Device) -> Result<LoadedModel> {
        if !spec.options.is_null() && !spec.options.is_object() {
            return Err(Error::InvalidInput(
                "builtin model options must be a JSON object".into(),
            ));
        }
        // Reject precisions before mapping weights unless the adapter implements
        // its full tensor conversion path (including audio preprocessing).
        if spec.dtype != DType::F32
            && !(["voxtral", "gemma4"].contains(&self.0) && spec.dtype == DType::BF16)
        {
            return Err(Error::Unsupported(format!(
                "{} adapter does not support the requested precision (F32 required; Voxtral/Gemma4 also support BF16)",
                self.0
            )));
        }
        if self.0 == "musicgen" {
            return Ok(LoadedModel::Generator(Box::new(
                super::musicgen::MusicGen::load(&spec.directory, device, &spec.options)?,
            )));
        }
        if self.0 == "gemma4" {
            let mut options = spec.options.clone();
            if options.is_null() {
                options = serde_json::json!({});
            }
            options["dtype"] = if spec.dtype == DType::BF16 {
                "bf16"
            } else {
                "f32"
            }
            .into();
            let model = super::gemma4::Gemma4::load(
                &spec.directory,
                device,
                &options,
                spec.input_format.unwrap_or(AudioFormat::mono(16000)?),
            )?;
            return match task {
                AudioTask::Recognition => Ok(LoadedModel::Asr(Box::new(model))),
                AudioTask::Understanding => Ok(LoadedModel::Understanding(Box::new(model))),
                AudioTask::Realtime => Ok(LoadedModel::Realtime(Box::new(model.into_realtime()?))),
                _ => Err(Error::Unsupported("Gemma4 task not supported".into())),
            };
        }
        if self.0 == "voxtral" {
            let mut options = spec.options.clone();
            if options.is_null() {
                options = serde_json::json!({});
            }
            options["dtype"] = if spec.dtype == DType::BF16 {
                "bf16"
            } else {
                "f32"
            }
            .into();
            let model = super::voxtral::Voxtral::load(
                &spec.directory,
                device,
                &options,
                spec.input_format.unwrap_or(AudioFormat::mono(16000)?),
            )?;
            return match task {
                AudioTask::Recognition => Ok(LoadedModel::Asr(Box::new(model))),
                AudioTask::Understanding => Ok(LoadedModel::Understanding(Box::new(model))),
                AudioTask::Realtime => Ok(LoadedModel::Realtime(Box::new(model.into_realtime()?))),
                _ => Err(Error::Unsupported("Voxtral task not supported".into())),
            };
        }
        if self.0 == "csm" {
            let model = super::csm::Csm::load(&spec.directory, device, &spec.options)?;
            return match task {
                AudioTask::Synthesis => Ok(LoadedModel::Tts(Box::new(model))),
                AudioTask::Realtime => Ok(LoadedModel::Realtime(Box::new(model))),
                _ => Err(Error::Unsupported("CSM task not supported".into())),
            };
        }
        if ["metavoice", "quantized_metavoice"].contains(&self.0) {
            let mut options = spec.options.clone();
            if options.is_null() {
                options = serde_json::json!({});
            }
            if self.0 == "quantized_metavoice" {
                options["quantized"] = true.into();
            }
            return Ok(LoadedModel::Tts(Box::new(
                super::metavoice::MetaVoice::load(&spec.directory, device, &options)?,
            )));
        }
        if self.0 == "lfm3_audio" {
            let model = super::lfm3::Lfm3::load(
                &spec.directory,
                device,
                spec.input_format.unwrap_or(AudioFormat::mono(16000)?),
                &spec.options,
            )?;
            return match task {
                AudioTask::Recognition => Ok(LoadedModel::Asr(Box::new(model))),
                AudioTask::Synthesis => Ok(LoadedModel::Tts(Box::new(model))),
                AudioTask::Realtime => Ok(LoadedModel::Realtime(Box::new(model))),
                _ => Err(Error::Unsupported("LFM3 task not supported".into())),
            };
        }
        if task == AudioTask::Codec
            || (task == AudioTask::Realtime && ["encodec", "snac", "dac"].contains(&self.0))
        {
            let codec = match self.0 {
                "mimi" => super::codecs::load_mimi(&spec.directory, device, &spec.options)?,
                "encodec" => super::codecs::load_encodec(&spec.directory, device, &spec.options)?,
                "snac" => super::codecs::load_snac(&spec.directory, device, &spec.options)?,
                "dac" => super::codecs::load_dac(&spec.directory, device, &spec.options)?,
                _ => return Err(Error::Unsupported(format!("{} has no codec task", self.0))),
            };
            if task == AudioTask::Codec {
                return Ok(LoadedModel::Codec(codec));
            }
            let max_frames = codec.format().audio_format.sample_rate() as usize * 30;
            let source = spec.input_format.unwrap_or(codec.format().audio_format);
            return Ok(LoadedModel::Realtime(Box::new(
                super::codec_stream::CodecStream::new(codec, source, max_frames)?,
            )));
        }
        match (self.0, task) {
            ("whisper", AudioTask::Recognition) => Ok(LoadedModel::Asr(super::load::load_whisper(
                &spec.directory,
                device,
                &spec.options,
            )?)),
            ("parler_tts", AudioTask::Synthesis) => Ok(LoadedModel::Tts(super::load::load_parler(
                &spec.directory,
                device,
                &spec.options,
            )?)),
            ("mimi", AudioTask::Realtime) => Ok(LoadedModel::Realtime(super::load::load_mimi(
                &spec.directory,
                device,
                spec.input_format.unwrap_or(AudioFormat::mono(24000)?),
                &spec.options,
            )?)),
            ("whisper", AudioTask::Realtime) => {
                let model = super::load::load_whisper(&spec.directory, device, &spec.options)?;
                let input = spec.input_format.unwrap_or(model.input_format());
                let window = if let Some(value) = spec.options.get("window_frames") {
                    value
                        .as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| Error::InvalidInput("window_frames must fit usize".into()))?
                } else if let Some(value) = spec.options.get("window_ms") {
                    let millis = value.as_u64().ok_or_else(|| {
                        Error::InvalidInput("window_ms must be a positive integer".into())
                    })?;
                    usize::try_from(
                        (model.input_format().sample_rate() as u128 * millis as u128) / 1000,
                    )
                    .map_err(|_| Error::InvalidInput("window_ms overflow".into()))?
                } else {
                    (model.input_format().sample_rate() as usize * 5).min(model.max_input_frames())
                };
                let options = spec
                    .options
                    .get("transcription")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()?
                    .unwrap_or_default();
                Ok(LoadedModel::Realtime(Box::new(AsrStream::new(
                    model, input, window, options,
                )?)))
            }
            ("parler_tts", AudioTask::Realtime) => {
                let model = super::load::load_parler(&spec.directory, device, &spec.options)?;
                Ok(LoadedModel::Realtime(Box::new(TtsStream::new(
                    model,
                    SynthesisRequest::new(""),
                    65536,
                )?)))
            }
            _ => Err(Error::Unsupported(format!("{} task {task:?}", self.0))),
        }
    }
}

/// Convenience for a caller using the standard built-in model catalogue.
pub fn load(directory: &Path, task: AudioTask, device: &Device) -> Result<LoadedModel> {
    ModelRegistry::standard().load(&ModelSpec::new(directory), task, device)
}
