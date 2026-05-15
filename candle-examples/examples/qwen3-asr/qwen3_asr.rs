//! Qwen3-ASR high-level API used by the example binary.

use anyhow::Result;
use candle::Device;

pub use crate::audio::input::AudioInput;
pub use crate::inference::streaming::AsrStream;
pub use crate::inference::types::{AsrTranscription, Batch, StreamOptions, TranscribeOptions};
pub use crate::model::weights::LoadOptions;
pub use crate::processor::AsrProcessor;

#[derive(Debug)]
pub struct Qwen3Asr {
    device: Device,
    config: crate::config::AsrConfig,
    processor: AsrProcessor,
    model: crate::model::AsrModel,
}

impl Qwen3Asr {
    /// Canonical language names supported by the official Qwen3-ASR stack.
    pub fn supported_languages(&self) -> &'static [&'static str] {
        crate::inference::utils::SUPPORTED_LANGUAGES
    }

    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let (config, model) =
            crate::model::weights::load_model_from_pretrained(model_id_or_path, device, opts)?;
        let thinker_type = config
            .thinker_config
            .model_type
            .as_deref()
            .unwrap_or_default();
        if thinker_type.contains("forced_aligner") {
            anyhow::bail!(
                "loaded a forced aligner checkpoint (thinker_config.model_type={thinker_type:?}); use the forced aligner API instead"
            );
        }
        let tokenizer = crate::processor::tokenizer::Tokenizer::from_pretrained(model_id_or_path)?;
        let processor = AsrProcessor::new(tokenizer);
        Ok(Self {
            device: device.clone(),
            config,
            processor,
            model,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &crate::config::AsrConfig {
        &self.config
    }

    pub fn processor(&self) -> &AsrProcessor {
        &self.processor
    }

    pub fn model(&self) -> &crate::model::AsrModel {
        &self.model
    }

    pub fn transcribe(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<Vec<AsrTranscription>> {
        crate::inference::transcribe::transcribe(
            &self.model,
            &self.processor,
            &self.device,
            &audio,
            &opts,
        )
    }

    pub fn start_stream(&self, opts: StreamOptions) -> Result<AsrStream<'_>> {
        crate::inference::streaming::start_stream(&self.model, &self.processor, &self.device, &opts)
    }

    pub fn require_ready(&self) -> Result<()> {
        self.processor.require_ready()
    }
}
