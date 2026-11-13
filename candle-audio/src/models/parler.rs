use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::parler_tts,
};
use tokenizers::Tokenizer;

/// Persistent Parler sampling configuration. The default temperature of zero
/// selects greedy decoding, where the request seed has no effect. A positive
/// temperature enables seeded sampling; top-k/top-p optionally restrict it.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParlerSamplingOptions {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
}
impl ParlerSamplingOptions {
    pub fn from_model_options(options: &serde_json::Value) -> Result<Self> {
        let options = if options.is_null() {
            Self::default()
        } else {
            if !options.is_object() {
                return Err(Error::InvalidInput(
                    "Parler sampling options must be a JSON object".into(),
                ));
            }
            serde_json::from_value(options.clone()).map_err(|error| {
                Error::InvalidInput(format!("invalid Parler sampling options: {error}"))
            })?
        };
        options.validate()?;
        Ok(options)
    }
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0. {
            return Err(Error::InvalidInput(
                "Parler temperature must be finite and nonnegative".into(),
            ));
        }
        if self.top_k == Some(0) {
            return Err(Error::InvalidInput("Parler top_k must be positive".into()));
        }
        if self
            .top_p
            .is_some_and(|p| !p.is_finite() || p <= 0. || p > 1.)
        {
            return Err(Error::InvalidInput(
                "Parler top_p must be finite and in (0, 1]".into(),
            ));
        }
        if self.temperature == 0. && (self.top_k.is_some() || self.top_p.is_some()) {
            return Err(Error::InvalidInput(
                "Parler top_k/top_p require a positive temperature to enable sampling".into(),
            ));
        }
        Ok(())
    }
    /// Build a fresh sampler for an utterance, so earlier requests or resets
    /// cannot advance this request's RNG stream.
    pub fn logits_processor(&self, seed: u64) -> Result<LogitsProcessor> {
        self.validate()?;
        let temperature = self.temperature;
        let sampling = if temperature == 0. {
            Sampling::ArgMax
        } else {
            match (self.top_k, self.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        Ok(LogitsProcessor::from_sampling(seed, sampling))
    }
}

/// Native Parler-TTS adapter. DAC is a non-streaming vocoder here: output chunks
/// are delivered only after synthesis, and capabilities explicitly say Buffered.
pub struct Parler {
    model: parler_tts::Model,
    tokenizer: Tokenizer,
    device: Device,
    format: AudioFormat,
    max_positions: usize,
    output_chunk_frames: usize,
    last_finish: FinishReason,
    sampling_options: ParlerSamplingOptions,
}
impl Parler {
    pub fn new(
        model: parler_tts::Model,
        config: &parler_tts::Config,
        tokenizer: Tokenizer,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            model,
            tokenizer,
            device,
            format: AudioFormat::mono(config.audio_encoder.sampling_rate)?,
            max_positions: config.decoder.max_position_embeddings,
            output_chunk_frames: 2048,
            last_finish: FinishReason::Completed,
            sampling_options: ParlerSamplingOptions::default(),
        })
    }
    pub fn set_output_chunk_frames(&mut self, frames: usize) -> Result<()> {
        if frames == 0 {
            return Err(Error::InvalidInput(
                "output chunk size must be nonzero".into(),
            ));
        }
        self.output_chunk_frames = frames;
        Ok(())
    }
    /// Change sampling for subsequent requests. Invalid options leave the
    /// current configuration intact; synthesis resets retain these options.
    pub fn set_sampling_options(&mut self, options: ParlerSamplingOptions) -> Result<()> {
        options.validate()?;
        self.sampling_options = options;
        Ok(())
    }
    pub fn sampling_options(&self) -> &ParlerSamplingOptions {
        &self.sampling_options
    }
    fn infer(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        if request.text.trim().is_empty() || request.text.len() > 65_536 || request.max_steps == 0 {
            return Err(Error::InvalidInput(
                "TTS needs nonempty text <= 65536 bytes and nonzero max_steps".into(),
            ));
        }
        if request.voice.is_some() || request.language.as_ref().is_some_and(|v| v != "en") {
            return Err(Error::Unsupported(
                "Parler uses descriptive voice conditioning; this adapter supports English".into(),
            ));
        }
        let description = request
            .description
            .as_deref()
            .unwrap_or("A clear female voice in a quiet recording.");
        if description.len() > 65_536 {
            return Err(Error::LimitExceeded(
                "TTS description exceeds 65536 bytes".into(),
            ));
        }
        let encode = |text: &str| -> Result<Tensor> {
            let tokens = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
            Ok(Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?)
        };
        let prompt = encode(&request.text)?;
        let description = encode(description)?;
        if description.dim(1)? > self.max_positions {
            return Err(Error::LimitExceeded(
                "TTS description exceeds conditioning token budget".into(),
            ));
        }
        if prompt.dim(1)?.saturating_add(request.max_steps) > self.max_positions {
            return Err(Error::LimitExceeded(
                "TTS prompt plus generation exceeds decoder context".into(),
            ));
        }
        let lp = self.sampling_options.logits_processor(request.seed)?;
        let codes = self.model.generate_with_callback_detailed(
            &prompt,
            &description,
            lp,
            request.max_steps,
            |_| {
                cancel
                    .check()
                    .map_err(|error| candle::Error::Msg(error.to_string()))
            },
        );
        cancel.check()?;
        let generation = codes?;
        self.last_finish = match generation.finish_reason {
            parler_tts::GenerationFinishReason::EndOfSequence => FinishReason::EndOfSequence,
            parler_tts::GenerationFinishReason::Length => FinishReason::Length,
        };
        let codes = generation.audio_codes;
        if codes.dim(1)? == 0 {
            return Err(Error::InvalidInput("TTS generated no audio codes".into()));
        }
        let codes = codes
            .to_device(&self.device)?
            .to_dtype(DType::I64)?
            .unsqueeze(0)?;
        // Decode with the codec's exact receptive-field halo. Synchronization
        // between bounded chunks makes cancellation effective during vocoding
        // and limits individual GPU workloads for long utterances.
        let decoded = self.model.audio_encoder.decode_codes_chunked_with_callback(
            &codes,
            candle_transformers::models::dac::DEFAULT_DECODE_CHUNK_SIZE,
            |_| {
                cancel
                    .check()
                    .map_err(|error| candle::Error::Msg(error.to_string()))
            },
        );
        cancel.check()?;
        let pcm = decoded?.i((0, 0))?.to_vec1::<f32>()?;
        cancel.check()?;
        for chunk in pcm.chunks(self.output_chunk_frames) {
            cancel.check()?;
            sink.emit(AudioEvent::Audio {
                turn: 0,
                audio: AudioBuffer::new(self.format, chunk.to_vec())?,
            })?;
        }
        Ok(())
    }
}
impl TtsModel for Parler {
    fn finish_reason(&self) -> FinishReason {
        self.last_finish
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "parler_tts".into(),
            tasks: vec![AudioTask::Synthesis],
            languages: vec!["en".into()],
            output_formats: vec![self.output_format()],
            max_text_bytes: Some(65_536),
            ..Default::default()
        }
    }
    fn output_format(&self) -> AudioFormat {
        self.format
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Buffered
    }
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        self.last_finish = FinishReason::Completed;
        let result = self.infer(request, cancel, sink);
        match &result {
            Err(Error::Cancelled) => self.last_finish = FinishReason::Cancelled,
            Err(Error::DeadlineExceeded) => self.last_finish = FinishReason::Deadline,
            _ => {}
        }
        self.reset()?;
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.model.decoder.clear_kv_cache();
        self.model.text_encoder.clear_kv_cache();
        Ok(())
    }
}
