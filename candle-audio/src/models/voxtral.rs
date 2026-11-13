//! Native Voxtral 2507 audio-conditioned text generation. This is the original
//! encoder/decoder architecture, not Voxtral Realtime or Voxtral TTS.
use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::voxtral as native};
use std::{collections::BTreeMap, path::Path};
use tekken::{SpecialTokenPolicy, Tekkenizer};

pub struct Voxtral {
    model: native::VoxtralForConditionalGeneration,
    tokenizer: Tekkenizer,
    specials: BTreeMap<String, u32>,
    cache: native::VoxtralCache,
    saved: Option<(native::VoxtralCache, usize)>,
    position: usize,
    device: Device,
    source: AudioFormat,
    max_context: usize,
    max_steps: usize,
    filters: Vec<f32>,
}
impl Voxtral {
    pub fn load(
        directory: &Path,
        device: &Device,
        options: &serde_json::Value,
        input_source: AudioFormat,
    ) -> Result<Self> {
        let dtype = match options.get("dtype").and_then(serde_json::Value::as_str) {
            None if options.get("dtype").is_none() => DType::F32,
            Some("f32" | "float32") => DType::F32,
            Some("bf16" | "bfloat16") => DType::BF16,
            _ => {
                return Err(Error::Unsupported(
                    "Voxtral dtype must be f32 or bf16".into(),
                ))
            }
        };
        Self::load_with_dtype(directory, device, options, input_source, dtype)
    }
    pub fn load_with_dtype(
        directory: &Path,
        device: &Device,
        options: &serde_json::Value,
        input_source: AudioFormat,
        dtype: DType,
    ) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::BF16) {
            return Err(Error::Unsupported(
                "Voxtral supports F32 or BF16 weights".into(),
            ));
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if metadata["model_type"] != "voxtral" {
            return Err(Error::Unsupported(
                "expected a Voxtral 2507 checkpoint".into(),
            ));
        }
        let mut config: native::VoxtralConfig = serde_json::from_value(metadata)?;
        let context = option_usize(options, "max_context", 4096)?;
        let max_steps = option_usize(options, "max_steps", 512)?;
        if context == 0
            || context > config.text_config.max_position_embeddings
            || !(1..=4096).contains(&max_steps)
            || max_steps >= context
        {
            return Err(Error::InvalidInput(
                "invalid Voxtral context/generation budget".into(),
            ));
        }
        validate_config(&config)?;
        let processor: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("preprocessor_config.json"))?)?;
        validate_processor(&processor)?;
        config.text_config.max_position_embeddings = context;
        config.text_config.use_flash_attn = false;
        let tokenizer_path = directory.join("tekken.json");
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&tokenizer_path)?)?;
        let specials: BTreeMap<String, u32> = json["special_tokens"]
            .as_array()
            .ok_or_else(|| Error::InvalidInput("tekken.json is missing special_tokens".into()))?
            .iter()
            .map(|token| {
                let text = token["token_str"]
                    .as_str()
                    .ok_or_else(|| Error::InvalidInput("invalid Tekken special token".into()))?;
                let rank = token["rank"]
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        Error::InvalidInput("invalid Tekken special token rank".into())
                    })?;
                Ok((text.to_owned(), rank))
            })
            .collect::<Result<_>>()?;
        for token in [
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[BEGIN_AUDIO]",
            "[AUDIO]",
            "[TRANSCRIBE]",
        ] {
            if !specials.contains_key(token) {
                return Err(Error::InvalidInput(format!(
                    "Tekken tokenizer is missing {token}"
                )));
            }
        }
        if specials["[AUDIO]"] as usize != config.audio_token_id {
            return Err(Error::InvalidInput(
                "Voxtral tokenizer/config audio token mismatch".into(),
            ));
        }
        if specials
            .values()
            .any(|&id| id as usize >= config.text_config.vocab_size)
        {
            return Err(Error::InvalidInput(
                "Voxtral tokenizer special token exceeds the checkpoint vocabulary".into(),
            ));
        }
        let tokenizer = Tekkenizer::from_file(tokenizer_path)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        // SAFETY: checkpoint mappings are read-only and must remain unchanged
        // during this model's lifetime.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&super::load::weights(directory)?, dtype, device)?
        };
        let model = native::VoxtralForConditionalGeneration::new(&config, vb)?;
        let cache = native::VoxtralCache::new(true, dtype, &config.text_config, device)?;
        Ok(Self {
            model,
            tokenizer,
            specials,
            cache,
            saved: None,
            position: 0,
            device: device.clone(),
            source: input_source,
            max_context: context,
            max_steps,
            filters: super::whisper::Whisper::default_mel_filters(128)?,
        })
    }
    pub fn into_realtime(self) -> Result<UnderstandingStream<Self>> {
        let source = self.source;
        let max_steps = self.max_steps;
        UnderstandingStream::new(
            self,
            source,
            source.sample_rate() as usize * 30,
            max_steps,
            0,
        )
    }
    /// Language context retained by successful understanding requests.
    pub fn context_tokens(&self) -> usize {
        self.position
    }
    fn token(&self, name: &str) -> u32 {
        self.specials[name]
    }
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false, false)
            .map_err(|e| Error::InvalidInput(e.to_string()))
    }
    fn run(
        &mut self,
        request: &AudioUnderstandingRequest,
        transcription: Option<&TranscriptionOptions>,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AudioUnderstandingResult> {
        cancel.check()?;
        let steps = request.max_steps;
        let mut total_frames = 0usize;
        let mut embeddings = Vec::new();
        let mut counts = Vec::new();
        if request.audio.len() > 16 {
            return Err(Error::LimitExceeded(
                "Voxtral request exceeds 16 audio clips".into(),
            ));
        }
        for clip in &request.audio {
            if clip.is_empty() || clip.duration() > 30. {
                return Err(Error::LimitExceeded(
                    "Voxtral clips must contain at most 30 seconds of audio".into(),
                ));
            }
            let pcm = AudioConverter::convert(clip, 16000)?;
            total_frames = total_frames
                .checked_add(pcm.frames())
                .filter(|n| *n <= 480000)
                .ok_or_else(|| {
                    Error::LimitExceeded("Voxtral request audio exceeds 30 seconds".into())
                })?;
            let mut samples = pcm.into_samples();
            samples.resize(480000, 0.);
            cancel.check()?;
            let features = native::extract_features(&samples, &self.filters, &self.device)?;
            let encoded = self.model.get_audio_embeds(&features)?;
            counts.push(encoded.dim(0)?);
            embeddings.push(encoded);
        }
        let embeddings = Tensor::cat(&embeddings, 0)?;
        let mut input = Vec::new();
        if self.position == 0 {
            input.push(self.token("<s>"));
        }
        input.push(self.token("[INST]"));
        for count in counts {
            input.push(self.token("[BEGIN_AUDIO]"));
            input.extend(std::iter::repeat_n(self.token("[AUDIO]"), count));
        }
        if transcription.is_none() {
            input.extend(self.encode_text(&request.prompt)?);
        }
        input.push(self.token("[/INST]"));
        if let Some(options) = transcription {
            if let Some(language) = options.language.as_deref() {
                input.extend(self.encode_text(&format!("lang:{language}"))?);
            }
            input.push(self.token("[TRANSCRIBE]"));
        }
        if self
            .position
            .checked_add(input.len())
            .and_then(|n| n.checked_add(steps + 1))
            .is_none_or(|n| n > self.max_context)
        {
            return Err(Error::LimitExceeded(
                "Voxtral conversation context is full; reset the conversation".into(),
            ));
        }
        let snapshot = (self.cache.clone(), self.position);
        self.saved = Some(snapshot.clone());
        let result = (|| {
            let mut logits = self.model.forward_with_audio_embeds(
                &Tensor::new(input.as_slice(), &self.device)?.unsqueeze(0)?,
                &embeddings,
                &mut self.cache,
                self.position,
            )?;
            self.position += input.len();
            let mut sampler = LogitsProcessor::new(request.seed, Some(0.), None);
            let mut generated = Vec::new();
            let eos = self.token("</s>");
            let mut reason = FinishReason::Length;
            for step in 0..steps {
                cancel.check()?;
                let token = sampler.sample(&logits.i(0)?.to_dtype(DType::F32)?)?;
                if token == eos {
                    reason = FinishReason::EndOfSequence;
                    break;
                }
                generated.push(token);
                // Consume every emitted token, including the last budgeted
                // token, so retained context includes the complete response.
                logits = self.model.forward(
                    &Tensor::new(&[[token]], &self.device)?,
                    None,
                    &mut self.cache,
                    self.position,
                )?;
                self.position += 1;
                if step + 1 == steps {
                    break;
                }
            }
            cancel.check()?;
            self.model.forward(
                &Tensor::new(&[[eos]], &self.device)?,
                None,
                &mut self.cache,
                self.position,
            )?;
            self.position += 1;
            let text = self
                .tokenizer
                .decode(&generated, SpecialTokenPolicy::Ignore)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
            if !text.is_empty() {
                sink.emit(AudioEvent::Text {
                    turn: 0,
                    text: text.clone(),
                })?;
            }
            cancel.check()?;
            Ok(AudioUnderstandingResult {
                text,
                finish_reason: reason,
            })
        })();
        if result.is_err() {
            self.cache = snapshot.0;
            self.position = snapshot.1;
            self.saved = None;
        }
        result
    }
}
impl AudioUnderstandingModel for Voxtral {
    fn understand(
        &mut self,
        request: &AudioUnderstandingRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AudioUnderstandingResult> {
        request.validate()?;
        self.run(request, None, cancel, sink)
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "voxtral".into(),
            tasks: vec![AudioTask::Understanding, AudioTask::Recognition],
            languages: ["en", "es", "fr", "pt", "hi", "de", "nl", "it"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            input_formats: vec![AudioFormat::mono(16000).unwrap()],
            max_input_frames: Some(480000),
            max_text_bytes: Some(16384),
            preserves_conversation: true,
            ..Default::default()
        }
    }
    fn reset(&mut self) -> Result<()> {
        self.cache.reset();
        self.position = 0;
        self.saved = None;
        Ok(())
    }
    fn end_turn(&mut self) -> Result<()> {
        self.saved = None;
        Ok(())
    }
    fn cancel_turn(&mut self) -> Result<()> {
        if let Some((cache, position)) = self.saved.take() {
            self.cache = cache;
            self.position = position;
        }
        Ok(())
    }
}
impl AsrModel for Voxtral {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::mono(16000).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        480000
    }
    fn transcribe(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<Transcript> {
        Ok(self.transcribe_detailed(audio, options, cancel)?.transcript)
    }
    fn transcribe_detailed(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<TranscriptionResult> {
        if options.translate {
            return Err(Error::Unsupported(
                "use Voxtral audio understanding with an explicit translation prompt".into(),
            ));
        }
        if audio.format() != self.input_format()
            || audio.is_empty()
            || audio.frames() > self.max_input_frames()
        {
            return Err(Error::InvalidInput(
                "Voxtral ASR expects nonempty mono 16 kHz audio up to 30 seconds".into(),
            ));
        }
        if options
            .language
            .as_deref()
            .is_some_and(|lang| !["en", "es", "fr", "pt", "hi", "de", "nl", "it"].contains(&lang))
        {
            return Err(Error::Unsupported(
                "unsupported Voxtral transcription language".into(),
            ));
        }
        AudioUnderstandingModel::reset(self)?;
        let request = AudioUnderstandingRequest {
            audio: vec![audio.clone()],
            prompt: String::new(),
            max_steps: self.max_steps,
            seed: 0,
        };
        let result = self.run(&request, Some(options), cancel, &mut |_| Ok(()));
        AudioUnderstandingModel::reset(self)?;
        let result = result?;
        Ok(TranscriptionResult {
            transcript: Transcript {
                text: result.text,
                start: 0.,
                end: audio.duration(),
            },
            finish_reason: result.finish_reason,
            language: options.language.clone(),
        })
    }
    fn reset(&mut self) -> Result<()> {
        AudioUnderstandingModel::reset(self)
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut d = AudioUnderstandingModel::descriptor(self);
        d.tasks = vec![AudioTask::Recognition];
        d.preserves_conversation = false;
        d
    }
}

fn option_usize(options: &serde_json::Value, name: &str, default: usize) -> Result<usize> {
    match options.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::InvalidInput(format!("{name} must be a nonnegative integer"))),
    }
}
pub fn validate_config(cfg: &native::VoxtralConfig) -> Result<()> {
    let a = &cfg.audio_config;
    let t = &cfg.text_config;
    if a.num_mel_bins != 128
        || a.max_source_positions != 1500
        || a.scale_embedding
        || a.activation_function != "gelu"
        || a.hidden_size == 0
        || a.intermediate_size != a.hidden_size.saturating_mul(4)
        || a.num_attention_heads == 0
        || a.num_key_value_heads != a.num_attention_heads
        || !a.hidden_size.is_multiple_of(a.num_attention_heads)
        || a.head_dim != a.hidden_size / a.num_attention_heads
        || t.hidden_size == 0
        || t.num_attention_heads == 0
        || t.num_key_value_heads == 0
        || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
        || !t
            .head_dim
            .unwrap_or(t.hidden_size / t.num_attention_heads)
            .is_multiple_of(2)
        || t.max_position_embeddings == 0
        || t.rope_theta <= 0.
        || t.head_dim.is_none() && !t.hidden_size.is_multiple_of(t.num_attention_heads)
        || cfg.audio_token_id >= t.vocab_size
    {
        return Err(Error::Unsupported(
            "unsupported or inconsistent Voxtral architecture/configuration".into(),
        ));
    }
    Ok(())
}

pub fn validate_processor(json: &serde_json::Value) -> Result<()> {
    for (key, expected) in [
        ("feature_size", 128.),
        ("sampling_rate", 16000.),
        ("n_fft", 400.),
        ("hop_length", 160.),
        ("chunk_length", 30.),
        ("dither", 0.),
        ("padding_value", 0.),
    ] {
        if json[key].as_f64() != Some(expected) {
            return Err(Error::Unsupported(format!(
                "unsupported Voxtral preprocessor {key}; expected {expected}"
            )));
        }
    }
    if json["padding_side"] != "right"
        || json["feature_extractor_type"] != "WhisperFeatureExtractor"
    {
        return Err(Error::Unsupported(
            "Voxtral requires right-padded WhisperFeatureExtractor input".into(),
        ));
    }
    Ok(())
}
