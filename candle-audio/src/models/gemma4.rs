//! Native Gemma 4 audio-conditioned text, with the checkpoint's audio and text
//! branches. This adapter does not synthesize audio or load a vision tower.
use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::gemma4 as native};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Gemma4 {
    model: native::AudioTextModel,
    tokenizer: Tokenizer,
    device: Device,
    source: AudioFormat,
    position: usize,
    saved: Option<(native::text::TextModel, usize)>,
    max_context: usize,
    max_steps: usize,
    eos: Vec<u32>,
    turn_end: u32,
}
impl Gemma4 {
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
                    "Gemma4 dtype must be f32 or bf16".into(),
                ))
            }
        };
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if json["model_type"] != "gemma4" {
            return Err(Error::Unsupported(
                "expected a multimodal Gemma4 checkpoint".into(),
            ));
        }
        let mut cfg: native::config::Gemma4Config = serde_json::from_value(json.clone())?;
        validate_config(&cfg)?;
        let processor: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("processor_config.json"))?)?;
        validate_processor(&processor)?;
        let max_context = option_usize(options, "max_context", 4096)?;
        let max_steps = option_usize(options, "max_steps", 512)?;
        if max_context == 0
            || max_context > cfg.text_config.max_position_embeddings
            || max_steps >= max_context
            || !(1..=4096).contains(&max_steps)
        {
            return Err(Error::InvalidInput(
                "invalid Gemma4 context/generation budget".into(),
            ));
        }
        cfg.text_config.max_position_embeddings = max_context;
        cfg.text_config.use_flash_attn = false;
        let tokenizer = super::load::tokenizer(directory)?;
        for token in [
            "<bos>",
            "<eos>",
            "<|turn>",
            "<turn|>",
            "<|audio>",
            "<audio|>",
            "<|audio|>",
        ] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(Error::InvalidInput(format!(
                    "Gemma4 tokenizer is missing {token}"
                )));
            }
        }
        if tokenizer.token_to_id("<|audio|>") != Some(cfg.audio_token_id as u32) {
            return Err(Error::InvalidInput(
                "Gemma4 audio token/config mismatch".into(),
            ));
        }
        let turn_end = tokenizer.token_to_id("<turn|>").unwrap();
        let eos = match &json["eos_token_id"] {
            serde_json::Value::Array(ids) => ids
                .iter()
                .map(|id| {
                    id.as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| Error::InvalidInput("invalid Gemma4 EOS token".into()))
                })
                .collect::<Result<Vec<_>>>()?,
            value => vec![value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or_else(|| tokenizer.token_to_id("<eos>").unwrap())],
        };
        // SAFETY: mappings are read-only; checkpoint files must remain unchanged
        // throughout the model's lifetime.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&super::load::weights(directory)?, dtype, device)?
        };
        let model = native::AudioTextModel::new(&cfg, vb)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            source: input_source,
            position: 0,
            saved: None,
            max_context,
            max_steps,
            eos,
            turn_end,
        })
    }
    pub fn into_realtime(self) -> Result<UnderstandingStream<Self>> {
        let source = self.source;
        let steps = self.max_steps;
        UnderstandingStream::new(self, source, source.sample_rate() as usize * 30, steps, 0)
    }
    /// Language context retained by successful understanding requests.
    pub fn context_tokens(&self) -> usize {
        self.position
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::InvalidInput(e.to_string()))?
            .get_ids()
            .to_vec())
    }
}
impl AudioUnderstandingModel for Gemma4 {
    fn understand(
        &mut self,
        request: &AudioUnderstandingRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AudioUnderstandingResult> {
        request.validate()?;
        cancel.check()?;
        let mut features = Vec::new();
        let mut counts = Vec::new();
        let mut total = 0usize;
        if request.audio.len() > 16 {
            return Err(Error::LimitExceeded(
                "Gemma4 request exceeds 16 audio clips".into(),
            ));
        }
        for audio in &request.audio {
            if audio.duration() > 30. {
                return Err(Error::LimitExceeded(
                    "Gemma4 audio exceeds 30 seconds".into(),
                ));
            }
            let audio = AudioConverter::convert(audio, 16000)?;
            total = total
                .checked_add(audio.frames())
                .filter(|n| *n <= 480000)
                .ok_or_else(|| {
                    Error::LimitExceeded("Gemma4 total request audio exceeds 30 seconds".into())
                })?;
            let mel = extract_features(&audio, &self.device, cancel)?;
            cancel.check()?;
            let encoded = self.model.encode_audio(&mel.mel, &mel.padding_mask)?;
            counts.push(encoded.dim(0)?);
            features.push(encoded);
        }
        let features = Tensor::cat(&features, 0)?;
        let mut tokens = self.encode(if self.position == 0 {
            "<bos><|turn>user\n"
        } else {
            "<|turn>user\n"
        })?;
        for count in counts {
            tokens.extend(self.encode("<|audio>")?);
            tokens.extend(std::iter::repeat_n(
                self.model.cfg.audio_token_id as u32,
                count,
            ));
            tokens.extend(self.encode("<audio|>")?);
        }
        tokens.extend(self.encode(&request.prompt)?);
        tokens.extend(self.encode("<turn|>\n<|turn>model\n")?);
        let ending = self.encode("<turn|>\n")?;
        if self
            .position
            .checked_add(tokens.len())
            .and_then(|n| n.checked_add(request.max_steps + ending.len()))
            .is_none_or(|n| n > self.max_context)
        {
            return Err(Error::LimitExceeded(
                "Gemma4 conversation context is full; reset the conversation".into(),
            ));
        }
        let snapshot = (self.model.language_model.clone(), self.position);
        self.saved = Some(snapshot.clone());
        let result = (|| {
            let mut logits = self.model.forward_with_audio_embeddings(
                &Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?,
                &features,
                self.position,
            )?;
            self.position += tokens.len();
            let mut sampler = LogitsProcessor::new(request.seed, Some(0.), None);
            let mut generated = Vec::new();
            let mut reason = FinishReason::Length;
            for _ in 0..request.max_steps {
                cancel.check()?;
                let token = sampler.sample(&logits.i((0, 0))?.to_dtype(DType::F32)?)?;
                if self.eos.contains(&token) || token == self.turn_end {
                    reason = FinishReason::EndOfSequence;
                    break;
                }
                generated.push(token);
                logits = self
                    .model
                    .forward(&Tensor::new(&[[token]], &self.device)?, self.position)?;
                self.position += 1;
            }
            cancel.check()?;
            // Close retained assistant context even when the response hit its
            // budget. The returned reason still records truncation explicitly.
            self.model.forward(
                &Tensor::new(ending.as_slice(), &self.device)?.unsqueeze(0)?,
                self.position,
            )?;
            self.position += ending.len();
            let text = self
                .tokenizer
                .decode(&generated, true)
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
            self.model.language_model = snapshot.0;
            self.position = snapshot.1;
            self.saved = None;
        }
        result
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "gemma4".into(),
            tasks: vec![AudioTask::Understanding, AudioTask::Recognition],
            input_formats: vec![AudioFormat::mono(16000).unwrap()],
            max_input_frames: Some(480000),
            max_text_bytes: Some(16384),
            max_context_tokens: Some(self.max_context),
            input_streaming: Some(StreamingMode::Buffered),
            preserves_conversation: true,
            ..Default::default()
        }
    }
    fn reset(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        self.position = 0;
        self.saved = None;
        Ok(())
    }
    fn end_turn(&mut self) -> Result<()> {
        self.saved = None;
        Ok(())
    }
    fn cancel_turn(&mut self) -> Result<()> {
        if let Some((model, position)) = self.saved.take() {
            self.model.language_model = model;
            self.position = position;
        }
        Ok(())
    }
}
impl AsrModel for Gemma4 {
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
        if options.translate || options.language.is_some() {
            return Err(Error::Unsupported(
                "use a Gemma4 audio understanding prompt for translation or language conditioning"
                    .into(),
            ));
        }
        if audio.format() != self.input_format() {
            return Err(Error::InvalidInput(
                "Gemma4 ASR expects mono 16 kHz audio".into(),
            ));
        }
        AudioUnderstandingModel::reset(self)?;
        let mut request = AudioUnderstandingRequest::new(vec![audio.clone()], "Transcribe the speech verbatim. Return only the transcription, in the original language.");
        request.max_steps = self.max_steps;
        let result = self.understand(&request, cancel, &mut |_| Ok(()));
        AudioUnderstandingModel::reset(self)?;
        let result = result?;
        Ok(TranscriptionResult {
            transcript: Transcript {
                text: result.text,
                start: 0.,
                end: audio.duration(),
            },
            finish_reason: result.finish_reason,
            language: None,
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

pub fn validate_config(cfg: &native::config::Gemma4Config) -> Result<()> {
    let Some(audio) = &cfg.audio_config else {
        return Err(Error::Unsupported(
            "Gemma4 checkpoint has no audio encoder".into(),
        ));
    };
    if audio.input_feat_size != 128
        || audio.hidden_size == 0
        || !audio.hidden_size.is_multiple_of(2)
        || audio.conf_num_attention_heads == 0
        || !audio
            .hidden_size
            .is_multiple_of(audio.conf_num_attention_heads)
        || audio.conf_attention_chunk_size == 0
        || audio.conf_attention_context_left < 2
        || audio.conf_attention_logit_cap <= 0.
        || audio.gradient_clipping <= 0.
        || audio.conf_conv_kernel_size == 0
        || audio.output_proj_dims == Some(0)
        || audio.conf_reduction_factor != 1
        || audio.sscp_conv_channel_size.len() != 2
        || audio.sscp_conv_channel_size.contains(&0)
        || audio.sscp_conv_kernel_size != [[3, 3], [3, 3]]
        || audio.sscp_conv_stride_size != [[2, 2], [2, 2]]
        || cfg.text_config.enable_moe_block
        || cfg
            .text_config
            .layer_types
            .iter()
            .any(|kind| !["full_attention", "sliding_attention"].contains(&kind.as_str()))
        || cfg.text_config.pad_token_id >= cfg.text_config.vocab_size
        || cfg.text_config.hidden_size_per_layer_input > 0
            && cfg.text_config.pad_token_id >= cfg.text_config.vocab_size_per_layer_input
    {
        return Err(Error::Unsupported(
            "unsupported or inconsistent Gemma4 audio architecture".into(),
        ));
    }
    Ok(())
}
pub fn validate_processor(json: &serde_json::Value) -> Result<()> {
    let f = &json["feature_extractor"];
    for (key, expected) in [
        ("feature_size", 128.),
        ("sampling_rate", 16000.),
        ("fft_length", 512.),
        ("frame_length", 320.),
        ("hop_length", 160.),
        ("min_frequency", 0.),
        ("max_frequency", 8000.),
        ("preemphasis", 0.),
        ("dither", 0.),
        ("input_scale_factor", 1.),
        ("mel_floor", 0.001),
        ("padding_value", 0.),
    ] {
        if f[key]
            .as_f64()
            .is_none_or(|value| (value - expected).abs() > 1e-9)
        {
            return Err(Error::Unsupported(format!(
                "unsupported Gemma4 feature extractor {key}; expected {expected}"
            )));
        }
    }
    if f["fft_overdrive"].as_bool() != Some(false)
        || !f["per_bin_mean"].is_null()
        || !f["per_bin_stddev"].is_null()
    {
        return Err(Error::Unsupported(
            "unsupported Gemma4 FFT/normalization preprocessing".into(),
        ));
    }
    Ok(())
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

pub struct AudioFeatures {
    /// `(1, time, 128)` natural-log HTK magnitude mel features.
    pub mel: Tensor,
    /// `(1, time)`, with 1 for padding and 0 for valid frames.
    pub padding_mask: Tensor,
}

/// Gemma4AudioFeatureExtractor's semicausal 20 ms Hann windows, 10 ms
/// hop, 512-point magnitude FFT and unnormalized HTK mel filters. This differs
/// from Whisper's power spectrogram, centering and log10 normalization.
pub fn extract_features(
    audio: &AudioBuffer,
    device: &Device,
    cancel: &CancellationToken,
) -> Result<AudioFeatures> {
    if audio.format() != AudioFormat::mono(16000)?
        || audio.frames() <= 160
        || audio.frames() > 480000
    {
        return Err(Error::InvalidInput(
            "Gemma4 features require mono 16 kHz audio with 161..=480000 frames".into(),
        ));
    }
    cancel.check()?;
    let padded = audio.frames().div_ceil(128) * 128;
    let frames = (padded + 160 - 321) / 160 + 1;
    let window: Vec<f32> = (0..320)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 320.).cos()) as f32)
        .collect();
    let max_mel = 2595. * (1f64 + 8000. / 700.).log10();
    let hz: Vec<f64> = (0..130)
        .map(|i| 700. * (10f64.powf((i as f64 * max_mel / 129.) / 2595.) - 1.))
        .collect();
    let mut filters = vec![0f64; 128 * 257];
    for bin in 0..257 {
        let frequency = bin as f64 * 8000. / 256.;
        for mel in 0..128 {
            filters[mel * 257 + bin] = ((frequency - hz[mel]) / (hz[mel + 1] - hz[mel]))
                .min((hz[mel + 2] - frequency) / (hz[mel + 2] - hz[mel + 1]))
                .max(0.);
        }
    }
    let mut mel = Vec::with_capacity(frames * 128);
    let mut mask = Vec::with_capacity(frames);
    for frame in 0..frames {
        cancel.check()?;
        let valid = frame * 160 + 160 < audio.frames();
        mask.push(if valid { 0f32 } else { 1. });
        if !valid {
            mel.extend(std::iter::repeat_n(0., 128));
            continue;
        }
        let mut real = vec![0f64; 512];
        let mut imag = vec![0f64; 512];
        for i in 0..320 {
            let position = frame * 160 + i;
            if position >= 160 && position - 160 < audio.frames() {
                real[i] = (audio.samples()[position - 160] * window[i]) as f64;
            }
        }
        fft(&mut real, &mut imag);
        let magnitude: Vec<f64> = real[..257]
            .iter()
            .zip(&imag[..257])
            .map(|(re, im)| re.hypot(*im))
            .collect();
        for filter in filters.as_chunks::<257>().0 {
            let energy: f64 = filter
                .iter()
                .zip(&magnitude)
                .map(|(weight, value)| weight * value)
                .sum();
            mel.push((energy + 0.001).ln() as f32);
        }
    }
    Ok(AudioFeatures {
        mel: Tensor::from_vec(mel, (1, frames, 128), device)?,
        padding_mask: Tensor::from_vec(mask, (1, frames), device)?,
    })
}
fn fft(real: &mut [f64], imag: &mut [f64]) {
    let n = real.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -std::f64::consts::TAU / len as f64;
        let (sin, cos) = angle.sin_cos();
        for start in (0..n).step_by(len) {
            let (mut wr, mut wi) = (1., 0.);
            for offset in 0..len / 2 {
                let a = start + offset;
                let b = a + len / 2;
                let tr = wr * real[b] - wi * imag[b];
                let ti = wr * imag[b] + wi * real[b];
                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;
                (wr, wi) = (wr * cos - wi * sin, wr * sin + wi * cos);
            }
        }
        len *= 2;
    }
}
