use crate::*;
use candle::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper as w;
use tokenizers::Tokenizer;

pub enum WhisperModel {
    Normal(w::model::Whisper),
    Quantized(w::quantized_model::Whisper),
}
impl WhisperModel {
    pub fn config(&self) -> &w::Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }
    fn encode(&mut self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, true),
            Self::Quantized(m) => m.encoder.forward(x, true),
        }
    }
    fn decode(&mut self, x: &Tensor, audio: &Tensor, first: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, audio, first),
            Self::Quantized(m) => m.decoder.forward(x, audio, first),
        }
    }
    fn logits(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
    fn reset(&mut self) {
        match self {
            Self::Normal(m) => m.reset_kv_cache(),
            Self::Quantized(m) => m.reset_kv_cache(),
        }
    }
}

/// Greedy, segment-level transcription/translation for normal or quantized
/// Whisper weights. Long recordings use `AsrStream` to bound model context.
pub struct Whisper {
    model: WhisperModel,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    device: Device,
    max_tokens: usize,
}
impl Whisper {
    pub fn new(
        model: WhisperModel,
        tokenizer: Tokenizer,
        mel_filters: Vec<f32>,
        device: Device,
    ) -> Result<Self> {
        let cfg = model.config();
        if cfg.num_mel_bins == 0
            || mel_filters.len() != cfg.num_mel_bins * (w::N_FFT / 2 + 1)
            || cfg.max_source_positions == 0
            || cfg.max_source_positions > w::N_FRAMES / 2
            || cfg.max_target_positions < 8
            || mel_filters.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "invalid Whisper filters or context size".into(),
            ));
        }
        for special in [
            w::SOT_TOKEN,
            w::EOT_TOKEN,
            w::NO_TIMESTAMPS_TOKEN,
            w::TRANSCRIBE_TOKEN,
            w::TRANSLATE_TOKEN,
        ] {
            if tokenizer.token_to_id(special).is_none() {
                return Err(Error::InvalidInput(format!(
                    "Whisper tokenizer is missing {special}"
                )));
            }
        }
        Ok(Self {
            model,
            tokenizer,
            mel_filters,
            device,
            max_tokens: 224,
        })
    }
    pub fn default_mel_filters(num_mel_bins: usize) -> Result<Vec<f32>> {
        let bytes = match num_mel_bins {
            80 => include_bytes!("../../assets/melfilters.bytes").as_slice(),
            128 => include_bytes!("../../assets/melfilters128.bytes").as_slice(),
            n => {
                return Err(Error::Unsupported(format!(
                    "no bundled {n}-bin mel filters"
                )))
            }
        };
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    pub fn set_max_tokens(&mut self, max_tokens: usize) -> Result<()> {
        if max_tokens == 0 {
            return Err(Error::InvalidInput("max_tokens must be nonzero".into()));
        }
        self.max_tokens = max_tokens;
        Ok(())
    }
    fn token(&self, name: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(name)
            .ok_or_else(|| Error::Unsupported(format!("missing Whisper token {name}")))
    }
    fn infer(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<TranscriptionResult> {
        cancel.check()?;
        if audio.format() != self.input_format() || audio.frames() > self.max_input_frames() {
            return Err(Error::InvalidInput(
                "Whisper expects mono 16 kHz PCM within its context limit".into(),
            ));
        }
        if audio.is_empty() {
            return Ok(TranscriptionResult {
                transcript: Transcript {
                    text: String::new(),
                    start: 0.,
                    end: 0.,
                },
                finish_reason: FinishReason::Completed,
                language: options.language.clone(),
            });
        }
        let cfg = self.model.config();
        let bins = cfg.num_mel_bins;
        let mel = w::audio::pcm_to_mel(cfg, audio.samples(), &self.mel_filters);
        let frames = mel.len() / bins;
        let mel = Tensor::from_vec(mel, (1, bins, frames), &self.device)?;
        let mel = mel.narrow(2, 0, cfg.max_source_positions * 2)?;
        let encoded = self.model.encode(&mel)?;
        cancel.check()?;
        let mut tokens = vec![self.token(w::SOT_TOKEN)?];
        // English-only checkpoints retain language tokens in their tokenizer.
        // Model vocabulary size, rather than tokenizer contents, distinguishes
        // OpenAI's monolingual checkpoints from multilingual ones.
        let multilingual = self.model.config().vocab_size >= 51_865;
        if !multilingual
            && (options.translate || options.language.as_deref().is_some_and(|s| s != "en"))
        {
            return Err(Error::Unsupported("English-only Whisper supports English transcription, not translation or other languages".into()));
        }
        let mut language = (!multilingual).then(|| "en".to_owned());
        let languages: Vec<(u32, String)> = self
            .tokenizer
            .get_vocab(true)
            .iter()
            .filter_map(|(s, &id)| {
                let code = s.strip_prefix("<|")?.strip_suffix("|>")?;
                ((2..=3).contains(&code.len()) && code.bytes().all(|b| b.is_ascii_lowercase()))
                    .then(|| (id, code.to_owned()))
            })
            .collect();
        if multilingual {
            if let Some(requested) = &options.language {
                tokens.push(self.token(&format!("<|{requested}|>"))?);
                language = Some(requested.clone());
            } else {
                let x = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = self.model.decode(&x, &encoded, true)?;
                let logits = self.model.logits(&hidden)?.i((0, 0))?.to_vec1::<f32>()?;
                let (language_id, code) = languages
                    .into_iter()
                    .filter(|(id, _)| (*id as usize) < logits.len())
                    .max_by(|(a, _), (b, _)| {
                        logits[*a as usize]
                            .total_cmp(&logits[*b as usize])
                            .then_with(|| b.cmp(a))
                    })
                    .ok_or_else(|| {
                        Error::InvalidInput("no language tokens fit model vocabulary".into())
                    })?;
                tokens.push(language_id);
                language = Some(code);
            }
            tokens.push(self.token(if options.translate {
                w::TRANSLATE_TOKEN
            } else {
                w::TRANSCRIBE_TOKEN
            })?);
        }
        tokens.push(self.token(w::NO_TIMESTAMPS_TOKEN)?);
        let eot = self.token(w::EOT_TOKEN)?;
        let prefix_len = tokens.len();
        let cfg = self.model.config();
        let mut suppress = vec![0f32; cfg.vocab_size];
        for &id in &cfg.suppress_tokens {
            if let Some(v) = suppress.get_mut(id as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
        let suppress = Tensor::new(suppress, &self.device)?;
        let limit = self
            .max_tokens
            .min(cfg.max_target_positions.saturating_sub(prefix_len));
        let mut finish_reason = FinishReason::Length;
        for step in 0..limit {
            cancel.check()?;
            let x = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = self.model.decode(&x, &encoded, step == 0)?;
            let logits = self
                .model
                .logits(&hidden.i((.., tokens.len() - 1.., ..))?)?
                .i((0, 0))?;
            let logits = logits.broadcast_add(&suppress)?.to_vec1::<f32>()?;
            let next = logits
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(id, _)| id as u32)
                .ok_or_else(|| {
                    Error::InvalidInput("Whisper produced no finite token logits".into())
                })?;
            if next == eot {
                finish_reason = FinishReason::EndOfSequence;
                break;
            }
            tokens.push(next);
        }
        cancel.check()?;
        let text = self
            .tokenizer
            .decode(&tokens[prefix_len..], true)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(TranscriptionResult {
            transcript: Transcript {
                text: text.trim().to_owned(),
                start: 0.,
                end: audio.duration(),
            },
            finish_reason,
            language,
        })
    }
}
impl AsrModel for Whisper {
    fn descriptor(&self) -> ModelCapabilities {
        let mut languages = if self.model.config().vocab_size < 51_865 {
            vec!["en".to_owned()]
        } else {
            self.tokenizer
                .get_vocab(true)
                .into_iter()
                .filter_map(|(token, id)| {
                    let code = token.strip_prefix("<|")?.strip_suffix("|>")?;
                    ((2..=3).contains(&code.len())
                        && code.bytes().all(|b| b.is_ascii_lowercase())
                        && (id as usize) < self.model.config().vocab_size)
                        .then(|| code.to_owned())
                })
                .collect()
        };
        languages.sort();
        languages.dedup();
        ModelCapabilities {
            model_id: "whisper".into(),
            tasks: vec![AudioTask::Recognition],
            languages,
            input_formats: vec![self.input_format()],
            max_input_frames: Some(self.max_input_frames()),
            max_context_tokens: Some(self.model.config().max_target_positions),
            input_streaming: Some(StreamingMode::Buffered),
            ..Default::default()
        }
    }
    fn input_format(&self) -> AudioFormat {
        AudioFormat::mono(w::SAMPLE_RATE as u32).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        (self.model.config().max_source_positions * 2 * w::HOP_LENGTH).min(w::N_SAMPLES)
    }
    fn transcribe(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<Transcript> {
        self.transcribe_detailed(audio, options, cancel)
            .map(|r| r.transcript)
    }
    fn transcribe_detailed(
        &mut self,
        audio: &AudioBuffer,
        options: &TranscriptionOptions,
        cancel: &CancellationToken,
    ) -> Result<TranscriptionResult> {
        let result = self.infer(audio, options, cancel);
        self.model.reset();
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset();
        Ok(())
    }
}
