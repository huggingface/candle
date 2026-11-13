//! Native Sesame CSM generation with a 32-codebook Mimi codec.
use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{csm, mimi},
};
use std::path::Path;
use tokenizers::Tokenizer;

const MAX_POSITIONS: usize = 2048;
const MAX_TEXT_BYTES: usize = 65_536;

fn speaker(request: &ConditionedSynthesisRequest) -> Result<u32> {
    let voice = match request.request.voice.as_deref() {
        None => None,
        Some("0") => Some(0),
        Some("1") => Some(1),
        Some(_) => {
            return Err(Error::Unsupported(
                "CSM voice must be speaker ID 0 or 1".into(),
            ))
        }
    };
    if request
        .speaker_id
        .zip(voice)
        .is_some_and(|(id, voice)| id != voice)
    {
        return Err(Error::InvalidInput(
            "CSM voice and speaker_id disagree".into(),
        ));
    }
    let speaker = request.speaker_id.or(voice).unwrap_or(0);
    if speaker > 1 {
        return Err(Error::Unsupported("CSM speaker_id must be 0 or 1".into()));
    }
    Ok(speaker)
}

struct Generation {
    tokens: Tensor,
    mask: Tensor,
    sampler: LogitsProcessor,
    remaining: usize,
    codes: Vec<Vec<u32>>,
    emitted: usize,
    backup: csm::Model,
    backup_pos: usize,
}

/// Text and transcribed speech references are fed to the CSM backbone in order.
/// Successful turns retain their KV cache; cancellation restores the previous
/// committed turn. Mimi decodes each sampled frame while generation is active.
pub struct Csm {
    model: csm::Model,
    config: csm::Config,
    tokenizer: Tokenizer,
    mimi: mimi::Model,
    device: Device,
    format: AudioFormat,
    position: usize,
    pending: Option<Generation>,
    text: String,
    temperature: f64,
    finish: FinishReason,
}

fn validate_config(config: &csm::Config, mimi: &mimi::Config) -> Result<()> {
    if config.audio_num_codebooks != 32
        || mimi.quantizer_n_q != 32
        || mimi.channels != 1
        || mimi.sample_rate != 24_000.
        || mimi.frame_rate != 12.5
        || config.audio_vocab_size < mimi.quantizer_bins
        || config.text_vocab_size == 0
    {
        return Err(Error::Unsupported(
            "CSM requires Mimi 24kHz mono, 12.5Hz, 32 codebooks and compatible vocabulary".into(),
        ));
    }
    Ok(())
}

fn validate_request(request: &ConditionedSynthesisRequest) -> Result<()> {
    request.validate()?;
    let r = &request.request;
    if r.text.trim().is_empty() || r.text.len() > MAX_TEXT_BYTES || r.max_steps == 0 {
        return Err(Error::InvalidInput(
            "CSM needs nonempty text <=65536 bytes and nonzero max_steps".into(),
        ));
    }
    if r.description.is_some()
        || request.speaker_embedding.is_some()
        || r.language
            .as_deref()
            .is_some_and(|language| language != "en")
    {
        return Err(Error::Unsupported(
            "CSM uses speaker_id and transcribed audio references; this adapter supports English"
                .into(),
        ));
    }
    speaker(request)?;
    if request
        .references
        .iter()
        .any(|r| r.speaker_id.unwrap_or(0) > 1)
    {
        return Err(Error::Unsupported("CSM speaker_id must be 0 or 1".into()));
    }
    let mut bytes = r.text.len();
    let mut seconds = 0.;
    for reference in &request.references {
        let transcript = reference
            .transcript
            .as_ref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::InvalidInput("CSM audio references require a nonempty transcript".into())
            })?;
        bytes = bytes
            .checked_add(transcript.len())
            .ok_or_else(|| Error::LimitExceeded("CSM reference text overflow".into()))?;
        seconds += reference.audio.frames() as f64 / reference.audio.format().sample_rate() as f64;
    }
    if bytes > MAX_TEXT_BYTES || seconds > 30. {
        return Err(Error::LimitExceeded(
            "CSM reference budget is 65536 total text bytes and 30 seconds of audio".into(),
        ));
    }
    Ok(())
}

impl Csm {
    pub fn new(
        model: csm::Model,
        config: csm::Config,
        tokenizer: Tokenizer,
        mimi: mimi::Model,
        device: Device,
    ) -> Result<Self> {
        validate_config(&config, mimi.config())?;
        Ok(Self {
            model,
            config,
            tokenizer,
            mimi,
            device,
            format: AudioFormat::mono(24000)?,
            position: 0,
            pending: None,
            text: String::new(),
            temperature: 0.7,
            finish: FinishReason::Completed,
        })
    }

    /// Local layout: config.json, tokenizer.json, model.safetensors (or shards),
    /// and mimi/{config.json,model.safetensors}. `mimi_directory` may override it.
    /// The tokenizer must be the Llama 3.2 tokenizer used by this checkpoint.
    pub fn load(directory: &Path, device: &Device, options: &serde_json::Value) -> Result<Self> {
        let config: csm::Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let mimi_directory = options
            .get("mimi_directory")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    Error::InvalidInput("mimi_directory must be a path string".into())
                })
            })
            .transpose()?
            .map(|path| directory.join(path))
            .unwrap_or_else(|| directory.join("mimi"));
        let mimi_config =
            super::load::mimi_config(&mimi_directory, &serde_json::json!({"num_codebooks":32}))?;
        validate_config(&config, &mimi_config)?;
        let temperature = options
            .get("temperature")
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite() && *value > 0.)
                    .ok_or_else(|| {
                        Error::InvalidInput("temperature must be positive and finite".into())
                    })
            })
            .transpose()?
            .unwrap_or(0.7);
        let tokenizer = super::load::tokenizer(directory)?;
        // SAFETY: checkpoint files must remain unchanged while their read-only mappings are live.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &super::load::weights(directory)?,
                DType::F32,
                device,
            )?
        };
        let model = csm::Model::new(&config, vb)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &super::load::weights(&mimi_directory)?,
                DType::F32,
                device,
            )?
        };
        let mimi = mimi::Model::new(mimi_config, vb)?;
        let mut adapter = Self::new(model, config, tokenizer, mimi, device.clone())?;
        adapter.temperature = temperature;
        Ok(adapter)
    }

    fn text_tokens(&self, text: &str, speaker: u32, budget: usize) -> Result<(Tensor, Tensor)> {
        let text = format!("[{speaker}]{text}<|end_of_text|>");
        let ids = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        if ids.len() > budget {
            return Err(Error::LimitExceeded(
                "CSM text exceeds remaining conversation context".into(),
            ));
        }
        if ids.is_empty()
            || ids
                .get_ids()
                .iter()
                .any(|id| *id as usize >= self.config.text_vocab_size)
        {
            return Err(Error::InvalidInput(
                "CSM tokenizer produced empty or out-of-vocabulary text".into(),
            ));
        }
        Ok(self.model.text_tokens_and_mask(ids.get_ids())?)
    }

    /// Begin a request without running the language model. Call poll_output to
    /// generate at most one Mimi frame per poll (plus terminal cache finalization).
    pub fn start_conditioned(
        &mut self,
        request: &ConditionedSynthesisRequest,
        cancel: &CancellationToken,
    ) -> Result<()> {
        cancel.check()?;
        validate_request(request)?;
        if self.pending.is_some() {
            return Err(Error::InvalidState(
                "CSM generation is already active".into(),
            ));
        }
        let mut remaining = MAX_POSITIONS
            .checked_sub(self.position)
            .and_then(|n| n.checked_sub(request.request.max_steps))
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                Error::LimitExceeded(
                    "CSM output budget exceeds remaining conversation context".into(),
                )
            })?;
        let mut tokens = Vec::new();
        let mut masks = Vec::new();
        for reference in &request.references {
            let (t, m) = self.text_tokens(
                reference.transcript.as_deref().unwrap(),
                reference.speaker_id.unwrap_or(0),
                remaining,
            )?;
            remaining -= t.dim(1)?;
            tokens.push(t);
            masks.push(m);
            let audio = AudioConverter::convert(&reference.audio, self.format.sample_rate())?;
            let frame_size = 1920;
            let padding = audio.frames().div_ceil(frame_size) * frame_size - audio.frames();
            let pcm = Tensor::new(audio.samples(), &self.device)?
                .reshape((1, 1, audio.frames()))?
                .pad_with_zeros(2, 0, padding)?;
            self.mimi.reset_state();
            cancel.check()?;
            let codes = self.mimi.encode(&pcm)?.i(0)?.t()?.to_vec2::<u32>()?;
            self.mimi.reset_state();
            cancel.check()?;
            remaining = remaining.checked_sub(codes.len() + 1).ok_or_else(|| {
                Error::LimitExceeded(
                    "CSM audio reference exceeds remaining conversation context".into(),
                )
            })?;
            for frame in codes.into_iter().chain(std::iter::once(vec![0; 32])) {
                let (t, m) = self.model.audio_tokens_and_mask(frame)?;
                tokens.push(t);
                masks.push(m);
            }
        }
        let (t, m) = self.text_tokens(&request.request.text, speaker(request)?, remaining)?;
        tokens.push(t);
        masks.push(m);
        let tokens = Tensor::cat(&tokens, 1)?;
        let mask = Tensor::cat(&masks, 1)?;
        let required = self
            .position
            .checked_add(tokens.dim(1)?)
            .and_then(|n| n.checked_add(request.request.max_steps));
        if required.is_none_or(|n| n > MAX_POSITIONS) {
            return Err(Error::LimitExceeded("CSM request and output budget exceed the remaining 2048-position conversation context; reset conversation or lower max_steps".into()));
        }
        self.mimi.reset_state();
        self.pending = Some(Generation {
            tokens,
            mask,
            sampler: LogitsProcessor::new(request.request.seed, Some(self.temperature), None),
            remaining: request.request.max_steps,
            codes: Vec::new(),
            emitted: 0,
            backup: self.model.clone(),
            backup_pos: self.position,
        });
        self.finish = FinishReason::Completed;
        Ok(())
    }

    fn step(
        &mut self,
        generation: &mut Generation,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<bool> {
        cancel.check()?;
        let frame = self.model.generate_frame_with_callback(
            &generation.tokens,
            &generation.mask,
            self.position,
            &mut generation.sampler,
            |_| {
                cancel
                    .check()
                    .map_err(|error| candle::Error::Msg(error.to_string()))
            },
        );
        cancel.check()?;
        let frame = frame?;
        self.position += generation.tokens.dim(1)?;
        generation.remaining -= 1;
        let eos = frame.iter().all(|code| *code == 0);
        let (tokens, mask) = self.model.audio_tokens_and_mask(frame.clone())?;
        generation.tokens = tokens;
        generation.mask = mask;
        cancel.check()?;
        if !eos {
            if frame
                .iter()
                .any(|code| *code as usize >= self.mimi.config().quantizer_bins)
            {
                return Err(Error::InvalidState(
                    "CSM generated an invalid Mimi code".into(),
                ));
            }
            generation.codes.push(frame.clone());
            let codes = Tensor::new(frame.as_slice(), &self.device)?.reshape((1, 32, 1))?;
            let pcm = self.mimi.decode_step(&codes.into())?;
            cancel.check()?;
            if let Some(pcm) = pcm.as_option() {
                let samples = pcm.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let available = generation.codes.len() * 1920 - generation.emitted;
                let samples = &samples[..samples.len().min(available)];
                if !samples.is_empty() {
                    sink.emit(AudioEvent::Audio {
                        turn: 0,
                        audio: AudioBuffer::new(self.format, samples.to_vec())?,
                    })?;
                    generation.emitted += samples.len();
                }
            }
        }
        let done = eos || generation.remaining == 0;
        if done {
            self.finish = if eos {
                FinishReason::EndOfSequence
            } else {
                FinishReason::Length
            };
            // Feed the final sampled frame so it remains in the next turn's context.
            cancel.check()?;
            let result = self.model.generate_frame_with_callback(
                &generation.tokens,
                &generation.mask,
                self.position,
                &mut generation.sampler,
                |_| {
                    cancel
                        .check()
                        .map_err(|error| candle::Error::Msg(error.to_string()))
                },
            );
            cancel.check()?;
            result?;
            self.position += 1;
            cancel.check()?;
            // Causal convolutions may retain a final tail. Recover exactly that
            // tail from the full decoder, without emitting invented padding audio.
            let expected = generation.codes.len() * 1920;
            if generation.emitted < expected {
                self.mimi.reset_state();
                let codes = Tensor::new(generation.codes.clone(), &self.device)?
                    .t()?
                    .unsqueeze(0)?;
                let pcm = self
                    .mimi
                    .decode(&codes)?
                    .i((0, 0))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                if pcm.len() < expected {
                    return Err(Error::InvalidState(
                        "Mimi decoder did not produce the expected CSM duration".into(),
                    ));
                }
                cancel.check()?;
                sink.emit(AudioEvent::Audio {
                    turn: 0,
                    audio: AudioBuffer::new(
                        self.format,
                        pcm[generation.emitted..expected].to_vec(),
                    )?,
                })?;
            }
            self.mimi.reset_state();
        }
        Ok(done)
    }

    fn rollback(&mut self, generation: Generation) {
        self.model = generation.backup;
        self.position = generation.backup_pos;
        self.mimi.reset_state();
    }

    pub fn context_positions(&self) -> usize {
        self.position
    }
    fn description(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "csm".into(),
            tasks: vec![AudioTask::Synthesis, AudioTask::Realtime],
            languages: vec!["en".into()],
            voices: vec!["0".into(), "1".into()],
            output_formats: vec![self.format],
            max_text_bytes: Some(MAX_TEXT_BYTES),
            accepts_audio_reference: true,
            preserves_conversation: true,
            independent_output: true,
            ..Default::default()
        }
    }
}

impl TtsModel for Csm {
    fn output_format(&self) -> AudioFormat {
        self.format
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Native
    }
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        self.synthesize_conditioned(
            &ConditionedSynthesisRequest::new(request.clone()),
            cancel,
            sink,
        )
    }
    fn synthesize_conditioned(
        &mut self,
        request: &ConditionedSynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        self.start_conditioned(request, cancel)?;
        while self.pending.is_some() {
            RealtimeModel::poll_output(self, cancel, sink)?;
        }
        Ok(())
    }
    fn finish_reason(&self) -> FinishReason {
        self.finish
    }
    fn reset(&mut self) -> Result<()> {
        RealtimeModel::reset(self)
    }
    fn descriptor(&self) -> ModelCapabilities {
        self.description()
    }
}

impl RealtimeModel for Csm {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: None,
            output_format: Some(self.format),
            max_input_frames_per_turn: None,
            text_input: true,
            text_output: false,
            streaming: StreamingMode::Native,
            output_streaming: Some(StreamingMode::Native),
        }
    }
    fn push_audio(
        &mut self,
        _audio: &AudioBuffer,
        _cancel: &CancellationToken,
        _sink: &mut dyn EventSink,
    ) -> Result<()> {
        Err(Error::Unsupported("CSM requires audio references paired with transcripts; use start_conditioned or synthesize_conditioned".into()))
    }
    fn push_text(
        &mut self,
        text: &str,
        cancel: &CancellationToken,
        _sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        if self.pending.is_some() {
            return Err(Error::InvalidState("CSM generation is active".into()));
        }
        if self.text.len().saturating_add(text.len()) > MAX_TEXT_BYTES {
            return Err(Error::LimitExceeded("CSM text exceeds 65536 bytes".into()));
        }
        self.text.push_str(text);
        Ok(())
    }
    fn commit(&mut self, cancel: &CancellationToken, _sink: &mut dyn EventSink) -> Result<()> {
        let request = ConditionedSynthesisRequest::new(SynthesisRequest::new(self.text.clone()));
        self.start_conditioned(&request, cancel)?;
        self.text.clear();
        Ok(())
    }
    fn output_pending(&self) -> bool {
        self.pending.is_some()
    }
    fn poll_output(
        &mut self,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<PollStatus> {
        let Some(mut generation) = self.pending.take() else {
            cancel.check()?;
            return Ok(PollStatus::Complete);
        };
        match self.step(&mut generation, cancel, sink) {
            Ok(true) => Ok(PollStatus::Complete),
            Ok(false) => {
                self.pending = Some(generation);
                Ok(PollStatus::Pending)
            }
            Err(error) => {
                self.finish = match error {
                    Error::Cancelled => FinishReason::Cancelled,
                    Error::DeadlineExceeded => FinishReason::Deadline,
                    _ => self.finish,
                };
                self.rollback(generation);
                Err(error)
            }
        }
    }
    fn end_turn(&mut self) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::InvalidState(
                "CSM output must be drained before ending turn".into(),
            ));
        }
        self.text.clear();
        Ok(())
    }
    fn cancel_turn(&mut self) -> Result<()> {
        if let Some(generation) = self.pending.take() {
            self.rollback(generation);
        }
        self.text.clear();
        self.finish = FinishReason::Cancelled;
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.pending = None;
        self.position = 0;
        self.text.clear();
        self.model.clear_kv_cache();
        self.mimi.reset_state();
        self.finish = FinishReason::Completed;
        Ok(())
    }
    fn reset_conversation(&mut self) -> Result<()> {
        RealtimeModel::reset(self)
    }
    fn finish_reason(&self) -> FinishReason {
        self.finish
    }
    fn descriptor(&self) -> ModelCapabilities {
        self.description()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csm_requires_the_32_codebook_mimi_variant() -> Result<()> {
        let config = csm::Config {
            audio_num_codebooks: 32,
            audio_vocab_size: 2051,
            backbone_flavor: csm::Flavor::Llama1B,
            decoder_flavor: csm::Flavor::Llama100M,
            text_vocab_size: 128256,
        };
        validate_config(&config, &mimi::Config::v0_1(Some(32)))?;
        assert!(matches!(
            validate_config(&config, &mimi::Config::v0_1(Some(16))),
            Err(Error::Unsupported(_))
        ));
        let mut stereo = mimi::Config::v0_1(Some(32));
        stereo.channels = 2;
        assert!(matches!(
            validate_config(&config, &stereo),
            Err(Error::Unsupported(_))
        ));
        Ok(())
    }
    #[test]
    fn reference_contract_requires_transcripts_and_known_speakers() -> Result<()> {
        let mut request = ConditionedSynthesisRequest::new(SynthesisRequest::new("hello"));
        request.references.push(AudioReference {
            audio: AudioBuffer::new(AudioFormat::mono(24000)?, vec![0.; 1920])?,
            transcript: None,
            speaker_id: Some(0),
        });
        assert!(matches!(
            validate_request(&request),
            Err(Error::InvalidInput(_))
        ));
        request.references[0].transcript = Some("reference".into());
        validate_request(&request)?;
        request.speaker_id = Some(2);
        assert!(matches!(
            validate_request(&request),
            Err(Error::Unsupported(_))
        ));
        Ok(())
    }
    #[test]
    fn speaker_embedding_is_explicitly_unsupported() {
        let mut request = ConditionedSynthesisRequest::new(SynthesisRequest::new("hello"));
        request.speaker_embedding = Some(vec![0.; 256]);
        assert!(matches!(
            validate_request(&request),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn advertised_voices_select_speaker_tokens() -> Result<()> {
        let mut request = ConditionedSynthesisRequest::new(SynthesisRequest::new("hello"));
        request.request.voice = Some("1".into());
        validate_request(&request)?;
        assert_eq!(speaker(&request)?, 1);
        request.speaker_id = Some(0);
        assert!(matches!(
            validate_request(&request),
            Err(Error::InvalidInput(_))
        ));
        Ok(())
    }
}
