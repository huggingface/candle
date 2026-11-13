//! Native LFM3-Audio ASR, synthesis and causal-prefill speech sessions.
//! No external inference process is used by this adapter.
use crate::*;
use candle::{Device, Tensor};
use candle_transformers::models::{lfm3::CacheState, lfm3_audio as native};
use std::{path::Path, sync::Arc};
use tokenizers::Tokenizer;

const LANGUAGES: &[(&str, &str)] = &[
    ("en", "American English"),
    ("es", "Spanish"),
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

const DEFAULT_AUDIO_PREFILL_FRAMES: usize = 32_000;
const MAX_AUDIO_PREFILL_FRAMES: usize = 480_000;

fn audio_prefill_frames(options: &serde_json::Value) -> Result<usize> {
    if !options.is_null() && !options.is_object() {
        return Err(Error::InvalidInput("LFM3 options must be an object".into()));
    }
    let frames = match options.get("audio_prefill_frames") {
        None => DEFAULT_AUDIO_PREFILL_FRAMES,
        Some(value) => value.as_u64().and_then(|n|usize::try_from(n).ok())
            .ok_or_else(|| Error::InvalidInput("audio_prefill_frames must be an integer in1..480000, measured in mono16kHz PCM frames".into()))?,
    };
    if !(1..=MAX_AUDIO_PREFILL_FRAMES).contains(&frames) {
        return Err(Error::InvalidInput(
            "audio_prefill_frames must be in1..480000 mono16kHz PCM frames".into(),
        ));
    }
    Ok(frames)
}

/// Only the unencoded PCM tail is queued. Complete batches are immediately
/// prefetched; native AudioStream still owns its unchanged causal prefix state.
struct AudioPrefillQueue {
    frames: usize,
    samples: Vec<f32>,
    finished: bool,
}
impl AudioPrefillQueue {
    fn new(frames: usize) -> Self {
        Self {
            frames,
            samples: Vec::with_capacity(frames),
            finished: false,
        }
    }
    fn push(
        &mut self,
        mut samples: &[f32],
        cancel: &CancellationToken,
        emit: &mut impl FnMut(&[f32], bool) -> Result<()>,
    ) -> Result<()> {
        if self.finished {
            return Err(Error::InvalidState("LFM3 audio prefill is finished".into()));
        }
        cancel.check()?;
        while !samples.is_empty() {
            cancel.check()?;
            let count = (self.frames - self.samples.len()).min(samples.len());
            self.samples.extend_from_slice(&samples[..count]);
            samples = &samples[count..];
            if self.samples.len() == self.frames {
                self.flush(false, cancel, emit)?;
            }
        }
        Ok(())
    }
    fn flush(
        &mut self,
        finalize: bool,
        cancel: &CancellationToken,
        emit: &mut impl FnMut(&[f32], bool) -> Result<()>,
    ) -> Result<()> {
        if self.finished {
            return Err(Error::InvalidState("LFM3 audio prefill is finished".into()));
        }
        cancel.check()?;
        if finalize || !self.samples.is_empty() {
            emit(&self.samples, finalize)?;
            self.samples.clear();
        }
        self.finished = finalize;
        cancel.check()
    }
    fn reset(&mut self) {
        self.samples.clear();
        self.finished = false;
    }
}

pub struct Lfm3 {
    model: native::Model,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    source: AudioFormat,
    converter: AudioConverter,
    audio_stream: native::AudioStream,
    audio_prefill: AudioPrefillQueue,
    saved_context: Option<CacheState>,
    started: bool,
    has_audio: bool,
    received: usize,
    text_bytes: usize,
    max_audio_frames: usize,
    max_context: usize,
    generation_steps: usize,
    last_finish: FinishReason,
}
impl Lfm3 {
    pub fn load(
        directory: &Path,
        device: &Device,
        source: AudioFormat,
        options: &serde_json::Value,
    ) -> Result<Self> {
        // Reject bad batching options before opening checkpoint artifacts.
        let audio_prefill_frames = audio_prefill_frames(options)?;
        let cfg = native::Config::from_json(&std::fs::read(directory.join("config.json"))?)?;
        let tokenizer = Arc::new(super::load::tokenizer(directory)?);
        for token in [
            "<|startoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|audio|>",
            "<|audio_start|>",
            "<|mixed_start|>",
        ] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(Error::InvalidInput(format!(
                    "LFM3 tokenizer is missing {token}"
                )));
            }
        }
        let max_context = options
            .get("max_context_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        let generation_steps = options
            .get("max_steps")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        if max_context == 0
            || max_context > cfg.text_config.max_position_embeddings
            || generation_steps == 0
            || generation_steps >= max_context
        {
            return Err(Error::InvalidInput(
                "invalid LFM3 context/generation budget".into(),
            ));
        }
        let model_path = directory.join(
            options
                .get("model_file")
                .and_then(|v| v.as_str())
                .unwrap_or("model.gguf"),
        );
        let projector_path = directory.join(
            options
                .get("projector_file")
                .and_then(|v| v.as_str())
                .unwrap_or("mmproj.gguf"),
        );
        let model = native::Model::from_gguf(model_path, projector_path, &cfg, device)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            source,
            converter: AudioConverter::new(source, 16000)?,
            audio_stream: native::AudioStream::new(480000)?,
            audio_prefill: AudioPrefillQueue::new(audio_prefill_frames),
            saved_context: None,
            started: false,
            has_audio: false,
            received: 0,
            text_bytes: 0,
            max_audio_frames: 480000,
            max_context,
            generation_steps,
            last_finish: FinishReason::Completed,
        })
    }
    /// A fresh independent session sharing immutable model weights/tokenizer.
    pub fn fork(&self, source: AudioFormat) -> Result<Self> {
        Ok(Self {
            model: self.model.fork(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            source,
            converter: AudioConverter::new(source, 16000)?,
            audio_stream: native::AudioStream::new(self.max_audio_frames)?,
            audio_prefill: AudioPrefillQueue::new(self.audio_prefill.frames),
            saved_context: None,
            started: false,
            has_audio: false,
            received: 0,
            text_bytes: 0,
            max_audio_frames: self.max_audio_frames,
            max_context: self.max_context,
            generation_steps: self.generation_steps,
            last_finish: FinishReason::Completed,
        })
    }
    pub fn context_tokens(&self) -> usize {
        self.model.language.position()
    }
    /// PCM16kHz frames coalesced before realtime acoustic prefill. This does not
    /// change offline ASR/TTS and may only be changed between completed turns.
    pub fn set_audio_prefill_frames(&mut self, frames: usize) -> Result<()> {
        if !(1..=MAX_AUDIO_PREFILL_FRAMES).contains(&frames) {
            return Err(Error::InvalidInput(
                "audio_prefill_frames must be in1..480000".into(),
            ));
        }
        if self.started || !self.audio_prefill.samples.is_empty() {
            return Err(Error::InvalidState(
                "change audio_prefill_frames only between turns".into(),
            ));
        }
        self.audio_prefill = AudioPrefillQueue::new(frames);
        Ok(())
    }
    pub fn audio_prefill_frames(&self) -> usize {
        self.audio_prefill.frames
    }
    pub fn pending_audio_prefill_frames(&self) -> usize {
        self.audio_prefill.samples.len()
    }
    fn embed(&self, text: &str) -> Result<Tensor> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        if ids.is_empty() {
            return Err(Error::InvalidInput("empty LFM3 prompt".into()));
        }
        Ok(self
            .model
            .language
            .embed(&Tensor::new(ids.get_ids(), &self.device)?.unsqueeze(0)?)?)
    }
    fn capacity(&self, rows: usize, reserve: usize) -> Result<()> {
        if self
            .model
            .language
            .position()
            .checked_add(rows)
            .and_then(|v| v.checked_add(reserve))
            .is_none_or(|n| n > self.max_context)
        {
            return Err(Error::LimitExceeded("LFM3 conversation context is full; reset the conversation or select a larger validated context budget".into()));
        }
        Ok(())
    }
    fn task_prefix(system: &str) -> String {
        format!("<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n")
    }
    fn language(code: &str) -> Result<&'static str> {
        LANGUAGES
            .iter()
            .find(|(id, _)| *id == code)
            .map(|(_, name)| *name)
            .ok_or_else(|| Error::Unsupported(format!("unsupported LFM3 task language {code:?}")))
    }
    fn generate(
        &mut self,
        prefill: &Tensor,
        mode: native::GenerationMode,
        steps: usize,
        seed: u64,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<(String, FinishReason)> {
        cancel.check()?;
        self.capacity(prefill.dim(1)?, steps)?;
        let mut text = String::new();
        let mut decoder = self.tokenizer.decode_stream(true);
        let mut callback_error = None;
        let options = native::GenerationOptions {
            mode,
            max_steps: steps,
            seed,
            eos_tokens: vec![
                self.tokenizer.token_to_id("<|im_end|>").unwrap(),
                self.tokenizer
                    .token_to_id("<|endoftext|>")
                    .unwrap_or(124895),
            ],
            ..Default::default()
        };
        let result = self.model.generate(prefill, &options, |event| {
            let output = (|| -> Result<()> {
                cancel.check()?;
                match event {
                    native::GenerationEvent::Token(id) => {
                        if let Some(delta) = decoder
                            .step(id)
                            .map_err(|e| Error::InvalidInput(e.to_string()))?
                        {
                            text.push_str(&delta);
                            sink.emit(AudioEvent::Text {
                                turn: 0,
                                text: delta,
                            })?;
                        }
                    }
                    native::GenerationEvent::Audio(pcm) => sink.emit(AudioEvent::Audio {
                        turn: 0,
                        audio: AudioBuffer::new(AudioFormat::mono(24000)?, pcm)?,
                    })?,
                }
                Ok(())
            })();
            if let Err(error) = output {
                callback_error = Some(error);
                candle::bail!("audio adapter callback stopped generation");
            }
            Ok(())
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        cancel.check()?;
        let reason = match result? {
            native::FinishReason::Eos => FinishReason::EndOfSequence,
            native::FinishReason::Length => FinishReason::Length,
        };
        self.last_finish = reason;
        Ok((text, reason))
    }
    fn clear_turn(&mut self) -> Result<()> {
        self.model.reset_audio();
        self.converter.reset();
        self.audio_stream.reset();
        self.audio_prefill.reset();
        self.started = false;
        self.has_audio = false;
        self.received = 0;
        self.text_bytes = 0;
        self.saved_context = None;
        Ok(())
    }
    fn start_turn(&mut self, cancel: &CancellationToken) -> Result<()> {
        if self.started {
            return Ok(());
        }
        cancel.check()?;
        self.saved_context = Some(self.model.language.snapshot_cache());
        self.model.reset_audio();
        let prefix = if self.model.language.position() == 0 {
            Self::task_prefix("Respond with interleaved text and audio. Keep your answer concise.")
        } else {
            "<|im_start|>user\n".into()
        };
        let embeddings = self.embed(&prefix)?;
        self.capacity(embeddings.dim(1)?, self.generation_steps + 16)?;
        self.model.language.forward_hidden(&embeddings)?;
        self.started = true;
        Ok(())
    }
    fn prefill_queued_audio(
        &mut self,
        pcm: &[f32],
        flush: Option<bool>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let model = &mut self.model;
        let stream = &mut self.audio_stream;
        let reserve = self.generation_steps + 16;
        let maximum = self.max_context;
        let mut prefill = |pcm: &[f32], finalize| {
            cancel.check()?;
            let embedding = model.encode_audio_chunk(stream, pcm, finalize)?;
            let rows = embedding.dim(1)?;
            if model
                .language
                .position()
                .checked_add(rows)
                .and_then(|n| n.checked_add(reserve))
                .is_none_or(|n| n > maximum)
            {
                return Err(Error::LimitExceeded("LFM3 conversation context is full; reset the conversation or select a larger validated context budget".into()));
            }
            if rows > 0 {
                model.language.forward_hidden(&embedding)?;
            }
            cancel.check()
        };
        self.audio_prefill.push(pcm, cancel, &mut prefill)?;
        if let Some(finalize) = flush {
            self.audio_prefill.flush(finalize, cancel, &mut prefill)?;
        }
        Ok(())
    }
    fn descriptor_for(&self, tasks: Vec<AudioTask>, languages: Vec<String>) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "lfm3_audio".into(),
            tasks,
            languages,
            input_formats: vec![AudioFormat::mono(16000).unwrap()],
            output_formats: vec![AudioFormat::mono(24000).unwrap()],
            max_input_frames: Some(self.max_audio_frames),
            max_text_bytes: Some(16384),
            max_context_tokens: Some(self.max_context),
            input_streaming: Some(StreamingMode::Buffered),
            output_streaming: Some(StreamingMode::Native),
            ..Default::default()
        }
    }
}
impl AsrModel for Lfm3 {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::mono(16000).unwrap()
    }
    fn max_input_frames(&self) -> usize {
        self.max_audio_frames
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut d = self.descriptor_for(
            vec![AudioTask::Recognition],
            LANGUAGES.iter().map(|(s, _)| s.to_string()).collect(),
        );
        d.output_formats.clear();
        d.output_streaming = None;
        d
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
        if audio.format() != self.input_format() || audio.frames() > self.max_audio_frames {
            return Err(Error::InvalidInput(
                "LFM3 ASR expects mono16kHz audio up to30 seconds".into(),
            ));
        }
        cancel.check()?;
        self.model.reset();
        let result = (|| {
            let language = options
                .language
                .as_deref()
                .map(Self::language)
                .transpose()?;
            let prompt = if options.translate {
                format!(
                    "Translate the following speech{} into English in text form.",
                    language.map(|l| format!(" from {l}")).unwrap_or_default()
                )
            } else {
                language
                    .map(|l| {
                        format!(
                            "Perform ASR in {}.",
                            if l == "American English" {
                                "English"
                            } else {
                                l
                            }
                        )
                    })
                    .unwrap_or("Perform ASR.".into())
            };
            let before = self.embed(&Self::task_prefix(&prompt))?;
            let acoustic = self.model.encode_audio(audio.samples())?;
            cancel.check()?;
            let after = self.embed("<|im_end|>\n<|im_start|>assistant\n")?;
            let embeddings = Tensor::cat(&[&before, &acoustic, &after], 1)?;
            let (text, finish_reason) = self.generate(
                &embeddings,
                native::GenerationMode::Text,
                self.generation_steps,
                42,
                cancel,
                &mut |_| Ok(()),
            )?;
            Ok(TranscriptionResult {
                transcript: Transcript {
                    text: text.trim().into(),
                    start: 0.,
                    end: audio.duration(),
                },
                finish_reason,
                language: options.language.clone(),
            })
        })();
        self.model.reset();
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset();
        self.clear_turn()
    }
}
impl TtsModel for Lfm3 {
    fn finish_reason(&self) -> FinishReason {
        self.last_finish
    }
    fn output_format(&self) -> AudioFormat {
        AudioFormat::mono(24000).unwrap()
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Native
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut d = self.descriptor_for(
            vec![AudioTask::Synthesis],
            LANGUAGES.iter().map(|(s, _)| s.to_string()).collect(),
        );
        d.input_formats.clear();
        d.input_streaming = None;
        d.voices = vec!["female".into(), "male".into()];
        d
    }
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        if request.text.trim().is_empty()
            || request.text.len() > 16384
            || request.max_steps == 0
            || request.max_steps > 4096
        {
            return Err(Error::InvalidInput(
                "LFM3 TTS requires text up to16384bytes and max_steps in1..4096".into(),
            ));
        }
        if request.description.is_some() {
            return Err(Error::Unsupported("LFM3 TTS uses language and female/male voice conditioning, not descriptive voice prompts".into()));
        }
        let voice = request.voice.as_deref().unwrap_or("female");
        if !["female", "male"].contains(&voice) {
            return Err(Error::Unsupported(format!(
                "unsupported LFM3 voice {voice:?}"
            )));
        }
        let language = Self::language(request.language.as_deref().unwrap_or("en"))?;
        cancel.check()?;
        self.model.reset();
        let result = (|| {
            let prompt = format!(
                "{}{}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>",
                Self::task_prefix(&format!("Perform TTS in {language} using a {voice} voice.")),
                request.text
            );
            let prefill = self.embed(&prompt)?;
            self.generate(
                &prefill,
                native::GenerationMode::Audio,
                request.max_steps,
                request.seed,
                cancel,
                sink,
            )?;
            Ok(())
        })();
        self.model.reset();
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset();
        self.clear_turn()
    }
}
impl RealtimeModel for Lfm3 {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: Some(self.source),
            output_format: Some(AudioFormat::mono(24000).unwrap()),
            max_input_frames_per_turn: Some(self.source.sample_rate() as usize * 30),
            text_input: true,
            text_output: true,
            streaming: StreamingMode::Native,
            output_streaming: Some(StreamingMode::Native),
        }
    }
    fn descriptor(&self) -> ModelCapabilities {
        let mut d = self.descriptor_for(vec![AudioTask::Realtime], vec!["en".into()]);
        d.input_formats = vec![self.source];
        d.max_input_frames = self.capabilities().max_input_frames_per_turn;
        d.input_streaming = Some(StreamingMode::Native);
        d.preserves_conversation = true;
        d
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        _sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        if audio.format() != self.source {
            return Err(Error::InvalidInput(
                "LFM3 source format changed during turn".into(),
            ));
        }
        if audio.is_empty() {
            return Ok(());
        }
        self.start_turn(cancel)?;
        self.has_audio = true;
        let pcm = self.converter.push(audio)?;
        self.received = self
            .received
            .checked_add(pcm.frames())
            .filter(|n| *n <= self.max_audio_frames)
            .ok_or_else(|| Error::LimitExceeded("LFM3 audio exceeds30-second turn limit".into()))?;
        self.prefill_queued_audio(pcm.samples(), None, cancel)
    }
    fn push_text(
        &mut self,
        text: &str,
        cancel: &CancellationToken,
        _sink: &mut dyn EventSink,
    ) -> Result<()> {
        let total = self
            .text_bytes
            .checked_add(text.len())
            .filter(|n| *n <= 16384)
            .ok_or_else(|| Error::LimitExceeded("LFM3 turn text exceeds16384bytes".into()))?;
        if text.trim().is_empty() {
            return Err(Error::InvalidInput("LFM3 text input is empty".into()));
        }
        self.start_turn(cancel)?;
        // Preserve audio/text ordering: queued earlier audio must reach the
        // causal encoder before this text is appended to the language context.
        self.prefill_queued_audio(&[], Some(false), cancel)?;
        let embeddings = self.embed(text)?;
        self.capacity(embeddings.dim(1)?, self.generation_steps + 16)?;
        self.model.language.forward_hidden(&embeddings)?;
        self.text_bytes = total;
        cancel.check()
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        if !self.started {
            return Err(Error::InvalidState("no LFM3 input to commit".into()));
        }
        if self.has_audio {
            let tail = self.converter.finish()?;
            self.received += tail.frames();
            self.prefill_queued_audio(tail.samples(), Some(true), cancel)?;
        }
        let suffix = self.embed("<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>")?;
        self.generate(
            &suffix,
            native::GenerationMode::Interleaved,
            self.generation_steps,
            42,
            cancel,
            sink,
        )?;
        let end = self.embed("<|im_end|>\n")?;
        self.capacity(end.dim(1)?, 0)?;
        self.model.language.forward_hidden(&end)?;
        Ok(())
    }
    fn end_turn(&mut self) -> Result<()> {
        self.clear_turn()
    }
    fn cancel_turn(&mut self) -> Result<()> {
        if let Some(cache) = self.saved_context.take() {
            self.model.language.restore_cache(&cache)?;
        }
        self.clear_turn()
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset();
        self.clear_turn()
    }
    fn reset_conversation(&mut self) -> Result<()> {
        <Self as RealtimeModel>::reset(self)
    }
    fn finish_reason(&self) -> FinishReason {
        self.last_finish
    }
}

#[cfg(test)]
mod prefill_tests {
    use super::*;
    #[test]
    fn prefill_options_reject_invalid_values_before_checkpoint_io() {
        assert_eq!(
            audio_prefill_frames(&serde_json::Value::Null).unwrap(),
            32000
        );
        for value in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(480001),
            serde_json::json!(1.5),
            serde_json::json!("1600"),
            serde_json::Value::Null,
        ] {
            let options = serde_json::json!({"audio_prefill_frames":value});
            assert!(matches!(
                Lfm3::load(
                    Path::new("missing-model"),
                    &Device::Cpu,
                    AudioFormat::mono(16000).unwrap(),
                    &options
                ),
                Err(Error::InvalidInput(_))
            ));
        }
        for n in [1, 1600, 32000, 480000] {
            assert_eq!(
                audio_prefill_frames(&serde_json::json!({"audio_prefill_frames":n})).unwrap(),
                n
            );
        }
    }
    #[test]
    fn irregular_prefill_batches_are_bounded_lossless_and_finalize_once() {
        let mut queue = AudioPrefillQueue::new(4);
        let token = CancellationToken::default();
        let samples = (0..19).map(|n| n as f32).collect::<Vec<_>>();
        let mut calls = vec![];
        let mut emit = |pcm: &[f32], finalize| {
            assert!(pcm.len() <= 4);
            calls.push((pcm.to_vec(), finalize));
            Ok(())
        };
        let mut offset = 0;
        for size in [1, 3, 9, 2, 4] {
            queue
                .push(&samples[offset..offset + size], &token, &mut emit)
                .unwrap();
            offset += size;
            assert!(queue.samples.len() < 4);
        }
        queue.flush(true, &token, &mut emit).unwrap();
        assert!(queue.flush(true, &token, &mut emit).is_err());
        assert!(queue.push(&[99.], &token, &mut emit).is_err());
        assert_eq!(calls.len(), 5);
        assert_eq!(
            calls
                .iter()
                .flat_map(|(pcm, _)| pcm.iter().copied())
                .collect::<Vec<_>>(),
            samples
        );
        assert!(calls[..4]
            .iter()
            .all(|(pcm, finalize)| pcm.len() == 4 && !finalize));
        assert_eq!(calls.last().unwrap(), &(vec![16., 17., 18.], true));
    }
    #[test]
    fn text_boundary_flushes_earlier_audio_without_finalizing_the_stream() {
        let token = CancellationToken::default();
        let mut queue = AudioPrefillQueue::new(4);
        let mut events = vec![];
        queue
            .push(&[1., 2.], &token, &mut |pcm, done| {
                events.push((pcm.to_vec(), done));
                Ok(())
            })
            .unwrap();
        assert!(events.is_empty());
        queue
            .flush(false, &token, &mut |pcm, done| {
                events.push((pcm.to_vec(), done));
                Ok(())
            })
            .unwrap();
        events.push((vec![], false)); // the text insertion occurs after this flush
        queue
            .push(&[3., 4., 5., 6., 7.], &token, &mut |pcm, done| {
                events.push((pcm.to_vec(), done));
                Ok(())
            })
            .unwrap();
        queue
            .flush(true, &token, &mut |pcm, done| {
                events.push((pcm.to_vec(), done));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            events,
            vec![
                (vec![1., 2.], false),
                (vec![], false),
                (vec![3., 4., 5., 6.], false),
                (vec![7.], true)
            ]
        );
    }
    #[test]
    fn cancelled_or_failed_prefill_keeps_a_bounded_tail_until_reset() {
        let mut queue = AudioPrefillQueue::new(4);
        let token = CancellationToken::default();
        let result = queue.push(&[1., 2., 3., 4., 5.], &token, &mut |_, _| {
            token.cancel();
            Err(Error::Cancelled)
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(queue.samples, [1., 2., 3., 4.]);
        assert!(matches!(
            queue.flush(true, &token, &mut |_, _| panic!("cancelled prefill")),
            Err(Error::Cancelled)
        ));
        queue.reset();
        assert!(queue.samples.is_empty() && !queue.finished);
        let fresh = CancellationToken::default();
        queue
            .push(&[9.], &fresh, &mut |_, _| panic!("tail is not full"))
            .unwrap();
        let mut output = vec![];
        queue
            .flush(true, &fresh, &mut |pcm, _| {
                output.extend_from_slice(pcm);
                Ok(())
            })
            .unwrap();
        assert_eq!(output, [9.]);
        queue.reset();
        assert!(queue
            .push(&[0.; 4], &fresh, &mut |_, _| Err(Error::InvalidInput(
                "sink failed".into()
            )))
            .is_err());
        assert_eq!(queue.samples.len(), 4);
        queue.reset();
        assert!(queue.samples.is_empty());
    }
}
