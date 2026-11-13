use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::parler_tts};
use tokenizers::Tokenizer;

/// Native Parler-TTS adapter. DAC is a non-streaming vocoder here: output chunks
/// are delivered only after synthesis, and capabilities explicitly say Buffered.
pub struct Parler {
    model: parler_tts::Model,
    tokenizer: Tokenizer,
    device: Device,
    format: AudioFormat,
    max_positions: usize,
    output_chunk_frames: usize,
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
        let encode = |text: &str| -> Result<Tensor> {
            let tokens = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
            Ok(Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?)
        };
        let prompt = encode(&request.text)?;
        let description = encode(description)?;
        if prompt.dim(1)?.saturating_add(request.max_steps) > self.max_positions {
            return Err(Error::LimitExceeded(
                "TTS prompt plus generation exceeds decoder context".into(),
            ));
        }
        let lp = LogitsProcessor::new(request.seed, Some(0.), None);
        let codes =
            self.model
                .generate_with_callback(&prompt, &description, lp, request.max_steps, |_| {
                    if cancel.is_cancelled() {
                        candle::bail!("audio generation cancelled");
                    }
                    Ok(())
                });
        cancel.check()?;
        let codes = codes?;
        if codes.dim(1)? == 0 {
            return Err(Error::InvalidInput("TTS generated no audio codes".into()));
        }
        let codes = codes
            .to_device(&self.device)?
            .to_dtype(DType::I64)?
            .unsqueeze(0)?;
        let pcm = self
            .model
            .audio_encoder
            .decode_codes(&codes)?
            .i((0, 0))?
            .to_vec1::<f32>()?;
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
        let result = self.infer(request, cancel, sink);
        self.reset()?;
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.model.decoder.clear_kv_cache();
        self.model.text_encoder.clear_kv_cache();
        Ok(())
    }
}
