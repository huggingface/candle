//! Bounded buffered reconstruction for codecs that are not causal streaming models.
use crate::*;

pub struct CodecStream {
    codec: Box<dyn AudioCodec + Send>,
    input: AudioFormat,
    converter: AudioConverter,
    pending: Vec<f32>,
    max_frames: usize,
}
impl CodecStream {
    pub fn new(
        codec: Box<dyn AudioCodec + Send>,
        input: AudioFormat,
        max_frames: usize,
    ) -> Result<Self> {
        if max_frames == 0 {
            return Err(Error::InvalidInput(
                "codec turn budget must be positive".into(),
            ));
        }
        let output = codec.format().audio_format;
        let policy = if output.channels() == 1 {
            ChannelPolicy::Average
        } else {
            ChannelPolicy::Preserve
        };
        let converter = AudioConverter::with_format(input, output, policy)?;
        Ok(Self {
            codec,
            input,
            converter,
            pending: Vec::new(),
            max_frames,
        })
    }
    fn append(&mut self, audio: AudioBuffer) -> Result<()> {
        let channels = audio.format().channels() as usize;
        let len = self
            .pending
            .len()
            .checked_add(audio.samples().len())
            .filter(|n| n / channels <= self.max_frames)
            .ok_or_else(|| {
                Error::LimitExceeded("buffered codec turn exceeds its input budget".into())
            })?;
        self.pending.reserve(len - self.pending.len());
        self.pending.extend_from_slice(audio.samples());
        Ok(())
    }
}
impl RealtimeModel for CodecStream {
    fn capabilities(&self) -> RealtimeCapabilities {
        let output = self.codec.format().audio_format;
        RealtimeCapabilities {
            input_format: Some(self.input),
            output_format: Some(output),
            max_input_frames_per_turn: Some(
                (self.max_frames as u128 * self.input.sample_rate() as u128
                    / output.sample_rate() as u128) as usize,
            ),
            text_input: false,
            text_output: false,
            streaming: StreamingMode::Buffered,
            output_streaming: Some(StreamingMode::Buffered),
        }
    }
    fn descriptor(&self) -> ModelCapabilities {
        let caps = self.capabilities();
        ModelCapabilities {
            model_id: "codec_reconstruction".into(),
            tasks: vec![AudioTask::Codec, AudioTask::Realtime],
            input_formats: vec![self.input],
            output_formats: caps.output_format.into_iter().collect(),
            max_input_frames: caps.max_input_frames_per_turn,
            input_streaming: Some(StreamingMode::Buffered),
            output_streaming: Some(StreamingMode::Buffered),
            ..Default::default()
        }
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        _: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        let converted = self.converter.push(audio)?;
        self.append(converted)
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        cancel.check()?;
        let tail = self.converter.finish()?;
        self.append(tail)?;
        let format = self.codec.format().audio_format;
        let input = AudioBuffer::new(format, std::mem::take(&mut self.pending))?;
        let codes = self.codec.encode_multiscale(&input, cancel)?;
        let output = self.codec.decode_multiscale(&codes, cancel)?;
        for chunk in output.samples().chunks(2048 * format.channels() as usize) {
            cancel.check()?;
            sink.emit(AudioEvent::Audio {
                turn: 0,
                audio: AudioBuffer::new(format, chunk.to_vec())?,
            })?;
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.codec.reset()?;
        self.pending.clear();
        self.converter.reset();
        Ok(())
    }
}
