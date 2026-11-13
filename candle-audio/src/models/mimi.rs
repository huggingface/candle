use crate::*;
use candle::{Device, IndexOp, Tensor};
use candle_transformers::models::mimi as m;
use std::collections::VecDeque;

/// Native streaming neural-codec reconstruction. This adapter demonstrates
/// causal audio-in/audio-out models; it is a codec, not a conversational model.
/// Input conversion and the final incomplete codec frame are flushed at commit.
pub struct Mimi {
    model: m::Model,
    device: Device,
    format: AudioFormat,
    input_format: AudioFormat,
    converter: AudioConverter,
    frame_size: usize,
    pending: VecDeque<f32>,
    received: usize,
    emitted: usize,
    max_model_frames: usize,
}
impl Mimi {
    pub fn new(mut model: m::Model, device: Device, input_format: AudioFormat) -> Result<Self> {
        let cfg = model.config();
        let frame_size = cfg.sample_rate / cfg.frame_rate;
        if cfg.channels != 1
            || !frame_size.is_finite()
            || frame_size < 1.
            || frame_size.fract() != 0.
            || cfg.sample_rate.fract() != 0.
        {
            return Err(Error::Unsupported(
                "Mimi adapter needs mono audio and an integral codec frame size".into(),
            ));
        }
        let format = AudioFormat::mono(cfg.sample_rate as u32)?;
        let hop = cfg.seanet.ratios.iter().product::<usize>();
        let max_model_frames = cfg
            .transformer
            .max_seq_len
            .checked_mul(hop)
            .and_then(|n| n.checked_sub(frame_size as usize * 5))
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                Error::InvalidInput("Mimi context is too short for the codec flush".into())
            })?;
        let converter = AudioConverter::new(input_format, format.sample_rate())?;
        model.reset_state();
        Ok(Self {
            model,
            device,
            format,
            input_format,
            converter,
            frame_size: frame_size as usize,
            pending: VecDeque::new(),
            received: 0,
            emitted: 0,
            max_model_frames,
        })
    }
    fn frame(
        &mut self,
        pcm: Vec<f32>,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        let pcm = Tensor::from_vec(pcm, (1, 1, self.frame_size), &self.device)?;
        let codes = self.model.encode_step(&pcm.into())?;
        cancel.check()?;
        let pcm = self.model.decode_step(&codes)?;
        if let Some(pcm) = pcm.as_option() {
            let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
            let available = self.received.saturating_sub(self.emitted).min(pcm.len());
            if available > 0 {
                cancel.check()?;
                sink.emit(AudioEvent::Audio {
                    turn: 0,
                    audio: AudioBuffer::new(self.format, pcm[..available].to_vec())?,
                })?;
                self.emitted += available;
            }
        }
        Ok(())
    }
    fn accept(
        &mut self,
        pcm: AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        let received = self
            .received
            .checked_add(pcm.frames())
            .ok_or_else(|| Error::LimitExceeded("codec duration overflow".into()))?;
        if received > self.max_model_frames {
            return Err(Error::LimitExceeded(
                "Mimi turn exceeds its rotary-position context; commit and start a new turn".into(),
            ));
        }
        self.received = received;
        for part in pcm.samples().chunks(self.frame_size) {
            self.pending.extend(part);
            while self.pending.len() >= self.frame_size {
                let frame = self.pending.drain(..self.frame_size).collect();
                self.frame(frame, cancel, sink)?;
            }
        }
        Ok(())
    }
}
impl RealtimeModel for Mimi {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            input_format: Some(self.input_format),
            output_format: Some(self.format),
            max_input_frames_per_turn: Some(
                (self.max_model_frames as u128 * self.input_format.sample_rate() as u128
                    / self.format.sample_rate() as u128) as usize,
            ),
            text_input: false,
            text_output: false,
            streaming: StreamingMode::Native,
            output_streaming: Some(StreamingMode::Native),
        }
    }
    fn push_audio(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        let pcm = self.converter.push(audio)?;
        self.accept(pcm, cancel, sink)
    }
    fn commit(&mut self, cancel: &CancellationToken, sink: &mut dyn EventSink) -> Result<()> {
        cancel.check()?;
        let tail = self.converter.finish()?;
        self.accept(tail, cancel, sink)?;
        if !self.pending.is_empty() {
            let mut frame: Vec<_> = self.pending.drain(..).collect();
            frame.resize(self.frame_size, 0.);
            self.frame(frame, cancel, sink)?;
        }
        // Causal convolutions may still hold a short tail. Bound the drain so
        // a misconfigured model cannot loop indefinitely or invent duration.
        for _ in 0..4 {
            if self.emitted >= self.received {
                break;
            }
            self.frame(vec![0.; self.frame_size], cancel, sink)?;
        }
        if self.emitted != self.received {
            return Err(Error::InvalidState(
                "codec failed to flush its audio tail".into(),
            ));
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset_state();
        self.converter.reset();
        self.pending.clear();
        self.received = 0;
        self.emitted = 0;
        Ok(())
    }
}
