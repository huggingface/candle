use crate::{Error, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::VecDeque;

/// Interleaved f32 PCM. Sample rates and channel counts cannot be zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawFormat")]
pub struct AudioFormat {
    sample_rate: u32,
    channels: u16,
}

#[derive(Deserialize)]
struct RawFormat {
    sample_rate: u32,
    channels: u16,
}
impl TryFrom<RawFormat> for AudioFormat {
    type Error = Error;
    fn try_from(v: RawFormat) -> Result<Self> {
        Self::new(v.sample_rate, v.channels)
    }
}
impl AudioFormat {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if !(1..=384_000).contains(&sample_rate) || !(1..=64).contains(&channels) {
            return Err(Error::InvalidInput(
                "expected 1..384000 Hz and 1..64 channels".into(),
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
        })
    }
    pub fn mono(sample_rate: u32) -> Result<Self> {
        Self::new(sample_rate, 1)
    }
    pub fn sample_rate(self) -> u32 {
        self.sample_rate
    }
    pub fn channels(self) -> u16 {
        self.channels
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AudioBuffer {
    format: AudioFormat,
    samples: Vec<f32>,
}
impl<'de> Deserialize<'de> for AudioBuffer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            format: AudioFormat,
            samples: Vec<f32>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.format, raw.samples).map_err(serde::de::Error::custom)
    }
}
impl AudioBuffer {
    pub fn new(format: AudioFormat, samples: Vec<f32>) -> Result<Self> {
        if !samples.len().is_multiple_of(format.channels as usize)
            || samples.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidInput(
                "PCM must contain complete interleaved frames and finite samples".into(),
            ));
        }
        Ok(Self { format, samples })
    }
    pub fn format(&self) -> AudioFormat {
        self.format
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
    pub fn frames(&self) -> usize {
        self.samples.len() / self.format.channels as usize
    }
    pub fn duration(&self) -> f64 {
        self.frames() as f64 / self.format.sample_rate as f64
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Explicit channel conversion. Source channel numbering starts at zero.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelPolicy {
    Average,
    Preserve,
    Select(u16),
    Map(Vec<u16>),
}

const MAX_RESAMPLER_BLOCK_SAMPLES: usize = 1 << 20;
const MAX_CONVERSION_SAMPLES: usize = 8 << 20;
const MAX_RESAMPLE_RATIO: u32 = 256;
const SINC_LEN: usize = 256;
const SINC_OVERSAMPLING: usize = 128;
// rubato 0.16 uses floor(block * ratio + 10); retain one extra frame
// before construction and verify output_frames_max again before processing.
const OUTPUT_MARGIN: usize = 11;

fn validate_ratio(input: AudioFormat, output: AudioFormat) -> Result<()> {
    let (a, b) = (input.sample_rate as u64, output.sample_rate as u64);
    if a > b * MAX_RESAMPLE_RATIO as u64 || b > a * MAX_RESAMPLE_RATIO as u64 {
        return Err(Error::Unsupported(format!("sample-rate conversion ratio must be within 1/{MAX_RESAMPLE_RATIO}..={MAX_RESAMPLE_RATIO}")));
    }
    Ok(())
}
fn expected_frames(input_frames: u64, input: AudioFormat, output: AudioFormat) -> Result<u64> {
    let frames = (input_frames as u128 * output.sample_rate as u128
        + input.sample_rate as u128 / 2)
        / input.sample_rate as u128;
    u64::try_from(frames)
        .map_err(|_| Error::LimitExceeded("converted audio duration overflow".into()))
}
fn call_samples(frames: u128, channels: usize) -> Result<usize> {
    let samples=frames.checked_mul(channels as u128).filter(|n|*n<=MAX_CONVERSION_SAMPLES as u128).ok_or_else(||Error::LimitExceeded(format!("conversion call exceeds {MAX_CONVERSION_SAMPLES} interleaved output samples; use smaller streaming chunks")))?;
    Ok(samples as usize)
}
fn block_samples(frames: usize, channels: usize) -> Result<()> {
    if frames
        .checked_mul(channels)
        .is_none_or(|n| n > MAX_RESAMPLER_BLOCK_SAMPLES)
    {
        return Err(Error::LimitExceeded(
            "resampler block exceeds bounded scratch memory".into(),
        ));
    }
    Ok(())
}

/// Stateful band-limited sample-rate/channel conversion, with exact tail drain.
///
/// Ratios outside 1/256..=256 are rejected before constructing an interpolator;
/// AudioFormat itself still permits synthetic rates such as 1 Hz. Rubato scratch
/// output is limited to 1 Mi interleaved floats per block by adapting the input
/// block size. Each push/finish/convert_to call is limited to 8 Mi output floats
/// (32 MiB PCM). Longer streams remain supported through smaller push calls.
/// Resource-limit rejection occurs before consuming input or changing state.
pub struct AudioConverter {
    input: AudioFormat,
    output: AudioFormat,
    policy: ChannelPolicy,
    resampler: Option<SincFixedIn<f32>>,
    pending: Vec<VecDeque<f32>>,
    input_frames: u64,
    output_frames: u64,
    delay: usize,
    finished: bool,
}
impl AudioConverter {
    /// Maximum interleaved PCM samples returned by one conversion call.
    pub const MAX_OUTPUT_SAMPLES_PER_CALL: usize = MAX_CONVERSION_SAMPLES;
    /// Compatibility constructor: average all input channels into mono.
    pub fn new(input: AudioFormat, output_rate: u32) -> Result<Self> {
        Self::with_format(
            input,
            AudioFormat::mono(output_rate)?,
            ChannelPolicy::Average,
        )
    }
    pub fn with_format(
        input: AudioFormat,
        output: AudioFormat,
        policy: ChannelPolicy,
    ) -> Result<Self> {
        let valid = match &policy {
            ChannelPolicy::Average => output.channels == 1,
            ChannelPolicy::Preserve => output.channels == input.channels,
            ChannelPolicy::Select(channel) => output.channels == 1 && *channel < input.channels,
            ChannelPolicy::Map(channels) => {
                channels.len() == output.channels as usize
                    && channels.iter().all(|channel| *channel < input.channels)
            }
        };
        if !valid {
            return Err(Error::InvalidInput(
                "channel policy does not match input/output formats".into(),
            ));
        }
        validate_ratio(input, output)?;
        let channels = output.channels as usize;
        let resampler = if input.sample_rate == output.sample_rate {
            None
        } else {
            let available = MAX_RESAMPLER_BLOCK_SAMPLES / channels - OUTPUT_MARGIN;
            let block = ((available as u64 * input.sample_rate as u64 / output.sample_rate as u64)
                as usize)
                .min(1024);
            if block == 0 {
                return Err(Error::LimitExceeded(
                    "resampling ratio cannot fit a bounded output block".into(),
                ));
            }
            let output_bound = (block as u64 * output.sample_rate as u64)
                .div_ceil(input.sample_rate as u64) as usize
                + OUTPUT_MARGIN;
            block_samples(output_bound, channels)?;
            // Both constructor allocations are bounded before SincFixedIn::new:
            // the fixed sinc table, and its per-channel history/input buffer.
            block_samples(SINC_LEN * SINC_OVERSAMPLING, 1)?;
            block_samples(block + 2 * SINC_LEN, channels)?;
            let params = SincInterpolationParameters {
                sinc_len: SINC_LEN,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: SINC_OVERSAMPLING,
                window: WindowFunction::BlackmanHarris2,
            };
            let resampler = SincFixedIn::new(
                output.sample_rate as f64 / input.sample_rate as f64,
                1.,
                params,
                block,
                channels,
            )
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
            block_samples(resampler.output_frames_max(), channels)?;
            Some(resampler)
        };
        let delay = resampler.as_ref().map_or(0, |r| r.output_delay());
        let block = resampler.as_ref().map_or(0, |r| r.input_frames_max());
        Ok(Self {
            input,
            output,
            policy,
            resampler,
            pending: (0..channels)
                .map(|_| VecDeque::with_capacity(block))
                .collect(),
            input_frames: 0,
            output_frames: 0,
            delay,
            finished: false,
        })
    }
    pub fn output_format(&self) -> AudioFormat {
        self.output
    }
    fn map_frame(&self, frame: &[f32], out: &mut Vec<f32>) {
        match &self.policy {
            ChannelPolicy::Average => {
                out.push((frame.iter().map(|x| *x as f64).sum::<f64>() / frame.len() as f64) as f32)
            }
            ChannelPolicy::Preserve => out.extend_from_slice(frame),
            ChannelPolicy::Select(channel) => out.push(frame[*channel as usize]),
            ChannelPolicy::Map(channels) => {
                out.extend(channels.iter().map(|channel| frame[*channel as usize]))
            }
        }
    }
    pub fn push(&mut self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        if self.finished {
            return Err(Error::InvalidState("converter was finished".into()));
        }
        if audio.format != self.input {
            return Err(Error::InvalidInput(
                "audio format changed mid-stream".into(),
            ));
        }
        let channels = self.output.channels as usize;
        let next_input = self
            .input_frames
            .checked_add(audio.frames() as u64)
            .ok_or_else(|| Error::LimitExceeded("audio duration overflow".into()))?;
        expected_frames(next_input, self.input, self.output)?;
        let maximum = if let Some(resampler) = &self.resampler {
            block_samples(resampler.output_frames_max(), channels)?;
            let pending = self.pending[0]
                .len()
                .checked_add(audio.frames())
                .ok_or_else(|| Error::LimitExceeded("resampler input count overflow".into()))?;
            let blocks = pending / resampler.input_frames_next();
            (blocks as u128 * resampler.output_frames_max() as u128)
                .saturating_sub(self.delay as u128)
        } else {
            audio.frames() as u128
        };
        let capacity = call_samples(maximum, channels)?;
        self.output_frames
            .checked_add(
                u64::try_from(maximum)
                    .map_err(|_| Error::LimitExceeded("converted duration overflow".into()))?,
            )
            .ok_or_else(|| Error::LimitExceeded("converted duration overflow".into()))?;
        // All budget checks above precede any input consumption/state changes.
        let mut output = Vec::with_capacity(capacity);
        self.input_frames = next_input;
        if self.resampler.is_none() {
            for frame in audio.samples.chunks_exact(self.input.channels as usize) {
                self.map_frame(frame, &mut output);
            }
            self.output_frames += audio.frames() as u64;
            return AudioBuffer::new(self.output, output);
        }
        let block = self.resampler.as_ref().unwrap().input_frames_next();
        let mut mapped = Vec::with_capacity(channels);
        for frame in audio.samples.chunks_exact(self.input.channels as usize) {
            mapped.clear();
            self.map_frame(frame, &mut mapped);
            for (channel, value) in self.pending.iter_mut().zip(&mapped) {
                channel.push_back(*value);
            }
            if self.pending[0].len() == block {
                self.process_block(false, None, &mut output)?;
            }
        }
        AudioBuffer::new(self.output, output)
    }
    fn process_block(
        &mut self,
        partial: bool,
        limit: Option<usize>,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let resampler = self.resampler.as_mut().unwrap();
        let channels = self.output.channels as usize;
        // process() allocates output_frames_max per channel, including its margin.
        block_samples(resampler.output_frames_max(), channels)?;
        let maximum = resampler
            .output_frames_max()
            .saturating_sub(self.delay)
            .min(limit.unwrap_or(usize::MAX));
        call_samples(
            (output.len() / channels) as u128 + maximum as u128,
            channels,
        )?;
        let block = resampler.input_frames_next();
        block_samples(block, channels)?;
        let input: Vec<Vec<f32>> = self
            .pending
            .iter_mut()
            .map(|channel| {
                let n = channel.len().min(block);
                let mut values: Vec<_> = channel.drain(..n).collect();
                if partial {
                    values.resize(block, 0.);
                }
                values
            })
            .collect();
        let result = resampler
            .process(&input, None)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let skip = self.delay.min(result[0].len());
        self.delay -= skip;
        let produced = (result[0].len() - skip).min(limit.unwrap_or(usize::MAX));
        for frame in skip..skip + produced {
            for channel in &result {
                output.push(channel[frame]);
            }
        }
        self.output_frames = self
            .output_frames
            .checked_add(produced as u64)
            .ok_or_else(|| Error::LimitExceeded("converted duration overflow".into()))?;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<AudioBuffer> {
        if self.finished {
            return Err(Error::InvalidState("converter was finished".into()));
        }
        let expected = expected_frames(self.input_frames, self.input, self.output)?;
        let remaining = expected.checked_sub(self.output_frames).ok_or_else(|| {
            Error::InvalidState("resampler exceeded expected audio duration".into())
        })?;
        let capacity = call_samples(remaining as u128, self.output.channels as usize)?;
        if let Some(resampler) = &self.resampler {
            block_samples(resampler.output_frames_max(), self.output.channels as usize)?;
        }
        let mut output = Vec::with_capacity(capacity);
        if self.resampler.is_some() {
            while self.output_frames < expected {
                self.process_block(
                    true,
                    Some((expected - self.output_frames) as usize),
                    &mut output,
                )?;
            }
        }
        self.finished = true;
        AudioBuffer::new(self.output, output)
    }
    pub fn reset(&mut self) {
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
            self.delay = resampler.output_delay();
        }
        for channel in &mut self.pending {
            channel.clear();
        }
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
    }
    pub fn convert(audio: &AudioBuffer, output_rate: u32) -> Result<AudioBuffer> {
        Self::convert_to(
            audio,
            AudioFormat::mono(output_rate)?,
            ChannelPolicy::Average,
        )
    }
    pub fn convert_to(
        audio: &AudioBuffer,
        output: AudioFormat,
        policy: ChannelPolicy,
    ) -> Result<AudioBuffer> {
        validate_ratio(audio.format, output)?;
        let expected = expected_frames(audio.frames() as u64, audio.format, output)?;
        let capacity = call_samples(expected as u128, output.channels as usize)?;
        let mut converter = Self::with_format(audio.format, output, policy)?;
        let first = converter.push(audio)?;
        let tail = converter.finish()?;
        let mut samples = Vec::with_capacity(capacity);
        samples.extend(first.into_samples());
        samples.extend(tail.into_samples());
        AudioBuffer::new(output, samples)
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn extreme_ratios_are_rejected_without_constructing_a_resampler() {
        let tiny = AudioFormat::mono(1).unwrap();
        let huge = AudioFormat::new(384000, 64).unwrap();
        assert!(matches!(
            AudioConverter::with_format(tiny, huge, ChannelPolicy::Map(vec![0; 64])),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            AudioConverter::with_format(huge, tiny, ChannelPolicy::Average),
            Err(Error::Unsupported(_))
        ));
        assert!(AudioConverter::with_format(tiny, tiny, ChannelPolicy::Preserve).is_ok());
        assert!(AudioConverter::new(tiny, 256).is_ok());
        assert!(AudioConverter::new(AudioFormat::mono(256).unwrap(), 1).is_ok());
        assert!(AudioConverter::new(tiny, 257).is_err());
    }
    #[test]
    fn adaptive_blocks_bound_real_rubato_output_and_constructor_history() {
        for (input, output) in [
            (1000, 256000),
            (8000, 384000),
            (384000, 8000),
            (48000, 16000),
            (256, 1),
        ] {
            let input = AudioFormat::new(input, 64).unwrap();
            let output = AudioFormat::new(output, 64).unwrap();
            let converter =
                AudioConverter::with_format(input, output, ChannelPolicy::Preserve).unwrap();
            let resampler = converter.resampler.as_ref().unwrap();
            assert!(resampler.output_frames_max() * 64 <= MAX_RESAMPLER_BLOCK_SAMPLES);
            assert!(
                (resampler.input_frames_max() + 2 * SINC_LEN) * 64 <= MAX_RESAMPLER_BLOCK_SAMPLES
            );
            assert!(converter.pending.iter().all(|channel| channel.is_empty()));
            if output.sample_rate() / input.sample_rate() >= 48 {
                assert!(resampler.input_frames_max() < 1024);
            }
        }
    }
    #[test]
    fn oversized_upsampling_push_and_one_shot_reject_before_mutation() {
        let input = AudioFormat::mono(1000).unwrap();
        let output = AudioFormat::new(256000, 64).unwrap();
        // Four KiB of input would expand to 64 MiB. Never execute that process.
        let audio = AudioBuffer::new(input, vec![0.; 1024]).unwrap();
        let mut converter =
            AudioConverter::with_format(input, output, ChannelPolicy::Map(vec![0; 64])).unwrap();
        let delay = converter.delay;
        assert!(matches!(
            converter.push(&audio),
            Err(Error::LimitExceeded(_))
        ));
        assert_eq!(converter.input_frames, 0);
        assert_eq!(converter.output_frames, 0);
        assert_eq!(converter.delay, delay);
        assert!(converter.pending.iter().all(VecDeque::is_empty));
        assert!(matches!(
            AudioConverter::convert_to(&audio, output, ChannelPolicy::Map(vec![0; 64])),
            Err(Error::LimitExceeded(_))
        ));
    }
    #[test]
    fn mapping_only_budget_rejection_is_recoverable_and_finish_is_guarded() {
        let input = AudioFormat::mono(10).unwrap();
        let output = AudioFormat::new(10, 64).unwrap();
        let mut converter =
            AudioConverter::with_format(input, output, ChannelPolicy::Map(vec![0; 64])).unwrap();
        let audio = AudioBuffer::new(input, vec![0.; MAX_CONVERSION_SAMPLES / 64 + 1]).unwrap();
        assert!(matches!(
            converter.push(&audio),
            Err(Error::LimitExceeded(_))
        ));
        let small = converter
            .push(&AudioBuffer::new(input, vec![0.25, 0.5]).unwrap())
            .unwrap();
        assert_eq!(small.frames(), 2);
        assert_eq!(small.samples().len(), 128);
        let chunk = AudioBuffer::new(input, vec![0.25; 8192]).unwrap();
        for _ in 0..17 {
            assert_eq!(converter.push(&chunk).unwrap().frames(), 8192);
        }
        assert!(converter.output_frames as usize * 64 > MAX_CONVERSION_SAMPLES);
        assert!(converter.finish().unwrap().is_empty());
        // Fault-inject an excessive undrained tail to exercise finish's preflight
        // without allocating a huge input or asking rubato to process it.
        let mut converter = AudioConverter::new(input, 20).unwrap();
        converter.input_frames = MAX_CONVERSION_SAMPLES as u64;
        let delay = converter.delay;
        assert!(matches!(converter.finish(), Err(Error::LimitExceeded(_))));
        assert_eq!(converter.delay, delay);
        assert!(!converter.finished);
        assert_eq!(converter.output_frames, 0);
    }
    #[test]
    fn irregular_synthetic_rates_preserve_exact_tail_and_pending_is_bounded() {
        for (input_rate, output_rate) in [(1, 10), (10, 1), (8000, 384000), (384000, 8000)] {
            let format = AudioFormat::mono(input_rate).unwrap();
            let output = AudioFormat::mono(output_rate).unwrap();
            let samples = (0..137)
                .map(|i| (i as f32 * 0.031).sin())
                .collect::<Vec<_>>();
            let audio = AudioBuffer::new(format, samples).unwrap();
            let expected =
                AudioConverter::convert_to(&audio, output, ChannelPolicy::Preserve).unwrap();
            let mut converter =
                AudioConverter::with_format(format, output, ChannelPolicy::Preserve).unwrap();
            let mut actual = Vec::new();
            for chunk in audio.samples().chunks(7) {
                actual.extend(
                    converter
                        .push(&AudioBuffer::new(format, chunk.to_vec()).unwrap())
                        .unwrap()
                        .into_samples(),
                );
                assert!(
                    converter.pending[0].len()
                        < converter.resampler.as_ref().unwrap().input_frames_max()
                );
            }
            actual.extend(converter.finish().unwrap().into_samples());
            assert_eq!(actual, expected.samples());
            assert_eq!(
                expected.frames(),
                ((137u64 * output_rate as u64 + input_rate as u64 / 2) / input_rate as u64)
                    as usize
            );
        }
    }
}
