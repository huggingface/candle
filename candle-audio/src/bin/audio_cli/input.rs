//! File PCM readers shared by streaming and bounded CLI tasks.
use anyhow::{Context, Result};
use candle_audio::{AudioBuffer, AudioFormat};
use clap::{Args, ValueEnum};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

pub(super) const MAX_INPUT_SAMPLES: usize = 16_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(super) enum AudioEncoding {
    #[default]
    Wav,
    S16le,
    F32le,
}

#[derive(Clone, Debug, Default, Args)]
pub(super) struct AudioInputOptions {
    /// WAV container, or headerless interleaved little-endian PCM.
    #[arg(long, value_enum, default_value = "wav")]
    pub input_format: AudioEncoding,
    /// Required for raw PCM; WAV uses its header.
    #[arg(long)]
    pub input_sample_rate: Option<u32>,
    /// Required for raw PCM; samples are interleaved by channel.
    #[arg(long)]
    pub input_channels: Option<u16>,
}
impl AudioInputOptions {
    fn raw_format(&self) -> Result<Option<AudioFormat>> {
        if self.input_format == AudioEncoding::Wav {
            anyhow::ensure!(
                self.input_sample_rate.is_none() && self.input_channels.is_none(),
                "WAV uses its header; input-sample-rate/input-channels require a raw input-format"
            );
            return Ok(None);
        }
        let rate = self
            .input_sample_rate
            .context("raw PCM requires --input-sample-rate")?;
        let channels = self
            .input_channels
            .context("raw PCM requires --input-channels")?;
        Ok(Some(AudioFormat::new(rate, channels)?))
    }
}

#[derive(Clone, Debug, Default, Args)]
pub(super) struct ReferenceInputOptions {
    /// Encoding of --audio-reference, independently of primary model options.
    #[arg(long, value_enum)]
    pub reference_format: Option<AudioEncoding>,
    #[arg(long)]
    pub reference_sample_rate: Option<u32>,
    #[arg(long)]
    pub reference_channels: Option<u16>,
}
impl ReferenceInputOptions {
    pub fn validate(&self, has_reference: bool) -> Result<()> {
        anyhow::ensure!(
            has_reference
                || (self.reference_format.is_none()
                    && self.reference_sample_rate.is_none()
                    && self.reference_channels.is_none()),
            "reference-format/reference-sample-rate/reference-channels require --audio-reference"
        );
        if has_reference {
            self.input_options()
                .raw_format()
                .context("audio reference metadata")?;
        }
        Ok(())
    }
    pub fn input_options(&self) -> AudioInputOptions {
        AudioInputOptions {
            input_format: self.reference_format.unwrap_or_default(),
            input_sample_rate: self.reference_sample_rate,
            input_channels: self.reference_channels,
        }
    }
}

enum Reader {
    Wav(hound::WavReader<BufReader<File>>),
    Raw {
        reader: BufReader<File>,
        encoding: AudioEncoding,
        remaining_frames: usize,
    },
}
pub(super) struct AudioSource {
    reader: Reader,
    pub format: AudioFormat,
    total_frames: usize,
}
impl AudioSource {
    pub fn open(path: &Path, options: &AudioInputOptions) -> Result<Self> {
        let raw_format = options.raw_format()?;
        let metadata =
            std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
        anyhow::ensure!(metadata.is_file(), "audio input must be a regular file");
        match raw_format {
            None => {
                let reader = hound::WavReader::open(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                let spec = reader.spec();
                let format = AudioFormat::new(spec.sample_rate, spec.channels)?;
                anyhow::ensure!(
                    match spec.sample_format {
                        hound::SampleFormat::Float => spec.bits_per_sample == 32,
                        hound::SampleFormat::Int => (1..=32).contains(&spec.bits_per_sample),
                    },
                    "unsupported WAV sample representation"
                );
                let total_frames = reader.duration() as usize;
                Ok(Self {
                    reader: Reader::Wav(reader),
                    format,
                    total_frames,
                })
            }
            Some(format) => {
                let width = if options.input_format == AudioEncoding::S16le {
                    2u64
                } else {
                    4
                };
                let frame_bytes = width * format.channels() as u64;
                anyhow::ensure!(
                    metadata.len().is_multiple_of(frame_bytes),
                    "raw PCM contains an incomplete sample or channel frame"
                );
                let total_frames = usize::try_from(metadata.len() / frame_bytes)
                    .context("raw PCM frame count exceeds usize")?;
                let reader = BufReader::new(File::open(path)?);
                Ok(Self {
                    reader: Reader::Raw {
                        reader,
                        encoding: options.input_format,
                        remaining_frames: total_frames,
                    },
                    format,
                    total_frames,
                })
            }
        }
    }
    pub fn total_frames(&self) -> usize {
        self.total_frames
    }
    /// Validate floating-point payloads in bounded chunks before loading weights.
    /// Integer PCM needs no content scan; framing and format are checked by open.
    pub fn preflight(path: &Path, options: &AudioInputOptions) -> Result<()> {
        let mut source = Self::open(path, options)?;
        let floating = match &source.reader {
            Reader::Wav(reader) => reader.spec().sample_format == hound::SampleFormat::Float,
            Reader::Raw { encoding, .. } => *encoding == AudioEncoding::F32le,
        };
        if floating {
            let frames = (8192 / source.format.channels() as usize).max(1);
            while !source.read(frames)?.is_empty() {}
        }
        Ok(())
    }
    pub fn read(&mut self, frames: usize) -> Result<AudioBuffer> {
        let channels = self.format.channels() as usize;
        let count = frames
            .checked_mul(channels)
            .context("PCM chunk size overflow")?;
        anyhow::ensure!(
            count <= MAX_INPUT_SAMPLES,
            "PCM read exceeds 16 million samples; use smaller streaming chunks"
        );
        let pcm = match &mut self.reader {
            Reader::Wav(reader) => {
                let spec = reader.spec();
                match spec.sample_format {
                    hound::SampleFormat::Float => reader
                        .samples::<f32>()
                        .take(count)
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                    hound::SampleFormat::Int => {
                        let scale = 2f64.powi(spec.bits_per_sample as i32 - 1);
                        reader
                            .samples::<i32>()
                            .take(count)
                            .map(|v| v.map(|x| (x as f64 / scale) as f32))
                            .collect::<std::result::Result<Vec<_>, _>>()?
                    }
                }
            }
            Reader::Raw {
                reader,
                encoding,
                remaining_frames,
            } => {
                let count = frames
                    .min(*remaining_frames)
                    .checked_mul(channels)
                    .context("raw PCM sample count overflow")?;
                let width = if *encoding == AudioEncoding::S16le {
                    2
                } else {
                    4
                };
                let mut bytes = vec![0u8; count * width];
                reader
                    .read_exact(&mut bytes)
                    .context("raw PCM file is incomplete")?;
                *remaining_frames -= count / channels;
                if *encoding == AudioEncoding::S16le {
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|sample| i16::from_le_bytes(*sample) as f32 / 32768.)
                        .collect()
                } else {
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|sample| f32::from_le_bytes(*sample))
                        .collect()
                }
            }
        };
        Ok(AudioBuffer::new(self.format, pcm)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options(input_format: AudioEncoding) -> AudioInputOptions {
        AudioInputOptions {
            input_format,
            input_sample_rate: Some(8000),
            input_channels: Some(2),
        }
    }
    #[test]
    fn s16le_stereo_chunks_preserve_the_final_frame_and_drain() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("samples.pcm");
        let samples = [i16::MIN, i16::MAX, 0, 16384, -16384, 8192];
        std::fs::write(
            &path,
            samples
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut source = AudioSource::open(&path, &options(AudioEncoding::S16le))?;
        assert_eq!(source.total_frames(), 3);
        assert_eq!(source.format, AudioFormat::new(8000, 2)?);
        assert_eq!(source.read(2)?.samples(), &[-1., 32767. / 32768., 0., 0.5]);
        assert_eq!(source.read(2)?.samples(), &[-0.5, 0.25]);
        assert!(source.read(2)?.is_empty());
        Ok(())
    }
    #[test]
    fn f32le_stereo_is_exact_and_nonfinite_payloads_fail_preflight() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("samples.pcm");
        let samples = [0.25f32, -0.5, 1., -0.75];
        std::fs::write(
            &path,
            samples
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut source = AudioSource::open(&path, &options(AudioEncoding::F32le))?;
        assert_eq!(source.read(1)?.samples(), &samples[..2]);
        assert_eq!(source.read(1)?.samples(), &samples[2..]);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            std::fs::write(
                &path,
                [value, 0.]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )?;
            assert!(AudioSource::preflight(&path, &options(AudioEncoding::F32le)).is_err());
        }
        Ok(())
    }
    #[test]
    fn raw_requires_metadata_complete_frames_and_bounded_reads() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("samples.pcm");
        for bytes in [vec![0u8; 3], vec![0; 6]] {
            std::fs::write(&path, bytes)?;
            assert!(AudioSource::open(&path, &options(AudioEncoding::S16le)).is_err());
        }
        std::fs::write(&path, [0u8; 8])?;
        let mut source = AudioSource::open(&path, &options(AudioEncoding::F32le))?;
        assert!(source.read(MAX_INPUT_SAMPLES).is_err());
        assert!(AudioInputOptions {
            input_format: AudioEncoding::F32le,
            ..Default::default()
        }
        .raw_format()
        .is_err());
        assert!(AudioInputOptions {
            input_channels: Some(2),
            ..Default::default()
        }
        .raw_format()
        .is_err());
        // Detect truncation after metadata preflight as well as at open time.
        std::fs::write(&path, [0u8; 4])?;
        assert!(source.read(1).is_err());
        Ok(())
    }
}
