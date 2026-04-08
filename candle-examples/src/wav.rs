use std::io::prelude::*;

pub trait Sample {
    fn to_i16(&self) -> i16;
    fn from_i16(v: i16) -> Self;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
    fn from_i16(v: i16) -> Self {
        v as f32 / 32768.0
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
    fn from_i16(v: i16) -> Self {
        v as f64 / 32768.0
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
    fn from_i16(v: i16) -> Self {
        v
    }
}

/// Write mono PCM samples as a WAV file (16-bit, single channel).
pub fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
) -> std::io::Result<()> {
    let len = 12u32; // header
    let len = len + 24u32; // fmt
    let len = len + samples.len() as u32 * 2 + 8; // data
    let n_channels = 1u16;
    let bytes_per_second = sample_rate * 2 * n_channels as u32;
    w.write_all(b"RIFF")?;
    w.write_all(&(len - 8).to_le_bytes())?; // total length minus 8 bytes
    w.write_all(b"WAVE")?;

    // Format block
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // block len minus 8 bytes
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&n_channels.to_le_bytes())?; // one channel
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&bytes_per_second.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?; // 2 bytes of data per sample
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data block
    w.write_all(b"data")?;
    w.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;
    for sample in samples.iter() {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(())
}

/// Write multi-channel PCM samples as a WAV file (16-bit).
///
/// `channels` is a slice of per-channel sample slices, all the same length.
/// For stereo: `&[&left[..], &right[..]]`.
pub fn write_pcm_as_wav_stereo<W: Write, S: Sample>(
    w: &mut W,
    channels: &[&[S]],
    sample_rate: u32,
) -> std::io::Result<()> {
    let n_channels = channels.len() as u16;
    let n_frames = channels[0].len();
    let data_bytes = (n_frames * n_channels as usize * 2) as u32;
    let block_align = n_channels * 2;
    let bytes_per_second = sample_rate * block_align as u32;

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVE")?;

    // fmt chunk
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&n_channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&bytes_per_second.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk (interleaved)
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    for frame in 0..n_frames {
        for ch in channels {
            w.write_all(&ch[frame].to_i16().to_le_bytes())?;
        }
    }
    Ok(())
}

/// Header information from a WAV file.
pub struct WavHeader {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

/// Read a 16-bit PCM WAV file and return per-channel sample vectors.
///
/// Returns `(header, channels)` where `channels[i]` contains the samples for
/// channel `i`. Mono files have one channel, stereo files have two, etc.
pub fn read_pcm_from_wav<R: Read + Seek, S: Sample>(
    r: &mut R,
) -> std::io::Result<(WavHeader, Vec<Vec<S>>)> {
    // RIFF header (12 bytes)
    let mut riff = [0u8; 12];
    r.read_exact(&mut riff)?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a WAV file"));
    }

    let mut header = WavHeader {
        sample_rate: 0,
        channels: 0,
        bits_per_sample: 0,
    };
    let mut data_buf = Vec::new();

    // Walk chunks
    loop {
        let mut ch = [0u8; 8];
        if r.read_exact(&mut ch).is_err() {
            break;
        }
        let id: [u8; 4] = ch[0..4].try_into().unwrap();
        let size = u32::from_le_bytes(ch[4..8].try_into().unwrap()) as usize;

        if &id == b"fmt " {
            let mut fmt = vec![0u8; size];
            r.read_exact(&mut fmt)?;
            header.channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
            header.sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
            header.bits_per_sample = u16::from_le_bytes(fmt[14..16].try_into().unwrap());
        } else if &id == b"data" {
            data_buf.resize(size, 0u8);
            r.read_exact(&mut data_buf)?;
            break;
        } else {
            // Skip unknown chunk (RIFF chunks are padded to even size)
            let skip = ((size + 1) & !1) as i64;
            r.seek(std::io::SeekFrom::Current(skip))?;
        }
    }

    if data_buf.is_empty() || header.channels == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no data chunk found in WAV",
        ));
    }
    if header.bits_per_sample != 16 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "only 16-bit WAV supported, got {}-bit",
                header.bits_per_sample
            ),
        ));
    }

    let nc = header.channels as usize;
    let n_frames = data_buf.len() / (2 * nc);
    let mut channels: Vec<Vec<S>> = (0..nc).map(|_| Vec::with_capacity(n_frames)).collect();

    for frame in 0..n_frames {
        for (ch, channel) in channels.iter_mut().enumerate() {
            let idx = (frame * nc + ch) * 2;
            let sample = i16::from_le_bytes([data_buf[idx], data_buf[idx + 1]]);
            channel.push(S::from_i16(sample));
        }
    }

    Ok((header, channels))
}
