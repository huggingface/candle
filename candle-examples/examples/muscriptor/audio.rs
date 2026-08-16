//! Audio decoding (any symphonia-supported format, downmixed to mono) and
//! fractional sinc resampling.

use anyhow::Result;

/// Decode an audio file to mono f32 samples, averaging all channels.
pub fn pcm_decode_mono<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::conv::FromSample;

    fn accumulate<T>(
        samples: &mut Vec<f32>,
        data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
    ) where
        T: symphonia::core::sample::Sample,
        f32: FromSample<T>,
    {
        let n_channels = data.spec().channels.count().max(1);
        let scale = 1. / n_channels as f32;
        let base = samples.len();
        samples.resize(base + data.frames(), 0.);
        for ch in 0..n_channels {
            for (out, v) in samples[base..].iter_mut().zip(data.chan(ch)) {
                *out += f32::from_sample(*v) * scale;
            }
        }
    }

    let src = std::fs::File::open(path)?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &Default::default(),
        &Default::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no supported audio track"))?;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let track_id = track.id;
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // Only the normal end of stream terminates the loop; a decode /
            // container error mid-file must not be mistaken for EOF.
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::F64(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::U8(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::U16(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::U24(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::U32(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::S8(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::S16(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::S24(buf) => accumulate(&mut samples, buf),
            AudioBufferRef::S32(buf) => accumulate(&mut samples, buf),
        }
    }
    if sample_rate == 0 {
        sample_rate = decoder
            .codec_params()
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
    }
    Ok((samples, sample_rate))
}

fn sinc(x: f64) -> f64 {
    if x == 0. {
        1.
    } else {
        x.sin() / x
    }
}

/// Fractional resampling with a windowed-sinc kernel bank (Julius O. Smith's
/// algorithm, matching the `julius.resample_frac` implementation used by the
/// reference: `zeros = 24`, `rolloff = 0.945`, replicate padding).
pub fn resample(x: &[f32], old_sr: usize, new_sr: usize) -> Vec<f32> {
    if old_sr == new_sr || x.is_empty() {
        return x.to_vec();
    }
    let gcd = {
        let (mut a, mut b) = (old_sr, new_sr);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let old_sr = old_sr / gcd;
    let new_sr = new_sr / gcd;
    const ZEROS: f64 = 24.;
    const ROLLOFF: f64 = 0.945;
    // The anti-aliasing lowpass sits at rolloff * min(sr) / 2.
    let sr = old_sr.min(new_sr) as f64 * ROLLOFF;
    let width = (ZEROS * old_sr as f64 / sr).ceil() as usize;
    let kernel_len = 2 * width + old_sr;

    // One kernel per output phase within a block of new_sr samples.
    let mut kernels = vec![0f64; new_sr * kernel_len];
    for i in 0..new_sr {
        let kernel = &mut kernels[i * kernel_len..(i + 1) * kernel_len];
        let mut sum = 0.;
        for (k, v) in kernel.iter_mut().enumerate() {
            let idx = k as f64 - width as f64;
            let t = (-(i as f64) / new_sr as f64 + idx / old_sr as f64) * sr;
            let t = t.clamp(-ZEROS, ZEROS) * std::f64::consts::PI;
            let window = (t / ZEROS / 2.).cos().powi(2);
            *v = sinc(t) * window;
            sum += *v;
        }
        for v in kernel.iter_mut() {
            *v /= sum;
        }
    }

    // Replicate-pad by `width` left and `width + old_sr` right, then apply
    // each kernel with a stride of old_sr.
    let padded_len = x.len() + 2 * width + old_sr;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend(std::iter::repeat_n(x[0] as f64, width));
    padded.extend(x.iter().map(|&v| v as f64));
    padded.extend(std::iter::repeat_n(
        *x.last().unwrap() as f64,
        width + old_sr,
    ));

    let out_len = x.len() * new_sr / old_sr;
    let n_blocks = (padded_len - kernel_len) / old_sr + 1;
    let mut out = vec![0f32; out_len];
    for block in 0..n_blocks {
        let input = &padded[block * old_sr..block * old_sr + kernel_len];
        for i in 0..new_sr {
            let pos = block * new_sr + i;
            if pos >= out_len {
                break;
            }
            let kernel = &kernels[i * kernel_len..(i + 1) * kernel_len];
            out[pos] = input.iter().zip(kernel).map(|(&a, &b)| a * b).sum::<f64>() as f32;
        }
    }
    out
}
