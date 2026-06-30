use candle::{Result, Tensor};

// https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/data/audio_utils.py#L57
pub fn normalize_loudness(
    wav: &Tensor,
    sample_rate: u32,
    loudness_compressor: bool,
) -> Result<Tensor> {
    let energy = wav.sqr()?.mean_all()?.sqrt()?.to_vec0::<f32>()?;
    if energy < 2e-3 {
        return Ok(wav.clone());
    }
    let wav_array = wav.to_vec1::<f32>()?;
    let mut meter = crate::bs1770::ChannelLoudnessMeter::new(sample_rate);
    meter.push(wav_array.into_iter());
    let power = meter.as_100ms_windows();
    let loudness = match crate::bs1770::gated_mean(power) {
        None => return Ok(wav.clone()),
        Some(gp) => gp.loudness_lkfs() as f64,
    };
    let delta_loudness = -14. - loudness;
    let gain = 10f64.powf(delta_loudness / 20.);
    let wav = (wav * gain)?;
    if loudness_compressor {
        wav.tanh()
    } else {
        Ok(wav)
    }
}

#[cfg(feature = "symphonia")]
pub fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::conv::FromSample;
    use symphonia::core::audio::Audio;
    use symphonia::core::audio::GenericAudioBufferRef;
    use symphonia::core::codecs::audio::AudioDecoderOptions;

    fn conv<T>(
        samples: &mut Vec<f32>,
        data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
    ) where
        T: symphonia::core::audio::sample::Sample,
        f32: symphonia::core::audio::conv::FromSample<T>,
    {
        if let Some(ch0) = data.plane(0) {
            samples.extend(ch0.iter().map(|v| f32::from_sample(*v)))
        }
    }

    // Open the media source.
    let src = std::fs::File::open(path).map_err(candle::Error::wrap)?;

    // Create the media source stream.
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = symphonia::core::formats::probe::Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source.
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .map_err(candle::Error::wrap)?;

    // Find the first audio track with a known (decodable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.as_ref().is_some_and(|s| s.is_audio()))
        .ok_or_else(|| candle::Error::Msg("no supported audio tracks".to_string()))?;

    // Use the default options for the decoder.
    let dec_opts: AudioDecoderOptions = Default::default();

    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| candle::Error::Msg("codec parameters missing".to_string()))?;

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&codec_params, &dec_opts)
        .map_err(|_| candle::Error::Msg("unsupported codec".to_string()))?;
    let track_id = track.id;
    let sample_rate = codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    // The decode loop.
    while let Ok(Some(packet)) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.

        use std::borrow::Cow;
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet).map_err(candle::Error::wrap)? {
            GenericAudioBufferRef::F32(buf) => {
                if let Some(ch0) = buf.plane(0) {
                    pcm_data.extend(ch0)
                }
            }
            GenericAudioBufferRef::U8(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::U16(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::U24(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::U32(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::S8(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::S16(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::S24(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::S32(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
            GenericAudioBufferRef::F64(data) => conv(&mut pcm_data, Cow::Borrowed(data)),
        }
    }
    Ok((pcm_data, sample_rate))
}

#[cfg(feature = "rubato")]
pub fn resample(pcm_in: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let input = rubato::audioadapter_buffers::owned::InterleavedOwned::new_from(
        pcm_in.to_vec(),
        1,
        pcm_in.len(),
    )
    .map_err(candle::Error::wrap)?;
    let mut resampler = rubato::Fft::<f32>::new(
        sr_in as usize,
        sr_out as usize,
        1024,
        1,
        1,
        rubato::FixedSync::Input,
    )
    .map_err(candle::Error::wrap)?;
    let output_len = resampler.process_all_needed_output_len(pcm_in.len());
    let mut output =
        rubato::audioadapter_buffers::owned::InterleavedOwned::new(0f32, 1, output_len);
    let (_in_len, out_len) = resampler
        .process_all_into_buffer(&input, &mut output, pcm_in.len(), None)
        .map_err(candle::Error::wrap)?;
    let mut pcm_out = output.take_data();
    pcm_out.truncate(out_len);

    Ok(pcm_out)
}
