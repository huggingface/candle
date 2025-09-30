use candle::{BackendStorage, Result, Tensor};

// https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/data/audio_utils.py#L57
pub fn normalize_loudness<B: BackendStorage>(
    wav: &Tensor<B>,
    sample_rate: u32,
    loudness_compressor: bool,
) -> Result<Tensor<B>> {
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
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::conv::FromSample;

    fn conv<T>(
        samples: &mut Vec<f32>,
        data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
    ) where
        T: symphonia::core::sample::Sample,
        f32: symphonia::core::conv::FromSample<T>,
    {
        samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
    }

    // Open the media source.
    let src = std::fs::File::open(path).map_err(candle::Error::wrap)?;

    // Create the media source stream.
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = symphonia::core::probe::Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(candle::Error::wrap)?;
    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| candle::Error::Msg("no supported audio tracks".to_string()))?;

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|_| candle::Error::Msg("unsupported codec".to_string()))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    // The decode loop.
    while let Ok(packet) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet).map_err(candle::Error::wrap)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

#[cfg(feature = "rubato")]
pub fn resample(pcm_in: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let mut pcm_out =
        Vec::with_capacity((pcm_in.len() as f64 * sr_out as f64 / sr_in as f64) as usize + 1024);

    let mut resampler = rubato::FftFixedInOut::<f32>::new(sr_in as usize, sr_out as usize, 1024, 1)
        .map_err(candle::Error::wrap)?;
    let mut output_buffer = resampler.output_buffer_allocate(true);
    let mut pos_in = 0;
    while pos_in + resampler.input_frames_next() < pcm_in.len() {
        let (in_len, out_len) = resampler
            .process_into_buffer(&[&pcm_in[pos_in..]], &mut output_buffer, None)
            .map_err(candle::Error::wrap)?;
        pos_in += in_len;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    if pos_in < pcm_in.len() {
        let (_in_len, out_len) = resampler
            .process_partial_into_buffer(Some(&[&pcm_in[pos_in..]]), &mut output_buffer, None)
            .map_err(candle::Error::wrap)?;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    Ok(pcm_out)
}
