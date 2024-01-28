use anyhow::Result;
use std::path::PathBuf;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub(crate) fn pcm_decode(path: PathBuf) -> Result<Vec<f32>> {
    // Open the media source.
    let src = std::fs::File::open(path)?;

    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");
    let track_id = track.id;
    let mut input = Vec::new();
    // The decode loop.
    while let Ok(packet) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            // Pop the old head of the metadata queue.
            format.metadata().pop();

            // Consume the new metadata at the head of the metadata queue.
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(AudioBufferRef::F32(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample);
                }
            }
            Ok(AudioBufferRef::U8(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            Ok(AudioBufferRef::U16(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            Ok(AudioBufferRef::U32(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            Ok(AudioBufferRef::S8(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            Ok(AudioBufferRef::S16(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            Ok(AudioBufferRef::S32(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            Ok(AudioBufferRef::F64(buf)) => {
                for &sample in buf.chan(0) {
                    input.push(sample as f32 / 32768.0);
                }
            }
            _ => {}
        }
    }
    Ok(input)
}
