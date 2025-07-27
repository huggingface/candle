/// Simple linear interpolation resampling
/// For production use, consider using a proper resampling library
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return audio.to_vec();
    }

    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (audio.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let input_index = i as f64 / ratio;
        let input_index_floor = input_index.floor() as usize;
        let input_index_ceil = (input_index_floor + 1).min(audio.len() - 1);

        if input_index_floor >= audio.len() {
            break;
        }

        // Linear interpolation
        let frac = input_index - input_index_floor as f64;
        let interpolated = if input_index_floor == input_index_ceil {
            audio[input_index_floor]
        } else {
            audio[input_index_floor] * (1.0 - frac as f32) + audio[input_index_ceil] * frac as f32
        };

        output.push(interpolated);
    }

    output
}
