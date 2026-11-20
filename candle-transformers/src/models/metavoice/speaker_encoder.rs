//! MetaVoice's three-layer speaker encoder and Librosa-compatible power mel frontend.
//! Reference: metavoice-src de3fa211ac4621e03a5f990651aeecc64da418f5.
use candle::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder, RNN};
use rustfft::{num_complex::Complex, FftPlanner};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub sampling_rate: usize,
    pub partial_n_frames: usize,
    pub model_hidden_size: usize,
    pub model_embedding_size: usize,
    pub model_num_layers: usize,
    pub mel_window_length: usize,
    pub mel_window_step: usize,
    pub mel_n_channels: usize,
}
impl Config {
    pub fn cfg() -> Self {
        Self {
            sampling_rate: 16000,
            partial_n_frames: 160,
            model_hidden_size: 256,
            model_embedding_size: 256,
            model_num_layers: 3,
            mel_window_length: 25,
            mel_window_step: 10,
            mel_n_channels: 40,
        }
    }
    fn fft_and_hop(&self) -> Result<(usize, usize)> {
        let fft = self
            .sampling_rate
            .checked_mul(self.mel_window_length)
            .map(|n| n / 1000)
            .filter(|n| *n > 0 && *n <= 16384);
        let hop = self
            .sampling_rate
            .checked_mul(self.mel_window_step)
            .map(|n| n / 1000)
            .filter(|n| *n > 0);
        match (fft, hop) {
            (Some(fft), Some(hop))
                if self.mel_n_channels > 0
                    && self.mel_n_channels <= 512
                    && self.partial_n_frames > 0 =>
            {
                Ok((fft, hop))
            }
            _ => candle::bail!("invalid speaker encoder sampling/window/mel configuration"),
        }
    }
}

/// Slaney's piecewise linear/log mel scale and area-normalized triangular filters,
/// as used by librosa.feature.melspectrogram (htk=false, norm='slaney').
pub fn mel_filters(config: &Config) -> Result<Vec<f32>> {
    let (fft, _) = config.fft_and_hop()?;
    let hz_to_mel = |hz: f64| {
        if hz < 1000. {
            hz / (200. / 3.)
        } else {
            15. + (hz / 1000.).ln() / (6.4f64.ln() / 27.)
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15. {
            mel * (200. / 3.)
        } else {
            1000. * ((mel - 15.) * (6.4f64.ln() / 27.)).exp()
        }
    };
    let max = hz_to_mel(config.sampling_rate as f64 / 2.);
    let edges: Vec<_> = (0..config.mel_n_channels + 2)
        .map(|i| mel_to_hz(max * i as f64 / (config.mel_n_channels + 1) as f64))
        .collect();
    let bins = fft / 2 + 1;
    let mut filters = vec![0.; config.mel_n_channels * bins];
    for mel in 0..config.mel_n_channels {
        let norm = 2. / (edges[mel + 2] - edges[mel]);
        for bin in 0..bins {
            let hz = bin as f64 * config.sampling_rate as f64 / fft as f64;
            let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
            // Librosa constructs float32 triangles, then normalizes them in place.
            filters[mel * bins + bin] = ((lower.min(upper).max(0.) as f32) as f64 * norm) as f32;
        }
    }
    Ok(filters)
}

/// Centered, constant-padded, periodic-Hann power spectrogram, returned time-major.
/// This frontend intentionally does not take a logarithm or apply Whisper scaling.
pub fn power_mel_spectrogram(wav: &[f32], filters: &[f32], config: &Config) -> Result<Vec<f32>> {
    power_mel_with_callback(wav, filters, config, || Ok(()))
}
fn power_mel_with_callback(
    wav: &[f32],
    filters: &[f32],
    config: &Config,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<f32>> {
    let (fft, hop) = config.fft_and_hop()?;
    let bins = fft / 2 + 1;
    if wav.is_empty()
        || wav.iter().any(|v| !v.is_finite())
        || filters.len() != config.mel_n_channels * bins
        || filters.iter().any(|v| !v.is_finite())
    {
        candle::bail!("speaker encoder needs finite nonempty PCM and correctly shaped mel filters");
    }
    let mut planner = FftPlanner::<f64>::new();
    let transform = planner.plan_fft_forward(fft);
    let window: Vec<_> = (0..fft)
        .map(|i| 0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / fft as f64).cos())
        .collect();
    let frames = wav.len() / hop + 1;
    let mut output = vec![0.; frames * config.mel_n_channels];
    let mut spectrum = vec![Complex::new(0., 0.); fft];
    let mut power = vec![0f32; bins];
    for frame in 0..frames {
        check()?;
        for i in 0..fft {
            let position = frame * hop + i;
            let sample = position
                .checked_sub(fft / 2)
                .and_then(|i| wav.get(i))
                .copied()
                .unwrap_or(0.);
            spectrum[i] = Complex::new(sample as f64 * window[i], 0.);
        }
        transform.process(&mut spectrum);
        for (power, spectrum) in power.iter_mut().zip(&spectrum) {
            let re = spectrum.re as f32;
            let im = spectrum.im as f32;
            *power = re * re + im * im;
        }
        for mel in 0..config.mel_n_channels {
            output[frame * config.mel_n_channels + mel] = power
                .iter()
                .zip(&filters[mel * bins..(mel + 1) * bins])
                .map(|(p, w)| p * w)
                .sum();
        }
    }
    Ok(output)
}

pub struct Model {
    lstms: Vec<candle_nn::LSTM>,
    linear: Linear,
    cfg: Config,
    device: Device,
}
pub type Slice = (usize, usize);
impl Model {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        cfg.fft_and_hop()?;
        if cfg.model_num_layers == 0 || cfg.model_hidden_size == 0 || cfg.model_embedding_size == 0
        {
            candle::bail!("speaker encoder dimensions must be nonzero");
        }
        let mut lstms = Vec::with_capacity(cfg.model_num_layers);
        for layer_idx in 0..cfg.model_num_layers {
            let input = if layer_idx == 0 {
                cfg.mel_n_channels
            } else {
                cfg.model_hidden_size
            };
            let config = candle_nn::LSTMConfig {
                layer_idx,
                ..Default::default()
            };
            lstms.push(candle_nn::lstm(
                input,
                cfg.model_hidden_size,
                config,
                vb.pp("lstm"),
            )?);
        }
        let linear = linear_b(
            cfg.model_hidden_size,
            cfg.model_embedding_size,
            true,
            vb.pp("linear"),
        )?;
        Ok(Self {
            lstms,
            linear,
            cfg,
            device: vb.device().clone(),
        })
    }
    pub fn config(&self) -> &Config {
        &self.cfg
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn partial_slices(
        config: &Config,
        n_samples: usize,
        rate: f64,
        min_coverage: f64,
    ) -> Result<(Vec<Slice>, Vec<Slice>)> {
        let (_, hop) = config.fft_and_hop()?;
        if n_samples == 0
            || !rate.is_finite()
            || rate <= 0.
            || !min_coverage.is_finite()
            || !(0. ..=1.).contains(&min_coverage)
        {
            candle::bail!("invalid speaker reference length, partial rate or coverage");
        }
        let step = (config.sampling_rate as f64 / rate / hop as f64).round_ties_even() as usize;
        if step == 0 || step > config.partial_n_frames {
            candle::bail!("speaker partial rate leaves gaps or has zero frame step");
        }
        let frames = n_samples
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("speaker duration overflow".into()))?
            .div_ceil(hop);
        let steps = frames
            .saturating_add(step)
            .saturating_add(1)
            .saturating_sub(config.partial_n_frames)
            .max(1);
        let mut mel: Vec<_> = (0..steps)
            .step_by(step)
            .map(|start| (start, start + config.partial_n_frames))
            .collect();
        let mut wav: Vec<_> = mel
            .iter()
            .map(|(start, end)| (start * hop, end * hop))
            .collect();
        let last = wav.last().unwrap();
        let coverage = (n_samples as f64 - last.0 as f64) / (last.1 - last.0) as f64;
        if coverage < min_coverage && mel.len() > 1 {
            mel.pop();
            wav.pop();
        }
        Ok((wav, mel))
    }
    pub fn embed_utterance(
        &self,
        wav: &[f32],
        filters: &[f32],
        rate: f64,
        min_coverage: f64,
        device: &Device,
    ) -> Result<Tensor> {
        self.embed_utterance_with_callback(wav, filters, rate, min_coverage, device, || Ok(()))
    }
    pub fn embed_utterance_with_callback(
        &self,
        wav: &[f32],
        filters: &[f32],
        rate: f64,
        min_coverage: f64,
        device: &Device,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Tensor> {
        check()?;
        let (wav_slices, mel_slices) =
            Self::partial_slices(&self.cfg, wav.len(), rate, min_coverage)?;
        let max = wav_slices.last().unwrap().1;
        let mut padded = wav.to_vec();
        padded.resize(padded.len().max(max), 0.);
        let mel = power_mel_with_callback(&padded, filters, &self.cfg, &mut check)?;
        let mut mels = Vec::with_capacity(
            mel_slices.len() * self.cfg.partial_n_frames * self.cfg.mel_n_channels,
        );
        for (start, end) in &mel_slices {
            mels.extend_from_slice(
                &mel[start * self.cfg.mel_n_channels..end * self.cfg.mel_n_channels],
            );
        }
        let mels = Tensor::from_vec(
            mels,
            (
                mel_slices.len(),
                self.cfg.partial_n_frames,
                self.cfg.mel_n_channels,
            ),
            device,
        )?;
        let partials = self.forward_with_callback(&mels, &mut check)?;
        normalize(&partials.mean(0)?.unsqueeze(0)?)?.squeeze(0)
    }
    pub fn forward_with_callback(
        &self,
        xs: &Tensor,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Tensor> {
        let (batch, frames, features) = xs.dims3()?;
        if batch == 0 || frames == 0 || features != self.cfg.mel_n_channels {
            candle::bail!("speaker encoder expects nonempty batch,time,mel features");
        }
        let mut xs = xs.clone();
        let mut hidden = None;
        for (index, layer) in self.lstms.iter().enumerate() {
            let mut state = layer.zero_state(batch)?;
            let mut outputs = Vec::with_capacity(frames);
            for frame in 0..frames {
                check()?;
                state = layer.step(&xs.i((.., frame, ..))?.contiguous()?, &state)?;
                if index + 1 < self.lstms.len() {
                    outputs.push(state.h().clone());
                }
            }
            if index + 1 < self.lstms.len() {
                xs = Tensor::stack(&outputs, 1)?;
            } else {
                hidden = Some(state.h().clone());
            }
        }
        let raw = hidden.unwrap().apply(&self.linear)?.relu()?;
        normalize(&raw)
    }
}
fn normalize(values: &Tensor) -> Result<Tensor> {
    let norm = values.sqr()?.sum_keepdim(1)?.sqrt()?;
    if norm
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?
        .iter()
        .any(|v| !v.is_finite() || *v <= 1e-12)
    {
        candle::bail!("speaker encoder produced an invalid or zero embedding");
    }
    values.broadcast_div(&norm)
}
impl Module for Model {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_callback(xs, || Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_frontend_and_slices_match_independent_librosa() -> Result<()> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/metavoice_speaker_encoder.json"
        ))
        .unwrap();
        let pcm: Vec<f32> = serde_json::from_value(fixture["pcm"].clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(fixture["mel"].clone()).unwrap();
        let cfg = Config::cfg();
        let filters = mel_filters(&cfg)?;
        let actual = power_mel_spectrogram(&pcm, &filters, &cfg)?;
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - e).abs() < 2e-6 + 1e-4 * e.abs(),
                "mel[{index}] native={a} reference={e}"
            );
        }
        for case in fixture["slices"].as_array().unwrap() {
            let (wav, mel) =
                Model::partial_slices(&cfg, case["samples"].as_u64().unwrap() as usize, 1.3, 0.75)?;
            let expected_wav: Vec<Slice> = serde_json::from_value(case["wav"].clone()).unwrap();
            let expected_mel: Vec<Slice> = serde_json::from_value(case["mel"].clone()).unwrap();
            assert_eq!(wav, expected_wav);
            assert_eq!(mel, expected_mel);
        }
        assert!(Model::partial_slices(&cfg, 0, 1.3, 0.75).is_err());
        assert!(Model::partial_slices(&cfg, 16000, 0., 0.75).is_err());
        Ok(())
    }
}
