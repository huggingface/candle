#![cfg(feature = "audio")]
//Adapted from: https://github.com/tanmayb123/OpenAI-Whisper-CoreML

use std::sync::Arc;

use candle::{Device, Tensor, WithDType};
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};

use super::*;

pub trait Float:
    num_traits::Float
    + num_traits::FloatConst
    + num_traits::NumAssign
    + std::fmt::Debug
    + realfft::FftNum
    + WithDType
{
}

impl Float for f32 {}
impl Float for f64 {}

pub struct SpectrogramGenerator<F: Float> {
    fft_plan: Arc<dyn RealToComplex<F>>,
    hann_window: Vec<F>,
    mels: Tensor,
}

impl<F: Float> SpectrogramGenerator<F> {
    pub fn new(mels: Vec<F>, n_mels: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        Self {
            fft_plan: planner.plan_fft_forward(N_FFT),
            hann_window: Self::hann_window(),
            mels: Tensor::from_vec(mels, (n_mels, N_FFT / 2 + 1), &candle::Device::Cpu).unwrap(),
        }
    }

    fn hann_window() -> Vec<F> {
        let two_pi = F::PI() + F::PI();
        let half = F::from(0.5).unwrap();
        let one = F::from(1.0).unwrap();
        let fft_size_t = F::from(N_FFT).unwrap();

        (0..N_FFT)
            .map(|i| half * (one - ((two_pi * F::from(i).unwrap()) / fft_size_t).cos()))
            .collect()
    }

    fn fft(&self, audio: &[F]) -> Vec<Complex<F>> {
        let mut input = audio.to_vec();

        for i in 0..N_FFT {
            input[i] *= self.hann_window[i];
        }

        let mut spectrum = self.fft_plan.make_output_vec();
        self.fft_plan
            .process(input.as_mut_slice(), &mut spectrum)
            .unwrap();
        spectrum
    }

    fn mel_spectrogram(&self, audio: &[F]) -> Vec<F> {
        let f4 = F::from(4.0).unwrap();
        let f8 = F::from(8.0).unwrap();

        let n_frames = (audio.len() - N_FFT) / HOP_LENGTH;
        let right_padding = N_SAMPLES + FFT_PAD; //padding is all 0s, so we can ignore it

        let mut spectrogram = vec![vec![F::zero(); n_frames]; N_FFT / 2 + 1];
        for i in (0..audio.len() - right_padding).step_by(HOP_LENGTH) {
            if i / HOP_LENGTH >= n_frames {
                break;
            }
            let fft = self.fft(&audio[i..i + N_FFT]);
            let spectrogram_col = fft.iter().map(|c| c.norm_sqr()).collect::<Vec<F>>();
            for (j, v) in spectrogram_col.iter().enumerate() {
                spectrogram[j][i / HOP_LENGTH] = *v;
            }
        }

        let flattened = spectrogram.iter().flatten().map(|v| *v).collect::<Vec<F>>();
        let spec_t = Tensor::from_vec(flattened, (N_FFT / 2 + 1, n_frames), &Device::Cpu).unwrap();

        let mel_spec = self.mels.matmul(&spec_t).unwrap();

        //Moving the below to tensor operations would be optimal
        let mut mel_spec_v = mel_spec.flatten_all().unwrap().to_vec1::<F>().unwrap();
        for v in mel_spec_v.iter_mut() {
            *v = <F as num_traits::Float>::max(*v, F::from(1e-10).unwrap()).log10();
        }
        let max = mel_spec_v
            .iter()
            .fold(F::min_value(), |a, &b| <F as num_traits::Float>::max(a, b));
        for v in mel_spec_v.iter_mut() {
            *v = (<F as num_traits::Float>::max(max - f8, *v) + f4) / f4;
        }
        mel_spec_v
    }

    pub fn generate(&self, audio: Vec<F>) -> Vec<F> {
        if audio.is_empty() {
            panic!("Audio is empty");
        }
        let padded = Self::pad_audio(audio, N_SAMPLES);
        self.mel_spectrogram(&padded)
    }

    //The padding done by OAI is as follows:
    //1. First explicitly pad with (CHUNK_LENGTH * SAMPLE_RATE) (480,000) zeros
    //2. Perform a reflection padding of FFT_PAD (200) samples, this is done internally in `torch.stft`
    pub fn pad_audio(samples: Vec<F>, padding: usize) -> Vec<F> {
        let padded_len = FFT_PAD + samples.len() + padding + FFT_PAD;
        let mut padded_samples = vec![F::zero(); padded_len];

        let mut reflect_padding = vec![F::zero(); FFT_PAD];
        for i in 0..FFT_PAD {
            reflect_padding[i] = samples[FFT_PAD - i];
        }

        padded_samples[0..FFT_PAD].copy_from_slice(&reflect_padding);
        padded_samples[FFT_PAD..(FFT_PAD + samples.len())].copy_from_slice(&samples);
        padded_samples
    }
}

pub fn pcm_to_mel<F: Float>(cfg: &super::Config, samples: &[F], filters: &[F]) -> Vec<F> {
    let generator = SpectrogramGenerator::new(filters.to_vec(), cfg.num_mel_bins);
    generator.generate(samples.to_vec())
}

#[cfg(test)]
mod tests {
    use hf_hub::{
        api::sync::{Api, ApiRepo},
        Repo, RepoType,
    };
    use std::path::Path;

    use candle::{Result, Tensor};

    use super::*;

    fn load_mels() -> Vec<f32> {
        let mel_bytes = include_bytes!("melfilters.bytes").as_slice();
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );
        mel_filters
    }

    fn load_audio(dataset: ApiRepo) -> Result<Vec<f32>> {
        let mut input = std::fs::File::open(dataset.get("samples_gb0.wav").unwrap())?;
        let (header, data) = wav::read(&mut input)?;
        if header.sampling_rate != SAMPLE_RATE as u32 {
            panic!("wav file must have a {} sampling rate", SAMPLE_RATE)
        }
        let data = data.as_sixteen().expect("expected 16 bit wav file");
        let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect();
        Ok(pcm_data)
    }

    #[test]
    fn test_log_mel() -> Result<()> {
        let mel_filters = load_mels();

        let api = Api::new().unwrap();
        let dataset = api.dataset("Narsil/candle-examples".to_string());
        let repo = api.repo(Repo::with_revision(
            String::from("openai/whisper-tiny.en"),
            RepoType::Model,
            String::from("refs/pr/15"),
        ));
        let pcm_data = load_audio(dataset).unwrap();

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(repo.get("config.json").unwrap()).unwrap(),
        )
        .unwrap();

        let mel = pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &candle::Device::Cpu,
        )
        .unwrap();
        let ground = Tensor::read_npy(Path::new("./src/models/whisper/ground.npy"))
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        assert!(mel.all_close(&ground, 1e-4).unwrap());
        Ok(())
    }
}
