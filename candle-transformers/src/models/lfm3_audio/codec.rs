//! Incremental LiquidCodec FSQ decoder and overlap-add inverse STFT.
use super::super::lfm3::{self, Linear, Stack, Weights};
use candle::{DType, Device, Module, Result, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};
const FFT: usize = 1280;
const HOP: usize = 320;
const STACK: usize = 6;
const HIDDEN: usize = 512;
const LEVELS: [u32; 4] = [8, 7, 6, 6];
#[derive(Clone)]
pub struct Codec {
    project: Linear,
    from_quantizer: Linear,
    decoder: Stack,
    spectrum: Linear,
    device: Device,
    audio: Vec<f32>,
    envelope: Vec<f32>,
    window: Vec<f32>,
    skip: usize,
    frames: usize,
    finished: bool,
}
impl Codec {
    pub fn new(weights: &Weights<'_>) -> Result<Self> {
        let config:lfm3::Config=serde_json::from_value(serde_json::json!({"vocab_size":1,"hidden_size":512,"intermediate_size":2304,"num_hidden_layers":8,"num_attention_heads":8,"num_key_value_heads":2,"head_dim":64,"max_position_embeddings":128000,"conv_L_cache":7,"conv_bias":false,"sliding_window":30,"use_sandwich_norm":true,"layer_types":["conv","conv","sliding_attention","conv","sliding_attention","conv","sliding_attention","conv"],"rope_parameters":{"sliding_attention":{"rope_theta":1000000.}}})).map_err(candle::Error::wrap)?;
        Ok(Self {
            project: weights.linear(
                48,
                256,
                "audio_tokenizer.quantizer.project_out",
                "a.gen.wav.quant.first.out_proj",
                true,
            )?,
            from_quantizer: weights.linear(
                256,
                HIDDEN,
                "audio_tokenizer.from_quantizer",
                "a.gen.wav.tfm.in_proj",
                true,
            )?,
            decoder: Stack::new(
                &config,
                weights,
                "audio_tokenizer.decoder.lfm",
                "a.gen.wav.tfm.",
                true,
                true,
                true,
            )?,
            spectrum: weights.linear(
                HIDDEN,
                2 * (FFT / 2 + 1),
                "audio_tokenizer.decoder.lin",
                "a.gen.wav.tfm.out_proj",
                true,
            )?,
            device: weights.device().clone(),
            audio: vec![0.; FFT],
            envelope: vec![0.; FFT],
            window: (0..FFT)
                .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT as f32).cos())
                .collect(),
            skip: (FFT - HOP) / 2,
            frames: 0,
            finished: false,
        })
    }
    pub fn reset(&mut self) {
        self.decoder.reset();
        self.audio.fill(0.);
        self.envelope.fill(0.);
        self.skip = (FFT - HOP) / 2;
        self.frames = 0;
        self.finished = false;
    }
    fn take(&mut self, count: usize, out: &mut Vec<f32>) {
        for i in 0..count {
            if self.skip > 0 {
                self.skip -= 1;
            } else {
                out.push(if self.envelope[i] > 1e-10 {
                    self.audio[i] / self.envelope[i]
                } else {
                    0.
                });
            }
        }
        self.audio.copy_within(count.., 0);
        self.envelope.copy_within(count.., 0);
        self.audio[FFT - count..].fill(0.);
        self.envelope[FFT - count..].fill(0.);
    }
    pub fn decode_frame(&mut self, codes: &[u32]) -> Result<Vec<f32>> {
        if self.finished || codes.len() != 12 || codes.iter().any(|&c| c >= 2016) {
            candle::bail!("LiquidCodec requires twelve nonterminal FSQ codes and an open stream")
        }
        let mut latent = Vec::with_capacity(48);
        for &code in codes {
            let mut basis = 1;
            for level in LEVELS {
                let half = level / 2;
                latent.push(((code / basis % level) as f32 - half as f32) / half as f32);
                basis *= level;
            }
        }
        let latent = Tensor::from_vec(latent, (1, 1, 48), &self.device)?;
        let hidden = self
            .from_quantizer
            .forward(&self.project.forward(&latent)?)?
            .broadcast_as((1, STACK, HIDDEN))?
            .contiguous()?;
        let states = self.decoder.forward(&hidden)?;
        let spectra = self
            .spectrum
            .forward(&states)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(FFT);
        let mut values = vec![Complex32::default(); FFT];
        let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let mut out = Vec::new();
        for spectrum in spectra {
            let bins = FFT / 2 + 1;
            for k in 0..bins {
                values[k] = Complex32::from_polar(spectrum[k].exp(), spectrum[bins + k]);
            }
            values[0].im = 0.;
            values[FFT / 2].im = 0.;
            for k in 1..FFT / 2 {
                values[FFT - k] = values[k].conj();
            }
            fft.process_with_scratch(&mut values, &mut scratch);
            for (i, v) in values.iter().enumerate() {
                self.audio[i] += v.re / FFT as f32 * self.window[i];
                self.envelope[i] += self.window[i] * self.window[i];
            }
            self.take(HOP, &mut out);
        }
        self.frames += 1;
        Ok(out)
    }
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            candle::bail!("LiquidCodec stream was already finished")
        };
        self.finished = true;
        let mut out = Vec::new();
        if self.frames > 0 {
            self.take((FFT - HOP) / 2, &mut out);
        }
        Ok(out)
    }
}
