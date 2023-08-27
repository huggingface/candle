use crate::model::{Config, Whisper};
use anyhow::Error as E;
use candle::{safetensors::Load, DType, Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use rand::{distributions::Distribution, rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use yew_agent::{HandlerId, Public, WorkerLink};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

pub const DTYPE: DType = DType::F32;

// Audio parameters.
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const N_MELS: usize = 80;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000 samples in a 30-second chunk
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 frames in a mel spectrogram input

pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
pub const LOGPROB_THRESHOLD: f64 = -1.0;
pub const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

// Tokenizer dependent bits.
pub const SOT_TOKEN: u32 = 50257;
pub const EOT_TOKEN: u32 = 50256;
pub const NO_SPEECH_TOKEN: u32 = 50361;
// From the _get_suppress_tokens function + 50362 (no timestamp)
// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/decoding.py#L605
pub const SUPPRESS_TOKENS: [u32; 91] = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357,
    366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782,
    1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959,
    10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992,
    19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549,
    47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361, 50362,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

pub struct Decoder {
    model: Whisper,
    mel_filters: Vec<f32>,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
}

impl Decoder {
    fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        mel_filters: Vec<f32>,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if SUPPRESS_TOKENS.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        Ok(Self {
            model,
            mel_filters,
            tokenizer,
            suppress_tokens,
        })
    }

    fn decode(&self, mel: &Tensor, t: f64, rng: &mut StdRng) -> anyhow::Result<DecodingResult> {
        let model = &self.model;
        let audio_features = model.encoder.forward(mel)?;
        console_log!("audio features: {:?}", audio_features.dims());
        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![SOT_TOKEN];
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let logits = model.decoder.forward(&tokens_t, &audio_features)?;
            let logits = logits.squeeze(0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                no_speech_prob = softmax(&logits.get(0)?, 0)?
                    .get(NO_SPEECH_TOKEN as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (seq_len, _) = logits.dims2()?;
            let logits = logits
                .get(seq_len - 1)?
                .broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .get(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == EOT_TOKEN || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(
        &self,
        segment: &Tensor,
        rng: &mut StdRng,
    ) -> anyhow::Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, _> = self.decode(segment, t, rng);
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    console_log!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&self, mel: &Tensor) -> anyhow::Result<Vec<Segment>> {
        let mut rng = StdRng::seed_from_u64(299792458);
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment, &mut rng)?;
            seek += segment_size;
            if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
                console_log!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            console_log!("{seek}: {segment:?}");
            segments.push(segment)
        }
        Ok(segments)
    }

    fn load(md: ModelData) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let tokenizer = Tokenizer::from_bytes(&md.tokenizer).map_err(anyhow::Error::msg)?;

        let mel_filters = safetensors::tensor::SafeTensors::deserialize(&md.mel_filters)?;
        let mel_filters = mel_filters.tensor("mel_80")?.load(&device)?;
        console_log!("loaded mel filters {:?}", mel_filters.shape());
        let mel_filters = mel_filters.flatten_all()?.to_vec1::<f32>()?;
        let weights = safetensors::tensor::SafeTensors::deserialize(&md.weights)?;
        let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
        let config = Config::tiny_en();
        let whisper = Whisper::load(&vb, config)?;
        console_log!("done loading model");
        let decoder = Self::new(whisper, tokenizer, mel_filters, &device)?;
        Ok(decoder)
    }

    fn convert_and_run(&self, wav_input: &[u8]) -> anyhow::Result<Vec<Segment>> {
        let device = Device::Cpu;
        let mut wav_input = std::io::Cursor::new(wav_input);
        let (header, data) = wav::read(&mut wav_input)?;
        console_log!("loaded wav data: {header:?}");
        if header.sampling_rate != SAMPLE_RATE as u32 {
            anyhow::bail!("wav file must have a {SAMPLE_RATE} sampling rate");
        }
        let data = data.as_sixteen().expect("expected 16 bit wav file");
        let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect();
        console_log!("pcm data loaded {}", pcm_data.len());
        let mel = crate::audio::pcm_to_mel(&pcm_data, &self.mel_filters)?;
        let mel_len = mel.len();
        let mel = Tensor::from_vec(mel, (1, N_MELS, mel_len / N_MELS), &device)?;
        console_log!("loaded mel: {:?}", mel.dims());
        let segments = self.run(&mel)?;
        Ok(segments)
    }
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transfered via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub tokenizer: Vec<u8>,
    pub mel_filters: Vec<u8>,
    pub weights: Vec<u8>,
}

pub struct Worker {
    link: WorkerLink<Self>,
    decoder: Option<Decoder>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    DecodeTask { wav_bytes: Vec<u8> },
}

#[derive(Serialize, Deserialize)]
pub enum WorkerOutput {
    Decoded(Vec<Segment>),
    WeightsLoaded,
}

impl yew_agent::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = Result<WorkerOutput, String>;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Self {
            link,
            decoder: None,
        }
    }

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        let output = match msg {
            WorkerInput::ModelData(md) => match Decoder::load(md) {
                Ok(decoder) => {
                    self.decoder = Some(decoder);
                    Ok(WorkerOutput::WeightsLoaded)
                }
                Err(err) => Err(format!("model creation error {err:?}")),
            },
            WorkerInput::DecodeTask { wav_bytes } => match &self.decoder {
                None => Err("model has not been set".to_string()),
                Some(decoder) => decoder
                    .convert_and_run(&wav_bytes)
                    .map(WorkerOutput::Decoded)
                    .map_err(|e| e.to_string()),
            },
        };
        self.link.respond(id, output);
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}
