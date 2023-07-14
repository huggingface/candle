#![allow(dead_code)]
use anyhow::Error as E;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

mod audio;
mod model;
use model::{Config, Whisper};

const DTYPE: DType = DType::F32;

// Audio parameters.
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const N_MELS: usize = 80;
const HOP_LENGTH: usize = 160;
const CHUNK_LENGTH: usize = 30;
const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000 samples in a 30-second chunk
const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 frames in a mel spectrogram input
const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2; // the initial convolutions has stride 2
const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP_LENGTH; // 10ms per audio frame
const TOKENS_PER_SECOND: usize = SAMPLE_RATE / N_SAMPLES_PER_TOKEN; // 20ms per audio token

const NO_SPEECH_THRESHOLD: f64 = 0.6;
const LOGPROB_THRESHOLD: f64 = -1.0;
const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

// Tokenizer dependent bits.
const SOT_TOKEN: u32 = 50257;
const EOT_TOKEN: u32 = 50256;
const NO_SPEECH_TOKEN: u32 = 50361;
const NO_TIMESTAMP_TOKEN: u32 = 50362;
// From the _get_suppress_tokens function + 50362 (no timestamp)
// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/decoding.py#L605
const SUPPRESS_TOKENS: [u32; 91] = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357,
    366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782,
    1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959,
    10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992,
    19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549,
    47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361, 50362,
];

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Whisper,
    rng: rand::rngs::StdRng,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
}

impl Decoder {
    fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        seed: u64,
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
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            suppress_tokens,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> anyhow::Result<DecodingResult> {
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
                no_speech_prob = logits
                    .get(0)?
                    .softmax(0)?
                    .get(NO_SPEECH_TOKEN as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (seq_len, _) = logits.shape().r2()?;
            let logits = logits
                .get(seq_len - 1)?
                .broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = (&logits / t)?.softmax(0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
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
            let prob = logits
                .softmax(candle::D::Minus1)?
                .get(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == EOT_TOKEN || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self
            .tokenizer
            .decode(tokens.clone(), true)
            .map_err(E::msg)?;
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

    fn decode_with_fallback(&mut self, segment: &Tensor) -> anyhow::Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, _> = self.decode(segment, t);
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

    fn run(&mut self, mel: &Tensor) -> anyhow::Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.shape().r3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
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
}

async fn fetch_url(url: &str) -> Result<Vec<u8>, JsValue> {
    use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};
    let window = web_sys::window().ok_or("window")?;
    let mut opts = RequestInit::new();
    let opts = opts
        .method("GET")
        .mode(RequestMode::Cors)
        .cache(RequestCache::NoCache);

    let request = Request::new_with_str_and_init(url, opts)?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into()?;
    let data = JsFuture::from(resp.blob()?).await?;
    let blob = web_sys::Blob::from(data);
    let array_buffer = JsFuture::from(blob.array_buffer()).await?;
    let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
    Ok(data)
}

fn w<T: ToString>(x: T) -> String {
    x.to_string()
}

async fn run_impl() -> Result<(), JsValue> {
    let device = Device::Cpu;
    let tokenizer_config = fetch_url("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_config).map_err(w)?;

    let mel_filters = fetch_url("mel_filters.safetensors").await?;
    let mel_filters = candle::safetensors::SafeTensors::from_buffer(&mel_filters).map_err(w)?;
    let mel_filters = mel_filters.tensor("mel_80", &device).map_err(w)?;
    console_log!("loaded mel filters {:?}", mel_filters.shape());
    let mel_filters = mel_filters
        .flatten_all()
        .map_err(w)?
        .to_vec1::<f32>()
        .map_err(w)?;

    let wav_input = fetch_url("jfk.wav").await?;
    let mut wav_input = std::io::Cursor::new(wav_input);
    let (header, data) = wav::read(&mut wav_input).map_err(w)?;
    console_log!("loaded wav data: {header:?}");
    if header.sampling_rate != SAMPLE_RATE as u32 {
        Err(format!(
            "wav file must have a {} sampling rate",
            SAMPLE_RATE
        ))?
    }
    let data = data.as_sixteen().expect("expected 16 bit wav file");
    let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
        .iter()
        .map(|v| *v as f32 / 32768.)
        .collect();
    console_log!("pcm data loaded {}", pcm_data.len());
    let mel = audio::pcm_to_mel(&pcm_data, &mel_filters).map_err(w)?;
    let mel_len = mel.len();
    let mel = Tensor::from_vec(mel, (1, N_MELS, mel_len / N_MELS), &device).map_err(w)?;
    console_log!("loaded mel: {:?}", mel.dims());

    let weights = fetch_url("tiny.en.safetensors").await?;
    let weights = candle::safetensors::SafeTensors::from_buffer(&weights).map_err(w)?;
    let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
    let config = Config::tiny_en();
    let model = Whisper::load(&vb, config).map_err(w)?;
    let mut dc = Decoder::new(model, tokenizer, 299792458, &device).map_err(w)?;
    dc.run(&mel).map_err(w)?;
    Ok(())
}

fn test_fn_impl() -> anyhow::Result<String> {
    let t1 = Tensor::randn((3, 4), DType::F32, &Device::Cpu, 0., 1.)?;
    let t2 = Tensor::randn((4, 2), DType::F32, &Device::Cpu, 0., 1.)?;
    let t = t1.matmul(&t2)?;
    console_log!("matmul result: {t}");
    let res = format!("Hello Candle!\n\nt1:\n{t1}\n\nt2:\n{t2}\n\nt1@t2:\n{t}\n");
    Ok(res)
}

#[wasm_bindgen]
pub fn test_fn() -> std::result::Result<(), JsValue> {
    let result = match test_fn_impl() {
        Ok(v) => v,
        Err(err) => format!("error: {err:?}"),
    };
    let window = web_sys::window().expect("no global `window` exists");
    let document = window.document().expect("should have a document on window");
    let p_element = document.create_element("p")?;
    p_element.set_text_content(Some(&result));
    let body = document.body().expect("document should have a body");
    body.append_child(&p_element)?;
    Ok(())
}

#[wasm_bindgen]
pub async fn run_fn() -> std::result::Result<(), JsValue> {
    console_log!("run_fn starting...");
    run_impl().await?;
    Ok(())
}
