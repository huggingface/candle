//! Native LFM3-Audio frontend and multimodal language model components.
use super::lfm3::{self, Linear, Stack, Weights};
use candle::{Device, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm};
use serde::Deserialize;
pub mod codec;
pub mod depth;
pub mod frontend;
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    #[serde(alias = "dim_in")]
    pub num_mel_bins: usize,
    #[serde(alias = "dim_conv")]
    pub conv_channels: usize,
    #[serde(alias = "dim_out")]
    pub hidden_size: usize,
    pub sliding_window: usize,
    #[serde(alias = "lfm")]
    pub text_config: serde_json::Value,
}
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(alias = "lfm")]
    pub text_config: lfm3::Config,
    #[serde(alias = "encoder")]
    pub audio_encoder_config: EncoderConfig,
    #[serde(alias = "depthformer")]
    pub depth_config: lfm3::Config,
    #[serde(alias = "codebooks")]
    pub num_codebooks: usize,
    #[serde(alias = "audio_vocab_size")]
    pub codebook_size: usize,
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
    #[serde(default = "default_maskgit")]
    pub maskgit_steps: usize,
}
fn default_maskgit() -> usize {
    3
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        self.text_config.validate()?;
        self.depth_config.validate()?;
        if self.num_codebooks != 12
            || self.codebook_size != 2017
            || self.maskgit_steps == 0
            || self.maskgit_steps > self.num_codebooks
            || self.interleaved_n_text == 0
            || self.interleaved_n_audio == 0
            || self.audio_encoder_config.num_mel_bins != 128
        {
            candle::bail!("unsupported LFM3-Audio codec, MaskGIT, or frontend configuration");
        }
        Ok(())
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let cfg: Self = serde_json::from_slice(bytes).map_err(candle::Error::wrap)?;
        cfg.validate()?;
        Ok(cfg)
    }
}

#[derive(Clone)]
pub struct AudioEncoder {
    conv_in: Conv2d,
    conv_mid: Conv2d,
    lin_mid: Conv2d,
    conv_end: Conv2d,
    lin_end: Conv2d,
    lin_out: Conv1d,
    stack: Stack,
}
impl AudioEncoder {
    pub fn new(config: &EncoderConfig, weights: &Weights<'_>) -> Result<Self> {
        let channels = config.conv_channels;
        let depth = Conv2dConfig {
            groups: channels,
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let make = |name: &str, index: usize, input, output, kernel, cfg| -> Result<Conv2d> {
            Ok(Conv2d::new(
                weights.dense(
                    (output, input, kernel, kernel),
                    &format!("encoder.downsample.{name}.weight"),
                    &format!("a.conv1d.{index}.weight"),
                )?,
                Some(weights.dense(
                    output,
                    &format!("encoder.downsample.{name}.bias"),
                    &format!("a.conv1d.{index}.bias"),
                )?),
                cfg,
            ))
        };
        let mut text = config.text_config.clone();
        let object = text
            .as_object_mut()
            .ok_or_else(|| candle::Error::Msg("invalid LFM3 audio encoder config".into()))?;
        let auto = object
            .get("block_auto_adjust_ff_dim")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if auto {
            let intermediate = object["intermediate_size"].as_u64().unwrap_or(0) as usize;
            let multiple = object["block_multiple_of"].as_u64().unwrap_or(256) as usize;
            let multiplier = object
                .get("block_ffn_dim_multiplier")
                .and_then(|v| v.as_f64());
            let size = 2 * intermediate / 3;
            let size = multiplier.map_or(size, |m| {
                ((size as f64 * m) as usize).div_ceil(multiple) * multiple
            });
            object.insert("intermediate_size".into(), size.into());
        }
        object.insert("use_sandwich_norm".into(), false.into());
        object.insert("sliding_window".into(), config.sliding_window.into());
        if let Some(layers) = object.get_mut("layer_types").and_then(|v| v.as_array_mut()) {
            for kind in layers {
                if kind == "full_attention" {
                    *kind = "sliding_attention".into();
                }
            }
        }
        let cfg: lfm3::Config = serde_json::from_value(text).map_err(candle::Error::wrap)?;
        Ok(Self {
            conv_in: make("conv_in", 0, 1, channels, 3, conv)?,
            conv_mid: make("conv_mid", 2, 1, channels, 3, depth)?,
            lin_mid: make("lin_mid", 3, channels, channels, 1, Conv2dConfig::default())?,
            conv_end: make("conv_end", 5, 1, channels, 3, depth)?,
            lin_end: make("lin_end", 6, channels, channels, 1, Conv2dConfig::default())?,
            lin_out: Conv1d::new(
                weights.dense(
                    (config.hidden_size, config.num_mel_bins * channels / 8, 1),
                    "encoder.downsample.lin_out.weight",
                    "a.pre_encode.out.weight",
                )?,
                Some(weights.dense(
                    config.hidden_size,
                    "encoder.downsample.lin_out.bias",
                    "a.pre_encode.out.bias",
                )?),
                Conv1dConfig::default(),
            ),
            stack: Stack::new(&cfg, weights, "encoder.lfm.lfm", "a.", true, false, false)?,
        })
    }
    pub fn reset(&mut self) {
        self.stack.reset();
    }
    pub fn forward(&mut self, features: &Tensor) -> Result<Tensor> {
        let x = self.conv_in.forward(&features.unsqueeze(1)?)?.silu()?;
        let x = self.lin_mid.forward(&self.conv_mid.forward(&x)?)?.silu()?;
        let x = self.lin_end.forward(&self.conv_end.forward(&x)?)?.silu()?;
        self.stack
            .forward(&self.lin_out.forward(&x.flatten(1, 2)?)?.transpose(1, 2)?)
    }
}
#[derive(Clone)]
pub struct Projector {
    norm: LayerNorm,
    input: Linear,
    output: Linear,
}
impl Projector {
    pub fn new(input: usize, output: usize, w: &Weights<'_>) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::new(
                w.dense(input, "mlp.model.0.weight", "mm.a.mlp.0.weight")?,
                w.dense(input, "mlp.model.0.bias", "mm.a.mlp.0.bias")?,
                1e-5,
            ),
            input: w.linear(input, output, "mlp.model.1", "mm.a.mlp.1", true)?,
            output: w.linear(output, output, "mlp.model.3", "mm.a.mlp.3", true)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.output
            .forward(&self.input.forward(&self.norm.forward(x)?)?.gelu_erf()?)
    }
}
/// Audio/text inference state. Speech generation is provided by the depth decoder
/// and LiquidCodec components, sharing this backbone's generated hidden states.
#[derive(Clone)]
pub struct Model {
    pub language: lfm3::Model,
    pub config: Config,
    depth: depth::DepthDecoder,
    codec: codec::Codec,
    encoder: AudioEncoder,
    projector: Projector,
    device: Device,
}
impl Model {
    pub fn from_gguf(
        model: impl AsRef<std::path::Path>,
        projector: impl AsRef<std::path::Path>,
        config: &Config,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let language = lfm3::Model::from_gguf(model, &config.text_config, device)?;
        let w = Weights::Gguf(crate::quantized_var_builder::VarBuilder::from_gguf(
            projector, device,
        )?);
        let encoder = AudioEncoder::new(&config.audio_encoder_config, &w)?;
        let projector = Projector::new(
            config.audio_encoder_config.hidden_size,
            config.text_config.hidden_size,
            &w,
        )?;
        let depth = depth::DepthDecoder::new(config, &w)?;
        let codec = codec::Codec::new(&w)?;
        Ok(Self {
            depth,
            codec,
            config: config.clone(),
            language,
            encoder,
            projector,
            device: device.clone(),
        })
    }
    pub fn encode_audio(&mut self, samples: &[f32]) -> Result<Tensor> {
        let mel = frontend::log_mel(samples, &self.device)?;
        self.projector.forward(&self.encoder.forward(&mel)?)
    }
    pub fn fork(&self) -> Self {
        let mut model = self.clone();
        model.reset();
        model
    }
    pub fn reset_audio(&mut self) {
        self.encoder.reset();
        self.codec.reset();
    }
    pub fn reset(&mut self) {
        self.language.reset();
        self.reset_audio();
    }
    pub fn forward_tokens(&mut self, tokens: &Tensor) -> Result<Tensor> {
        self.language.forward(tokens)
    }
    pub fn forward_embeds(&mut self, embeddings: &Tensor) -> Result<Tensor> {
        let hidden = self.language.forward_hidden(embeddings)?;
        self.language.project(&hidden)
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMode {
    Text,
    Audio,
    Interleaved,
}
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub mode: GenerationMode,
    pub max_steps: usize,
    pub seed: u64,
    pub audio_temperature: f64,
    pub audio_top_k: usize,
    pub eos_tokens: Vec<u32>,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            mode: GenerationMode::Text,
            max_steps: 512,
            seed: 42,
            audio_temperature: 1.,
            audio_top_k: 128,
            eos_tokens: vec![124900],
        }
    }
}
pub enum GenerationEvent {
    Token(u32),
    Audio(Vec<f32>),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    Length,
}
impl Model {
    /// Prefill a sequence of token/audio embeddings and generate natively. The
    /// callback is invoked between model steps and may stop work by returning an
    /// error (including an application's cooperative cancellation error).
    pub fn generate(
        &mut self,
        prefill: &Tensor,
        options: &GenerationOptions,
        mut emit: impl FnMut(GenerationEvent) -> Result<()>,
    ) -> Result<FinishReason> {
        use candle::IndexOp;
        use rand::SeedableRng;
        if options.max_steps == 0 {
            candle::bail!("LFM3 generation budget must be positive")
        }
        self.codec.reset();
        let mut rng = rand::rngs::StdRng::seed_from_u64(options.seed);
        let hidden = self.language.forward_hidden(prefill)?;
        let mut hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
        let mut mode = options.mode;
        let mut speaking = mode == GenerationMode::Audio;
        let mut text_done = false;
        let mut slots = self.config.interleaved_n_text;
        let mut history = Vec::new();
        let mut reason = FinishReason::Length;
        for _ in 0..options.max_steps {
            // An empty event performs no user-visible output but is not needed:
            // every frame/token callback occurs before the next model step.
            let interleaved = mode == GenerationMode::Interleaved;
            let input = if mode == GenerationMode::Audio
                || (mode == GenerationMode::Interleaved && speaking)
            {
                let mut codes = self.depth.sample(
                    &hidden,
                    &history,
                    &mut rng,
                    options.audio_temperature,
                    options.audio_top_k,
                )?;
                let terminal = codes.contains(&((self.config.codebook_size - 1) as u32));
                if terminal {
                    codes.fill((self.config.codebook_size - 1) as u32);
                    mode = GenerationMode::Text;
                    speaking = false;
                } else {
                    let pcm = self.codec.decode_frame(&codes)?;
                    history.push(codes.clone());
                    if !pcm.is_empty() {
                        emit(GenerationEvent::Audio(pcm))?;
                    }
                }
                if interleaved && !text_done {
                    slots -= 1;
                    if slots == 0 {
                        speaking = !speaking;
                        slots = if speaking {
                            self.config.interleaved_n_audio
                        } else {
                            self.config.interleaved_n_text
                        };
                    }
                }
                self.depth.feedback(&codes)?
            } else {
                let logits = self
                    .language
                    .project(&hidden)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let token = logits
                    .iter()
                    .enumerate()
                    .filter(|(id, v)| {
                        v.is_finite()
                            && ![125031usize, 125032].contains(id)
                            && (mode != GenerationMode::Text || ![125012usize, 125015].contains(id))
                    })
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .ok_or_else(|| candle::Error::Msg("LFM3 has no finite token logits".into()))?;
                if options.eos_tokens.contains(&token) {
                    reason = FinishReason::Eos;
                    break;
                }
                emit(GenerationEvent::Token(token))?;
                if token == 125014 && interleaved {
                    text_done = true;
                    speaking = true;
                } else if !interleaved {
                    if token == 125012 {
                        mode = GenerationMode::Audio;
                        speaking = true;
                    } else if token == 125015 {
                        mode = GenerationMode::Interleaved;
                        speaking = false;
                        slots = self.config.interleaved_n_text;
                    }
                }
                if interleaved && !text_done {
                    slots -= 1;
                    if slots == 0 {
                        speaking = !speaking;
                        slots = if speaking {
                            self.config.interleaved_n_audio
                        } else {
                            self.config.interleaved_n_text
                        };
                    }
                }
                self.language
                    .embed(&Tensor::new(&[token], &self.device)?.reshape((1, 1))?)?
            };
            hidden = self.language.forward_hidden(&input)?;
        }
        let tail = self.codec.finish()?;
        if !tail.is_empty() {
            emit(GenerationEvent::Audio(tail))?;
        }
        Ok(reason)
    }
}

/// Bounded streaming frontend state. The current native implementation recomputes
/// the causal encoder prefix and emits only stable new rows; it preserves exact
/// chunking semantics while avoiding an unbounded recording. A later optimized
/// implementation may replace this with equivalent convolution/encoder caches.
pub struct AudioStream {
    samples: Vec<f32>,
    emitted: usize,
    max_samples: usize,
    finished: bool,
}
impl AudioStream {
    pub fn new(max_samples: usize) -> Result<Self> {
        if max_samples == 0 {
            candle::bail!("LFM3 audio stream limit must be positive")
        };
        Ok(Self {
            samples: Vec::new(),
            emitted: 0,
            max_samples,
            finished: false,
        })
    }
    pub fn reset(&mut self) {
        self.samples.clear();
        self.emitted = 0;
        self.finished = false;
    }
}
impl Model {
    pub fn encode_audio_chunk(
        &mut self,
        stream: &mut AudioStream,
        samples: &[f32],
        finalize: bool,
    ) -> Result<Tensor> {
        if stream.finished
            || stream
                .samples
                .len()
                .checked_add(samples.len())
                .is_none_or(|n| n > stream.max_samples)
            || samples.iter().any(|v| !v.is_finite())
        {
            candle::bail!("invalid, closed, or over-limit LFM3 audio stream")
        }
        stream.samples.extend_from_slice(samples);
        let complete_mels = if stream.samples.len() < 400 {
            0
        } else {
            (stream.samples.len() - 400) / 160 + 1
        };
        let stable = complete_mels / 8;
        if !finalize && stable <= stream.emitted {
            return Tensor::zeros(
                (1, 0, self.config.text_config.hidden_size),
                candle::DType::F32,
                &self.device,
            );
        }
        let all = self.encode_audio(&stream.samples)?;
        let count = if finalize {
            all.dim(1)?
        } else {
            stable.min(all.dim(1)?)
        };
        let output = all
            .narrow(1, stream.emitted, count - stream.emitted)?
            .contiguous()?;
        stream.emitted = count;
        stream.finished = finalize;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned LFM3 GGUF pair and config; runs native CPU inference"]
    fn native_checkpoint_components() -> Result<()> {
        let directory = std::path::PathBuf::from(
            std::env::var("CANDLE_LFM3_MODEL_DIR").expect("set CANDLE_LFM3_MODEL_DIR"),
        );
        let config_path = std::env::var("CANDLE_LFM3_CONFIG").expect("set CANDLE_LFM3_CONFIG");
        let config = Config::from_json(&std::fs::read(config_path)?)?;
        let mut model = Model::from_gguf(
            directory.join("LFM3-Audio-2B-Q4_K_M.gguf"),
            directory.join("mmproj-LFM3-Audio-2B-Q8_0.gguf"),
            &config,
            &Device::Cpu,
        )?;
        let audio = model.encode_audio(&vec![0.; 1600])?;
        assert_eq!(audio.dims(), &[1, 2, 2048]);
        let mut stream = AudioStream::new(1600)?;
        let a = model.encode_audio_chunk(&mut stream, &vec![0.; 1280], false)?;
        let b = model.encode_audio_chunk(&mut stream, &vec![0.; 320], false)?;
        let c = model.encode_audio_chunk(&mut stream, &[], true)?;
        let streamed = Tensor::cat(&[a, b, c], 1)?;
        let difference = (&audio - streamed)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            difference < 1e-4,
            "streaming encoder changed stable rows {difference}"
        );

        assert!(audio
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        let logits = model.forward_tokens(&Tensor::new(&[[124894u32, 124895]], &Device::Cpu)?)?;
        assert_eq!(logits.dims(), &[1, 1, 128000]);
        assert!(logits
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        model.reset();
        let prefill = model
            .language
            .embed(&Tensor::new(&[[124894u32, 125012]], &Device::Cpu)?)?;
        let mut frames = 0;
        let reason = model.generate(
            &prefill,
            &GenerationOptions {
                mode: GenerationMode::Audio,
                max_steps: 2,
                ..Default::default()
            },
            |event| {
                if let GenerationEvent::Audio(audio) = event {
                    assert!(audio.iter().all(|v| v.is_finite()));
                    frames += audio.len();
                }
                Ok(())
            },
        )?;
        assert_eq!(reason, FinishReason::Length);
        assert!(frames > 0, "native codec generated no frames");
        eprintln!("native LFM3 component smoke generated {frames} PCM frames");
        Ok(())
    }
}
