//! Buffered native MusicGen text-to-music generation and audio continuation.
//! Audio prompts are reconstructed by the codec before the generated extension.
use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::musicgen::{self, GenConfig, MusicgenForConditionalGeneration},
};
use std::path::Path;
use tokenizers::Tokenizer;

// A long audio reference is prefetched in bounded causal blocks. Checks occur
// between blocks instead of waiting for one full 30-second-reference forward.
fn prefill_logits(
    decoder: &mut musicgen::MusicgenForCausalLM,
    ids: &Tensor,
    condition: &Tensor,
    cancel: &CancellationToken,
) -> Result<Tensor> {
    let length = ids.dim(1)?;
    let mut last = None;
    for offset in (0..length).step_by(32) {
        cancel.check()?;
        let size = (length - offset).min(32);
        let logits = decoder
            .forward_conditioned(&ids.narrow(1, offset, size)?.contiguous()?, Some(condition))?;
        last = Some(logits.i((.., size - 1, ..))?);
    }
    last.ok_or_else(|| Error::InvalidInput("MusicGen prefill cannot be empty".into()))
}

fn continuation_codes(
    decoder: &mut musicgen::MusicgenForCausalLM,
    pattern: &musicgen::DelayPattern,
    condition: &Tensor,
    request: &AudioGenerationRequest,
    sampler: &mut LogitsProcessor,
    cancel: &CancellationToken,
) -> Result<Tensor> {
    let books = pattern.codebooks().len();
    let device = condition.device();
    let mut codes = pattern.codebooks().to_vec();
    for book in &mut codes {
        book.resize(pattern.total_frames(), 0);
    }
    let mut current = pattern.prefill()?;
    let mut input_length = pattern.prefill_length();
    for position in pattern.prefill_length()..pattern.delayed_length() {
        cancel.check()?;
        let guidance = request.guidance_scale > 1.;
        let ids = if guidance {
            [current.as_slice(), current.as_slice()].concat()
        } else {
            current.clone()
        };
        let ids = Tensor::from_vec(
            ids,
            (if guidance { 2 * books } else { books }, input_length),
            device,
        )?;
        let logits = prefill_logits(decoder, &ids, condition, cancel)?;
        let logits = if guidance {
            let conditional = logits.narrow(0, 0, books)?;
            let unconditional = logits.narrow(0, books, books)?;
            (&unconditional + ((conditional - &unconditional)? * request.guidance_scale)?)?
        } else {
            logits
        };
        current = Vec::with_capacity(books);
        for (book, values) in codes.iter_mut().enumerate() {
            let sampled = sampler.sample(&logits.i(book)?)?;
            let token = match pattern.fixed_token(position, book)? {
                Some(token) => token,
                None => {
                    let frame = musicgen::delayed_frame(
                        position,
                        book,
                        pattern.channels(),
                        pattern.total_frames(),
                    )
                    .ok_or_else(|| {
                        Error::InvalidState("MusicGen continuation mask lost its code frame".into())
                    })?;
                    values[frame] = sampled;
                    sampled
                }
            };
            current.push(token);
        }
        input_length = 1;
    }
    Tensor::from_vec(
        codes.into_iter().flatten().collect::<Vec<_>>(),
        (1, books, pattern.total_frames()),
        device,
    )
    .map_err(Error::from)
}

pub struct MusicGen {
    model: MusicgenForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
    format: AudioFormat,
    max_steps: usize,
    finish: FinishReason,
}
impl MusicGen {
    pub fn load(directory: &Path, device: &Device, options: &serde_json::Value) -> Result<Self> {
        let config: GenConfig =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        config.validate()?;
        let frames_per_second = config.encodec.sampling_rate
            / config.encodec.upsampling_ratios.iter().product::<usize>();
        let available = config
            .musicgen
            .max_position_embeddings
            .saturating_sub(config.musicgen.num_codebooks / config.musicgen.audio_channels);
        let max_steps = options
            .get("max_steps")
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        Error::InvalidInput("MusicGen max_steps must be an integer".into())
                    })
            })
            .transpose()?
            .unwrap_or(30 * frames_per_second)
            .min(30 * frames_per_second)
            .min(available);
        if max_steps == 0 {
            return Err(Error::InvalidInput(
                "MusicGen output budget must be positive".into(),
            ));
        }
        let format = AudioFormat::new(
            config.encodec.sampling_rate as u32,
            config.musicgen.audio_channels as u16,
        )?;
        let weights = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &super::load::weights(directory)?,
                DType::F32,
                device,
            )?
        };
        let model = MusicgenForConditionalGeneration::load(weights, config)?;
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            format,
            max_steps,
            finish: FinishReason::Completed,
        })
    }
    pub fn frame_rate(&self) -> f64 {
        let cfg = &self.model.config().encodec;
        cfg.sampling_rate as f64 / cfg.upsampling_ratios.iter().product::<usize>() as f64
    }
    fn continuation(
        &mut self,
        reference: &AudioBuffer,
        request: &AudioGenerationRequest,
        condition: &Tensor,
        sampler: &mut LogitsProcessor,
        cancel: &CancellationToken,
        expected_prefix: usize,
    ) -> Result<Tensor> {
        cancel.check()?;
        let tensor = Tensor::from_vec(
            reference.samples().to_vec(),
            (
                1,
                reference.frames(),
                reference.format().channels() as usize,
            ),
            &self.device,
        )?
        .transpose(1, 2)?
        .contiguous()?;
        let prefix = self.model.encode_audio(&tensor)?;
        cancel.check()?;
        let (batch, books, frames) = prefix.dims3()?;
        if batch != 1
            || frames != expected_prefix
            || books != self.model.config().musicgen.num_codebooks
        {
            return Err(Error::InvalidInput(
                "MusicGen reference encoder returned an incompatible code shape".into(),
            ));
        }
        let prefix = prefix.i(0)?.to_vec2::<u32>()?;
        let cfg = &self.model.config().musicgen;
        let pattern = musicgen::DelayPattern::new(
            prefix,
            request.max_steps,
            cfg.audio_channels,
            cfg.pad_token_id as u32,
        )?;
        continuation_codes(
            &mut self.model.decoder,
            &pattern,
            condition,
            request,
            sampler,
            cancel,
        )
    }

    fn prepare(
        &self,
        request: &AudioGenerationRequest,
        cancel: &CancellationToken,
    ) -> Result<(Vec<u32>, usize)> {
        request.validate()?;
        cancel.check()?;
        let hop = self
            .model
            .config()
            .encodec
            .upsampling_ratios
            .iter()
            .product::<usize>();
        let prefix_frames = match &request.audio_reference {
            None => 0,
            Some(reference) => {
                if reference.is_empty() || reference.format() != self.format {
                    return Err(Error::InvalidInput("MusicGen reference must be nonempty and match its advertised PCM format; convert explicitly".into()));
                }
                reference.frames().div_ceil(hop)
            }
        };
        let total_frames = prefix_frames
            .checked_add(request.max_steps)
            .ok_or_else(|| Error::LimitExceeded("MusicGen total duration overflow".into()))?;
        if total_frames > self.max_steps {
            return Err(Error::LimitExceeded(format!(
                "MusicGen reconstructed prompt plus continuation exceeds {} codec frames",
                self.max_steps
            )));
        }
        if request.max_steps > self.max_steps {
            return Err(Error::LimitExceeded(format!(
                "MusicGen output exceeds {} codec frames",
                self.max_steps
            )));
        }
        let tokens = self
            .tokenizer
            .encode(request.prompt.as_str(), true)
            .map_err(|e| Error::InvalidInput(e.to_string()))?
            .get_ids()
            .to_vec();
        if tokens.len() > 512 {
            return Err(Error::LimitExceeded(
                "MusicGen text condition exceeds 512 tokens".into(),
            ));
        }
        Ok((tokens, prefix_frames))
    }
    fn run(
        &mut self,
        request: &AudioGenerationRequest,
        cancel: &CancellationToken,
        sink: &mut dyn FnMut(AudioBuffer) -> Result<()>,
        prepared: (Vec<u32>, usize),
    ) -> Result<()> {
        let (tokens, prefix_frames) = prepared;
        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        self.model.reset();
        let condition = self.model.encode_text(&tokens)?;
        cancel.check()?;
        let cfg = &self.model.config().musicgen;
        let books = cfg.num_codebooks;
        let channels = cfg.audio_channels;
        let pad = cfg.pad_token_id as u32;
        let guidance = request.guidance_scale > 1.;
        let condition = if guidance {
            Tensor::cat(&[&condition, &condition.zeros_like()?], 0)?
        } else {
            condition
        };
        let sampling = match (request.temperature, request.top_k, request.top_p) {
            (None, _, _) => Sampling::ArgMax,
            (Some(temperature), Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            (Some(temperature), Some(k), None) => Sampling::TopK { k, temperature },
            (Some(temperature), None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(temperature), None, None) => Sampling::All { temperature },
        };
        let mut sampler = LogitsProcessor::from_sampling(request.seed, sampling);
        let codes = if let Some(reference) = &request.audio_reference {
            self.continuation(
                reference,
                request,
                &condition,
                &mut sampler,
                cancel,
                prefix_frames,
            )?
        } else {
            let mut codes = vec![vec![0u32; request.max_steps]; books];
            let mut current = vec![pad; books];
            let total = request.max_steps + books / channels - 1;
            for position in 1..=total {
                cancel.check()?;
                let ids = if guidance {
                    [current.as_slice(), current.as_slice()].concat()
                } else {
                    current.clone()
                };
                let ids = Tensor::from_vec(
                    ids,
                    (if guidance { 2 * books } else { books }, 1),
                    &self.device,
                )?;
                let logits = self
                    .model
                    .decoder
                    .forward_conditioned(&ids, Some(&condition))?
                    .squeeze(1)?;
                let logits = if guidance {
                    let cond = logits.narrow(0, 0, books)?;
                    let uncond = logits.narrow(0, books, books)?;
                    (&uncond + ((cond - &uncond)? * request.guidance_scale)?)?
                } else {
                    logits
                };
                for (book, values) in codes.iter_mut().enumerate() {
                    // Sample every row before applying the mask, as Transformers does.
                    let sampled = sampler.sample(&logits.i(book)?)?;
                    current[book] = match musicgen::delayed_frame(
                        position,
                        book,
                        channels,
                        request.max_steps,
                    ) {
                        Some(frame) => {
                            values[frame] = sampled;
                            sampled
                        }
                        None => pad,
                    };
                }
            }
            cancel.check()?;
            Tensor::from_vec(
                codes.into_iter().flatten().collect::<Vec<_>>(),
                (1, books, request.max_steps),
                &self.device,
            )?
        };
        let audio = self.model.decode_audio_with_callback(&codes, || {
            cancel
                .check()
                .map_err(|error| candle::Error::Msg(error.to_string()))
        });
        // Preserve the typed cancellation/deadline error across the core tensor
        // callback, and suppress the entire buffered response on cancellation.
        cancel.check()?;
        let audio = audio?;
        let (batch, channels, frames) = audio.dims3()?;
        if batch != 1 || channels != self.format.channels() as usize {
            return Err(Error::InvalidInput(
                "MusicGen decoded an incompatible PCM shape".into(),
            ));
        }
        let samples = audio
            .i(0)?
            .transpose(0, 1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        if frames > self.format.sample_rate() as usize * 30 {
            return Err(Error::LimitExceeded(
                "MusicGen decoded audio exceeds 30 seconds".into(),
            ));
        }
        sink(AudioBuffer::new(self.format, samples)?)?;
        self.finish = FinishReason::Length;
        Ok(())
    }
}
impl AudioGenerationModel for MusicGen {
    fn descriptor(&self) -> ModelCapabilities {
        let accepts_reference = self.max_steps >= 2;
        let hop = self
            .model
            .config()
            .encodec
            .upsampling_ratios
            .iter()
            .product::<usize>();
        ModelCapabilities {
            model_id: "musicgen".into(),
            tasks: vec![AudioTask::AudioGeneration],
            output_formats: vec![self.format],
            input_formats: if accepts_reference {
                vec![self.format]
            } else {
                Vec::new()
            },
            max_input_frames: accepts_reference.then_some(self.max_steps.saturating_sub(1) * hop),
            accepts_audio_reference: accepts_reference,
            max_text_bytes: Some(16384),
            max_context_tokens: Some(self.model.config().musicgen.max_position_embeddings),
            input_streaming: accepts_reference.then_some(StreamingMode::Buffered),
            output_streaming: Some(StreamingMode::Buffered),
            ..Default::default()
        }
    }
    fn generate(
        &mut self,
        request: &AudioGenerationRequest,
        cancel: &CancellationToken,
        sink: &mut dyn FnMut(AudioBuffer) -> Result<()>,
    ) -> Result<()> {
        let prepared = match self.prepare(request, cancel) {
            Ok(prepared) => prepared,
            Err(error) => {
                match &error {
                    Error::Cancelled => self.finish = FinishReason::Cancelled,
                    Error::DeadlineExceeded => self.finish = FinishReason::Deadline,
                    _ => {}
                }
                return Err(error);
            }
        };
        let result = self.run(request, cancel, sink, prepared);
        self.model.reset();
        match &result {
            Err(Error::Cancelled) => self.finish = FinishReason::Cancelled,
            Err(Error::DeadlineExceeded) => self.finish = FinishReason::Deadline,
            Err(_) => self.finish = FinishReason::Completed,
            _ => {}
        }
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset();
        self.finish = FinishReason::Completed;
        Ok(())
    }
    fn finish_reason(&self) -> FinishReason {
        self.finish
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuation_codes_match_official_transformers_generate_exactly() -> Result<()> {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/musicgen-continuation.json"
        ))?;
        let values = |value: &serde_json::Value| {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let device = Device::Cpu;
        let cfg: musicgen::Config = serde_json::from_value(f["config"].clone())?;
        let weights = f["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                let shape = value["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                Ok((
                    name.clone(),
                    Tensor::from_vec(values(&value["values"]), shape, &device)?,
                ))
            })
            .collect::<candle::Result<std::collections::HashMap<_, _>>>()?;
        let mut decoder = musicgen::MusicgenForCausalLM::load(
            VarBuilder::from_tensors(weights, DType::F32, &device),
            &cfg,
        )?;
        let prefix = f["continuation"]["prefix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let prefix = prefix
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        let pattern = musicgen::DelayPattern::new(prefix.clone(), 4, 1, 8)?;
        let context = Tensor::from_vec(values(&f["context"]), (1, 3, 8), &device)?;
        let mut request = AudioGenerationRequest::new("fixture");
        request.max_steps = 4;
        request.temperature = None;
        request.guidance_scale = 1.;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let actual = continuation_codes(
            &mut decoder,
            &pattern,
            &context,
            &request,
            &mut sampler,
            &CancellationToken::default(),
        )?;
        let expected = f["continuation"]["expected_codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(actual.flatten_all()?.to_vec1::<u32>()?, expected);
        for (book, expected) in prefix.iter().enumerate() {
            assert_eq!(actual.i((0, book, ..3))?.to_vec1::<u32>()?, *expected);
        }
        decoder.reset();
        let long = f["continuation"]["long_prefix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let long = long
            .as_chunks::<33>()
            .0
            .iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        let long_pattern = musicgen::DelayPattern::new(long, 4, 1, 8)?;
        let actual = continuation_codes(
            &mut decoder,
            &long_pattern,
            &context,
            &request,
            &mut sampler,
            &CancellationToken::default(),
        )?;
        let expected = f["continuation"]["long_expected_codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            actual.flatten_all()?.to_vec1::<u32>()?,
            expected,
            "bounded prefill must preserve HF full-prefill continuation"
        );
        decoder.reset();
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            continuation_codes(
                &mut decoder,
                &pattern,
                &context,
                &request,
                &mut sampler,
                &cancelled
            ),
            Err(Error::Cancelled)
        ));
        Ok(())
    }
}
