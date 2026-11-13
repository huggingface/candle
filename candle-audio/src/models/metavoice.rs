//! MetaVoice 1B: speaker-conditioned first stage, six-codebook refinement,
//! and native EnCodec decoding. Both safetensors and GGUF first stages run in Candle.
use crate::*;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{
        encodec,
        metavoice::{adapters, gpt, speaker_encoder, tokenizers, transformer},
        quantized_metavoice::transformer as quantized,
    },
};
use std::path::{Path, PathBuf};

const AUDIO_VOCAB: u32 = 1024;
const EOS: u32 = 2048;
const MAX_TEXT_BYTES: usize = 65_536;

pub enum FirstStage {
    Normal(transformer::Model),
    Quantized(quantized::Model),
}
impl FirstStage {
    fn clear(&mut self) {
        match self {
            Self::Normal(model) => model.clear_kv_cache(),
            Self::Quantized(model) => model.clear_kv_cache(),
        }
    }
    fn forward(
        &mut self,
        tokens: &Tensor,
        speaker: &Tensor,
        position: usize,
        cancel: &CancellationToken,
    ) -> Result<Tensor> {
        let check = |_| {
            cancel
                .check()
                .map_err(|error| candle::Error::Msg(error.to_string()))
        };
        let result = match self {
            Self::Normal(model) => model.forward_with_callback(tokens, speaker, position, check),
            Self::Quantized(model) => model.forward_with_callback(tokens, speaker, position, check),
        };
        cancel.check()?;
        Ok(result?)
    }
}

/// MetaVoice is a buffered synthesizer. It accepts a 256-value speaker embedding,
/// a checkpoint `spk_emb`, or audio references when the native speaker encoder
/// checkpoint is loaded. Reference transcripts are not used for voice extraction.
pub struct MetaVoice {
    first: FirstStage,
    first_config: transformer::Config,
    second: gpt::Model,
    tokenizer: tokenizers::BPE,
    encodec: encodec::Model,
    default_speaker: Option<Tensor>,
    speaker_encoder: Option<(speaker_encoder::Model, Vec<f32>)>,
    device: Device,
    codec_device: Device,
    format: AudioFormat,
    guidance_scale: f64,
    temperature: f64,
    finish: FinishReason,
    chunk_frames: usize,
}

fn validate_request(request: &ConditionedSynthesisRequest) -> Result<()> {
    request.validate()?;
    let r = &request.request;
    if r.text.trim().is_empty() || r.text.len() > MAX_TEXT_BYTES || r.max_steps == 0 {
        return Err(Error::InvalidInput(
            "MetaVoice needs nonempty text <=65536 bytes and nonzero max_steps".into(),
        ));
    }
    if request.speaker_id.is_some()
        || r.voice.is_some()
        || r.description.is_some()
        || r.language
            .as_deref()
            .is_some_and(|language| language != "en")
    {
        return Err(Error::Unsupported("this MetaVoice adapter supports English, speaker embeddings and audio references; speaker IDs, named voices and descriptions are unsupported".into()));
    }
    if request
        .speaker_embedding
        .as_ref()
        .is_some_and(|speaker| speaker.len() != 256)
    {
        return Err(Error::InvalidInput(
            "MetaVoice speaker embedding must have 256 values".into(),
        ));
    }
    if !request.references.is_empty() && request.speaker_embedding.is_some() {
        return Err(Error::InvalidInput(
            "MetaVoice needs either audio references or a speaker embedding, not both".into(),
        ));
    }
    validate_references(&request.references)?;
    Ok(())
}

fn validate_references(references: &[AudioReference]) -> Result<()> {
    if references.len() > 8
        || references
            .iter()
            .map(|reference| reference.audio.duration())
            .sum::<f64>()
            > 30.
    {
        return Err(Error::LimitExceeded(
            "MetaVoice reference budget is eight clips and 30 seconds total".into(),
        ));
    }
    if references
        .iter()
        .any(|reference| reference.audio.is_empty())
    {
        return Err(Error::InvalidInput(
            "MetaVoice references must contain audio".into(),
        ));
    }
    let mut speaker = None;
    for reference in references {
        if let Some(id) = reference.speaker_id {
            if speaker.is_some_and(|previous| previous != id) {
                return Err(Error::InvalidInput(
                    "MetaVoice reference clips must belong to one speaker".into(),
                ));
            }
            speaker = Some(id);
        }
    }
    Ok(())
}

fn validate_metadata(meta: &serde_json::Value) -> Result<()> {
    if meta.get("speaker_cond").and_then(|v| v.as_bool()) != Some(true)
        || meta.get("speaker_emb_size").and_then(|v| v.as_u64()) != Some(256)
    {
        return Err(Error::Unsupported(
            "MetaVoice v0.1 metadata must enable 256-dimensional speaker conditioning".into(),
        ));
    }
    Ok(())
}

fn validate_model_args(args: &serde_json::Value) -> Result<()> {
    for (name, expected) in [
        ("n_layer", 24),
        ("n_head", 16),
        ("n_embd", 2048),
        ("block_size", 2048),
    ] {
        if args.get(name).and_then(|v| v.as_u64()) != Some(expected) {
            return Err(Error::Unsupported(format!(
                "MetaVoice1B v0.1 requires {name}={expected}"
            )));
        }
    }
    for (name, expected) in [("bias", false), ("causal", true), ("spk_emb_on_text", true)] {
        if args.get(name).and_then(|v| v.as_bool()) != Some(expected) {
            return Err(Error::Unsupported(format!(
                "MetaVoice1B v0.1 requires {name}={expected}"
            )));
        }
    }
    if args.get("vocab_sizes") != Some(&serde_json::json!([2562]))
        || args.get("norm_type").and_then(|v| v.as_str()) != Some("rmsnorm")
        || args.get("nonlinearity_type").and_then(|v| v.as_str()) != Some("swiglu")
    {
        return Err(Error::Unsupported(
            "unsupported MetaVoice first-stage vocabulary, normalization or activation".into(),
        ));
    }
    Ok(())
}

fn hierarchies(
    prompt: &[u32],
    ids1: &[u32],
    ids2: &[u32],
    block_size: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    if prompt.iter().any(|id| !(2049..=2561).contains(id))
        || ids1.iter().chain(ids2).any(|id| *id >= AUDIO_VOCAB)
    {
        return Err(Error::InvalidInput(
            "MetaVoice tokenizer/audio codes are outside the supported v0.1 vocabularies".into(),
        ));
    }
    let needed = prompt
        .len()
        .checked_add(ids1.len().max(ids2.len()))
        .and_then(|n| n.checked_add(1));
    if needed.is_none_or(|n| n > block_size) {
        return Err(Error::LimitExceeded(
            "MetaVoice second-stage context would truncate text or audio".into(),
        ));
    }
    let mut h1: Vec<_> = prompt
        .iter()
        .map(|id| id - 1024)
        .chain(ids1.iter().copied())
        .chain([AUDIO_VOCAB])
        .collect();
    let mut h2: Vec<_> = std::iter::repeat_n(AUDIO_VOCAB, prompt.len())
        .chain(ids2.iter().copied())
        .chain([AUDIO_VOCAB])
        .collect();
    h1.resize(block_size, AUDIO_VOCAB);
    h2.resize(block_size, AUDIO_VOCAB);
    Ok((h1, h2))
}

fn path_option(
    directory: &Path,
    options: &serde_json::Value,
    name: &str,
    default: &str,
) -> Result<PathBuf> {
    match options.get(name) {
        None => Ok(directory.join(default)),
        Some(value) => value
            .as_str()
            .map(|path| directory.join(path))
            .ok_or_else(|| Error::InvalidInput(format!("{name} must be a path string"))),
    }
}

impl MetaVoice {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        first: FirstStage,
        first_config: transformer::Config,
        second: gpt::Model,
        tokenizer: tokenizers::BPE,
        encodec: encodec::Model,
        default_speaker: Option<Tensor>,
        device: Device,
        codec_device: Device,
    ) -> Result<Self> {
        if first_config.speaker_emb_dim != 256
            || first_config.vocab_size != 2562
            || tokenizer.offset != 2049
            || tokenizer.end_of_text != 512
            || tokenizer.ranks.values().any(|rank| *rank >= 512)
            || second.config().vocab_sizes != [1538, 1025]
            || second.config().target_vocab_sizes != [1025; 6]
            || first_config.block_size < 2
            || second.config().block_size < 2
        {
            return Err(Error::Unsupported("MetaVoice adapter requires the 1B v0.1 text/audio vocabularies and 256-dimensional speaker embeddings".into()));
        }
        let default_speaker = default_speaker
            .map(|speaker| {
                if speaker.elem_count() != 256 {
                    return Err(Error::InvalidInput(
                        "spk_emb tensor must contain 256 values".into(),
                    ));
                }
                let speaker = speaker
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                if speaker.iter().any(|value| !value.is_finite()) {
                    return Err(Error::InvalidInput(
                        "spk_emb tensor contains nonfinite values".into(),
                    ));
                }
                Ok(Tensor::new(speaker.as_slice(), &device)?.reshape((1, 1, 256))?)
            })
            .transpose()?;
        Ok(Self {
            first,
            first_config,
            second,
            tokenizer,
            encodec,
            default_speaker,
            speaker_encoder: None,
            device,
            codec_device,
            format: AudioFormat::mono(24000)?,
            guidance_scale: 3.,
            temperature: 1.,
            finish: FinishReason::Completed,
            chunk_frames: 4096,
        })
    }

    pub fn with_speaker_encoder(mut self, encoder: speaker_encoder::Model) -> Result<Self> {
        let config = encoder.config();
        if config.model_embedding_size != 256
            || config.mel_n_channels != 40
            || config.sampling_rate != 16000
        {
            return Err(Error::Unsupported(
                "MetaVoice speaker encoder requires 256 embeddings and 40-channel 16kHz mels"
                    .into(),
            ));
        }
        let filters = speaker_encoder::mel_filters(config)?;
        self.speaker_encoder = Some((encoder, filters));
        Ok(self)
    }

    /// Extract and average normalized embeddings from clips of one speaker.
    /// PCM is converted to mono16kHz; transcripts are not needed by this encoder.
    pub fn reference_embedding(
        &self,
        references: &[AudioReference],
        cancel: &CancellationToken,
    ) -> Result<Vec<f32>> {
        cancel.check()?;
        validate_references(references)?;
        if references.is_empty() {
            return Err(Error::InvalidInput(
                "MetaVoice needs at least one speaker reference".into(),
            ));
        }
        let (encoder, filters) = self.speaker_encoder.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "load speaker_encoder.safetensors to enable native MetaVoice audio references"
                    .into(),
            )
        })?;
        let mut embeddings = Vec::with_capacity(references.len());
        for reference in references {
            cancel.check()?;
            let audio = AudioConverter::convert(&reference.audio, 16000)?;
            if audio
                .samples()
                .iter()
                .map(|sample| (*sample as f64).powi(2))
                .sum::<f64>()
                / (audio.frames() as f64)
                < 1e-12
            {
                return Err(Error::InvalidInput(
                    "MetaVoice speaker reference is silent".into(),
                ));
            }
            let result = encoder.embed_utterance_with_callback(
                audio.samples(),
                filters,
                1.3,
                0.75,
                encoder.device(),
                || {
                    cancel
                        .check()
                        .map_err(|error| candle::Error::Msg(error.to_string()))
                },
            );
            cancel.check()?;
            embeddings.push(result?);
        }
        let embedding = Tensor::stack(&embeddings, 0)?.mean(0)?;
        let norm = embedding.sqr()?.sum_all()?.sqrt()?;
        let norm_value = norm.to_scalar::<f32>()?;
        if !norm_value.is_finite() || norm_value <= 1e-12 {
            return Err(Error::InvalidInput(
                "MetaVoice speaker references produced an invalid embedding".into(),
            ));
        }
        Ok(embedding.broadcast_div(&norm)?.to_vec1::<f32>()?)
    }

    /// Local files follow the Candle MetaVoice example: first_stage.meta.json,
    /// first_stage.safetensors (or first_stage_q4k.gguf), second_stage.safetensors,
    /// spk_emb.safetensors (optional with request conditioning), and
    /// encodec/{config.json,model.safetensors}. Optional speaker_encoder.safetensors
    /// enables native audio references (`speaker_encoder_weights` overrides its path).
    /// All artifact paths can be overridden in options.
    pub fn load(directory: &Path, device: &Device, options: &serde_json::Value) -> Result<Self> {
        let quantized = match options.get("quantized") {
            None => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| Error::InvalidInput("quantized must be a boolean".into()))?,
        };
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(path_option(
            directory,
            options,
            "first_stage_meta",
            "first_stage.meta.json",
        )?)?)?;
        validate_metadata(&meta)?;
        let config_file = path_option(
            directory,
            options,
            "first_stage_config",
            "first_stage.model_args.json",
        )?;
        if config_file.exists() {
            let args = serde_json::from_slice(&std::fs::read(&config_file)?)?;
            validate_model_args(&args)?;
        } else if options.get("first_stage_config").is_some() {
            return Err(Error::InvalidInput(format!(
                "MetaVoice first-stage config does not exist: {}",
                config_file.display()
            )));
        }
        let tokenizer = tokenizers::BPE::from_json(
            meta.get("tokenizer").ok_or_else(|| {
                Error::InvalidInput("first_stage.meta.json has no tokenizer".into())
            })?,
            512,
        )?;
        if tokenizer.offset != 2049 || tokenizer.ranks.values().any(|rank| *rank >= 512) {
            return Err(Error::Unsupported(
                "MetaVoice v0.1 tokenizer needs offset2049 and512 byte-pair ranks".into(),
            ));
        }
        let number = |name: &str, default: f64, positive: bool| -> Result<f64> {
            match options.get(name) {
                None => Ok(default),
                Some(value) => value
                    .as_f64()
                    .filter(|n| n.is_finite() && (!positive || *n > 0.))
                    .ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "{name} must be finite{}",
                            if positive { " and positive" } else { "" }
                        ))
                    }),
            }
        };
        let guidance_scale = number("guidance_scale", 3., false)?;
        let temperature = number("temperature", 1., true)?;
        let first_config = transformer::Config::cfg1b_v0_1();
        let first_file = path_option(
            directory,
            options,
            "first_stage_weights",
            if quantized {
                "first_stage_q4k.gguf"
            } else {
                "first_stage.safetensors"
            },
        )?;
        let first = if quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                first_file, device,
            )?;
            FirstStage::Quantized(quantized::Model::new(&first_config, vb)?)
        } else {
            // SAFETY: files must remain unchanged while their read-only mappings are live.
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[first_file], DType::F32, device)? };
            FirstStage::Normal(transformer::Model::new(&first_config, vb)?)
        };
        let second_file = path_option(
            directory,
            options,
            "second_stage_weights",
            "second_stage.safetensors",
        )?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[second_file], DType::F32, device)? };
        let second = gpt::Model::new(gpt::Config::cfg1b_v0_1(), vb)?;
        let codec_device = match options.get("codec_cpu") {
            None | Some(serde_json::Value::Bool(false)) => device.clone(),
            Some(serde_json::Value::Bool(true)) => Device::Cpu,
            _ => return Err(Error::InvalidInput("codec_cpu must be a boolean".into())),
        };
        let codec_directory = path_option(directory, options, "encodec_directory", "encodec")?;
        let config: encodec::Config =
            serde_json::from_slice(&std::fs::read(codec_directory.join("config.json"))?)?;
        config.validate()?;
        if config.sampling_rate != 24000
            || config.audio_channels != 1
            || config.codebook_size != 1024
            || config.num_quantizers() < 8
        {
            return Err(Error::Unsupported(
                "MetaVoice requires mono EnCodec 24kHz with at least 8 codebooks of size 1024"
                    .into(),
            ));
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &super::load::weights(&codec_directory)?,
                DType::F32,
                &codec_device,
            )?
        };
        let encodec = encodec::Model::new(&config, vb)?;
        let speaker_file = path_option(
            directory,
            options,
            "speaker_embedding_file",
            "spk_emb.safetensors",
        )?;
        let speaker = if speaker_file.exists() {
            let tensors = candle::safetensors::load(&speaker_file, &Device::Cpu)?;
            Some(
                tensors
                    .get("spk_emb")
                    .ok_or_else(|| {
                        Error::InvalidInput("speaker checkpoint has no spk_emb tensor".into())
                    })?
                    .clone(),
            )
        } else if options.get("speaker_embedding_file").is_some() {
            return Err(Error::InvalidInput(format!(
                "speaker checkpoint does not exist: {}",
                speaker_file.display()
            )));
        } else {
            None
        };
        let mut model = Self::new(
            first,
            first_config,
            second,
            tokenizer,
            encodec,
            speaker,
            device.clone(),
            codec_device,
        )?;
        model.guidance_scale = guidance_scale;
        model.temperature = temperature;
        let encoder_file = path_option(
            directory,
            options,
            "speaker_encoder_weights",
            "speaker_encoder.safetensors",
        )?;
        if encoder_file.exists() {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[encoder_file], DType::F32, device)?
            };
            model = model.with_speaker_encoder(speaker_encoder::Model::new(
                speaker_encoder::Config::cfg(),
                vb,
            )?)?;
        } else if options.get("speaker_encoder_weights").is_some() {
            return Err(Error::InvalidInput(format!(
                "speaker encoder checkpoint does not exist: {}",
                encoder_file.display()
            )));
        }
        Ok(model)
    }

    fn infer(
        &mut self,
        request: &ConditionedSynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        cancel.check()?;
        validate_request(request)?;
        let prompt = self.tokenizer.encode(&request.request.text)?;
        let max_positions = prompt.len().checked_add(request.request.max_steps);
        if max_positions.is_none_or(|n| n > self.first_config.block_size)
            || prompt.len() + 1 >= self.second.config().block_size
        {
            return Err(Error::LimitExceeded(
                "MetaVoice text and generation budget exceed first/second-stage context".into(),
            ));
        }
        let speaker = match &request.speaker_embedding {
            Some(values) => Tensor::new(values.as_slice(), &self.device)?.reshape((1, 1, 256))?,
            None if !request.references.is_empty() => Tensor::new(
                self.reference_embedding(&request.references, cancel)?,
                &self.device,
            )?
            .reshape((1, 1, 256))?,
            None => self.default_speaker.clone().ok_or_else(|| {
                Error::InvalidInput(
                    "MetaVoice requires a speaker embedding or checkpoint spk_emb tensor".into(),
                )
            })?,
        };
        // Validate tokenizer mapping before altering the transformer cache.
        hierarchies(&prompt, &[], &[], self.second.config().block_size)?;
        self.first.clear();
        let mut tokens = prompt.clone();
        let mut sampler =
            LogitsProcessor::new(request.request.seed, Some(self.temperature), Some(0.95));
        let mut n1 = 0usize;
        let mut n2 = 0usize;
        self.finish = FinishReason::Length;
        for index in 0..request.request.max_steps {
            cancel.check()?;
            let size = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - size;
            let input = Tensor::new(&tokens[start..], &self.device)?;
            let input = Tensor::stack(&[&input, &input], 0)?;
            let logits = self.first.forward(&input, &speaker, start, cancel)?;
            cancel.check()?;
            let logits = ((logits.i((0, 0))? * self.guidance_scale)?
                + (logits.i((1, 0))? * (1. - self.guidance_scale))?)?
                .to_dtype(DType::F32)?;
            let next = sampler.sample(&logits)?;
            if next == EOS {
                tokens.push(next);
                self.finish = FinishReason::EndOfSequence;
                break;
            }
            if next < AUDIO_VOCAB {
                n1 += 1;
            } else if next < 2 * AUDIO_VOCAB {
                n2 += 1;
            }
            // Stage two has a smaller context. Stop with explicit Length metadata
            // before appending a code that would require truncating its input.
            if prompt.len() + n1.max(n2) + 1 > self.second.config().block_size {
                break;
            }
            tokens.push(next);
        }
        cancel.check()?;
        let (_, ids1, ids2) =
            adapters::FlattenedInterleavedEncodec2Codebook::new(AUDIO_VOCAB).decode(&tokens);
        if ids1.is_empty() || ids2.is_empty() {
            if self.finish == FinishReason::EndOfSequence {
                return Ok(());
            }
            return Err(Error::LimitExceeded(
                "MetaVoice token budget ended before a complete pair of audio codebooks".into(),
            ));
        }
        let (h1, h2) = hierarchies(&prompt, &ids1, &ids2, self.second.config().block_size)?;
        let input = Tensor::stack(
            &[
                Tensor::new(h1, &self.device)?,
                Tensor::new(h2, &self.device)?,
            ],
            0,
        )?
        .unsqueeze(0)?;
        cancel.check()?;
        let logits = self.second.forward_with_callback(&input, |_| {
            cancel
                .check()
                .map_err(|error| candle::Error::Msg(error.to_string()))
        });
        cancel.check()?;
        let logits = logits?;
        let mut sampler =
            LogitsProcessor::new(request.request.seed.wrapping_add(1337), Some(1.), None);
        let mut codes = Vec::with_capacity(logits.len());
        for hierarchy in logits {
            let logits = hierarchy.squeeze(0)?;
            let mut samples = Vec::with_capacity(logits.dim(0)?);
            for position in 0..logits.dim(0)? {
                cancel.check()?;
                samples.push(sampler.sample(&logits.i(position)?.to_dtype(DType::F32)?)?);
            }
            codes.push(samples);
        }
        let codes = Tensor::new(codes, &self.device)?.unsqueeze(0)?;
        let codes = Tensor::cat(&[input, codes], 1)?.i(0)?.to_vec2::<u32>()?;
        let (_, audio) = adapters::TiltedEncodec::new(AUDIO_VOCAB).decode(&codes);
        if audio.len() != 8 || audio.iter().any(Vec::is_empty) {
            return Err(Error::InvalidState(
                "MetaVoice refinement produced no complete 8-codebook audio frames".into(),
            ));
        }
        let audio = Tensor::new(audio, &self.codec_device)?.unsqueeze(0)?;
        cancel.check()?;
        let pcm = self
            .encodec
            .decode(&audio)?
            .i((0, 0))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        cancel.check()?;
        for chunk in pcm.chunks(self.chunk_frames) {
            cancel.check()?;
            sink.emit(AudioEvent::Audio {
                turn: 0,
                audio: AudioBuffer::new(self.format, chunk.to_vec())?,
            })?;
        }
        Ok(())
    }
}

impl TtsModel for MetaVoice {
    fn output_format(&self) -> AudioFormat {
        self.format
    }
    fn streaming_mode(&self) -> StreamingMode {
        StreamingMode::Buffered
    }
    fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        self.synthesize_conditioned(
            &ConditionedSynthesisRequest::new(request.clone()),
            cancel,
            sink,
        )
    }
    fn synthesize_conditioned(
        &mut self,
        request: &ConditionedSynthesisRequest,
        cancel: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<()> {
        let result = self.infer(request, cancel, sink);
        self.first.clear();
        if let Err(error) = &result {
            self.finish = match error {
                Error::Cancelled => FinishReason::Cancelled,
                Error::DeadlineExceeded => FinishReason::Deadline,
                _ => self.finish,
            };
        }
        result
    }
    fn reset(&mut self) -> Result<()> {
        self.first.clear();
        self.finish = FinishReason::Completed;
        Ok(())
    }
    fn finish_reason(&self) -> FinishReason {
        self.finish
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: if matches!(&self.first, FirstStage::Quantized(_)) {
                "quantized_metavoice"
            } else {
                "metavoice"
            }
            .into(),
            tasks: vec![AudioTask::Synthesis],
            languages: vec!["en".into()],
            output_formats: vec![self.format],
            max_text_bytes: Some(MAX_TEXT_BYTES),
            accepts_audio_reference: self.speaker_encoder.is_some(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_checkpoint_architecture_before_mapping_weights() -> Result<()> {
        let mut meta = serde_json::json!({"speaker_cond":true,"speaker_emb_size":256});
        validate_metadata(&meta)?;
        meta["speaker_cond"] = serde_json::json!(false);
        assert!(matches!(
            validate_metadata(&meta),
            Err(Error::Unsupported(_))
        ));
        // Pinned first_stage.model_args.json from lmz/candle-metavoice v0.1.
        let mut args = serde_json::json!({"n_layer":24,"n_head":16,"n_embd":2048,"block_size":2048,"bias":false,"causal":true,"spk_emb_on_text":true,"vocab_sizes":[2562],"norm_type":"rmsnorm","nonlinearity_type":"swiglu"});
        validate_model_args(&args)?;
        args["block_size"] = serde_json::json!(4096);
        assert!(matches!(
            validate_model_args(&args),
            Err(Error::Unsupported(_))
        ));
        Ok(())
    }
    #[test]
    fn refinement_does_not_silently_truncate() -> Result<()> {
        let (h1, h2) = hierarchies(&[2049, 2561], &[2, 3], &[4, 5], 5)?;
        assert_eq!(h1, [1025, 1537, 2, 3, 1024]);
        assert_eq!(h2, [1024, 1024, 4, 5, 1024]);
        assert!(matches!(
            hierarchies(&[2049, 2561], &[2, 3], &[4, 5], 4),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            hierarchies(&[2048], &[], &[], 5),
            Err(Error::InvalidInput(_))
        ));
        Ok(())
    }
    #[test]
    fn validates_embedding_and_reference_conditioning() -> Result<()> {
        let mut request = ConditionedSynthesisRequest::new(SynthesisRequest::new("hello"));
        request.speaker_embedding = Some(vec![0.; 255]);
        assert!(matches!(
            validate_request(&request),
            Err(Error::InvalidInput(_))
        ));
        request.speaker_embedding = Some(vec![0.; 256]);
        validate_request(&request)?;
        request.references.push(AudioReference {
            audio: AudioBuffer::new(AudioFormat::mono(24000)?, vec![0.; 32])?,
            transcript: None,
            speaker_id: None,
        });
        assert!(matches!(
            validate_request(&request),
            Err(Error::InvalidInput(_))
        ));
        request.speaker_embedding = None;
        validate_request(&request)?;
        request
            .references
            .extend(vec![request.references[0].clone(); 8]);
        assert!(matches!(
            validate_request(&request),
            Err(Error::LimitExceeded(_))
        ));
        Ok(())
    }

    #[test]
    fn native_transformer_layer_cancellation_and_cache_recovery() -> Result<()> {
        let device = Device::Cpu;
        let mut config = transformer::Config::cfg1b_v0_1();
        config.n_layer = 2;
        config.n_head = 2;
        config.n_local_heads = Some(2);
        config.dim = 8;
        config.intermediate_size = Some(16);
        config.block_size = 16;
        let mut model = transformer::Model::new(&config, VarBuilder::zeros(DType::F32, &device))?;
        let tokens = Tensor::new(&[[2049u32, 2050], [2049, 2050]], &device)?;
        let speaker = Tensor::zeros((1, 1, 256), DType::F32, &device)?;
        let mut stopped = false;
        let result = model.forward_with_callback(&tokens, &speaker, 0, |layer| {
            if layer == 1 {
                stopped = true;
                return Err(candle::Error::Msg("cancelled layer fixture".into()));
            }
            Ok(())
        });
        assert!(stopped && result.is_err());
        model.clear_kv_cache();
        let logits = model.forward(&tokens, &speaker, 0)?;
        assert_eq!(logits.dims(), [2, 1, 2562]);
        Ok(())
    }

    #[test]
    fn native_refinement_layer_cancellation() -> Result<()> {
        let device = Device::Cpu;
        let mut config = gpt::Config::cfg1b_v0_1();
        config.n_layer = 2;
        config.n_head = 2;
        config.n_embd = 12;
        config.block_size = 8;
        config.swiglu_multiple_of = Some(4);
        let model = gpt::Model::new(config, VarBuilder::zeros(DType::F32, &device))?;
        let input = Tensor::new(
            &[[[1025u32, 1537, 0, 1024], [1024, 1024, 0, 1024]]],
            &device,
        )?;
        let mut stopped = false;
        assert!(model
            .forward_with_callback(&input, |layer| {
                if layer == 1 {
                    stopped = true;
                    return Err(candle::Error::Msg("cancelled refinement fixture".into()));
                }
                Ok(())
            })
            .is_err());
        assert!(stopped);
        let output = model.forward(&input)?;
        assert_eq!(output.len(), 6);
        assert!(output.iter().all(|tensor| tensor.dims() == [1, 4, 1025]));
        Ok(())
    }
}
