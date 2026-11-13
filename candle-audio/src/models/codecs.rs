//! Native codec adapters. PCM is explicitly interleaved; codes are frame-major.
use crate::*;
use candle::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::{dac, encodec, mimi, snac};
use std::path::Path;

fn input(
    audio: &AudioBuffer,
    format: AudioFormat,
    max: usize,
    cancel: &CancellationToken,
) -> Result<()> {
    cancel.check()?;
    if audio.format() != format {
        return Err(Error::InvalidInput(
            "codec PCM format mismatch; convert explicitly before encoding".into(),
        ));
    }
    if audio.frames() > max {
        return Err(Error::LimitExceeded(
            "codec input exceeds configured frame budget".into(),
        ));
    }
    Ok(())
}
fn tensor(audio: &AudioBuffer, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        audio.samples().to_vec(),
        (1, audio.frames(), audio.format().channels() as usize),
        device,
    )?
    .transpose(1, 2)?
    .contiguous()?)
}
fn pcm(
    values: Tensor,
    format: AudioFormat,
    length: Option<usize>,
    cancel: &CancellationToken,
) -> Result<AudioBuffer> {
    cancel.check()?;
    let (batch, channels, frames) = values.dims3()?;
    if batch != 1 || channels != format.channels() as usize {
        return Err(Error::InvalidInput(
            "codec decoded an incompatible PCM shape".into(),
        ));
    }
    let length = length.unwrap_or(frames);
    if length > frames {
        return Err(Error::InvalidInput(
            "codec decoded fewer samples than the original length".into(),
        ));
    }
    let values = values
        .narrow(2, 0, length)?
        .i(0)?
        .transpose(0, 1)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    AudioBuffer::new(format, values)
}
fn codes_from_tensor(
    values: Tensor,
    format: CodecFormat,
    original: usize,
    cancel: &CancellationToken,
) -> Result<AudioCodes> {
    cancel.check()?;
    let (batch, books, _) = values.dims3()?;
    if batch != 1 || books != format.num_codebooks {
        return Err(Error::InvalidInput(
            "codec encoded an incompatible code shape".into(),
        ));
    }
    let values = values
        .i(0)?
        .transpose(0, 1)?
        .flatten_all()?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    AudioCodes::new(format, values)?.with_original_audio_frames(original)
}
fn codes_tensor(codes: &AudioCodes, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        codes.codes().to_vec(),
        (codes.frames(), codes.format().num_codebooks),
        device,
    )?
    .t()?
    .unsqueeze(0)?
    .contiguous()?)
}
fn check_codes(
    codes: &AudioCodes,
    format: &CodecFormat,
    max: usize,
    cancel: &CancellationToken,
) -> Result<()> {
    cancel.check()?;
    if codes.format() != format {
        return Err(Error::InvalidInput("codec code format mismatch".into()));
    }
    let estimated = (codes.frames() as f64 / format.frame_rate
        * format.audio_format.sample_rate() as f64)
        .ceil();
    if estimated > max as f64 + format.audio_format.sample_rate() as f64 {
        return Err(Error::LimitExceeded(
            "codec codes exceed decode frame budget".into(),
        ));
    }
    Ok(())
}
fn max_frames(format: AudioFormat) -> usize {
    format.sample_rate() as usize * 30
}
fn empty(format: CodecFormat) -> Result<AudioCodes> {
    AudioCodes::new(format, Vec::new())?.with_original_audio_frames(0)
}

pub struct EncodecCodec {
    model: encodec::Model,
    config: encodec::Config,
    device: Device,
    format: CodecFormat,
    max_frames: usize,
}
impl EncodecCodec {
    pub fn new(model: encodec::Model, config: encodec::Config, device: Device) -> Result<Self> {
        config.validate()?;
        let audio_format =
            AudioFormat::new(config.sampling_rate as u32, config.audio_channels as u16)?;
        let hop = config.upsampling_ratios.iter().product::<usize>();
        let format = CodecFormat {
            audio_format,
            frame_rate: config.sampling_rate as f64 / hop as f64,
            num_codebooks: config.num_quantizers(),
            cardinality: config.codebook_size as u32,
        };
        format.validate()?;
        Ok(Self {
            model,
            config,
            device,
            format,
            max_frames: max_frames(audio_format),
        })
    }
    pub fn with_max_input_frames(mut self, frames: usize) -> Result<Self> {
        if frames == 0 {
            return Err(Error::InvalidInput("codec budget must be positive".into()));
        }
        self.max_frames = frames;
        Ok(self)
    }
    pub fn with_num_codebooks(mut self, count: usize) -> Result<Self> {
        if count == 0 || count > self.config.num_quantizers() {
            return Err(Error::InvalidInput(
                "EnCodec codebook count out of range".into(),
            ));
        }
        self.format.num_codebooks = count;
        Ok(self)
    }
}
impl AudioCodec for EncodecCodec {
    fn format(&self) -> CodecFormat {
        self.format.clone()
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "encodec".into(),
            tasks: vec![AudioTask::Codec],
            input_streaming: Some(StreamingMode::Buffered),
            output_streaming: Some(StreamingMode::Buffered),
            input_formats: vec![self.format.audio_format],
            output_formats: vec![self.format.audio_format],
            max_input_frames: Some(self.max_frames),
            ..Default::default()
        }
    }
    fn encode(&mut self, audio: &AudioBuffer, cancel: &CancellationToken) -> Result<AudioCodes> {
        input(audio, self.format.audio_format, self.max_frames, cancel)?;
        if audio.is_empty() {
            return empty(self.format());
        }
        let channels = self.format.audio_format.channels() as usize;
        let chunk = self
            .config
            .chunk_length_s
            .map(|seconds| (seconds * self.config.sampling_rate as f64) as usize)
            .unwrap_or(audio.frames())
            .max(1);
        let stride = ((1. - self.config.overlap.unwrap_or(0.)) * chunk as f64).floor() as usize;
        if stride == 0 {
            return Err(Error::InvalidInput(
                "EnCodec chunk stride must be positive".into(),
            ));
        }
        let mut all_codes = Vec::new();
        let mut segments = Vec::new();
        let mut code_offset = 0;
        for offset in (0..audio.frames()).step_by(stride) {
            cancel.check()?;
            let length = chunk.min(audio.frames() - offset);
            let samples = &audio.samples()[offset * channels..(offset + length) * channels];
            let scale = if self.config.normalize {
                let energy = samples
                    .chunks_exact(channels)
                    .map(|frame| {
                        let mono = frame.iter().map(|v| *v as f64).sum::<f64>() / channels as f64;
                        mono * mono
                    })
                    .sum::<f64>();
                (energy / length as f64).sqrt() as f32 + 1e-8
            } else {
                1.
            };
            let scaled = AudioBuffer::new(
                self.format.audio_format,
                samples.iter().map(|value| *value / scale).collect(),
            )?;
            let values = self.model.encode(&tensor(&scaled, &self.device)?)?.narrow(
                1,
                0,
                self.format.num_codebooks,
            )?;
            let codes = codes_from_tensor(values, self.format(), length, cancel)?;
            segments.push(CodecSegment {
                code_offset,
                code_frames: codes.frames(),
                audio_offset: offset,
                audio_frames: length,
                scale,
            });
            code_offset += codes.frames();
            all_codes.extend_from_slice(codes.codes());
            if self.config.chunk_length_s.is_none() {
                break;
            }
        }
        AudioCodes::new(self.format(), all_codes)?
            .with_original_audio_frames(audio.frames())?
            .with_segments(segments)
    }
    fn decode(&mut self, codes: &AudioCodes, cancel: &CancellationToken) -> Result<AudioBuffer> {
        check_codes(codes, &self.format, self.max_frames, cancel)?;
        if codes.frames() == 0 {
            return AudioBuffer::new(self.format.audio_format, Vec::new());
        }
        if codes.segments().is_empty() {
            return pcm(
                self.model.decode(&codes_tensor(codes, &self.device)?)?,
                self.format.audio_format,
                codes.original_audio_frames(),
                cancel,
            );
        }
        let channels = self.format.audio_format.channels() as usize;
        let length = codes.original_audio_frames().unwrap_or_else(|| {
            codes
                .segments()
                .iter()
                .map(|segment| segment.audio_offset + segment.audio_frames)
                .max()
                .unwrap_or(0)
        });
        if length > self.max_frames {
            return Err(Error::LimitExceeded(
                "EnCodec decoded audio exceeds frame budget".into(),
            ));
        }
        let mut output = vec![0f64; length * channels];
        let mut weights = vec![0f64; length];
        let window_length = codes.segments()[0].audio_frames;
        for segment in codes.segments() {
            cancel.check()?;
            let start = segment.code_offset * self.format.num_codebooks;
            let end = start + segment.code_frames * self.format.num_codebooks;
            let chunk_codes = AudioCodes::new(self.format(), codes.codes()[start..end].to_vec())?;
            let decoded = pcm(
                self.model
                    .decode(&codes_tensor(&chunk_codes, &self.device)?)?,
                self.format.audio_format,
                Some(segment.audio_frames),
                cancel,
            )?;
            for (frame, samples) in decoded.samples().chunks_exact(channels).enumerate() {
                let index = segment.audio_offset + frame;
                if index >= length {
                    break;
                }
                let time = (frame + 1) as f64 / (window_length + 1) as f64;
                let weight = 0.5 - (time - 0.5).abs();
                weights[index] += weight;
                for (channel, value) in samples.iter().enumerate() {
                    output[index * channels + channel] +=
                        *value as f64 * segment.scale as f64 * weight;
                }
            }
        }
        if weights.iter().any(|value| *value <= 0.) {
            return Err(Error::InvalidInput(
                "EnCodec chunk metadata leaves a gap in decoded audio".into(),
            ));
        }
        AudioBuffer::new(
            self.format.audio_format,
            output
                .into_iter()
                .enumerate()
                .map(|(index, value)| (value / weights[index / channels]) as f32)
                .collect(),
        )
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct MimiCodec {
    model: mimi::Model,
    device: Device,
    format: CodecFormat,
    max_frames: usize,
    frame_size: usize,
}
impl MimiCodec {
    pub fn new(mut model: mimi::Model, device: Device) -> Result<Self> {
        let cfg = model.config();
        let audio_format = AudioFormat::new(cfg.sample_rate as u32, cfg.channels as u16)?;
        let ratio = cfg.sample_rate / cfg.frame_rate;
        if !ratio.is_finite() || ratio < 1. || ratio.fract() != 0. {
            return Err(Error::InvalidInput(
                "Mimi codec requires an integral frame length".into(),
            ));
        }
        let hop = cfg.seanet.ratios.iter().product::<usize>();
        let max = cfg
            .transformer
            .max_seq_len
            .saturating_mul(hop)
            .saturating_sub(ratio as usize * 2);
        let format = CodecFormat {
            audio_format,
            frame_rate: cfg.frame_rate,
            num_codebooks: cfg.quantizer_n_q,
            cardinality: cfg.quantizer_bins as u32,
        };
        format.validate()?;
        model.reset_state();
        Ok(Self {
            model,
            device,
            format,
            max_frames: max.min(max_frames(audio_format)),
            frame_size: ratio as usize,
        })
    }
}
impl AudioCodec for MimiCodec {
    fn format(&self) -> CodecFormat {
        self.format.clone()
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "mimi".into(),
            tasks: vec![AudioTask::Codec],
            input_streaming: Some(StreamingMode::Native),
            output_streaming: Some(StreamingMode::Native),
            input_formats: vec![self.format.audio_format],
            output_formats: vec![self.format.audio_format],
            max_input_frames: Some(self.max_frames),
            ..Default::default()
        }
    }
    fn encode(&mut self, audio: &AudioBuffer, cancel: &CancellationToken) -> Result<AudioCodes> {
        input(audio, self.format.audio_format, self.max_frames, cancel)?;
        if audio.is_empty() {
            return empty(self.format());
        }
        self.model.reset_state();
        let values = tensor(audio, &self.device)?;
        let padding = audio.frames().div_ceil(self.frame_size) * self.frame_size - audio.frames();
        let result = self
            .model
            .encode(&values.pad_with_zeros(D::Minus1, 0, padding)?);
        self.model.reset_state();
        codes_from_tensor(result?, self.format(), audio.frames(), cancel)
    }
    fn decode(&mut self, codes: &AudioCodes, cancel: &CancellationToken) -> Result<AudioBuffer> {
        check_codes(codes, &self.format, self.max_frames, cancel)?;
        if codes.frames() == 0 {
            return AudioBuffer::new(self.format.audio_format, Vec::new());
        }
        self.model.reset_state();
        let result = self.model.decode(&codes_tensor(codes, &self.device)?);
        self.model.reset_state();
        pcm(
            result?,
            self.format.audio_format,
            codes.original_audio_frames(),
            cancel,
        )
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset_state();
        Ok(())
    }
}

pub struct DacCodec {
    model: dac::Model,
    device: Device,
    format: CodecFormat,
    max_frames: usize,
}
impl DacCodec {
    pub fn new(model: dac::Model, config: &dac::Config, device: Device) -> Result<Self> {
        let audio_format = AudioFormat::mono(config.sampling_rate)?;
        let format = CodecFormat {
            audio_format,
            frame_rate: config.sampling_rate as f64 / model.hop_length() as f64,
            num_codebooks: config.num_codebooks,
            cardinality: config.codebook_size as u32,
        };
        format.validate()?;
        Ok(Self {
            model,
            device,
            format,
            max_frames: max_frames(audio_format),
        })
    }
}
impl AudioCodec for DacCodec {
    fn format(&self) -> CodecFormat {
        self.format.clone()
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "dac".into(),
            tasks: vec![AudioTask::Codec],
            input_streaming: Some(StreamingMode::Buffered),
            output_streaming: Some(StreamingMode::Buffered),
            input_formats: vec![self.format.audio_format],
            output_formats: vec![self.format.audio_format],
            max_input_frames: Some(self.max_frames),
            ..Default::default()
        }
    }
    fn encode(&mut self, audio: &AudioBuffer, cancel: &CancellationToken) -> Result<AudioCodes> {
        input(audio, self.format.audio_format, self.max_frames, cancel)?;
        if audio.is_empty() {
            return empty(self.format());
        }
        codes_from_tensor(
            self.model.encode(&tensor(audio, &self.device)?)?,
            self.format(),
            audio.frames(),
            cancel,
        )
    }
    fn decode(&mut self, codes: &AudioCodes, cancel: &CancellationToken) -> Result<AudioBuffer> {
        check_codes(codes, &self.format, self.max_frames, cancel)?;
        if codes.frames() == 0 {
            return AudioBuffer::new(self.format.audio_format, Vec::new());
        }
        pcm(
            self.model
                .decode_codes(&codes_tensor(codes, &self.device)?)?,
            self.format.audio_format,
            codes.original_audio_frames(),
            cancel,
        )
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct SnacCodec {
    model: snac::Model,
    device: Device,
    format: CodecFormat,
    strides: Vec<usize>,
    max_frames: usize,
}
impl SnacCodec {
    pub fn new(model: snac::Model, device: Device) -> Result<Self> {
        let cfg = model.config();
        cfg.validate()?;
        let audio_format = AudioFormat::mono(cfg.sampling_rate as u32)?;
        let format = CodecFormat {
            audio_format,
            frame_rate: cfg.sampling_rate as f64 / model.hop_length as f64,
            num_codebooks: cfg.vq_strides.len(),
            cardinality: cfg.codebook_size as u32,
        };
        format.validate()?;
        let strides = cfg.vq_strides.clone();
        Ok(Self {
            model,
            device,
            format,
            strides,
            max_frames: max_frames(audio_format),
        })
    }
}
impl AudioCodec for SnacCodec {
    fn format(&self) -> CodecFormat {
        self.format.clone()
    }
    fn descriptor(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_id: "snac".into(),
            tasks: vec![AudioTask::Codec],
            input_streaming: Some(StreamingMode::Buffered),
            output_streaming: Some(StreamingMode::Buffered),
            input_formats: vec![self.format.audio_format],
            output_formats: vec![self.format.audio_format],
            max_input_frames: Some(self.max_frames),
            ..Default::default()
        }
    }
    fn encode(&mut self, audio: &AudioBuffer, cancel: &CancellationToken) -> Result<AudioCodes> {
        let codes = self.encode_multiscale(audio, cancel)?;
        let frames = codes
            .codebooks()
            .iter()
            .zip(&self.strides)
            .map(|(book, stride)| book.len() * stride)
            .max()
            .unwrap_or(0);
        let mut expanded = Vec::with_capacity(frames * self.format.num_codebooks);
        for frame in 0..frames {
            for (book, stride) in codes.codebooks().iter().zip(&self.strides) {
                expanded.push(book[frame / stride]);
            }
        }
        AudioCodes::new(self.format(), expanded)?.with_original_audio_frames(audio.frames())
    }
    fn decode(&mut self, codes: &AudioCodes, cancel: &CancellationToken) -> Result<AudioBuffer> {
        check_codes(codes, &self.format, self.max_frames, cancel)?;
        let mut books = vec![Vec::new(); self.format.num_codebooks];
        for (book, stride) in self.strides.iter().enumerate() {
            if !codes.frames().is_multiple_of(*stride) {
                return Err(Error::InvalidInput(
                    "SNAC expanded codes do not align to codebook stride".into(),
                ));
            }
            for start in (0..codes.frames()).step_by(*stride) {
                let code = codes.codes()[start * self.format.num_codebooks + book];
                for offset in 1..*stride {
                    if codes.codes()[(start + offset) * self.format.num_codebooks + book] != code {
                        return Err(Error::InvalidInput(
                            "SNAC coarse-code repetitions are inconsistent".into(),
                        ));
                    }
                }
                books[book].push(code);
            }
        }
        let mut multiscale = MultiScaleAudioCodes::new(self.format(), self.strides.clone(), books)?;
        if let Some(length) = codes.original_audio_frames() {
            multiscale = multiscale.with_original_audio_frames(length)?;
        }
        self.decode_multiscale(&multiscale, cancel)
    }
    fn encode_multiscale(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
    ) -> Result<MultiScaleAudioCodes> {
        input(audio, self.format.audio_format, self.max_frames, cancel)?;
        let books = if audio.is_empty() {
            vec![Vec::new(); self.format.num_codebooks]
        } else {
            self.model
                .encode(&tensor(audio, &self.device)?)?
                .into_iter()
                .map(|values| values.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>())
                .collect::<candle::Result<Vec<_>>>()?
        };
        cancel.check()?;
        MultiScaleAudioCodes::new(self.format(), self.strides.clone(), books)?
            .with_original_audio_frames(audio.frames())
    }
    fn decode_multiscale(
        &mut self,
        codes: &MultiScaleAudioCodes,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer> {
        cancel.check()?;
        if codes.format() != &self.format || codes.codebook_strides() != self.strides {
            return Err(Error::InvalidInput(
                "SNAC multiscale format mismatch".into(),
            ));
        }
        let spans: Vec<_> = codes
            .codebooks()
            .iter()
            .zip(&self.strides)
            .map(|(book, stride)| book.len() * stride)
            .collect();
        if spans.iter().any(|span| *span != spans[0]) {
            return Err(Error::InvalidInput(
                "SNAC codebooks cover different durations".into(),
            ));
        }
        if spans[0] == 0 {
            return AudioBuffer::new(self.format.audio_format, Vec::new());
        }
        let samples = (spans[0] as f64 / self.format.frame_rate
            * self.format.audio_format.sample_rate() as f64)
            .ceil();
        if samples > self.max_frames as f64 + self.format.audio_format.sample_rate() as f64 {
            return Err(Error::LimitExceeded(
                "SNAC codes exceed decode budget".into(),
            ));
        }
        let values = codes
            .codebooks()
            .iter()
            .map(|book| Tensor::new(book.as_slice(), &self.device)?.unsqueeze(0))
            .collect::<candle::Result<Vec<_>>>()?;
        let refs = values.iter().collect::<Vec<_>>();
        pcm(
            self.model.decode(&refs)?,
            self.format.audio_format,
            codes.original_audio_frames(),
            cancel,
        )
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

fn vb<'a>(directory: &Path, device: &Device) -> Result<VarBuilder<'a>> {
    // Checkpoints must remain unchanged while their read-only mappings are used.
    Ok(unsafe {
        VarBuilder::from_mmaped_safetensors(&super::load::weights(directory)?, DType::F32, device)?
    })
}
pub fn load_encodec(
    directory: &Path,
    device: &Device,
    options: &serde_json::Value,
) -> Result<Box<dyn AudioCodec + Send>> {
    let config: encodec::Config =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    config.validate()?;
    let model = encodec::Model::new(&config, vb(directory, device)?)?;
    let mut adapter = EncodecCodec::new(model, config, device.clone())?;
    if let Some(count) = options
        .get("num_codebooks")
        .and_then(|value| value.as_u64())
    {
        adapter = adapter.with_num_codebooks(count as usize)?;
    }
    Ok(Box::new(adapter))
}
pub fn load_mimi(
    directory: &Path,
    device: &Device,
    options: &serde_json::Value,
) -> Result<Box<dyn AudioCodec + Send>> {
    let config = super::load::mimi_config(directory, options)?;
    let model = mimi::Model::new(config, vb(directory, device)?)?;
    Ok(Box::new(MimiCodec::new(model, device.clone())?))
}
pub fn load_snac(
    directory: &Path,
    device: &Device,
    _options: &serde_json::Value,
) -> Result<Box<dyn AudioCodec + Send>> {
    let config: snac::Config =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    config.validate()?;
    // Official SNAC releases publish a PyTorch state dict. Accept it natively
    // as well as the converted safetensors used by earlier Candle examples.
    let weights = if directory.join("model.safetensors").is_file()
        || directory.join("model.safetensors.index.json").is_file()
    {
        vb(directory, device)?
    } else {
        VarBuilder::from_pth(directory.join("pytorch_model.bin"), DType::F32, device)?
    };
    let model = snac::Model::new(&config, weights)?;
    Ok(Box::new(SnacCodec::new(model, device.clone())?))
}
pub fn load_dac(
    directory: &Path,
    device: &Device,
    _options: &serde_json::Value,
) -> Result<Box<dyn AudioCodec + Send>> {
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let config: dac::Config = serde_json::from_value(json.clone())?;
    let encoder_dim = json
        .get("encoder_hidden_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(64) as usize;
    let decoder_dim = json
        .get("decoder_hidden_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(1536) as usize;
    let codebook_dim = json
        .get("codebook_dim")
        .and_then(|value| value.as_u64())
        .unwrap_or(8) as usize;
    let rates = |name: &str, default: &[usize]| -> Result<Vec<usize>> {
        match json.get(name) {
            None => Ok(default.to_vec()),
            Some(value) => serde_json::from_value(value.clone()).map_err(Error::from),
        }
    };
    let encoder = rates("downsampling_ratios", &[2, 4, 8, 8])?;
    let decoder = rates("upsampling_ratios", &[8, 8, 4, 2])?;
    let model = dac::Model::with_architecture(
        &config,
        encoder_dim,
        &encoder,
        decoder_dim,
        &decoder,
        codebook_dim,
        vb(directory, device)?,
    )?;
    Ok(Box::new(DacCodec::new(model, &config, device.clone())?))
}
