use crate::*;

/// Codec configuration. Codes are frame-major, with `num_codebooks` codes per frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodecFormat {
    pub audio_format: AudioFormat,
    pub frame_rate: f64,
    pub num_codebooks: usize,
    pub cardinality: u32,
}
impl CodecFormat {
    pub fn validate(&self) -> Result<()> {
        if !self.frame_rate.is_finite()
            || self.frame_rate <= 0.
            || self.num_codebooks == 0
            || self.cardinality == 0
        {
            return Err(Error::InvalidInput(
                "invalid codec frame rate, codebook count or cardinality".into(),
            ));
        }
        Ok(())
    }
}

/// An independently encoded frame/chunk, with gain and overlap placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodecSegment {
    pub code_offset: usize,
    pub code_frames: usize,
    pub audio_offset: usize,
    pub audio_frames: usize,
    pub scale: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AudioCodes {
    format: CodecFormat,
    codes: Vec<u32>,
    original_audio_frames: Option<usize>,
    segments: Vec<CodecSegment>,
}
impl<'de> Deserialize<'de> for AudioCodes {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            format: CodecFormat,
            codes: Vec<u32>,
            #[serde(default)]
            original_audio_frames: Option<usize>,
            #[serde(default)]
            segments: Vec<CodecSegment>,
        }
        let value = Raw::deserialize(deserializer)?;
        Self::new(value.format, value.codes)
            .and_then(|mut codes| {
                codes.original_audio_frames = value.original_audio_frames;
                codes.validate_original_frames()?;
                codes = codes.with_segments(value.segments)?;
                Ok(codes)
            })
            .map_err(serde::de::Error::custom)
    }
}
impl AudioCodes {
    pub fn new(format: CodecFormat, codes: Vec<u32>) -> Result<Self> {
        format.validate()?;
        if !codes.len().is_multiple_of(format.num_codebooks)
            || codes.iter().any(|code| *code >= format.cardinality)
        {
            return Err(Error::InvalidInput(
                "codec codes contain an incomplete frame or out-of-vocabulary value".into(),
            ));
        }
        Ok(Self {
            format,
            codes,
            original_audio_frames: None,
            segments: Vec::new(),
        })
    }
    fn validate_original_frames(&self) -> Result<()> {
        if let Some(frames) = self.original_audio_frames {
            let capacity = (self.frames() as f64 / self.format.frame_rate
                * self.format.audio_format.sample_rate() as f64)
                .ceil();
            if frames as f64 > capacity {
                return Err(Error::InvalidInput(
                    "original audio exceeds encoded frame capacity".into(),
                ));
            }
        }
        Ok(())
    }
    pub fn with_original_audio_frames(mut self, frames: usize) -> Result<Self> {
        self.original_audio_frames = Some(frames);
        self.validate_original_frames()?;
        Ok(self)
    }
    pub fn original_audio_frames(&self) -> Option<usize> {
        self.original_audio_frames
    }
    pub fn segments(&self) -> &[CodecSegment] {
        &self.segments
    }
    pub fn with_segments(mut self, segments: Vec<CodecSegment>) -> Result<Self> {
        let mut offset = 0usize;
        for segment in &segments {
            if segment.code_offset != offset
                || segment.code_frames == 0
                || segment.audio_frames == 0
                || !segment.scale.is_finite()
                || segment.scale <= 0.
            {
                return Err(Error::InvalidInput("invalid codec chunk metadata".into()));
            }
            offset = offset
                .checked_add(segment.code_frames)
                .ok_or_else(|| Error::InvalidInput("codec chunk offset overflow".into()))?;
            segment
                .audio_offset
                .checked_add(segment.audio_frames)
                .ok_or_else(|| Error::InvalidInput("codec chunk audio offset overflow".into()))?;
            let capacity = (segment.code_frames as f64 / self.format.frame_rate
                * self.format.audio_format.sample_rate() as f64)
                .ceil();
            if segment.audio_frames as f64 > capacity {
                return Err(Error::InvalidInput(
                    "codec chunk audio exceeds code capacity".into(),
                ));
            }
        }
        if !segments.is_empty() && offset != self.frames() {
            return Err(Error::InvalidInput(
                "codec chunks do not cover code sequence".into(),
            ));
        }
        self.segments = segments;
        Ok(self)
    }
    pub fn format(&self) -> &CodecFormat {
        &self.format
    }
    pub fn frames(&self) -> usize {
        self.codes.len() / self.format.num_codebooks
    }
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }
    pub fn into_codes(self) -> Vec<u32> {
        self.codes
    }
}

/// Native codec encode/decode. Streaming codecs can override the step/flush methods.
/// Default step methods are intentionally unsupported rather than pretending that
/// independently encoding chunks is equivalent to causal codec streaming.
pub trait AudioCodec {
    fn format(&self) -> CodecFormat;
    fn descriptor(&self) -> ModelCapabilities {
        let format = self.format();
        ModelCapabilities {
            tasks: vec![AudioTask::Codec],
            input_formats: vec![format.audio_format],
            output_formats: vec![format.audio_format],
            ..Default::default()
        }
    }
    fn encode(&mut self, audio: &AudioBuffer, cancel: &CancellationToken) -> Result<AudioCodes>;
    fn decode(&mut self, codes: &AudioCodes, cancel: &CancellationToken) -> Result<AudioBuffer>;
    fn reset(&mut self) -> Result<()>;
    fn encode_multiscale(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
    ) -> Result<MultiScaleAudioCodes> {
        self.encode(audio, cancel)
            .map(MultiScaleAudioCodes::from_uniform)
    }
    fn decode_multiscale(
        &mut self,
        codes: &MultiScaleAudioCodes,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer> {
        self.decode(&codes.clone().into_uniform()?, cancel)
    }
    fn streaming(&self) -> bool {
        false
    }
    fn encode_step(
        &mut self,
        _audio: &AudioBuffer,
        _cancel: &CancellationToken,
    ) -> Result<Option<AudioCodes>> {
        Err(Error::Unsupported(
            "codec does not implement streaming encode".into(),
        ))
    }
    fn decode_step(
        &mut self,
        _codes: &AudioCodes,
        _cancel: &CancellationToken,
    ) -> Result<Option<AudioBuffer>> {
        Err(Error::Unsupported(
            "codec does not implement streaming decode".into(),
        ))
    }
}
impl<T: AudioCodec + ?Sized> AudioCodec for Box<T> {
    fn format(&self) -> CodecFormat {
        (**self).format()
    }
    fn descriptor(&self) -> ModelCapabilities {
        (**self).descriptor()
    }
    fn encode(&mut self, audio: &AudioBuffer, cancel: &CancellationToken) -> Result<AudioCodes> {
        (**self).encode(audio, cancel)
    }
    fn decode(&mut self, codes: &AudioCodes, cancel: &CancellationToken) -> Result<AudioBuffer> {
        (**self).decode(codes, cancel)
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
    fn encode_multiscale(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
    ) -> Result<MultiScaleAudioCodes> {
        (**self).encode_multiscale(audio, cancel)
    }
    fn decode_multiscale(
        &mut self,
        codes: &MultiScaleAudioCodes,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer> {
        (**self).decode_multiscale(codes, cancel)
    }
    fn streaming(&self) -> bool {
        (**self).streaming()
    }
    fn encode_step(
        &mut self,
        audio: &AudioBuffer,
        cancel: &CancellationToken,
    ) -> Result<Option<AudioCodes>> {
        (**self).encode_step(audio, cancel)
    }
    fn decode_step(
        &mut self,
        codes: &AudioCodes,
        cancel: &CancellationToken,
    ) -> Result<Option<AudioBuffer>> {
        (**self).decode_step(codes, cancel)
    }
}

/// Ragged codebooks for codecs such as SNAC. `codebook_strides` are measured in
/// units of the finest code frame; no invented padding codes are inserted.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MultiScaleAudioCodes {
    format: CodecFormat,
    codebook_strides: Vec<usize>,
    codebooks: Vec<Vec<u32>>,
    original_audio_frames: Option<usize>,
    segments: Vec<CodecSegment>,
}
impl MultiScaleAudioCodes {
    pub fn new(
        format: CodecFormat,
        codebook_strides: Vec<usize>,
        codebooks: Vec<Vec<u32>>,
    ) -> Result<Self> {
        format.validate()?;
        if codebooks.len() != format.num_codebooks
            || codebook_strides.len() != codebooks.len()
            || codebook_strides.contains(&0)
            || codebooks
                .iter()
                .flatten()
                .any(|code| *code >= format.cardinality)
        {
            return Err(Error::InvalidInput("invalid multiscale codec codes".into()));
        }
        Ok(Self {
            format,
            codebook_strides,
            codebooks,
            original_audio_frames: None,
            segments: Vec::new(),
        })
    }
    pub fn format(&self) -> &CodecFormat {
        &self.format
    }
    pub fn codebook_strides(&self) -> &[usize] {
        &self.codebook_strides
    }
    pub fn codebooks(&self) -> &[Vec<u32>] {
        &self.codebooks
    }
    pub fn original_audio_frames(&self) -> Option<usize> {
        self.original_audio_frames
    }
    pub fn with_original_audio_frames(mut self, frames: usize) -> Result<Self> {
        let capacity = self
            .codebooks
            .iter()
            .zip(&self.codebook_strides)
            .map(|(codes, stride)| codes.len().saturating_mul(*stride))
            .min()
            .unwrap_or(0);
        let samples = (capacity as f64 / self.format.frame_rate
            * self.format.audio_format.sample_rate() as f64)
            .ceil();
        if frames as f64 > samples {
            return Err(Error::InvalidInput(
                "original audio exceeds multiscale encoded capacity".into(),
            ));
        }
        self.original_audio_frames = Some(frames);
        Ok(self)
    }
    pub fn into_uniform(self) -> Result<AudioCodes> {
        if self.codebook_strides.iter().any(|stride| *stride != 1)
            || self
                .codebooks
                .windows(2)
                .any(|books| books[0].len() != books[1].len())
        {
            return Err(Error::Unsupported(
                "multiscale codebooks are not uniform".into(),
            ));
        }
        let frames = self.codebooks.first().map_or(0, Vec::len);
        let mut values = Vec::with_capacity(frames * self.format.num_codebooks);
        for frame in 0..frames {
            for book in &self.codebooks {
                values.push(book[frame]);
            }
        }
        let mut codes = AudioCodes::new(self.format, values)?;
        codes.original_audio_frames = self.original_audio_frames;
        codes = codes.with_segments(self.segments)?;
        Ok(codes)
    }
    pub fn from_uniform(codes: AudioCodes) -> Self {
        let mut books = vec![Vec::with_capacity(codes.frames()); codes.format.num_codebooks];
        for frame in codes.codes.chunks_exact(codes.format.num_codebooks) {
            for (book, code) in books.iter_mut().zip(frame) {
                book.push(*code);
            }
        }
        Self {
            codebook_strides: vec![1; codes.format.num_codebooks],
            format: codes.format,
            codebooks: books,
            original_audio_frames: codes.original_audio_frames,
            segments: codes.segments,
        }
    }
}
impl<'de> Deserialize<'de> for MultiScaleAudioCodes {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            format: CodecFormat,
            codebook_strides: Vec<usize>,
            codebooks: Vec<Vec<u32>>,
            #[serde(default)]
            original_audio_frames: Option<usize>,
            #[serde(default)]
            segments: Vec<CodecSegment>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let mut codes = Self::new(raw.format, raw.codebook_strides, raw.codebooks)
            .map_err(serde::de::Error::custom)?;
        if let Some(frames) = raw.original_audio_frames {
            codes = codes
                .with_original_audio_frames(frames)
                .map_err(serde::de::Error::custom)?;
        }
        if !raw.segments.is_empty() {
            let uniform = codes
                .into_uniform()
                .map_err(serde::de::Error::custom)?
                .with_segments(raw.segments)
                .map_err(serde::de::Error::custom)?;
            codes = Self::from_uniform(uniform);
        }
        Ok(codes)
    }
}
