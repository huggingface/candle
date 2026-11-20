//! Text-conditioned audio and music generation, independently of speech synthesis.
use crate::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioGenerationRequest {
    pub prompt: String,
    pub seed: u64,
    /// Number of new codec frames (MusicGen uses 50 Hz). A MusicGen reference
    /// is additional context; its reconstructed prefix is included in output.
    pub max_steps: usize,
    pub guidance_scale: f64,
    /// `None` selects greedy decoding.
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    /// Optional continuation prompt, in the model's advertised input format.
    /// MusicGen returns its codec reconstruction followed by new audio; the
    /// prefix is not a bit-exact copy and may round up to a codec frame.
    pub audio_reference: Option<AudioBuffer>,
}
impl AudioGenerationRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            seed: 0,
            max_steps: 250,
            guidance_scale: 3.,
            temperature: Some(1.),
            top_k: Some(250),
            top_p: None,
            audio_reference: None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.prompt.trim().is_empty() || self.prompt.len() > 16384 {
            return Err(Error::InvalidInput(
                "audio generation prompt must contain 1–16384 bytes".into(),
            ));
        }
        if self.max_steps == 0
            || self.max_steps > 4096
            || !self.guidance_scale.is_finite()
            || self.guidance_scale < 1.
            || self.temperature.is_some_and(|v| !v.is_finite() || v <= 0.)
            || self.top_k == Some(0)
            || self
                .top_p
                .is_some_and(|v| !v.is_finite() || v <= 0. || v > 1.)
        {
            return Err(Error::InvalidInput(
                "invalid audio generation sampling or step budget".into(),
            ));
        }
        Ok(())
    }
}
/// PCM callbacks may be buffered until the model's vocoder finishes. This trait
/// does not imply native streaming or synthesis with named voices.
pub trait AudioGenerationModel {
    fn descriptor(&self) -> ModelCapabilities;
    fn generate(
        &mut self,
        request: &AudioGenerationRequest,
        cancel: &CancellationToken,
        sink: &mut dyn FnMut(AudioBuffer) -> Result<()>,
    ) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
    fn finish_reason(&self) -> FinishReason;
}
impl<T: AudioGenerationModel + ?Sized> AudioGenerationModel for Box<T> {
    fn descriptor(&self) -> ModelCapabilities {
        (**self).descriptor()
    }
    fn generate(
        &mut self,
        request: &AudioGenerationRequest,
        cancel: &CancellationToken,
        sink: &mut dyn FnMut(AudioBuffer) -> Result<()>,
    ) -> Result<()> {
        (**self).generate(request, cancel, sink)
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
    fn finish_reason(&self) -> FinishReason {
        (**self).finish_reason()
    }
}
