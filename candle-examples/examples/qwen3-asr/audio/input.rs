//! Audio input types accepted by the high-level inference API.

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum AudioInput<'a> {
    Path(&'a Path),
    Waveform {
        samples: &'a [f32],
        sample_rate: u32,
    },
}
