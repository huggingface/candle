//! Compiled SPIR-V binaries, generated from `shaders/*.comp` by `build.rs` via naga.

const AFFINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine.spv"));
const REDUCE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce.spv"));

/// The set of compiled compute shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Reduce,
}

impl Source {
    /// Size of the push constant block for this shader. The pipeline
    /// layout must declare exactly the bytes the shader accesses,
    /// otherwise dispatch fails validation.
    pub fn push_constant_size(self) -> u32 {
        match self {
            Self::Affine => 12,
            Self::Reduce => 8,
        }
    }

    /// Raw SPIR-V words (little-endian u32) for this shader.
    pub fn spv_words(self) -> Vec<u32> {
        let bytes: &[u8] = match self {
            Self::Affine => AFFINE_SPV,
            Self::Reduce => REDUCE_SPV,
        };
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}
