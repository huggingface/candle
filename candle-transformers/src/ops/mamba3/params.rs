//! Configuration for Mamba-3 GPU operators.

#[derive(Debug, Clone, Copy)]
pub struct Mamba3Params {
    pub chunk_size: usize,
    pub headdim: usize,
    pub d_state: usize,
    pub nheads: usize,
    pub mimo_rank: usize,
    pub num_rope_angles: usize,
    pub a_floor: f32,
}

impl Mamba3Params {
    pub fn siso_chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn mimo_chunk_size(&self) -> usize {
        self.chunk_size / self.mimo_rank.max(1)
    }

    pub fn is_mimo(&self) -> bool {
        self.mimo_rank > 1
    }
}
