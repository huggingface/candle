use rocm_rs::hip::Dim3;

/// Buffer offset helper for kernel operations
pub struct BufferOffset {
    pub offset_in_bytes: usize,
}

impl BufferOffset {
    /// Create a zero offset for a buffer
    pub fn zero_offset<T>(_: &rocm_rs::hip::DeviceMemory<T>) -> Self {
        Self { offset_in_bytes: 0 }
    }
}

const BLOCK_SIZE: u32 = 256;

/// Calculate grid and block configuration for a given number of elements.
///
/// Returns (grid_dims, block_dims) suitable for kernel launch.
pub fn grid_block_config(num_elems: usize) -> (Dim3, Dim3) {
    let num_blocks = ((num_elems as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).min(65535);
    (
        Dim3 {
            x: num_blocks,
            y: 1,
            z: 1,
        },
        Dim3 {
            x: BLOCK_SIZE,
            y: 1,
            z: 1,
        },
    )
}
