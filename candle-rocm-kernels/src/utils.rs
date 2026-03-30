pub struct BufferOffset {
    pub offset_in_bytes: usize,
}

impl BufferOffset {
    pub fn zero_offset<T>(_: &rocm_rs::hip::DeviceMemory<T>) -> Self {
        Self { offset_in_bytes: 0 }
    }
}

pub fn launch_config(num_elems: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    const BLOCK_SIZE: u32 = 256;
    let num_blocks = ((num_elems as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).min(65535);
    ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))
}

pub fn launch_config_for_num_elems(num_elems: usize) -> (u32, u32, u32) {
    const BLOCK_SIZE: usize = 256;
    let num_blocks = ((num_elems + BLOCK_SIZE - 1) / BLOCK_SIZE).min(65535);
    (num_blocks as u32, 1, 1)
}

pub fn get_grid_block_config(num_elems: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    launch_config(num_elems)
}
