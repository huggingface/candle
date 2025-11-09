use crate::metal::{BlitCommandEncoder, CommandBufferPool, ComputeCommandEncoder};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandQueue, MTLCounterSet};
use std::sync::Arc;

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CounterSet = Retained<ProtocolObject<dyn MTLCounterSet>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;
const DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5;

/// Commands manager backed by a command buffer pool for improved CPU-GPU parallelism.
///
/// This struct maintains a pool of command buffers, allowing
/// the pool to balance load across multiple buffers and improve GPU utilization.
/// The API remains the same as the original single-buffer implementation, but the
/// implementation now uses the pool internally.
///
/// Thread-safe: Can be shared across threads safely.
pub struct Commands {
    pool: Arc<CommandBufferPool>,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse().unwrap_or(50),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val.parse().unwrap_or(4),
            _ => DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE,
        };

        let pool = CommandBufferPool::new(command_queue, pool_size, compute_per_buffer)?;

        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let (recycled, encoder) = self.pool.acquire_compute_encoder()?;
        Ok((recycled, encoder))
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let (recycled, encoder) = self.pool.acquire_blit_encoder()?;
        Ok((recycled, encoder))
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.pool.flush_and_wait()
    }
}
