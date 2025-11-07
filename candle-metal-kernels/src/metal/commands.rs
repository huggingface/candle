use crate::metal::{BlitCommandEncoder, CommandBufferPool, ComputeCommandEncoder};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandQueue, MTLCounterSet};
use std::sync::Arc;

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CounterSet = Retained<ProtocolObject<dyn MTLCounterSet>>;

/// Commands manager backed by a command buffer pool for improved CPU-GPU parallelism.
///
/// This struct maintains a pool of command buffers instead of a single buffer, allowing
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
    /// Creates a new Commands manager with a command buffer pool.
    ///
    /// # Arguments
    /// * `command_queue` - The Metal command queue to use
    ///
    /// # Environment Variables
    /// * `CANDLE_METAL_COMPUTE_PER_BUFFER` - Max encoders per buffer (default: 50)
    /// * `CANDLE_METAL_COMMAND_POOL_SIZE` - Number of buffers in pool (default: 4)
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse().unwrap_or(50),
            _ => 50,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val.parse().unwrap_or(4),
            _ => 4,
        };

        let pool = CommandBufferPool::new(command_queue, pool_size, compute_per_buffer)?;

        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let encoder = self.pool.acquire_compute_encoder()?;
        Ok((false, encoder))
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let encoder = self.pool.acquire_blit_encoder()?;
        Ok((false, encoder))
    }

    /// Flushes all pending work in the pool and waits for completion.
    ///
    /// This ensures all buffered encoders are submitted to the GPU and waits
    /// for them to complete execution. Useful as a synchronization point.
    ///
    /// This method is now immutable (`&self` instead of `&mut self`).
    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.pool.flush_and_wait()
    }

    /// Flushes all pending work without waiting (async from GPU perspective).
    ///
    /// Useful when you want to submit pending work but don't need to wait for results.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        self.pool.flush()
    }
}
