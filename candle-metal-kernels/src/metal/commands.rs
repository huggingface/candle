use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus, ComputeCommandEncoder,
};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue, MTLCommandBuffer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;
const DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5;

/// Creates a new command buffer from the queue with an attached semaphore for tracking its state.
pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
}

/// A pool entry containing a command buffer, its usage count, and synchronization primitives.
/// The `state` mutex guards the current buffer and the in-flight list for coherent updates.
/// `compute_count` and `semaphore` remain accessible without locking for selection/coordination.
pub struct CommandBufferEntry {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

pub struct Commands {
    /// Maintains a pool of command buffers, allowing
    /// the pool to balance load across multiple buffers and improve GPU utilization.
    /// Can be shared across threads safely.
    pool: Vec<Arc<CommandBufferEntry>>,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE),
            _ => DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE,
        };

        let pool = (0..pool_size)
            .map(|_| Self::create_pool_entry(&command_queue))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            pool,
            command_queue,
            compute_per_buffer,
        })
    }

    fn create_pool_entry(
        command_queue: &CommandQueue,
    ) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(command_queue, Arc::clone(&semaphore))?;

        Ok(Arc::new(CommandBufferEntry {
            state: Mutex::new(EntryState {
                current: cb,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.compute_command_encoder())
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.blit_command_encoder())
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    /// Run a closure with the current command buffer for the selected pool entry.
    /// This participates in the same compute_per_buffer accounting and recycling as
    /// encoder-based paths, so external encodes share the same commit cadence.
    pub fn with_command_buffer<F, R>(&self, f: F) -> Result<R, MetalKernelError>
    where
        F: FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> R,
    {
        let entry = self.select_entry()?;
        let mut state = entry.state.lock()?;

        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;
        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }

        let cb = state.current.clone();
        drop(state);

        let out = f(cb.as_ref());
        entry.semaphore.set_status(CommandStatus::Available);
        Ok(out)
    }

    // Selects an entry from the pool using a two-phase strategy:
    /// 1. Try non-blocking: find any available buffer without waiting
    /// 2. Fallback: select the least-loaded buffer and wait for availability
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Phase 1: Try to find an available buffer without blocking
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Phase 2: Select the buffer with the most work and wait for it
        let entry = self
            .pool
            .iter()
            .max_by_key(|e| e.compute_count.load(Ordering::Acquire))
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Command buffer pool is empty".to_string(),
            ))?;

        let entry = Arc::clone(entry);
        {
            let mut guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }

        Ok(entry)
    }

    /// Creates an encoder from the selected entry, recycling the buffer if needed.
    /// When recycling, the old committed buffer is moved to `in_flight` so we can later wait on it.
    fn finalize_entry<F, E>(
        &self,
        entry: Arc<CommandBufferEntry>,
        create_encoder: F,
    ) -> Result<(bool, E), MetalKernelError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        let mut state = entry.state.lock()?;

        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }

        let encoder = create_encoder(&mut state.current);

        Ok((flush, encoder))
    }

    /// Flushes all buffers and waits for their completion.
    /// Commits any pending work on the current buffers, moves them to in-flight,
    /// then waits on all in-flight buffers including those from prior recycles.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Under state lock, commit current if it has pending work and swap to a fresh one.
            let to_wait: Vec<CommandBuffer> = {
                // Ensure no active encoder is still encoding on this entry.
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));

                let mut state = entry.state.lock()?;

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(&entry, &mut state, 0)?;
                }

                // Drain `in_flight` into a local vec to wait without holding the lock.
                // Replaces `state.in_flight` with an empty vec and returns its previous contents.
                std::mem::take(&mut state.in_flight)
            };

            for cb in to_wait {
                Self::ensure_completed(&cb)?;
            }
        }

        Ok(())
    }

    /// Flushes all buffers without waiting for completion.
    /// Commits any pending work and moves current buffers to in-flight.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let _guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut state = entry.state.lock()?;

            if entry.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&entry, &mut state, 0)?;
            }
        }

        Ok(())
    }

    /// Commit the current command buffer, swap in a fresh one, push the old into `in_flight`,
    /// and reset `compute_count` to `reset_to`.
    fn commit_swap_locked(
        &self,
        entry: &CommandBufferEntry,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        state.current.commit();
        let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        entry.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // TODO: Avoid redundant allocation before drop
        let _ = self.flush();
    }
}
