use objc2_metal::MTLCommandQueue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandQueue, CommandSemaphore, CommandStatus,
    ComputeCommandEncoder,
};
use crate::MetalKernelError;

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

/// A pool entry containing a command buffer, its usage count, and synchronization primitives.
/// The mutex guards the command buffer while the atomic count and semaphore can be accessed
/// without locking for efficient load balancing.
pub struct CommandBufferEntry {
    command_buffer: Mutex<CommandBuffer>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

/// A pool of command buffers that distributes work across multiple buffers to improve
/// parallelism between CPU encoding and GPU execution.
///
/// The pool automatically recycles buffers when they reach their compute limit and
/// balances load across available entries.
pub struct CommandBufferPool {
    pool: Vec<Arc<CommandBufferEntry>>,
    command_queue: CommandQueue,
    compute_per_buffer: usize,
}

impl CommandBufferPool {
    pub fn new(
        command_queue: CommandQueue,
        pool_size: usize,
        compute_per_buffer: usize,
    ) -> Result<Self, MetalKernelError> {
        if pool_size == 0 {
            return Err(MetalKernelError::FailedToCreateResource(
                "Pool size must be greater than 0".to_string(),
            ));
        }

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
            command_buffer: Mutex::new(cb),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn acquire_compute_encoder(&self) -> Result<ComputeCommandEncoder, MetalKernelError> {
        let entry = self.select_entry()?;
        let encoder = self.finalize_entry(entry, |cb| cb.compute_command_encoder())?;
        Ok(encoder)
    }

    pub fn acquire_blit_encoder(&self) -> Result<BlitCommandEncoder, MetalKernelError> {
        let entry = self.select_entry()?;
        let encoder = self.finalize_entry(entry, |cb| cb.blit_command_encoder())?;
        Ok(encoder)
    }

    /// Selects an entry from the pool using a two-phase strategy:
    /// 1. Try non-blocking: find any available buffer without waiting
    /// 2. Fallback: select the least-loaded buffer and wait for availability
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Phase 1: Try to find an available buffer without blocking
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    entry.semaphore.cond.notify_one();
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Phase 2: Select the buffer with the least work and wait for it
        let entry = self
            .pool
            .iter()
            .min_by_key(|e| e.compute_count.load(Ordering::Acquire))
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
        entry.semaphore.cond.notify_one();
        Ok(entry)
    }

    /// Creates an encoder from the selected entry, recycling the buffer if needed.
    fn finalize_entry<F, E>(
        &self,
        entry: Arc<CommandBufferEntry>,
        create_encoder: F,
    ) -> Result<E, MetalKernelError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        let encoder = {
            let mut cb = entry.command_buffer.lock().map_err(|_| {
                MetalKernelError::FailedToCreateResource(
                    "Command buffer mutex poisoned".to_string(),
                )
            })?;

            let count = entry.compute_count.load(Ordering::Acquire);
            // Recycle: commit and create fresh buffer if limit reached
            if count >= self.compute_per_buffer {
                cb.commit();
                let new_cb =
                    create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
                *cb = new_cb;
                entry.compute_count.store(0, Ordering::Release);
            }

            entry.compute_count.fetch_add(1, Ordering::Release);
            create_encoder(&mut cb)
        };

        Ok(encoder)
    }

    /// Flushes all buffers and waits for their completion.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let pending_count = entry.compute_count.load(Ordering::Acquire);

            if pending_count > 0 {
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));

                let mut cb = entry.command_buffer.lock().map_err(|_| {
                    MetalKernelError::FailedToCreateResource(
                        "Command buffer mutex poisoned".to_string(),
                    )
                })?;

                cb.commit();
                cb.wait_until_completed();

                let new_cb =
                    create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
                *cb = new_cb;
                entry.compute_count.store(0, Ordering::Release);
            }
        }

        Ok(())
    }

    /// Flushes all buffers without waiting for completion.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            self.flush_entry(entry)?;
        }
        Ok(())
    }

    fn flush_entry(&self, entry: &Arc<CommandBufferEntry>) -> Result<(), MetalKernelError> {
        let pending_count = entry.compute_count.load(Ordering::Acquire);

        if pending_count > 0 {
            let _guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut cb = entry.command_buffer.lock().map_err(|_| {
                MetalKernelError::FailedToCreateResource(
                    "Command buffer mutex poisoned".to_string(),
                )
            })?;

            cb.commit();
            let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
            *cb = new_cb;
            entry.compute_count.store(0, Ordering::Release);
        }

        Ok(())
    }
}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
