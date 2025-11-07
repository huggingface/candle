use objc2_metal::MTLCommandQueue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandQueue, CommandSemaphore, CommandStatus,
    ComputeCommandEncoder,
};
use crate::MetalKernelError;

/// Creates a new command buffer for the given queue with synchronization via semaphore.
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

pub struct CommandBufferEntry {
    command_buffer: Mutex<CommandBuffer>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

/// Pool of command buffers for efficient reuse and scheduling.
///
/// Uses a two-tier strategy for entry selection:
/// 1. Fast path: Scans for an immediately available entry
/// 2. Fallback: Waits for the least-loaded entry
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

        // let mut pool = Vec::with_capacity(pool_size);
        // for id in 0..pool_size {
        //     let semaphore = Arc::new(CommandSemaphore::new());
        //     let cb = create_command_buffer(&command_queue, Arc::clone(&semaphore))?;
        //     pool.push(Arc::new(CommandBufferEntry {
        //         command_buffer: Mutex::new(cb),
        //         compute_count: AtomicUsize::new(0),
        //         semaphore,
        //     }))
        // }

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
        let encoder = self.finalize_compute_entry(&entry)?;
        Ok(encoder) // Just return the encoder directly!
    }

    pub fn acquire_blit_encoder(&self) -> Result<BlitCommandEncoder, MetalKernelError> {
        let entry = self.select_entry()?;
        let encoder = self.finalize_blit_entry(&entry)?;
        Ok(encoder) // Just return the encoder directly!
    }

    /// Selects a command buffer entry using a two-tier strategy:
    /// 1. Fast path: Try to find an available entry without blocking
    /// 2. Fallback: Wait for the least-loaded entry to become available
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Try fast path: scan for immediately available entry
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    entry.semaphore.cond.notify_one();
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Fallback: wait for the least-loaded entry to become available
        let entry = self
            .pool
            .iter()
            .min_by_key(|e| e.compute_count.load(Ordering::Relaxed))
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

    /// Finalizes the entry by creating or reusing a compute encoder.
    /// Commits and replaces the command buffer if it has reached the compute limit.
    fn finalize_compute_entry(
        &self,
        entry: &Arc<CommandBufferEntry>,
    ) -> Result<ComputeCommandEncoder, MetalKernelError> {
        let encoder = {
            let mut cb = entry.command_buffer.lock().map_err(|_| {
                MetalKernelError::FailedToCreateResource(
                    "Command buffer mutex poisoned".to_string(),
                )
            })?;

            let count = entry.compute_count.load(Ordering::Relaxed);
            if count >= self.compute_per_buffer {
                // Commit current buffer and create new one
                cb.commit();
                let new_cb =
                    create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
                *cb = new_cb;
                entry.compute_count.store(0, Ordering::Relaxed);
            }

            entry.compute_count.fetch_add(1, Ordering::Relaxed);
            cb.compute_command_encoder()
        };

        Ok(encoder)
    }

    fn finalize_blit_entry(
        &self,
        entry: &Arc<CommandBufferEntry>,
    ) -> Result<BlitCommandEncoder, MetalKernelError> {
        let encoder = {
            let mut cb = entry.command_buffer.lock().map_err(|_| {
                MetalKernelError::FailedToCreateResource(
                    "Command buffer mutex poisoned".to_string(),
                )
            })?;

            let count = entry.compute_count.load(Ordering::Relaxed);
            if count >= self.compute_per_buffer {
                cb.commit();
                let new_cb =
                    create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
                *cb = new_cb;
                entry.compute_count.store(0, Ordering::Relaxed);
            }

            entry.compute_count.fetch_add(1, Ordering::Relaxed);
            cb.blit_command_encoder()
        };

        Ok(encoder)
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            self.flush_entry(entry)?;
        }
        Ok(())
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let pending_count = entry.compute_count.load(Ordering::Relaxed);

            if pending_count > 0 {
                let mut cb = entry.command_buffer.lock().map_err(|_| {
                    MetalKernelError::FailedToCreateResource(
                        "Command buffer mutex poisoned".to_string(),
                    )
                })?;

                // Commit and wait for completion
                cb.commit();
                cb.wait_until_completed();

                // Create fresh buffer and reset state
                let new_cb =
                    create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
                *cb = new_cb;
                entry.compute_count.store(0, Ordering::Relaxed);
                drop(cb);

                // Reset status to available
                entry.semaphore.set_status(CommandStatus::Available);
            }
        }

        Ok(())
    }

    fn flush_entry(&self, entry: &Arc<CommandBufferEntry>) -> Result<(), MetalKernelError> {
        let pending_count = entry.compute_count.load(Ordering::Relaxed);

        if pending_count > 0 {
            let mut cb = entry.command_buffer.lock().map_err(|_| {
                MetalKernelError::FailedToCreateResource(
                    "Command buffer mutex poisoned".to_string(),
                )
            })?;

            cb.commit();
            // Create a new command buffer to replace the committed one
            let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
            *cb = new_cb;
            entry.compute_count.store(0, Ordering::Relaxed);
        }

        Ok(())
    }
}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
