use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
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

/// A pool of command buffers that distributes work across multiple buffers to improve
/// parallelism between CPU encoding and GPU execution.
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
            state: Mutex::new(EntryState {
                current: cb,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn acquire_compute_encoder(
        &self,
    ) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        let (flushed, encoder) = self.finalize_entry(entry, |cb| cb.compute_command_encoder())?;
        Ok((flushed, encoder))
    }

    pub fn acquire_blit_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        let (flushed, encoder) = self.finalize_entry(entry, |cb| cb.blit_command_encoder())?;
        Ok((flushed, encoder))
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
        let flushed = count >= self.compute_per_buffer;

        if flushed {
            state.current.commit();
            let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
            let old_cb = std::mem::replace(&mut state.current, new_cb);
            state.in_flight.push(old_cb);
            entry.compute_count.store(1, Ordering::Release);
        }

        let encoder = create_encoder(&mut state.current);

        Ok((flushed, encoder))
    }

    /// Flushes all buffers and waits for their completion.
    /// Commits any pending work on the current buffers, moves them to in-flight,
    /// then waits on all in-flight buffers including those from prior recycles.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Ensure no active encoder is still encoding on this entry.
            {
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));
            }

            // Under state lock, commit current if it has pending work and swap to a fresh one.
            let to_wait: Vec<CommandBuffer> = {
                let mut state = entry.state.lock()?;

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    state.current.commit();
                    let new_cb =
                        create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
                    let old_cb = std::mem::replace(&mut state.current, new_cb);
                    state.in_flight.push(old_cb);
                    entry.compute_count.store(0, Ordering::Release);
                }

                // Drain in_flight into a local vec to wait without holding the lock.
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
            self.flush_entry(entry)?;
        }

        Ok(())
    }

    fn flush_entry(&self, entry: &Arc<CommandBufferEntry>) -> Result<(), MetalKernelError> {
        // Ensure no active encoder is still encoding on this entry.
        let _guard = entry
            .semaphore
            .wait_until(|s| matches!(s, CommandStatus::Available));

        let mut state = entry.state.lock()?;

        if entry.compute_count.load(Ordering::Acquire) > 0 {
            state.current.commit();
            let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
            let old_cb = std::mem::replace(&mut state.current, new_cb);
            state.in_flight.push(old_cb);
            entry.compute_count.store(0, Ordering::Release);
        }

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
                    .map(|e| e.into_owned())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
