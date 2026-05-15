use crate::metal::{
    BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline, Device,
    Fence, PrevCeOutputs, ResidencySet,
};
use crate::MetalKernelError;
use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;

fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer, MetalKernelError> {
    command_queue.commandBuffer().map(CommandBuffer::new).ok_or(
        MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()),
    )
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: MutexGuard<'a, EntryState>,
}

impl AsRef<ComputeCommandEncoder> for CommandsGuard<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.guard.current_encoder.as_ref().unwrap()
    }
}

impl CommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.as_ref().set_compute_pipeline_state(pipeline);
    }
}

/// RAII guard for blit command encoder operations.
pub struct BlitCommandsGuard<'a> {
    _guard: MutexGuard<'a, EntryState>,
    state: BlitCommandEncoder,
}

impl<'a> AsRef<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_ref(&self) -> &BlitCommandEncoder {
        &self.state
    }
}

impl<'a> AsMut<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_mut(&mut self) -> &mut BlitCommandEncoder {
        &mut self.state
    }
}

impl BlitCommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.as_mut()
            .copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size)
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.as_mut().fill_buffer(buffer, range, value);
    }
}

impl Drop for BlitCommandsGuard<'_> {
    fn drop(&mut self) {
        self.as_ref().end_encoding();
    }
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
    current_encoder: Option<ComputeCommandEncoder>,
}

impl EntryState {
    pub fn new(cb: CommandBuffer) -> EntryState {
        EntryState {
            current: cb,
            in_flight: vec![],
            current_encoder: None,
        }
    }
}

pub struct Commands {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    device: Device,
    /// Global cross-encoder output map. Maps buffer pointer to the fence of the last encoder
    /// that wrote it, enabling cross-command-buffer ordering for HazardTrackingModeUntracked.
    prev_ce_outputs: PrevCeOutputs,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: &ResidencySet,
    ) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        if let Some(raw) = residency_set.raw() {
            command_queue.addResidencySet(raw);
        }

        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue)?;

        Ok(Self {
            state: Mutex::new(EntryState::new(cb)),
            compute_count: AtomicUsize::new(0),
            command_queue,
            compute_per_buffer,
            device,
            prev_ce_outputs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        if state_guard.current_encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device));
            let enc = state_guard.current.compute_command_encoder(&fence);
            // Wait for all prior encoder fences before the first dispatch.
            // Using HazardTrackingModeUntracked implies that Metal does not automatically flush GPU caches
            // at encoder or command buffer boundaries.
            {
                use std::collections::HashSet;
                let map = self.prev_ce_outputs.lock().unwrap();
                let mut seen = HashSet::new();
                for f in map.values() {
                    let ptr = Arc::as_ptr(f) as usize;
                    if seen.insert(ptr) {
                        enc.wait_for_fence(f);
                    }
                }
            }
            state_guard.current_encoder = Some(enc);
        }

        Ok(CommandsGuard { guard: state_guard })
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // End compute encoder before starting blit.
        if let Some(enc) = state_guard.current_encoder.take() {
            self.end_encoding(enc);
        }

        let fence = Arc::new(Fence::new(&self.device));
        let encoder = state_guard
            .current
            .blit_command_encoder(&fence, &self.prev_ce_outputs);

        // Wait for all prior encoder fences before any blit commands execute.
        // Required for HazardTrackingModeUntracked: GPU caches are not auto-flushed.
        {
            use std::collections::HashSet;
            let map = self.prev_ce_outputs.lock().unwrap();
            let mut seen = HashSet::new();
            for f in map.values() {
                let ptr = Arc::as_ptr(f) as usize;
                if seen.insert(ptr) {
                    encoder.wait_for_fence(f);
                }
            }
        }

        Ok(BlitCommandsGuard {
            _guard: state_guard,
            state: encoder,
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        let to_wait = {
            let mut state = self.state.lock()?;
            if self.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&mut state, 0)?;
            }
            std::mem::take(&mut state.in_flight)
        };

        // Wait only on the last CB. Metal executes CBs in queue order, so all earlier
        // CBs are guaranteed complete when the last one is. Calling waitUntilCompleted on
        // each CB individually pays OS notification latency (~1-2ms) N times unnecessarily.
        if let Some(last) = to_wait.last() {
            Self::ensure_completed(last)?;
        }
        // Check earlier CBs for errors (no need to block — they're already done).
        for cb in &to_wait[..to_wait.len().saturating_sub(1)] {
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
        }

        self.prev_ce_outputs.lock()?.clear();

        Ok(())
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.state.lock()?;
        if self.compute_count.load(Ordering::Acquire) > 0 {
            self.commit_swap_locked(&mut state, 0)?;
        }
        Ok(())
    }

    fn commit_swap_locked(
        &self,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }

        match state.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                state.current.commit();
            }
            _ => {}
        }
        let new_cb = create_command_buffer(&self.command_queue)?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        self.compute_count.store(reset_to, Ordering::Release);

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

    fn end_encoding(&self, encoder: ComputeCommandEncoder) {
        use objc2_metal::MTLCommandEncoder as _;
        use objc2_metal::MTLComputeCommandEncoder as _;

        let all_outputs = {
            let s = encoder.state.lock().unwrap();
            s.all_outputs.clone()
        };

        {
            let mut prev_ce_outputs = self.prev_ce_outputs.lock().unwrap();
            // Register our outputs so subsequent encoders can wait for us.
            for output in all_outputs.iter() {
                let _ = prev_ce_outputs.insert(*output, encoder.fence.clone());
            }
        }

        // Signal this encoder's completion fence and end encoding.
        encoder.raw.updateFence(encoder.fence.raw());

        // Schedule cleanup of our output entries once the GPU completes.
        if !all_outputs.is_empty() {
            let fence_for_cleanup = Arc::clone(&encoder.fence);
            let map_for_cleanup = Arc::clone(&self.prev_ce_outputs);
            let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let mut map = map_for_cleanup.lock().unwrap();
                for &buf in &all_outputs {
                    if let Some(f) = map.get(&buf) {
                        if Arc::ptr_eq(f, &fence_for_cleanup) {
                            map.remove(&buf);
                        }
                    }
                }
            });
            unsafe {
                encoder
                    .command_buffer
                    .addCompletedHandler(RcBlock::as_ptr(&block))
            };
        }

        encoder.raw.endEncoding();
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
