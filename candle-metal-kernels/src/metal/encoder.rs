use crate::metal::{Buffer, CommandSemaphore, CommandStatus, ComputePipeline};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::{
    collections::HashSet,
    ffi::c_void,
    ptr,
    sync::{Arc, Mutex},
};

struct EncoderState {
    /// Buffer ptr values written since the last barrier (RAW/WAW detection).
    prev_outputs: HashSet<usize>,
    /// Buffer ptr values written by the current op, promoted to prev_outputs after dispatch.
    next_outputs: HashSet<usize>,
    /// Buffer ptr values read since the last barrier (WAR detection).
    prev_inputs: HashSet<usize>,
    /// Buffer ptr values read by the current op, promoted to prev_inputs after dispatch.
    next_inputs: HashSet<usize>,
    /// Whether a barrier is needed before the next dispatch.
    needs_barrier: bool,
}

impl EncoderState {
    fn new() -> Self {
        EncoderState {
            prev_outputs: HashSet::new(),
            next_outputs: HashSet::new(),
            prev_inputs: HashSet::new(),
            next_inputs: HashSet::new(),
            needs_barrier: false,
        }
    }
}

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
    /// Barrier tracking state, shared between the original encoder and any clone_refs.
    state: Arc<Mutex<EncoderState>>,
    /// If true, Drop calls `end_encoding`.
    end_on_drop: bool,
    /// Only meaningful when `end_on_drop` is false.
    /// If true, Drop signals the semaphore `Available` (clone_ref).
    signal_on_drop: bool,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}

impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            state: Arc::new(Mutex::new(EncoderState::new())),
            end_on_drop: true,
            signal_on_drop: true,
        }
    }

    /// Convert this encoder into a long lived original encoder.
    /// Drop becomes a noop. Only explicit end_encoding() calls will signal the semaphore.
    pub(crate) fn make_long_lived(&mut self) {
        self.end_on_drop = false;
        self.signal_on_drop = false;
    }

    /// Return a non-owning clone that shares barrier tracking state with this encoder.
    /// Dropping the clone signals the semaphore (`Available`) but does not call metal `endEncoding`.
    /// Only the original encoder ends the encoding.
    pub(crate) fn clone_ref(&self) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw: self.raw.clone(),
            semaphore: Arc::clone(&self.semaphore),
            state: Arc::clone(&self.state),
            end_on_drop: false,
            signal_on_drop: true,
        }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    /// Insert a memory barrier only when a RAW, WAW, or WAR hazard is detected.
    ///
    /// RAW/WAW: input or output buffer ptr was in prev_outputs (written since last barrier).
    /// WAR: output buffer ptr was in prev_inputs (read since last barrier, now being recycled).
    ///
    /// On barrier: replace both prev sets with the current op's sets.
    /// On no barrier: accumulate both prev sets with the current op's sets.
    fn auto_barrier(&self) {
        let mut s = self.state.lock().unwrap();
        if s.needs_barrier {
            self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            s.needs_barrier = false;
            s.prev_outputs = std::mem::take(&mut s.next_outputs);
            s.prev_inputs = std::mem::take(&mut s.next_inputs);
        } else {
            let next_out = std::mem::take(&mut s.next_outputs);
            s.prev_outputs.extend(next_out);
            let next_in = std::mem::take(&mut s.next_inputs);
            s.prev_inputs.extend(next_in);
        }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.auto_barrier();
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.auto_barrier();
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    /// Set a buffer as an input. Checks for RAW hazard and tracks the read.
    pub fn set_input_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&ptr) {
                s.needs_barrier = true;
            }
            s.next_inputs.insert(ptr);
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    /// Set a buffer as an output. Checks for WAW and WAR hazards, tracks the write.
    pub fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        if let Some(buf) = buffer {
            let ptr = buf.raw_ptr() as usize;
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&ptr) || s.prev_inputs.contains(&ptr) {
                s.needs_barrier = true;
            }
            s.next_outputs.insert(ptr);
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    /// Ends the Metal encoding session and resets intra encoder state.
    /// Signals the semaphore as `Available`.
    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        *self.state.lock().unwrap() = EncoderState::new();
        self.signal_encoding_ended();
    }

    /// End the Metal encoding session without signaling the semaphore.
    /// Used when the caller will immediately create a new encoder on the same
    /// command buffer (e.g. blit encoder) and needs the semaphore to stay `Encoding`.
    pub(crate) fn end_encoding_silent(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        *self.state.lock().unwrap() = EncoderState::new();
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        if self.end_on_drop {
            self.end_encoding();
        } else if self.signal_on_drop {
            // Clone ref. Release semaphore slot. Encoding continues on original.
            self.signal_encoding_ended();
        }
        // Original encoder means noop. `end_encoding` must be called explicitly.
    }
}

pub struct BlitCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder { raw, semaphore }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        self.signal_encoding_ended();
    }

    pub fn set_label(&self, label: &str) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    pub fn copy_from_buffer(
        &self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}
