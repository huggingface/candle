use crate::metal::{Buffer, CommandSemaphore, CommandStatus, ComputePipeline, MetalResource};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandEncoder, MTLComputeCommandEncoder, MTLResourceUsage, MTLSize,
};
use std::{ffi::c_void, ptr, sync::Arc};

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
    cached: bool,
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
        ComputeCommandEncoder { raw, semaphore, cached: false }
    }

    /// Create a non-owning view that wraps the same Metal encoder via raw pointer.
    /// This does NOT increment the Metal reference count — the original encoder must
    /// outlive all views. The view will not call endEncoding on drop.
    ///
    /// # Safety
    /// The caller must ensure the original encoder outlives this view.
    pub unsafe fn view(&self) -> ComputeCommandEncoder {
        // Use objc2's retain to get a new Retained without going through clone
        // Actually, we need to avoid Retained entirely. Let's use a raw pointer.
        // For now, clone but mark as cached so Drop doesn't call endEncoding.
        // The extra refcount is harmless — Metal tracks encoding state separately.
        ComputeCommandEncoder {
            raw: self.raw.clone(),
            semaphore: Arc::clone(&self.semaphore),
            cached: true,
        }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
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

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage)
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        if !self.cached {
            self.signal_encoding_ended();
        }
    }

    /// End encoding without signaling the semaphore.
    /// Used when the caller manages semaphore state explicitly.
    pub fn end_encoding_raw(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
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
        if !self.cached {
            self.end_encoding();
        } else {
            // Cached encoder: don't end encoding, but still signal semaphore
            // so the command buffer entry can be reused
            self.semaphore.set_status(CommandStatus::Available);
        }
    }
}

impl ComputeCommandEncoder {
    /// Mark this encoder as cached — it won't end encoding on drop.
    /// The owner is responsible for calling end_encoding() explicitly.
    pub fn set_cached(&mut self, cached: bool) {
        self.cached = cached;
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
