use crate::{DType, Result};
use candle_metal_kernels::Kernels;
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, MTLResourceOptions, NSUInteger,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

use super::MetalError;

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

type BufferMap = HashMap<(NSUInteger, MTLResourceOptions), Vec<Arc<Buffer>>>;
type AllocatedBuffers = Arc<RwLock<BufferMap>>;

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: metal::Device,

    /// Single command queue for the entire device.
    pub(crate) command_queue: CommandQueue,
    /// Single command buffer for the entire device.
    pub(crate) command_buffer: Arc<RwLock<Option<CommandBuffer>>>,
    /// Single command encoder for the entire device.
    pub(crate) command_encoder: Arc<RwLock<Option<ComputeCommandEncoder>>>,
    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Simple allocator struct.
    /// The buffers are stored in size buckets since ML tends to use similar shapes over and over.
    /// We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
    /// (could be linked to FFI communication overhead).
    ///
    /// Whenever a buffer has a strong_count==1, we can reuse it, it means it was dropped in the
    /// graph calculation, and only we the allocator kept a reference to it, therefore it's free
    /// to be reused. However, in order for this to work, we need to guarantee the order of
    /// operation, so that this buffer is not being used by another kernel at the same time.
    /// Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
    ///
    /// Whenever we actually allocate a new buffer, we make a full sweep to clean up unused buffers
    /// (strong_count = 1).
    pub(crate) buffers: AllocatedBuffers,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &metal::Device {
        &self.device
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let mut command_buffer_lock = self.command_buffer.try_write().map_err(MetalError::from)?;
        let command_buffer = command_buffer_lock.to_owned();

        if let Some(command_buffer) = command_buffer {
            Ok(command_buffer)
        } else {
            let new_command_buffer = self.command_queue.new_command_buffer().to_owned();
            *command_buffer_lock = Some(new_command_buffer);
            Ok(command_buffer_lock.to_owned().unwrap())
        }
    }

    pub fn command_encoder(&self) -> Result<ComputeCommandEncoder> {
        let mut command_encoder_lock =
            self.command_encoder.try_write().map_err(MetalError::from)?;
        let command_encoder = command_encoder_lock.to_owned();

        if let Some(command_encoder) = command_encoder {
            Ok(command_encoder)
        } else {
            let command_buffer = self.command_buffer()?;
            let new_command_encoder = command_buffer.new_compute_command_encoder().to_owned();
            *command_encoder_lock = Some(ComputeCommandEncoder::from(new_command_encoder));
            Ok(command_encoder_lock.to_owned().unwrap())
        }
    }

    /// Ends the current command buffer
    pub fn close_compute_buffer(&self) -> Result<()> {
        let mut command_buffer_lock = self.command_buffer.try_write().map_err(MetalError::from)?;
        let command_buffer = command_buffer_lock.to_owned();

        if let Some(command_buffer) = command_buffer {
            command_buffer.commit();
            command_buffer.wait_until_completed();
            *command_buffer_lock = None;
        }

        Ok(())
    }

    /// Ends the current encoder's encoding, this does not setup a new encoder, the consumer must do that.
    pub fn end_compute_encoding(&self) -> Result<()> {
        let mut command_encoder_lock =
            self.command_encoder.try_write().map_err(MetalError::from)?;
        let command_encoder = command_encoder_lock.to_owned();
        if let Some(command_encoder) = command_encoder {
            command_encoder.end_encoding();
            *command_encoder_lock = None;
        }
        Ok(())
    }

    pub fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.try_write().map_err(MetalError::from)?;
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(*s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLPrivate](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer data cannot be read on the CPU directly.
    ///
    /// [`name`] is only used to keep track of the resource origin in case of bugs
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = (element_count * dtype.size_in_bytes()) as NSUInteger;
        self.allocate_buffer(size, MTLResourceOptions::StorageModePrivate, name)
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer can be read on the CPU but will require manual
    /// synchronization when the CPU memory is modified
    /// Used as a bridge to gather data back from the GPU
    pub fn new_buffer_managed(&self, size: NSUInteger) -> Result<Arc<Buffer>> {
        self.allocate_buffer(size, MTLResourceOptions::StorageModeManaged, "managed")
    }

    /// Creates a new buffer from data.
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        let new_buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            size,
            MTLResourceOptions::StorageModeManaged,
        );
        let mut buffers = self.buffers.try_write().map_err(MetalError::from)?;
        let subbuffers = buffers
            .entry((size, MTLResourceOptions::StorageModeManaged))
            .or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        self.end_compute_encoding()?;

        let buffer = self.allocate_buffer(
            size_in_bytes as NSUInteger,
            MTLResourceOptions::StorageModePrivate,
            "allocate_zeros",
        )?;
        let command_buffer = self.command_buffer()?;
        let blit = command_buffer.new_blit_command_encoder();
        blit.fill_buffer(
            &buffer,
            metal::NSRange {
                location: 0,
                length: buffer.length(),
            },
            0,
        );
        blit.end_encoding();
        Ok(buffer)
    }

    fn find_available_buffer(
        &self,
        size: NSUInteger,
        option: MTLResourceOptions,
        buffers: &RwLockWriteGuard<BufferMap>,
    ) -> Option<Arc<Buffer>> {
        let mut best_buffer: Option<&Arc<Buffer>> = None;
        let mut best_buffer_size: NSUInteger = NSUInteger::MAX;
        for ((buffer_size, buffer_option), subbuffers) in buffers.iter() {
            if buffer_size >= &size && buffer_size < &best_buffer_size && buffer_option == &option {
                for sub in subbuffers {
                    if Arc::strong_count(sub) == 1 {
                        best_buffer = Some(sub);
                        best_buffer_size = *buffer_size;
                    }
                }
            }
        }
        best_buffer.cloned()
    }

    /// The critical allocator algorithm
    fn allocate_buffer(
        &self,
        size: NSUInteger,
        option: MTLResourceOptions,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.try_write().map_err(MetalError::from)?;
        if let Some(b) = self.find_available_buffer(size, option, &buffers) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }

        let size = buf_size(size);
        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);

        let new_buffer = self.device.new_buffer(size as NSUInteger, option);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());

        Ok(new_buffer)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(self);
        descriptor.set_output_url(path);

        capture
            .start_capture(&descriptor)
            .map_err(MetalError::from)?;
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.end_compute_encoding()?;
        self.close_compute_buffer()?;
        Ok(())
    }
}

fn buf_size(size: NSUInteger) -> NSUInteger {
    (size - 1).next_power_of_two() as NSUInteger
}
