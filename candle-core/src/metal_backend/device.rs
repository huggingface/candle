use crate::{DType, Result};
use candle_metal_kernels::Kernels;
use metal::{
    Buffer, BufferRef, CommandBuffer, CommandQueue, ComputeCommandEncoder, MTLResourceOptions,
    NSUInteger,
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
type ActiveCommandBuffer = Arc<RwLock<Option<CommandBuffer>>>;
type ActiveComputeCommandEncoder = Arc<RwLock<Option<ComputeCommandEncoder>>>;

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    id: DeviceId,

    /// Metal device instance
    /// <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    device: metal::Device,

    /// Single command queue for the entire device.
    /// <https://developer.apple.com/documentation/metal/mtlcommandqueue?language=objc>
    command_queue: CommandQueue,

    /// Single command buffer for the entire device.
    /// <https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc>
    /// The command buffer is lazily created on the first call to [`command_buffer`]. This is to ensure that we
    /// don't need to manage cleaning up an empty command buffer on the last tensor, which usually has a call to [`to_cpu`].
    command_buffer: ActiveCommandBuffer,

    /// Track the count of encoded commands.
    /// We do this to cap the maximum number of commands that can be encoded in a single command buffer.
    /// This is to prevent system crashes due to OOM errors.
    command_buffer_accesses: Arc<RwLock<usize>>,

    /// Maximum number of commands that can be encoded in a single command buffer.
    max_command_buffer_accesses: usize,

    /// The active command encoder for use by the kernels, specifically is of type compute encoder.
    /// <https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc>
    /// Some notes on the command encoder:
    /// - The command encoder is lazily created on calls to [`command_encoder`].
    /// - Multiple command encoders CAN be created on a single command buffer, but only one can be active at a time.
    /// - The only way a new command encoder should be created is via the [`command_encoder`] function.
    /// - The only way a command encoder should be ended is via the [`end_compute_encoding`] function.
    command_encoder: ActiveComputeCommandEncoder,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    kernels: Arc<Kernels>,

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
    buffers: AllocatedBuffers,

    /// Seed for random number generation.
    seed: Arc<Mutex<Buffer>>,
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

impl From<metal::Device> for MetalDevice {
    fn from(device: metal::Device) -> Self {
        let id = DeviceId::new();
        let command_queue = device.new_command_queue();
        let kernels = Arc::new(Kernels::new());
        let buffers = Arc::new(RwLock::new(HashMap::new()));
        let seed = Arc::new(Mutex::new(device.new_buffer_with_data(
            [299792458].as_ptr() as *const c_void,
            4,
            MTLResourceOptions::StorageModeManaged,
        )));
        let max_command_buffer_accesses = std::env::var("CANDLE_MAX_COMMAND_BUFFER_ACCESSES")
            .unwrap_or("100".to_string())
            .parse()
            .unwrap();
        Self {
            id,
            device,
            command_queue,
            command_buffer: Arc::new(RwLock::new(None)),
            command_encoder: Arc::new(RwLock::new(None)),
            command_buffer_accesses: Arc::new(RwLock::new(0)),
            kernels,
            buffers,
            seed,
            max_command_buffer_accesses,
        }
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

    pub fn seed(&self) -> Arc<Mutex<Buffer>> {
        self.seed.clone()
    }

    /// Returns the current active command buffer, if the buffer is not present, this will allocate one
    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        // Check if the command buffer has reached the maximum number of accesses
        let accesses = {
            let accesses = self
                .command_buffer_accesses
                .try_read()
                .map_err(MetalError::from)?;
            accesses.clone()
        };

        // If the command buffer has reached the maximum number of accesses, we need to close it
        if accesses >= self.max_command_buffer_accesses {
            self.close_compute_buffer()?;
        } else {
            let mut accesses = self
                .command_buffer_accesses
                .try_write()
                .map_err(MetalError::from)?;
            *accesses += 1;
        }

        // Provision a new command buffer if there is none
        let mut command_buffer_lock = self.command_buffer.try_write().map_err(MetalError::from)?;
        let command_buffer = command_buffer_lock.to_owned();
        if let Some(command_buffer) = command_buffer {
            Ok(command_buffer)
        } else {
            let new_command_buffer = self.command_queue.new_command_buffer().to_owned();
            *command_buffer_lock = Some(new_command_buffer.clone());
            Ok(new_command_buffer)
        }
    }

    /// Returns the current active command encoder, if the encoder is not present, this will allocate one
    pub fn command_encoder(&self) -> Result<ComputeCommandEncoder> {
        let mut command_encoder_lock =
            self.command_encoder.try_write().map_err(MetalError::from)?;
        let command_encoder = command_encoder_lock.to_owned();

        if let Some(command_encoder) = command_encoder {
            Ok(command_encoder)
        } else {
            let command_buffer = self.command_buffer()?;
            let new_command_encoder = command_buffer.new_compute_command_encoder().to_owned();
            *command_encoder_lock = Some(ComputeCommandEncoder::from(new_command_encoder.clone()));
            Ok(new_command_encoder)
        }
    }

    /// Ends the current command buffer
    pub fn close_compute_buffer(&self) -> Result<()> {
        let mut command_buffer_lock = self.command_buffer.try_write().map_err(MetalError::from)?;
        let command_buffer = command_buffer_lock.to_owned();

        if let Some(command_buffer) = command_buffer {
            let mut accesses = self
                .command_buffer_accesses
                .try_write()
                .map_err(MetalError::from)?;
            command_buffer.commit();
            command_buffer.wait_until_completed();
            *command_buffer_lock = None;
            *accesses = 0;
        }

        Ok(())
    }

    /// If the user wants to end the current command encoder, they should call this function
    /// In the case that there is no active command encoder, this function will not return an error,
    /// it instead is a no-op.
    pub fn end_compute_encoding(&self) -> Result<()> {
        let mut command_encoder_lock =
            self.command_encoder.try_write().map_err(MetalError::from)?;
        let command_encoder = command_encoder_lock.as_ref();
        if let Some(command_encoder) = command_encoder {
            command_encoder.end_encoding();
            *command_encoder_lock = None;
        }
        Ok(())
    }

    /// Drops all buffers that are not currently in use, see the struct docs for more details about the buffer allocator
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

    /// Returns the pre-existing kernels that have been compiled so far
    pub fn kernels(&self) -> &Kernels {
        &self.kernels
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

    /// Allocates a new buffer with zeros.
    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let data = vec![0u8; size_in_bytes];
        self.new_buffer_with_data(&data)
    }

    /// Finds the best buffer to reuse.
    /// The best buffer is the one that is the smallest and can fit the requested size.
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

            // Early exit if we found a buffer that exactly fits the size
            if best_buffer.is_some() && best_buffer_size == size {
                break;
            }
        }
        best_buffer.cloned()
    }

    /// The critical allocator algorithm
    /// Allocates a new buffer with the given size and options. If there already exists a buffer
    /// that can be reused, it will be returned instead to avoid the overhead of creating a new buffer.
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

        // Buffers on metal should be powers of two, this is to encourage buffer reuse
        let size = (size - 1).next_power_of_two() as NSUInteger;

        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);
        let new_buffer = self.device.new_buffer(size as NSUInteger, option);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());

        Ok(new_buffer)
    }

    /// Copies data from one buffer to another.
    /// This method is blocking as it relies on GPU-CPU synchronization.
    /// This method does not end the current command buffer, it is the responsibility of the caller
    /// to do so via the [`close_compute_buffer`] function. This is to allow for multiple copies to be made
    /// without the overhead of creating a new command buffer each time in the case of not requiring the data
    /// to be accessible on CPU memory.
    pub fn copy_buffer(
        &self,
        source_buffer: &BufferRef,
        source_offset: NSUInteger,
        destination_buffer: &BufferRef,
        destination_offset: NSUInteger,
        size: NSUInteger,
    ) -> Result<()> {
        // Ensure that current work is complete before copying data
        self.synchronize()?;

        // Setup a new blit encoder and copy the data
        // There is no need to setup a new compute command encoder since it is handled by
        // the [`command_encoder`] function.
        let command_buffer = self.command_queue().new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            source_buffer,
            source_offset,
            destination_buffer,
            destination_offset,
            size,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
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

    /// Ends the current compute encoder and command buffer.
    /// If no compute buffer is allocated, this is a no-op.
    /// If you are looking to wait for data to be available for or compute to finish, please leverage
    /// memory barriers instead (https://developer.apple.com/documentation/metal/mtlrendercommandencoder/2967441-memorybarrierwithresources?language=objc).
    pub fn synchronize(&self) -> Result<()> {
        // Command buffers cannot be committed if there is an active encoder.
        self.end_compute_encoding()?;

        // Commit the command buffer and wait for it to finish
        self.close_compute_buffer()?;

        Ok(())
    }
}
