use crate::{DType, Result};
use candle_metal_kernels::Kernels;
use metal::{Buffer, CommandBuffer, CommandQueue, MTLResourceOptions, NSUInteger};
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
    /// One command buffer at a time.
    /// The scheduler works by allowing multiple
    /// [ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// on a single command buffer. Using a single command buffer would be fastest on the GPU but
    /// prevents overlapping of CPU and GPU commands (because command buffer needs to be committed
    /// to start to work).
    /// Despite what the documentation says, command buffers are NOT ordered. They are ordered
    /// for their START time, but there's no guarantee that command buffer1 will finish before
    /// command buffer2 starts (or there are metal bugs there)
    pub(crate) command_buffer: Arc<RwLock<CommandBuffer>>,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    /// Arc, RwLock because of the interior mutability.
    pub(crate) command_buffer_index: Arc<RwLock<usize>>,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    pub(crate) compute_per_buffer: usize,
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
        let mut command_buffer_lock = self.command_buffer.write().map_err(MetalError::from)?;
        let mut command_buffer = command_buffer_lock.to_owned();
        let mut index = self
            .command_buffer_index
            .write()
            .map_err(MetalError::from)?;
        if *index > self.compute_per_buffer {
            command_buffer.commit();
            command_buffer = self.command_queue.new_command_buffer().to_owned();
            *command_buffer_lock = command_buffer.clone();
            *index = 0;

            self.drop_unused_buffers()?;
        }
        *index += 1;
        Ok(command_buffer)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let mut command_buffer = self.command_buffer.write().map_err(MetalError::from)?;
        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                panic!("Already committed");
            }
            _ => {}
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        *command_buffer = self.command_queue.new_command_buffer().to_owned();

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
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        let subbuffers = buffers
            .entry((size, MTLResourceOptions::StorageModeManaged))
            .or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(
            size_in_bytes as NSUInteger,
            MTLResourceOptions::StorageModePrivate,
            "allocate_zeros",
        )?;
        let command_buffer = self.command_buffer()?;
        command_buffer.set_label("zeros");
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

    fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
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

    /// The critical allocator algorithm
    fn allocate_buffer(
        &self,
        size: NSUInteger,
        option: MTLResourceOptions,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
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
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            descriptor.set_output_url(path);
        } else {
            let path = std::env::current_dir()?.join(path);
            descriptor.set_output_url(path);
        }

        capture
            .start_capture(&descriptor)
            .map_err(MetalError::from)?;
        Ok(())
    }
}

fn buf_size(size: NSUInteger) -> NSUInteger {
    size.saturating_sub(1).next_power_of_two() as NSUInteger
}
