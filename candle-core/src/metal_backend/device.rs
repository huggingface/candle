use crate::{DType, Result};

#[cfg(feature = "ug")]
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{
    metal::{
        BlitCommandsGuard, Buffer, BufferMap, Commands, CommandsGuard, Device, MTLResourceOptions,
        ResidencySet,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use super::MetalError;

/// Unique identifier for metal devices.
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

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<Commands>,

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
    pub(crate) buffers: Arc<RwLock<BufferMap>>,

    /// Same as `buffers` but uses `PRIVATE_RESOURCE_OPTIONS` (StorageModePrivate on macOS).
    /// Intermediate compute buffers don't need CPU access so Private avoids coherency overhead.
    pub(crate) private_buffers: Arc<RwLock<BufferMap>>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
    /// Residency set registered on the command queue.
    pub(crate) residency_set: Arc<ResidencySet>,
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);
// Resource options used for `new_private_buffer`. This uses `private` where supported.
#[cfg(target_os = "ios")]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = RESOURCE_OPTIONS;
#[cfg(not(target_os = "ios"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModePrivate.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    #[cfg(all(feature = "ug", not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        candle_ug::metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        for subbuffers in buffers.values_mut() {
            subbuffers.retain(|s| {
                if Arc::strong_count(s) == 1 {
                    self.residency_set.remove(s);
                    false
                } else {
                    true
                }
            });
        }
        let mut private_buffers = self.private_buffers.write().map_err(MetalError::from)?;
        for subbuffers in private_buffers.values_mut() {
            subbuffers.retain(|s| {
                if Arc::strong_count(s) == 1 {
                    self.residency_set.remove(s);
                    false
                } else {
                    true
                }
            });
        }
        Ok(())
    }

    pub fn command_encoder<'a>(&'a self) -> Result<CommandsGuard<'a>> {
        let command_encoder = self.commands.command_encoder().map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>> {
        let command_encoder = self
            .commands
            .blit_command_encoder()
            .map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        self.commands
            .wait_until_completed()
            .map_err(MetalError::from)?;

        self.drop_unused_buffers()?;
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Creates a new buffer (not necessarily zeroed).
    ///
    /// Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
    /// Falls back to StorageModeShared on iOS where Private is not always available.
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        let mut buffers = self.private_buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers) {
            return Ok(b.clone());
        }
        let size = buf_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        self.residency_set.insert(&new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Creates a new private buffer (not necessarily zeroed).
    ///
    /// This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
    pub fn new_private_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let buffer = Arc::new(buffer);
        self.residency_set.insert(&buffer);
        Ok(buffer)
    }

    /// Creates a new buffer from data.
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;

        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        self.residency_set.insert(&new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let mut blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        /*
        // Alternative impl
        if size_in_bytes > 0 {
            let encoder = self.command_encoder()?;
            call_const_fill(
                &self.device,
                &encoder,
                &self.kernels,
                "fill_u8",
                size_in_bytes,
                &buffer,
                0u8,
            )
            .map_err(crate::Error::wrap)?;
        }
        */
        Ok(buffer)
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }
        let size = buf_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        self.residency_set.insert(&new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(self.device().as_ref());
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        } else {
            let path = std::env::current_dir()?.join(path);
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        }

        capture
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::from(e.to_string()))?;
        Ok(())
    }
}

fn buf_size(size: usize) -> usize {
    size.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buf_size_exact_powers_of_two() {
        assert_eq!(buf_size(1), 1);
        assert_eq!(buf_size(2), 2);
        assert_eq!(buf_size(4), 4);
        assert_eq!(buf_size(8), 8);
        assert_eq!(buf_size(16), 16);
        assert_eq!(buf_size(1024), 1024);
    }

    #[test]
    fn test_buf_size_rounds_up() {
        assert_eq!(buf_size(3), 4);
        assert_eq!(buf_size(5), 8);
        assert_eq!(buf_size(6), 8);
        assert_eq!(buf_size(7), 8);
        assert_eq!(buf_size(9), 16);
        assert_eq!(buf_size(1000), 1024);
        assert_eq!(buf_size(1025), 2048);
    }

    #[test]
    fn test_buf_size_bf16_f16_scalar() {
        // BF16 and F16 are 2 bytes per element. A scalar tensor requests
        // a 2-byte buffer. This must not be rounded down to 1.
        assert_eq!(buf_size(2), 2);
    }
}

fn find_available_buffer(size: usize, buffers: &BufferMap) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
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
