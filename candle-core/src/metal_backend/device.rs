use crate::{DType, Result};
use candle_metal_kernels::{
    metal::{
        BlitCommandEncoder, Buffer, BufferMap, Commands, ComputeCommandEncoder, ComputePipeline,
        Device, MTLResourceOptions,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};
use std::ffi::CStr;
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, RwLock,
};

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
pub(crate) struct AllocationPolicy {
    /// Maximum number of bytes we allow to be newly allocated since the last
    /// synchronization point before forcing a sync to reclaim temporaries.
    pending_allocation_bytes_limit: usize,
    /// Maximum bytes to keep cached for reuse.
    cache_limit_bytes: usize,
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        const DEFAULT_PENDING: usize = 4 * 1024 * 1024 * 1024; // 4 GiB
        const MIN_PENDING: usize = 512 * 1024 * 1024; // 512 MiB
        const MAX_PENDING: usize = 12 * 1024 * 1024 * 1024; // 12 GiB
        const HW_MEMSIZE_KEY: &CStr = c"hw.memsize";
        const IOGPU_WIRED_LIMIT_MB_KEY: &CStr = c"iogpu.wired_limit_mb";

        fn parse_env_mebibytes(var: &str) -> Option<usize> {
            std::env::var(var)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .and_then(|mb| mb.checked_mul(1024 * 1024))
        }
        fn sysctl_u64(name: &CStr) -> Option<u64> {
            use libc::c_void;
            unsafe {
                let mut value: u64 = 0;
                let mut len = core::mem::size_of::<u64>();
                if libc::sysctlbyname(
                    name.as_ptr(),
                    &mut value as *mut u64 as *mut c_void,
                    &mut len as *mut usize,
                    std::ptr::null_mut(),
                    0,
                ) != 0
                {
                    return None;
                }
                if len == 0 {
                    None
                } else {
                    Some(value)
                }
            }
        }

        fn system_memory_bytes() -> Option<usize> {
            const MEBIBYTE: usize = 1024 * 1024;
            const SYSTEM_RESERVE_FRACTION: usize = 4; // Keep at least 25% for the OS.
            const SYSTEM_RESERVE_MIN: usize = 2 * 1024 * 1024 * 1024; // 2 GiB floor.

            let hw_total = sysctl_u64(HW_MEMSIZE_KEY).and_then(|bytes| {
                if bytes == 0 {
                    None
                } else {
                    Some(bytes as usize)
                }
            })?;

            let reserve = std::cmp::max(hw_total / SYSTEM_RESERVE_FRACTION, SYSTEM_RESERVE_MIN);
            let hw_budget = hw_total.saturating_sub(reserve);
            if hw_budget == 0 {
                return None;
            }

            let wired_limit_bytes = sysctl_u64(IOGPU_WIRED_LIMIT_MB_KEY).and_then(|limit_mb| {
                if limit_mb == 0 {
                    return None;
                }
                (limit_mb as usize).checked_mul(MEBIBYTE)
            });

            if let Some(wired) = wired_limit_bytes {
                Some(std::cmp::min(wired, hw_budget))
            } else {
                Some(hw_budget)
            }
        }

        let pending_limit = parse_env_mebibytes("CANDLE_METAL_PENDING_LIMIT_MB")
            .or_else(|| system_memory_bytes().map(|mem| (mem / 3).clamp(MIN_PENDING, MAX_PENDING)))
            .unwrap_or(DEFAULT_PENDING);

        let cache_limit = parse_env_mebibytes("CANDLE_METAL_CACHE_LIMIT_MB")
            .unwrap_or_else(|| std::cmp::max(pending_limit / 2, 64 * 1024 * 1024));

        crate::metal_backend::device::AllocationPolicy {
            pending_allocation_bytes_limit: pending_limit,
            cache_limit_bytes: cache_limit,
        }
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<RwLock<Commands>>,

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

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Bytes newly allocated since the last GPU synchronization point. This is
    /// compared against `allocation_policy.pending_allocation_bytes_limit` to
    /// decide when to force a sync and reclaim temporaries.
    pub(crate) pending_allocation_bytes: Arc<AtomicUsize>,
    /// Allocation thresholds and cache budget.
    pub(crate) allocation_policy: AllocationPolicy,
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());
//| MTLResourceOptions::HazardTrackingModeUntracked.bits(),
//);

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
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        ug_metal::code_gen::gen(&mut buf, func_name, &kernel)?;
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
        self.trim_buffer_cache_to(self.allocation_policy.cache_limit_bytes)
    }

    fn trim_buffer_cache_to(&self, limit: usize) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        let mut cached_bytes = 0usize;
        for (size, subbuffers) in buffers.iter() {
            for buffer in subbuffers.iter() {
                if Arc::strong_count(buffer) == 1 {
                    cached_bytes += *size;
                }
            }
        }
        if cached_bytes <= limit {
            return Ok(());
        }

        let mut bytes_to_drop = cached_bytes - limit;
        let mut empty_keys = Vec::new();
        for (size, subbuffers) in buffers.iter_mut() {
            if bytes_to_drop == 0 {
                break;
            }
            subbuffers.retain(|buffer| {
                if bytes_to_drop == 0 {
                    return true;
                }
                if Arc::strong_count(buffer) == 1 {
                    bytes_to_drop = bytes_to_drop.saturating_sub(*size);
                    false
                } else {
                    true
                }
            });
            if subbuffers.is_empty() {
                empty_keys.push(*size);
            }
        }
        for key in empty_keys {
            buffers.remove(&key);
        }
        Ok(())
    }

    pub fn command_encoder(&self) -> Result<ComputeCommandEncoder> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        let (flush, command_encoder) = commands.command_encoder().map_err(MetalError::from)?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        let (flush, command_encoder) = commands.blit_command_encoder().map_err(MetalError::from)?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.wait_until_completed().map_err(MetalError::from)?;
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Creates a new buffer (not necessarily zeroed).
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        self.allocate_buffer(size)
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
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        blit.end_encoding();
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
        subbuffers.push(new_buffer.clone());
        drop(buffers);
        self.on_new_allocation(size)?;
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

    fn on_new_allocation(&self, size: usize) -> Result<()> {
        let pending = self
            .pending_allocation_bytes
            .fetch_add(size, Ordering::AcqRel)
            .saturating_add(size);
        if pending >= self.allocation_policy.pending_allocation_bytes_limit {
            // Ensure the GPU processed the backlog so buffers can be reused.
            self.wait_until_completed()?;
            self.pending_allocation_bytes.store(0, Ordering::Release);
            // Drop part of the cache to keep the resident set under control.
            let target = self.allocation_policy.cache_limit_bytes / 2;
            self.trim_buffer_cache_to(target)?;
        }
        Ok(())
    }
}

fn buf_size(size: usize) -> usize {
    size.saturating_sub(1).next_power_of_two()
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
