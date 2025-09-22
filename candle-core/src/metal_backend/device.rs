use crate::{BackendDevice, DType, Result, TryConvertStorage};
use candle_metal_kernels::{
    metal::{
        Buffer, BufferMap, CommandBuffer, Commands, ComputePipeline, Device, MTLResourceOptions,
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
}

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

impl AsRef<MetalDevice> for MetalDevice {
    fn as_ref(&self) -> &MetalDevice {
        self
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

    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let mut commands = self.commands.write().map_err(MetalError::from)?;
        let (flushed, command_buffer) = commands.command_buffer().map_err(MetalError::from)?;
        if flushed {
            self.drop_unused_buffers()?
        }
        Ok(command_buffer.clone())
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let mut commands = self.commands.write().map_err(MetalError::from)?;
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
        let size = element_count * dtype.size_in_bytes();
        self.allocate_buffer(size, MTLResourceOptions::StorageModePrivate, name)
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer can be read on the CPU but will require manual
    /// synchronization when the CPU memory is modified
    /// Used as a bridge to gather data back from the GPU
    pub fn new_buffer_managed(&self, size: usize) -> Result<Arc<Buffer>> {
        self.allocate_buffer(size, MTLResourceOptions::StorageModeManaged, "managed")
    }

    /// Creates a new buffer from data.
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(
                data.as_ptr().cast(),
                size,
                MTLResourceOptions::StorageModeManaged,
            )
            .map_err(MetalError::from)?;
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
            size_in_bytes,
            MTLResourceOptions::StorageModePrivate,
            "allocate_zeros",
        )?;
        let command_buffer = self.command_buffer()?;
        command_buffer.set_label("zeros");
        let blit = command_buffer.blit_command_encoder();
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        blit.end_encoding();
        Ok(buffer)
    }

    /// The critical allocator algorithm
    fn allocate_buffer(
        &self,
        size: usize,
        option: MTLResourceOptions,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, option, &buffers) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }

        let size = buf_size(size);
        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(size, option)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
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

impl BackendDevice<MetalStorage> for MetalDevice {
    const SUPPORTS_BF16: bool = true;

    fn new(ordinal: usize) -> Result<Self> {
        let device = Device::all().swap_remove(ordinal);
        let command_queue = device.new_command_queue().map_err(MetalError::from)?;
        let kernels = Arc::new(Kernels::new());
        let seed = Arc::new(Mutex::new(
            device
                .new_buffer_with_data(
                    [299792458u64].as_ptr() as *const c_void,
                    4,
                    MTLResourceOptions::StorageModeManaged,
                )
                .map_err(MetalError::from)?,
        ));
        let commands = Commands::new(command_queue).map_err(MetalError::from)?;
        Ok(Self {
            id: DeviceId::new(),
            device,
            commands: Arc::new(RwLock::new(commands)),
            buffers: Arc::new(RwLock::new(HashMap::new())),
            kernels,
            seed,
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal {
            gpu_id: self.registry_id() as usize,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn is_cpu(&self) -> bool {
        false
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let buffer = self.new_buffer(shape.elem_count(), dtype, "alloc-uninit")?;
        Ok(MetalStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let size = shape.elem_count() * dtype.size_in_bytes();
        let buffer = self.allocate_zeros(size)?;
        Ok(MetalStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<MetalStorage> {
        let (count, buffer) = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::U32(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::I64(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::BF16(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::F16(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::F32(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::F64(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorageRef::F8E4M3(_) => crate::bail!("Metal device does not yet support F8E4M3."),
        };
        Ok(MetalStorage::new(buffer?, self.clone(), count, T::DTYPE))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<MetalStorage> {
        let (count, buffer) = match storage {
            CpuStorage::U8(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::U32(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::I64(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::BF16(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F16(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F32(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F64(storage) => (storage.len(), self.new_buffer_with_data(storage)),
            CpuStorage::F8E4M3(_) => crate::bail!("Metal device does not yet support F8E4M3."),
        };
        Ok(MetalStorage::new(
            buffer?,
            self.clone(),
            count,
            storage.dtype(),
        ))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<MetalStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn storage<A: crate::NdArray>(&self, array: A) -> Result<MetalStorage> {
        let storage = array.to_cpu_storage();
        let storage = self.storage_from_cpu_storage_owned(storage)?;
        Ok(storage)
    }

    fn storage_owned<S: crate::WithDType>(&self, data: Vec<S>) -> Result<MetalStorage> {
        let storage = S::to_cpu_storage_owned(data);
        let storage = self.storage_from_cpu_storage_owned(storage)?;
        Ok(storage)
    }

    fn rand_uniform<T: crate::FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        min: T,
        max: T,
    ) -> Result<MetalStorage> {
        let name = match dtype {
            DType::F64 => "rand_uniform_f64",
            DType::F32 => "rand_uniform_f32",
            DType::F16 => "rand_uniform_f16",
            DType::BF16 => "rand_uniform_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self.new_buffer(shape.elem_count(), dtype, "rand_uniform")?;
        let command_buffer = self.command_buffer()?;
        candle_metal_kernels::call_random_uniform(
            &self.device,
            &command_buffer,
            &self.kernels,
            name,
            min.to_f64() as f32,
            max.to_f64() as f32,
            shape.elem_count(),
            &self.seed.lock().unwrap(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(MetalStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn rand_normal<T: crate::FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: T,
        stddev: T,
    ) -> Result<MetalStorage> {
        let name = match dtype {
            DType::F64 => "rand_uniform_f64",
            DType::F32 => "rand_normal_f32",
            DType::F16 => "rand_normal_f16",
            DType::BF16 => "rand_normal_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self.new_buffer(shape.elem_count(), dtype, "rand_normal")?;
        let command_buffer = self.command_buffer()?;
        candle_metal_kernels::call_random_normal(
            &self.device,
            &command_buffer,
            &self.kernels,
            name,
            mean.to_f64() as f32,
            stddev.to_f64() as f32,
            shape.elem_count(),
            &self.seed.lock().unwrap(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(MetalStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let seed_buffer = self.seed.try_lock().map_err(MetalError::from)?;
        let contents = seed_buffer.data();
        unsafe {
            std::ptr::copy([seed].as_ptr(), contents as *mut u64, 1);
        }
        seed_buffer.did_modify_range(NSRange::new(0, 8));

        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.wait_until_completed()
    }
}

fn buf_size(size: usize) -> usize {
    size.saturating_sub(1).next_power_of_two()
}

fn find_available_buffer(
    size: usize,
    option: MTLResourceOptions,
    buffers: &BufferMap,
) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
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
