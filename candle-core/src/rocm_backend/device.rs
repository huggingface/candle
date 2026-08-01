use crate::Result;
use candle_rocm_kernels::KernelCache;
use std::sync::{Arc, Mutex, RwLock};

use super::alloc::{RocmAllocator, SendSyncDeviceMemory};
#[cfg(feature = "miopen")]
use super::wrappers::SendSyncMIOpenHandle;
use super::wrappers::{RocmBlas, SendSyncPseudoRng, SendSyncRocblasHandle, SendSyncStream};
use super::{rocm_error, RocmError, WrapErr};
use rocm_rs::hip::Device as HipDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct RocmDevice {
    pub(super) id: DeviceId,
    pub(super) device: Arc<HipDevice>,
    pub(crate) stream: Arc<SendSyncStream>,
    pub(crate) allocator: Arc<RocmAllocator>,
    rocrand: Arc<Mutex<SendSyncPseudoRng>>,
    pub(super) seed_value: Arc<RwLock<u64>>,
    pub(crate) blas: Arc<SendSyncRocblasHandle>,
    #[cfg(feature = "miopen")]
    pub(crate) miopen: Arc<SendSyncMIOpenHandle>,
    /// `KernelCache` already guards its own maps, so there is no outer lock:
    /// the previous `Mutex<KernelCache>` serialised every launch on the device,
    /// FFI call included, for a lookup that was already synchronised.
    kernel_manager: Arc<KernelCache>,
    /// Compute units, read once at init and used to size elementwise grids.
    multiprocessor_count: u32,
    /// Device copies of `[dims, strides…]` vectors; see
    /// [`super::params_from_vec`].
    param_cache: Arc<super::ParamCache>,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({:?})", self.id)
    }
}

impl RocmDevice {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = HipDevice::new(device_id as i32)?;
        device.set_current()?;
        // `HipDevice::get_stream` is `hipStreamCreate` with flags 0, i.e. a
        // *blocking* stream that legacy-synchronises with the null stream. The
        // backend no longer leans on that — every operation it issues is on
        // this stream, which is what makes the caching allocator's reuse sound
        // (see `alloc`) — but anything added later that touches the null stream
        // would be silently ordered by it, and would stop being ordered if the
        // stream ever became `hipStreamNonBlocking`.
        let stream = Arc::new(SendSyncStream(device.get_stream()?));
        let allocator = Arc::new(RocmAllocator::new(stream.clone()));

        let mut rocrand = SendSyncPseudoRng::new(rocm_rs::rocrand::rng_type::PSEUDO_DEFAULT)
            .map_err(|e| rocm_error(format!("Failed to create rocrand generator: {}", e)))?;
        // Put the generator on *this* device's stream rather than leaving it
        // on the null stream: it writes into allocator memory, and the
        // allocator recycles a block on the assumption that all work touching
        // it is ordered by the one stream.
        {
            use rocm_rs::rocrand::Generator;
            // SAFETY: `stream` outlives the generator — both are owned by this
            // device, and `rocrand` is dropped first.
            unsafe { rocrand.set_stream(rocm_rs::hip::stream_to_rocrand(&stream)) }
                .map_err(|e| rocm_error(format!("Failed to set rocrand stream: {}", e)))?;
        }
        let seed = 299792458u64;
        rocrand
            .set_seed(seed)
            .map_err(|e| rocm_error(format!("Failed to set rocrand seed: {}", e)))?;

        let blas = SendSyncRocblasHandle::new().map_err(|e| RocmError::Rocblas(e.to_string()))?;
        blas.set_stream(&stream)
            .map_err(|e| RocmError::Rocblas(e.to_string()))?;

        #[cfg(feature = "miopen")]
        let miopen =
            SendSyncMIOpenHandle::new(&stream).map_err(|e| RocmError::MIOpen(e.to_string()))?;

        // Keyed on this device's own architecture: a second GPU of a different
        // generation must not reuse device 0's code objects.
        let kernel_manager = Arc::new(
            KernelCache::new(device_id)
                .map_err(|e| rocm_error(format!("Failed to create kernel cache: {}", e)))?,
        );

        Ok(Self {
            id: DeviceId::new(),
            multiprocessor_count: multiprocessor_count(device_id as i32)?,
            device: Arc::new(device),
            stream,
            allocator,
            rocrand: Arc::new(Mutex::new(rocrand)),
            seed_value: Arc::new(RwLock::new(seed)),
            blas: Arc::new(blas),
            #[cfg(feature = "miopen")]
            miopen: Arc::new(miopen),
            kernel_manager,
            param_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        })
    }

    /// A device with a stream of its own, the counterpart of
    /// `CudaDevice::new_with_stream`.
    ///
    /// On CUDA the two constructors genuinely differ: `new` adopts the context's
    /// *per-thread default* stream, which the rest of the process shares, and
    /// `new_with_stream` creates a private one. Here there is nothing to choose
    /// between, because [`Self::new`] already creates a private stream —
    /// `HipDevice::get_stream` is `hipStreamCreate`, not a handle on the null
    /// stream — so every ROCm device has had one all along.
    ///
    /// It is deliberately *not* a way to hand the device a stream you own, and
    /// two devices never share one. The allocator recycles a freed block into
    /// the next allocation of that size with no event standing between the two
    /// uses, and what makes that sound is that all of the work touching the
    /// block is queued on a single stream; see [`super::alloc`].
    pub fn new_with_stream(device_id: usize) -> Result<Self> {
        Self::new(device_id)
    }

    /// Whether buffers record HIP events to order their uses across streams.
    /// Always `false` here.
    ///
    /// The CUDA answer varies because cudarc gives every `CudaSlice` a read and
    /// a write event, waits on them whenever the slice is touched from a stream
    /// other than its own, and waits on both again before freeing it. That is a
    /// correctness mechanism, not a diagnostic — it is what lets a context hold
    /// more than one stream — and `disable_event_tracking` is the opt-out for
    /// callers who would rather synchronise by hand.
    ///
    /// This backend has no such mechanism and needs none: it orders everything
    /// through the one stream per device described in [`super::alloc`], so the
    /// state CUDA callers reach for by disabling event tracking is the only
    /// state ROCm has.
    pub fn is_event_tracking(&self) -> bool {
        false
    }

    /// The counterpart of `CudaDevice::disable_event_tracking`. A no-op.
    ///
    /// Its postcondition — that no buffer allocated afterwards tracks its uses
    /// with events — already holds before the call, as
    /// [`Self::is_event_tracking`] reports. It exists so that a call site tuned
    /// for CUDA compiles and means the same thing when pointed at ROCm.
    ///
    /// # Safety
    ///
    /// Nothing to uphold today. It stays `unsafe` to match the CUDA signature,
    /// and so that a future ROCm implementation of event tracking could start
    /// relying on the caller's own synchronisation without a breaking change.
    pub unsafe fn disable_event_tracking(&self) {}

    /// This device's rocBLAS handle, the counterpart of
    /// `CudaDevice::cublas_handle`.
    ///
    /// The returned [`RocmBlas`] carries the stream the handle is bound to and
    /// offers no way to rebind it; see its docs for why that matters.
    pub fn rocblas_handle(&self) -> RocmBlas {
        RocmBlas::new(self.blas.clone(), self.stream.clone())
    }

    /// Which MMQ tile geometry this device's `quantized.cu` was compiled with;
    /// see `quantized::rocm::mmq`.
    pub(crate) fn mmq_tiles(&self) -> candle_rocm_kernels::MmqTiles {
        self.kernel_manager.mmq_tiles()
    }

    pub(crate) fn param_cache(&self) -> &super::ParamCache {
        &self.param_cache
    }

    /// Compute units on this device.
    ///
    /// Used to size the grid of the grid-stride elementwise kernels; see
    /// [`super::launch_config`].
    pub(crate) fn multiprocessor_count(&self) -> u32 {
        self.multiprocessor_count
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Makes this device current for the calling thread.
    ///
    /// HIP's current device is thread-local and `hipMalloc`/`hipModuleLaunchKernel`
    /// resolve it from that TLS slot — `rocm-rs`' `HipDevice` carries no device
    /// handle into either. So a second `RocmDevice::new(1)` leaves device 1
    /// current for the constructing thread, and a `RocmStorage` used from a
    /// worker thread would allocate on that thread's default device. Every entry
    /// point that allocates, copies or launches re-binds first; `hipSetDevice` on
    /// the already-current device is a TLS store plus a bounds check.
    pub(crate) fn bind(&self) -> Result<()> {
        self.device.set_current()?;
        Ok(())
    }

    pub fn alloc<T>(&self, len: usize) -> Result<SendSyncDeviceMemory<T>> {
        self.bind()?;
        SendSyncDeviceMemory::new(&self.allocator, len)
            .map_err(|e| rocm_error(format!("Failed to allocate ROCm memory: {}", e)))
    }

    pub fn alloc_zeros<T: Default + Clone>(&self, len: usize) -> Result<SendSyncDeviceMemory<T>> {
        self.bind()?;
        let mut mem = SendSyncDeviceMemory::new(&self.allocator, len)
            .map_err(|e| rocm_error(format!("Failed to allocate ROCm memory: {}", e)))?;
        mem.memset(0)
            .map_err(|e| rocm_error(format!("Failed to memset: {}", e)))?;
        Ok(mem)
    }

    pub fn clone_htod<T: Clone>(&self, src: &[T]) -> Result<SendSyncDeviceMemory<T>> {
        self.bind()?;
        let count = src.len();
        let mut dst = SendSyncDeviceMemory::new(&self.allocator, count)
            .map_err(|e| rocm_error(format!("Failed to allocate ROCm memory: {}", e)))?;
        dst.copy_from_host(src)
            .map_err(|e| rocm_error(format!("Failed to copy host to device: {}", e)))?;
        Ok(dst)
    }

    pub fn clone_dtoh<T: Default + Clone>(&self, src: &SendSyncDeviceMemory<T>) -> Result<Vec<T>> {
        self.bind()?;
        let count = src.count();
        // The `vec![T::default(); count]` looks like a wasted memset before the
        // copy overwrites every byte, and `Vec::with_capacity` + `set_len` was
        // tried. It measured *no* faster on a 256 MiB readback (80.9 ms against
        // 80.3 ms, inside the noise): `vec![0; n]` of a zeroable type is a
        // `calloc`, so the pages arrive lazily zeroed from the kernel and are
        // faulted in by the copy either way. Not worth the `unsafe`.
        let mut dst: Vec<T> = vec![T::default(); count];
        src.copy_to_host(&mut dst)
            .map_err(|e| rocm_error(format!("Failed to copy device to host: {}", e)))?;
        Ok(dst)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| rocm_error(format!("Synchronize failed: {}", e)))
    }

    #[cfg(feature = "miopen")]
    pub(crate) fn miopen(&self) -> &SendSyncMIOpenHandle {
        &self.miopen
    }

    /// Get a reference to the underlying HIP stream.
    /// This is public so that candle-nn and other crates can launch custom kernels.
    pub fn stream(&self) -> &rocm_rs::hip::Stream {
        &self.stream.0
    }

    /// Locks the rocrand generator.
    ///
    /// The lock is deliberately not unwrapped: a failure inside any `rand_*`
    /// call would poison the mutex and turn every later call into a panic.
    pub(super) fn rocrand(&self) -> Result<std::sync::MutexGuard<'_, SendSyncPseudoRng>> {
        self.rocrand
            .lock()
            .map_err(|_| rocm_error("Failed to lock rocrand generator".to_string()))
    }

    /// Get or load a kernel function from the cache.
    /// This is public so that candle-nn and other crates can launch custom kernels.
    ///
    /// Nothing is held across the returned `Function`: the cache's own locks are
    /// released here, so a launch by one thread no longer blocks every other
    /// thread's lookup.
    pub fn get_or_load_func(
        &self,
        kernel_name: &str,
        module: &candle_rocm_kernels::Module,
    ) -> crate::Result<rocm_rs::hip::Function> {
        self.bind()?;
        self.kernel_manager.function(module, kernel_name).w()
    }

    /// [`Self::get_or_load_func`] for a kernel candle does not ship.
    ///
    /// The counterpart of `CudaDevice::get_or_load_custom_func`, and the reason
    /// it takes a source rather than a compiled object: the CUDA backend hands
    /// cudarc a PTX string, while here `source` is CUDA-syntax HIP that
    /// [`candle_rocm_kernels::KernelCache`] runs through `hipcc` on first use
    /// and caches on disk, keyed on the source text among other things. So an
    /// edited kernel recompiles and an unchanged one is free from the second
    /// process onwards.
    ///
    /// The source is compiled exactly as candle's own modules are — the HIP
    /// shim is force-included and `cuda_utils.cuh`, `compatibility.cuh` and
    /// `binary_op_macros.cuh` are on the include path — so a downstream kernel
    /// can be written in the same dialect as `candle-kernels/src/*.cu`.
    ///
    /// `module_name` names the on-disk cache entry and should be unique across
    /// the custom modules of a process; it cannot collide with a built-in. It
    /// is not the whole key though — the source is part of it in memory as well
    /// as on disk, so reusing a name for a revised source compiles and loads the
    /// revision rather than handing back the module it replaces.
    ///
    /// That costs a SHA-256 pass over `source` per call, so a launch loop should
    /// keep the returned function rather than ask for it again: the handle
    /// borrows nothing and modules are never unloaded, which is also why every
    /// distinct source stays resident for the life of the process.
    pub fn get_or_load_custom_func(
        &self,
        kernel_name: &str,
        module_name: &str,
        source: &str,
    ) -> crate::Result<rocm_rs::hip::Function> {
        self.bind()?;
        self.kernel_manager
            .custom_function(module_name, source, kernel_name)
            .w()
    }

    /// Compile a `ug` micro-kernel for this device, the counterpart of
    /// `CudaDevice::compile`.
    ///
    /// CUDA hands the generated source to nvrtc; here it goes through the same
    /// `hipcc` path as every other kernel candle compiles, so the result is
    /// disk-cached and a repeated call with an unchanged kernel is free from the
    /// second process onwards. The generated HIP carries no `#include` of its
    /// own — it relies on the shim [`Self::get_or_load_custom_func`] force
    /// includes, exactly as the sources shared with the CUDA backend do.
    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<rocm_rs::hip::Function> {
        let mut buf = vec![];
        candle_ug::rocm::code_gen::gen(&mut buf, func_name, &kernel)?;
        let source = String::from_utf8(buf)?;
        // The source is part of the cache key, so a distinct kernel compiled
        // under a name already seen still recompiles rather than aliasing.
        self.get_or_load_custom_func(func_name, &format!("candle_ug_{func_name}"), &source)
    }
}

/// Compute units on device `ordinal`.
///
/// Read once at device init: `hipDeviceGetAttribute` is a driver call and the
/// answer is fixed for the life of the process.
fn multiprocessor_count(ordinal: i32) -> Result<u32> {
    use rocm_rs::hip::bindings;
    let mut count: std::os::raw::c_int = 0;
    // SAFETY: `count` is only read once the status has been checked.
    let status = unsafe {
        bindings::hipDeviceGetAttribute(
            &mut count,
            bindings::hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount,
            ordinal,
        )
    };
    if status != bindings::hipError_t_hipSuccess {
        return Err(rocm_error(format!(
            "hipDeviceGetAttribute(MultiprocessorCount) failed for device {ordinal} \
             with error {status}"
        )));
    }
    Ok(count.max(1) as u32)
}
