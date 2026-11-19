//! Location in memory that contains data.
//!
//! A Vulkan buffer is very similar to a buffer that you would use in programming languages in
//! general, in the sense that it is a location in memory that contains data. The difference
//! between a Vulkan buffer and a regular buffer is that the content of a Vulkan buffer is
//! accessible from the GPU.
//!
//! Vulkano does not perform any specific marshalling of buffer data. The representation of the
//! buffer in memory is identical between the CPU and GPU. Because the Rust compiler is allowed to
//! reorder struct fields at will by default when using `#[repr(Rust)]`, it is advised to mark each
//! struct requiring input assembly as `#[repr(C)]`. This forces Rust to follow the standard C
//! procedure. Each element is laid out in memory in the order of declaration and aligned to a
//! multiple of their alignment.
//!
//! # Multiple levels of abstraction
//!
//! - The low-level implementation of a buffer is [`RawBuffer`], which corresponds directly to a
//!   `VkBuffer`, and as such doesn't hold onto any memory.
//! - [`Buffer`] is a `RawBuffer` with memory bound to it, and with state tracking.
//! - [`Subbuffer`] is what you will use most of the time, as it is what all the APIs expect. It is
//!   a reference to a portion of a `Buffer`. `Subbuffer` also has a type parameter, which is a
//!   hint for how the data in the portion of the buffer is going to be interpreted.
//!
//! # `Subbuffer` allocation
//!
//! There are two ways to get a `Subbuffer`:
//!
//! - By using the functions on `Buffer`, which create a new buffer and memory allocation each
//!   time, and give you a `Subbuffer` that has an entire `Buffer` dedicated to it.
//! - By using the [`SubbufferAllocator`], which creates `Subbuffer`s by suballocating existing
//!   `Buffer`s such that the `Buffer`s can keep being reused.
//!
//! Which of these you should choose depends on the use case. For example, if you need to upload
//! data to the device each frame, then you should use `SubbufferAllocator`. Same goes for if you
//! need to download data very frequently, or if you need to allocate a lot of intermediary buffers
//! that are only accessed by the device. On the other hand, if you need to upload some data just
//! once, or you can keep reusing the same buffer (because its size is unchanging) it's best to
//! use a dedicated `Buffer` for that.
//!
//! # Buffer usage
//!
//! When you create a buffer, you have to specify its *usage*. In other words, you have to
//! specify the way it is going to be used. Trying to use a buffer in a way that wasn't specified
//! when you created it will result in a runtime error.
//!
//! You can use buffers for the following purposes:
//!
//! - Can contain arbitrary data that can be transferred from/to other buffers and images.
//! - Can be read and modified from a shader.
//! - Can be used as a source of vertices and indices.
//! - Can be used as a source of list of models for draw indirect commands.
//!
//! Accessing a buffer from a shader can be done in the following ways:
//!
//! - As a uniform buffer. Uniform buffers are read-only.
//! - As a storage buffer. Storage buffers can be read and written.
//! - As a uniform texel buffer. Contrary to a uniform buffer, the data is interpreted by the GPU
//!   and can be for example normalized.
//! - As a storage texel buffer. Additionally, some data formats can be modified with atomic
//!   operations.
//!
//! Using uniform/storage texel buffers requires creating a *buffer view*. See [the `view` module]
//! for how to create a buffer view.
//!
//! See also [the `shader` module documentation] for information about how buffer contents need to
//! be laid out in accordance with the shader interface.
//!
//! [`SubbufferAllocator`]: allocator::SubbufferAllocator
//! [the `view` module]: view
//! [the `shader` module documentation]: crate::shader

pub use self::{subbuffer::*, sys::*, usage::*};
use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum},
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationType, DeviceLayout, MemoryAllocator,
            MemoryAllocatorError,
        },
        DedicatedAllocation, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        ExternalMemoryProperties, MemoryRequirements, ResourceMemory,
    },
    range_map::RangeMap,
    self_referential::borrow_wrapper_impls,
    sync::{future::AccessError, AccessConflict, CurrentAccess, Sharing},
    DeviceAddress, DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError,
    Version, VulkanError, VulkanObject,
};
use ash::vk;
use parking_lot::{Mutex, MutexGuard};
use std::{
    error::Error,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::NonZero,
    ops::Range,
    sync::Arc,
};

pub mod allocator;
pub mod subbuffer;
pub mod sys;
mod usage;
pub mod view;

/// A storage for raw bytes.
///
/// Unlike [`RawBuffer`], a `Buffer` has memory backing it, and can be used normally.
///
/// See [the module-level documentation] for more information about buffers.
///
/// # Examples
///
/// Sometimes, you need a buffer that is rarely accessed by the host. To get the best performance
/// in this case, one should use a buffer in device-local memory, that is inaccessible from the
/// host. As such, to initialize or otherwise access such a buffer, we need a *staging buffer*.
///
/// The following example outlines the general strategy one may take when initializing a
/// device-local buffer.
///
/// ```
/// use vulkano::{
///     buffer::{BufferUsage, Buffer, BufferCreateInfo},
///     command_buffer::{
///         AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
///         PrimaryCommandBufferAbstract,
///     },
///     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
///     sync::GpuFuture,
///     DeviceSize,
/// };
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let command_buffer_allocator: std::sync::Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator> = return;
/// #
/// // Simple iterator to construct test data.
/// let data = (0..10_000).map(|i| i as f32);
///
/// // Create a host-accessible buffer initialized with the data.
/// let temporary_accessible_buffer = Buffer::from_iter(
///     &memory_allocator,
///     &BufferCreateInfo {
///         // Specify that this buffer will be used as a transfer source.
///         usage: BufferUsage::TRANSFER_SRC,
///         ..Default::default()
///     },
///     &AllocationCreateInfo {
///         // Specify use for upload to the device.
///         memory_type_filter: MemoryTypeFilter::PREFER_HOST
///             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
///         ..Default::default()
///     },
///     data,
/// )
/// .unwrap();
///
/// // Create a buffer in device-local memory with enough space for a slice of `10_000` floats.
/// let device_local_buffer = Buffer::new_slice::<f32>(
///     &memory_allocator,
///     &BufferCreateInfo {
///         // Specify use as a storage buffer and transfer destination.
///         usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
///         ..Default::default()
///     },
///     &AllocationCreateInfo {
///         // Specify use by the device only.
///         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
///         ..Default::default()
///     },
///     10_000 as DeviceSize,
/// )
/// .unwrap();
///
/// // Create a one-time command to copy between the buffers.
/// let mut cbb = AutoCommandBufferBuilder::primary(
///     command_buffer_allocator.clone(),
///     queue.queue_family_index(),
///     CommandBufferUsage::OneTimeSubmit,
/// )
/// .unwrap();
/// cbb.copy_buffer(CopyBufferInfo::buffers(
///     temporary_accessible_buffer,
///     device_local_buffer.clone(),
/// ))
/// .unwrap();
/// let cb = cbb.build().unwrap();
///
/// // Execute the copy command and wait for completion before proceeding.
/// cb.execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap()
///     .wait(None /* timeout */)
///     .unwrap()
/// ```
///
/// [the module-level documentation]: self
#[derive(Debug)]
pub struct Buffer {
    inner: RawBuffer,
    memory: BufferMemory,
    state: Mutex<BufferState>,
}

/// The type of backing memory that a buffer can have.
#[derive(Debug)]
#[non_exhaustive]
pub enum BufferMemory {
    /// The buffer is backed by normal memory, bound with [`bind_memory`].
    ///
    /// [`bind_memory`]: RawBuffer::bind_memory
    Normal(ResourceMemory),

    /// The buffer is backed by sparse memory, bound with [`bind_sparse`].
    ///
    /// [`bind_sparse`]: crate::device::QueueGuard::bind_sparse
    Sparse,

    /// The buffer is backed by memory not managed by vulkano.
    External,
}

impl Buffer {
    /// Creates a new `Buffer` and writes `data` in it. Returns a [`Subbuffer`] spanning the whole
    /// buffer.
    ///
    /// <div class="vulkano-alert-warning">
    ///
    /// > Warning
    /// >
    /// > This only works with memory types that are host-visible. If you want to upload data to a
    /// > buffer allocated in device-local memory, you will need to create a staging buffer and
    /// > copy the contents over.
    ///
    /// </div>
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    /// - Panics if the chosen memory type is not host-visible.
    pub fn from_data<T>(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        data: T,
    ) -> Result<Subbuffer<T>, Validated<AllocateBufferError>>
    where
        T: BufferContents,
    {
        let buffer = Buffer::new_sized(allocator, create_info, allocation_info)?;

        {
            let mut write_guard = buffer.write().unwrap();
            *write_guard = data;
        }

        Ok(buffer)
    }

    /// Creates a new `Buffer` and writes all elements of `iter` in it. Returns a [`Subbuffer`]
    /// spanning the whole buffer.
    ///
    /// <div class="vulkano-alert-warning">
    ///
    /// > Warning
    /// >
    /// > This only works with memory types that are host-visible. If you want to upload data to a
    /// > buffer allocated in device-local memory, you will need to create a staging buffer and
    /// > copy the contents over.
    ///
    /// </div>
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    /// - Panics if the chosen memory type is not host-visible.
    /// - Panics if `iter` is empty.
    pub fn from_iter<T, I>(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        iter: I,
    ) -> Result<Subbuffer<[T]>, Validated<AllocateBufferError>>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let buffer = Buffer::new_slice(
            allocator,
            create_info,
            allocation_info,
            iter.len().try_into().unwrap(),
        )?;

        {
            let mut write_guard = buffer.write().unwrap();

            for (o, i) in write_guard.iter_mut().zip(iter) {
                *o = i;
            }
        }

        Ok(buffer)
    }

    /// Creates a new uninitialized `Buffer` for sized data. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    pub fn new_sized<T>(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
    ) -> Result<Subbuffer<T>, Validated<AllocateBufferError>>
    where
        T: BufferContents,
    {
        let layout = T::LAYOUT.unwrap_sized();
        let buffer = Subbuffer::new(Buffer::try_new(
            allocator,
            create_info,
            allocation_info,
            layout,
        )?);

        Ok(unsafe { buffer.reinterpret_unchecked() })
    }

    /// Creates a new uninitialized `Buffer` for a slice. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    /// - Panics if `len` is zero.
    pub fn new_slice<T>(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        len: DeviceSize,
    ) -> Result<Subbuffer<[T]>, Validated<AllocateBufferError>>
    where
        T: BufferContents,
    {
        Buffer::new_unsized(allocator, create_info, allocation_info, len)
    }

    /// Creates a new uninitialized `Buffer` for unsized data. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    /// - Panics if `len` is zero.
    pub fn new_unsized<T>(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        len: DeviceSize,
    ) -> Result<Subbuffer<T>, Validated<AllocateBufferError>>
    where
        T: BufferContents + ?Sized,
    {
        let layout = T::LAYOUT.layout_for_len(len).unwrap();
        let buffer = Subbuffer::new(Buffer::try_new(
            allocator,
            create_info,
            allocation_info,
            layout,
        )?);

        Ok(unsafe { buffer.reinterpret_unchecked() })
    }

    /// Creates a new uninitialized `Buffer` with the given `layout`, panicking on a validation
    /// error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    /// - Panics if `create_info.size` is not zero.
    ///
    /// [`try_new`]: Self::try_new
    #[track_caller]
    pub fn new(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, AllocateBufferError> {
        Self::new_inner(
            allocator.clone().as_dyn(),
            create_info,
            allocation_info,
            layout,
        )
    }

    fn new_inner(
        allocator: Arc<dyn MemoryAllocator>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, AllocateBufferError> {
        match Self::try_new_inner(allocator, create_info, allocation_info, layout) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new uninitialized `Buffer` with the given `layout`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    pub fn try_new(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, Validated<AllocateBufferError>> {
        Self::try_new_inner(
            allocator.clone().as_dyn(),
            create_info,
            allocation_info,
            layout,
        )
    }

    pub(crate) fn try_new_inner(
        allocator: Arc<dyn MemoryAllocator>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, Validated<AllocateBufferError>> {
        assert!(!create_info
            .flags
            .contains(BufferCreateFlags::SPARSE_BINDING));

        assert_eq!(
            create_info.size, 0,
            "`Buffer::new*` functions set the `create_info.size` field themselves, you should not \
             set it yourself"
        );

        let create_info = BufferCreateInfo {
            size: layout.size(),
            ..*create_info
        };

        let raw_buffer = RawBuffer::try_new(allocator.device(), &create_info)
            .map_err(|err| err.map(AllocateBufferError::CreateBuffer))?;
        let mut requirements = *raw_buffer.memory_requirements();
        requirements.layout = requirements.layout.align_to(layout.alignment()).unwrap();

        let allocation = allocator
            .try_allocate(
                &requirements,
                AllocationType::Linear,
                allocation_info,
                Some(DedicatedAllocation::Buffer(&raw_buffer)),
            )
            .map_err(|err| err.map(AllocateBufferError::AllocateMemory))?;
        let allocation = unsafe { ResourceMemory::from_allocation_inner(allocator, allocation) };

        let buffer = raw_buffer
            .try_bind_memory(allocation)
            .map_err(|(err, _, _)| {
                err.map(AllocateBufferError::BindMemory)
                    .map_validation(|err| err.add_context("RawBuffer::bind_memory"))
            })?;

        Ok(Arc::new(buffer))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, AllocateBufferError> {
        unsafe {
            Self::new_unchecked_inner(
                allocator.clone().as_dyn(),
                create_info,
                allocation_info,
                layout,
            )
        }
    }

    unsafe fn new_unchecked_inner(
        allocator: Arc<dyn MemoryAllocator>,
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, AllocateBufferError> {
        assert!(!create_info
            .flags
            .contains(BufferCreateFlags::SPARSE_BINDING));

        assert_eq!(
            create_info.size, 0,
            "`Buffer::new*` functions set the `create_info.size` field themselves, you should not \
             set it yourself"
        );

        let create_info = BufferCreateInfo {
            size: layout.size(),
            ..*create_info
        };

        // SAFETY: Enforced by the caller.
        let raw_buffer = unsafe { RawBuffer::new_unchecked(allocator.device(), &create_info) }
            .map_err(AllocateBufferError::CreateBuffer)?;
        let mut requirements = *raw_buffer.memory_requirements();
        requirements.layout = requirements.layout.align_to(layout.alignment()).unwrap();

        // SAFETY: Enforced by the caller.
        let allocation = unsafe {
            allocator.allocate_unchecked(
                &requirements,
                AllocationType::Linear,
                allocation_info,
                Some(DedicatedAllocation::Buffer(&raw_buffer)),
            )
        }
        .map_err(AllocateBufferError::AllocateMemory)?;
        let allocation = unsafe { ResourceMemory::from_allocation_inner(allocator, allocation) };

        // SAFETY: Enforced by the caller.
        let buffer = unsafe { raw_buffer.bind_memory_unchecked(allocation) }
            .map_err(|(err, _, _)| AllocateBufferError::BindMemory(err))?;

        Ok(Arc::new(buffer))
    }

    fn from_raw(inner: RawBuffer, memory: BufferMemory) -> Self {
        let state = Mutex::new(BufferState::new(inner.size()));

        Buffer {
            inner,
            memory,
            state,
        }
    }

    /// Returns the type of memory that is backing this buffer.
    #[inline]
    pub fn memory(&self) -> &BufferMemory {
        &self.memory
    }

    /// Returns the memory requirements for this buffer.
    #[inline]
    pub fn memory_requirements(&self) -> &MemoryRequirements {
        self.inner.memory_requirements()
    }

    /// Returns the flags the buffer was created with.
    #[inline]
    pub fn flags(&self) -> BufferCreateFlags {
        self.inner.flags()
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.inner.size()
    }

    /// Returns the usage the buffer was created with.
    #[inline]
    pub fn usage(&self) -> BufferUsage {
        self.inner.usage()
    }

    /// Returns the sharing the buffer was created with.
    #[inline]
    pub fn sharing(&self) -> Sharing<'_> {
        self.inner.sharing()
    }

    /// Returns the external memory handle types that are supported with this buffer.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.inner.external_memory_handle_types()
    }

    /// Returns the device address for this buffer, panicking on a validation error.
    ///
    /// This is a shortcut for `try_device_address().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_device_address`] returns a [`ValidationError`].
    ///
    /// [`try_device_address`]: Self::try_device_address
    #[track_caller]
    pub fn device_address(&self) -> NonZero<DeviceAddress> {
        self.try_device_address().unwrap()
    }

    /// Returns the device address for this buffer.
    // TODO: Caching?
    pub fn try_device_address(&self) -> Result<NonZero<DeviceAddress>, Box<ValidationError>> {
        self.validate_device_address()?;

        Ok(unsafe { self.device_address_unchecked() })
    }

    fn validate_device_address(&self) -> Result<(), Box<ValidationError>> {
        let device = self.device();

        if !device.enabled_features().buffer_device_address {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "buffer_device_address",
                )])]),
                vuids: &["VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324"],
                ..Default::default()
            }));
        }

        if !self.usage().intersects(BufferUsage::SHADER_DEVICE_ADDRESS) {
            return Err(Box::new(ValidationError {
                context: "self.usage()".into(),
                problem: "does not contain `BufferUsage::SHADER_DEVICE_ADDRESS`".into(),
                vuids: &["VUID-VkBufferDeviceAddressInfo-buffer-02601"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn device_address_unchecked(&self) -> NonZero<DeviceAddress> {
        let device = self.device();

        let info_vk = vk::BufferDeviceAddressInfo::default().buffer(self.handle());

        let ptr = {
            let fns = device.fns();
            let func = if device.api_version() >= Version::V1_2 {
                fns.v1_2.get_buffer_device_address
            } else if device.enabled_extensions().khr_buffer_device_address {
                fns.khr_buffer_device_address.get_buffer_device_address_khr
            } else {
                fns.ext_buffer_device_address.get_buffer_device_address_ext
            };
            unsafe { func(device.handle(), &info_vk) }
        };

        NonZero::new(ptr).unwrap()
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, BufferState> {
        self.state.lock()
    }
}

unsafe impl VulkanObject for Buffer {
    type Handle = vk::Buffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for Buffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl PartialEq for Buffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// Error that can happen when allocating a new buffer.
#[derive(Clone, Debug)]
pub enum AllocateBufferError {
    CreateBuffer(VulkanError),
    AllocateMemory(MemoryAllocatorError),
    BindMemory(VulkanError),
}

impl Error for AllocateBufferError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CreateBuffer(err) => Some(err),
            Self::AllocateMemory(err) => Some(err),
            Self::BindMemory(err) => Some(err),
        }
    }
}

impl Display for AllocateBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateBuffer(_) => write!(f, "creating the buffer failed"),
            Self::AllocateMemory(_) => write!(f, "allocating memory for the buffer failed"),
            Self::BindMemory(_) => write!(f, "binding memory to the buffer failed"),
        }
    }
}

impl From<AllocateBufferError> for Validated<AllocateBufferError> {
    fn from(err: AllocateBufferError) -> Self {
        Self::Error(err)
    }
}

/// The current state of a buffer.
#[derive(Debug)]
pub(crate) struct BufferState {
    ranges: RangeMap<DeviceSize, BufferRangeState>,
}

impl BufferState {
    fn new(size: DeviceSize) -> Self {
        BufferState {
            ranges: [(
                0..size,
                BufferRangeState {
                    current_access: CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    },
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    pub(crate) fn check_cpu_read(&self, range: Range<DeviceSize>) -> Result<(), AccessConflict> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive => return Err(AccessConflict::HostWrite),
                CurrentAccess::GpuExclusive { .. } => return Err(AccessConflict::DeviceWrite),
                CurrentAccess::Shared { .. } => (),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn cpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => {
                    *cpu_reads += 1;
                }
                _ => unreachable!("Buffer is being written by the CPU or GPU"),
            }
        }
    }

    pub(crate) unsafe fn cpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => *cpu_reads -= 1,
                _ => unreachable!("Buffer was not locked for CPU read"),
            }
        }
    }

    pub(crate) fn check_cpu_write(&self, range: Range<DeviceSize>) -> Result<(), AccessConflict> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive => return Err(AccessConflict::HostWrite),
                CurrentAccess::GpuExclusive { .. } => return Err(AccessConflict::DeviceWrite),
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                CurrentAccess::Shared { cpu_reads, .. } if *cpu_reads > 0 => {
                    return Err(AccessConflict::HostRead);
                }
                CurrentAccess::Shared { .. } => return Err(AccessConflict::DeviceRead),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn cpu_write_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            state.current_access = CurrentAccess::CpuExclusive;
        }
    }

    pub(crate) unsafe fn cpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::CpuExclusive => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    }
                }
                _ => unreachable!("Buffer was not locked for CPU write"),
            }
        }
    }

    pub(crate) fn check_gpu_read(&self, range: Range<DeviceSize>) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared { .. } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. }
                | CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads += 1,
                _ => unreachable!("Buffer is being written by the CPU"),
            }
        }
    }

    pub(crate) unsafe fn gpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. } => *gpu_reads -= 1,
                CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads -= 1,
                _ => unreachable!("Buffer was not locked for GPU read"),
            }
        }
    }

    pub(crate) fn check_gpu_write(&self, range: Range<DeviceSize>) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_write_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes += 1,
                &mut CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads,
                } => {
                    state.current_access = CurrentAccess::GpuExclusive {
                        gpu_reads,
                        gpu_writes: 1,
                    }
                }
                _ => unreachable!("Buffer is being accessed by the CPU"),
            }
        }
    }

    pub(crate) unsafe fn gpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                &mut CurrentAccess::GpuExclusive {
                    gpu_reads,
                    gpu_writes: 1,
                } => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads,
                    }
                }
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes -= 1,
                _ => unreachable!("Buffer was not locked for GPU write"),
            }
        }
    }
}

/// The current state of a specific range of bytes in a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferRangeState {
    current_access: CurrentAccess,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a buffer.
    BufferCreateFlags = BufferCreateFlags(u32);

    /// The buffer will be backed by sparse memory binding (through the [`bind_sparse`] queue
    /// command) instead of regular binding (through [`bind_memory`]).
    ///
    /// The [`sparse_binding`] feature must be enabled on the device.
    ///
    /// [`bind_sparse`]: crate::device::queue::QueueGuard::bind_sparse
    /// [`bind_memory`]: sys::RawBuffer::bind_memory
    /// [`sparse_binding`]: crate::device::DeviceFeatures::sparse_binding
    SPARSE_BINDING = SPARSE_BINDING,

    /// The buffer can be used without being fully resident in memory at the time of use.
    ///
    /// This requires the [`BufferCreateFlags::SPARSE_BINDING`] flag as well.
    ///
    /// The [`sparse_residency_buffer`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_buffer`]: crate::device::DeviceFeatures::sparse_residency_buffer
    SPARSE_RESIDENCY = SPARSE_RESIDENCY,

    /* TODO: enable
    /// The buffer's memory can alias with another buffer or a different part of the same buffer.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_aliased`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_aliased`]: crate::device::DeviceFeatures::sparse_residency_aliased
    SPARSE_ALIASED = SPARSE_ALIASED,*/

    /// The buffer is protected, and can only be used in combination with protected memory and other
    /// protected objects.
    ///
    /// The device API version must be at least 1.1.
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),

    /* TODO: enable
    /// The buffer's device address can be saved and reused on a subsequent run.
    ///
    /// The device API version must be at least 1.2, or either the [`khr_buffer_device_address`] or
    /// [`ext_buffer_device_address`] extension must be enabled on the device.
    DEVICE_ADDRESS_CAPTURE_REPLAY = DEVICE_ADDRESS_CAPTURE_REPLAY {
        api_version: V1_2,
        device_extensions: [khr_buffer_device_address, ext_buffer_device_address],
    },*/
}

/// The buffer configuration to query in [`PhysicalDevice::external_buffer_properties`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalBufferInfo<'a> {
    /// The flags that will be used.
    pub flags: BufferCreateFlags,

    /// The usage that the buffer will have.
    pub usage: BufferUsage,

    /// The external handle type that will be used with the buffer.
    pub handle_type: ExternalMemoryHandleType,

    pub _ne: crate::NonExhaustive<'a>,
}

impl ExternalBufferInfo<'_> {
    /// Returns a default `ExternalBufferInfo` with the provided `handle_type`.
    #[inline]
    pub const fn new(handle_type: ExternalMemoryHandleType) -> Self {
        Self {
            flags: BufferCreateFlags::empty(),
            usage: BufferUsage::empty(),
            handle_type,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn handle_type(handle_type: ExternalMemoryHandleType) -> Self {
        Self::new(handle_type)
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            usage,
            handle_type,
            _ne: _,
        } = self;

        flags
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("flags")
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalBufferInfo-flags-parameter"])
            })?;

        usage
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("usage")
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalBufferInfo-usage-parameter"])
            })?;

        if usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPhysicalDeviceExternalBufferInfo-usage-requiredbitmask"],
                ..Default::default()
            }));
        }

        handle_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("handle_type")
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalBufferInfo-handleType-parameter"])
            })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PhysicalDeviceExternalBufferInfo<'static> {
        let &Self {
            flags,
            usage,
            handle_type,
            _ne: _,
        } = self;

        vk::PhysicalDeviceExternalBufferInfo::default()
            .flags(flags.into())
            .usage(usage.into())
            .handle_type(handle_type.into())
    }

    pub(crate) fn to_owned(&self) -> ExternalBufferInfo<'static> {
        ExternalBufferInfo {
            _ne: crate::NE,
            ..*self
        }
    }
}

borrow_wrapper_impls!(ExternalBufferInfo<'_>, PartialEq, Eq, Hash);

/// The external memory properties supported for buffers with a given configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalBufferProperties {
    /// The properties for external memory.
    pub external_memory_properties: ExternalMemoryProperties,
}

impl ExternalBufferProperties {
    pub(crate) fn to_mut_vk() -> vk::ExternalBufferProperties<'static> {
        vk::ExternalBufferProperties::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::ExternalBufferProperties<'_>) -> Self {
        let vk::ExternalBufferProperties {
            external_memory_properties,
            ..
        } = val_vk;

        Self {
            external_memory_properties: ExternalMemoryProperties::from_vk(
                external_memory_properties,
            ),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// An enumeration of all valid index types.
    IndexType = IndexType(i32);

    /// Indices are 8-bit unsigned integers.
    U8 = UINT8_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_index_type_uint8)]),
    ]),

    /// Indices are 16-bit unsigned integers.
    U16 = UINT16,

    /// Indices are 32-bit unsigned integers.
    U32 = UINT32,
}

impl IndexType {
    /// Returns the size in bytes of indices of this type.
    #[inline]
    pub fn size(self) -> DeviceSize {
        match self {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        }
    }
}

/// A buffer holding index values, which index into buffers holding vertex data.
#[derive(Clone, Debug)]
pub enum IndexBuffer {
    /// An index buffer containing unsigned 8-bit indices.
    ///
    /// The [`index_type_uint8`] feature must be enabled on the device.
    ///
    /// [`index_type_uint8`]: crate::device::DeviceFeatures::index_type_uint8
    U8(Subbuffer<[u8]>),

    /// An index buffer containing unsigned 16-bit indices.
    U16(Subbuffer<[u16]>),

    /// An index buffer containing unsigned 32-bit indices.
    U32(Subbuffer<[u32]>),
}

impl IndexBuffer {
    /// Returns an `IndexType` value corresponding to the type of the buffer.
    #[inline]
    pub fn index_type(&self) -> IndexType {
        match self {
            Self::U8(_) => IndexType::U8,
            Self::U16(_) => IndexType::U16,
            Self::U32(_) => IndexType::U32,
        }
    }

    /// Returns the buffer reinterpreted as a buffer of bytes.
    #[inline]
    pub fn as_bytes(&self) -> &Subbuffer<[u8]> {
        match self {
            IndexBuffer::U8(buffer) => buffer.as_bytes(),
            IndexBuffer::U16(buffer) => buffer.as_bytes(),
            IndexBuffer::U32(buffer) => buffer.as_bytes(),
        }
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> DeviceSize {
        match self {
            IndexBuffer::U8(buffer) => buffer.len(),
            IndexBuffer::U16(buffer) => buffer.len(),
            IndexBuffer::U32(buffer) => buffer.len(),
        }
    }
}

impl From<Subbuffer<[u8]>> for IndexBuffer {
    #[inline]
    fn from(value: Subbuffer<[u8]>) -> Self {
        Self::U8(value)
    }
}

impl From<Subbuffer<[u16]>> for IndexBuffer {
    #[inline]
    fn from(value: Subbuffer<[u16]>) -> Self {
        Self::U16(value)
    }
}

impl From<Subbuffer<[u32]>> for IndexBuffer {
    #[inline]
    fn from(value: Subbuffer<[u32]>) -> Self {
        Self::U32(value)
    }
}

/// This is intended for use by the `BufferContents` derive macro only.
#[doc(hidden)]
pub struct AssertParamIsBufferContents<T: BufferContents + ?Sized>(PhantomData<T>);
