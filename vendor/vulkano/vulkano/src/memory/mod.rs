//! Device memory allocation and memory pools.
//!
//! By default, memory allocation is automatically handled by the vulkano library when you create
//! a buffer or an image. But if you want more control, you have the possibility to customize the
//! memory allocation strategy.
//!
//! # Memory types and heaps
//!
//! A physical device is composed of one or more **memory heaps**. A memory heap is a pool of
//! memory that can be allocated.
//!
//! ```
//! # let physical_device: vulkano::device::physical::PhysicalDevice = return;
//! #
//! // Enumerating memory heaps.
//! for (index, heap) in physical_device
//!     .memory_properties()
//!     .memory_heaps
//!     .iter()
//!     .enumerate()
//! {
//!     println!("Heap #{:?} has a capacity of {:?} bytes", index, heap.size);
//! }
//! ```
//!
//! However you can't allocate directly from a memory heap. A memory heap is shared amongst one or
//! multiple **memory types**, which you can allocate memory from. Each memory type has different
//! characteristics.
//!
//! A memory type may or may not be visible to the host. In other words, it may or may not be
//! directly writable by the CPU. A memory type may or may not be device-local. A device-local
//! memory type has a much quicker access time from the GPU than a non-device-local type. Note
//! that non-device-local memory types are still accessible by the device, they are just slower.
//!
//! ```
//! # let physical_device: vulkano::device::physical::PhysicalDevice = return;
//! #
//! // Enumerating memory types.
//! for ty in physical_device.memory_properties().memory_types.iter() {
//!     println!("Memory type belongs to heap #{:?}", ty.heap_index);
//!     println!("Property flags: {:?}", ty.property_flags);
//! }
//! ```
//!
//! Memory types are order from "best" to "worse". In other words, the implementation prefers that
//! you use the memory types that are earlier in the list. This means that selecting a memory type
//! should always be done by enumerating them and taking the first one that matches our criteria.
//!
//! ## In practice
//!
//! In practice, desktop machines usually have two memory heaps: one that represents the RAM of
//! the CPU, and one that represents the RAM of the GPU. The CPU's RAM is host-accessible but not
//! device-local, while the GPU's RAM is not host-accessible but is device-local.
//!
//! Mobile machines usually have a single memory heap that is "equally local" to both the CPU and
//! the GPU. It is both host-accessible and device-local.
//!
//! # Allocating memory and memory pools
//!
//! Allocating memory can be done by calling `DeviceMemory::allocate()`.
//!
//! Here is an example:
//!
//! ```
//! use vulkano::memory::{DeviceMemory, MemoryAllocateInfo};
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! #
//! // Taking the first memory type for the sake of this example.
//! let memory_type_index = 0;
//!
//! let memory = DeviceMemory::allocate(
//!     &device,
//!     &MemoryAllocateInfo {
//!         allocation_size: 1024,
//!         memory_type_index,
//!         ..Default::default()
//!     },
//! )
//! .expect("failed to allocate memory");
//!
//! // The memory is automatically freed when `memory` is destroyed.
//! ```
//!
//! However allocating and freeing memory is very slow (up to several hundred milliseconds
//! sometimes). Instead you are strongly encouraged to use a memory pool. A memory pool is not
//! a Vulkan concept but a vulkano concept.
//!
//! A memory pool is any object that implements the `MemoryPool` trait. You can implement that
//! trait on your own structure and then use it when you create buffers and images so that they
//! get memory from that pool. By default if you don't specify any pool when creating a buffer or
//! an image, an instance of `StandardMemoryPool` that is shared by the `Device` object is used.

use self::allocator::{
    align_up, AllocationHandle, AllocationType, DeviceLayout, MemoryAlloc, MemoryAllocator,
    Suballocation,
};
pub use self::{alignment::*, device_memory::*};
use crate::{
    buffer::sys::RawBuffer,
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    image::sys::RawImage,
    macros::vulkan_bitflags,
    sync::HostAccessError,
    DeviceSize, Validated, ValidationError, Version, VulkanError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use std::{
    cmp,
    mem::ManuallyDrop,
    num::NonZero,
    ops::{Bound, Range, RangeBounds, RangeTo},
    ptr::NonNull,
    sync::Arc,
};

mod alignment;
pub mod allocator;
mod device_memory;
pub mod sparse;

/// Memory that can be bound to resources.
///
/// Most commonly you will want to obtain this by first using a [memory allocator] and then
/// [constructing this object from its allocation]. Alternatively, if you want to bind a whole
/// block of `DeviceMemory` to a resource, or can't go through an allocator, you can use [the
/// dedicated constructor].
///
/// [memory allocator]: MemoryAllocator
/// [the dedicated constructor]: Self::new_dedicated
#[derive(Debug)]
pub struct ResourceMemory {
    device_memory: ManuallyDrop<DeviceOwnedDebugWrapper<Arc<DeviceMemory>>>,
    offset: DeviceSize,
    size: DeviceSize,
    allocation_type: AllocationType,
    allocation_handle: AllocationHandle,
    suballocation_handle: Option<AllocationHandle>,
    allocator: Option<Arc<dyn MemoryAllocator>>,
}

impl ResourceMemory {
    /// Creates a new `ResourceMemory` that has a whole device memory block dedicated to it. You
    /// may use this when you obtain the memory in a way other than through the use of a memory
    /// allocator, for instance by importing memory.
    ///
    /// This is safe because we take ownership of the device memory, so that there can be no
    /// aliasing resources. On the other hand, the device memory can never be reused: it will be
    /// freed once the returned object is dropped.
    pub fn new_dedicated(device_memory: DeviceMemory) -> Self {
        unsafe { Self::new_dedicated_unchecked(&Arc::new(device_memory)) }
    }

    /// Same as [`new_dedicated`], except that this allows creating aliasing resources.
    ///
    /// # Safety
    ///
    /// - Two resources must not alias each other, and if they do, you must ensure correct
    ///   synchronization yourself.
    ///
    /// [`new_dedicated`]: Self::new_dedicated
    pub unsafe fn new_dedicated_unchecked(device_memory: &Arc<DeviceMemory>) -> Self {
        let size = device_memory.allocation_size();

        unsafe { Self::from_device_memory_unchecked(device_memory, 0, size) }
    }

    /// Creates a new `ResourceMemory` from the given portion of the given device memory block. You
    /// may use this when you need to portion an existing memory block in a specific way. Note that
    /// when you don't have this requirement of placing resources at specific offsets, you should
    /// use a memory allocator instead.
    ///
    /// # Safety
    ///
    /// - Two resources must not alias each other (as returned by
    ///   [`RawBuffer::memory_requirements`] or [`RawImage::memory_requirements`]), and if they do,
    ///   you must ensure correct synchronization yourself.
    ///
    /// # Panics
    ///
    /// - Panics if `offset + size` is greater than `device_memory.allocation_size()`.
    pub unsafe fn from_device_memory_unchecked(
        device_memory: &Arc<DeviceMemory>,
        offset: DeviceSize,
        size: DeviceSize,
    ) -> Self {
        assert!(offset <= device_memory.allocation_size());
        assert!(size <= device_memory.allocation_size() - offset);

        ResourceMemory {
            offset,
            size,
            allocation_type: AllocationType::Unknown,
            allocation_handle: AllocationHandle::null(),
            suballocation_handle: None,
            allocator: None,
            device_memory: ManuallyDrop::new(DeviceOwnedDebugWrapper(device_memory.clone())),
        }
    }

    /// Creates a new `ResourceMemory` from an allocation of a memory allocator.
    ///
    /// Ownership of `allocation` is semantically transferred to this object, and it is deallocated
    /// when the returned object is dropped.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `allocator`.
    /// - `allocation` must never be deallocated.
    #[inline]
    pub unsafe fn from_allocation(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        allocation: MemoryAlloc,
    ) -> Self {
        unsafe { Self::from_allocation_inner(allocator.clone().as_dyn(), allocation) }
    }

    pub(crate) unsafe fn from_allocation_inner(
        allocator: Arc<dyn MemoryAllocator>,
        allocation: MemoryAlloc,
    ) -> Self {
        if let Some(suballocation) = allocation.suballocation {
            ResourceMemory {
                offset: suballocation.offset,
                size: suballocation.size,
                allocation_type: suballocation.allocation_type,
                allocation_handle: allocation.allocation_handle,
                suballocation_handle: Some(suballocation.handle),
                allocator: Some(allocator),
                device_memory: ManuallyDrop::new(DeviceOwnedDebugWrapper(allocation.device_memory)),
            }
        } else {
            ResourceMemory {
                offset: 0,
                size: allocation.device_memory.allocation_size(),
                allocation_type: AllocationType::Unknown,
                allocation_handle: allocation.allocation_handle,
                suballocation_handle: None,
                allocator: Some(allocator),
                device_memory: ManuallyDrop::new(DeviceOwnedDebugWrapper(allocation.device_memory)),
            }
        }
    }

    /// Returns the underlying block of [`DeviceMemory`].
    #[inline]
    pub fn device_memory(&self) -> &Arc<DeviceMemory> {
        &self.device_memory
    }

    /// Returns the offset (in bytes) within the [`DeviceMemory`] block where this `ResourceMemory`
    /// beings.
    ///
    /// If this `ResourceMemory` is not a [suballocation], then this will be `0`.
    ///
    /// [suballocation]: Suballocation
    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    /// Returns the size (in bytes) of the `ResourceMemory`.
    ///
    /// If this `ResourceMemory` is not a [suballocation], then this will be equal to the
    /// [allocation size] of the [`DeviceMemory`] block.
    ///
    /// [suballocation]: Suballocation
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the type of resources that can be bound to this `ResourceMemory`.
    ///
    /// If this `ResourceMemory` is not a [suballocation], then this will be
    /// [`AllocationType::Unknown`].
    ///
    /// [suballocation]: Suballocation
    #[inline]
    pub fn allocation_type(&self) -> AllocationType {
        self.allocation_type
    }

    fn suballocation(&self) -> Option<Suballocation> {
        self.suballocation_handle.map(|handle| Suballocation {
            offset: self.offset,
            size: self.size,
            allocation_type: self.allocation_type,
            handle,
        })
    }

    /// Returns the mapped pointer to a range of the `ResourceMemory`, or returns [`None`] if out
    /// of bounds.
    ///
    /// `range` is specified in bytes relative to the beginning of `self` and must fall within the
    /// range of the memory mapping given to [`DeviceMemory::map`].
    ///
    /// See [`MappingState::slice`] for the safety invariants of the returned pointer.
    #[inline]
    pub fn mapped_slice(
        &self,
        range: impl RangeBounds<DeviceSize>,
    ) -> Option<Result<NonNull<[u8]>, HostAccessError>> {
        let mut range = self::range(range, ..self.size())?;
        range.start += self.offset();
        range.end += self.offset();

        let res = if let Some(state) = self.device_memory().mapping_state() {
            state.slice(range).ok_or(HostAccessError::OutOfMappedRange)
        } else {
            Err(HostAccessError::NotHostMapped)
        };

        Some(res)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn mapped_slice_unchecked(
        &self,
        range: impl RangeBounds<DeviceSize>,
    ) -> Result<NonNull<[u8]>, HostAccessError> {
        let mut range = range_unchecked(range, ..self.size());
        range.start += self.offset();
        range.end += self.offset();

        if let Some(state) = self.device_memory().mapping_state() {
            state.slice(range).ok_or(HostAccessError::OutOfMappedRange)
        } else {
            Err(HostAccessError::NotHostMapped)
        }
    }

    // TODO: Expose (in a better way).
    #[doc(hidden)]
    #[inline]
    pub fn atom_size(&self) -> Option<DeviceAlignment> {
        let memory = self.device_memory();

        (!memory.is_coherent()).then_some(memory.atom_size())
    }

    /// Invalidates the host cache for a range of the `ResourceMemory`, panicking on a validation
    /// error.
    ///
    /// If the device memory is not [host-coherent], you must call this function before the memory
    /// is read by the host, if the device previously wrote to the memory. It has no effect if the
    /// memory is host-coherent.
    ///
    /// This is a shortcut for `try_invalidate_range().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the device that have not been propagated into the host
    ///   cache, then there must not be any references in Rust code to any portion of the specified
    ///   `memory_range`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_invalidate_range`] returns a [`ValidationError`].
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    /// [`try_invalidate_range`]: Self::try_invalidate_range
    #[inline]
    #[track_caller]
    pub unsafe fn invalidate_range(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_invalidate_range(memory_range) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Invalidates the host cache for a range of the `ResourceMemory`.
    ///
    /// If the device memory is not [host-coherent], you must call this function before the memory
    /// is read by the host, if the device previously wrote to the memory. It has no effect if the
    /// memory is host-coherent.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the device that have not been propagated into the host
    ///   cache, then there must not be any references in Rust code to any portion of the specified
    ///   `memory_range`.
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    #[inline]
    pub unsafe fn try_invalidate_range(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_memory_range(memory_range)?;

        unsafe {
            self.device_memory()
                .try_invalidate_range(&self.create_memory_range(memory_range))
        }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn invalidate_range_unchecked(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), VulkanError> {
        unsafe {
            self.device_memory()
                .invalidate_range_unchecked(&self.create_memory_range(memory_range))
        }
    }

    /// Flushes the host cache for a range of the `ResourceMemory`, panicking on a validation
    /// error.
    ///
    /// If the device memory is not [host-coherent], you must call this function after writing to
    /// the memory, if the device is going to read the memory. It has no effect if the memory is
    /// host-coherent.
    ///
    /// This is a shortcut for `try_flush_range().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a device queue, that access any
    ///   portion of the specified `memory_range`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_flush_range`] returns a [`ValidationError`].
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    /// [`try_flush_range`]: Self::try_flush_range
    #[inline]
    #[track_caller]
    pub unsafe fn flush_range(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_flush_range(memory_range) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Flushes the host cache for a range of the `ResourceMemory`.
    ///
    /// If the device memory is not [host-coherent], you must call this function after writing to
    /// the memory, if the device is going to read the memory. It has no effect if the memory is
    /// host-coherent.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a device queue, that access any
    ///   portion of the specified `memory_range`.
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    #[inline]
    pub unsafe fn try_flush_range(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_memory_range(memory_range)?;

        unsafe {
            self.device_memory()
                .try_flush_range(&self.create_memory_range(memory_range))
        }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn flush_range_unchecked(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), VulkanError> {
        unsafe {
            self.device_memory()
                .flush_range_unchecked(&self.create_memory_range(memory_range))
        }
    }

    fn validate_memory_range(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let &MappedMemoryRange {
            offset,
            size,
            _ne: _,
        } = memory_range;

        if !(offset <= self.size() && size.is_none_or(|size| size <= self.size() - offset)) {
            return Err(Box::new(ValidationError {
                context: "memory_range".into(),
                problem: "is not contained within the allocation".into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    fn create_memory_range(
        &self,
        memory_range: &MappedMemoryRange<'_>,
    ) -> MappedMemoryRange<'static> {
        let &MappedMemoryRange {
            offset,
            size,
            _ne: _,
        } = memory_range;

        let memory = self.device_memory();
        let offset = self.offset() + offset;
        let mut size = size.unwrap_or(self.size() - memory_range.offset);

        // VUID-VkMappedMemoryRange-size-01390
        if memory_range.offset + size == self.size() {
            // We can align the end of the range like this without aliasing other allocations,
            // because the memory allocator must ensure that all allocations are aligned to the
            // atom size for non-host-coherent host-visible memory.
            let end = cmp::min(
                align_up(offset + size, memory.atom_size()),
                memory.allocation_size(),
            );
            size = end - offset;
        }

        MappedMemoryRange {
            offset,
            size: Some(size),
            _ne: crate::NE,
        }
    }

    pub(crate) fn to_vk_bind_buffer_memory_info(
        &self,
        buffer_vk: vk::Buffer,
    ) -> vk::BindBufferMemoryInfo<'static> {
        let &Self {
            ref device_memory,
            offset,
            ..
        } = self;

        vk::BindBufferMemoryInfo::default()
            .buffer(buffer_vk)
            .memory(device_memory.handle())
            .memory_offset(offset)
    }

    pub(crate) fn to_vk_bind_image_memory_info(
        &self,
        image_vk: vk::Image,
    ) -> vk::BindImageMemoryInfo<'static> {
        let &Self {
            ref device_memory,
            offset,
            ..
        } = self;

        vk::BindImageMemoryInfo::default()
            .image(image_vk)
            .memory(device_memory.handle())
            .memory_offset(offset)
    }
}

impl Drop for ResourceMemory {
    #[inline]
    fn drop(&mut self) {
        let device_memory = unsafe { ManuallyDrop::take(&mut self.device_memory) }.0;

        if let Some(allocator) = &self.allocator {
            let allocation = MemoryAlloc {
                device_memory,
                suballocation: self.suballocation(),
                allocation_handle: self.allocation_handle,
            };

            // SAFETY: Enforced by the safety contract of [`ResourceMemory::from_allocation`].
            unsafe { allocator.deallocate(allocation) };
        }
    }
}

unsafe impl DeviceOwned for ResourceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device_memory().device()
    }
}

/// Properties of the memory in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryProperties {
    /// The available memory types.
    pub memory_types: Vec<MemoryType>,

    /// The available memory heaps.
    pub memory_heaps: Vec<MemoryHeap>,
}

impl MemoryProperties {
    pub(crate) fn to_mut_vk2() -> vk::PhysicalDeviceMemoryProperties2KHR<'static> {
        vk::PhysicalDeviceMemoryProperties2KHR::default()
    }

    pub(crate) fn from_vk2(val_vk: &vk::PhysicalDeviceMemoryProperties2<'_>) -> Self {
        let vk::PhysicalDeviceMemoryProperties2 {
            memory_properties, ..
        } = val_vk;

        Self::from_vk(memory_properties)
    }

    pub(crate) fn from_vk(val_vk: &vk::PhysicalDeviceMemoryProperties) -> Self {
        let memory_types_vk = val_vk.memory_types_as_slice();
        let memory_heaps_vk = val_vk.memory_heaps_as_slice();

        Self {
            memory_types: memory_types_vk.iter().map(MemoryType::from_vk).collect(),
            memory_heaps: memory_heaps_vk.iter().map(MemoryHeap::from_vk).collect(),
        }
    }
}

/// A memory type in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryType {
    /// The properties of this memory type.
    pub property_flags: MemoryPropertyFlags,

    /// The index of the memory heap that this memory type corresponds to.
    pub heap_index: u32,
}

impl MemoryType {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub(crate) fn from_vk(val_vk: &vk::MemoryType) -> Self {
        let &vk::MemoryType {
            property_flags,
            heap_index,
        } = val_vk;

        MemoryType {
            property_flags: property_flags.into(),
            heap_index,
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Properties of a memory type.
    MemoryPropertyFlags = MemoryPropertyFlags(u32);

    /// The memory is located on the device, and is allocated from a heap that also has the
    /// [`DEVICE_LOCAL`] flag set.
    ///
    /// For some devices, particularly integrated GPUs, the device shares memory with the host and
    /// all memory may be device-local, so the distinction is moot. However, if the device has
    /// non-device-local memory, it is usually faster for the device to access device-local memory.
    /// Therefore, device-local memory is preferred for data that will only be accessed by
    /// the device.
    ///
    /// If the device and host do not share memory, data transfer between host and device may
    /// involve sending the data over the data bus that connects the two. Accesses are faster if
    /// they do not have to cross this barrier: device-local memory is fast for the device to
    /// access, but slower to access by the host. However, there are devices that share memory with
    /// the host, yet have distinct device-local and non-device local memory types. In that case,
    /// the speed difference may not be large.
    ///
    /// For data transfer between host and device, it is most efficient if the memory is located
    /// at the destination of the transfer. Thus, if [`HOST_VISIBLE`] versions of both are
    /// available, device-local memory is preferred for host-to-device data transfer, while
    /// non-device-local memory is preferred for device-to-host data transfer. This is because data
    /// is usually written only once but potentially read several times, and because reads can take
    /// advantage of caching while writes cannot.
    ///
    /// Devices may have memory types that are neither `DEVICE_LOCAL` nor [`HOST_VISIBLE`]. This
    /// is regular host memory that is made available to the device exclusively. Although it will be
    /// slower to access from the device than `DEVICE_LOCAL` memory, it can be faster than
    /// [`HOST_VISIBLE`] memory. It can be used as overflow space if the device is out of memory.
    ///
    /// [`DEVICE_LOCAL`]: MemoryHeapFlags::DEVICE_LOCAL
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    DEVICE_LOCAL = DEVICE_LOCAL,

    /// The memory can be mapped into the memory space of the host and accessed as regular RAM.
    ///
    /// Memory of this type is required to transfer data between the host and the device. If
    /// the memory is going to be accessed by the device more than a few times, it is recommended
    /// to copy the data to non-`HOST_VISIBLE` memory first if it is available.
    ///
    /// `HOST_VISIBLE` memory is always at least either [`HOST_COHERENT`] or [`HOST_CACHED`],
    /// but it can be both.
    ///
    /// [`HOST_COHERENT`]: MemoryPropertyFlags::HOST_COHERENT
    /// [`HOST_CACHED`]: MemoryPropertyFlags::HOST_CACHED
    HOST_VISIBLE = HOST_VISIBLE,

    /// Host access to the memory does not require calling [`invalidate_range`] to make device
    /// writes visible to the host, nor [`flush_range`] to flush host writes back to the device.
    ///
    /// [`invalidate_range`]: DeviceMemory::invalidate_range
    /// [`flush_range`]: DeviceMemory::flush_range
    HOST_COHERENT = HOST_COHERENT,

    /// The memory is cached by the host.
    ///
    /// `HOST_CACHED` memory is fast for reads and random access from the host, so it is preferred
    /// for device-to-host data transfer. Memory that is [`HOST_VISIBLE`] but not `HOST_CACHED` is
    /// often slow for all accesses other than sequential writing, so it is more suited for
    /// host-to-device transfer, and it is often beneficial to write the data in sequence.
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    HOST_CACHED = HOST_CACHED,

    /// Allocations made from the memory are lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation based on need. You can call
    /// [`DeviceMemory::commitment`] to query how much memory is currently committed to an
    /// allocation.
    ///
    /// Memory of this type can only be used on images created with a certain flag, and is never
    /// [`HOST_VISIBLE`].
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    LAZILY_ALLOCATED = LAZILY_ALLOCATED,

    /// The memory can only be accessed by the device, and allows protected queue access.
    ///
    /// Memory of this type is never [`HOST_VISIBLE`], [`HOST_COHERENT`] or [`HOST_CACHED`].
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`HOST_COHERENT`]: MemoryPropertyFlags::HOST_COHERENT
    /// [`HOST_CACHED`]: MemoryPropertyFlags::HOST_CACHED
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),

    /// Device accesses to the memory are automatically made available and visible to other device
    /// accesses.
    ///
    /// Memory of this type is slower to access by the device, so it is best avoided for general
    /// purpose use. Because of its coherence properties, however, it may be useful for debugging.
    DEVICE_COHERENT = DEVICE_COHERENT_AMD
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(amd_device_coherent_memory)]),
    ]),

    /// The memory is not cached on the device.
    ///
    /// `DEVICE_UNCACHED` memory is always also [`DEVICE_COHERENT`].
    ///
    /// [`DEVICE_COHERENT`]: MemoryPropertyFlags::DEVICE_COHERENT
    DEVICE_UNCACHED = DEVICE_UNCACHED_AMD
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(amd_device_coherent_memory)]),
    ]),

    /// Other devices can access the memory via remote direct memory access (RDMA).
    RDMA_CAPABLE = RDMA_CAPABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_external_memory_rdma)]),
    ]),
}

/// The current memory budget and usage of each memory heap of a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryBudget {
    /// For each memory heap, an estimate of how much memory the process can allocate before
    /// allocation is likely to fail or cause performance degradation. Indexed identically to
    /// [`MemoryProperties::memory_heaps`].
    pub heap_budget: Vec<DeviceSize>,

    /// For each memory heap, an estimate of how much memory the process is currently using.
    /// Indexed identically to [`MemoryProperties::memory_heaps`].
    pub heap_usage: Vec<DeviceSize>,
}

impl MemoryBudget {
    pub(crate) fn to_mut_vk2() -> vk::PhysicalDeviceMemoryBudgetPropertiesEXT<'static> {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default()
    }

    pub(crate) fn from_vk(
        val_vk: &vk::PhysicalDeviceMemoryBudgetPropertiesEXT<'_>,
        memory_heap_count: usize,
    ) -> Self {
        let &vk::PhysicalDeviceMemoryBudgetPropertiesEXT {
            heap_budget,
            heap_usage,
            ..
        } = val_vk;

        Self {
            heap_budget: heap_budget[..memory_heap_count].to_vec(),
            heap_usage: heap_usage[..memory_heap_count].to_vec(),
        }
    }
}

/// A memory heap in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryHeap {
    /// The size of the heap in bytes.
    pub size: DeviceSize,

    /// Attributes of the heap.
    pub flags: MemoryHeapFlags,
}

impl MemoryHeap {
    pub(crate) fn from_vk(val_vk: &vk::MemoryHeap) -> Self {
        let &vk::MemoryHeap { size, flags } = val_vk;

        Self {
            size,
            flags: flags.into(),
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Attributes of a memory heap.
    MemoryHeapFlags = MemoryHeapFlags(u32);

    /// The heap corresponds to device-local memory.
    DEVICE_LOCAL = DEVICE_LOCAL,

    /// If used on a logical device that represents more than one physical device, allocations are
    /// replicated across each physical device's instance of this heap.
    MULTI_INSTANCE = MULTI_INSTANCE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([InstanceExtension(khr_device_group_creation)]),
    ]),
}

/// Represents requirements expressed by the Vulkan implementation when it comes to binding memory
/// to a resource.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MemoryRequirements {
    /// Memory layout required for the resource.
    pub layout: DeviceLayout,

    /// Indicates which memory types can be used. Each bit that is set to 1 means that the memory
    /// type whose index is the same as the position of the bit can be used.
    pub memory_type_bits: u32,

    /// Whether implementation prefers to use dedicated allocations (in other words, allocate
    /// a whole block of memory dedicated to this resource alone).
    /// This will be `false` if the device API version is less than 1.1 and the
    /// [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
    /// extension is not enabled on the device.
    pub prefers_dedicated_allocation: bool,

    /// Whether implementation requires the use of a dedicated allocation (in other words, allocate
    /// a whole block of memory dedicated to this resource alone).
    /// This will be `false` if the device API version is less than 1.1 and the
    /// [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
    /// extension is not enabled on the device.
    pub requires_dedicated_allocation: bool,
}

impl MemoryRequirements {
    pub(crate) fn to_mut_vk2(
        extensions_vk: &mut MemoryRequirements2ExtensionsVk,
    ) -> vk::MemoryRequirements2<'_> {
        let mut val_vk = vk::MemoryRequirements2::default();

        let MemoryRequirements2ExtensionsVk { dedicated_vk } = extensions_vk;

        if let Some(next) = dedicated_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_mut_vk2_extensions(device: &Device) -> MemoryRequirements2ExtensionsVk {
        let dedicated_vk = (device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
            .then(|| {
                debug_assert!(
                    device.api_version() >= Version::V1_1
                        || device.enabled_extensions().khr_get_memory_requirements2
                );

                vk::MemoryDedicatedRequirements::default()
            });

        MemoryRequirements2ExtensionsVk { dedicated_vk }
    }

    pub(crate) fn from_vk2(
        val_vk: &vk::MemoryRequirements2<'_>,
        extensions_vk: &MemoryRequirements2ExtensionsVk,
    ) -> Self {
        let &vk::MemoryRequirements2 {
            memory_requirements:
                vk::MemoryRequirements {
                    size,
                    alignment,
                    memory_type_bits,
                },
            ..
        } = val_vk;

        let mut val = Self {
            layout: DeviceLayout::from_size_alignment(size, alignment).unwrap(),
            memory_type_bits,
            prefers_dedicated_allocation: false,
            requires_dedicated_allocation: false,
        };

        let MemoryRequirements2ExtensionsVk { dedicated_vk } = extensions_vk;

        if let Some(val_vk) = dedicated_vk {
            let &vk::MemoryDedicatedRequirements {
                prefers_dedicated_allocation,
                requires_dedicated_allocation,
                ..
            } = val_vk;

            val = Self {
                prefers_dedicated_allocation: prefers_dedicated_allocation != 0,
                requires_dedicated_allocation: requires_dedicated_allocation != 0,
                ..val
            };
        }

        val
    }
}

pub(crate) struct MemoryRequirements2ExtensionsVk {
    pub(crate) dedicated_vk: Option<vk::MemoryDedicatedRequirements<'static>>,
}

/// Indicates a specific resource to allocate memory for.
///
/// Using dedicated allocations can yield better performance, but requires the
/// [`khr_dedicated_allocation`](crate::device::DeviceExtensions::khr_dedicated_allocation)
/// extension to be enabled on the device.
///
/// If a dedicated allocation is performed, it must not be bound to any resource other than the
/// one that was passed with the enumeration.
#[derive(Clone, Copy, Debug)]
pub enum DedicatedAllocation<'a> {
    /// Allocation dedicated to a buffer.
    Buffer(&'a RawBuffer),
    /// Allocation dedicated to an image.
    Image(&'a RawImage),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DedicatedTo {
    Buffer(NonZero<u64>),
    Image(NonZero<u64>),
}

impl From<DedicatedAllocation<'_>> for DedicatedTo {
    fn from(dedicated_allocation: DedicatedAllocation<'_>) -> Self {
        match dedicated_allocation {
            DedicatedAllocation::Buffer(buffer) => Self::Buffer(buffer.id()),
            DedicatedAllocation::Image(image) => Self::Image(image.id()),
        }
    }
}

/// The properties for exporting or importing external memory, when a buffer or image is created
/// with a specific configuration.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ExternalMemoryProperties {
    /// Whether a dedicated memory allocation is required for the queried external handle type.
    pub dedicated_only: bool,

    /// Whether memory can be exported to an external source with the queried
    /// external handle type.
    pub exportable: bool,

    /// Whether memory can be imported from an external source with the queried
    /// external handle type.
    pub importable: bool,

    /// Which external handle types can be re-exported after the queried external handle type has
    /// been imported.
    pub export_from_imported_handle_types: ExternalMemoryHandleTypes,

    /// Which external handle types can be enabled along with the queried external handle type
    /// when creating the buffer or image.
    pub compatible_handle_types: ExternalMemoryHandleTypes,
}

impl ExternalMemoryProperties {
    pub(crate) fn from_vk(val_vk: &vk::ExternalMemoryProperties) -> Self {
        let &vk::ExternalMemoryProperties {
            external_memory_features,
            export_from_imported_handle_types,
            compatible_handle_types,
        } = val_vk;

        Self {
            dedicated_only: external_memory_features
                .intersects(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
            exportable: external_memory_features
                .intersects(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
            importable: external_memory_features
                .intersects(vk::ExternalMemoryFeatureFlags::IMPORTABLE),
            export_from_imported_handle_types: export_from_imported_handle_types.into(),
            compatible_handle_types: compatible_handle_types.into(),
        }
    }
}

/// The properties of a Unix file descriptor when it is imported.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryFdProperties {
    /// A bitmask of the indices of memory types that can be used with the file.
    pub memory_type_bits: u32,
}

impl MemoryFdProperties {
    pub(crate) fn to_mut_vk() -> vk::MemoryFdPropertiesKHR<'static> {
        vk::MemoryFdPropertiesKHR::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::MemoryFdPropertiesKHR<'_>) -> Self {
        let &vk::MemoryFdPropertiesKHR {
            memory_type_bits, ..
        } = val_vk;

        Self { memory_type_bits }
    }
}

/// The properties of a Windows handle when it is imported.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryWin32HandleProperties {
    /// A bitmask of the indices of memory types that can be used with the handle.
    pub memory_type_bits: u32,
}

impl MemoryWin32HandleProperties {
    pub(crate) fn to_mut_vk() -> vk::MemoryWin32HandlePropertiesKHR<'static> {
        vk::MemoryWin32HandlePropertiesKHR::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::MemoryWin32HandlePropertiesKHR<'_>) -> Self {
        let &vk::MemoryWin32HandlePropertiesKHR {
            memory_type_bits, ..
        } = val_vk;

        Self { memory_type_bits }
    }
}

#[inline(always)]
pub(crate) fn is_aligned(offset: DeviceSize, alignment: DeviceAlignment) -> bool {
    offset & (alignment.as_devicesize() - 1) == 0
}

/// Performs bounds-checking of a Vulkan memory range. Analog of `std::slice::range`.
pub(crate) fn range(
    range: impl RangeBounds<DeviceSize>,
    bounds: RangeTo<DeviceSize>,
) -> Option<Range<DeviceSize>> {
    let len = bounds.end;

    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(end) => end.checked_add(1)?,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };

    (start <= end && end <= len).then_some(Range { start, end })
}

/// Converts a `RangeBounds` into a `Range` without doing any bounds checking.
pub(crate) fn range_unchecked(
    range: impl RangeBounds<DeviceSize>,
    bounds: RangeTo<DeviceSize>,
) -> Range<DeviceSize> {
    let len = bounds.end;

    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(end) => end + 1,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };

    debug_assert!(start <= end && end <= len);

    Range { start, end }
}
