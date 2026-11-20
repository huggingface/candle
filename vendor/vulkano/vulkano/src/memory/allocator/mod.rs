//! In Vulkan, suballocation of [`DeviceMemory`] is left to the application, because every
//! application has slightly different needs and one can not incorporate an allocator into the
//! driver that would perform well in all cases. Vulkano stays true to this sentiment, but aims to
//! reduce the burden on the user as much as possible. You have a toolbox of [suballocators] to
//! choose from that cover all allocation algorithms, which you can compose into any kind of
//! [hierarchy] you wish. This way you have maximum flexibility while still only using a few
//! `DeviceMemory` blocks and not writing any of the very error-prone code.
//!
//! If you just want to allocate memory and don't have any special needs, look no further than the
//! [`StandardMemoryAllocator`].
//!
//! # Why not just allocate `DeviceMemory`?
//!
//! But the driver has an allocator! Otherwise you wouldn't be able to allocate `DeviceMemory`,
//! right? Indeed, but that allocation is very expensive. Not only that, there is also a pretty low
//! limit on the number of allocations by the drivers. See, everything in Vulkan tries to keep you
//! away from allocating `DeviceMemory` too much. These limits are used by the implementation to
//! optimize on its end, while the application optimizes on the other end.
//!
//! # Alignment
//!
//! At the end of the day, memory needs to be backed by hardware somehow. A *memory cell* stores a
//! single *bit*, bits are grouped into *bytes* and bytes are grouped into *words*. Intuitively, it
//! should make sense that accessing single bits at a time would be very inefficient. That is why
//! computers always access a whole word of memory at once, at least. That means that if you tried
//! to do an unaligned access, you would need to access twice the number of memory locations.
//!
//! Example aligned access, performing bitwise NOT on the (64-bit) word at offset 0x08:
//!
//! ```plain
//!     | 08                      | 10                      | 18
//! ----+-------------------------+-------------------------+----
//! ••• | 35 35 35 35 35 35 35 35 | 01 23 45 67 89 ab cd ef | •••
//! ----+-------------------------+-------------------------+----
//!     ,            |            ,
//!     +------------|------------+
//!     '            v            '
//! ----+-------------------------+-------------------------+----
//! ••• | ca ca ca ca ca ca ca ca | 01 23 45 67 89 ab cd ef | •••
//! ----+-------------------------+-------------------------+----
//! ```
//!
//! Same example as above, but this time unaligned with a word at offset 0x0a:
//!
//! ```plain
//!     | 08    0a                | 10                      | 18
//! ----+-------------------------+-------------------------+----
//! ••• | cd ef 35 35 35 35 35 35 | 35 35 01 23 45 67 89 ab | •••
//! ----+-------------------------+-------------------------+----
//!            ,            |            ,
//!            +------------|------------+
//!            '            v            '
//! ----+-------------------------+-------------------------+----
//! ••• | cd ef ca ca ca ca ca ca | ca ca 01 23 45 67 89 ab | •••
//! ----+-------------------------+-------------------------+----
//! ```
//!
//! As you can see, in the unaligned case the hardware would need to read both the word at offset
//! 0x08 and the word at the offset 0x10 and then shift the bits from one register into the other.
//! Safe to say it should to be avoided, and this is why we need alignment. This example also goes
//! to show how inefficient unaligned writes are. Say you pieced together your word as described,
//! and now you want to perform the bitwise NOT and write the result back. Difficult, isn't it?
//! That's due to the fact that even though the chunks occupy different ranges in memory, they are
//! still said to *alias* each other, because if you try to write to one memory location, you would
//! be overwriting 2 or more different chunks of data.
//!
//! ## Pages
//!
//! It doesn't stop at the word, though. Words are further grouped into *pages*. These are
//! typically power-of-two multiples of the word size, much like words are typically powers of two
//! themselves. You can easily extend the concepts from the previous examples to pages if you think
//! of the examples as having a page size of 1 word. Two resources are said to alias if they share
//! a page, and therefore should be aligned to the page size. What the page size is depends on the
//! context, and a computer might have multiple different ones for different parts of hardware.
//!
//! ## Memory requirements
//!
//! A Vulkan device might have any number of reasons it would want certain alignments for certain
//! resources. For example, the device might have different caches for different types of
//! resources, which have different page sizes. Maybe the device wants to store images in some
//! other cache compared to buffers which needs different alignment. Or maybe images of different
//! layouts require different alignment, or buffers with different usage/mapping do. The specifics
//! don't matter in the end, this just goes to illustrate the point. This is why memory
//! requirements in Vulkan vary not only with the Vulkan implementation, but also with the type of
//! resource.
//!
//! ## Buffer-image granularity
//!
//! This unfortunately named granularity is the page size which a linear resource neighboring a
//! non-linear resource must be aligned to in order for them not to alias. The difference between
//! the memory requirements of the individual resources and the [buffer-image granularity] is that
//! the memory requirements only apply to the resource they are for, while the buffer-image
//! granularity applies to two neighboring resources. For example, you might create two buffers,
//! which might have two different memory requirements, but as long as those are satisfied, you can
//! put these buffers cheek to cheek. On the other hand, if one of them is an (optimal layout)
//! image, then they must not share any page, whose size is given by this granularity. The Vulkan
//! implementation can use this for additional optimizations if it needs to, or report a
//! granularity of 1.
//!
//! # Fragmentation
//!
//! Memory fragmentation refers to the wastage of memory that results from alignment requirements
//! and/or dynamic memory allocation. As such, some level of fragmentation is always going to be
//! inevitable. Different allocation algorithms each have their own characteristics and trade-offs
//! in relation to fragmentation.
//!
//! ## Internal Fragmentation
//!
//! This type of fragmentation arises from alignment requirements. These might be imposed by the
//! Vulkan implementation or the application itself.
//!
//! Say for example your allocations need to be aligned to 64B, then any allocation whose size is
//! not a multiple of the alignment will need padding at the end:
//!
//! ```plain
//!     | 0x040            | 0x080            | 0x0c0            | 0x100
//! ----+------------------+------------------+------------------+--------
//!     | ############     | ################ | ########         | #######
//! ••• | ### 48 B ###     | ##### 64 B ##### | # 32 B #         | ### •••
//!     | ############     | ################ | ########         | #######
//! ----+------------------+------------------+------------------+--------
//! ```
//!
//! If this alignment is imposed by the Vulkan implementation, then there's nothing one can do
//! about this. Simply put, that space is unusable. One also shouldn't want to do anything about
//! it, since these requirements have very good reasons, as described in further detail in previous
//! sections. They prevent resources from aliasing so that performance is optimal.
//!
//! It might seem strange that the application would want to cause internal fragmentation itself,
//! but this is often a good trade-off to reduce or even completely eliminate external
//! fragmentation. Internal fragmentation is very predictable, which makes it easier to deal with.
//!
//! ## External fragmentation
//!
//! With external fragmentation, what happens is that while the allocations might be using their
//! own memory totally efficiently, the way they are arranged in relation to each other would
//! prevent a new contiguous chunk of memory to be allocated even though there is enough free space
//! left. That is why this fragmentation is said to be external to the allocations. Also, the
//! allocations together with the fragments in-between add overhead both in terms of space and time
//! to the allocator, because it needs to keep track of more things overall.
//!
//! As an example, take these 4 allocations within some block, with the rest of the block assumed
//! to be full:
//!
//! ```plain
//! +-----+-------------------+-------+-----------+-- - - --+
//! |     |                   |       |           |         |
//! |  A  |         B         |   C   |     D     |   •••   |
//! |     |                   |       |           |         |
//! +-----+-------------------+-------+-----------+-- - - --+
//! ```
//!
//! The allocations were all done in order, and naturally there is no fragmentation at this point.
//! Now if we free B and D, since these are done out of order, we will be left with holes between
//! the other allocations, and we won't be able to fit allocation E anywhere:
//!
//! ```plain
//! +-----+-------------------+-------+-----------+-- - - --+       +-------------------------+
//! |     |                   |       |           |         |   ?   |                         |
//! |  A  |                   |   C   |           |   •••   |  <==  |            E            |
//! |     |                   |       |           |         |       |                         |
//! +-----+-------------------+-------+-----------+-- - - --+       +-------------------------+
//! ```
//!
//! So fine, we use a different block for E, and just use this block for allocations that fit:
//!
//! ```plain
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! |     |   |     |         |       |     |     |         |
//! |  A  | H |  I  |    J    |   C   |  F  |  G  |   •••   |
//! |     |   |     |         |       |     |     |         |
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! ```
//!
//! Sure, now let's free some shall we? And voilà, the problem just became much worse:
//!
//! ```plain
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! |     |   |     |         |       |     |     |         |
//! |  A  |   |  I  |    J    |       |  F  |     |   •••   |
//! |     |   |     |         |       |     |     |         |
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! ```
//!
//! # Leakage
//!
//! Memory leaks happen when allocations are kept alive past their shelf life. This most often
//! occurs because of [cyclic references]. If you have structures that have cycles, then make sure
//! you read the documentation for [`Arc`]/[`Rc`] carefully to avoid memory leaks. You can also
//! introduce memory leaks willingly by using [`mem::forget`] or [`Box::leak`] to name a few. In
//! all of these examples the memory can never be reclaimed, but that doesn't have to be the case
//! for something to be considered a leak. Say for example you have a [region] which you
//! suballocate, and at some point you drop all the suballocations. When that happens, the region
//! can be returned (freed) to the next level up the hierarchy, or it can be reused by another
//! suballocator. But if you happen to keep alive just one suballocation for the duration of the
//! program for instance, then the whole region is also kept as it is for that time (and keep in
//! mind this bubbles up the hierarchy). Therefore, for the program, that memory might be a leak
//! depending on the allocator, because some allocators wouldn't be able to reuse the entire rest
//! of the region. You must always consider the lifetime of your resources when choosing the
//! appropriate allocator.
//!
//! [suballocators]: Suballocator
//! [hierarchy]: Suballocator#memory-hierarchies
//! [buffer-image granularity]: crate::device::DeviceProperties::buffer_image_granularity
//! [cyclic references]: Arc#breaking-cycles-with-weak
//! [`Rc`]: std::rc::Rc
//! [region]: Suballocator#regions

use self::{aliasable_box::AliasableBox, array_vec::ArrayVec, suballocator::Region};
pub use self::{
    layout::DeviceLayout,
    suballocator::{
        AllocationType, BuddyAllocator, BumpAllocator, FreeListAllocator, Suballocation,
        Suballocator, SuballocatorError, TlsfAllocator,
    },
};
use super::{
    DedicatedAllocation, DeviceAlignment, DeviceMemory, ExternalMemoryHandleTypes,
    MemoryAllocateFlags, MemoryAllocateInfo, MemoryMapInfo, MemoryProperties, MemoryPropertyFlags,
    MemoryRequirements,
};
use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    DeviceSize, Validated, Version, VulkanError,
};
use ash::vk::{self, MAX_MEMORY_TYPES};
use parking_lot::{Mutex, MutexGuard};
use slabbin::SlabAllocator;
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    iter::FusedIterator,
    ops::BitOr,
    ptr, slice,
    sync::Arc,
};

mod layout;
pub mod suballocator;

/// General-purpose memory allocators which allocate from any memory type dynamically as needed.
///
/// # Safety
///
/// - `allocate`, `allocate_from_type` and `allocate_dedicated` must return a memory block that is
///   in bounds of its device memory.
/// - `allocate` and `allocate_from_type` must return a memory block that doesn't alias any other
///   currently allocated memory blocks:
///   - Two currently allocated memory blocks must not share any memory locations, meaning that the
///     intersection of the byte ranges of the two memory blocks must be empty.
///   - Two neighboring currently allocated memory blocks must not share any [page] whose size is
///     given by the [buffer-image granularity], unless either both were allocated with
///     [`AllocationType::Linear`] or both were allocated with [`AllocationType::NonLinear`].
///   - For all [host-visible] memory types that are not [host-coherent], all memory blocks must be
///     aligned to the [non-coherent atom size].
///   - The size does **not** have to be padded to the alignment. That is, as long the offset is
///     aligned and the memory blocks don't share any memory locations, a memory block is not
///     considered to alias another even if the padded size shares memory locations with another
///     memory block.
/// - A memory block must stay allocated until either `deallocate` is called on it or the allocator
///   is dropped. If the allocator is cloned, it must produce the same allocator, and memory blocks
///   must stay allocated until either `deallocate` is called on the memory block using any of the
///   clones or all of the clones have been dropped.
///
/// [page]: self#pages
/// [buffer-image granularity]: self#buffer-image-granularity
/// [host-visible]: MemoryPropertyFlags::HOST_VISIBLE
/// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
/// [non-coherent atom size]: crate::device::DeviceProperties::non_coherent_atom_size
pub unsafe trait MemoryAllocator: DeviceOwned + Send + Sync + 'static {
    /// Finds the most suitable memory type index in `memory_type_bits` using the given `filter`.
    /// Returns [`None`] if the requirements are too strict and no memory type is able to satisfy
    /// them.
    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        filter: MemoryTypeFilter,
    ) -> Option<u32>;

    /// Allocates memory from a specific memory type.
    ///
    /// # Arguments
    ///
    /// - `memory_type_index` - The index of the memory type to allocate from.
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `never_allocate` - If `true` then the allocator should never allocate `DeviceMemory`,
    ///   instead only suballocate from existing blocks.
    fn try_allocate_from_type(
        &self,
        memory_type_index: u32,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>>;

    /// Allocates memory from a specific memory type without doing any checks.
    ///
    /// <div class="vulkano-alert-caution">
    ///
    /// > Caution
    /// >
    /// > You should only call this when necessary. Humans are famously bad at guessing what needs
    /// > to be optimized, so **the only way** to know is by profiling. Have you profiled to make
    /// > sure that this actually makes a difference? In 99% of cases (not an exaggeration),
    /// > calling the unchecked version of one of our functions/methods makes absolutely no
    /// > difference, so make sure first.
    ///
    /// </div>
    ///
    /// # Arguments
    ///
    /// - `memory_type_index` - The index of the memory type to allocate from.
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `never_allocate` - If `true` then the allocator should never allocate `DeviceMemory`,
    ///   instead only suballocate from existing blocks.
    ///
    /// # Safety
    ///
    /// - If calling [`try_allocate_from_type`] with the same arguments would return a
    ///   [`ValidationError`], calling this method is *undefined behavior*!
    ///
    /// [`try_allocate_from_type`]: Self::try_allocate_from_type
    unsafe fn allocate_from_type_unchecked(
        &self,
        memory_type_index: u32,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, MemoryAllocatorError>;

    /// Allocates memory according to requirements.
    ///
    /// # Arguments
    ///
    /// - `requirements` - Requirements of the resource you want to allocate memory for.
    ///
    ///   If you plan to bind this memory directly to a non-sparse resource, then this must
    ///   correspond to the value returned by either [`RawBuffer::memory_requirements`] or
    ///   [`RawImage::memory_requirements`] for the respective buffer or image.
    ///
    /// - `allocation_type` - What type of resource this allocation will be used for.
    ///
    ///   This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    ///   images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    ///   bind memory allocated with the [`NonLinear`] type to buffers and linear images. You
    ///   should never use the [`Unknown`] type unless you have to, as that can be less memory
    ///   efficient.
    ///
    /// - `dedicated_allocation` - Allows a dedicated allocation to be created.
    ///
    ///   You should always fill this field in if you are allocating memory for a non-sparse
    ///   resource, otherwise the allocator won't be able to create a dedicated allocation if one
    ///   is required or recommended.
    ///
    ///   This argument is silently ignored (treated as `None`) if the device API version is below
    ///   1.1 and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// [`RawBuffer::memory_requirements`]: crate::buffer::sys::RawBuffer::memory_requirements
    /// [`RawImage::memory_requirements`]: crate::image::sys::RawImage::memory_requirements
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    fn try_allocate(
        &self,
        requirements: &MemoryRequirements,
        allocation_type: AllocationType,
        create_info: &AllocationCreateInfo<'_>,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>>;

    /// Allocates memory according to requirements without doing any checks.
    ///
    /// <div class="vulkano-alert-caution">
    ///
    /// > Caution
    /// >
    /// > You should only call this when necessary. Humans are famously bad at guessing what needs
    /// > to be optimized, so **the only way** to know is by profiling. Have you profiled to make
    /// > sure that this actually makes a difference? In 99% of cases (not an exaggeration),
    /// > calling the unchecked version of one of our functions/methods makes absolutely no
    /// > difference, so make sure first.
    ///
    /// </div>
    ///
    /// # Arguments
    ///
    /// - `requirements` - Requirements of the resource you want to allocate memory for.
    ///
    ///   If you plan to bind this memory directly to a non-sparse resource, then this must
    ///   correspond to the value returned by either [`RawBuffer::memory_requirements`] or
    ///   [`RawImage::memory_requirements`] for the respective buffer or image.
    ///
    /// - `allocation_type` - What type of resource this allocation will be used for.
    ///
    ///   This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    ///   images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    ///   bind memory allocated with the [`NonLinear`] type to buffers and linear images. You
    ///   should never use the [`Unknown`] type unless you have to, as that can be less memory
    ///   efficient.
    ///
    /// - `dedicated_allocation` - Allows a dedicated allocation to be created.
    ///
    ///   You should always fill this field in if you are allocating memory for a non-sparse
    ///   resource, otherwise the allocator won't be able to create a dedicated allocation if one
    ///   is required or recommended.
    ///
    ///   This argument is silently ignored (treated as `None`) if the device API version is below
    ///   1.1 and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// # Safety
    ///
    /// - If calling [`try_allocate`] with the same arguments would return a [`ValidationError`],
    ///   calling this method is *undefined behavior*!
    ///
    /// [`RawBuffer::memory_requirements`]: crate::buffer::sys::RawBuffer::memory_requirements
    /// [`RawImage::memory_requirements`]: crate::image::sys::RawImage::memory_requirements
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    /// [`try_allocate`]: Self::try_allocate
    unsafe fn allocate_unchecked(
        &self,
        requirements: &MemoryRequirements,
        allocation_type: AllocationType,
        create_info: &AllocationCreateInfo<'_>,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, MemoryAllocatorError>;

    /// Creates an allocation with a whole device memory block dedicated to it.
    fn try_allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>>;

    /// Creates an allocation with a whole device memory block dedicated to it without doing any
    /// checks.
    ///
    /// <div class="vulkano-alert-caution">
    ///
    /// > Caution
    /// >
    /// > You should only call this when necessary. Humans are famously bad at guessing what needs
    /// > to be optimized, so **the only way** to know is by profiling. Have you profiled to make
    /// > sure that this actually makes a difference? In 99% of cases (not an exaggeration),
    /// > calling the unchecked version of one of our functions/methods makes absolutely no
    /// > difference, so make sure first.
    ///
    /// </div>
    ///
    /// # Safety
    ///
    /// - If calling [`try_allocate_dedicated`] with the same arguments would return a
    ///   [`ValidationError`], calling this method is *undefined behavior*!
    ///
    /// [`try_allocate_dedicated`]: Self::try_allocate_dedicated
    unsafe fn allocate_dedicated_unchecked(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, MemoryAllocatorError>;

    /// Deallocates the given `allocation`.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `self`.
    unsafe fn deallocate(&self, allocation: MemoryAlloc);

    /// Coerces a reference to the allocator to a reference to a trait object.
    fn as_dyn(self: Arc<Self>) -> Arc<dyn MemoryAllocator>;
}

impl Debug for dyn MemoryAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("MemoryAllocator").finish_non_exhaustive()
    }
}

/// Describes what memory property flags are required, preferred and not preferred when picking a
/// memory type index.
///
/// # Examples
///
/// #### Direct device access only
///
/// For resources that the device frequently accesses, e.g. textures, render targets, or
/// intermediary buffers, you want device-local memory without any host access:
///
/// ```
/// # use vulkano::{
/// #     image::{Image, ImageCreateInfo, ImageUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let format = return;
/// # let extent = return;
/// #
/// let texture = Image::new(
///     &memory_allocator,
///     &ImageCreateInfo {
///         format,
///         extent,
///         usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
///         ..Default::default()
///     },
///     &AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// ```
///
/// #### Sequential writes from host, indirect device access
///
/// For staging, the resource is only ever written to sequentially. Also, since the device will
/// only read the staging resource once, it would yield no benefit to place it in device-local
/// memory, in fact it would be wasteful. Therefore, it's best to put it in host-local memory:
///
/// ```
/// # use vulkano::{
/// #     buffer::{Buffer, BufferCreateInfo, BufferUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// let staging_buffer = Buffer::new_sized(
///     &memory_allocator,
///     &BufferCreateInfo {
///         usage: BufferUsage::TRANSFER_SRC,
///         ..Default::default()
///     },
///     &AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_HOST |
///             MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// #
/// # let staging_buffer: vulkano::buffer::Subbuffer<u32> = staging_buffer;
/// ```
///
/// #### Sequential writes from host, direct device access
///
/// For resources that the device accesses directly, aka a buffer/image used for anything other
/// than transfers, it's probably best to put it in device-local memory:
///
/// ```
/// # use vulkano::{
/// #     buffer::{Buffer, BufferCreateInfo, BufferUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// let uniform_buffer = Buffer::new_sized(
///     &memory_allocator,
///     &BufferCreateInfo {
///         usage: BufferUsage::UNIFORM_BUFFER,
///         ..Default::default()
///     },
///     &AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE |
///             MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// #
/// # let uniform_buffer: vulkano::buffer::Subbuffer<u32> = uniform_buffer;
/// ```
///
/// #### Readback to host
///
/// For readback, e.g. getting the results of a compute shader back to the host:
///
/// ```
/// # use vulkano::{
/// #     buffer::{Buffer, BufferCreateInfo, BufferUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// let readback_buffer = Buffer::new_sized(
///     &memory_allocator,
///     &BufferCreateInfo {
///         usage: BufferUsage::TRANSFER_DST,
///         ..Default::default()
///     },
///     &AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_HOST |
///             MemoryTypeFilter::HOST_RANDOM_ACCESS,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// #
/// # let readback_buffer: vulkano::buffer::Subbuffer<u32> = readback_buffer;
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryTypeFilter {
    pub required_flags: MemoryPropertyFlags,
    pub preferred_flags: MemoryPropertyFlags,
    pub not_preferred_flags: MemoryPropertyFlags,
}

impl MemoryTypeFilter {
    /// Prefers picking a memory type with the [`DEVICE_LOCAL`] flag. **It does not affect whether
    /// the memory is host-visible**. You need to combine this with either the
    /// [`HOST_SEQUENTIAL_WRITE`] or [`HOST_RANDOM_ACCESS`] filter for that.
    ///
    /// Memory being device-local means that it is fastest to access for the device. However,
    /// for dedicated GPUs, getting memory in and out of VRAM requires to go through the PCIe bus,
    /// which is very slow and should therefore only be done when necessary.
    ///
    /// This filter is best suited for anything that the host doesn't access, but the device
    /// accesses frequently. For example textures, render targets, and intermediary buffers.
    ///
    /// For memory that the host does access but less frequently than the device, such as updating
    /// a uniform buffer each frame, device-local memory may also be preferred. In this case,
    /// because the memory is only written once before being consumed by the device and becoming
    /// outdated, it doesn't matter that the data is potentially transferred over the PCIe bus
    /// since it only happens once. Since this is only a preference, if you have some requirements
    /// such as the memory being [`HOST_VISIBLE`], those requirements will take precedence.
    ///
    /// For memory that the host doesn't access, and the device doesn't access directly, you may
    /// still prefer device-local memory if the memory is used regularly. For instance, an image
    /// that is swapped each frame.
    ///
    /// Keep in mind that for implementations with unified memory, there's only system RAM. That
    /// means that even if the implementation reports a memory type that is not `HOST_VISIBLE` and
    /// is `DEVICE_LOCAL`, its still the same unified memory as all other memory. However, it may
    /// still be faster to access. On the other hand, some such implementations don't report any
    /// memory types that are not `HOST_VISIBLE`, which makes sense since it's all system RAM. In
    /// that case you wouldn't need to do staging.
    ///
    /// Don't use this together with [`PREFER_HOST`], that makes no sense.
    ///
    /// [`DEVICE_LOCAL`]: MemoryPropertyFlags::DEVICE_LOCAL
    /// [`HOST_SEQUENTIAL_WRITE`]: Self::HOST_SEQUENTIAL_WRITE
    /// [`HOST_RANDOM_ACCESS`]: Self::HOST_RANDOM_ACCESS
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`PREFER_HOST`]: Self::PREFER_HOST
    pub const PREFER_DEVICE: Self = Self {
        required_flags: MemoryPropertyFlags::empty(),
        preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
        not_preferred_flags: MemoryPropertyFlags::empty(),
    };

    /// Prefers picking a memory type without the [`DEVICE_LOCAL`] flag. **It does not affect
    /// whether the memory is host-visible**. You need to combine this with either the
    /// [`HOST_SEQUENTIAL_WRITE`] or [`HOST_RANDOM_ACCESS`] filter for that.
    ///
    /// This option is best suited for resources that the host does access, but device doesn't
    /// access directly, such as staging buffers and readback buffers.
    ///
    /// For memory that the host does access but less frequently than the device, such as updating
    /// a uniform buffer each frame, you may still get away with using host-local memory if the
    /// updates are small and local enough. In such cases, the memory should be able to be quickly
    /// cached by the device, such that the data potentially being transferred over the PCIe bus
    /// wouldn't matter.
    ///
    /// For memory that the host doesn't access, and the device doesn't access directly, you may
    /// still prefer host-local memory if the memory is rarely used, such as for manually paging
    /// parts of device-local memory out in order to free up space on the device.
    ///
    /// Don't use this together with [`PREFER_DEVICE`], that makes no sense.
    ///
    /// [`DEVICE_LOCAL`]: MemoryPropertyFlags::DEVICE_LOCAL
    /// [`HOST_SEQUENTIAL_WRITE`]: Self::HOST_SEQUENTIAL_WRITE
    /// [`HOST_RANDOM_ACCESS`]: Self::HOST_RANDOM_ACCESS
    /// [`PREFER_DEVICE`]: Self::PREFER_DEVICE
    pub const PREFER_HOST: Self = Self {
        required_flags: MemoryPropertyFlags::empty(),
        preferred_flags: MemoryPropertyFlags::empty(),
        not_preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
    };

    /// This guarantees picking a memory type that has the [`HOST_VISIBLE`] flag. **It does not
    /// affect whether the memory is device-local or host-local**. You need to combine this with
    /// either the [`PREFER_DEVICE`] or [`PREFER_HOST`] filter for that.
    ///
    /// Using this filter allows the allocator to pick a memory type that is uncached and
    /// write-combined, which is ideal for sequential writes. However, this optimization might lead
    /// to poor performance for anything else. What counts as a sequential write is any kind of
    /// loop that writes memory locations in order, such as iterating over a slice while writing
    /// each element in order, or equivalently using [`slice::copy_from_slice`]. Copying sized data
    /// also counts, as rustc should write the memory locations in order. If you have a struct,
    /// make sure you write it member-by-member.
    ///
    /// Example use cases include staging buffers, as well as any other kind of buffer that you
    /// only write to from the host, like a uniform or vertex buffer.
    ///
    /// Don't use this together with [`HOST_RANDOM_ACCESS`], that makes no sense. If you do both a
    /// sequential write and read or random access, then you should use `HOST_RANDOM_ACCESS`
    /// instead. However, you could also consider using different allocations for the two purposes
    /// to get the most performance out, if that's possible.
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`PREFER_DEVICE`]: Self::PREFER_DEVICE
    /// [`PREFER_HOST`]: Self::PREFER_HOST
    /// [`HOST_RANDOM_ACCESS`]: Self::HOST_RANDOM_ACCESS
    pub const HOST_SEQUENTIAL_WRITE: Self = Self {
        required_flags: MemoryPropertyFlags::HOST_VISIBLE,
        preferred_flags: MemoryPropertyFlags::empty(),
        not_preferred_flags: MemoryPropertyFlags::HOST_CACHED,
    };

    /// This guarantees picking a memory type that has the [`HOST_VISIBLE`] and [`HOST_CACHED`]
    /// flags, which is best suited for readback and/or random access. **It does not affect whether
    /// the memory is device-local or host-local**. You need to combine this with either the
    /// [`PREFER_DEVICE`] or [`PREFER_HOST`] filter for that.
    ///
    /// Example use cases include using the device for things other than rendering and getting the
    /// results back to the host. That might be compute shading, or image or video manipulation, or
    /// screenshotting.
    ///
    /// Don't use this together with [`HOST_SEQUENTIAL_WRITE`], that makes no sense. If you are
    /// sure you only ever need to sequentially write to the allocation, then using
    /// `HOST_SEQUENTIAL_WRITE` instead will yield better performance.
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`HOST_CACHED`]: MemoryPropertyFlags::HOST_CACHED
    /// [`PREFER_DEVICE`]: Self::PREFER_DEVICE
    /// [`PREFER_HOST`]: Self::PREFER_HOST
    /// [`HOST_SEQUENTIAL_WRITE`]: Self::HOST_SEQUENTIAL_WRITE
    pub const HOST_RANDOM_ACCESS: Self = Self {
        required_flags: MemoryPropertyFlags::HOST_VISIBLE.union(MemoryPropertyFlags::HOST_CACHED),
        preferred_flags: MemoryPropertyFlags::empty(),
        not_preferred_flags: MemoryPropertyFlags::empty(),
    };

    /// Returns a `MemoryTypeFilter` with none of the flags set.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            required_flags: MemoryPropertyFlags::empty(),
            preferred_flags: MemoryPropertyFlags::empty(),
            not_preferred_flags: MemoryPropertyFlags::empty(),
        }
    }

    /// Returns the union of `self` and `other`.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self {
            required_flags: self.required_flags.union(other.required_flags),
            preferred_flags: self.preferred_flags.union(other.preferred_flags),
            not_preferred_flags: self.not_preferred_flags.union(other.not_preferred_flags),
        }
    }
}

impl BitOr for MemoryTypeFilter {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

/// Parameters to create a new [allocation] using a [memory allocator].
///
/// [allocation]: MemoryAlloc
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub struct AllocationCreateInfo<'a> {
    /// Filter used to narrow down the memory type to be selected.
    ///
    /// The default value is [`MemoryTypeFilter::PREFER_DEVICE`].
    pub memory_type_filter: MemoryTypeFilter,

    /// Allows you to further constrain the possible choices of memory types, by only allowing the
    /// memory type indices that have a corresponding bit at the same index set to 1.
    ///
    /// The default value is [`u32::MAX`].
    pub memory_type_bits: u32,

    /// How eager the allocator should be to allocate [`DeviceMemory`].
    ///
    /// The default value is [`MemoryAllocatePreference::Unknown`].
    pub allocate_preference: MemoryAllocatePreference,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AllocationCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AllocationCreateInfo<'_> {
    /// Returns a default `AllocationCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            memory_type_bits: u32::MAX,
            allocate_preference: MemoryAllocatePreference::Unknown,
            _ne: crate::NE,
        }
    }
}

/// Describes whether allocating [`DeviceMemory`] is desired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryAllocatePreference {
    /// There is no known preference, let the allocator decide.
    Unknown,

    /// The allocator should never allocate `DeviceMemory` and should instead only suballocate from
    /// existing blocks.
    ///
    /// This option is best suited if you can not afford the overhead of allocating `DeviceMemory`.
    NeverAllocate,

    /// The allocator should always allocate `DeviceMemory`.
    ///
    /// This option is best suited if you are allocating a long-lived resource that you know could
    /// benefit from having a dedicated allocation.
    AlwaysAllocate,
}

/// An allocation made using a [memory allocator].
///
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub struct MemoryAlloc {
    /// The underlying block of device memory.
    pub device_memory: Arc<DeviceMemory>,

    /// The suballocation within the device memory block, or [`None`] if this is a dedicated
    /// allocation.
    pub suballocation: Option<Suballocation>,

    /// An opaque handle identifying the allocation inside the allocator.
    pub allocation_handle: AllocationHandle,
}

/// An opaque handle identifying an allocation inside an allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(not(doc), repr(transparent))]
pub struct AllocationHandle(*mut ());

unsafe impl Send for AllocationHandle {}
unsafe impl Sync for AllocationHandle {}

impl AllocationHandle {
    /// Creates a null `AllocationHandle`.
    ///
    /// Use this if you don't have anything that you need to associate with the allocation.
    #[inline]
    pub const fn null() -> Self {
        AllocationHandle(ptr::null_mut())
    }

    /// Stores a pointer in an `AllocationHandle`.
    ///
    /// Use this if you want to associate an allocation with some (host) heap allocation.
    #[inline]
    pub const fn from_ptr(ptr: *mut ()) -> Self {
        AllocationHandle(ptr)
    }

    /// Stores an index inside an `AllocationHandle`.
    ///
    /// Use this if you want to associate an allocation with some index.
    #[inline]
    pub const fn from_index(index: usize) -> Self {
        AllocationHandle(ptr::without_provenance_mut(index))
    }

    /// Retrieves a previously-stored pointer from the `AllocationHandle`.
    ///
    /// If this handle hasn't been created using [`from_ptr`] then this will return an invalid
    /// pointer, dereferencing which is undefined behavior.
    ///
    /// [`from_ptr`]: Self::from_ptr
    #[inline]
    pub const fn as_ptr(self) -> *mut () {
        self.0
    }

    /// Retrieves a previously-stored index from the `AllocationHandle`.
    ///
    /// If this handle hasn't been created using [`from_index`] then this will return a bogus
    /// result.
    ///
    /// [`from_index`]: Self::from_index
    #[inline]
    pub fn as_index(self) -> usize {
        self.0.addr()
    }
}

/// Error that can be returned when creating an [allocation] using a [memory allocator].
///
/// [allocation]: MemoryAlloc
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub enum MemoryAllocatorError {
    /// Allocating [`DeviceMemory`] failed.
    AllocateDeviceMemory(VulkanError),

    /// Finding a suitable memory type failed.
    ///
    /// This is returned from [`MemoryAllocator::try_allocate`] when
    /// [`MemoryAllocator::find_memory_type_index`] returns [`None`].
    FindMemoryType,

    /// There is not enough memory in the pool.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and there is not
    /// enough memory in the pool.
    OutOfPoolMemory,

    /// A dedicated allocation is required but was explicitly forbidden.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and the
    /// implementation requires a dedicated allocation.
    DedicatedAllocationRequired,

    /// The block size for the allocator was exceeded.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and the allocation
    /// size exceeded the block size for all heaps of suitable memory types.
    BlockSizeExceeded,
}

impl Error for MemoryAllocatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::AllocateDeviceMemory(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for MemoryAllocatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let msg = match self {
            Self::AllocateDeviceMemory(_) => "allocating device memory failed",
            Self::FindMemoryType => "finding a suitable memory type failed",
            Self::OutOfPoolMemory => "the pool doesn't have enough free space",
            Self::DedicatedAllocationRequired => {
                "a dedicated allocation is required but was explicitly forbidden"
            }
            Self::BlockSizeExceeded => {
                "the allocation size was greater than the block size for all heaps of suitable \
                memory types and dedicated allocations were explicitly forbidden"
            }
        };

        f.write_str(msg)
    }
}

/// Standard memory allocator intended as a global and general-purpose allocator.
///
/// This type of allocator is what you should always use, unless you know, for a fact, that it is
/// not suited to the task.
///
/// See also [`GenericMemoryAllocator`] for details about the allocation algorithm, and
/// [`TlsfAllocator`] for details about the suballocation algorithm.
pub type StandardMemoryAllocator = GenericMemoryAllocator<TlsfAllocator>;

/// A generic implementation of a [memory allocator].
///
/// The allocator keeps [a pool of `DeviceMemory` blocks] for each memory type and uses the type
/// parameter `S` to [suballocate] these blocks. You can also configure the sizes of these blocks.
/// This means that you can have as many `GenericMemoryAllocator`s as you you want for different
/// needs, or for performance reasons, as long as the block sizes are configured properly so that
/// too much memory isn't wasted.
///
/// See also [the `MemoryAllocator` implementation].
///
/// # Mapping behavior
///
/// Every time a new `DeviceMemory` block is allocated, it is mapped in full automatically as long
/// as it resides in host-visible memory. It remains mapped until it is dropped, which only happens
/// if the allocator is dropped. In other words, all eligible blocks are persistently mapped, so
/// you don't need to worry about whether or not your host-visible allocations are host-accessible.
///
/// # `DeviceMemory` allocation
///
/// If an allocation is created with the [`MemoryAllocatePreference::Unknown`] option, and the
/// allocator deems the allocation too big for suballocation (larger than half the block size), or
/// the implementation prefers or requires a dedicated allocation, then that allocation is made a
/// dedicated allocation. Using [`MemoryAllocatePreference::NeverAllocate`], a dedicated allocation
/// is never created, even if the allocation is larger than the block size or a dedicated
/// allocation is required. In such a case an error is returned instead. Using
/// [`MemoryAllocatePreference::AlwaysAllocate`], a dedicated allocation is always created.
///
/// In all other cases, `DeviceMemory` is only allocated if a pool runs out of memory and needs
/// another block. No `DeviceMemory` is allocated when the allocator is created, the blocks are
/// only allocated once they are needed.
///
/// [memory allocator]: MemoryAllocator
/// [a pool of `DeviceMemory` blocks]: DeviceMemoryPool
/// [suballocate]: Suballocator
/// [the `MemoryAllocator` implementation]: Self#impl-MemoryAllocator-for-GenericMemoryAllocator<S>
#[derive(Debug)]
pub struct GenericMemoryAllocator<S> {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    buffer_image_granularity: DeviceAlignment,
    // Each memory type has a pool of `DeviceMemory` blocks.
    pools: ArrayVec<DeviceMemoryPool<S>, MAX_MEMORY_TYPES>,
    // Global mask of memory types.
    memory_type_bits: u32,
    dedicated_allocation: bool,
    export_handle_types: ArrayVec<ExternalMemoryHandleTypes, MAX_MEMORY_TYPES>,
    flags: MemoryAllocateFlags,
    // How many `DeviceMemory` allocations should be allowed before restricting them.
    max_allocations: u32,
}

impl<S> GenericMemoryAllocator<S> {
    // This is a false-positive, we only use this const for static initialization.
    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY_POOL: DeviceMemoryPool<S> = DeviceMemoryPool {
        blocks: Mutex::new(DeviceMemoryBlockVec::new()),
        property_flags: MemoryPropertyFlags::empty(),
        atom_size: DeviceAlignment::MIN,
        block_size: 0,
    };

    /// Creates a new `GenericMemoryAllocator<S>` using the provided suballocator `S` for
    /// suballocation of [`DeviceMemory`] blocks.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.block_sizes` doesn't contain as many elements as the number of
    ///   memory types.
    /// - Panics if `create_info.export_handle_types` is non-empty and doesn't contain as many
    ///   elements as the number of memory types.
    pub fn new(device: &Arc<Device>, create_info: &GenericMemoryAllocatorCreateInfo<'_>) -> Self {
        let &GenericMemoryAllocatorCreateInfo {
            block_sizes,
            mut memory_type_bits,
            dedicated_allocation,
            export_handle_types,
            mut device_address,
            _ne: _,
        } = create_info;

        let MemoryProperties {
            memory_types,
            memory_heaps,
        } = device.physical_device().memory_properties();

        if !block_sizes.is_empty() {
            assert_eq!(
                block_sizes.len(),
                memory_types.len(),
                "`create_info.block_sizes` must be empty or contain as many elements as the number
                of memory types",
            );
        }

        if !export_handle_types.is_empty() {
            assert_eq!(
                export_handle_types.len(),
                memory_types.len(),
                "`create_info.export_handle_types` must be empty or contain as many elements as \
                the number of memory types",
            );
        }

        let buffer_image_granularity = device
            .physical_device()
            .properties()
            .buffer_image_granularity;

        let memory_types = &device.physical_device().memory_properties().memory_types;

        let mut pools = ArrayVec::new(memory_types.len(), [Self::EMPTY_POOL; MAX_MEMORY_TYPES]);

        for (index, memory_type) in memory_types.iter().enumerate() {
            const LARGE_HEAP_THRESHOLD: DeviceSize = 1024 * 1024 * 1024;

            let property_flags = memory_type.property_flags;
            pools[index].property_flags = property_flags;

            if property_flags.intersects(MemoryPropertyFlags::HOST_VISIBLE)
                && !property_flags.intersects(MemoryPropertyFlags::HOST_COHERENT)
            {
                pools[index].atom_size =
                    device.physical_device().properties().non_coherent_atom_size;
            }

            pools[index].block_size = block_sizes.get(index).copied().unwrap_or_else(|| {
                let heap_size = memory_heaps[memory_type.heap_index as usize].size;

                if heap_size >= LARGE_HEAP_THRESHOLD {
                    256 * 1024 * 1024
                } else {
                    64 * 1024 * 1024
                }
            });
        }

        if memory_type_bits == 0 {
            memory_type_bits = u32::MAX;

            for (index, memory_type) in memory_types.iter().enumerate() {
                if memory_type.property_flags.intersects(
                    MemoryPropertyFlags::LAZILY_ALLOCATED
                        | MemoryPropertyFlags::PROTECTED
                        | MemoryPropertyFlags::DEVICE_COHERENT
                        | MemoryPropertyFlags::RDMA_CAPABLE,
                ) {
                    // VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
                    // VUID-vkAllocateMemory-deviceCoherentMemory-02790
                    // Lazily allocated memory would just cause problems for suballocation in
                    // general.
                    memory_type_bits &= !(1 << index);
                }
            }
        }

        let export_handle_types = {
            let mut types = ArrayVec::new(
                export_handle_types.len(),
                [ExternalMemoryHandleTypes::empty(); MAX_MEMORY_TYPES],
            );
            types.copy_from_slice(export_handle_types);

            types
        };

        // VUID-VkMemoryAllocateInfo-flags-03331
        device_address &= device.enabled_features().buffer_device_address
            && !device.enabled_extensions().ext_buffer_device_address;
        // Providers of `VkMemoryAllocateFlags`
        device_address &=
            device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_device_group;

        let flags = if device_address {
            MemoryAllocateFlags::DEVICE_ADDRESS
        } else {
            MemoryAllocateFlags::empty()
        };

        let max_memory_allocation_count = device
            .physical_device()
            .properties()
            .max_memory_allocation_count;
        let max_allocations = max_memory_allocation_count / 4 * 3;

        GenericMemoryAllocator {
            device: InstanceOwnedDebugWrapper(device.clone()),
            buffer_image_granularity,
            pools,
            dedicated_allocation,
            export_handle_types,
            flags,
            memory_type_bits,
            max_allocations,
        }
    }

    /// Returns the pools of [`DeviceMemory`] blocks that are currently allocated. Each memory type
    /// index has a corresponding element in the slice.
    #[inline]
    pub fn pools(&self) -> &[DeviceMemoryPool<S>] {
        &self.pools
    }

    #[cold]
    fn try_allocate_device_memory(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<Arc<DeviceMemory>, Validated<VulkanError>> {
        let mut memory = DeviceMemory::allocate(
            &self.device,
            &MemoryAllocateInfo {
                allocation_size,
                memory_type_index,
                dedicated_allocation,
                export_handle_types,
                flags: self.flags,
                ..Default::default()
            },
        )?;

        if self.pools[memory_type_index as usize]
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_VISIBLE)
        {
            // SAFETY:
            // - We checked that the memory is host-visible.
            // - The memory can't be mapped already, because we just allocated it.
            // - Mapping the whole range is always valid.
            unsafe { memory.map_unchecked(&MemoryMapInfo::default()) }?;
        }

        Ok(Arc::new(memory))
    }

    #[cold]
    unsafe fn allocate_device_memory_unchecked(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<Arc<DeviceMemory>, VulkanError> {
        // SAFETY: Enforced by the caller.
        let mut memory = unsafe {
            DeviceMemory::allocate_unchecked(
                &self.device,
                &MemoryAllocateInfo {
                    allocation_size,
                    memory_type_index,
                    dedicated_allocation,
                    export_handle_types,
                    flags: self.flags,
                    ..Default::default()
                },
                None,
            )
        }?;

        if self.pools[memory_type_index as usize]
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_VISIBLE)
        {
            // SAFETY:
            // - We checked that the memory is host-visible.
            // - The memory can't be mapped already, because we just allocated it.
            // - Mapping the whole range is always valid.
            unsafe { memory.map_unchecked(&MemoryMapInfo::default()) }?;
        }

        Ok(Arc::new(memory))
    }
}

unsafe impl<S: Suballocator + Send + 'static> MemoryAllocator for GenericMemoryAllocator<S> {
    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        filter: MemoryTypeFilter,
    ) -> Option<u32> {
        let required_flags = filter.required_flags.into();
        let preferred_flags = filter.preferred_flags.into();
        let not_preferred_flags = filter.not_preferred_flags.into();

        self.pools
            .iter()
            .map(|pool| vk::MemoryPropertyFlags::from(pool.property_flags))
            .enumerate()
            // Filter out memory types which are supported by the memory type bits and have the
            // required flags set.
            .filter(|&(index, flags)| {
                memory_type_bits & (1 << index) != 0 && flags & required_flags == required_flags
            })
            // Rank memory types with more of the preferred flags higher, and ones with more of the
            // not preferred flags lower.
            .min_by_key(|&(_, flags)| {
                (!flags & preferred_flags).as_raw().count_ones()
                    + (flags & not_preferred_flags).as_raw().count_ones()
            })
            .map(|(index, _)| index as u32)
    }

    /// Allocates memory from a specific memory type.
    ///
    /// # Arguments
    ///
    /// - `memory_type_index` - The index of the memory type to allocate from.
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `never_allocate` - If `true` then the allocator should never allocate `DeviceMemory`,
    ///   instead only suballocate from existing blocks.
    ///
    /// # Panics
    ///
    /// - Panics if `memory_type_index` is not less than the number of available memory types.
    ///
    /// # Errors
    ///
    /// - Returns [`AllocateDeviceMemory`] if allocating a new block failed.
    /// - Returns [`OutOfPoolMemory`] if `never_allocate` is `true` and the pool doesn't have
    ///   enough free space.
    /// - Returns [`BlockSizeExceeded`] if `create_info.layout.size()` is greater than the block
    ///   size corresponding to the heap that the memory type corresponding to `memory_type_index`
    ///   resides in.
    ///
    /// [`AllocateDeviceMemory`]: MemoryAllocatorError::AllocateDeviceMemory
    /// [`OutOfPoolMemory`]: MemoryAllocatorError::OutOfPoolMemory
    /// [`BlockSizeExceeded`]: MemoryAllocatorError::BlockSizeExceeded
    fn try_allocate_from_type(
        &self,
        memory_type_index: u32,
        mut layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>> {
        let size = layout.size();
        let pool = &self.pools[memory_type_index as usize];

        if size > pool.block_size {
            return Err(Validated::Error(MemoryAllocatorError::BlockSizeExceeded));
        }

        layout = layout.align_to(pool.atom_size).unwrap();

        let blocks = &mut *pool.blocks.lock();
        let vec = &mut blocks.vec;

        // TODO: Incremental sorting
        vec.sort_by_key(|block| block.free_size());
        let (Ok(idx) | Err(idx)) = vec.binary_search_by_key(&size, |block| block.free_size());

        for block in &mut vec[idx..] {
            if let Ok(allocation) =
                block.allocate(layout, allocation_type, self.buffer_image_granularity)
            {
                return Ok(allocation);
            }
        }

        if never_allocate {
            return Err(Validated::Error(MemoryAllocatorError::OutOfPoolMemory));
        }

        // The pool doesn't have enough real estate, so we need a new block.
        let block = {
            let export_handle_types = if !self.export_handle_types.is_empty() {
                self.export_handle_types[memory_type_index as usize]
            } else {
                ExternalMemoryHandleTypes::empty()
            };
            let mut i = 0;

            loop {
                let allocation_size = pool.block_size >> i;

                match self.try_allocate_device_memory(
                    memory_type_index,
                    allocation_size,
                    None,
                    export_handle_types,
                ) {
                    Ok(device_memory) => {
                        break DeviceMemoryBlock::new(device_memory, &blocks.block_allocator);
                    }
                    // Retry up to 3 times, halving the allocation size each time so long as the
                    // resulting size is still large enough.
                    Err(Validated::Error(
                        VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory,
                    )) if i < 3 && pool.block_size >> (i + 1) >= size => {
                        i += 1;
                    }
                    Err(Validated::Error(err)) => {
                        return Err(Validated::Error(
                            MemoryAllocatorError::AllocateDeviceMemory(err),
                        ));
                    }
                    Err(Validated::ValidationError(err)) => {
                        return Err(Validated::ValidationError(err));
                    }
                }
            }
        };

        vec.push(block);
        let block = vec.last_mut().unwrap();

        match block.allocate(layout, allocation_type, self.buffer_image_granularity) {
            Ok(allocation) => Ok(allocation),
            // This can't happen as we always allocate a block of sufficient size.
            Err(SuballocatorError::OutOfRegionMemory) => unreachable!(),
            // This can't happen as the block is fresher than Febreze and we're still holding an
            // exclusive lock.
            Err(SuballocatorError::FragmentedRegion) => unreachable!(),
        }
    }

    /// Allocates memory from a specific memory type without doing any checks.
    ///
    /// <div class="vulkano-alert-caution">
    ///
    /// > Caution
    /// >
    /// > You should only call this when necessary. Humans are famously bad at guessing what needs
    /// > to be optimized, so **the only way** to know is by profiling. Have you profiled to make
    /// > sure that this actually makes a difference? In 99% of cases (not an exaggeration),
    /// > calling the unchecked version of one of our functions/methods makes absolutely no
    /// > difference, so make sure first.
    ///
    /// </div>
    ///
    /// # Arguments
    ///
    /// - `memory_type_index` - The index of the memory type to allocate from.
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `never_allocate` - If `true` then the allocator should never allocate `DeviceMemory`,
    ///   instead only suballocate from existing blocks.
    ///
    /// # Safety
    ///
    /// - If calling [`try_allocate_from_type`] with the same arguments would return a
    ///   [`ValidationError`], calling this method is *undefined behavior*!
    ///
    /// # Panics
    ///
    /// - Panics if `memory_type_index` is not less than the number of available memory types.
    ///
    /// # Errors
    ///
    /// - Returns [`AllocateDeviceMemory`] if allocating a new block failed.
    /// - Returns [`OutOfPoolMemory`] if `never_allocate` is `true` and the pool doesn't have
    ///   enough free space.
    /// - Returns [`BlockSizeExceeded`] if `create_info.layout.size()` is greater than the block
    ///   size corresponding to the heap that the memory type corresponding to `memory_type_index`
    ///   resides in.
    ///
    /// [`try_allocate_from_type`]: Self::try_allocate_from_type
    /// [`AllocateDeviceMemory`]: MemoryAllocatorError::AllocateDeviceMemory
    /// [`OutOfPoolMemory`]: MemoryAllocatorError::OutOfPoolMemory
    /// [`BlockSizeExceeded`]: MemoryAllocatorError::BlockSizeExceeded
    unsafe fn allocate_from_type_unchecked(
        &self,
        memory_type_index: u32,
        mut layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        let size = layout.size();
        let pool = &self.pools[memory_type_index as usize];

        if size > pool.block_size {
            return Err(MemoryAllocatorError::BlockSizeExceeded);
        }

        layout = layout.align_to(pool.atom_size).unwrap();

        let blocks = &mut *pool.blocks.lock();
        let vec = &mut blocks.vec;

        // TODO: Incremental sorting
        vec.sort_by_key(|block| block.free_size());
        let (Ok(idx) | Err(idx)) = vec.binary_search_by_key(&size, |block| block.free_size());

        for block in &mut vec[idx..] {
            if let Ok(allocation) =
                block.allocate(layout, allocation_type, self.buffer_image_granularity)
            {
                return Ok(allocation);
            }
        }

        if never_allocate {
            return Err(MemoryAllocatorError::OutOfPoolMemory);
        }

        // The pool doesn't have enough real estate, so we need a new block.
        let block = {
            let export_handle_types = if !self.export_handle_types.is_empty() {
                self.export_handle_types[memory_type_index as usize]
            } else {
                ExternalMemoryHandleTypes::empty()
            };
            let mut i = 0;

            loop {
                let allocation_size = pool.block_size >> i;

                // SAFETY: Enforced by the caller.
                match unsafe {
                    self.allocate_device_memory_unchecked(
                        memory_type_index,
                        allocation_size,
                        None,
                        export_handle_types,
                    )
                } {
                    Ok(device_memory) => {
                        break DeviceMemoryBlock::new(device_memory, &blocks.block_allocator);
                    }
                    // Retry up to 3 times, halving the allocation size each time so long as the
                    // resulting size is still large enough.
                    Err(VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory)
                        if i < 3 && pool.block_size >> (i + 1) >= size =>
                    {
                        i += 1;
                    }
                    Err(err) => {
                        return Err(MemoryAllocatorError::AllocateDeviceMemory(err));
                    }
                }
            }
        };

        vec.push(block);
        let block = vec.last_mut().unwrap();

        match block.allocate(layout, allocation_type, self.buffer_image_granularity) {
            Ok(allocation) => Ok(allocation),
            // This can't happen as we always allocate a block of sufficient size.
            Err(SuballocatorError::OutOfRegionMemory) => unreachable!(),
            // This can't happen as the block is fresher than Febreze and we're still holding an
            // exclusive lock.
            Err(SuballocatorError::FragmentedRegion) => unreachable!(),
        }
    }

    /// Allocates memory according to requirements.
    ///
    /// # Arguments
    ///
    /// - `requirements` - Requirements of the resource you want to allocate memory for.
    ///
    ///   If you plan to bind this memory directly to a non-sparse resource, then this must
    ///   correspond to the value returned by either [`RawBuffer::memory_requirements`] or
    ///   [`RawImage::memory_requirements`] for the respective buffer or image.
    ///
    /// - `allocation_type` - What type of resource this allocation will be used for.
    ///
    ///   This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    ///   images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    ///   bind memory allocated with the [`NonLinear`] type to buffers and linear images. You
    ///   should never use the [`Unknown`] type unless you have to, as that can be less memory
    ///   efficient.
    ///
    /// - `dedicated_allocation` - Allows a dedicated allocation to be created.
    ///
    ///   You should always fill this field in if you are allocating memory for a non-sparse
    ///   resource, otherwise the allocator won't be able to create a dedicated allocation if one
    ///   is required or recommended.
    ///
    ///   This argument is silently ignored (treated as `None`) if the device API version is below
    ///   1.1 and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// # Errors
    ///
    /// - Returns [`AllocateDeviceMemory`] if allocating a new block failed.
    /// - Returns [`FindMemoryType`] if finding a suitable memory type failed. This can happen if
    ///   the `create_info.requirements` correspond to those of an optimal image but
    ///   `create_info.memory_type_filter` requires host access.
    /// - Returns [`OutOfPoolMemory`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and none of the pools of suitable memory
    ///   types have enough free space.
    /// - Returns [`DedicatedAllocationRequired`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and
    ///   `create_info.requirements.requires_dedicated_allocation` is `true`.
    /// - Returns [`BlockSizeExceeded`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and `create_info.requirements.size` is
    ///   greater than the block size for all heaps of suitable memory types.
    ///
    /// [`RawBuffer::memory_requirements`]: crate::buffer::sys::RawBuffer::memory_requirements
    /// [`RawImage::memory_requirements`]: crate::image::sys::RawImage::memory_requirements
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    /// [`AllocateDeviceMemory`]: MemoryAllocatorError::AllocateDeviceMemory
    /// [`FindMemoryType`]: MemoryAllocatorError::FindMemoryType
    /// [`OutOfPoolMemory`]: MemoryAllocatorError::OutOfPoolMemory
    /// [`DedicatedAllocationRequired`]: MemoryAllocatorError::DedicatedAllocationRequired
    /// [`BlockSizeExceeded`]: MemoryAllocatorError::BlockSizeExceeded
    fn try_allocate(
        &self,
        requirements: &MemoryRequirements,
        allocation_type: AllocationType,
        create_info: &AllocationCreateInfo<'_>,
        mut dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>> {
        let &MemoryRequirements {
            layout,
            mut memory_type_bits,
            mut prefers_dedicated_allocation,
            requires_dedicated_allocation,
        } = requirements;

        memory_type_bits &= self.memory_type_bits;
        memory_type_bits &= create_info.memory_type_bits;

        let &AllocationCreateInfo {
            memory_type_filter,
            memory_type_bits: _,
            allocate_preference,
            _ne: _,
        } = create_info;

        let size = layout.size();

        let mut memory_type_index = self
            .find_memory_type_index(memory_type_bits, memory_type_filter)
            .ok_or(Validated::Error(MemoryAllocatorError::FindMemoryType))?;

        if !self.dedicated_allocation && !requires_dedicated_allocation {
            dedicated_allocation = None;
        }

        let export_handle_types = if self.export_handle_types.is_empty() {
            ExternalMemoryHandleTypes::empty()
        } else {
            self.export_handle_types[memory_type_index as usize]
        };

        loop {
            let pool = &self.pools[memory_type_index as usize];

            let res = match allocate_preference {
                MemoryAllocatePreference::Unknown => {
                    // VUID-vkBindBufferMemory-buffer-01444
                    // VUID-vkBindImageMemory-image-01445
                    if requires_dedicated_allocation {
                        self.try_allocate_dedicated(
                            memory_type_index,
                            size,
                            dedicated_allocation,
                            export_handle_types,
                        )
                    } else {
                        if size > pool.block_size / 2 {
                            prefers_dedicated_allocation = true;
                        }
                        if self.device.allocation_count() > self.max_allocations
                            && size <= pool.block_size
                        {
                            prefers_dedicated_allocation = false;
                        }

                        if prefers_dedicated_allocation {
                            self.try_allocate_dedicated(
                                memory_type_index,
                                size,
                                dedicated_allocation,
                                export_handle_types,
                            )
                            // Fall back to suballocation.
                            .or_else(|err| {
                                self.try_allocate_from_type(
                                    memory_type_index,
                                    layout,
                                    allocation_type,
                                    true, // A dedicated allocation already failed.
                                )
                                .map_err(|_| err)
                            })
                        } else {
                            self.try_allocate_from_type(
                                memory_type_index,
                                layout,
                                allocation_type,
                                false,
                            )
                            // Fall back to dedicated allocation. It is possible that the 1/8
                            // block size tried was greater than the allocation size, so
                            // there's hope.
                            .or_else(|_| {
                                self.try_allocate_dedicated(
                                    memory_type_index,
                                    size,
                                    dedicated_allocation,
                                    export_handle_types,
                                )
                            })
                        }
                    }
                }
                MemoryAllocatePreference::NeverAllocate => {
                    if requires_dedicated_allocation {
                        return Err(Validated::Error(
                            MemoryAllocatorError::DedicatedAllocationRequired,
                        ));
                    }

                    self.try_allocate_from_type(memory_type_index, layout, allocation_type, true)
                }
                MemoryAllocatePreference::AlwaysAllocate => self.try_allocate_dedicated(
                    memory_type_index,
                    size,
                    dedicated_allocation,
                    export_handle_types,
                ),
            };

            match res {
                Ok(allocation) => return Ok(allocation),
                // Try a different memory type.
                Err(err) => {
                    memory_type_bits &= !(1 << memory_type_index);
                    memory_type_index = self
                        .find_memory_type_index(memory_type_bits, memory_type_filter)
                        .ok_or(err)?;
                }
            }
        }
    }

    /// Allocates memory according to requirements without doing any checks.
    ///
    /// <div class="vulkano-alert-caution">
    ///
    /// > Caution
    /// >
    /// > You should only call this when necessary. Humans are famously bad at guessing what needs
    /// > to be optimized, so **the only way** to know is by profiling. Have you profiled to make
    /// > sure that this actually makes a difference? In 99% of cases (not an exaggeration),
    /// > calling the unchecked version of one of our functions/methods makes absolutely no
    /// > difference, so make sure first.
    ///
    /// </div>
    ///
    /// # Arguments
    ///
    /// - `requirements` - Requirements of the resource you want to allocate memory for.
    ///
    ///   If you plan to bind this memory directly to a non-sparse resource, then this must
    ///   correspond to the value returned by either [`RawBuffer::memory_requirements`] or
    ///   [`RawImage::memory_requirements`] for the respective buffer or image.
    ///
    /// - `allocation_type` - What type of resource this allocation will be used for.
    ///
    ///   This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    ///   images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    ///   bind memory allocated with the [`NonLinear`] type to buffers and linear images. You
    ///   should never use the [`Unknown`] type unless you have to, as that can be less memory
    ///   efficient.
    ///
    /// - `dedicated_allocation` - Allows a dedicated allocation to be created.
    ///
    ///   You should always fill this field in if you are allocating memory for a non-sparse
    ///   resource, otherwise the allocator won't be able to create a dedicated allocation if one
    ///   is required or recommended.
    ///
    ///   This argument is silently ignored (treated as `None`) if the device API version is below
    ///   1.1 and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// # Safety
    ///
    /// - If calling [`try_allocate`] with the same arguments would return a [`ValidationError`],
    ///   calling this method is *undefined behavior*!
    ///
    /// # Errors
    ///
    /// - Returns [`AllocateDeviceMemory`] if allocating a new block failed.
    /// - Returns [`FindMemoryType`] if finding a suitable memory type failed. This can happen if
    ///   the `create_info.requirements` correspond to those of an optimal image but
    ///   `create_info.memory_type_filter` requires host access.
    /// - Returns [`OutOfPoolMemory`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and none of the pools of suitable memory
    ///   types have enough free space.
    /// - Returns [`DedicatedAllocationRequired`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and
    ///   `create_info.requirements.requires_dedicated_allocation` is `true`.
    /// - Returns [`BlockSizeExceeded`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and `create_info.requirements.size` is
    ///   greater than the block size for all heaps of suitable memory types.
    ///
    /// [`RawBuffer::memory_requirements`]: crate::buffer::sys::RawBuffer::memory_requirements
    /// [`RawImage::memory_requirements`]: crate::image::sys::RawImage::memory_requirements
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    /// [`try_allocate`]: Self::try_allocate
    /// [`AllocateDeviceMemory`]: MemoryAllocatorError::AllocateDeviceMemory
    /// [`FindMemoryType`]: MemoryAllocatorError::FindMemoryType
    /// [`OutOfPoolMemory`]: MemoryAllocatorError::OutOfPoolMemory
    /// [`DedicatedAllocationRequired`]: MemoryAllocatorError::DedicatedAllocationRequired
    /// [`BlockSizeExceeded`]: MemoryAllocatorError::BlockSizeExceeded
    unsafe fn allocate_unchecked(
        &self,
        requirements: &MemoryRequirements,
        allocation_type: AllocationType,
        create_info: &AllocationCreateInfo<'_>,
        mut dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        let &MemoryRequirements {
            layout,
            mut memory_type_bits,
            mut prefers_dedicated_allocation,
            requires_dedicated_allocation,
        } = requirements;

        memory_type_bits &= self.memory_type_bits;
        memory_type_bits &= create_info.memory_type_bits;

        let &AllocationCreateInfo {
            memory_type_filter,
            memory_type_bits: _,
            allocate_preference,
            _ne: _,
        } = create_info;

        let size = layout.size();

        let mut memory_type_index = self
            .find_memory_type_index(memory_type_bits, memory_type_filter)
            .ok_or(MemoryAllocatorError::FindMemoryType)?;

        if !self.dedicated_allocation && !requires_dedicated_allocation {
            dedicated_allocation = None;
        }

        let export_handle_types = if self.export_handle_types.is_empty() {
            ExternalMemoryHandleTypes::empty()
        } else {
            self.export_handle_types[memory_type_index as usize]
        };

        loop {
            let pool = &self.pools[memory_type_index as usize];

            let res = match allocate_preference {
                MemoryAllocatePreference::Unknown => {
                    // VUID-vkBindBufferMemory-buffer-01444
                    // VUID-vkBindImageMemory-image-01445
                    if requires_dedicated_allocation {
                        // SAFETY: Enforced by the caller.
                        unsafe {
                            self.allocate_dedicated_unchecked(
                                memory_type_index,
                                size,
                                dedicated_allocation,
                                export_handle_types,
                            )
                        }
                    } else {
                        if size > pool.block_size / 2 {
                            prefers_dedicated_allocation = true;
                        }
                        if self.device.allocation_count() > self.max_allocations
                            && size <= pool.block_size
                        {
                            prefers_dedicated_allocation = false;
                        }

                        if prefers_dedicated_allocation {
                            // SAFETY: Enforced by the caller.
                            unsafe {
                                self.allocate_dedicated_unchecked(
                                    memory_type_index,
                                    size,
                                    dedicated_allocation,
                                    export_handle_types,
                                )
                            }
                            // Fall back to suballocation.
                            .or_else(|err| {
                                // SAFETY: Enforced by the caller.
                                unsafe {
                                    self.allocate_from_type_unchecked(
                                        memory_type_index,
                                        layout,
                                        allocation_type,
                                        true, // A dedicated allocation already failed.
                                    )
                                }
                                .map_err(|_| err)
                            })
                        } else {
                            // SAFETY: Enforced by the caller.
                            unsafe {
                                self.allocate_from_type_unchecked(
                                    memory_type_index,
                                    layout,
                                    allocation_type,
                                    false,
                                )
                            }
                            // Fall back to dedicated allocation. It is possible that the 1/8
                            // block size tried was greater than the allocation size, so
                            // there's hope.
                            .or_else(|_| {
                                // SAFETY: Enforced by the caller.
                                unsafe {
                                    self.allocate_dedicated_unchecked(
                                        memory_type_index,
                                        size,
                                        dedicated_allocation,
                                        export_handle_types,
                                    )
                                }
                            })
                        }
                    }
                }
                MemoryAllocatePreference::NeverAllocate => {
                    if requires_dedicated_allocation {
                        return Err(MemoryAllocatorError::DedicatedAllocationRequired);
                    }

                    unsafe {
                        self.allocate_from_type_unchecked(
                            memory_type_index,
                            layout,
                            allocation_type,
                            true,
                        )
                    }
                }
                MemoryAllocatePreference::AlwaysAllocate => unsafe {
                    self.allocate_dedicated_unchecked(
                        memory_type_index,
                        size,
                        dedicated_allocation,
                        export_handle_types,
                    )
                },
            };

            match res {
                Ok(allocation) => return Ok(allocation),
                // Try a different memory type.
                Err(err) => {
                    memory_type_bits &= !(1 << memory_type_index);
                    memory_type_index = self
                        .find_memory_type_index(memory_type_bits, memory_type_filter)
                        .ok_or(err)?;
                }
            }
        }
    }

    #[cold]
    fn try_allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>> {
        let device_memory = self
            .try_allocate_device_memory(
                memory_type_index,
                allocation_size,
                dedicated_allocation,
                export_handle_types,
            )
            .map_err(|err| err.map(MemoryAllocatorError::AllocateDeviceMemory))?;

        Ok(MemoryAlloc {
            device_memory,
            suballocation: None,
            allocation_handle: AllocationHandle::null(),
        })
    }

    #[cold]
    unsafe fn allocate_dedicated_unchecked(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        let device_memory = unsafe {
            self.allocate_device_memory_unchecked(
                memory_type_index,
                allocation_size,
                dedicated_allocation,
                export_handle_types,
            )
        }
        .map_err(MemoryAllocatorError::AllocateDeviceMemory)?;

        Ok(MemoryAlloc {
            device_memory,
            suballocation: None,
            allocation_handle: AllocationHandle::null(),
        })
    }

    unsafe fn deallocate(&self, allocation: MemoryAlloc) {
        if let Some(suballocation) = allocation.suballocation {
            let memory_type_index = allocation.device_memory.memory_type_index();
            let blocks = self.pools[memory_type_index as usize].blocks.lock();
            let vec = &blocks.vec;
            let block_ptr = allocation
                .allocation_handle
                .as_ptr()
                .cast::<DeviceMemoryBlock<S>>();

            // TODO: Maybe do a similar check for dedicated blocks.
            debug_assert!(
                vec.iter().any(|block| &raw const **block == block_ptr),
                "attempted to deallocate a memory block that does not belong to this allocator",
            );

            // SAFETY: The caller must guarantee that `allocation` refers to one allocated by
            // `self`, therefore `block_ptr` must be the same one we gave out on allocation. We
            // know that this pointer must be valid, because all blocks are boxed and pinned in
            // memory and because a block isn't dropped until the allocator itself is dropped, at
            // which point it would be impossible to call this method. We also know that it must be
            // valid to create a reference to the block, because we locked the pool it belongs to.
            let block = unsafe { &mut *block_ptr };

            // SAFETY: The caller must guarantee that `allocation` refers to a currently allocated
            // allocation of `self`.
            unsafe { block.deallocate(suballocation) };
        }
    }

    fn as_dyn(self: Arc<Self>) -> Arc<dyn MemoryAllocator> {
        self
    }
}

unsafe impl<T: MemoryAllocator + ?Sized> MemoryAllocator for Arc<T> {
    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        filter: MemoryTypeFilter,
    ) -> Option<u32> {
        (**self).find_memory_type_index(memory_type_bits, filter)
    }

    fn try_allocate_from_type(
        &self,
        memory_type_index: u32,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>> {
        (**self).try_allocate_from_type(memory_type_index, layout, allocation_type, never_allocate)
    }

    unsafe fn allocate_from_type_unchecked(
        &self,
        memory_type_index: u32,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        unsafe {
            (**self).allocate_from_type_unchecked(
                memory_type_index,
                layout,
                allocation_type,
                never_allocate,
            )
        }
    }

    fn try_allocate(
        &self,
        requirements: &MemoryRequirements,
        allocation_type: AllocationType,
        create_info: &AllocationCreateInfo<'_>,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>> {
        (**self).try_allocate(
            requirements,
            allocation_type,
            create_info,
            dedicated_allocation,
        )
    }

    unsafe fn allocate_unchecked(
        &self,
        requirements: &MemoryRequirements,
        allocation_type: AllocationType,
        create_info: &AllocationCreateInfo<'_>,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        unsafe {
            (**self).allocate_unchecked(
                requirements,
                allocation_type,
                create_info,
                dedicated_allocation,
            )
        }
    }

    fn try_allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, Validated<MemoryAllocatorError>> {
        (**self).try_allocate_dedicated(
            memory_type_index,
            allocation_size,
            dedicated_allocation,
            export_handle_types,
        )
    }

    unsafe fn allocate_dedicated_unchecked(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        unsafe {
            (**self).allocate_dedicated_unchecked(
                memory_type_index,
                allocation_size,
                dedicated_allocation,
                export_handle_types,
            )
        }
    }

    unsafe fn deallocate(&self, allocation: MemoryAlloc) {
        unsafe { (**self).deallocate(allocation) }
    }

    fn as_dyn(self: Arc<Self>) -> Arc<dyn MemoryAllocator> {
        self
    }
}

unsafe impl<S> DeviceOwned for GenericMemoryAllocator<S> {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// A pool of [`DeviceMemory`] blocks within [`GenericMemoryAllocator`], specific to a memory type.
#[derive(Debug)]
pub struct DeviceMemoryPool<S> {
    blocks: Mutex<DeviceMemoryBlockVec<S>>,
    // This is cached here for faster access, so we don't need to hop through 3 pointers.
    property_flags: MemoryPropertyFlags,
    atom_size: DeviceAlignment,
    block_size: DeviceSize,
}

impl<S> DeviceMemoryPool<S> {
    /// Returns an iterator over the [`DeviceMemory`] blocks in the pool.
    ///
    /// # Locking behavior
    ///
    /// This locks the pool for the lifetime of the returned iterator. Creating other iterators, or
    /// allocating/deallocating in the meantime, can therefore lead to a deadlock as long as this
    /// iterator isn't dropped.
    #[inline]
    pub fn blocks(&self) -> DeviceMemoryBlocks<'_, S> {
        DeviceMemoryBlocks {
            inner: MutexGuard::leak(self.blocks.lock()).vec.iter(),
            // SAFETY: We have just locked the pool above.
            _guard: unsafe { DeviceMemoryPoolGuard::new(self) },
        }
    }
}

impl<S> Drop for DeviceMemoryPool<S> {
    fn drop(&mut self) {
        let blocks = self.blocks.get_mut();

        for block in &mut blocks.vec {
            unsafe { AliasableBox::drop(block, &blocks.block_allocator) };
        }
    }
}

#[derive(Debug)]
struct DeviceMemoryBlockVec<S> {
    vec: Vec<AliasableBox<DeviceMemoryBlock<S>>>,
    block_allocator: SlabAllocator<DeviceMemoryBlock<S>>,
}

impl<S> DeviceMemoryBlockVec<S> {
    const fn new() -> Self {
        DeviceMemoryBlockVec {
            vec: Vec::new(),
            block_allocator: SlabAllocator::new(32),
        }
    }
}

/// A [`DeviceMemory`] block within a [`DeviceMemoryPool`].
#[derive(Debug)]
pub struct DeviceMemoryBlock<S> {
    device_memory: Arc<DeviceMemory>,
    suballocator: S,
    allocation_count: usize,
}

impl<S: Suballocator> DeviceMemoryBlock<S> {
    fn new(
        device_memory: Arc<DeviceMemory>,
        block_allocator: &SlabAllocator<Self>,
    ) -> AliasableBox<Self> {
        let suballocator = S::new(
            Region::new(0, device_memory.allocation_size())
                .expect("we somehow managed to allocate more than `DeviceLayout::MAX_SIZE` bytes"),
        );

        AliasableBox::new(
            DeviceMemoryBlock {
                device_memory,
                suballocator,
                allocation_count: 0,
            },
            block_allocator,
        )
    }

    unsafe fn deallocate(&mut self, suballocation: Suballocation) {
        unsafe { self.suballocator.deallocate(suballocation) };

        self.allocation_count -= 1;

        // For bump allocators, reset the free-start once there are no remaining allocations.
        if self.allocation_count == 0 {
            self.suballocator.reset();
        }
    }

    /// Returns the [`DeviceMemory`] backing this block.
    #[inline]
    pub fn device_memory(&self) -> &Arc<DeviceMemory> {
        &self.device_memory
    }

    /// Returns the suballocator used to suballocate this block.
    #[inline]
    pub fn suballocator(&self) -> &S {
        &self.suballocator
    }

    /// Returns the number of suballocations currently allocated from the block.
    #[inline]
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    fn free_size(&self) -> DeviceSize {
        self.suballocator.free_size()
    }
}

impl<S: Suballocator> AliasableBox<DeviceMemoryBlock<S>> {
    fn allocate(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<MemoryAlloc, SuballocatorError> {
        let suballocation =
            self.suballocator
                .allocate(layout, allocation_type, buffer_image_granularity)?;

        self.allocation_count += 1;

        // It is paramount to soundness that the pointer we give out doesn't go through `DerefMut`,
        // as such a pointer would become invalidated when another allocation is made.
        let ptr = AliasableBox::as_mut_ptr(self);

        Ok(MemoryAlloc {
            device_memory: self.device_memory.clone(),
            suballocation: Some(suballocation),
            allocation_handle: AllocationHandle::from_ptr(<*mut _>::cast(ptr)),
        })
    }
}

/// An iterator over the [`DeviceMemoryBlock`]s within a [`DeviceMemoryPool`].
pub struct DeviceMemoryBlocks<'a, S> {
    inner: slice::Iter<'a, AliasableBox<DeviceMemoryBlock<S>>>,
    _guard: DeviceMemoryPoolGuard<'a, S>,
}

impl<'a, S> Iterator for DeviceMemoryBlocks<'a, S> {
    type Item = &'a DeviceMemoryBlock<S>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|block| &**block)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.inner.count()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.inner.next_back().map(|block| &**block)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(|block| &**block)
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, |acc, block| f(acc, &**block))
    }
}

impl<S> DoubleEndedIterator for DeviceMemoryBlocks<'_, S> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|block| &**block)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth_back(n).map(|block| &**block)
    }
}

impl<S> ExactSizeIterator for DeviceMemoryBlocks<'_, S> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<S> FusedIterator for DeviceMemoryBlocks<'_, S> {}

struct DeviceMemoryPoolGuard<'a, S> {
    pool: &'a DeviceMemoryPool<S>,
}

impl<'a, S> DeviceMemoryPoolGuard<'a, S> {
    /// # Safety
    ///
    /// - The `pool.blocks` mutex must be locked.
    unsafe fn new(pool: &'a DeviceMemoryPool<S>) -> Self {
        DeviceMemoryPoolGuard { pool }
    }
}

impl<S> Drop for DeviceMemoryPoolGuard<'_, S> {
    fn drop(&mut self) {
        // SAFETY: Enforced by the caller of `DeviceMemoryPoolGuard::new`.
        unsafe { self.pool.blocks.force_unlock() };
    }
}

/// Parameters to create a new [`GenericMemoryAllocator`].
#[derive(Clone, Debug)]
pub struct GenericMemoryAllocatorCreateInfo<'a> {
    /// Lets you configure the block sizes for each memory type.
    ///
    /// Must be either empty or contain one entry for each memory type. The allocator keeps a pool
    /// of [`DeviceMemory`] blocks for each memory type, and every time a new block is allocated,
    /// the block size corresponding to the memory type index is looked up here and used for the
    /// allocation.
    ///
    /// The block size is going to be the maximum size of a `DeviceMemory` block that is tried. If
    /// allocating a block with the size fails, the allocator tries 1/2, 1/4 and 1/8 of the block
    /// size in that order until one succeeds, else a dedicated allocation is attempted for the
    /// allocation. If an allocation is created with a size greater than half the block size it is
    /// always made a dedicated allocation. All of this doesn't apply when using
    /// [`MemoryAllocatePreference::NeverAllocate`] however.
    ///
    /// If you leave the default, block sizes are selected automatically.
    ///
    /// The default value is empty.
    pub block_sizes: &'a [DeviceSize],

    /// Lets you configure the allocator's global mask of memory type indices. Only the memory type
    /// indices that have a corresponding bit at the same index set will be allocated from when
    /// calling [`allocate`], otherwise [`MemoryAllocatorError::FindMemoryType`] is returned.
    ///
    /// You may use this to disallow problematic memory types, for instance ones with the
    /// [`PROTECTED`] flag, or any other flags you don't want.
    ///
    /// If you leave the default, unsuited memory types are excluded automatically.
    ///
    /// The default value is `0`.
    ///
    /// [`allocate`]: struct.GenericMemoryAllocator.html#method.allocate
    /// [`PROTECTED`]: MemoryPropertyFlags::PROTECTED
    pub memory_type_bits: u32,

    /// Whether the allocator should use the dedicated allocation APIs.
    ///
    /// This means that when the allocator decides that an allocation should not be suballocated,
    /// but rather have its own block of [`DeviceMemory`], that that allocation will be made a
    /// dedicated allocation. Otherwise they are still given their own block of device memory, just
    /// that that block won't be [dedicated] to a particular resource.
    ///
    /// Dedicated allocations are an optimization which may result in better performance, so there
    /// really is no reason to disable this option, unless the restrictions that they bring with
    /// them are a problem. Namely, a dedicated allocation must only be used for the resource it
    /// was created for. Meaning that reusing the memory for something else is not possible,
    /// suballocating it is not possible, and aliasing it is also not possible.
    ///
    /// This option is silently ignored (treated as `false`) if the device API version is below 1.1
    /// and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// The default value is `true`.
    ///
    /// [dedicated]: DeviceMemory::is_dedicated
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    pub dedicated_allocation: bool,

    /// Lets you configure the external memory handle types that the [`DeviceMemory`] blocks will
    /// be allocated with.
    ///
    /// Must be either empty or contain one element for each memory type. When `DeviceMemory` is
    /// allocated, the external handle types corresponding to the memory type index are looked up
    /// here and used for the allocation.
    ///
    /// The default value is empty.
    pub export_handle_types: &'a [ExternalMemoryHandleTypes],

    /// Whether the allocator should allocate the [`DeviceMemory`] blocks with the
    /// [`DEVICE_ADDRESS`] flag set.
    ///
    /// This is required if you want to allocate memory for buffers that have the
    /// [`SHADER_DEVICE_ADDRESS`] usage set. For this option too, there is no reason to disable it.
    ///
    /// This option is silently ignored (treated as `false`) if the [`buffer_device_address`]
    /// feature is not enabled on the device or if the [`ext_buffer_device_address`] extension is
    /// enabled on the device. It is also ignored if the device API version is below 1.1 and the
    /// [`khr_device_group`] extension is not enabled on the device.
    ///
    /// The default value is `true`.
    ///
    /// [`DEVICE_ADDRESS`]: MemoryAllocateFlags::DEVICE_ADDRESS
    /// [`SHADER_DEVICE_ADDRESS`]: crate::buffer::BufferUsage::SHADER_DEVICE_ADDRESS
    /// [`buffer_device_address`]: crate::device::DeviceFeatures::buffer_device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    pub device_address: bool,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for GenericMemoryAllocatorCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl GenericMemoryAllocatorCreateInfo<'_> {
    /// Returns a default `GenericMemoryAllocatorCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            block_sizes: &[],
            memory_type_bits: u32::MAX,
            dedicated_allocation: true,
            export_handle_types: &[],
            device_address: true,
            _ne: crate::NE,
        }
    }
}

/// Returns the smallest value greater or equal to `val` that is a multiple of `alignment`.
///
/// <div class="vulkano-alert-note">
///
/// > Note
/// >
/// > Returns zero on overflow.
///
/// </div>
#[inline(always)]
pub const fn align_up(val: DeviceSize, alignment: DeviceAlignment) -> DeviceSize {
    align_down(val.wrapping_add(alignment.as_devicesize() - 1), alignment)
}

/// Returns the largest value smaller or equal to `val` that is a multiple of `alignment`.
#[inline(always)]
pub const fn align_down(val: DeviceSize, alignment: DeviceAlignment) -> DeviceSize {
    val & !(alignment.as_devicesize() - 1)
}

mod array_vec {
    use std::{
        fmt,
        ops::{Deref, DerefMut},
    };

    /// Minimal implementation of an `ArrayVec`. Useful when a `Vec` is needed but there is a known
    /// limit on the number of elements, so that it can occupy real estate on the stack.
    #[derive(Clone, Copy)]
    pub(super) struct ArrayVec<T, const N: usize> {
        len: usize,
        data: [T; N],
    }

    impl<T, const N: usize> ArrayVec<T, N> {
        pub fn new(len: usize, data: [T; N]) -> Self {
            assert!(len <= N);

            ArrayVec { len, data }
        }
    }

    impl<T: fmt::Debug, const N: usize> fmt::Debug for ArrayVec<T, N> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&**self, f)
        }
    }

    impl<T, const N: usize> Deref for ArrayVec<T, N> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            // SAFETY: `self.len <= N`.
            unsafe { self.data.get_unchecked(0..self.len) }
        }
    }

    impl<T, const N: usize> DerefMut for ArrayVec<T, N> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            // SAFETY: `self.len <= N`.
            unsafe { self.data.get_unchecked_mut(0..self.len) }
        }
    }
}

pub(crate) mod aliasable_box {
    use slabbin::SlabAllocator;
    use std::{
        fmt,
        marker::PhantomData,
        ops::{Deref, DerefMut},
        panic::{RefUnwindSafe, UnwindSafe},
        ptr::NonNull,
    };

    pub struct AliasableBox<T> {
        ptr: NonNull<T>,
        marker: PhantomData<T>,
    }

    unsafe impl<T: Send> Send for AliasableBox<T> {}
    unsafe impl<T: Sync> Sync for AliasableBox<T> {}

    impl<T: UnwindSafe> UnwindSafe for AliasableBox<T> {}
    impl<T: RefUnwindSafe> RefUnwindSafe for AliasableBox<T> {}

    impl<T> Unpin for AliasableBox<T> {}

    impl<T> AliasableBox<T> {
        pub fn new(value: T, allocator: &SlabAllocator<T>) -> Self {
            let ptr = allocator.allocate();

            unsafe { ptr.as_ptr().write(value) };

            AliasableBox {
                ptr,
                marker: PhantomData,
            }
        }

        pub fn as_mut_ptr(this: &mut Self) -> *mut T {
            this.ptr.as_ptr()
        }

        pub unsafe fn drop(this: &mut Self, allocator: &SlabAllocator<T>) {
            unsafe { this.ptr.as_ptr().drop_in_place() };
            unsafe { allocator.deallocate(this.ptr) };
        }
    }

    impl<T: fmt::Debug> fmt::Debug for AliasableBox<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&**self, f)
        }
    }

    impl<T> Deref for AliasableBox<T> {
        type Target = T;

        #[inline]
        fn deref(&self) -> &Self::Target {
            unsafe { self.ptr.as_ref() }
        }
    }

    impl<T> DerefMut for AliasableBox<T> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { self.ptr.as_mut() }
        }
    }
}
