//! Suballocators are used to divide a *region* into smaller *suballocations*.
//!
//! See also [the parent module] for details about memory allocation in Vulkan.
//!
//! [the parent module]: super

pub use self::{
    buddy::BuddyAllocator, bump::BumpAllocator, free_list::FreeListAllocator, region::Region,
    tlsf::TlsfAllocator,
};
use super::{align_down, AllocationHandle, DeviceAlignment, DeviceLayout};
use crate::{image::ImageTiling, DeviceSize};
use std::{
    error::Error,
    fmt::{self, Debug, Display},
    ops::Range,
};

mod buddy;
mod bump;
mod free_list;
mod tlsf;

/// Suballocators are used to divide a *region* into smaller *suballocations*.
///
/// # Regions
///
/// As the name implies, a region is a contiguous portion of memory. It may be the whole dedicated
/// block of [`DeviceMemory`], or only a part of it. Or it may be a buffer, or only a part of a
/// buffer. Regions are just allocations like any other, but we use this term to refer specifically
/// to an allocation that is to be suballocated. Every suballocator is created with a region to
/// work with.
///
/// # Free-lists
///
/// A free-list, also kind of predictably, refers to a list of (sub)allocations within a region
/// that are currently free. Every (sub)allocator that can free allocations dynamically (in any
/// order) needs to keep a free-list of some sort. This list is then consulted when new allocations
/// are made, and can be used to coalesce neighboring allocations that are free into bigger ones.
///
/// # Memory hierarchies
///
/// Different applications have wildly different allocation needs, and there's no way to cover them
/// all with a single type of allocator. Furthermore, different allocators have different
/// trade-offs and are best suited to specific tasks. To account for all possible use-cases,
/// Vulkano offers the ability to create *memory hierarchies*. We refer to the `DeviceMemory` as
/// the root of any such hierarchy, even though technically the driver has levels that are further
/// up, because those `DeviceMemory` blocks need to be allocated from physical memory pages
/// themselves, but since those levels are not accessible to us we don't need to consider them. You
/// can create any number of levels/branches from there, bounded only by the amount of available
/// memory within a `DeviceMemory` block. You can suballocate the root into regions, which are then
/// suballocated into further regions and so on, creating hierarchies of arbitrary height.
///
/// # Examples
///
/// The suballocator API was designed to be completely agnostic to what it's suballocating, such
/// that it can suballocate anything from bytes to array elements as well as suballocations
/// thereof. Here are some examples of common patterns.
///
/// #### Suballocating bytes
///
/// Allocating a lot of small buffers is inefficient and should be avoided. You can instead use a
/// suballocator to suballocate one buffer into smaller ones. Note that while we use a buffer here
/// as an example, as mentioned above, the suballocator is completely agnostic to what it's
/// suballocating, so you could instead be suballocating any other kind of buffer, including a
/// non-Vulkan one.
///
/// No Vulkan implementation actually cares about [`BufferUsage`] flags. Therefore, using one
/// buffer per usage flag is wasteful. You should instead allocate everything you can in one buffer
/// for optimal performance. The only exceptions are buffer usage flags such as [`UNIFORM_BUFFER`],
/// which has stricter limits, and [`SHADER_DEVICE_ADDRESS`], [`ACCELERATION_STRUCTURE_STORAGE`],
/// and [`SHADER_BINDING_TABLE`], which may influence the available memory types.
///
/// ```
/// use vulkano::{
///     buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
///     memory::{
///         allocator::{
///             suballocator::Region, AllocationCreateInfo, AllocationType, DeviceLayout,
///             Suballocator, TlsfAllocator,
///         },
///         DeviceAlignment,
///     },
/// };
///
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// // Allocate a byte buffer with all the flags we'll need. Note the alignment of 4; this must be
/// // the maximum alignment that any suballocation can have.
/// let buffer = Subbuffer::from(
///     Buffer::new(
///         &memory_allocator,
///         &BufferCreateInfo {
///             usage: BufferUsage::STORAGE_BUFFER
///                 | BufferUsage::INDEX_BUFFER
///                 | BufferUsage::VERTEX_BUFFER,
///             ..Default::default()
///         },
///         &AllocationCreateInfo::default(),
///         DeviceLayout::from_size_alignment(16 * 1024 * 1024, 4).unwrap(),
///     )
///     .unwrap(),
/// );
///
/// // Since we want to suballocate the whole buffer, the region is created to match the buffer.
/// let allocator = TlsfAllocator::new(Region::new(0, buffer.len()).unwrap());
///
/// // We can then allocate whatever type of data we need and reinterpret the bytes to that type.
/// # let index_count = return;
/// let index_buffer_allocation = allocator
///     .allocate_buffer(DeviceLayout::new_unsized::<[u32]>(index_count).unwrap())
///     .unwrap();
/// let index_buffer = buffer
///     .slice(index_buffer_allocation.as_range())
///     .reinterpret::<[u32]>();
///
/// # type MyVertex = u32;
/// #
/// # let vertex_count = return;
/// let vertex_buffer_allocation = allocator
///     .allocate_buffer(DeviceLayout::new_unsized::<[MyVertex]>(vertex_count).unwrap())
///     .unwrap();
/// let vertex_buffer = buffer
///     .slice(vertex_buffer_allocation.as_range())
///     .reinterpret::<[MyVertex]>();
///
/// // ...use the allocations...
///
/// // Once the allocations are no longer in use, you should deallocate them or reset the allocator
/// // (if you know that no other allocations exist). That means that if you use the allocations in
/// // a command buffer, you must only deallocate them after the command buffer is no longer being
/// // executed. The best way to do this is to keep some frame-local data, where each frame in
/// // flight has its own data associated with it. At the beginning of each frame, after waiting
/// // for the frame to finish, you deallocate everything associated with that frame.
/// unsafe { allocator.deallocate(index_buffer_allocation) };
/// unsafe { allocator.deallocate(vertex_buffer_allocation) };
/// ```
///
/// #### Suballocating elements
///
/// Suballocating bytes like in the first example allows you to put multiple types of data into the
/// same buffer. However, if you have a buffer that contains nothing but an array, you can also
/// suballocate the array's elements instead of bytes. This is very useful for indirect draws,
/// where you are forced to specify offsets and sizes in elements rather than bytes. Again, while
/// we use a buffer here as an example, you could be suballocating any array, including a
/// non-Vulkan one.
///
/// ```no_run
/// use vulkano::{
///     buffer::Subbuffer,
///     memory::{
///         allocator::{
///             suballocator::Region, AllocationType, DeviceLayout, Suballocator, TlsfAllocator,
///         },
///         DeviceAlignment,
///     },
/// };
///
/// # type MyVertex = u32;
/// #
/// # fn allocate_vertex_buffer() -> Subbuffer<[MyVertex]> {
/// #     unimplemented!()
/// # }
/// #
/// // This could be its own buffer or a suballocation like in the first example.
/// let vertex_buffer: Subbuffer<[MyVertex]> = allocate_vertex_buffer();
///
/// // Since we want to suballocate the whole buffer, the region is created to match the buffer.
/// // Note that unlike in the first example, this time the region is not in bytes.
/// let allocator = TlsfAllocator::new(Region::new(0, vertex_buffer.len()).unwrap());
///
/// // We can then allocate some elements. Note that unlike in the first example, we have to use
/// // `DeviceLayout::from_size_alignment` directly. This is because our layout is in elements
/// // rather than bytes. The other constructors of `DeviceLayout` exist to create a layout in
/// // bytes from type information and element counts. Since we need elements and our
/// // `vertex_count` is in elements, we use that as the size. Note also that the alignment loses
/// // its meaning because it's also in elements. An alignment of, say, 4 is not going to create an
/// // allocation that is aligned to 4 bytes. Therefore, using any other alignment than 1 is
/// // nonsensical.
/// # let vertex_count = return;
/// let allocation = allocator
///     .allocate_buffer(DeviceLayout::from_size_alignment(vertex_count, 1).unwrap())
///     .unwrap();
/// let vertex_subbuffer = vertex_buffer.slice(allocation.as_range());
///
/// // ...use the allocations...
///
/// // Once the allocations are no longer in use, you should deallocate them or reset the allocator
/// // (if you know that no other allocations exist). That means that if you use the allocations in
/// // a command buffer, you must only deallocate them after the command buffer is no longer being
/// // executed. The best way to do this is to keep some frame-local data, where each frame in
/// // flight has its own data associated with it. At the beginning of each frame, after waiting
/// // for the frame to finish, you deallocate everything associated with that frame.
/// unsafe { allocator.deallocate(allocation) };
/// ```
///
/// #### Suballocating relative to a parent allocation
///
/// There are two ways to suballocate a suballocation. One way is to create a region that matches
/// the suballocation to be suballocated; that is, with an offset of 0 and size equal to its size.
/// This is usually the way to go and what the other examples do. However, when you need your
/// offsets to be relative to some parent of the suballocation to be suballocated, you can instead
/// use a region with an offset that of its offset from the parent. This way, all allocation
/// offsets will be relative to the parent, but still in bounds of the child.
///
/// One use case for this is to ensure an allocation is aligned relative to the `DeviceMemory`
/// block. When you allocate a buffer using a memory allocator, it's going to be aligned according
/// to the alignment used when allocating it. Suballocating this buffer with an alignment greater
/// than its alignment is a mistake. This is because the suballocation is going to be aligned to at
/// most the alignment of the buffer even if you allocate the suballocation with a greater
/// alignment. The simplest way to fix this is to ensure that the buffer is aligned to the maximum
/// (or greater) alignment any suballocation can have. However, if you don't know what alignment
/// that is, you can use the approach in this example. Note that aligning relative to the
/// `DeviceMemory` block means that the suballocations will be at most as aligned as the
/// `DeviceMemory` block, so the same problem presents itself. However, Vulkan guarantees that host
/// memory mappings are aligned to at least 64 bytes. As for the device, `DeviceMemory` blocks will
/// always be aligned such that they satisfy any requirements imposed by the implementation.
///
/// ```
/// use vulkano::{
///     buffer::{Buffer, BufferCreateInfo, BufferUsage, BufferMemory, Subbuffer},
///     memory::{
///         allocator::{
///             suballocator::Region, AllocationCreateInfo, AllocationType, DeviceLayout,
///             Suballocator, TlsfAllocator,
///         },
///         DeviceAlignment,
///     },
/// };
///
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// // Allocate a byte buffer with all the flags we'll need. Note that unlike in the first example,
/// // the alignment is 1.
/// let buffer = Subbuffer::from(
///     Buffer::new(
///         &memory_allocator,
///         &BufferCreateInfo {
///             usage: BufferUsage::STORAGE_BUFFER
///                 | BufferUsage::INDEX_BUFFER
///                 | BufferUsage::VERTEX_BUFFER,
///             ..Default::default()
///         },
///         &AllocationCreateInfo::default(),
///         DeviceLayout::from_size_alignment(16 * 1024 * 1024, 1).unwrap(),
///     )
///     .unwrap(),
/// );
///
/// let offset = match buffer.buffer().memory() {
///     BufferMemory::Normal(allocation) => allocation.offset(),
///     _ => unreachable!(),
/// };
///
/// // Since we want to suballocate the parent `DeviceMemory` block, but only the portion that's
/// // taken up by the buffer, the region is created with the offset and size of the buffer.
/// let allocator = TlsfAllocator::new(Region::new(offset, buffer.len()).unwrap());
///
/// // We can then allocate whatever type of data we need. Note that unlike in the first example,
/// // we don't know what the alignment might be. We subtract the offset of the buffer to get a
/// // suballocation relative to the buffer rather than the parent `DeviceMemory` block. Don't
/// // forget to add the offset back when you deallocate!
/// # let layout: DeviceLayout = return;
/// let mut allocation = allocator.allocate_buffer(layout).unwrap();
/// allocation.offset -= offset;
/// let subbuffer = buffer.slice(allocation.as_range());
///
/// // ...use the allocations...
///
/// // Once the allocations are no longer in use, you should deallocate them or reset the allocator
/// // (if you know that no other allocations exist). That means that if you use the allocations in
/// // a command buffer, you must only deallocate them after the command buffer is no longer being
/// // executed. The best way to do this is to keep some frame-local data, where each frame in
/// // flight has its own data associated with it. At the beginning of each frame, after waiting
/// // for the frame to finish, you deallocate everything associated with that frame.
/// allocation.offset += offset;
/// unsafe { allocator.deallocate(allocation) };
/// ```
///
/// # Safety
///
/// First consider using the provided implementations as there should be no reason to implement
/// this trait, but if you **must**:
///
/// - `allocate` must return a memory block that is in bounds of the region.
/// - `allocate` must return a memory block that doesn't alias any other currently allocated memory
///   blocks:
///   - Two currently allocated memory blocks must not share any memory locations, meaning that the
///     intersection of the byte ranges of the two memory blocks must be empty.
///   - Two neighboring currently allocated memory blocks must not share any [page] whose size is
///     given by the [buffer-image granularity], unless either both were allocated with
///     [`AllocationType::Linear`] or both were allocated with [`AllocationType::NonLinear`].
///   - The size does **not** have to be padded to the alignment. That is, as long the offset is
///     aligned and the memory blocks don't share any memory locations, a memory block is not
///     considered to alias another even if the padded size shares memory locations with another
///     memory block.
/// - A memory block must stay allocated until either `deallocate` is called on it or the allocator
///   is dropped. If the allocator is cloned, it must produce the same allocator, and memory blocks
///   must stay allocated until either `deallocate` is called on the memory block using any of the
///   clones or all of the clones have been dropped.
///
/// [`DeviceMemory`]: crate::memory::DeviceMemory
/// [`BufferUsage`]: crate::buffer::BufferUsage
/// [`UNIFORM_BUFFER`]: crate::buffer::BufferUsage::UNIFORM_BUFFER
/// [`SHADER_DEVICE_ADDRESS`]: crate::buffer::BufferUsage::SHADER_DEVICE_ADDRESS
/// [`ACCELERATION_STRUCTURE_STORAGE`]: crate::buffer::BufferUsage::ACCELERATION_STRUCTURE_STORAGE
/// [`SHADER_BINDING_TABLE`]: crate::buffer::BufferUsage::SHADER_BINDING_TABLE
/// [page]: super#pages
/// [buffer-image granularity]: super#buffer-image-granularity
pub unsafe trait Suballocator {
    /// The type of iterator returned by [`suballocations`].
    ///
    /// [`suballocations`]: Self::suballocations
    type Suballocations<'a>: Iterator<Item = SuballocationNode>
        + DoubleEndedIterator
        + ExactSizeIterator
    where
        Self: Sized + 'a;

    /// Creates a new suballocator for the given [region].
    ///
    /// [region]: Self#regions
    fn new(region: Region) -> Self
    where
        Self: Sized;

    /// Creates a new suballocation within the [region], assuming that only buffers are being
    /// allocated from it.
    ///
    /// This is **not to be used when both buffers and images are being allocated from the
    /// region**. It's strictly a shortcut for the most common use case: suballocating a buffer
    /// into smaller buffers. In other words, you shouldn't use this function when this isn't your
    /// use case. Instead, you should use [`allocate`], where you get to fill in the allocation
    /// type and buffer-image granularity. This method hard-codes them as
    /// [`AllocationType::Linear`] and [`DeviceAlignment::MIN`], respectively.
    ///
    /// See [`allocate`] for more info.
    ///
    /// [region]: Self#regions
    /// [`allocate`]: Self::allocate
    fn allocate_buffer(
        &mut self,
        layout: DeviceLayout,
    ) -> Result<Suballocation, SuballocatorError> {
        self.allocate(layout, AllocationType::Linear, DeviceAlignment::MIN)
    }

    /// Creates a new suballocation within the [region].
    ///
    /// # Arguments
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `buffer_image_granularity` - The [buffer-image granularity] device property.
    ///
    ///   This is provided as an argument here rather than on construction of the allocator to
    ///   allow for optimizations: if you are only ever going to be creating allocations with the
    ///   same `allocation_type` using this allocator, then you may hard-code this to
    ///   [`DeviceAlignment::MIN`], in which case, after inlining, the logic for aligning the
    ///   allocation to the buffer-image-granularity based on the allocation type of surrounding
    ///   allocations can be optimized out.
    ///
    ///   You don't need to consider the buffer-image granularity for instance when suballocating a
    ///   buffer, or when suballocating a [`DeviceMemory`] block that's only ever going to be used
    ///   for optimal images. However, if you do allocate both linear and non-linear resources and
    ///   don't specify the buffer-image granularity device property here, **you will get undefined
    ///   behavior down the line**. Note that [`AllocationType::Unknown`] counts as both linear and
    ///   non-linear at the same time: if you always use this as the `allocation_type` using this
    ///   allocator, then it is valid to set this to `DeviceAlignment::MIN`, but **you must ensure
    ///   all allocations are aligned to the buffer-image granularity at minimum**.
    ///
    /// [region]: Self#regions
    /// [buffer-image granularity]: super#buffer-image-granularity
    /// [`DeviceMemory`]: crate::memory::DeviceMemory
    fn allocate(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError>;

    /// Deallocates the given `suballocation`.
    ///
    /// # Safety
    ///
    /// - `suballocation` must refer to a **currently allocated** suballocation of `self`.
    unsafe fn deallocate(&mut self, suballocation: Suballocation);

    /// Resets the suballocator, deallocating all currently allocated suballocations at once.
    fn reset(&mut self);

    /// Returns the total amount of free space that is left in the [region].
    ///
    /// [region]: Self#regions
    fn free_size(&self) -> DeviceSize;

    /// Returns an iterator over the current suballocations.
    fn suballocations(&self) -> Self::Suballocations<'_>
    where
        Self: Sized;
}

impl Debug for dyn Suballocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Suballocator").finish_non_exhaustive()
    }
}

mod region {
    use super::{DeviceLayout, DeviceSize};

    /// A [region] for a [suballocator] to allocate within. All [suballocations] will be in bounds
    /// of this region.
    ///
    /// In order to prevent arithmetic overflow when allocating, the region's end must not exceed
    /// [`DeviceLayout::MAX_SIZE`].
    ///
    /// The suballocator knowing the offset of the region rather than only the size allows you to
    /// easily suballocate suballocations. Otherwise, if regions were always relative, you would
    /// have to pick some maximum alignment for a suballocation before suballocating it further, to
    /// satisfy alignment requirements. However, you might not even know the maximum alignment
    /// requirement. Instead you can feed a suballocator a region that is aligned any which way,
    /// and it makes sure that the *absolute offset* of the suballocation has the requested
    /// alignment, meaning the offset that's already offset by the region's offset.
    ///
    /// There's one important caveat: if suballocating a suballocation, and the suballocation and
    /// the suballocation's suballocations aren't both only linear or only nonlinear, then the
    /// region must be aligned to the [buffer-image granularity]. Otherwise, there might be a
    /// buffer-image granularity conflict between the parent suballocator's allocations and the
    /// child suballocator's allocations.
    ///
    /// [region]: super::Suballocator#regions
    /// [suballocator]: super::Suballocator
    /// [suballocations]: super::Suballocation
    /// [buffer-image granularity]: super::super#buffer-image-granularity
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Region {
        offset: DeviceSize,
        size: DeviceSize,
    }

    impl Region {
        /// Creates a new `Region` from the given `offset` and `size`.
        ///
        /// Returns [`None`] if the end of the region would exceed [`DeviceLayout::MAX_SIZE`].
        #[inline]
        pub const fn new(offset: DeviceSize, size: DeviceSize) -> Option<Self> {
            if offset.saturating_add(size) <= DeviceLayout::MAX_SIZE {
                // SAFETY: We checked that the end of the region doesn't exceed
                // `DeviceLayout::MAX_SIZE`.
                Some(unsafe { Region::new_unchecked(offset, size) })
            } else {
                None
            }
        }

        /// Creates a new `Region` from the given `offset` and `size` without doing any checks.
        ///
        /// # Safety
        ///
        /// - The end of the region must not exceed [`DeviceLayout::MAX_SIZE`], that is the
        ///   infinite-precision sum of `offset` and `size` must not exceed the bound.
        #[inline]
        pub const unsafe fn new_unchecked(offset: DeviceSize, size: DeviceSize) -> Self {
            Region { offset, size }
        }

        /// Returns the offset where the region begins.
        #[inline]
        pub const fn offset(&self) -> DeviceSize {
            self.offset
        }

        /// Returns the size of the region.
        #[inline]
        pub const fn size(&self) -> DeviceSize {
            self.size
        }
    }
}

/// Tells the [suballocator] what type of resource will be bound to the allocation, so that it can
/// optimize memory usage while still respecting the [buffer-image granularity].
///
/// [suballocator]: Suballocator
/// [buffer-image granularity]: super#buffer-image-granularity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationType {
    /// The type of resource is unknown, it might be either linear or non-linear. What this means
    /// is that allocations created with this type must always be aligned to the buffer-image
    /// granularity.
    Unknown = 0,

    /// The resource is linear, e.g. buffers, linear images. A linear allocation following another
    /// linear allocation never needs to be aligned to the buffer-image granularity.
    Linear = 1,

    /// The resource is non-linear, e.g. optimal images. A non-linear allocation following another
    /// non-linear allocation never needs to be aligned to the buffer-image granularity.
    NonLinear = 2,
}

impl From<ImageTiling> for AllocationType {
    #[inline]
    fn from(tiling: ImageTiling) -> Self {
        match tiling {
            ImageTiling::Optimal => AllocationType::NonLinear,
            ImageTiling::Linear => AllocationType::Linear,
            ImageTiling::DrmFormatModifier => AllocationType::Unknown,
        }
    }
}

/// An allocation made using a [suballocator].
///
/// [suballocator]: Suballocator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Suballocation {
    /// The **absolute** offset within the [region]. That means that this is already offset by the
    /// region's offset, **not relative to beginning of the region**. This offset will be aligned
    /// to the requested alignment.
    ///
    /// [region]: Suballocator#regions
    pub offset: DeviceSize,

    /// The size of the allocation. This will be exactly equal to the requested size.
    pub size: DeviceSize,

    /// The type of resources that can be bound to this memory block. This will be exactly equal to
    /// the requested allocation type.
    pub allocation_type: AllocationType,

    /// An opaque handle identifying the allocation within the allocator.
    pub handle: AllocationHandle,
}

impl Suballocation {
    /// Returns the suballocation as a `DeviceSize` range.
    ///
    /// This is identical to `self.offset..self.offset + self.size`.
    #[inline]
    pub fn as_range(&self) -> Range<DeviceSize> {
        self.offset..self.offset + self.size
    }

    /// Returns the suballocation as a `usize` range.
    ///
    /// This is identical to `self.offset as usize..(self.offset + self.size) as usize`.
    #[inline]
    pub fn as_usize_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }
}

/// Error that can be returned when creating an [allocation] using a [suballocator].
///
/// [allocation]: Suballocation
/// [suballocator]: Suballocator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuballocatorError {
    /// There is no more space available in the region.
    OutOfRegionMemory,

    /// The region has enough free space to satisfy the request but is too fragmented.
    FragmentedRegion,
}

impl Error for SuballocatorError {}

impl Display for SuballocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::OutOfRegionMemory => "out of region memory",
            Self::FragmentedRegion => "the region is too fragmented",
        };

        f.write_str(msg)
    }
}

/// A node within a [suballocator]'s list/tree of suballocations.
///
/// [suballocator]: Suballocator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SuballocationNode {
    /// The **absolute** offset within the [region]. That means that this is already offset by the
    /// region's offset, **not relative to beginning of the region**.
    ///
    /// [region]: Suballocator#regions
    pub offset: DeviceSize,

    /// The size of the allocation.
    pub size: DeviceSize,

    /// Tells us if the allocation is free, and if not, what type of resources can be bound to it.
    pub allocation_type: SuballocationType,
}

impl SuballocationNode {
    /// Returns the suballocation as a `DeviceSize` range.
    ///
    /// This is identical to `self.offset..self.offset + self.size`.
    #[inline]
    pub fn as_range(&self) -> Range<DeviceSize> {
        self.offset..self.offset + self.size
    }

    /// Returns the suballocation as a `usize` range.
    ///
    /// This is identical to `self.offset as usize..(self.offset + self.size) as usize`.
    #[inline]
    pub fn as_usize_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }
}

/// Tells us if an allocation within a [suballocator]'s list/tree of suballocations is free, and if
/// not, what type of resources can be bound to it. The suballocator needs to keep track of this in
/// order to be able to respect the buffer-image granularity.
///
/// [suballocator]: Suballocator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuballocationType {
    /// The type of resource is unknown, it might be either linear or non-linear. What this means
    /// is that allocations created with this type must always be aligned to the buffer-image
    /// granularity.
    Unknown = 0,

    /// The resource is linear, e.g. buffers, linear images. A linear allocation following another
    /// linear allocation never needs to be aligned to the buffer-image granularity.
    Linear = 1,

    /// The resource is non-linear, e.g. optimal images. A non-linear allocation following another
    /// non-linear allocation never needs to be aligned to the buffer-image granularity.
    NonLinear = 2,

    /// The allocation is free. It can take on any of the allocation types once allocated.
    Free = 3,
}

impl From<AllocationType> for SuballocationType {
    #[inline]
    fn from(ty: AllocationType) -> Self {
        match ty {
            AllocationType::Unknown => SuballocationType::Unknown,
            AllocationType::Linear => SuballocationType::Linear,
            AllocationType::NonLinear => SuballocationType::NonLinear,
        }
    }
}

/// Checks if resources A and B share a page.
///
/// <div class="vulkano-alert-note">
///
/// > Note
/// >
/// > Assumes `a_offset + a_size <= b_offset`.
///
/// </div>
#[inline]
fn are_blocks_on_same_page(
    a_offset: DeviceSize,
    a_size: DeviceSize,
    b_offset: DeviceSize,
    page_size: DeviceAlignment,
) -> bool {
    debug_assert!(a_offset + a_size <= b_offset);

    let a_end = a_offset + a_size;
    let a_end_page = align_down(a_end.saturating_sub(1), page_size);
    let b_start_page = align_down(b_offset, page_size);

    a_end_page == b_start_page
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_queue::ArrayQueue;
    use parking_lot::Mutex;
    use std::{sync::Barrier, thread};

    const DUMMY_LAYOUT: DeviceLayout = DeviceLayout::from_size_alignment(1, 1).unwrap();

    #[test]
    fn are_blocks_on_same_page_literal_edge_cases() {
        const PAGE_SIZE: DeviceAlignment = DeviceAlignment::new(16).unwrap();
        const LAST_PAGE_OFFSET: DeviceSize = align_down(DeviceSize::MAX, PAGE_SIZE);

        // These two are technically not correct, but it does't matter because `a_end` is 0, and 0
        // is always aligned for every alignment, so aligning it up isn't going to do anything.
        assert!(are_blocks_on_same_page(0, 0, 0, PAGE_SIZE));
        assert!(are_blocks_on_same_page(0, 0, 1, PAGE_SIZE));

        assert!(are_blocks_on_same_page(0, 15, 15, PAGE_SIZE));
        assert!(!are_blocks_on_same_page(0, 15, 16, PAGE_SIZE));
        assert!(!are_blocks_on_same_page(0, 16, 16, PAGE_SIZE));
        assert!(!are_blocks_on_same_page(0, 16, 17, PAGE_SIZE));
        assert!(are_blocks_on_same_page(0, 17, 17, PAGE_SIZE));

        assert!(are_blocks_on_same_page(16, 15, 31, PAGE_SIZE));
        assert!(!are_blocks_on_same_page(16, 15, 32, PAGE_SIZE));
        assert!(!are_blocks_on_same_page(16, 16, 32, PAGE_SIZE));
        assert!(!are_blocks_on_same_page(16, 16, 33, PAGE_SIZE));
        assert!(are_blocks_on_same_page(16, 17, 33, PAGE_SIZE));

        assert!(are_blocks_on_same_page(
            LAST_PAGE_OFFSET - 16,
            15,
            LAST_PAGE_OFFSET - 1,
            PAGE_SIZE,
        ));
        assert!(!are_blocks_on_same_page(
            LAST_PAGE_OFFSET - 16,
            15,
            LAST_PAGE_OFFSET,
            PAGE_SIZE,
        ));
        assert!(!are_blocks_on_same_page(
            LAST_PAGE_OFFSET - 16,
            16,
            LAST_PAGE_OFFSET,
            PAGE_SIZE,
        ));
        assert!(!are_blocks_on_same_page(
            LAST_PAGE_OFFSET - 16,
            16,
            LAST_PAGE_OFFSET + 1,
            PAGE_SIZE,
        ));
        assert!(are_blocks_on_same_page(
            LAST_PAGE_OFFSET - 16,
            17,
            LAST_PAGE_OFFSET + 1,
            PAGE_SIZE,
        ));

        assert!(are_blocks_on_same_page(
            LAST_PAGE_OFFSET,
            15,
            LAST_PAGE_OFFSET + 15,
            PAGE_SIZE,
        ));
    }

    #[test]
    fn tlsf_allocator_capacity() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100;
        const ALLOCATION_STEP: DeviceSize = 117;
        const MAX_ALLOCATION_SIZE: DeviceSize = ALLOCATION_STEP * THREADS;
        const REGION_SIZE: DeviceSize = (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2)
            * ALLOCATIONS_PER_THREAD
            + MAX_ALLOCATION_SIZE;

        let allocator = Mutex::new(TlsfAllocator::new(Region::new(0, REGION_SIZE).unwrap()));
        let allocs = ArrayQueue::new((ALLOCATIONS_PER_THREAD * THREADS) as usize);

        let barrier = &Barrier::new(THREADS as usize);

        // Using threads to randomize allocation order.
        thread::scope(|scope| {
            for i in 1..=THREADS {
                let (allocator, allocs) = (&allocator, &allocs);

                scope.spawn(move || {
                    let layout = DeviceLayout::from_size_alignment(i * ALLOCATION_STEP, 1).unwrap();

                    barrier.wait();

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        let mut allocator = allocator.lock();
                        let alloc = allocator
                            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                            .unwrap();
                        allocs.push(alloc).unwrap();
                    }
                });
            }
        });

        let mut allocator = allocator.into_inner();

        let layout = DeviceLayout::from_size_alignment(MAX_ALLOCATION_SIZE, 1).unwrap();
        assert!(allocator
            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), MAX_ALLOCATION_SIZE);
        assert_eq!(
            allocator.suballocations().count(),
            (ALLOCATIONS_PER_THREAD * THREADS) as usize + 1,
        );
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);

        let mut free_size = MAX_ALLOCATION_SIZE;

        for (index, alloc) in allocs.into_iter().enumerate() {
            unsafe { allocator.deallocate(alloc) };

            free_size += alloc.size;

            let allocated_allocs = (ALLOCATIONS_PER_THREAD * THREADS) as usize - index - 1;
            let total_allocs = if allocated_allocs == 0 {
                1
            } else {
                allocated_allocs + 2
            };

            assert_eq!(allocator.free_size(), free_size);
            assert_eq!(allocator.suballocations().count(), total_allocs);
            assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
        }

        assert!(allocator
            .allocate(
                DeviceLayout::from_size_alignment(REGION_SIZE, 1).unwrap(),
                AllocationType::Unknown,
                DeviceAlignment::MIN,
            )
            .is_err());
        assert_eq!(allocator.free_size(), REGION_SIZE);
        assert_eq!(allocator.suballocations().count(), 1);
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
    }

    #[test]
    fn tlsf_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 10 * 256;
        const LAYOUT: DeviceLayout = DeviceLayout::from_size_alignment(1, 256).unwrap();

        let mut allocator = TlsfAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut allocs = Vec::with_capacity(10);

        for _ in 0..9 {
            allocs.push(
                allocator
                    .allocate(LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                    .unwrap(),
            );
        }

        assert!(allocator
            .allocate(LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), 256);

        for alloc in allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }
    }

    #[test]
    fn tlsf_allocator_respects_granularity() {
        const GRANULARITY: DeviceAlignment = DeviceAlignment::new(256).unwrap();
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY.as_devicesize();

        let mut allocator = TlsfAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        {
            const ALLOCATIONS: DeviceSize = REGION_SIZE / tlsf::MIN_NODE_SIZE;

            let mut allocs = Vec::with_capacity(ALLOCATIONS as usize);

            for _ in 0..ALLOCATIONS {
                allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs {
                unsafe { allocator.deallocate(alloc) };
            }
        }

        {
            const ALLOCATIONS: DeviceSize = GRANULARITY.as_devicesize() / tlsf::MIN_NODE_SIZE;

            let alloc1 = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
                .unwrap();
            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), GRANULARITY.as_devicesize());

            let mut allocs = Vec::with_capacity(ALLOCATIONS as usize);

            for _ in 0..ALLOCATIONS {
                allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs {
                unsafe { allocator.deallocate(alloc) };
            }

            unsafe { allocator.deallocate(alloc1) };
        }
    }

    #[test]
    fn free_list_allocator_capacity() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100;
        const ALLOCATION_STEP: DeviceSize = 117;
        const REGION_SIZE: DeviceSize =
            (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2) * ALLOCATIONS_PER_THREAD;

        let allocator = Mutex::new(FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap()));
        let allocs = ArrayQueue::new((ALLOCATIONS_PER_THREAD * THREADS) as usize);

        let barrier = &Barrier::new(THREADS as usize);

        // Using threads to randomize allocation order.
        thread::scope(|scope| {
            for i in 1..=THREADS {
                let (allocator, allocs) = (&allocator, &allocs);

                scope.spawn(move || {
                    let layout = DeviceLayout::from_size_alignment(i * ALLOCATION_STEP, 1).unwrap();

                    barrier.wait();

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        let mut allocator = allocator.lock();
                        let alloc = allocator
                            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                            .unwrap();
                        allocs.push(alloc).unwrap();
                    }
                });
            }
        });

        let mut allocator = allocator.into_inner();

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), 0);
        assert_eq!(
            allocator.suballocations().count(),
            (ALLOCATIONS_PER_THREAD * THREADS) as usize,
        );
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);

        let mut free_size = 0;

        for (index, alloc) in allocs.into_iter().enumerate() {
            unsafe { allocator.deallocate(alloc) };

            free_size += alloc.size;

            assert_eq!(allocator.free_size(), free_size);
            assert_eq!(
                allocator.suballocations().count(),
                (ALLOCATIONS_PER_THREAD * THREADS) as usize - index,
            );
            assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
        }

        let alloc = allocator
            .allocate(
                DeviceLayout::from_size_alignment(REGION_SIZE, 1).unwrap(),
                AllocationType::Unknown,
                DeviceAlignment::MIN,
            )
            .unwrap();
        unsafe { allocator.deallocate(alloc) };

        assert_eq!(allocator.free_size(), REGION_SIZE);
        assert_eq!(allocator.suballocations().count(), 1);
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
    }

    #[test]
    fn free_list_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 10 * 256;
        const LAYOUT: DeviceLayout = DeviceLayout::from_size_alignment(1, 256).unwrap();

        let mut allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut allocs = Vec::with_capacity(10);

        for _ in 0..10 {
            allocs.push(
                allocator
                    .allocate(LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                    .unwrap(),
            );
        }

        assert!(allocator
            .allocate(LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), REGION_SIZE - 10);

        for alloc in allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }
    }

    #[test]
    fn free_list_allocator_respects_granularity() {
        const GRANULARITY: DeviceAlignment = DeviceAlignment::new(16).unwrap();
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY.as_devicesize();

        let mut allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut linear_allocs = Vec::with_capacity(REGION_SIZE as usize / 2);
        let mut nonlinear_allocs = Vec::with_capacity(REGION_SIZE as usize / 2);

        for i in 0..REGION_SIZE {
            if i % 2 == 0 {
                linear_allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                        .unwrap(),
                );
            } else {
                nonlinear_allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::NonLinear, GRANULARITY)
                        .unwrap(),
                );
            }
        }

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        assert_eq!(allocator.free_size(), 0);

        for alloc in linear_allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }

        let alloc = allocator
            .allocate(
                DeviceLayout::from_size_alignment(GRANULARITY.as_devicesize(), 1).unwrap(),
                AllocationType::Unknown,
                GRANULARITY,
            )
            .unwrap();
        unsafe { allocator.deallocate(alloc) };

        let alloc = allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
            .unwrap();
        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
            .is_err());
        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        unsafe { allocator.deallocate(alloc) };

        for alloc in nonlinear_allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }
    }

    #[test]
    fn free_list_allocator_respects_granularity_backwards() {
        const GRANULARITY: DeviceAlignment = DeviceAlignment::new(1024).unwrap();
        const REGION_SIZE: DeviceSize = 4 * GRANULARITY.as_devicesize();
        const HALF_PAGE: DeviceLayout = DeviceLayout::from_size_alignment(512, 1).unwrap();
        const PAGE_ALIGNED_HALF_PAGE: DeviceLayout =
            DeviceLayout::from_size_alignment(512, 1024).unwrap();

        let mut allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        // Occupy the upper half of the first page, leaving the lower half free.
        let lower = allocator
            .allocate(HALF_PAGE, AllocationType::Linear, GRANULARITY)
            .unwrap();
        let upper = allocator
            .allocate(HALF_PAGE, AllocationType::Linear, GRANULARITY)
            .unwrap();
        assert_eq!(upper.offset, 512);
        unsafe { allocator.deallocate(lower) };

        // A conflicting type must not take the free half, as that would put it on the same
        // page as `upper`.
        let nonlinear = allocator
            .allocate(
                PAGE_ALIGNED_HALF_PAGE,
                AllocationType::NonLinear,
                GRANULARITY,
            )
            .unwrap();
        assert_eq!(nonlinear.offset, GRANULARITY.as_devicesize());

        // The same type may.
        let linear = allocator
            .allocate(HALF_PAGE, AllocationType::Linear, GRANULARITY)
            .unwrap();
        assert_eq!(linear.offset, 0);

        for suballoc in [upper, nonlinear, linear] {
            unsafe { allocator.deallocate(suballoc) };
        }

        assert_eq!(allocator.free_size(), REGION_SIZE);
    }

    #[test]
    fn buddy_allocator_capacity() {
        const MAX_ORDER: usize = 10;
        const REGION_SIZE: DeviceSize = BuddyAllocator::MIN_NODE_SIZE << MAX_ORDER;

        let mut allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut allocs = Vec::with_capacity(1 << MAX_ORDER);

        for order in 0..=MAX_ORDER {
            let layout =
                DeviceLayout::from_size_alignment(BuddyAllocator::MIN_NODE_SIZE << order, 1)
                    .unwrap();

            for _ in 0..1 << (MAX_ORDER - order) {
                allocs.push(
                    allocator
                        .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(allocator.free_size(), 0);
            assert_eq!(allocator.suballocations().count(), 1 << (MAX_ORDER - order));
            assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);

            let mut last_offset = 0;

            for alloc in allocator.suballocations() {
                if alloc.offset == 0 {
                    continue;
                }

                assert!(last_offset < alloc.offset);
                last_offset = alloc.offset;
            }

            let mut free_size = 0;

            for (index, alloc) in allocs.drain(..).enumerate() {
                unsafe { allocator.deallocate(alloc) };

                free_size += alloc.size;

                let deallocated_allocs = index + 1;
                let allocated_allocs = (1 << (MAX_ORDER - order)) - deallocated_allocs;

                assert_eq!(allocator.free_size(), free_size);
                assert_eq!(
                    allocator.suballocations().count(),
                    deallocated_allocs.count_ones() as usize + allocated_allocs,
                );
                assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
            }
        }

        let mut orders = (0..MAX_ORDER).collect::<Vec<_>>();

        for mid in 0..MAX_ORDER {
            orders.rotate_left(mid);

            for &order in &orders {
                let layout =
                    DeviceLayout::from_size_alignment(BuddyAllocator::MIN_NODE_SIZE << order, 1)
                        .unwrap();

                allocs.push(
                    allocator
                        .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            let alloc = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(allocator.free_size(), 0);
            assert_eq!(allocator.suballocations().count(), MAX_ORDER + 1);
            assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);

            let mut last_offset = 0;

            for alloc in allocator.suballocations() {
                if alloc.offset == 0 {
                    continue;
                }

                assert!(last_offset < alloc.offset);
                last_offset = alloc.offset;
            }

            let mut free_size = 0;

            for alloc in allocs.drain(..) {
                unsafe { allocator.deallocate(alloc) };

                free_size += alloc.size;

                assert_eq!(allocator.free_size(), free_size);
                assert_eq!(allocator.suballocations().count(), MAX_ORDER + 1);
                assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
            }

            unsafe { allocator.deallocate(alloc) };

            assert_eq!(allocator.free_size(), REGION_SIZE);
            assert_eq!(allocator.suballocations().count(), 1);
            assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
        }
    }

    #[test]
    fn buddy_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 4096;

        let mut allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        {
            let layout = DeviceLayout::from_size_alignment(1, 4096).unwrap();

            let alloc = allocator
                .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
            assert!(allocator
                .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(
                allocator.free_size(),
                REGION_SIZE - BuddyAllocator::MIN_NODE_SIZE,
            );
            unsafe { allocator.deallocate(alloc) };
        }

        {
            let layout_a = DeviceLayout::from_size_alignment(1, 256).unwrap();
            let allocations_a = REGION_SIZE / layout_a.alignment().as_devicesize();
            let layout_b = DeviceLayout::from_size_alignment(1, 16).unwrap();
            let allocations_b = REGION_SIZE / layout_b.alignment().as_devicesize() - allocations_a;

            let mut allocs =
                Vec::with_capacity((REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE) as usize);

            for _ in 0..allocations_a {
                allocs.push(
                    allocator
                        .allocate(layout_a, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(layout_a, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(
                allocator.free_size(),
                REGION_SIZE - allocations_a * BuddyAllocator::MIN_NODE_SIZE,
            );

            for _ in 0..allocations_b {
                allocs.push(
                    allocator
                        .allocate(layout_b, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs {
                unsafe { allocator.deallocate(alloc) };
            }
        }
    }

    #[test]
    fn buddy_allocator_respects_granularity() {
        const GRANULARITY: DeviceAlignment = DeviceAlignment::new(256).unwrap();
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY.as_devicesize();

        let mut allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        {
            const ALLOCATIONS: DeviceSize = REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE;

            let mut allocs = Vec::with_capacity(ALLOCATIONS as usize);

            for _ in 0..ALLOCATIONS {
                allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs {
                unsafe { allocator.deallocate(alloc) };
            }
        }

        {
            let alloc1 = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
                .unwrap();
            let alloc2 = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
                .unwrap();
            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), 0);
            unsafe { allocator.deallocate(alloc1) };
            unsafe { allocator.deallocate(alloc2) };
        }
    }

    #[test]
    fn buddy_allocator_respects_region_offset() {
        const MAX_ORDER: usize = 31;
        const REGION_OFFSET: DeviceSize = align_down(
            DeviceLayout::MAX_SIZE - REGION_SIZE,
            DeviceAlignment::new(BuddyAllocator::MIN_NODE_SIZE).unwrap(),
        );
        const REGION_SIZE: DeviceSize = BuddyAllocator::MIN_NODE_SIZE << MAX_ORDER;

        let mut allocator = BuddyAllocator::new(Region::new(REGION_OFFSET, REGION_SIZE).unwrap());

        for order in (0..=MAX_ORDER).rev().take(10) {
            let size = BuddyAllocator::MIN_NODE_SIZE << order;
            let node_count = 1 << (MAX_ORDER - order);

            for index in 0..node_count {
                let layout = DeviceLayout::from_size_alignment(size, 1).unwrap();
                let allocation = allocator
                    .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                    .unwrap();
                assert_eq!(allocation.offset, REGION_OFFSET + index * size);
                assert_eq!(allocation.size, size);
            }

            allocator.reset();
        }
    }

    #[test]
    fn bump_allocator_respects_alignment() {
        const ALIGNMENT: DeviceSize = 16;
        const REGION_SIZE: DeviceSize = 10 * ALIGNMENT;

        let layout = DeviceLayout::from_size_alignment(1, ALIGNMENT).unwrap();
        let mut allocator = BumpAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        for _ in 0..10 {
            allocator
                .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
        }

        assert!(allocator
            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), ALIGNMENT - 1);
        assert_eq!(allocator.suballocations().count(), 2);
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);

        for _ in 0..ALIGNMENT - 1 {
            allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
        }

        assert!(allocator
            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), 0);
        assert_eq!(allocator.suballocations().count(), 1);
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);

        allocator.reset();
        assert_eq!(allocator.free_size(), REGION_SIZE);
        assert_eq!(allocator.suballocations().count(), 1);
        assert_eq!(suballocation_size_sum(&allocator), REGION_SIZE);
    }

    #[test]
    fn bump_allocator_respects_granularity() {
        const ALLOCATIONS: DeviceSize = 10;
        const GRANULARITY: DeviceAlignment = DeviceAlignment::new(1024).unwrap();
        const REGION_SIZE: DeviceSize = ALLOCATIONS * GRANULARITY.as_devicesize();

        let mut allocator = BumpAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        for i in 0..ALLOCATIONS {
            for _ in 0..GRANULARITY.as_devicesize() {
                allocator
                    .allocate(
                        DUMMY_LAYOUT,
                        if i % 2 == 0 {
                            AllocationType::NonLinear
                        } else {
                            AllocationType::Linear
                        },
                        GRANULARITY,
                    )
                    .unwrap();
            }
        }

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        assert_eq!(allocator.free_size(), 0);

        allocator.reset();

        for i in 0..ALLOCATIONS {
            allocator
                .allocate(
                    DUMMY_LAYOUT,
                    if i % 2 == 0 {
                        AllocationType::Linear
                    } else {
                        AllocationType::NonLinear
                    },
                    GRANULARITY,
                )
                .unwrap();
        }

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        assert_eq!(allocator.free_size(), GRANULARITY.as_devicesize() - 1);

        allocator.reset();
        assert_eq!(allocator.free_size(), REGION_SIZE);
    }

    fn suballocation_size_sum(allocator: &impl Suballocator) -> DeviceSize {
        allocator
            .suballocations()
            .map(|n| n.size)
            .sum::<DeviceSize>()
    }
}
