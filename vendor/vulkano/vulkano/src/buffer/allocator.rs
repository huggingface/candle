//! Efficiently allocates buffer memory.

use super::{
    sys::BufferCreateInfo, AllocateBufferError, Buffer, BufferContents, BufferMemory, BufferUsage,
    Subbuffer,
};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    memory::{
        allocator::{
            aliasable_box::AliasableBox, align_up, suballocator::Region, AllocationCreateInfo,
            AllocationHandle as SuballocationHandle, AllocationType, DeviceLayout, MemoryAllocator,
            MemoryAllocatorError, MemoryTypeFilter, StandardMemoryAllocator, Suballocation,
            Suballocator, SuballocatorError,
        },
        DeviceAlignment,
    },
    DeviceSize, Validated,
};
use crossbeam_queue::ArrayQueue;
use slabbin::SlabAllocator;
use std::{
    cell::UnsafeCell,
    cmp,
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    ops::Range,
    sync::Arc,
};

/// Efficiently allocates buffer memory.
///
/// Instead of allocating a separate buffer for each purpose, you can use this allocator, which
/// keeps a pool of `Buffer` blocks and suballocates them using the provided suballocator `S`.
#[derive(Debug)]
pub struct BufferAllocator<S> {
    memory_allocator: Arc<dyn MemoryAllocator>,
    block_size: DeviceSize,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
    buffer_alignment: DeviceAlignment,
    blocks: Vec<AliasableBox<BufferMemoryBlock<S>>>,
    block_allocator: SlabAllocator<BufferMemoryBlock<S>>,
}

impl<S> BufferAllocator<S> {
    /// Creates a new `BufferAllocator`.
    pub fn new(
        memory_allocator: &Arc<impl MemoryAllocator>,
        create_info: &BufferAllocatorCreateInfo<'_>,
    ) -> Self {
        Self::new_inner(memory_allocator.clone().as_dyn(), create_info)
    }

    fn new_inner(
        memory_allocator: Arc<dyn MemoryAllocator>,
        create_info: &BufferAllocatorCreateInfo<'_>,
    ) -> Self {
        let &BufferAllocatorCreateInfo {
            block_size,
            buffer_usage,
            memory_type_filter,
            _ne: _,
        } = create_info;

        let properties = memory_allocator.device().physical_device().properties();
        let buffer_alignment = [
            buffer_usage
                .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER)
                .then_some(properties.min_texel_buffer_offset_alignment),
            buffer_usage
                .contains(BufferUsage::UNIFORM_BUFFER)
                .then_some(properties.min_uniform_buffer_offset_alignment),
            buffer_usage
                .contains(BufferUsage::STORAGE_BUFFER)
                .then_some(properties.min_storage_buffer_offset_alignment),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(DeviceAlignment::MIN);

        BufferAllocator {
            memory_allocator,
            block_size,
            buffer_usage,
            memory_type_filter,
            buffer_alignment,
            blocks: Vec::new(),
            block_allocator: SlabAllocator::new(32),
        }
    }

    fn allocate_buffer(&self) -> Result<Arc<Buffer>, MemoryAllocatorError> {
        Buffer::try_new(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: self.buffer_usage,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: self.memory_type_filter,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(self.block_size, 1).unwrap(),
        )
        .map_err(|err| match err {
            Validated::Error(AllocateBufferError::AllocateMemory(err)) => err,
            // We don't use sparse-binding, concurrent sharing or external memory; therefore, the
            // other errors can't happen.
            _ => unreachable!("{err:?}"),
        })
    }
}

impl<S: Suballocator> BufferAllocator<S> {
    /// Creates an allocation for the given `layout`.
    #[inline]
    pub fn allocate(&mut self, layout: DeviceLayout) -> Result<BufferAlloc, MemoryAllocatorError> {
        if layout.size() > self.block_size {
            return Err(MemoryAllocatorError::BlockSizeExceeded);
        }

        // TODO: Incremental sorting
        self.blocks.sort_by_key(|block| block.free_size());
        let (Ok(idx) | Err(idx)) = self
            .blocks
            .binary_search_by_key(&layout.size(), |block| block.free_size());

        for block in &mut self.blocks[idx..] {
            if let Ok(allocation) = block.allocate(layout, self.buffer_alignment) {
                return Ok(allocation);
            }
        }

        let block = BufferMemoryBlock::new(self.allocate_buffer()?, &self.block_allocator);

        self.blocks.push(block);
        let block = self.blocks.last_mut().unwrap();

        match block.allocate(layout, self.buffer_alignment) {
            Ok(allocation) => Ok(allocation),
            Err(SuballocatorError::OutOfRegionMemory) => unreachable!(),
            Err(SuballocatorError::FragmentedRegion) => unreachable!(),
        }
    }

    /// Deallocates the given `allocation`.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `self`.
    #[inline]
    pub unsafe fn deallocate(&mut self, allocation: BufferAlloc) {
        let blocks = &self.blocks;
        let block_ptr = allocation.handle.block_ptr.cast::<BufferMemoryBlock<S>>();

        debug_assert!(
            blocks.iter().any(|block| &raw const **block == block_ptr),
            "attempted to deallocate a memory block that does not belong to this allocator",
        );

        // SAFETY: The caller must guarantee that `allocation` refers to one allocated by `self`;
        // therefore, `block_ptr` must be the same one we gave out on allocation. We know that this
        // pointer must be valid because all blocks are boxed and pinned in memory and because a
        // block isn't dropped until the allocator itself is dropped, at which point it would be
        // impossible to call this method. We also know that it must be valid to create a reference
        // to the block because we have exclusive access to the allocator.
        let block = unsafe { &mut *block_ptr };

        let suballocation = Suballocation {
            offset: allocation.offset,
            size: allocation.size,
            allocation_type: AllocationType::Linear,
            handle: allocation.handle.suballocation_handle,
        };

        // SAFETY: The caller must guarantee that `allocation` refers to a currently allocated
        // allocation of `self`.
        unsafe { block.deallocate(suballocation) };
    }

    /// Resets the allocator, deallocating all currently allocated allocations at once.
    #[inline]
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.reset();
        }
    }
}

#[derive(Debug)]
struct BufferMemoryBlock<S> {
    buffer: Arc<Buffer>,
    offset: DeviceSize,
    atom_size: DeviceAlignment,
    suballocator: S,
    allocation_count: usize,
}

impl<S: Suballocator> BufferMemoryBlock<S> {
    fn new(buffer: Arc<Buffer>, block_allocator: &SlabAllocator<Self>) -> AliasableBox<Self> {
        let allocation = match buffer.memory() {
            BufferMemory::Normal(allocation) => allocation,
            BufferMemory::Sparse | BufferMemory::External => unreachable!(),
        };
        let offset = allocation.offset();
        let atom_size = allocation.atom_size().unwrap_or(DeviceAlignment::MIN);
        let suballocator = S::new(
            Region::new(offset, buffer.size())
                .expect("we somehow managed to allocate more than `DeviceLayout::MAX_SIZE` bytes"),
        );

        AliasableBox::new(
            BufferMemoryBlock {
                buffer,
                offset,
                atom_size,
                suballocator,
                allocation_count: 0,
            },
            block_allocator,
        )
    }

    #[inline]
    unsafe fn deallocate(&mut self, mut suballocation: Suballocation) {
        suballocation.offset += self.offset;

        unsafe { self.suballocator.deallocate(suballocation) };

        self.allocation_count -= 1;

        // For bump allocators, reset the free-start once there are no remaining allocations.
        if self.allocation_count == 0 {
            self.suballocator.reset();
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.suballocator.reset();
        self.allocation_count = 0;
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.suballocator.free_size()
    }
}

impl<S: Suballocator> AliasableBox<BufferMemoryBlock<S>> {
    #[inline]
    fn allocate(
        &mut self,
        layout: DeviceLayout,
        buffer_alignment: DeviceAlignment,
    ) -> Result<BufferAlloc, SuballocatorError> {
        let layout = layout
            .align_to(cmp::max(self.atom_size, buffer_alignment))
            .unwrap();
        let mut suballocation = self.suballocator.allocate_buffer(layout)?;

        suballocation.offset -= self.offset;

        self.allocation_count += 1;

        // It is paramount to soundness that the pointer we give out doesn't go through `DerefMut`,
        // as such a pointer would become invalidated when another allocation is made.
        let block_ptr = AliasableBox::as_mut_ptr(self);

        Ok(BufferAlloc {
            buffer: self.buffer.clone(),
            offset: suballocation.offset,
            size: suballocation.size,
            handle: AllocationHandle {
                block_ptr: block_ptr.cast(),
                suballocation_handle: suballocation.handle,
            },
        })
    }
}

/// Parameters to create a new [`BufferAllocator`].
#[derive(Clone, Debug)]
pub struct BufferAllocatorCreateInfo<'a> {
    /// The block size in bytes.
    ///
    /// The allocator keeps a pool of [`Buffer`] blocks, and every time a new block is allocated,
    /// it is allocated with this size.
    ///
    /// The default value is `0`, which must be overridden.
    pub block_size: DeviceSize,

    /// The buffer usage that all allocated buffers should have.
    ///
    /// The default value is empty, which must be overridden.
    pub buffer_usage: BufferUsage,

    /// The memory type filter all buffers should be allocated with.
    ///
    /// The default value is [`MemoryTypeFilter::PREFER_DEVICE`].
    pub memory_type_filter: MemoryTypeFilter,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BufferAllocatorCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferAllocatorCreateInfo<'_> {
    /// Returns a default `BufferAllocatorCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        BufferAllocatorCreateInfo {
            block_size: 0,
            buffer_usage: BufferUsage::empty(),
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            _ne: crate::NE,
        }
    }
}

/// An allocation made using a [`BufferAllocator`].
#[derive(Clone, Debug)]
pub struct BufferAlloc {
    /// The underlying block of buffer memory.
    pub buffer: Arc<Buffer>,

    /// The offset from the start of the buffer memory. The offset **relative to the beginning of
    /// the device memory block** will be aligned to the requested alignment, **not relative to the
    /// beginning of the buffer memory block**.
    pub offset: DeviceSize,

    /// The size of the allocation. This will be exactly equal to the requested size.
    pub size: DeviceSize,

    /// An opaque handle identifying the allocation within the allocator.
    pub handle: AllocationHandle,
}

impl BufferAlloc {
    /// Returns the allocation as a `DeviceSize` range.
    ///
    /// This is identical to `self.offset..self.offset + self.size`.
    #[inline]
    pub fn as_range(&self) -> Range<DeviceSize> {
        self.offset..self.offset + self.size
    }

    /// Returns the allocation as a `usize` range.
    ///
    /// This is identical to `self.offset as usize..(self.offset + self.size) as usize`.
    #[inline]
    pub fn as_usize_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }
}

/// An opaque handle identifying an allocation inside a [`BufferAllocator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationHandle {
    block_ptr: *mut (),
    suballocation_handle: SuballocationHandle,
}

unsafe impl Send for AllocationHandle {}
unsafe impl Sync for AllocationHandle {}

const MAX_ARENAS: usize = 32;

/// Efficiently suballocates buffers into smaller subbuffers.
///
/// This allocator is especially suitable when you want to upload or download some data regularly
/// (for example, at each frame for a video game).
///
/// # Algorithm
///
/// The allocator keeps a pool of *arenas*. An arena is simply a buffer in which *arena allocation*
/// takes place, also known as *bump allocation* or *linear allocation*. Every time you allocate,
/// one of these arenas is suballocated. An arena is suballocated until it runs out of space, at
/// which point a free one is taken from the pool. If there is no arena that is currently
/// available, one will be allocated. After all subbuffers allocated from an arena are dropped, the
/// arena is automatically returned to the arena pool for reuse. If you try to allocate a subbuffer
/// larger than the current size of an arena, the arenas are automatically resized.
///
/// No memory is allocated when the allocator is created, be it on the Vulkan or Rust side. That
/// only happens once you allocate a subbuffer.
///
/// # Usage
///
/// Ideally, one arena should be able to fit all data you need to update per frame, so that each
/// arena is submitted and freed once per frame. This way, the arena pool would also contain as
/// many arenas as there are frames in flight on the thread. Otherwise, if your arenas are not able
/// to fit everything each frame, what will likely happen is that each subbuffer will be
/// allocated from an individual arena. This can impact efficiency both in terms of memory usage
/// (because each arena has the same size, even if some of the subbuffers are way smaller) as well
/// as performance, because the data could end up more physically separated in memory, which means
/// the GPU would need to hop from place to place a lot more during a frame.
///
/// Ideally the result is something roughly like this:
///
/// ```plain
/// +---------------------------------------------------------------------------------------------+
/// |                                        Memory Block                                         |
/// |-----+------+-----------------------+---------+-----------------------+------+---------+-----|
/// |     |      |     Frame 1 Arena     |         |     Frame 2 Arena     |      |         |     |
/// | ••• | Tex. |-------+-------+-------| Attach. |-------+-------+-------| Tex. | Attach. | ••• |
/// |     |      | Vert. | Index | Unif. |         | Vert. | Index | Unif. |      |         |     |
/// +-----+------+-------+-------+-------+---------+-------+-------+-------+------+---------+-----+
/// ```
///
/// Download or device-only usage is much the same. Try to make the arenas fit all the data you
/// need to store at once.
///
/// # Examples
///
/// ```
/// use vulkano::{
///     buffer::{
///         allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
///         BufferUsage,
///     },
///     command_buffer::{
///         AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
///     },
///     memory::allocator::MemoryTypeFilter,
///     sync::GpuFuture,
/// };
///
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let command_buffer_allocator: std::sync::Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator> = return;
/// #
/// // Create the buffer allocator.
/// let buffer_allocator = SubbufferAllocator::new(
///     memory_allocator.clone(),
///     SubbufferAllocatorCreateInfo {
///         buffer_usage: BufferUsage::TRANSFER_SRC,
///         memory_type_filter: MemoryTypeFilter::PREFER_HOST
///             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
///         ..Default::default()
///     },
/// );
///
/// for n in 0..25u32 {
///     // Each loop allocates a new subbuffer and stores `data` in it.
///     let data: [f32; 4] = [1.0, 0.5, n as f32 / 24.0, 0.0];
///     let subbuffer = buffer_allocator.allocate_sized().unwrap();
///     *subbuffer.write().unwrap() = data;
///
///     // You can then use `subbuffer` as if it was an entirely separate buffer.
///     AutoCommandBufferBuilder::primary(
///         command_buffer_allocator.clone(),
///         queue.queue_family_index(),
///         CommandBufferUsage::OneTimeSubmit,
///     )
///     .unwrap()
///     // For the sake of the example we just call `update_buffer` on the buffer, even though
///     // it is pointless to do that.
///     .update_buffer(subbuffer.clone(), &[0.2, 0.3, 0.4, 0.5])
///     .unwrap()
///     .build()
///     .unwrap()
///     .execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap();
/// }
/// ```
#[deprecated(since = "0.36.0", note = "use `BufferAllocator` instead")]
#[derive(Debug)]
pub struct SubbufferAllocator<A = StandardMemoryAllocator> {
    state: UnsafeCell<SubbufferAllocatorState<A>>,
}

#[allow(deprecated)]
impl<A> SubbufferAllocator<A>
where
    A: MemoryAllocator,
{
    /// Creates a new `SubbufferAllocator`.
    pub fn new(memory_allocator: Arc<A>, create_info: SubbufferAllocatorCreateInfo) -> Self {
        let SubbufferAllocatorCreateInfo {
            arena_size,
            buffer_usage,
            memory_type_filter,
            _ne: _,
        } = create_info;

        let properties = memory_allocator.device().physical_device().properties();
        let buffer_alignment = [
            buffer_usage
                .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER)
                .then_some(properties.min_texel_buffer_offset_alignment),
            buffer_usage
                .contains(BufferUsage::UNIFORM_BUFFER)
                .then_some(properties.min_uniform_buffer_offset_alignment),
            buffer_usage
                .contains(BufferUsage::STORAGE_BUFFER)
                .then_some(properties.min_storage_buffer_offset_alignment),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(DeviceAlignment::MIN);

        SubbufferAllocator {
            state: UnsafeCell::new(SubbufferAllocatorState {
                memory_allocator,
                buffer_usage,
                memory_type_filter,
                buffer_alignment,
                arena_size,
                arena: None,
                free_start: 0,
                reserve: None,
            }),
        }
    }

    /// Returns the current size of the arenas.
    pub fn arena_size(&self) -> DeviceSize {
        unsafe { &*self.state.get() }.arena_size
    }

    /// Sets the arena size to the provided `size`.
    ///
    /// The next time you allocate a subbuffer, a new arena will be allocated with the new size,
    /// and all subsequently allocated arenas will also share the new size.
    pub fn set_arena_size(&self, size: DeviceSize) {
        let state_ptr = self.state.get();
        let state = unsafe { &mut *state_ptr };
        state.arena_size = size;
        state.arena = None;
        state.reserve = None;
    }

    /// Ensures that the size of the current arena is at least `size`.
    ///
    /// If `size` is greater than the current arena size, then a new arena will be allocated with
    /// the new size, and all subsequently allocated arenas will also share the new size. Otherwise
    /// this has no effect.
    pub fn reserve(&self, size: DeviceSize) -> Result<(), MemoryAllocatorError> {
        if size > self.arena_size() {
            let state_ptr = self.state.get();
            let state = unsafe { &mut *state_ptr };
            state.arena_size = size;
            state.reserve = None;
            state.arena = Some(state.next_arena()?);
        }

        Ok(())
    }

    /// Allocates a subbuffer for sized data.
    pub fn allocate_sized<T>(&self) -> Result<Subbuffer<T>, MemoryAllocatorError>
    where
        T: BufferContents,
    {
        let layout = T::LAYOUT.unwrap_sized();

        let state_ptr = self.state.get();
        let state = unsafe { &mut *state_ptr };
        state
            .allocate(layout)
            .map(|subbuffer| unsafe { subbuffer.reinterpret_unchecked() })
    }

    /// Allocates a subbuffer for a slice.
    ///
    /// # Panics
    ///
    /// - Panics if `len` is zero.
    pub fn allocate_slice<T>(&self, len: DeviceSize) -> Result<Subbuffer<[T]>, MemoryAllocatorError>
    where
        T: BufferContents,
    {
        self.allocate_unsized(len)
    }

    /// Allocates a subbuffer for unsized data.
    ///
    /// # Panics
    ///
    /// - Panics if `len` is zero.
    pub fn allocate_unsized<T>(&self, len: DeviceSize) -> Result<Subbuffer<T>, MemoryAllocatorError>
    where
        T: BufferContents + ?Sized,
    {
        let layout = T::LAYOUT.layout_for_len(len).unwrap();

        unsafe { &mut *self.state.get() }
            .allocate(layout)
            .map(|subbuffer| unsafe { subbuffer.reinterpret_unchecked() })
    }

    /// Allocates a subbuffer with the given `layout`.
    pub fn allocate(&self, layout: DeviceLayout) -> Result<Subbuffer<[u8]>, MemoryAllocatorError> {
        unsafe { &mut *self.state.get() }.allocate(layout)
    }
}

#[allow(deprecated)]
unsafe impl<A> DeviceOwned for SubbufferAllocator<A>
where
    A: MemoryAllocator,
{
    fn device(&self) -> &Arc<Device> {
        unsafe { &*self.state.get() }.memory_allocator.device()
    }
}

#[derive(Debug)]
struct SubbufferAllocatorState<A> {
    memory_allocator: Arc<A>,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
    // The alignment required for the subbuffers.
    buffer_alignment: DeviceAlignment,
    // The current size of the arenas.
    arena_size: DeviceSize,
    // Contains the buffer that is currently being suballocated.
    arena: Option<Arc<Arena>>,
    // Offset pointing to the start of free memory within the arena.
    free_start: DeviceSize,
    // When an `Arena` is dropped, it returns itself here for reuse.
    reserve: Option<Arc<ArrayQueue<Arc<Buffer>>>>,
}

impl<A> SubbufferAllocatorState<A>
where
    A: MemoryAllocator,
{
    fn allocate(&mut self, layout: DeviceLayout) -> Result<Subbuffer<[u8]>, MemoryAllocatorError> {
        let size = layout.size();
        let alignment = cmp::max(layout.alignment(), self.buffer_alignment);

        loop {
            if self.arena.is_none() {
                // If the requested size is larger than the arenas, we need to resize them.
                if self.arena_size < size {
                    self.arena_size = size * 2;
                    // We need to drop our reference to the old pool to make sure the arenas are
                    // dropped once no longer in use, and replace it with a new pool that will not
                    // be polluted with the outdates arenas.
                    self.reserve = None;
                }
                self.arena = Some(self.next_arena()?);
                self.free_start = 0;
            }

            let arena = self.arena.as_ref().unwrap();
            let allocation = match arena.buffer.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse | BufferMemory::External => unreachable!(),
            };
            let arena_offset = allocation.offset();
            let atom_size = allocation.atom_size().unwrap_or(DeviceAlignment::MIN);

            let alignment = cmp::max(alignment, atom_size);
            let offset = align_up(arena_offset + self.free_start, alignment);

            if offset + size <= arena_offset + self.arena_size {
                let offset = offset - arena_offset;
                self.free_start = offset + size;

                return Ok(Subbuffer::from_arena(arena.clone(), offset, layout.size()));
            }

            // We reached the end of the arena, grab the next one.
            self.arena = None;
        }
    }

    fn next_arena(&mut self) -> Result<Arc<Arena>, MemoryAllocatorError> {
        if self.reserve.is_none() {
            self.reserve = Some(Arc::new(ArrayQueue::new(MAX_ARENAS)));
        }
        let reserve = self.reserve.as_ref().unwrap();

        reserve
            .pop()
            .map(Ok)
            .unwrap_or_else(|| self.create_arena())
            .map(|buffer| {
                Arc::new(Arena {
                    buffer: ManuallyDrop::new(DeviceOwnedDebugWrapper(buffer)),
                    reserve: reserve.clone(),
                })
            })
    }

    fn create_arena(&self) -> Result<Arc<Buffer>, MemoryAllocatorError> {
        Buffer::try_new(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: self.buffer_usage,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: self.memory_type_filter,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(self.arena_size, 1).unwrap(),
        )
        .map_err(|err| match err {
            Validated::Error(AllocateBufferError::AllocateMemory(err)) => err,
            // We don't use sparse-binding, concurrent sharing or external memory, therefore the
            // other errors can't happen.
            _ => unreachable!("{err:?}"),
        })
    }
}

#[derive(Debug)]
pub(super) struct Arena {
    buffer: ManuallyDrop<DeviceOwnedDebugWrapper<Arc<Buffer>>>,
    // Where we return the arena in our `Drop` impl.
    reserve: Arc<ArrayQueue<Arc<Buffer>>>,
}

impl Arena {
    pub(super) fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) }.0;
        let _ = self.reserve.push(buffer);
    }
}

impl PartialEq for Arena {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Arena {}

impl Hash for Arena {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
    }
}

/// Parameters to create a new [`SubbufferAllocator`].
#[derive(Clone, Debug)]
pub struct SubbufferAllocatorCreateInfo {
    /// Initial size of an arena in bytes.
    ///
    /// Ideally this should fit all the data you need to update per frame. So for example, if you
    /// need to allocate buffers of size 1K, 2K and 5K each frame, then this should be 8K. If your
    /// data is dynamically-sized then try to make an educated guess or simply leave the default.
    ///
    /// The default value is `0`.
    pub arena_size: DeviceSize,

    /// The buffer usage that all allocated buffers should have.
    ///
    /// The default value is empty, which must be overridden.
    pub buffer_usage: BufferUsage,

    /// The memory type filter all buffers should be allocated with.
    ///
    /// The default value is [`MemoryTypeFilter::PREFER_DEVICE`].
    pub memory_type_filter: MemoryTypeFilter,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for SubbufferAllocatorCreateInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SubbufferAllocatorCreateInfo {
    /// Returns a default `SubbufferAllocatorCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        SubbufferAllocatorCreateInfo {
            arena_size: 0,
            buffer_usage: BufferUsage::empty(),
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            _ne: crate::NE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(deprecated)]
    #[test]
    fn reserve() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer_allocator = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
        );
        assert_eq!(buffer_allocator.arena_size(), 0);

        buffer_allocator.reserve(83).unwrap();
        assert_eq!(buffer_allocator.arena_size(), 83);
    }

    #[allow(deprecated)]
    #[test]
    fn capacity_increase() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let buffer_allocator = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
        );
        assert_eq!(buffer_allocator.arena_size(), 0);

        buffer_allocator.allocate_sized::<u32>().unwrap();
        assert_eq!(buffer_allocator.arena_size(), 8);
    }
}
