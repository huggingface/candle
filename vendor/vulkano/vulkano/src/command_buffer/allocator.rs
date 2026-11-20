//! Traits and types for managing the allocation of command buffers and command pools.
//!
//! In Vulkano, creating a command buffer requires passing an implementation of the
//! [`CommandBufferAllocator`] trait. You can implement this trait yourself, or use the
//! Vulkano-provided [`StandardCommandBufferAllocator`].

use super::{
    pool::{
        CommandBufferAllocateInfo, CommandPool, CommandPoolAlloc, CommandPoolCreateFlags,
        CommandPoolCreateInfo, CommandPoolResetFlags,
    },
    CommandBufferLevel,
};
use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    Validated, ValidationError, VulkanError,
};
use crossbeam_queue::ArrayQueue;
use smallvec::SmallVec;
use std::{
    cell::UnsafeCell,
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    ptr,
    sync::{Arc, Weak},
};
use thread_local::ThreadLocal;

const MAX_POOLS: usize = 32;

/// Types that manage the memory of command buffers.
///
/// # Safety
///
/// A Vulkan command pool must be externally synchronized as if it owned the command buffers that
/// were allocated from it. This includes allocating from the pool, freeing from the pool,
/// resetting the pool or individual command buffers, and most importantly recording commands to
/// command buffers. The implementation of `CommandBufferAllocator` is expected to manage this.
///
/// The implementation of `allocate` must return a valid allocation that stays allocated until
/// either `deallocate` is called on it or the allocator is dropped. If the allocator is cloned, it
/// must produce the same allocator, and an allocation must stay allocated until either
/// `deallocate` is called on any of the clones or all clones have been dropped.
///
/// The implementation of `deallocate` is expected to free the command buffer, reset the command
/// buffer or its pool, or add it to a pool so that it gets reused. If the implementation frees the
/// command buffer or resets the command buffer or pool, it must not forget that this operation
/// must be externally synchronized. The implementation should not panic as it is used when
/// dropping command buffers.
///
/// Command buffers in the recording state can never be sent between threads in vulkano, which
/// means that the implementation of `allocate` can freely assume that the command buffer won't
/// leave the thread it was allocated on until it has finished recording. Note however that after
/// recording is finished, command buffers are free to be sent between threads, which means that
/// `deallocate` must account for the possibility that a command buffer can be deallocated from a
/// different thread than it was allocated from.
pub unsafe trait CommandBufferAllocator: DeviceOwned + Send + Sync + 'static {
    /// Allocates a command buffer.
    fn try_allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
    ) -> Result<CommandBufferAlloc, Validated<VulkanError>>;

    /// Allocates a command buffer without doing any checks.
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
    /// - If calling [`try_allocate`] with the same arguments would return a [`ValidationError`],
    ///   calling this method is *undefined behavior*!
    ///
    /// [`try_allocate`]: Self::try_allocate
    unsafe fn allocate_unchecked(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
    ) -> Result<CommandBufferAlloc, VulkanError>;

    /// Deallocates the given `allocation`.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `self`.
    unsafe fn deallocate(&self, allocation: CommandBufferAlloc);

    /// Coerces a reference to the allocator to a reference to a trait object.
    fn as_dyn(self: Arc<Self>) -> Arc<dyn CommandBufferAllocator>;
}

impl Debug for dyn CommandBufferAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("CommandBufferAllocator")
            .finish_non_exhaustive()
    }
}

/// An allocation made using a [command buffer allocator].
///
/// [command buffer allocator]: CommandBufferAllocator
#[derive(Debug)]
pub struct CommandBufferAlloc {
    /// The internal object that contains the command buffer.
    pub inner: CommandPoolAlloc,

    /// The command pool that the command buffer was allocated from.
    ///
    /// Using this for anything other than looking at the pool's metadata will lead to a Bad
    /// Time<sup>TM</sup>.
    pub pool: Arc<CommandPool>,

    /// An opaque handle identifying the allocation inside the allocator.
    pub handle: AllocationHandle,
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

/// Standard implementation of a command buffer allocator.
///
/// The intended way to use this allocator is to have one that is used globally for the duration of
/// the program, in order to avoid creating and destroying [`CommandPool`]s, as that is expensive.
/// Alternatively, you can have one locally on a thread for the duration of the thread.
///
/// Internally, this allocator keeps one or more `CommandPool`s per queue family index per thread,
/// using Thread-Local Storage. When a thread first allocates, an entry is reserved for the thread
/// and queue family combination. After a thread exits and the allocator wasn't dropped yet, its
/// entries are freed, but the pools it used are not dropped. The next time a new thread allocates
/// for the first time, the entries are reused along with the pools. If all threads drop their
/// reference to the allocator, all entries along with the allocator are dropped, even if the
/// threads didn't exit yet, which is why you should keep the allocator alive for as long as you
/// need to allocate so that the pools can keep being reused.
///
/// This allocator only needs to lock when a thread first allocates or when a thread that
/// previously allocated exits. In all other cases, allocation is lock-free.
#[derive(Debug)]
pub struct StandardCommandBufferAllocator {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    // Each queue family index points directly to its entry.
    pools: ThreadLocal<SmallVec<[UnsafeCell<Option<Entry>>; 8]>>,
    buffer_count: [usize; 2],
    command_pool_create_flags: CommandPoolCreateFlags,
}

impl StandardCommandBufferAllocator {
    /// Creates a new `StandardCommandBufferAllocator`.
    #[inline]
    pub fn new(
        device: &Arc<Device>,
        create_info: &StandardCommandBufferAllocatorCreateInfo<'_>,
    ) -> Self {
        let mut buffer_count = [0, 0];
        buffer_count[CommandBufferLevel::Primary as usize] = create_info.primary_buffer_count;
        buffer_count[CommandBufferLevel::Secondary as usize] = create_info.secondary_buffer_count;

        StandardCommandBufferAllocator {
            device: InstanceOwnedDebugWrapper(device.clone()),
            pools: ThreadLocal::new(),
            buffer_count,
            command_pool_create_flags: create_info.command_pool_create_flags,
        }
    }

    /// Tries to reset the [`CommandPool`] that's currently in use for the given queue family index
    /// on the current thread.
    ///
    /// If successful, the memory of the pool can be reused again along with all command buffers
    /// allocated from it. This is only possible if all command buffers allocated from the pool
    /// have been dropped.
    ///
    /// This has no effect if the entry wasn't initialized yet or if the entry was [cleared].
    ///
    /// # Panics
    ///
    /// - Panics if `queue_family_index` is not less than the number of queue families.
    ///
    /// [cleared]: Self::clear
    #[inline]
    pub fn try_reset_pool(
        &self,
        queue_family_index: u32,
        flags: CommandPoolResetFlags,
    ) -> Result<(), Validated<ResetCommandPoolError>> {
        let entry_ptr = self.entry(queue_family_index);

        if let Some(entry) = unsafe { &mut *entry_ptr }.as_mut() {
            entry.try_reset_pool(flags)
        } else {
            Ok(())
        }
    }

    /// Clears the entry for the given queue family index and the current thread. This does not
    /// mean that the pools are dropped immediately. A pool is kept alive for as long as command
    /// buffers allocated from it exist.
    ///
    /// This has no effect if the entry was not initialized yet.
    ///
    /// # Panics
    ///
    /// - Panics if `queue_family_index` is not less than the number of queue families.
    #[inline]
    pub fn clear(&self, queue_family_index: u32) {
        let entry_ptr = self.entry(queue_family_index);
        unsafe { *entry_ptr = None };
    }

    fn entry(&self, queue_family_index: u32) -> *mut Option<Entry> {
        let pools = self.pools.get_or(|| {
            self.device
                .physical_device()
                .queue_family_properties()
                .iter()
                .map(|_| UnsafeCell::new(None))
                .collect()
        });

        pools[queue_family_index as usize].get()
    }
}

unsafe impl CommandBufferAllocator for StandardCommandBufferAllocator {
    #[inline]
    fn try_allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
    ) -> Result<CommandBufferAlloc, Validated<VulkanError>> {
        let entry_ptr = self.entry(queue_family_index);
        let entry = unsafe { &mut *entry_ptr };
        let command_pool_create_info = CommandPoolCreateInfo {
            flags: self.command_pool_create_flags,
            queue_family_index,
            ..Default::default()
        };

        if entry.is_none() {
            *entry = Some(Entry::try_new(
                &self.device,
                &command_pool_create_info,
                &self.buffer_count,
                Arc::new(ArrayQueue::new(MAX_POOLS)),
            )?);
        }

        let entry = entry.as_mut().unwrap();

        entry.try_allocate(&command_pool_create_info, level, &self.buffer_count)
    }

    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
    ) -> Result<CommandBufferAlloc, VulkanError> {
        let entry_ptr = self.entry(queue_family_index);
        let entry = unsafe { &mut *entry_ptr };
        let command_pool_create_info = CommandPoolCreateInfo {
            flags: self.command_pool_create_flags,
            queue_family_index,
            ..Default::default()
        };

        if entry.is_none() {
            // SAFETY: Enforced by the caller.
            *entry = Some(unsafe {
                Entry::new_unchecked(
                    &self.device,
                    &command_pool_create_info,
                    &self.buffer_count,
                    Arc::new(ArrayQueue::new(MAX_POOLS)),
                )
            }?);
        }

        let entry = entry.as_mut().unwrap();

        // SAFETY: Enforced by the caller.
        unsafe { entry.allocate_unchecked(&command_pool_create_info, level, &self.buffer_count) }
    }

    #[inline]
    unsafe fn deallocate(&self, allocation: CommandBufferAlloc) {
        let ptr = allocation.handle.as_ptr().cast::<Pool>();

        // SAFETY: The caller must guarantee that `allocation` refers to one allocated by `self`,
        // therefore `ptr` must be the same one we gave out on allocation. We also know that the
        // pointer must be valid, because the caller must guarantee that the same allocation isn't
        // deallocated more than once. That means that since we cloned the `Arc` on allocation, at
        // least that strong reference must still keep it alive, and we can safely drop this clone
        // at the end of the scope here.
        let pool = unsafe { Arc::from_raw(ptr) };

        let level = allocation.inner.level();

        // This cannot panic because in order to have allocated a command buffer with the level in
        // the first place, the size of the pool for that level must have been non-zero.
        let buffer_reserve = pool.buffer_reserve[level as usize].as_ref().unwrap();

        let res = buffer_reserve.push(allocation.inner);

        // This cannot happen because every allocation is (supposed to be) returned to the pool
        // whence it came, so there must be enough room for it.
        debug_assert!(res.is_ok());

        // We have to make sure that we only reset the pool under this condition, because there
        // could be other references in other allocations.
        if Arc::strong_count(&pool) == 1 {
            // The pool reserve can be dropped from under us when an entry is cleared, in which
            // case we destroy the pool.
            if let Some(reserve) = pool.pool_reserve.upgrade() {
                // If there is not enough space in the reserve, we destroy the pool. The only way
                // this can happen is if something is resource hogging, forcing new pools to be
                // created such that the number exceeds `MAX_POOLS`, and then drops them all at
                // once.
                let _ = reserve.push(pool);
            }
        }
    }

    #[inline]
    fn as_dyn(self: Arc<Self>) -> Arc<dyn CommandBufferAllocator> {
        self
    }
}

unsafe impl<T: CommandBufferAllocator + ?Sized> CommandBufferAllocator for Arc<T> {
    #[inline]
    fn try_allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
    ) -> Result<CommandBufferAlloc, Validated<VulkanError>> {
        (**self).try_allocate(queue_family_index, level)
    }

    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
    ) -> Result<CommandBufferAlloc, VulkanError> {
        unsafe { (**self).allocate_unchecked(queue_family_index, level) }
    }

    #[inline]
    unsafe fn deallocate(&self, allocation: CommandBufferAlloc) {
        unsafe { (**self).deallocate(allocation) }
    }

    #[inline]
    fn as_dyn(self: Arc<Self>) -> Arc<dyn CommandBufferAllocator> {
        self
    }
}

unsafe impl DeviceOwned for StandardCommandBufferAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
struct Entry {
    // Contains the actual Vulkan command pool that is currently in use.
    pool: Arc<Pool>,
    // How many command buffers have been allocated from `pool.buffer_reserve`.
    allocations: [usize; 2],
    // When a `Pool` is about to be dropped, it is returned here for reuse.
    pool_reserve: Arc<ArrayQueue<Arc<Pool>>>,
}

// This is needed because of the blanket impl of `Send` on `Arc<T>`, which requires that `T` is
// `Send + Sync`. `Pool` is `Send + !Sync` because `CommandPool` is `!Sync`. That's fine however
// because we never access the Vulkan command pool concurrently.
unsafe impl Send for Entry {}

impl Entry {
    fn try_new(
        device: &Arc<Device>,
        command_pool_create_info: &CommandPoolCreateInfo<'_>,
        buffer_count: &[usize; 2],
        pool_reserve: Arc<ArrayQueue<Arc<Pool>>>,
    ) -> Result<Self, Validated<VulkanError>> {
        Ok(Entry {
            pool: Pool::try_new(
                device,
                command_pool_create_info,
                buffer_count,
                &pool_reserve,
            )?,
            allocations: [0; 2],
            pool_reserve,
        })
    }

    unsafe fn new_unchecked(
        device: &Arc<Device>,
        command_pool_create_info: &CommandPoolCreateInfo<'_>,
        buffer_count: &[usize; 2],
        pool_reserve: Arc<ArrayQueue<Arc<Pool>>>,
    ) -> Result<Self, VulkanError> {
        Ok(Entry {
            // SAFETY: Enforced by the caller.
            pool: unsafe {
                Pool::new_unchecked(
                    device,
                    command_pool_create_info,
                    buffer_count,
                    &pool_reserve,
                )
            }?,
            allocations: [0; 2],
            pool_reserve,
        })
    }

    fn try_allocate(
        &mut self,
        command_pool_create_info: &CommandPoolCreateInfo<'_>,
        level: CommandBufferLevel,
        buffer_count: &[usize; 2],
    ) -> Result<CommandBufferAlloc, Validated<VulkanError>> {
        if self.allocations[level as usize] >= buffer_count[level as usize] {
            // This can happen if there's only ever one allocation alive at any point in time. In
            // that case, when deallocating the last command buffer before reaching `buffer_count`,
            // there will be 2 references to the pool (one here and one in the allocation) and so
            // the pool won't be returned to the reserve when deallocating. However, since there
            // are no other allocations alive, there would be no other allocations that could
            // return it to the reserve. To avoid dropping the pool unnecessarily, we simply
            // continue using it. In the case where there are other references, we drop ours, at
            // which point an allocation still holding a reference will be able to put the pool
            // into the reserve when deallocated.
            //
            // TODO: This can still run into the A/B/A problem causing the pool to be dropped.
            if Arc::strong_count(&self.pool) == 1 {
                // SAFETY: We checked that the pool has a single strong reference above, meaning
                // that all the allocations we gave out must have been deallocated.
                unsafe { self.pool.inner.try_reset(CommandPoolResetFlags::empty()) }?;

                self.allocations = [0; 2];
            } else {
                if let Some(pool) = self.pool_reserve.pop() {
                    // SAFETY: We checked that the pool has a single strong reference when
                    // deallocating, meaning that all the allocations we gave out must have been
                    // deallocated.
                    unsafe { pool.inner.try_reset(CommandPoolResetFlags::empty()) }?;

                    self.pool = pool;
                    self.allocations = [0; 2];
                } else {
                    *self = Entry::try_new(
                        self.pool.inner.device(),
                        command_pool_create_info,
                        buffer_count,
                        self.pool_reserve.clone(),
                    )?;
                }
            }
        }

        let Some(buffer_reserve) = self.pool.buffer_reserve[level as usize].as_ref() else {
            return Err(Validated::ValidationError(Box::new(ValidationError {
                problem: format!(
                    "attempted to allocate a command buffer with level `{level:?}`, but the \
                    command buffer pool for that level was configured to be empty",
                )
                .into(),
                ..Default::default()
            })));
        };

        self.allocations[level as usize] += 1;

        Ok(CommandBufferAlloc {
            inner: buffer_reserve.pop().unwrap(),
            pool: self.pool.inner.clone(),
            handle: AllocationHandle::from_ptr(Arc::into_raw(self.pool.clone()) as _),
        })
    }

    unsafe fn allocate_unchecked(
        &mut self,
        command_pool_create_info: &CommandPoolCreateInfo<'_>,
        level: CommandBufferLevel,
        buffer_count: &[usize; 2],
    ) -> Result<CommandBufferAlloc, VulkanError> {
        if self.allocations[level as usize] >= buffer_count[level as usize] {
            // This can happen if there's only ever one allocation alive at any point in time. In
            // that case, when deallocating the last command buffer before reaching `buffer_count`,
            // there will be 2 references to the pool (one here and one in the allocation) and so
            // the pool won't be returned to the reserve when deallocating. However, since there
            // are no other allocations alive, there would be no other allocations that could
            // return it to the reserve. To avoid dropping the pool unnecessarily, we simply
            // continue using it. In the case where there are other references, we drop ours, at
            // which point an allocation still holding a reference will be able to put the pool
            // into the reserve when deallocated.
            //
            // TODO: This can still run into the A/B/A problem causing the pool to be dropped.
            if Arc::strong_count(&self.pool) == 1 {
                // SAFETY: We checked that the pool has a single strong reference above, meaning
                // that all the allocations we gave out must have been deallocated.
                unsafe {
                    self.pool
                        .inner
                        .reset_unchecked(CommandPoolResetFlags::empty())
                }?;

                self.allocations = [0; 2];
            } else {
                if let Some(pool) = self.pool_reserve.pop() {
                    // SAFETY: We checked that the pool has a single strong reference when
                    // deallocating, meaning that all the allocations we gave out must have been
                    // deallocated.
                    unsafe { pool.inner.reset_unchecked(CommandPoolResetFlags::empty()) }?;

                    self.pool = pool;
                    self.allocations = [0; 2];
                } else {
                    // SAFETY: Enforced by the caller.
                    *self = unsafe {
                        Entry::new_unchecked(
                            self.pool.inner.device(),
                            command_pool_create_info,
                            buffer_count,
                            self.pool_reserve.clone(),
                        )
                    }?;
                }
            }
        }

        let buffer_reserve = self.pool.buffer_reserve[level as usize].as_ref();

        // SAFETY: Enforced by the caller.
        let buffer_reserve = unsafe { buffer_reserve.unwrap_unchecked() };

        self.allocations[level as usize] += 1;

        Ok(CommandBufferAlloc {
            inner: buffer_reserve.pop().unwrap(),
            pool: self.pool.inner.clone(),
            handle: AllocationHandle::from_ptr(Arc::into_raw(self.pool.clone()) as _),
        })
    }

    fn try_reset_pool(
        &mut self,
        flags: CommandPoolResetFlags,
    ) -> Result<(), Validated<ResetCommandPoolError>> {
        if let Some(pool) = Arc::get_mut(&mut self.pool) {
            unsafe { pool.inner.try_reset(flags) }
                .map_err(|err| err.map(ResetCommandPoolError::VulkanError))?;

            self.allocations = [0; 2];

            Ok(())
        } else {
            Err(ResetCommandPoolError::InUse.into())
        }
    }
}

#[derive(Debug)]
struct Pool {
    // The Vulkan pool specific to a device's queue family.
    inner: Arc<CommandPool>,
    // List of existing command buffers that are available for reuse.
    buffer_reserve: [Option<ArrayQueue<CommandPoolAlloc>>; 2],
    // Where to return this pool once there are no more current allocations.
    pool_reserve: Weak<ArrayQueue<Arc<Self>>>,
}

impl Pool {
    fn try_new(
        device: &Arc<Device>,
        command_pool_create_info: &CommandPoolCreateInfo<'_>,
        buffer_counts: &[usize; 2],
        pool_reserve: &Arc<ArrayQueue<Arc<Self>>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        let inner = CommandPool::try_new(device, command_pool_create_info)?;

        let levels = [CommandBufferLevel::Primary, CommandBufferLevel::Secondary];
        let mut buffer_reserve = [None, None];

        for (level, &buffer_count) in levels.into_iter().zip(buffer_counts) {
            if buffer_count == 0 {
                continue;
            }

            let pool = ArrayQueue::new(buffer_count);

            let allocate_info = CommandBufferAllocateInfo {
                level,
                command_buffer_count: buffer_count.try_into().unwrap(),
                ..Default::default()
            };

            for allocation in inner.try_allocate_command_buffers(&allocate_info)? {
                let _ = pool.push(allocation);
            }

            buffer_reserve[level as usize] = Some(pool);
        }

        Ok(Arc::new(Pool {
            inner: Arc::new(inner),
            buffer_reserve,
            pool_reserve: Arc::downgrade(pool_reserve),
        }))
    }

    unsafe fn new_unchecked(
        device: &Arc<Device>,
        command_pool_create_info: &CommandPoolCreateInfo<'_>,
        buffer_counts: &[usize; 2],
        pool_reserve: &Arc<ArrayQueue<Arc<Self>>>,
    ) -> Result<Arc<Self>, VulkanError> {
        // SAFETY: Enforced by the caller.
        let inner = unsafe { CommandPool::new_unchecked(device, command_pool_create_info) }?;

        let levels = [CommandBufferLevel::Primary, CommandBufferLevel::Secondary];
        let mut buffer_reserve = [None, None];

        for (level, &buffer_count) in levels.into_iter().zip(buffer_counts) {
            if buffer_count == 0 {
                continue;
            }

            let pool = ArrayQueue::new(buffer_count);

            let allocate_info = CommandBufferAllocateInfo {
                level,
                command_buffer_count: buffer_count.try_into().unwrap(),
                ..Default::default()
            };

            for allocation in unsafe { inner.allocate_command_buffers_unchecked(&allocate_info) }? {
                let _ = pool.push(allocation);
            }

            buffer_reserve[level as usize] = Some(pool);
        }

        Ok(Arc::new(Pool {
            inner: Arc::new(inner),
            buffer_reserve,
            pool_reserve: Arc::downgrade(pool_reserve),
        }))
    }
}

/// Parameters to create a new [`StandardCommandBufferAllocator`].
#[derive(Clone, Debug)]
pub struct StandardCommandBufferAllocatorCreateInfo<'a> {
    /// How many primary command buffers should be allocated per pool.
    ///
    /// Each time a thread allocates using some queue family index, and either no pools were
    /// initialized yet or all pools are full, a new pool is created for that thread and queue
    /// family combination. This option tells the allocator how many primary command buffers should
    /// be allocated for that pool. It always allocates exactly this many command buffers at once
    /// for the pool, as that is more performant than allocating them one-by-one. What this means
    /// is that you should make sure that this is not too large, so that you don't end up wasting
    /// too much memory. You also don't want this to be too low, because that on the other hand
    /// would mean that the pool would have to be reset more often, or that more pools would need
    /// to be created, depending on the lifetime of the command buffers.
    ///
    /// The default value is `32`.
    pub primary_buffer_count: usize,

    /// Same as `primary_buffer_count` except for secondary command buffers.
    ///
    /// The default value is `0`.
    pub secondary_buffer_count: usize,

    /// Used to set the flags for pools that the allocator creates.
    ///
    /// The default value is empty.
    pub command_pool_create_flags: CommandPoolCreateFlags,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for StandardCommandBufferAllocatorCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StandardCommandBufferAllocatorCreateInfo<'_> {
    /// Returns a default `StandardCommandBufferAllocatorCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        StandardCommandBufferAllocatorCreateInfo {
            primary_buffer_count: 32,
            secondary_buffer_count: 0,
            command_pool_create_flags: CommandPoolCreateFlags::empty(),
            _ne: crate::NE,
        }
    }
}

/// Error that can be returned when resetting a [`CommandPool`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResetCommandPoolError {
    /// A runtime error occurred.
    VulkanError(VulkanError),

    /// The `CommandPool` is still in use.
    InUse,
}

impl Error for ResetCommandPoolError {}

impl Display for ResetCommandPoolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VulkanError(_) => write!(f, "a runtime error occurred"),
            Self::InUse => write!(f, "the command pool is still in use"),
        }
    }
}

impl From<VulkanError> for ResetCommandPoolError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl From<ResetCommandPoolError> for Validated<ResetCommandPoolError> {
    fn from(err: ResetCommandPoolError) -> Self {
        Self::Error(err)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CommandBufferAllocator, StandardCommandBufferAllocator,
        StandardCommandBufferAllocatorCreateInfo,
    };
    use crate::{
        command_buffer::{pool::CommandPoolCreateFlags, CommandBufferLevel},
        Version,
    };

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo {
                ..Default::default()
            },
        );
    }

    #[test]
    fn basic_alloc() {
        let (device, queue) = gfx_dev_and_queue!();
        let allocator = StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo {
                ..Default::default()
            },
        );
        let buffer = allocator
            .try_allocate(queue.queue_family_index(), CommandBufferLevel::Primary)
            .unwrap();
        assert_eq!(buffer.pool.flags(), CommandPoolCreateFlags::empty());
    }

    #[test]
    fn alloc_with_command_pool_create_flags() {
        let (device, queue) = gfx_dev_and_queue!();
        let desired_flags = {
            let mut flags = CommandPoolCreateFlags::TRANSIENT;

            if device.api_version() >= Version::V1_1 {
                flags |= CommandPoolCreateFlags::PROTECTED;
            }
            flags
        };
        let allocator = StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo {
                command_pool_create_flags: desired_flags,
                ..Default::default()
            },
        );
        let buffer = allocator
            .try_allocate(queue.queue_family_index(), CommandBufferLevel::Primary)
            .unwrap();
        assert_eq!(buffer.pool.flags(), desired_flags);
    }
}
