use super::{Device, DeviceOwned, QueueCreateFlags};
use crate::{
    command_buffer::{CommandBufferSubmitInfo, SemaphoreSubmitInfo, SubmitInfo},
    instance::{debug::DebugUtilsLabel, InstanceOwnedDebugWrapper},
    macros::{impl_id_counter, vulkan_bitflags},
    memory::sparse::BindSparseInfo,
    swapchain::{PresentInfo, SwapchainPresentInfo},
    sync::{fence::Fence, PipelineStages},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ash::vk;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    mem::{self, MaybeUninit},
    num::NonZero,
    sync::Arc,
};

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: vk::Queue,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: QueueCreateFlags,
    queue_family_index: u32,
    queue_index: u32, // index within family

    lock: Arc<dyn QueueMutex>,
}

impl Queue {
    pub(super) unsafe fn new(device: &Arc<Device>, queue_info: &DeviceQueueInfo<'_>) -> Arc<Self> {
        let queue_info_vk = queue_info.to_vk();

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();

            if device.api_version() >= Version::V1_1 {
                unsafe {
                    (fns.v1_1.get_device_queue2)(
                        device.handle(),
                        &queue_info_vk,
                        output.as_mut_ptr(),
                    )
                };
            } else {
                debug_assert!(queue_info_vk.flags.is_empty());
                debug_assert!(queue_info_vk.p_next.is_null());
                unsafe {
                    (fns.v1_0.get_device_queue)(
                        device.handle(),
                        queue_info_vk.queue_family_index,
                        queue_info_vk.queue_index,
                        output.as_mut_ptr(),
                    )
                };
            }

            unsafe { output.assume_init() }
        };

        let lock = Arc::new(DefaultQueueMutex::new());

        unsafe { Self::from_handle(device, handle, queue_info, lock) }
    }

    /// Creates a new `Queue` from a raw object handle.
    ///
    /// When creating queues this way, it is important to ensure that each queue locks its own
    /// mutex. Otherwise, it would be possible to lock one queue, then lock another queue, and end
    /// up with a deadlock (even on a single thread). It is assumed that this doesn't happen, as
    /// it's fundamentally incompatible with internally-locked queues.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle retrieved from `device`.
    /// - `create_info` must match the info used to create the object. If you aren't given the
    ///   `queue_index`, it's fine to make it up, as vulkano doesn't use it itself. Just make sure
    ///   nothing else uses it. If you aren't given any queue create flags, it's safe to assume
    ///   that there are none.
    /// - The external code you share the queue with must use the same queue lock as vulkano. The
    ///   `lock` parameter must be used to ensure this.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Queue,
        queue_info: &DeviceQueueInfo<'_>,
        lock: Arc<dyn QueueMutex>,
    ) -> Arc<Self> {
        let &DeviceQueueInfo {
            flags,
            queue_family_index,
            queue_index,
            _ne: _,
        } = queue_info;

        let id = Self::next_id();

        device.queue_locks.lock().push((id, lock.clone()));

        Arc::new(Queue {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id,
            flags,
            queue_family_index,
            queue_index,
            lock,
        })
    }

    /// Returns the device that this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the flags that the queue was created with.
    #[inline]
    pub fn flags(&self) -> QueueCreateFlags {
        self.flags
    }

    /// Returns the index of the queue family that this queue belongs to.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the index of this queue within its queue family.
    #[inline]
    pub fn queue_index(&self) -> u32 {
        self.queue_index
    }

    /// Locks the queue and then calls the provided closure, providing it with an object that
    /// can be used to perform operations on the queue, such as command buffer submissions.
    #[inline]
    pub fn with<'a, R>(self: &'a Arc<Self>, func: impl FnOnce(QueueGuard<'a>) -> R) -> R {
        self.lock.lock();

        // SAFETY: We have just locked the queue above.
        func(unsafe { QueueGuard::new(self) })
    }

    fn queue_family_properties(&self) -> &QueueFamilyProperties {
        &self.device().physical_device().queue_family_properties()[self.queue_family_index as usize]
    }
}

impl Drop for Queue {
    #[inline]
    fn drop(&mut self) {
        let mut queue_locks = self.device.queue_locks.lock();

        // This can't panic because we inserted our lock on creation and only remove it here.
        let index = queue_locks
            .iter()
            .position(|(id, _)| *id == self.id)
            .unwrap();

        queue_locks.remove(index);
    }
}

unsafe impl VulkanObject for Queue {
    type Handle = vk::Queue;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Queue {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Queue);

/// A trait allowing you to supply a custom queue lock implementation.
///
/// This is necessary when there is external code with which you share your queue(s). If the
/// external code doesn't get the memo that vulkano has locked the queue or vice versa then it
/// could happen that both use the queue at the same time, which would be undefined behavior.
/// Typically, you would use the external library's functions for locking and unlocking a queue in
/// order to implement this trait.
///
/// # Safety
///
/// `lock` and `unlock` must have the semantics of a mutual exclusion lock: after `lock` is called,
/// further calls to `lock` must block until `unlock` is called, at which point at most one thread
/// in the wait queue must be woken up and return from `lock`.
pub unsafe trait QueueMutex: Send + Sync + 'static {
    /// Locks the mutex.
    fn lock(&self);

    /// Unlocks the mutex.
    ///
    /// # Safety
    ///
    /// - The lock must have been locked.
    unsafe fn unlock(&self);
}

impl Debug for dyn QueueMutex {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("QueueMutex").finish_non_exhaustive()
    }
}

/// The default queue lock implementation.
///
/// This uses its own mutex internally that is not shared with any external code. As such, it
/// should not be used when external code is sharing a queue with vulkano.
pub struct DefaultQueueMutex {
    inner: Mutex<()>,
}

impl Default for DefaultQueueMutex {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultQueueMutex {
    /// Creates a new `DefaultQueueMutex`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(()),
        }
    }
}

// SAFETY: We lock the mutex correctly.
unsafe impl QueueMutex for DefaultQueueMutex {
    fn lock(&self) {
        mem::forget(self.inner.lock());
    }

    unsafe fn unlock(&self) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.force_unlock() };
    }
}

/// Parameters to retrieve a [`Queue`] from the device.
#[derive(Clone, Debug)]
pub struct DeviceQueueInfo<'a> {
    /// Additional properties of the queue.
    ///
    /// The default value is empty.
    pub flags: QueueCreateFlags,

    /// The queue family index the queue is from.
    ///
    /// The default value is `0`.
    pub queue_family_index: u32,

    /// The queue index within the queue family.
    ///
    /// The default value is `0`.
    pub queue_index: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DeviceQueueInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self {
            flags: QueueCreateFlags::empty(),
            queue_family_index: 0,
            queue_index: 0,
            _ne: crate::NE,
        }
    }
}

impl DeviceQueueInfo<'_> {
    pub(crate) fn to_vk(&self) -> vk::DeviceQueueInfo2<'static> {
        let &Self {
            flags,
            queue_family_index,
            queue_index,
            _ne: _,
        } = self;

        vk::DeviceQueueInfo2::default()
            .flags(flags.into())
            .queue_family_index(queue_family_index)
            .queue_index(queue_index)
    }
}

pub struct QueueGuard<'a> {
    queue: &'a Arc<Queue>,
}

impl<'a> QueueGuard<'a> {
    unsafe fn new(queue: &'a Arc<Queue>) -> Self {
        Self { queue }
    }

    /// Waits until all work on this queue has finished, then releases ownership of all resources
    /// that were in use by the queue.
    ///
    /// This is equivalent to submitting a fence to the queue, waiting on it, and then calling
    /// `cleanup_finished`.
    ///
    /// Just like [`Device::wait_idle`], you shouldn't have to call this function in a typical
    /// program.
    #[inline]
    pub fn wait_idle(&mut self) -> Result<(), VulkanError> {
        let fns = self.queue.device.fns();
        unsafe { (fns.v1_0.queue_wait_idle)(self.queue.handle) }
            .result()
            .map_err(VulkanError::from)
    }

    /// Bind or unbind memory to resources with sparse memory, panicking on a validation error.
    ///
    /// This is a shortcut for `try_bind_sparse().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of every `bind_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The semaphore must be already in the signaled state, or there must be a previously
    ///   submitted operation that will signal it.
    /// - When the wait operation is executed, no other queue must be waiting on the same
    ///   semaphore.
    ///
    /// For every element in the `buffer_binds`, `image_opaque_binds` and `image_binds`
    /// elements of every `bind_infos` element:
    /// - The buffers and images must be kept alive while the command is being executed.
    /// - The memory allocations must be kept alive while they are bound to a buffer or image.
    /// - Access to the affected regions of each buffer or image must be synchronized.
    ///
    /// For every semaphore in the `signal_semaphores` elements of every `bind_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - When the signal operation is executed, the semaphore must be in the unsignaled state.
    ///
    /// If `fence` is `Some`:
    /// - The fence must be kept alive while the command is being executed.
    /// - The fence must be unsignaled and must not be associated with any other command that is
    ///   still executing.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_bind_sparse`] returns a [`ValidationError`].
    ///
    /// [`try_bind_sparse`]: Self::try_bind_sparse
    #[inline]
    #[track_caller]
    pub unsafe fn bind_sparse(
        &mut self,
        bind_infos: &[BindSparseInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_bind_sparse(bind_infos, fence) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Bind or unbind memory to resources with sparse memory.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of every `bind_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The semaphore must be already in the signaled state, or there must be a previously
    ///   submitted operation that will signal it.
    /// - When the wait operation is executed, no other queue must be waiting on the same
    ///   semaphore.
    ///
    /// For every element in the `buffer_binds`, `image_opaque_binds` and `image_binds`
    /// elements of every `bind_infos` element:
    /// - The buffers and images must be kept alive while the command is being executed.
    /// - The memory allocations must be kept alive while they are bound to a buffer or image.
    /// - Access to the affected regions of each buffer or image must be synchronized.
    ///
    /// For every semaphore in the `signal_semaphores` elements of every `bind_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - When the signal operation is executed, the semaphore must be in the unsignaled state.
    ///
    /// If `fence` is `Some`:
    /// - The fence must be kept alive while the command is being executed.
    /// - The fence must be unsignaled and must not be associated with any other command that is
    ///   still executing.
    #[inline]
    pub unsafe fn try_bind_sparse(
        &mut self,
        bind_infos: &[BindSparseInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_bind_sparse(bind_infos, fence)?;

        Ok(unsafe { self.bind_sparse_unchecked(bind_infos, fence) }?)
    }

    fn validate_bind_sparse(
        &self,
        bind_infos: &[BindSparseInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::SPARSE_BINDING)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of this queue does not support \
                    sparse binding operations"
                    .into(),
                vuids: &["VUID-vkQueueBindSparse-queuetype"],
                ..Default::default()
            }));
        }

        let device = self.queue.device();

        if let Some(fence) = fence {
            // VUID-vkQueueBindSparse-commonparent
            assert_eq!(device, fence.device());
        }

        for (index, bind_sparse_info) in bind_infos.iter().enumerate() {
            bind_sparse_info
                .validate(device)
                .map_err(|err| err.add_context(format!("bind_infos[{}]", index)))?;
        }

        // unsafe
        // VUID-vkQueueBindSparse-fence-01113
        // VUID-vkQueueBindSparse-fence-01114
        // VUID-vkQueueBindSparse-pSignalSemaphores-01115
        // VUID-vkQueueBindSparse-pWaitSemaphores-01116
        // VUID-vkQueueBindSparse-pWaitSemaphores-03245

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_sparse_unchecked(
        &mut self,
        bind_infos: &[BindSparseInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        let bind_infos_fields2_vk: SmallVec<[_; 4]> = bind_infos
            .iter()
            .map(BindSparseInfo::to_vk_fields2)
            .collect();

        let bind_infos_fields1_vk: SmallVec<[_; 4]> = bind_infos
            .iter()
            .zip(&bind_infos_fields2_vk)
            .map(|(bind_info, fields2_vk)| bind_info.to_vk_fields1(fields2_vk))
            .collect();

        let bind_infos_vk: SmallVec<[_; 4]> = bind_infos
            .iter()
            .zip(&bind_infos_fields1_vk)
            .map(|(bind_info, fields1_vk)| bind_info.to_vk(fields1_vk))
            .collect();

        let fns = self.queue.device.fns();
        unsafe {
            (fns.v1_0.queue_bind_sparse)(
                self.queue.handle,
                bind_infos_vk.len() as u32,
                bind_infos_vk.as_ptr(),
                fence
                    .as_ref()
                    .map_or_else(Default::default, VulkanObject::handle),
            )
        }
        .result()
        .map_err(VulkanError::from)
    }

    /// Queues swapchain images for presentation to the surface, panicking on a validation error.
    ///
    /// This is a shortcut for `try_present().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of `present_info`:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The semaphore must be already in the signaled state, or there must be a previously
    ///   submitted operation that will signal it.
    /// - When the wait operation is executed, no other queue must be waiting on the same
    ///   semaphore.
    ///
    /// For every element of `present_info.swapchain_infos`:
    /// - `swapchain` must be kept alive while the command is being executed.
    /// - `image_index` must be an index previously acquired from the swapchain, and the present
    ///   operation must happen-after the acquire operation.
    /// - The swapchain image indicated by `swapchain` and `image_index` must be in the
    ///   [`ImageLayout::PresentSrc`] layout when the presentation operation is executed.
    /// - The swapchain image indicated by `swapchain` and `image_index` must not be accessed after
    ///   this function is called, until it is acquired again.
    /// - If `present_id` is `Some`, then it must be greater than any present ID previously used
    ///   for the same swapchain.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_present`] returns a [`ValidationError`].
    ///
    /// [`ImageLayout::PresentSrc`]: crate::image::ImageLayout::PresentSrc
    /// [`try_present`]: Self::try_present
    #[inline]
    #[track_caller]
    pub unsafe fn present(
        &mut self,
        present_info: &PresentInfo,
    ) -> Result<Vec<Result<bool, VulkanError>>, VulkanError> {
        match unsafe { self.try_present(present_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Queues swapchain images for presentation to the surface.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of `present_info`:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The semaphore must be already in the signaled state, or there must be a previously
    ///   submitted operation that will signal it.
    /// - When the wait operation is executed, no other queue must be waiting on the same
    ///   semaphore.
    ///
    /// For every element of `present_info.swapchain_infos`:
    /// - `swapchain` must be kept alive while the command is being executed.
    /// - `image_index` must be an index previously acquired from the swapchain, and the present
    ///   operation must happen-after the acquire operation.
    /// - The swapchain image indicated by `swapchain` and `image_index` must be in the
    ///   [`ImageLayout::PresentSrc`] layout when the presentation operation is executed.
    /// - The swapchain image indicated by `swapchain` and `image_index` must not be accessed after
    ///   this function is called, until it is acquired again.
    /// - If `present_id` is `Some`, then it must be greater than any present ID previously used
    ///   for the same swapchain.
    ///
    /// [`ImageLayout::PresentSrc`]: crate::image::ImageLayout::PresentSrc
    #[inline]
    pub unsafe fn try_present(
        &mut self,
        present_info: &PresentInfo,
    ) -> Result<Vec<Result<bool, VulkanError>>, Validated<VulkanError>> {
        self.validate_present(present_info)?;

        Ok(unsafe { self.present_unchecked(present_info) }?)
    }

    fn validate_present(&self, present_info: &PresentInfo) -> Result<(), Box<ValidationError>> {
        let device = self.queue.device();

        if !device.enabled_extensions().khr_swapchain {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_swapchain",
                )])]),
                ..Default::default()
            }));
        }

        present_info
            .validate(device)
            .map_err(|err| err.add_context("present_info"))?;

        let PresentInfo {
            wait_semaphores: _,
            swapchain_infos: swapchains,
            _ne: _,
        } = present_info;

        for (index, swapchain_info) in swapchains.iter().enumerate() {
            let SwapchainPresentInfo {
                swapchain,
                image_index: _,
                present_region: _,
                present_id: _,
                fence: _,
                present_mode: _,
                _ne: _,
            } = swapchain_info;

            let surface_support = unsafe {
                device
                    .physical_device
                    .surface_support_unchecked(self.queue.queue_family_index, swapchain.surface())
            };

            if !surface_support.unwrap_or_default() {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "present_info.swapchain_infos[{}].swapchain.surface()",
                        index
                    )
                    .into(),
                    problem: "the queue family of this queue does not support presenting to \
                        the surface"
                        .into(),
                    vuids: &["VUID-vkQueuePresentKHR-pSwapchains-01292"],
                    ..Default::default()
                }));
            }
        }

        // unsafe
        // VUID-vkQueuePresentKHR-pWaitSemaphores-01294
        // VUID-vkQueuePresentKHR-pWaitSemaphores-03268

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn present_unchecked(
        &mut self,
        present_info: &PresentInfo,
    ) -> Result<Vec<Result<bool, VulkanError>>, VulkanError> {
        let present_info_fields2_vk = present_info.to_vk_fields2();
        let present_info_fields1_vk = present_info.to_vk_fields1(&present_info_fields2_vk);
        let mut results_vk = present_info.to_vk_results();
        let mut present_info_extensions_vk =
            present_info.to_vk_extensions(&present_info_fields1_vk);
        let info_vk = present_info.to_vk(
            &present_info_fields1_vk,
            &mut results_vk,
            &mut present_info_extensions_vk,
        );

        let fns = self.queue.device().fns();
        let result = unsafe { (fns.khr_swapchain.queue_present_khr)(self.queue.handle, &info_vk) };

        // Per the documentation of `vkQueuePresentKHR`, certain results indicate that the whole
        // operation has failed, while others only indicate failure of a particular present.
        // If we got a result that is not one of these per-present ones, we return it directly.
        // Otherwise, we consider the present to be enqueued.
        if !matches!(
            result,
            vk::Result::SUCCESS
                | vk::Result::SUBOPTIMAL_KHR
                | vk::Result::ERROR_OUT_OF_DATE_KHR
                | vk::Result::ERROR_SURFACE_LOST_KHR
                | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
        ) {
            return Err(VulkanError::from(result));
        }

        let results = results_vk
            .into_iter()
            .map(|result| match result {
                vk::Result::SUCCESS => Ok(false),
                vk::Result::SUBOPTIMAL_KHR => Ok(true),
                err => Err(VulkanError::from(err)),
            })
            .collect();

        Ok(results)
    }

    /// Submits command buffers to a queue to be executed, panicking on a validation error.
    ///
    /// This is a shortcut for `try_submit().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of every `submit_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The safety requirements for semaphores, as detailed in the
    ///   [`semaphore`](crate::sync::semaphore#Safety) module documentation, must be followed.
    ///
    /// For every command buffer in the `command_buffers` elements of every `submit_infos` element,
    /// as well as any secondary command buffers recorded within it:
    /// - The command buffer, and any resources it uses, must be kept alive while the command is
    ///   being executed.
    /// - Any mutable resources used by the command buffer must be in the state (image layout,
    ///   query reset, event signal etc.) expected by the command buffer at the time it begins
    ///   executing, and must be synchronized appropriately.
    /// - If the command buffer's `usage` is [`CommandBufferUsage::OneTimeSubmit`], then it must
    ///   not have been previously submitted.
    /// - If the command buffer's `usage` is [`CommandBufferUsage::MultipleSubmit`], then it must
    ///   not be currently submitted and not yet completed.
    /// - If a recorded command performs a queue family transfer acquire operation, then a
    ///   corresponding queue family transfer release operation with matching parameters must have
    ///   been previously submitted, and must happen-before it.
    /// - If a recorded command references an [`Event`], then that `Event` must not be referenced
    ///   by a command that is currently executing on another queue.
    ///
    /// For every semaphore in the `signal_semaphores` elements of every `submit_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The safety requirements for semaphores, as detailed in the
    ///   [`semaphore`](crate::sync::semaphore#Safety) module documentation, must be followed.
    ///
    /// If `fence` is `Some`:
    /// - The fence must be kept alive while the command is being executed.
    /// - The safety requirements for fences, as detailed in the
    ///   [`fence`](crate::sync::fence#Safety) module documentation, must be followed.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_submit`] returns a [`ValidationError`].
    ///
    /// [`try_submit`]: Self::try_submit
    /// [`CommandBufferUsage::OneTimeSubmit`]: crate::command_buffer::CommandBufferUsage::OneTimeSubmit
    /// [`CommandBufferUsage::MultipleSubmit`]: crate::command_buffer::CommandBufferUsage::MultipleSubmit
    /// [`Event`]: crate::sync::event::Event
    #[inline]
    #[track_caller]
    pub unsafe fn submit(
        &mut self,
        submit_infos: &[SubmitInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_submit(submit_infos, fence) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Submits command buffers to a queue to be executed.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of every `submit_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The safety requirements for semaphores, as detailed in the
    ///   [`semaphore`](crate::sync::semaphore#Safety) module documentation, must be followed.
    ///
    /// For every command buffer in the `command_buffers` elements of every `submit_infos` element,
    /// as well as any secondary command buffers recorded within it:
    /// - The command buffer, and any resources it uses, must be kept alive while the command is
    ///   being executed.
    /// - Any mutable resources used by the command buffer must be in the state (image layout,
    ///   query reset, event signal etc.) expected by the command buffer at the time it begins
    ///   executing, and must be synchronized appropriately.
    /// - If the command buffer's `usage` is [`CommandBufferUsage::OneTimeSubmit`], then it must
    ///   not have been previously submitted.
    /// - If the command buffer's `usage` is [`CommandBufferUsage::MultipleSubmit`], then it must
    ///   not be currently submitted and not yet completed.
    /// - If a recorded command performs a queue family transfer acquire operation, then a
    ///   corresponding queue family transfer release operation with matching parameters must have
    ///   been previously submitted, and must happen-before it.
    /// - If a recorded command references an [`Event`], then that `Event` must not be referenced
    ///   by a command that is currently executing on another queue.
    ///
    /// For every semaphore in the `signal_semaphores` elements of every `submit_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The safety requirements for semaphores, as detailed in the
    ///   [`semaphore`](crate::sync::semaphore#Safety) module documentation, must be followed.
    ///
    /// If `fence` is `Some`:
    /// - The fence must be kept alive while the command is being executed.
    /// - The safety requirements for fences, as detailed in the
    ///   [`fence`](crate::sync::fence#Safety) module documentation, must be followed.
    ///
    /// [`CommandBufferUsage::OneTimeSubmit`]: crate::command_buffer::CommandBufferUsage::OneTimeSubmit
    /// [`CommandBufferUsage::MultipleSubmit`]: crate::command_buffer::CommandBufferUsage::MultipleSubmit
    /// [`Event`]: crate::sync::event::Event
    #[inline]
    pub unsafe fn try_submit(
        &mut self,
        submit_infos: &[SubmitInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_submit(submit_infos, fence)?;

        Ok(unsafe { self.submit_unchecked(submit_infos, fence) }?)
    }

    fn validate_submit(
        &self,
        submit_infos: &[SubmitInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), Box<ValidationError>> {
        let device = self.queue.device();
        let queue_family_properties = &device.physical_device().queue_family_properties()
            [self.queue.queue_family_index as usize];

        if let Some(fence) = fence {
            // VUID-vkQueueSubmit2-commonparent
            assert_eq!(device, fence.device());
        }

        let supported_stages = PipelineStages::from(queue_family_properties.queue_flags);

        for (index, submit_info) in submit_infos.iter().enumerate() {
            submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("submit_infos[{}]", index)))?;

            let &SubmitInfo {
                wait_semaphores,
                command_buffers,
                signal_semaphores,
                protected_submit: _,
                _ne: _,
            } = submit_info;

            for (semaphore_index, semaphore_submit_info) in wait_semaphores.iter().enumerate() {
                let &SemaphoreSubmitInfo {
                    semaphore: _,
                    value: _,
                    stages,
                    _ne: _,
                } = semaphore_submit_info;

                if !supported_stages.contains(stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "submit_infos[{}].wait_semaphores[{}].stages",
                            index, semaphore_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the queue"
                            .into(),
                        vuids: &["VUID-vkQueueSubmit2-stageMask-03870"],
                        ..Default::default()
                    }));
                }
            }

            for (command_buffer_index, command_buffer_submit_info) in
                command_buffers.iter().enumerate()
            {
                let CommandBufferSubmitInfo {
                    command_buffer,
                    _ne: _,
                } = command_buffer_submit_info;

                if command_buffer.queue_family_index() != self.queue.queue_family_index {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "submit_infos[{}].command_buffers[{}]\
                            .command_buffer.queue_family_index()",
                            index, command_buffer_index
                        )
                        .into(),
                        problem: "does not equal the queue family index of this queue".into(),
                        vuids: &["VUID-vkQueueSubmit2-commandBuffer-03878"],
                        ..Default::default()
                    }));
                }
            }

            for (semaphore_index, semaphore_submit_info) in signal_semaphores.iter().enumerate() {
                let &SemaphoreSubmitInfo {
                    semaphore: _,
                    value: _,
                    stages,
                    _ne: _,
                } = semaphore_submit_info;

                if !supported_stages.contains(stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "submit_infos[{}].signal_semaphores[{}].stages",
                            index, semaphore_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the queue"
                            .into(),
                        vuids: &["VUID-vkQueueSubmit2-stageMask-03870"],
                        ..Default::default()
                    }));
                }
            }
        }

        // unsafe
        // VUID-VkSubmitInfo2-semaphore-03882
        // VUID-VkSubmitInfo2-semaphore-03883
        // VUID-VkSubmitInfo2-semaphore-03884
        // VUID-vkQueueSubmit2-fence-04894
        // VUID-vkQueueSubmit2-fence-04895
        // VUID-vkQueueSubmit2-commandBuffer-03867
        // VUID-vkQueueSubmit2-semaphore-03868
        // VUID-vkQueueSubmit2-semaphore-03871
        // VUID-vkQueueSubmit2-semaphore-03873
        // VUID-vkQueueSubmit2-commandBuffer-03874
        // VUID-vkQueueSubmit2-commandBuffer-03875
        // VUID-vkQueueSubmit2-commandBuffer-03876
        // VUID-vkQueueSubmit2-commandBuffer-03877
        // VUID-vkQueueSubmit2-commandBuffer-03879

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn submit_unchecked(
        &mut self,
        submit_infos: &[SubmitInfo<'_>],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        if self.queue.device.enabled_features().synchronization2 {
            let submit_infos_fields1_vk: SmallVec<[_; 4]> = submit_infos
                .iter()
                .map(SubmitInfo::to_vk2_fields1)
                .collect();

            let submit_infos_vk: SmallVec<[_; 4]> = submit_infos
                .iter()
                .zip(&submit_infos_fields1_vk)
                .map(|(submit_info, fields1_vk)| submit_info.to_vk2(fields1_vk))
                .collect();

            let fns = self.queue.device.fns();

            if self.queue.device.api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.queue_submit2)(
                        self.queue.handle,
                        submit_infos_vk.len() as u32,
                        submit_infos_vk.as_ptr(),
                        fence
                            .as_ref()
                            .map_or_else(Default::default, VulkanObject::handle),
                    )
                }
            } else {
                debug_assert!(self.queue.device.enabled_extensions().khr_synchronization2);
                unsafe {
                    (fns.khr_synchronization2.queue_submit2_khr)(
                        self.queue.handle,
                        submit_infos_vk.len() as u32,
                        submit_infos_vk.as_ptr(),
                        fence
                            .as_ref()
                            .map_or_else(Default::default, VulkanObject::handle),
                    )
                }
            }
            .result()
            .map_err(VulkanError::from)
        } else {
            let fields1_vk: SmallVec<[_; 4]> = submit_infos
                .iter()
                .map(|submit_info| submit_info.to_vk_fields1())
                .collect();

            let mut submit_infos_extensions_vk: SmallVec<[_; 4]> = submit_infos
                .iter()
                .zip(&fields1_vk)
                .map(|(submit_info, submit_info_fields1_vk)| {
                    submit_info.to_vk_extensions(submit_info_fields1_vk)
                })
                .collect();

            let submit_infos_vk: SmallVec<[_; 4]> = submit_infos
                .iter()
                .zip(&fields1_vk)
                .zip(&mut submit_infos_extensions_vk)
                .map(|((submit_infos, fields1_vk), extensions_vk)| {
                    submit_infos.to_vk(fields1_vk, extensions_vk)
                })
                .collect();

            let fns = self.queue.device.fns();
            unsafe {
                (fns.v1_0.queue_submit)(
                    self.queue.handle,
                    submit_infos_vk.len() as u32,
                    submit_infos_vk.as_ptr(),
                    fence
                        .as_ref()
                        .map_or_else(Default::default, VulkanObject::handle),
                )
            }
            .result()
            .map_err(VulkanError::from)
        }
    }

    /// Opens a queue debug label region, panicking on a validation error.
    ///
    /// The [`ext_debug_utils`] extension must be enabled on the instance.
    ///
    /// This is a shortcut for `try_begin_debug_utils_label().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_begin_debug_utils_label`] returns a [`ValidationError`].
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    /// [`try_begin_debug_utils_label`]: Self::try_begin_debug_utils_label
    #[inline]
    #[track_caller]
    pub fn begin_debug_utils_label(&mut self, label_info: DebugUtilsLabel) {
        self.try_begin_debug_utils_label(label_info).unwrap()
    }

    /// Opens a queue debug label region.
    ///
    /// The [`ext_debug_utils`] extension must be enabled on the instance.
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    #[inline]
    pub fn try_begin_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        self.validate_begin_debug_utils_label(&label_info)?;

        unsafe { self.begin_debug_utils_label_unchecked(label_info) };

        Ok(())
    }

    fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn begin_debug_utils_label_unchecked(&mut self, label_info: DebugUtilsLabel) {
        let label_info_fields1_vk = label_info.to_vk_fields1();
        let label_info_vk = label_info.to_vk(&label_info_fields1_vk);

        let fns = self.queue.device.fns();
        unsafe {
            (fns.ext_debug_utils.queue_begin_debug_utils_label_ext)(
                self.queue.handle,
                &label_info_vk,
            )
        };
    }

    /// Closes a queue debug label region, panicking on a validation error.
    ///
    /// The [`ext_debug_utils`] must be enabled on the instance.
    ///
    /// This is a shortcut for `try_end_debug_utils_label().unwrap()`.
    ///
    /// # Safety
    ///
    /// - There must be an outstanding queue label region begun with `begin_debug_utils_label` in
    ///   the queue.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_end_debug_utils_label`] returns a [`ValidationError`].
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    /// [`try_end_debug_utils_label`]: Self::try_end_debug_utils_label
    #[inline]
    #[track_caller]
    pub unsafe fn end_debug_utils_label(&mut self) {
        unsafe { self.try_end_debug_utils_label() }.unwrap()
    }

    /// Closes a queue debug label region.
    ///
    /// The [`ext_debug_utils`] must be enabled on the instance.
    ///
    /// # Safety
    ///
    /// - There must be an outstanding queue label region begun with `begin_debug_utils_label` in
    ///   the queue.
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    #[inline]
    pub unsafe fn try_end_debug_utils_label(&mut self) -> Result<(), Box<ValidationError>> {
        self.validate_end_debug_utils_label()?;
        unsafe { self.end_debug_utils_label_unchecked() };

        Ok(())
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkQueueEndDebugUtilsLabelEXT-None-01911
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) {
        let fns = self.queue.device.fns();
        unsafe { (fns.ext_debug_utils.queue_end_debug_utils_label_ext)(self.queue.handle) };
    }

    /// Inserts a queue debug label, panicking on a validation error.
    ///
    /// The [`ext_debug_utils`] must be enabled on the instance.
    ///
    /// This is a shortcut for `try_insert_debug_utils_label().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_insert_debug_utils_label`] returns a [`ValidationError`].
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    /// [`try_insert_debug_utils_label`]: Self::try_insert_debug_utils_label
    #[inline]
    #[track_caller]
    pub fn insert_debug_utils_label(&mut self, label_info: DebugUtilsLabel) {
        self.try_insert_debug_utils_label(label_info).unwrap()
    }

    /// Inserts a queue debug label.
    ///
    /// The [`ext_debug_utils`] must be enabled on the instance.
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
    #[inline]
    pub fn try_insert_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        self.validate_insert_debug_utils_label(&label_info)?;

        unsafe { self.insert_debug_utils_label_unchecked(label_info) };

        Ok(())
    }

    fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn insert_debug_utils_label_unchecked(&mut self, label_info: DebugUtilsLabel) {
        let label_info_fields1_vk = label_info.to_vk_fields1();
        let label_info_vk = label_info.to_vk(&label_info_fields1_vk);

        let fns = self.queue.device.fns();
        unsafe {
            (fns.ext_debug_utils.queue_insert_debug_utils_label_ext)(
                self.queue.handle,
                &label_info_vk,
            )
        };
    }
}

impl Drop for QueueGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: The caller of `QueueGuard::new` must ensure that the queue has been locked.
        unsafe { self.queue.lock.unlock() };
    }
}

/// Properties of a queue family in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QueueFamilyProperties {
    /// Attributes of the queue family.
    pub queue_flags: QueueFlags,

    /// The number of queues available in this family.
    ///
    /// This guaranteed to be at least 1 (or else that family wouldn't exist).
    pub queue_count: u32,

    /// If timestamps are supported, the number of bits supported by timestamp operations.
    /// The returned value will be in the range 36..64.
    ///
    /// If timestamps are not supported, this is `None`.
    pub timestamp_valid_bits: Option<u32>,

    /// The minimum granularity supported for image transfers, in terms of `[width, height,
    /// depth]`.
    pub min_image_transfer_granularity: [u32; 3],
}

impl QueueFamilyProperties {
    pub(crate) fn to_mut_vk2() -> vk::QueueFamilyProperties2<'static> {
        vk::QueueFamilyProperties2::default()
    }

    pub(crate) fn from_vk2(val_vk: &vk::QueueFamilyProperties2<'_>) -> Self {
        let vk::QueueFamilyProperties2 {
            queue_family_properties,
            ..
        } = val_vk;

        Self::from_vk(queue_family_properties)
    }

    pub(crate) fn from_vk(val_vk: &vk::QueueFamilyProperties) -> Self {
        let &vk::QueueFamilyProperties {
            queue_flags,
            queue_count,
            timestamp_valid_bits,
            min_image_transfer_granularity,
        } = val_vk;

        Self {
            queue_flags: queue_flags.into(),
            queue_count,
            timestamp_valid_bits: (timestamp_valid_bits != 0).then_some(timestamp_valid_bits),
            min_image_transfer_granularity: [
                min_image_transfer_granularity.width,
                min_image_transfer_granularity.height,
                min_image_transfer_granularity.depth,
            ],
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Attributes of a queue or queue family.
    QueueFlags = QueueFlags(u32);

    /// Queues of this family can execute graphics operations.
    GRAPHICS = GRAPHICS,

    /// Queues of this family can execute compute operations.
    COMPUTE = COMPUTE,

    /// Queues of this family can execute transfer operations.
    TRANSFER = TRANSFER,

    /// Queues of this family can execute sparse memory management operations.
    SPARSE_BINDING = SPARSE_BINDING,

    /// Queues of this family can be created using the `protected` flag.
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),

    /// Queues of this family can execute video decode operations.
    VIDEO_DECODE = VIDEO_DECODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),

    /// Queues of this family can execute video encode operations.
    VIDEO_ENCODE = VIDEO_ENCODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),

    /// Queues of this family can execute optical flow operations.
    OPTICAL_FLOW = OPTICAL_FLOW_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_optical_flow)]),
    ]),
}

#[cfg(test)]
mod tests {
    use crate::sync::fence::Fence;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn empty_submit() {
        let (_device, queue) = gfx_dev_and_queue!();

        queue
            .with(|mut q| unsafe { q.submit(&[Default::default()], None) })
            .unwrap();
    }

    #[test]
    fn signal_fence() {
        let (device, queue) = gfx_dev_and_queue!();

        let fence = Arc::new(Fence::new(&device, &Default::default()).unwrap());
        assert!(!fence.is_signaled().unwrap());

        queue
            .with(|mut q| unsafe { q.submit(&[Default::default()], Some(&fence)) })
            .unwrap();

        fence.wait(Some(Duration::from_secs(5))).unwrap();
        assert!(fence.is_signaled().unwrap());
    }
}
