//! Memory and resource pool for recording command buffers.
//!
//! A command pool holds and manages the memory of one or more command buffers. If you destroy a
//! command pool, all command buffers recorded from it become invalid. This could lead to invalid
//! usage and unsoundness, so to ensure safety you must use a [command buffer allocator].
//!
//! [command buffer allocator]: crate::command_buffer::allocator

use crate::{
    command_buffer::CommandBufferLevel,
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;
use std::{cell::Cell, marker::PhantomData, mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// Represents a Vulkan command pool.
///
/// A command pool is always tied to a specific queue family. Command buffers allocated from a pool
/// can only be executed on the corresponding queue family.
///
/// This struct doesn't implement the `Sync` trait because Vulkan command pools are not thread
/// safe. In other words, you can only use a pool from one thread at a time.
#[derive(Debug)]
pub struct CommandPool {
    handle: vk::CommandPool,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: CommandPoolCreateFlags,
    queue_family_index: u32,

    // Unimplement `Sync`, as Vulkan command pools are not thread-safe.
    _marker: PhantomData<Cell<vk::CommandPool>>,
}

impl CommandPool {
    /// Creates a new `CommandPool`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &CommandPoolCreateInfo<'_>,
    ) -> Result<CommandPool, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `CommandPool`.
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &CommandPoolCreateInfo<'_>,
    ) -> Result<CommandPool, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &CommandPoolCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &CommandPoolCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_command_pool)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    /// Creates a new `CommandPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::CommandPool,
        create_info: &CommandPoolCreateInfo<'_>,
    ) -> CommandPool {
        let &CommandPoolCreateInfo {
            flags,
            queue_family_index,
            _ne: _,
        } = create_info;

        CommandPool {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            flags,
            queue_family_index,

            _marker: PhantomData,
        }
    }

    /// Returns the flags that the command pool was created with.
    #[inline]
    pub fn flags(&self) -> CommandPoolCreateFlags {
        self.flags
    }

    /// Returns the queue family on which command buffers of this pool can be executed.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Resets the pool, which resets all the command buffers that were allocated from it,
    /// panicking on a validation error.
    ///
    /// This is a shortcut for `try_reset().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The command buffers allocated from this pool must not be in the pending state.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_reset`] returns a [`ValidationError`].
    ///
    /// [`try_reset`]: Self::try_reset
    #[inline]
    #[track_caller]
    pub unsafe fn reset(&self, flags: CommandPoolResetFlags) -> Result<(), VulkanError> {
        match unsafe { self.try_reset(flags) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Resets the pool, which resets all the command buffers that were allocated from it.
    ///
    /// # Safety
    ///
    /// - The command buffers allocated from this pool must not be in the pending state.
    #[inline]
    pub unsafe fn try_reset(
        &self,
        flags: CommandPoolResetFlags,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_reset(flags)?;

        Ok(unsafe { self.reset_unchecked(flags) }?)
    }

    fn validate_reset(&self, flags: CommandPoolResetFlags) -> Result<(), Box<ValidationError>> {
        flags.validate_device(self.device()).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-vkResetCommandPool-flags-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_unchecked(&self, flags: CommandPoolResetFlags) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.reset_command_pool)(self.device.handle(), self.handle, flags.into()) }
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Allocates command buffers, panicking on a validation error.
    ///
    /// This is a shortcut for `try_allocate_command_buffers().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_allocate_command_buffers`] returns a [`ValidationError`].
    ///
    /// [`try_allocate_command_buffers`]: Self::try_allocate_command_buffers
    #[inline]
    #[track_caller]
    pub fn allocate_command_buffers(
        &self,
        allocate_info: &CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<CommandPoolAlloc>, VulkanError> {
        match self.try_allocate_command_buffers(allocate_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Allocates command buffers.
    #[inline]
    pub fn try_allocate_command_buffers(
        &self,
        allocate_info: &CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<CommandPoolAlloc>, Validated<VulkanError>> {
        self.validate_allocate_command_buffers(allocate_info)?;

        Ok(unsafe { self.allocate_command_buffers_unchecked(allocate_info) }?)
    }

    fn validate_allocate_command_buffers(
        &self,
        _allocate_info: &CommandBufferAllocateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn allocate_command_buffers_unchecked(
        &self,
        allocate_info: &CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<CommandPoolAlloc>, VulkanError> {
        let &CommandBufferAllocateInfo {
            level,
            command_buffer_count,
            _ne: _,
        } = allocate_info;

        // VUID-vkAllocateCommandBuffers-pAllocateInfo::commandBufferCount-arraylength
        if command_buffer_count == 0 {
            return Ok(Vec::new());
        }

        let output = {
            let allocate_info_vk = allocate_info.to_vk(self.handle);
            let command_buffer_count = command_buffer_count as usize;

            let fns = self.device.fns();
            let mut out = Vec::with_capacity(command_buffer_count);

            unsafe {
                (fns.v1_0.allocate_command_buffers)(
                    self.device.handle(),
                    &allocate_info_vk,
                    out.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { out.set_len(command_buffer_count) };

            out
        };

        let command_pool_allocs = output
            .into_iter()
            .map(|command_buffer| CommandPoolAlloc {
                handle: command_buffer,
                device: InstanceOwnedDebugWrapper(self.device.clone()),
                id: CommandPoolAlloc::next_id(),
                level,
            })
            .collect();

        Ok(command_pool_allocs)
    }

    /// Frees individual command buffers, panicking on a validation error.
    ///
    /// This is a shortcut for `try_free_command_buffers().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The `command_buffers` must have been allocated from this pool.
    /// - The `command_buffers` must not be in the pending state.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_free_command_buffers`] returns a [`ValidationError`].
    ///
    /// [`try_free_command_buffers`]: Self::try_free_command_buffers
    #[track_caller]
    pub unsafe fn free_command_buffers(
        &self,
        command_buffers: impl IntoIterator<Item = CommandPoolAlloc>,
    ) {
        unsafe { self.try_free_command_buffers(command_buffers) }.unwrap()
    }

    /// Frees individual command buffers.
    ///
    /// # Safety
    ///
    /// - The `command_buffers` must have been allocated from this pool.
    /// - The `command_buffers` must not be in the pending state.
    pub unsafe fn try_free_command_buffers(
        &self,
        command_buffers: impl IntoIterator<Item = CommandPoolAlloc>,
    ) -> Result<(), Box<ValidationError>> {
        let command_buffers: SmallVec<[_; 4]> = command_buffers.into_iter().collect();
        self.validate_free_command_buffers(&command_buffers)?;

        unsafe { self.free_command_buffers_unchecked(command_buffers) };

        Ok(())
    }

    fn validate_free_command_buffers(
        &self,
        _command_buffers: &[CommandPoolAlloc],
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkFreeCommandBuffers-pCommandBuffers-00047
        // VUID-vkFreeCommandBuffers-pCommandBuffers-parent
        // Unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn free_command_buffers_unchecked(
        &self,
        command_buffers: impl IntoIterator<Item = CommandPoolAlloc>,
    ) {
        let command_buffers_vk: SmallVec<[_; 4]> =
            command_buffers.into_iter().map(|cb| cb.handle).collect();

        let fns = self.device.fns();
        unsafe {
            (fns.v1_0.free_command_buffers)(
                self.device.handle(),
                self.handle,
                command_buffers_vk.len() as u32,
                command_buffers_vk.as_ptr(),
            )
        }
    }

    /// Trims a command pool, which recycles unused internal memory from the command pool back to
    /// the system, panicking on a validation error.
    ///
    /// Command buffers allocated from the pool are not affected by trimming.
    ///
    /// This function is supported only if the [`khr_maintenance1`] extension is enabled on the
    /// device. Otherwise an error is returned. Since this operation is purely an optimization it
    /// is legitimate to call this function and simply ignore any possible error.
    ///
    /// This is a shortcut for `try_trim().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_trim`] returns a [`ValidationError`].
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    /// [`try_trim`]: Self::try_trim
    #[inline]
    #[track_caller]
    pub fn trim(&self) {
        self.try_trim().unwrap()
    }

    /// Trims a command pool, which recycles unused internal memory from the command pool back to
    /// the system.
    ///
    /// Command buffers allocated from the pool are not affected by trimming.
    ///
    /// This function is supported only if the [`khr_maintenance1`] extension is enabled on the
    /// device. Otherwise an error is returned. Since this operation is purely an optimization it
    /// is legitimate to call this function and simply ignore any possible error.
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    #[inline]
    pub fn try_trim(&self) -> Result<(), Box<ValidationError>> {
        self.validate_trim()?;

        unsafe { self.trim_unchecked() };

        Ok(())
    }

    fn validate_trim(&self) -> Result<(), Box<ValidationError>> {
        if !(self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_maintenance1)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance1")]),
                ]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn trim_unchecked(&self) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_1 {
            unsafe {
                (fns.v1_1.trim_command_pool)(
                    self.device.handle(),
                    self.handle,
                    vk::CommandPoolTrimFlags::empty(),
                )
            };
        } else {
            unsafe {
                (fns.khr_maintenance1.trim_command_pool_khr)(
                    self.device.handle(),
                    self.handle,
                    vk::CommandPoolTrimFlagsKHR::empty(),
                )
            };
        }
    }
}

impl Drop for CommandPool {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.destroy_command_pool)(self.device.handle(), self.handle, ptr::null()) };
    }
}

unsafe impl VulkanObject for CommandPool {
    type Handle = vk::CommandPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for CommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(CommandPool);

/// Parameters to create an `CommandPool`.
#[derive(Clone, Debug)]
pub struct CommandPoolCreateInfo<'a> {
    /// Additional properties of the command pool.
    ///
    /// The default value is empty.
    pub flags: CommandPoolCreateFlags,

    /// The index of the queue family that this pool is created for. All command buffers allocated
    /// from this pool must be submitted on a queue belonging to that family.
    ///
    /// The default value is `u32::MAX`, which must be overridden.
    pub queue_family_index: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CommandPoolCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CommandPoolCreateInfo<'_> {
    /// Returns a default `CommandPoolCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: CommandPoolCreateFlags::empty(),
            queue_family_index: u32::MAX,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            queue_family_index,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkCommandPoolCreateInfo-flags-parameter"])
        })?;

        if !device
            .active_queue_family_indices()
            .contains(&queue_family_index)
        {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not active on the device".into(),
                vuids: &["VUID-vkCreateCommandPool-queueFamilyIndex-01937"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::CommandPoolCreateInfo<'static> {
        let &Self {
            flags,
            queue_family_index,
            _ne: _,
        } = self;

        vk::CommandPoolCreateInfo::default()
            .flags(flags.into())
            .queue_family_index(queue_family_index)
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Additional properties of the command pool.
    CommandPoolCreateFlags = CommandPoolCreateFlags(u32);

    /// A hint to the implementation that the command buffers allocated from this pool will be
    /// short-lived.
    TRANSIENT = TRANSIENT,

    /// Command buffers allocated from this pool can be reset individually.
    RESET_COMMAND_BUFFER = RESET_COMMAND_BUFFER,

    /// Specifies that command buffers allocated from the pool are protected command buffers.
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)])
    ]),
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Additional properties of the command pool reset operation.
    CommandPoolResetFlags = CommandPoolResetFlags(u32);

    /// A hint to the implementation that it should free all the memory internally allocated
    /// for this pool.
    RELEASE_RESOURCES = RELEASE_RESOURCES,
}

/// Parameters to allocate a `CommandPoolAlloc`.
#[derive(Clone, Debug)]
pub struct CommandBufferAllocateInfo<'a> {
    /// The level of command buffer to allocate.
    ///
    /// The default value is [`CommandBufferLevel::Primary`].
    pub level: CommandBufferLevel,

    /// The number of command buffers to allocate.
    ///
    /// The default value is `1`.
    pub command_buffer_count: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CommandBufferAllocateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CommandBufferAllocateInfo<'_> {
    /// Returns a default `CommandBufferAllocateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            level: CommandBufferLevel::Primary,
            command_buffer_count: 1,
            _ne: crate::NE,
        }
    }

    pub(crate) fn to_vk(
        &self,
        command_pool_vk: vk::CommandPool,
    ) -> vk::CommandBufferAllocateInfo<'static> {
        let &Self {
            level,
            command_buffer_count,
            _ne: _,
        } = self;

        vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool_vk)
            .level(level.into())
            .command_buffer_count(command_buffer_count)
    }
}

/// Opaque type that represents a command buffer allocated from a pool.
#[derive(Debug)]
pub struct CommandPoolAlloc {
    handle: vk::CommandBuffer,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,
    level: CommandBufferLevel,
}

impl CommandPoolAlloc {
    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.level
    }
}

unsafe impl VulkanObject for CommandPoolAlloc {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for CommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(CommandPoolAlloc);

#[cfg(test)]
mod tests {
    use super::{CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo};
    use crate::{
        command_buffer::{pool::CommandBufferAllocateInfo, CommandBufferLevel},
        Validated, Version,
    };

    #[test]
    fn basic_create() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = CommandPool::new(
            &device,
            &CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_with_flags() {
        let (device, queue) = gfx_dev_and_queue!();
        let desired_flags = {
            let mut flags = CommandPoolCreateFlags::TRANSIENT;

            if device.api_version() >= Version::V1_1 {
                flags |= CommandPoolCreateFlags::PROTECTED;
            }
            flags
        };
        let pool = CommandPool::new(
            &device,
            &CommandPoolCreateInfo {
                flags: desired_flags,
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(desired_flags, pool.flags());
    }

    #[test]
    fn queue_family_getter() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = CommandPool::new(
            &device,
            &CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pool.queue_family_index(), queue.queue_family_index());
    }

    #[test]
    fn check_queue_family_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        match CommandPool::try_new(
            &device,
            &CommandPoolCreateInfo {
                ..Default::default()
            },
        ) {
            Err(Validated::ValidationError(_)) => (),
            _ => panic!(),
        }
    }

    // TODO: test that trim works if VK_KHR_maintenance1 if enabled ; the test macro doesn't
    //       support enabling extensions yet

    #[test]
    fn basic_alloc() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = CommandPool::new(
            &device,
            &CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();
        let vec = pool
            .allocate_command_buffers(&CommandBufferAllocateInfo {
                level: CommandBufferLevel::Primary,
                command_buffer_count: 12,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(vec.len(), 12);
    }
}
