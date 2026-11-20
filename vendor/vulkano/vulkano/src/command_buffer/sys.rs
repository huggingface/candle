use super::{
    allocator::{CommandBufferAlloc, CommandBufferAllocator},
    pool::CommandPoolCreateFlags,
    CommandBufferInheritanceInfo, CommandBufferInheritanceInfoExtensionsVk,
    CommandBufferInheritanceInfoFields1Vk, CommandBufferLevel, CommandBufferUsage,
};
use crate::{
    device::{Device, DeviceOwned, QueueFamilyProperties},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use std::{fmt::Debug, mem::ManuallyDrop, sync::Arc};

/// A command buffer in the recording state.
///
/// This type corresponds directly to a `VkCommandBuffer` after it has been allocated and started
/// recording. It doesn't keep track of synchronization or resource lifetimes. As such, all
/// recorded commands are unsafe and it is the user's duty to make sure that data races are
/// protected against using manual synchronization and all resources used by the recorded commands
/// outlive the command buffer.
///
/// Note that command buffers in the recording state don't implement the `Send` and `Sync` traits.
/// Once a command buffer has finished recording, however, it *does* implement `Send` and `Sync`.
pub struct RecordingCommandBuffer {
    allocation: ManuallyDrop<CommandBufferAlloc>,
    allocator: Arc<dyn CommandBufferAllocator>,
    queue_family_index: u32,
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    inheritance_info: Option<CommandBufferInheritanceInfo>,
    pub(super) usage: CommandBufferUsage,
}

impl RecordingCommandBuffer {
    /// Allocates and begins recording a new command buffer, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        allocator: &Arc<impl CommandBufferAllocator + ?Sized>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, VulkanError> {
        Self::new_inner(
            allocator.clone().as_dyn(),
            queue_family_index,
            level,
            begin_info,
        )
    }

    #[inline]
    #[track_caller]
    fn new_inner(
        allocator: Arc<dyn CommandBufferAllocator>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, VulkanError> {
        match Self::try_new_inner(allocator, queue_family_index, level, begin_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Allocates and begins recording a new command buffer.
    #[inline]
    pub fn try_new(
        allocator: &Arc<impl CommandBufferAllocator + ?Sized>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::try_new_inner(
            allocator.clone().as_dyn(),
            queue_family_index,
            level,
            begin_info,
        )
    }

    #[inline]
    fn try_new_inner(
        allocator: Arc<dyn CommandBufferAllocator>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::validate_new(allocator.device(), queue_family_index, level, begin_info)?;

        let allocation = allocator.try_allocate(queue_family_index, level)?;

        Ok(unsafe {
            Self::new_unchecked_inner_inner(allocator, allocation, queue_family_index, begin_info)
        }?)
    }

    pub(super) fn validate_new(
        device: &Device,
        _queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkBeginCommandBuffer-commandBuffer-00049
        // VUID-vkBeginCommandBuffer-commandBuffer-00050
        // Guaranteed by `CommandBufferAllocator`.

        if level == CommandBufferLevel::Secondary && begin_info.inheritance_info.is_none() {
            return Err(Box::new(ValidationError {
                context: "begin_info.inheritance_info".into(),
                problem: "is `None` while `level` is `CommandBufferLevel::Secondary`".into(),
                vuids: &["VUID-vkBeginCommandBuffer-commandBuffer-00051"],
                ..Default::default()
            }));
        }

        begin_info
            .validate(device)
            .map_err(|err| err.add_context("begin_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        allocator: &Arc<impl CommandBufferAllocator + ?Sized>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, VulkanError> {
        unsafe {
            Self::new_unchecked_inner(
                allocator.clone().as_dyn(),
                queue_family_index,
                level,
                begin_info,
            )
        }
    }

    unsafe fn new_unchecked_inner(
        allocator: Arc<dyn CommandBufferAllocator>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, VulkanError> {
        let allocation = unsafe { allocator.allocate_unchecked(queue_family_index, level) }?;

        unsafe {
            Self::new_unchecked_inner_inner(allocator, allocation, queue_family_index, begin_info)
        }
    }

    unsafe fn new_unchecked_inner_inner(
        allocator: Arc<dyn CommandBufferAllocator>,
        allocation: CommandBufferAlloc,
        queue_family_index: u32,
        begin_info: &CommandBufferBeginInfo<'_>,
    ) -> Result<Self, VulkanError> {
        {
            let begin_info_fields2_vk = begin_info.to_vk_fields2();
            let mut begin_info_fields1_extensions_vk =
                begin_info.to_vk_fields1_extensions(&begin_info_fields2_vk);
            let begin_info_fields1_vk =
                begin_info.to_vk_fields1(&mut begin_info_fields1_extensions_vk);
            let begin_info_vk = begin_info.to_vk(&begin_info_fields1_vk);

            let fns = allocation.inner.device().fns();
            unsafe { (fns.v1_0.begin_command_buffer)(allocation.inner.handle(), &begin_info_vk) }
                .result()
                .map_err(VulkanError::from)?;
        }

        let &CommandBufferBeginInfo {
            usage,
            inheritance_info,
            _ne: _,
        } = begin_info;

        Ok(RecordingCommandBuffer {
            allocation: ManuallyDrop::new(allocation),
            allocator,
            inheritance_info: inheritance_info.cloned(),
            queue_family_index,
            usage,
        })
    }

    /// Ends the recording, returning a command buffer which can be submitted.
    #[inline]
    pub unsafe fn end(self) -> Result<CommandBuffer, VulkanError> {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.end_command_buffer)(self.handle()) }
            .result()
            .map_err(VulkanError::from)?;

        Ok(CommandBuffer { inner: self })
    }

    /// Returns the queue family index that this command buffer was created for.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.allocation.inner.level()
    }

    /// Returns the usage that the command buffer was created with.
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.usage
    }

    /// Returns the inheritance info of the command buffer, if it is a secondary command buffer.
    #[inline]
    pub fn inheritance_info(&self) -> Option<&CommandBufferInheritanceInfo> {
        self.inheritance_info.as_ref()
    }

    pub(in crate::command_buffer) fn queue_family_properties(&self) -> &QueueFamilyProperties {
        &self.device().physical_device().queue_family_properties()[self.queue_family_index as usize]
    }

    /// Returns `true` if the command buffer was created from a protected pool.
    #[inline]
    pub(crate) fn is_protected(&self) -> bool {
        self.allocation
            .pool
            .flags()
            .contains(CommandPoolCreateFlags::PROTECTED)
    }
}

impl Drop for RecordingCommandBuffer {
    #[inline]
    fn drop(&mut self) {
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        unsafe { self.allocator.deallocate(allocation) };
    }
}

unsafe impl VulkanObject for RecordingCommandBuffer {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.allocation.inner.handle()
    }
}

unsafe impl DeviceOwned for RecordingCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.allocation.inner.device()
    }
}

impl Debug for RecordingCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingCommandBuffer")
            .field("handle", &self.level())
            .field("level", &self.level())
            .field("usage", &self.usage)
            .finish()
    }
}

/// Parameters to begin recording a command buffer.
#[derive(Clone, Debug)]
pub struct CommandBufferBeginInfo<'a> {
    /// How the command buffer will be used.
    ///
    /// The default value is [`CommandBufferUsage::MultipleSubmit`].
    pub usage: CommandBufferUsage,

    /// For a secondary command buffer, this must be `Some`, containing the context that will be
    /// inherited from the primary command buffer. For a primary command buffer, this must be
    /// `None`.
    ///
    /// The default value is `None`.
    pub inheritance_info: Option<&'a CommandBufferInheritanceInfo>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CommandBufferBeginInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CommandBufferBeginInfo<'a> {
    /// Returns a default `CommandBufferBeginInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            usage: CommandBufferUsage::MultipleSubmit,
            inheritance_info: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            usage: _,
            inheritance_info,
            _ne: _,
        } = self;

        if let Some(inheritance_info) = &inheritance_info {
            inheritance_info
                .validate(device)
                .map_err(|err| err.add_context("inheritance_info"))?;
        } else {
            // VUID-vkBeginCommandBuffer-commandBuffer-02840
            // Ensured by the definition of the `CommandBufferUsage` enum.
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a BeginInfoFields1Vk<'_>,
    ) -> vk::CommandBufferBeginInfo<'a> {
        let &Self {
            usage,
            ref inheritance_info,
            _ne: _,
        } = self;

        let mut flags_vk = vk::CommandBufferUsageFlags::from(usage);

        if inheritance_info
            .as_ref()
            .is_some_and(|inheritance_info| inheritance_info.render_pass.is_some())
        {
            flags_vk |= vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;
        }

        let mut val_vk = vk::CommandBufferBeginInfo::default().flags(flags_vk);

        let BeginInfoFields1Vk {
            inheritance_info_vk,
        } = fields1_vk;

        if let Some(inheritance_info_vk) = inheritance_info_vk {
            val_vk = val_vk.inheritance_info(inheritance_info_vk);
        }

        val_vk
    }

    pub(crate) fn to_vk_fields1(
        &self,
        fields1_extensions_vk: &'a mut BeginInfoFields1ExtensionsVk<'_>,
    ) -> BeginInfoFields1Vk<'a> {
        let BeginInfoFields1ExtensionsVk {
            inheritance_info_vk,
        } = fields1_extensions_vk;

        let inheritance_info_vk = self
            .inheritance_info
            .as_ref()
            .zip(inheritance_info_vk.as_mut())
            .map(|(inheritance_info, inheritance_info_extensions_vk)| {
                inheritance_info.to_vk(inheritance_info_extensions_vk)
            });

        BeginInfoFields1Vk {
            inheritance_info_vk,
        }
    }

    pub(crate) fn to_vk_fields1_extensions(
        &self,
        fields2_vk: &'a BeginInfoFields2Vk,
    ) -> BeginInfoFields1ExtensionsVk<'a> {
        let BeginInfoFields2Vk {
            inheritance_info_fields1_vk,
        } = fields2_vk;

        let inheritance_info_vk = self
            .inheritance_info
            .as_ref()
            .zip(inheritance_info_fields1_vk.as_ref())
            .map(|(inheritance_info, inheritance_info_fields1_vk)| {
                inheritance_info.to_vk_extensions(inheritance_info_fields1_vk)
            });

        BeginInfoFields1ExtensionsVk {
            inheritance_info_vk,
        }
    }

    pub(crate) fn to_vk_fields2(&self) -> BeginInfoFields2Vk {
        let inheritance_info_fields1_vk = self
            .inheritance_info
            .as_ref()
            .map(|inheritance_info| inheritance_info.to_vk_fields1());

        BeginInfoFields2Vk {
            inheritance_info_fields1_vk,
        }
    }
}

pub(crate) struct BeginInfoFields1Vk<'a> {
    pub(crate) inheritance_info_vk: Option<vk::CommandBufferInheritanceInfo<'a>>,
}

pub(crate) struct BeginInfoFields1ExtensionsVk<'a> {
    pub(crate) inheritance_info_vk: Option<CommandBufferInheritanceInfoExtensionsVk<'a>>,
}

pub(crate) struct BeginInfoFields2Vk {
    pub(crate) inheritance_info_fields1_vk: Option<CommandBufferInheritanceInfoFields1Vk>,
}

/// A command buffer that has finished recording.
#[derive(Debug)]
pub struct CommandBuffer {
    inner: RecordingCommandBuffer,
}

// `RecordingCommandBuffer` is `!Send + !Sync` so that the implementation of
// `CommandBufferAllocator::allocate` can assume that a command buffer in the recording state
// doesn't leave the thread it was allocated on. However, as the safety contract states,
// `CommandBufferAllocator::deallocate` must account for the possibility that a command buffer is
// moved between threads after the recording is finished, and thus deallocated from another thread.
// That's why this is sound.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    /// Returns the queue family index that this command buffer was created for.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.inner.allocation.inner.level()
    }

    /// Returns the usage that the command buffer was created with.
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.inner.usage
    }

    /// Returns `true` if the command buffer was created from a protected pool.
    #[inline]
    pub(crate) fn is_protected(&self) -> bool {
        self.inner.is_protected()
    }

    /// Returns the inheritance info of the command buffer, if it is a secondary command buffer.
    #[inline]
    pub fn inheritance_info(&self) -> Option<&CommandBufferInheritanceInfo> {
        self.inner.inheritance_info.as_ref()
    }
}

unsafe impl VulkanObject for CommandBuffer {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.allocation.inner.handle()
    }
}

unsafe impl DeviceOwned for CommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.allocation.inner.device()
    }
}
