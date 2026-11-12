//! Low level implementation of buffers.
//!
//! This module contains low-level wrappers around the Vulkan buffer types. All
//! other buffer types of this library, and all custom buffer types
//! that you create must wrap around the types in this module.

use super::{Buffer, BufferCreateFlags, BufferMemory, BufferUsage};
use crate::{
    buffer::ExternalBufferInfo,
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    memory::{
        allocator::AllocationType, is_aligned, DedicatedTo, ExternalMemoryHandleTypes,
        MemoryAllocateFlags, MemoryPropertyFlags, MemoryRequirements, ResourceMemory,
    },
    sync::{OwnedSharing, Sharing},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use std::{marker::PhantomData, mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// A raw buffer, with no memory backing it.
///
/// This is the basic buffer type, a direct translation of a `VkBuffer` object, but it is mostly
/// useless in this form. After creating a raw buffer, you must call `bind_memory` to make a
/// complete buffer object.
#[derive(Debug)]
pub struct RawBuffer {
    handle: vk::Buffer,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: BufferCreateFlags,
    size: DeviceSize,
    usage: BufferUsage,
    sharing: OwnedSharing,
    external_memory_handle_types: ExternalMemoryHandleTypes,

    memory_requirements: MemoryRequirements,
    needs_destruction: bool,
}

impl RawBuffer {
    /// Creates a new `RawBuffer`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    /// - Panics if `create_info.sharing` is [`Concurrent`] with less than 2 items.
    /// - Panics if `create_info.size` is zero.
    /// - Panics if `create_info.usage` is empty.
    ///
    /// [`try_new`]: Self::try_new
    /// [`Concurrent`]: Sharing::Concurrent
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &BufferCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `RawBuffer`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.sharing` is [`Concurrent`] with less than 2 items.
    /// - Panics if `create_info.size` is zero.
    /// - Panics if `create_info.usage` is empty.
    ///
    /// [`Concurrent`]: Sharing::Concurrent
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &BufferCreateInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &BufferCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        // TODO: sparse_address_space_size and extended_sparse_address_space_size limits
        // VUID-vkCreateBuffer-flags-00911
        // VUID-vkCreateBuffer-flags-09383
        // VUID-vkCreateBuffer-flags-09384

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &BufferCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        let mut extensions_vk = create_info.to_vk_extensions();
        let create_info_vk = create_info.to_vk(&mut extensions_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_buffer)(
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

    /// Creates a new `RawBuffer` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    /// - If the buffer has memory bound to it, `bind_memory` must not be called on the returned
    ///   `RawBuffer`.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Buffer,
        create_info: &BufferCreateInfo<'_>,
    ) -> Self {
        unsafe { Self::from_handle_with_destruction(device, handle, create_info, true) }
    }

    /// Creates a new `RawBuffer` from a raw object handle. Unlike `from_handle`, the created
    /// `RawBuffer` does not destroy the inner buffer when dropped.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    /// - If the buffer has memory bound to it, `bind_memory` must not be called on the returned
    ///   `RawBuffer`.
    /// - Caller must ensure that the handle will not be destroyed for the lifetime of returned
    ///   `RawBuffer`.
    #[inline]
    pub unsafe fn from_handle_borrowed(
        device: &Arc<Device>,
        handle: vk::Buffer,
        create_info: &BufferCreateInfo<'_>,
    ) -> Self {
        unsafe { Self::from_handle_with_destruction(device, handle, create_info, false) }
    }

    unsafe fn from_handle_with_destruction(
        device: &Arc<Device>,
        handle: vk::Buffer,
        create_info: &BufferCreateInfo<'_>,
        needs_destruction: bool,
    ) -> Self {
        let &BufferCreateInfo {
            flags,
            size,
            usage,
            sharing,
            external_memory_handle_types,
            _ne: _,
        } = create_info;

        let mut memory_requirements = Self::get_memory_requirements(device, handle);

        debug_assert!(memory_requirements.layout.size() >= size);
        debug_assert!(memory_requirements.memory_type_bits != 0);

        // We have to manually enforce some additional requirements for some buffer types.
        let properties = device.physical_device().properties();
        if usage.intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(properties.min_texel_buffer_offset_alignment)
                .unwrap();
        }

        if usage.intersects(BufferUsage::STORAGE_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(properties.min_storage_buffer_offset_alignment)
                .unwrap();
        }

        if usage.intersects(BufferUsage::UNIFORM_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(properties.min_uniform_buffer_offset_alignment)
                .unwrap();
        }

        RawBuffer {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),
            flags,
            size,
            usage,
            sharing: sharing.to_owned(),
            external_memory_handle_types,
            memory_requirements,
            needs_destruction,
        }
    }

    fn get_memory_requirements(device: &Device, handle: vk::Buffer) -> MemoryRequirements {
        let info_vk = vk::BufferMemoryRequirementsInfo2::default().buffer(handle);

        let mut memory_requirements2_extensions_vk =
            MemoryRequirements::to_mut_vk2_extensions(device);
        let mut memory_requirements2_vk =
            MemoryRequirements::to_mut_vk2(&mut memory_requirements2_extensions_vk);

        let fns = device.fns();

        if device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_get_memory_requirements2
        {
            if device.api_version() >= Version::V1_1 {
                unsafe {
                    (fns.v1_1.get_buffer_memory_requirements2)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_get_memory_requirements2
                        .get_buffer_memory_requirements2_khr)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    )
                };
            }
        } else {
            unsafe {
                (fns.v1_0.get_buffer_memory_requirements)(
                    device.handle(),
                    handle,
                    &mut memory_requirements2_vk.memory_requirements,
                )
            };
        }

        // Unborrow
        let memory_requirements2_vk = vk::MemoryRequirements2 {
            _marker: PhantomData,
            ..memory_requirements2_vk
        };

        MemoryRequirements::from_vk2(
            &memory_requirements2_vk,
            &memory_requirements2_extensions_vk,
        )
    }

    /// Binds device memory to this buffer, panicking on a validation error.
    ///
    /// This is a shortcut for `try_bind_memory().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_bind_memory`] returns a [`ValidationError`].
    ///
    /// [`try_bind_memory`]: Self::try_bind_memory
    #[inline]
    #[track_caller]
    pub fn bind_memory(
        self,
        allocation: ResourceMemory,
    ) -> Result<Buffer, (VulkanError, RawBuffer, ResourceMemory)> {
        match self.try_bind_memory(allocation) {
            Ok(buffer) => Ok(buffer),
            Err((err, this, allocation)) => Err((err.unwrap(), this, allocation)),
        }
    }

    /// Binds device memory to this buffer.
    #[inline]
    pub fn try_bind_memory(
        self,
        allocation: ResourceMemory,
    ) -> Result<Buffer, (Validated<VulkanError>, RawBuffer, ResourceMemory)> {
        if let Err(err) = self.validate_bind_memory(&allocation) {
            return Err((err.into(), self, allocation));
        }

        unsafe { self.bind_memory_unchecked(allocation) }
            .map_err(|(err, buffer, allocation)| (err.into(), buffer, allocation))
    }

    fn validate_bind_memory(
        &self,
        allocation: &ResourceMemory,
    ) -> Result<(), Box<ValidationError>> {
        if self.flags().intersects(BufferCreateFlags::SPARSE_BINDING) {
            return Err(Box::new(ValidationError {
                context: "self.flags()".into(),
                problem: "contains `BufferCreateFlags::SPARSE_BINDING`".into(),
                vuids: &["VUID-VkBindBufferMemoryInfo-buffer-01030"],
                ..Default::default()
            }));
        }

        assert_ne!(allocation.allocation_type(), AllocationType::NonLinear);

        let physical_device = self.device().physical_device();

        let memory_requirements = &self.memory_requirements;
        let memory = allocation.device_memory();
        let memory_offset = allocation.offset();
        let memory_type =
            &physical_device.memory_properties().memory_types[memory.memory_type_index() as usize];

        // VUID-VkBindBufferMemoryInfo-commonparent
        assert_eq!(self.device(), memory.device());

        // VUID-VkBindBufferMemoryInfo-buffer-07459
        // Ensured by taking ownership of `RawBuffer`.

        // VUID-VkBindBufferMemoryInfo-memoryOffset-01031
        // Assume that `allocation` was created correctly.

        if memory_requirements.memory_type_bits & (1 << memory.memory_type_index()) == 0 {
            return Err(Box::new(ValidationError {
                problem: "`allocation.device_memory().memory_type_index()` is not a bit set in \
                    `self.memory_requirements().memory_type_bits`"
                    .into(),
                vuids: &["VUID-VkBindBufferMemoryInfo-memory-01035"],
                ..Default::default()
            }));
        }

        if !is_aligned(memory_offset, memory_requirements.layout.alignment()) {
            return Err(Box::new(ValidationError {
                problem: "`allocation.offset()` is not aligned according to \
                    `self.memory_requirements().layout.alignment()`"
                    .into(),
                vuids: &["VUID-VkBindBufferMemoryInfo-memoryOffset-01036"],
                ..Default::default()
            }));
        }

        if allocation.size() < memory_requirements.layout.size() {
            return Err(Box::new(ValidationError {
                problem: "`allocation.size()` is less than \
                    `self.memory_requirements().layout.size()`"
                    .into(),
                vuids: &["VUID-VkBindBufferMemoryInfo-size-01037"],
                ..Default::default()
            }));
        }

        if let Some(dedicated_to) = memory.dedicated_to() {
            match dedicated_to {
                DedicatedTo::Buffer(id) if id == self.id => {}
                _ => {
                    return Err(Box::new(ValidationError {
                        problem: "`allocation.device_memory()` is a dedicated allocation, but \
                            it is not dedicated to this buffer"
                            .into(),
                        vuids: &["VUID-VkBindBufferMemoryInfo-memory-01508"],
                        ..Default::default()
                    }));
                }
            }
            debug_assert!(memory_offset == 0); // This should be ensured by the allocator
        } else {
            if memory_requirements.requires_dedicated_allocation {
                return Err(Box::new(ValidationError {
                    problem: "`self.memory_requirements().requires_dedicated_allocation` is \
                        `true`, but `allocation.device_memory()` is not a dedicated allocation"
                        .into(),
                    vuids: &["VUID-VkBindBufferMemoryInfo-buffer-01444"],
                    ..Default::default()
                }));
            }
        }

        if self.flags.intersects(BufferCreateFlags::PROTECTED) {
            if !memory_type
                .property_flags
                .intersects(MemoryPropertyFlags::PROTECTED)
            {
                return Err(Box::new(ValidationError {
                    problem: "the buffer was created with `BufferCreateFlags::PROTECTED`, but \
                        the `property_flags` of the memory type of `allocation.device_memory()` \
                        does not contain `MemoryPropertyFlags::PROTECTED`"
                        .into(),
                    vuids: &["VUID-VkBindBufferMemoryInfo-None-01898"],
                    ..Default::default()
                }));
            }
        } else {
            if memory_type
                .property_flags
                .intersects(MemoryPropertyFlags::PROTECTED)
            {
                return Err(Box::new(ValidationError {
                    problem: "the buffer was not created with `BufferCreateFlags::PROTECTED`, but \
                        the `property_flags` of the memory type of `allocation.device_memory()` \
                        contains `MemoryPropertyFlags::PROTECTED`"
                        .into(),
                    vuids: &["VUID-VkBindBufferMemoryInfo-None-01899"],
                    ..Default::default()
                }));
            }
        }

        if !memory.export_handle_types().is_empty() {
            if !self
                .external_memory_handle_types
                .intersects(memory.export_handle_types())
            {
                return Err(Box::new(ValidationError {
                    problem:
                        "`allocation.device_memory().export_handle_types()` is not empty, but \
                        it does not share at least one handle type with \
                        `self.external_memory_handle_types()`"
                            .into(),
                    vuids: &["VUID-VkBindBufferMemoryInfo-memory-02726"],
                    ..Default::default()
                }));
            }

            for handle_type in memory.export_handle_types() {
                let external_buffer_properties = unsafe {
                    physical_device.external_buffer_properties_unchecked(&ExternalBufferInfo {
                        flags: self.flags,
                        usage: self.usage,
                        handle_type,
                        _ne: crate::NE,
                    })
                };

                if external_buffer_properties
                    .external_memory_properties
                    .dedicated_only
                    && !memory.is_dedicated()
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`allocation.device_memory().export_handle_types()` has the `{:?}` \
                            flag set, which requires a dedicated allocation as returned by \
                            `PhysicalDevice::external_buffer_properties`, but \
                            `allocation.device_memory()` is not a dedicated allocation",
                            handle_type,
                        )
                        .into(),
                        vuids: &["VUID-VkMemoryAllocateInfo-pNext-00639"],
                        ..Default::default()
                    }));
                }

                if !external_buffer_properties
                    .external_memory_properties
                    .exportable
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`allocation.device_memory().export_handle_types()` has the `{:?}` \
                            flag set, but the flag is not supported for exporting, as returned by \
                            `PhysicalDevice::external_buffer_properties`",
                            handle_type,
                        )
                        .into(),
                        vuids: &["VUID-VkExportMemoryAllocateInfo-handleTypes-00656"],
                        ..Default::default()
                    }));
                }

                if !external_buffer_properties
                    .external_memory_properties
                    .compatible_handle_types
                    .contains(memory.export_handle_types())
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`allocation.device_memory().export_handle_types()` has the `{:?}` \
                            flag set, but the flag is not compatible with the other flags set, as \
                            returned by `PhysicalDevice::external_buffer_properties`",
                            handle_type,
                        )
                        .into(),
                        vuids: &["VUID-VkExportMemoryAllocateInfo-handleTypes-00656"],
                        ..Default::default()
                    }));
                }
            }
        }

        if let Some(handle_type) = memory.imported_handle_type() {
            if !self.external_memory_handle_types.contains_enum(handle_type) {
                return Err(Box::new(ValidationError {
                    problem: "`allocation.device_memory()` is imported, but \
                        `self.external_memory_handle_types()` does not contain the imported \
                        handle type"
                        .into(),
                    vuids: &["VUID-VkBindBufferMemoryInfo-memory-02985"],
                    ..Default::default()
                }));
            }
        }

        if !self.device.enabled_extensions().ext_buffer_device_address
            && self.usage.intersects(BufferUsage::SHADER_DEVICE_ADDRESS)
            && !memory
                .flags()
                .intersects(MemoryAllocateFlags::DEVICE_ADDRESS)
        {
            return Err(Box::new(ValidationError {
                problem: "`self.usage()` contains `BufferUsage::SHADER_DEVICE_ADDRESS`, but \
                    `allocation.device_memory().flags()` does not contain \
                    `MemoryAllocateFlags::DEVICE_ADDRESS`"
                    .into(),
                vuids: &["VUID-VkBindBufferMemoryInfo-bufferDeviceAddress-03339"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_memory_unchecked(
        self,
        allocation: ResourceMemory,
    ) -> Result<Buffer, (VulkanError, RawBuffer, ResourceMemory)> {
        let bind_info_vk = allocation.to_vk_bind_buffer_memory_info(self.handle());

        let fns = self.device.fns();

        let result = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_bind_memory2
        {
            let bind_infos_vk = [bind_info_vk];

            if self.device.api_version() >= Version::V1_1 {
                unsafe {
                    (fns.v1_1.bind_buffer_memory2)(
                        self.device.handle(),
                        bind_infos_vk.len() as u32,
                        bind_infos_vk.as_ptr(),
                    )
                }
            } else {
                unsafe {
                    (fns.khr_bind_memory2.bind_buffer_memory2_khr)(
                        self.device.handle(),
                        bind_infos_vk.len() as u32,
                        bind_infos_vk.as_ptr(),
                    )
                }
            }
        } else {
            unsafe {
                (fns.v1_0.bind_buffer_memory)(
                    self.device.handle(),
                    bind_info_vk.buffer,
                    bind_info_vk.memory,
                    bind_info_vk.memory_offset,
                )
            }
        }
        .result();

        if let Err(err) = result {
            return Err((VulkanError::from(err), self, allocation));
        }

        Ok(Buffer::from_raw(self, BufferMemory::Normal(allocation)))
    }

    /// Converts a raw buffer into a full buffer without binding any memory.
    ///
    /// # Safety
    ///
    /// If `self.flags()` does not contain [`BufferCreateFlags::SPARSE_BINDING`]:
    ///
    /// - The buffer must already have a suitable memory allocation bound to it.
    ///
    /// If `self.flags()` does contain [`BufferCreateFlags::SPARSE_BINDING`]:
    ///
    /// - If `self.flags()` does not contain [`BufferCreateFlags::SPARSE_RESIDENCY`], then the
    ///   buffer must be fully bound with memory before its memory is accessed by the device.
    /// - If `self.flags()` contains [`BufferCreateFlags::SPARSE_RESIDENCY`], then you must ensure
    ///   that any reads from the buffer are prepared to handle unexpected or inconsistent values,
    ///   as determined by the [`residency_non_resident_strict`] device property.
    ///
    /// [`residency_non_resident_strict`]: crate::device::DeviceProperties::residency_non_resident_strict
    pub unsafe fn assume_bound(self) -> Buffer {
        let memory = if self.flags().intersects(BufferCreateFlags::SPARSE_BINDING) {
            BufferMemory::Sparse
        } else {
            BufferMemory::External
        };

        Buffer::from_raw(self, memory)
    }

    /// Returns the memory requirements for this buffer.
    pub fn memory_requirements(&self) -> &MemoryRequirements {
        &self.memory_requirements
    }

    /// Returns the flags the buffer was created with.
    #[inline]
    pub fn flags(&self) -> BufferCreateFlags {
        self.flags
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the usage the buffer was created with.
    #[inline]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Returns the sharing the buffer was created with.
    #[inline]
    pub fn sharing(&self) -> Sharing<'_> {
        self.sharing.as_ref()
    }

    /// Returns the external memory handle types that are supported with this buffer.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.external_memory_handle_types
    }
}

impl Drop for RawBuffer {
    #[inline]
    fn drop(&mut self) {
        if self.needs_destruction {
            let fns = self.device.fns();
            unsafe { (fns.v1_0.destroy_buffer)(self.device.handle(), self.handle, ptr::null()) };
        }
    }
}

unsafe impl VulkanObject for RawBuffer {
    type Handle = vk::Buffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RawBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(RawBuffer);

/// Parameters to create a new [`Buffer`].
#[derive(Clone, Debug)]
pub struct BufferCreateInfo<'a> {
    /// Additional properties of the buffer.
    ///
    /// The default value is empty.
    pub flags: BufferCreateFlags,

    /// Whether the buffer can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<'a>,

    /// The size in bytes of the buffer.
    ///
    /// When using the [`Buffer`] constructors, you must leave this at `0`. They fill this field
    /// based on the data type of the contents and the other parameters you provide, and then pass
    /// this create-info to [`RawBuffer::new`]. You must override the default when constructing
    /// [`RawBuffer`] directly.
    ///
    /// The default value is `0`.
    pub size: DeviceSize,

    /// How the buffer is going to be used.
    ///
    /// The default value is [`BufferUsage::empty()`], which must be overridden.
    pub usage: BufferUsage,

    /// The external memory handle types that are going to be used with the buffer.
    ///
    /// If this value is not empty, then the device API version must be at least 1.1, or the
    /// [`khr_external_memory`] extension must be enabled on the device.
    ///
    /// The default value is [`ExternalMemoryHandleTypes::empty()`].
    ///
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    pub external_memory_handle_types: ExternalMemoryHandleTypes,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BufferCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> BufferCreateInfo<'a> {
    /// Returns a default `BufferCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: BufferCreateFlags::empty(),
            sharing: Sharing::Exclusive,
            size: 0,
            usage: BufferUsage::empty(),
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            sharing,
            size,
            usage,
            external_memory_handle_types,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkBufferCreateInfo-flags-parameter"])
        })?;

        usage.validate_device(device).map_err(|err| {
            err.add_context("usage")
                .set_vuids(&["VUID-VkBufferCreateInfo-usage-parameter"])
        })?;

        if usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkBufferCreateInfo-usage-requiredbitmask"],
                ..Default::default()
            }));
        }

        if size == 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkBufferCreateInfo-size-00912"],
                ..Default::default()
            }));
        }

        match sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                if queue_family_indices.len() < 2 {
                    return Err(Box::new(ValidationError {
                        context: "sharing".into(),
                        problem: "is `Sharing::Concurrent`, but contains less than 2 elements"
                            .into(),
                        vuids: &["VUID-VkBufferCreateInfo-sharingMode-00914"],
                        ..Default::default()
                    }));
                }

                let queue_family_count =
                    device.physical_device().queue_family_properties().len() as u32;

                for (index, &queue_family_index) in queue_family_indices.iter().enumerate() {
                    if queue_family_indices[..index].contains(&queue_family_index) {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_indices".into(),
                            problem: format!(
                                "the queue family index in the list at index {} is contained in \
                                the list more than once",
                                index,
                            )
                            .into(),
                            vuids: &["VUID-VkBufferCreateInfo-sharingMode-01419"],
                            ..Default::default()
                        }));
                    }

                    if queue_family_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: format!("sharing[{}]", index).into(),
                            problem: "is not less than the number of queue families in the device"
                                .into(),
                            vuids: &["VUID-VkBufferCreateInfo-sharingMode-01419"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if flags.intersects(BufferCreateFlags::PROTECTED) {
            if !device.enabled_features().protected_memory {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `BufferCreateFlags::PROTECTED`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "protected_memory",
                    )])]),
                    vuids: &["VUID-VkBufferCreateInfo-flags-01887"],
                }));
            }

            // If VK_BUFFER_CREATE_SPARSE_ALIASED_BIT support is added, then it should be added to
            // this validation.
            if flags
                .intersects(BufferCreateFlags::SPARSE_BINDING | BufferCreateFlags::SPARSE_RESIDENCY)
            {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `BufferCreateFlags::PROTECTED`, but also \
                        contains one of the sparse memory flags"
                        .into(),
                    vuids: &["VUID-VkBufferCreateInfo-None-01888"],
                    ..Default::default()
                }));
            }

            // If VK_BUFFER_USAGE_2_VIDEO_DECODE_SRC_BIT_KHR,
            // VK_BUFFER_USAGE_2_VIDEO_ENCODE_DST_BIT_KHR,
            // or VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT get added, then they should be added to
            // the allowed list.
            if !(usage
                - (BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::UNIFORM_TEXEL_BUFFER
                    | BufferUsage::STORAGE_TEXEL_BUFFER
                    | BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::STORAGE_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS))
                .is_empty()
            {
                return Err(Box::new(ValidationError {
                    problem: "contains `BufferCreateFlags::PROTECTED`, but also contains \
                        usages other than [allowed list]"
                        .into(),
                    vuids: &["VUID-VkBufferCreateInfo-flags-09641"],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(BufferCreateFlags::SPARSE_BINDING) {
            if !device.enabled_features().sparse_binding {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `BufferCreateFlags::SPARSE_BINDING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "sparse_binding",
                    )])]),
                    vuids: &["VUID-VkBufferCreateInfo-flags-00915"],
                }));
            }
        }

        if flags.intersects(BufferCreateFlags::SPARSE_RESIDENCY) {
            if !flags.intersects(BufferCreateFlags::SPARSE_BINDING) {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `BufferCreateFlags::SPARSE_RESIDENCY`, but does not also \
                        contain `BufferCreateFlags::SPARSE_BINDING`"
                        .into(),
                    vuids: &["VUID-VkBufferCreateInfo-flags-00918"],
                    ..Default::default()
                }));
            }

            if !device.enabled_features().sparse_residency_buffer {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `BufferCreateFlags::SPARSE_RESIDENCY`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "sparse_residency_buffer",
                    )])]),
                    vuids: &["VUID-VkBufferCreateInfo-flags-00916"],
                }));
            }
        }

        if let Some(max_buffer_size) = device.physical_device().properties().max_buffer_size {
            if size > max_buffer_size {
                return Err(Box::new(ValidationError {
                    context: "size".into(),
                    problem: "exceeds the `max_buffer_size` limit".into(),
                    vuids: &["VUID-VkBufferCreateInfo-size-06409"],
                    ..Default::default()
                }));
            }
        }

        if !external_memory_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(Box::new(ValidationError {
                    context: "external_memory_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_memory")]),
                    ]),
                    ..Default::default()
                }));
            }

            external_memory_handle_types
                .validate_device(device)
                .map_err(|err| {
                    err.add_context("external_memory_handle_types")
                        .set_vuids(&["VUID-VkExternalMemoryBufferCreateInfo-handleTypes-parameter"])
                })?;

            // TODO:
            // VUID-VkBufferCreateInfo-pNext-00920
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &'a self,
        extensions_vk: &'a mut BufferCreateInfoExtensionsVk,
    ) -> vk::BufferCreateInfo<'a> {
        let &Self {
            flags,
            sharing,
            size,
            usage,
            external_memory_handle_types: _,
            _ne: _,
        } = self;

        let (sharing_mode_vk, queue_family_indices) = sharing.to_vk();

        let mut val_vk = vk::BufferCreateInfo::default()
            .flags(flags.into())
            .size(size)
            .usage(usage.into())
            .sharing_mode(sharing_mode_vk)
            .queue_family_indices(queue_family_indices);

        let BufferCreateInfoExtensionsVk { external_memory_vk } = extensions_vk;

        if let Some(next) = external_memory_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(&self) -> BufferCreateInfoExtensionsVk {
        let &Self {
            external_memory_handle_types,
            ..
        } = self;

        let external_memory_vk = (!external_memory_handle_types.is_empty()).then(|| {
            vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(external_memory_handle_types.into())
        });

        BufferCreateInfoExtensionsVk { external_memory_vk }
    }
}

pub(crate) struct BufferCreateInfoExtensionsVk {
    pub(crate) external_memory_vk: Option<vk::ExternalMemoryBufferCreateInfo<'static>>,
}

#[cfg(test)]
mod tests {
    use super::{BufferCreateFlags, BufferCreateInfo, BufferUsage, RawBuffer};
    use crate::{device::DeviceOwned, Validated};

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let buf = RawBuffer::new(
            &device,
            &BufferCreateInfo {
                size: 128,
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
        )
        .unwrap();
        let reqs = buf.memory_requirements();

        assert!(reqs.layout.size() >= 128);
        assert_eq!(buf.size(), 128);
        assert_eq!(buf.device(), &device);
    }

    #[test]
    fn validate_no_protected_sparse() {
        let (device, _) = gfx_dev_and_queue!(protected_memory, sparse_binding);
        match RawBuffer::try_new(
            &device,
            &BufferCreateInfo {
                size: 128,
                flags: (BufferCreateFlags::PROTECTED | BufferCreateFlags::SPARSE_BINDING),
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
        ) {
            Err(Validated::ValidationError(_)) => {}
            Err(Validated::Error(err)) => {
                panic!(
                    "Expected a ValidationError, but got a runtime error: {:?}",
                    err
                );
            }
            Ok(_) => {
                panic!("RawBuffer::try_new succeeded when it should have failed!");
            }
        };
    }

    #[test]
    fn validate_invalid_protected_usage() {
        let (device, _) = gfx_dev_and_queue!(protected_memory);
        match RawBuffer::try_new(
            &device,
            &BufferCreateInfo {
                size: 128,
                flags: BufferCreateFlags::PROTECTED,
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
        ) {
            Err(Validated::ValidationError(_)) => {}
            Err(Validated::Error(err)) => {
                panic!(
                    "Expected a ValidationError, but got a runtime error: {:?}",
                    err
                );
            }
            Ok(_) => {
                panic!("RawBuffer::try_new succeeded when it should have failed!");
            }
        };
    }

    /* Re-enable when sparse binding is properly implemented
    #[test]
    fn missing_feature_sparse_binding() {
        let (device, _) = gfx_dev_and_queue!();
        match RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 128,
                sparse: Some(BufferCreateFlags::empty()),
                usage: BufferUsage::transfer_dst,
                ..Default::default()
            },
        ) {
            Err(BufferError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"sparse_binding") => (),
            _ => panic!(),
        }
    }

    #[test]
    fn missing_feature_sparse_residency() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        match RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 128,
                sparse: Some(BufferCreateFlags {
                    sparse_residency: true,
                    sparse_aliased: false,
                    ..Default::default()
                }),
                usage: BufferUsage::transfer_dst,
                ..Default::default()
            },
        ) {
            Err(BufferError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"sparse_residency_buffer") => (),
            _ => panic!(),
        }
    }

    #[test]
    fn missing_feature_sparse_aliased() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        match RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 128,
                sparse: Some(BufferCreateFlags {
                    sparse_residency: false,
                    sparse_aliased: true,
                    ..Default::default()
                }),
                usage: BufferUsage::transfer_dst,
                ..Default::default()
            },
        ) {
            Err(BufferError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"sparse_residency_aliased") => (),
            _ => panic!(),
        }
    }
    */

    #[test]
    fn create_empty_buffer() {
        let (device, _) = gfx_dev_and_queue!();

        RawBuffer::try_new(
            &device,
            &BufferCreateInfo {
                size: 0,
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
        )
        .unwrap_err();
    }
}
