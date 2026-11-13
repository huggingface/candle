use crate::{
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorSetLayoutCreateFlags, DescriptorType},
        sys::RawDescriptorSet,
    },
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use smallvec::SmallVec;
use std::{cell::Cell, marker::PhantomData, mem::MaybeUninit, num::NonZero, ptr, slice, sync::Arc};

/// Pool that descriptors are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
#[derive(Debug)]
pub struct DescriptorPool {
    handle: vk::DescriptorPool,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: DescriptorPoolCreateFlags,
    max_sets: u32,
    pool_sizes: Vec<(DescriptorType, u32)>,
    max_inline_uniform_block_bindings: u32,

    // Unimplement `Sync`, as Vulkan descriptor pools are not thread safe.
    _marker: PhantomData<Cell<vk::DescriptorPool>>,
}

impl DescriptorPool {
    /// Creates a new `DescriptorPool`, panicking on a validation error.
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
        device: &Arc<Device>,
        create_info: &DescriptorPoolCreateInfo<'_>,
    ) -> Result<DescriptorPool, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `DescriptorPool`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &DescriptorPoolCreateInfo<'_>,
    ) -> Result<DescriptorPool, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &DescriptorPoolCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateDescriptorPool-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &DescriptorPoolCreateInfo<'_>,
    ) -> Result<DescriptorPool, VulkanError> {
        let create_info_fields1_vk = create_info.to_vk_fields1();
        let mut create_info_extensions_vk = create_info.to_vk_extensions();
        let create_info_vk =
            create_info.to_vk(&create_info_fields1_vk, &mut create_info_extensions_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_descriptor_pool)(
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

    /// Creates a new `DescriptorPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::DescriptorPool,
        create_info: &DescriptorPoolCreateInfo<'_>,
    ) -> DescriptorPool {
        let &DescriptorPoolCreateInfo {
            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = create_info;

        let mut pool_sizes = Vec::with_capacity(pool_sizes.len());

        for &(descriptor_type, pool_size) in create_info.pool_sizes {
            if let Some(entry) = pool_sizes.iter_mut().find(|(ty, _)| *ty == descriptor_type) {
                entry.1 += pool_size;
            } else {
                pool_sizes.push((descriptor_type, pool_size));
            }
        }

        DescriptorPool {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,

            _marker: PhantomData,
        }
    }

    /// Returns the flags that the descriptor pool was created with.
    #[inline]
    pub fn flags(&self) -> DescriptorPoolCreateFlags {
        self.flags
    }

    /// Returns the maximum number of sets that can be allocated from the pool.
    #[inline]
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }

    /// Returns the number of descriptors of each type that the pool was created with.
    ///
    /// Each `DescriptorType` is unique even if the pool was created with duplicate
    /// `DescriptorType`s for its pool sizes. The elements are in no particular order.
    #[inline]
    pub fn pool_sizes(&self) -> &[(DescriptorType, u32)] {
        &self.pool_sizes
    }

    /// Returns the maximum number of [`DescriptorType::InlineUniformBlock`] bindings that can
    /// be allocated from the descriptor pool.
    #[inline]
    pub fn max_inline_uniform_block_bindings(&self) -> u32 {
        self.max_inline_uniform_block_bindings
    }

    /// Allocates a descriptor set from the pool, panicking on a validation error.
    ///
    /// The `FragmentedPool` errors often can't be prevented. If the function returns this error,
    /// you should just create a new pool.
    ///
    /// This is a shortcut for `try_allocate_descriptor_set().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - When the pool is dropped, the returned descriptor set must not be in use by either the
    ///   host or device.
    /// - If the device API version is less than 1.1, and the [`khr_maintenance1`] extension is not
    ///   enabled on the device, then there must be a descriptor set remaining in the pool, and the
    ///   total number of descriptors of each type being allocated must not be greater than the
    ///   number of descriptors of that type remaining in the pool.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_allocate_descriptor_set`] returns a [`ValidationError`].
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    /// [`try_allocate_descriptor_set`]: Self::try_allocate_descriptor_set
    #[inline]
    #[track_caller]
    pub unsafe fn allocate_descriptor_set(
        &self,
        allocate_info: &DescriptorSetAllocateInfo<'_>,
    ) -> Result<DescriptorPoolAlloc, VulkanError> {
        match unsafe { self.try_allocate_descriptor_set(allocate_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Allocates a descriptor set from the pool.
    ///
    /// The `FragmentedPool` errors often can't be prevented. If the function returns this error,
    /// you should just create a new pool.
    ///
    /// # Safety
    ///
    /// - When the pool is dropped, the returned descriptor set must not be in use by either the
    ///   host or device.
    /// - If the device API version is less than 1.1, and the [`khr_maintenance1`] extension is not
    ///   enabled on the device, then there must be a descriptor set remaining in the pool, and the
    ///   total number of descriptors of each type being allocated must not be greater than the
    ///   number of descriptors of that type remaining in the pool.
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    #[inline]
    pub unsafe fn try_allocate_descriptor_set(
        &self,
        allocate_info: &DescriptorSetAllocateInfo<'_>,
    ) -> Result<DescriptorPoolAlloc, Validated<VulkanError>> {
        self.validate_allocate_descriptor_sets(slice::from_ref(allocate_info))?;

        Ok(unsafe { self.allocate_descriptor_set_unchecked(allocate_info) }?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn allocate_descriptor_set_unchecked(
        &self,
        allocate_info: &DescriptorSetAllocateInfo<'_>,
    ) -> Result<DescriptorPoolAlloc, VulkanError> {
        unsafe { self.allocate_descriptor_sets_unchecked_inner(slice::from_ref(allocate_info)) }
            .map(|allocs| {
                let [alloc] = allocs.into_inner().unwrap();

                alloc
            })
    }

    /// Allocates descriptor sets from the pool, one for each element in `allocate_info`, panicking
    /// on a validation error.
    ///
    /// The `FragmentedPool` errors often can't be prevented. If the function returns this error,
    /// you should just create a new pool.
    ///
    /// This is a shortcut for `try_allocate_descriptor_sets().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - When the pool is dropped, the returned descriptor sets must not be in use by either the
    ///   host or device.
    /// - If the device API version is less than 1.1, and the [`khr_maintenance1`] extension is not
    ///   enabled on the device, then the length of `allocate_infos` must not be greater than the
    ///   number of descriptor sets remaining in the pool, and the total number of descriptors of
    ///   each type being allocated must not be greater than the number of descriptors of that type
    ///   remaining in the pool.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_allocate_descriptor_sets`] returns a [`ValidationError`].
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    /// [`try_allocate_descriptor_sets`]: Self::try_allocate_descriptor_sets
    #[inline]
    #[track_caller]
    pub unsafe fn allocate_descriptor_sets(
        &self,
        allocate_infos: &[DescriptorSetAllocateInfo<'_>],
    ) -> Result<Vec<DescriptorPoolAlloc>, VulkanError> {
        match unsafe { self.try_allocate_descriptor_sets(allocate_infos) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Allocates descriptor sets from the pool, one for each element in `allocate_info`.
    ///
    /// The `FragmentedPool` errors often can't be prevented. If the function returns this error,
    /// you should just create a new pool.
    ///
    /// # Safety
    ///
    /// - When the pool is dropped, the returned descriptor sets must not be in use by either the
    ///   host or device.
    /// - If the device API version is less than 1.1, and the [`khr_maintenance1`] extension is not
    ///   enabled on the device, then the length of `allocate_infos` must not be greater than the
    ///   number of descriptor sets remaining in the pool, and the total number of descriptors of
    ///   each type being allocated must not be greater than the number of descriptors of that type
    ///   remaining in the pool.
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    #[inline]
    pub unsafe fn try_allocate_descriptor_sets(
        &self,
        allocate_infos: &[DescriptorSetAllocateInfo<'_>],
    ) -> Result<Vec<DescriptorPoolAlloc>, Validated<VulkanError>> {
        self.validate_allocate_descriptor_sets(allocate_infos)?;

        Ok(unsafe { self.allocate_descriptor_sets_unchecked(allocate_infos) }?)
    }

    fn validate_allocate_descriptor_sets(
        &self,
        allocate_infos: &[DescriptorSetAllocateInfo<'_>],
    ) -> Result<(), Box<ValidationError>> {
        for (index, info) in allocate_infos.iter().enumerate() {
            info.validate(self.device())
                .map_err(|err| err.add_context(format!("allocate_infos[{}]", index)))?;

            let &DescriptorSetAllocateInfo {
                layout,
                variable_descriptor_count: _,
                _ne: _,
            } = info;

            if layout
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                && !self
                    .flags
                    .intersects(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`allocate_infos[{}].layout.flags()` contains \
                        `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`, but \
                        `self.flags` does not contain \
                        `DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkDescriptorSetAllocateInfo-pSetLayouts-03044"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn allocate_descriptor_sets_unchecked(
        &self,
        allocate_infos: &[DescriptorSetAllocateInfo<'_>],
    ) -> Result<Vec<DescriptorPoolAlloc>, VulkanError> {
        unsafe { self.allocate_descriptor_sets_unchecked_inner(allocate_infos) }
            .map(SmallVec::into_vec)
    }

    unsafe fn allocate_descriptor_sets_unchecked_inner(
        &self,
        allocate_infos: &[DescriptorSetAllocateInfo<'_>],
    ) -> Result<SmallVec<[DescriptorPoolAlloc; 1]>, VulkanError> {
        let mut layouts_vk: SmallVec<[_; 1]> = SmallVec::with_capacity(allocate_infos.len());
        let mut variable_descriptor_counts: SmallVec<[_; 1]> =
            SmallVec::with_capacity(allocate_infos.len());
        let mut layouts: SmallVec<[_; 1]> = SmallVec::with_capacity(allocate_infos.len());

        for info in allocate_infos {
            let &DescriptorSetAllocateInfo {
                layout,
                variable_descriptor_count,
                _ne: _,
            } = info;

            layouts_vk.push(layout.handle());
            variable_descriptor_counts.push(variable_descriptor_count);
            layouts.push(layout.clone());
        }

        let mut output: SmallVec<[_; 1]> = SmallVec::new();

        if !layouts_vk.is_empty() {
            let mut variable_desc_count_alloc_info = None;

            let mut info_vk = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.handle)
                .set_layouts(&layouts_vk);

            if (self.device.api_version() >= Version::V1_2
                || self.device.enabled_extensions().ext_descriptor_indexing)
                && variable_descriptor_counts.iter().any(|c| *c != 0)
            {
                let next = variable_desc_count_alloc_info.insert(
                    vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                        .descriptor_counts(&variable_descriptor_counts),
                );
                info_vk = info_vk.push(next);
            }

            output.reserve(layouts_vk.len());

            let fns = self.device.fns();
            unsafe {
                (fns.v1_0.allocate_descriptor_sets)(
                    self.device.handle(),
                    &info_vk,
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)
            .map_err(|err| match err {
                VulkanError::OutOfHostMemory
                | VulkanError::OutOfDeviceMemory
                | VulkanError::OutOfPoolMemory => err,
                // According to the specs, because `VK_ERROR_FRAGMENTED_POOL` was added after
                // version 1.0 of Vulkan, any negative return value except out-of-memory errors
                // must be considered as a fragmented pool error.
                _ => VulkanError::FragmentedPool,
            })?;

            unsafe { output.set_len(layouts_vk.len()) };
        }

        let descriptor_pool_allocs = output
            .into_iter()
            .zip(layouts)
            .zip(variable_descriptor_counts)
            .map(
                |((handle, layout), variable_descriptor_count)| DescriptorPoolAlloc {
                    handle,
                    id: DescriptorPoolAlloc::next_id(),
                    layout: DeviceOwnedDebugWrapper(layout),
                    variable_descriptor_count,
                },
            )
            .collect();

        Ok(descriptor_pool_allocs)
    }

    /// Frees some descriptor sets, panicking on a validation error.
    ///
    /// Note that it is not mandatory to free sets. Destroying or resetting the pool destroys all
    /// the descriptor sets.
    ///
    /// This is a shortcut for `try_free_descriptor_sets().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - All elements of `descriptor_sets` must have been allocated from `self`, and not freed
    ///   previously.
    /// - All elements of `descriptor_sets` must not be in use by the host or device.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_free_descriptor_sets`] returns a [`ValidationError`].
    ///
    /// [`try_free_descriptor_sets`]: Self::try_free_descriptor_sets
    #[inline]
    #[track_caller]
    pub unsafe fn free_descriptor_sets(
        &self,
        descriptor_sets: impl IntoIterator<Item = RawDescriptorSet>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_free_descriptor_sets(descriptor_sets) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Frees some descriptor sets.
    ///
    /// Note that it is not mandatory to free sets. Destroying or resetting the pool destroys all
    /// the descriptor sets.
    ///
    /// # Safety
    ///
    /// - All elements of `descriptor_sets` must have been allocated from `self`, and not freed
    ///   previously.
    /// - All elements of `descriptor_sets` must not be in use by the host or device.
    #[inline]
    pub unsafe fn try_free_descriptor_sets(
        &self,
        descriptor_sets: impl IntoIterator<Item = RawDescriptorSet>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_free_descriptor_sets()?;

        Ok(unsafe { self.free_descriptor_sets_unchecked(descriptor_sets) }?)
    }

    fn validate_free_descriptor_sets(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .flags
            .intersects(DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        {
            return Err(Box::new(ValidationError {
                context: "self.flags()".into(),
                problem: "does not contain `DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET`".into(),
                vuids: &["VUID-vkFreeDescriptorSets-descriptorPool-00312"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn free_descriptor_sets_unchecked(
        &self,
        descriptor_sets: impl IntoIterator<Item = RawDescriptorSet>,
    ) -> Result<(), VulkanError> {
        let sets: SmallVec<[_; 8]> = descriptor_sets.into_iter().map(|s| s.handle()).collect();
        if !sets.is_empty() {
            let fns = self.device.fns();
            unsafe {
                (fns.v1_0.free_descriptor_sets)(
                    self.device.handle(),
                    self.handle,
                    sets.len() as u32,
                    sets.as_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    /// Resets the pool, panicking on a validation error.
    ///
    /// This destroys all descriptor sets and empties the pool.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - All descriptor sets that were previously allocated from `self` must not be in use by the
    ///   host or device.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub unsafe fn reset(&self) -> Result<(), VulkanError> {
        match unsafe { self.try_reset() } {
            Ok(()) => Ok(()),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    ///
    /// # Safety
    ///
    /// - All descriptor sets that were previously allocated from `self` must not be in use by the
    ///   host or device.
    #[inline]
    pub unsafe fn try_reset(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_reset()?;

        Ok(unsafe { self.reset_unchecked() }?)
    }

    fn validate_reset(&self) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_unchecked(&self) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        unsafe {
            (fns.v1_0.reset_descriptor_pool)(
                self.device.handle(),
                self.handle,
                vk::DescriptorPoolResetFlags::empty(),
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }
}

impl Drop for DescriptorPool {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe {
            (fns.v1_0.destroy_descriptor_pool)(self.device.handle(), self.handle, ptr::null())
        };
    }
}

unsafe impl VulkanObject for DescriptorPool {
    type Handle = vk::DescriptorPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(DescriptorPool);

/// Parameters to create a new `DescriptorPool`.
#[derive(Clone, Debug)]
pub struct DescriptorPoolCreateInfo<'a> {
    /// Additional properties of the descriptor pool.
    ///
    /// The default value is empty.
    pub flags: DescriptorPoolCreateFlags,

    /// The maximum number of descriptor sets that can be allocated from the pool.
    ///
    /// The default value is `0`, which must be overridden.
    pub max_sets: u32,

    /// The number of descriptors of each type to allocate for the pool.
    ///
    /// If the descriptor type is [`DescriptorType::InlineUniformBlock`], then the value is the
    /// number of bytes to allocate for such descriptors. The value must then be a multiple of 4.
    ///
    /// The default value is empty, which must be overridden.
    pub pool_sizes: &'a [(DescriptorType, u32)],

    /// The maximum number of [`DescriptorType::InlineUniformBlock`] bindings that can be allocated
    /// from the descriptor pool.
    ///
    /// If this is not 0, the device API version must be at least 1.3, or the
    /// [`khr_inline_uniform_block`](crate::device::DeviceExtensions::ext_inline_uniform_block)
    /// extension must be enabled on the device.
    ///
    /// The default value is 0.
    pub max_inline_uniform_block_bindings: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DescriptorPoolCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DescriptorPoolCreateInfo<'a> {
    /// Returns a default `DescriptorPoolCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: DescriptorPoolCreateFlags::empty(),
            max_sets: 0,
            pool_sizes: &[],
            max_inline_uniform_block_bindings: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkDescriptorPoolCreateInfo-flags-parameter"])
        })?;

        if max_sets == 0 {
            return Err(Box::new(ValidationError {
                context: "max_sets".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkDescriptorPoolCreateInfo-maxSets-00301"],
                ..Default::default()
            }));
        }

        // VUID-VkDescriptorPoolCreateInfo-pPoolSizes-parameter
        for &(descriptor_type, pool_size) in pool_sizes {
            flags.validate_device(device).map_err(|err| {
                err.add_context("pool_sizes")
                    .set_vuids(&["VUID-VkDescriptorPoolSize-type-parameter"])
            })?;

            if pool_size == 0 {
                return Err(Box::new(ValidationError {
                    context: format!("pool_sizes[DescriptorType::{:?}]", descriptor_type).into(),
                    problem: "is zero".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-descriptorCount-00302"],
                    ..Default::default()
                }));
            }

            if descriptor_type == DescriptorType::InlineUniformBlock {
                return Err(Box::new(ValidationError {
                    context: "pool_sizes[DescriptorType::InlineUniformBlock]".into(),
                    problem: "is not a multiple of 4".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-type-02218"],
                    ..Default::default()
                }));
            }
        }

        if max_inline_uniform_block_bindings != 0
            && !(device.api_version() >= Version::V1_3
                || device.enabled_extensions().ext_inline_uniform_block)
        {
            return Err(Box::new(ValidationError {
                context: "max_inline_uniform_block_bindings".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("ext_inline_uniform_block")]),
                ]),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a DescriptorPoolCreateInfoFields1Vk,
        extensions_vk: &'a mut DescriptorPoolCreateInfoExtensionsVk,
    ) -> vk::DescriptorPoolCreateInfo<'a> {
        let &Self {
            flags,
            max_sets,
            pool_sizes: _,
            max_inline_uniform_block_bindings: _,
            _ne: _,
        } = self;
        let DescriptorPoolCreateInfoFields1Vk { pool_sizes_vk } = fields1_vk;

        let mut val_vk = vk::DescriptorPoolCreateInfo::default()
            .flags(flags.into())
            .max_sets(max_sets)
            .pool_sizes(pool_sizes_vk);

        let DescriptorPoolCreateInfoExtensionsVk {
            inline_uniform_block_vk,
        } = extensions_vk;

        if let Some(next) = inline_uniform_block_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_fields1(&self) -> DescriptorPoolCreateInfoFields1Vk {
        let pool_sizes_vk = self
            .pool_sizes
            .iter()
            .map(|&(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty: ty.into(),
                descriptor_count,
            })
            .collect();

        DescriptorPoolCreateInfoFields1Vk { pool_sizes_vk }
    }

    pub(crate) fn to_vk_extensions(&self) -> DescriptorPoolCreateInfoExtensionsVk {
        let inline_uniform_block_vk = (self.max_inline_uniform_block_bindings != 0).then(|| {
            vk::DescriptorPoolInlineUniformBlockCreateInfo::default()
                .max_inline_uniform_block_bindings(self.max_inline_uniform_block_bindings)
        });

        DescriptorPoolCreateInfoExtensionsVk {
            inline_uniform_block_vk,
        }
    }
}

pub(crate) struct DescriptorPoolCreateInfoExtensionsVk {
    pub(crate) inline_uniform_block_vk:
        Option<vk::DescriptorPoolInlineUniformBlockCreateInfo<'static>>,
}

pub(crate) struct DescriptorPoolCreateInfoFields1Vk {
    pub(crate) pool_sizes_vk: SmallVec<[vk::DescriptorPoolSize; 8]>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a descriptor pool.
    DescriptorPoolCreateFlags = DescriptorPoolCreateFlags(u32);

    /// Individual descriptor sets can be freed from the pool. Otherwise you must reset or
    /// destroy the whole pool at once.
    FREE_DESCRIPTOR_SET = FREE_DESCRIPTOR_SET,

    /// The pool can allocate descriptor sets with a layout whose flags include
    /// [`DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`].
    ///
    /// A pool created with this flag can still allocate descriptor sets without the flag.
    /// However, descriptor copy operations are only allowed between pools of the same type;
    /// it is not possible to copy between a descriptor set whose pool has `UPDATE_AFTER_BIND`,
    /// and a descriptor set whose pool does not have this flag.
    UPDATE_AFTER_BIND = UPDATE_AFTER_BIND
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),

    /* TODO: enable
    // TODO: document
    HOST_ONLY = HOST_ONLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mutable_descriptor_type)]),
        RequiresAllOf([DeviceExtension(valve_mutable_descriptor_type)]),
    ]), */
}

/// Parameters to allocate a new `DescriptorPoolAlloc` from a `DescriptorPool`.
#[derive(Clone, Debug)]
pub struct DescriptorSetAllocateInfo<'a> {
    /// The descriptor set layout to create the set for.
    ///
    /// There is no default value.
    pub layout: &'a Arc<DescriptorSetLayout>,

    /// For layouts with a variable-count binding, the number of descriptors to allocate for that
    /// binding. This should be 0 for layouts that don't have a variable-count binding.
    ///
    /// The default value is 0.
    pub variable_descriptor_count: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> DescriptorSetAllocateInfo<'a> {
    /// Returns a default `DescriptorSetAllocateInfo` with the provided `layout`.
    #[inline]
    pub const fn new(layout: &'a Arc<DescriptorSetLayout>) -> Self {
        Self {
            layout,
            variable_descriptor_count: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            layout,
            variable_descriptor_count,
            _ne,
        } = self;

        // VUID-VkDescriptorSetAllocateInfo-commonparent
        assert_eq!(device, layout.device().as_ref());

        if layout
            .flags()
            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
        {
            return Err(Box::new(ValidationError {
                context: "layout.flags()".into(),
                problem: "contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`".into(),
                vuids: &["VUID-VkDescriptorSetAllocateInfo-pSetLayouts-00308"],
                ..Default::default()
            }));
        }

        if variable_descriptor_count > layout.variable_descriptor_count() {
            return Err(Box::new(ValidationError {
                problem: "`variable_descriptor_count` is greater than
                    `layout.variable_descriptor_count()`"
                    .into(),
                // vuids: https://github.com/KhronosGroup/Vulkan-Docs/issues/2244
                ..Default::default()
            }));
        }

        Ok(())
    }
}

/// Opaque type that represents a descriptor set allocated from a pool.
#[derive(Debug)]
pub struct DescriptorPoolAlloc {
    handle: vk::DescriptorSet,
    id: NonZero<u64>,
    layout: DeviceOwnedDebugWrapper<Arc<DescriptorSetLayout>>,
    variable_descriptor_count: u32,
}

impl DescriptorPoolAlloc {
    /// Returns the descriptor set layout of the descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.layout
    }

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.variable_descriptor_count
    }
}

unsafe impl VulkanObject for DescriptorPoolAlloc {
    type Handle = vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DescriptorPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

impl_id_counter!(DescriptorPoolAlloc);

#[cfg(test)]
mod tests {
    use super::{DescriptorPool, DescriptorPoolCreateInfo};
    use crate::{
        descriptor_set::{
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            pool::DescriptorSetAllocateInfo,
        },
        shader::ShaderStages,
    };

    #[test]
    fn pool_create() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = DescriptorPool::new(
            &device,
            &DescriptorPoolCreateInfo {
                max_sets: 10,
                pool_sizes: &[(DescriptorType::UniformBuffer, 1)],
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_max_set() {
        let (device, _) = gfx_dev_and_queue!();

        DescriptorPool::try_new(
            &device,
            &DescriptorPoolCreateInfo {
                max_sets: 0,
                pool_sizes: &[(DescriptorType::UniformBuffer, 1)],
                ..Default::default()
            },
        )
        .unwrap_err();
    }

    #[test]
    fn basic_alloc() {
        let (device, _) = gfx_dev_and_queue!();

        let set_layout = DescriptorSetLayout::new(
            &device,
            &DescriptorSetLayoutCreateInfo {
                bindings: &[DescriptorSetLayoutBinding {
                    stages: ShaderStages::all_graphics(),
                    ..DescriptorSetLayoutBinding::new(DescriptorType::UniformBuffer)
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let pool = DescriptorPool::new(
            &device,
            &DescriptorPoolCreateInfo {
                max_sets: 10,
                pool_sizes: &[(DescriptorType::UniformBuffer, 10)],
                ..Default::default()
            },
        )
        .unwrap();
        let sets = unsafe {
            pool.allocate_descriptor_sets(&[DescriptorSetAllocateInfo::new(&set_layout)])
        }
        .unwrap();
        assert_eq!(sets.len(), 1);
    }

    #[test]
    fn alloc_diff_device() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set_layout = DescriptorSetLayout::new(
            &device1,
            &DescriptorSetLayoutCreateInfo {
                bindings: &[DescriptorSetLayoutBinding {
                    stages: ShaderStages::all_graphics(),
                    ..DescriptorSetLayoutBinding::new(DescriptorType::UniformBuffer)
                }],
                ..Default::default()
            },
        )
        .unwrap();

        assert_should_panic!({
            let pool = DescriptorPool::new(
                &device2,
                &DescriptorPoolCreateInfo {
                    max_sets: 10,
                    pool_sizes: &[(DescriptorType::UniformBuffer, 10)],
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = unsafe {
                pool.allocate_descriptor_sets(&[DescriptorSetAllocateInfo::new(&set_layout)])
            };
        });
    }

    #[test]
    fn alloc_zero() {
        let (device, _) = gfx_dev_and_queue!();

        let pool = DescriptorPool::new(
            &device,
            &DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_sizes: &[(DescriptorType::UniformBuffer, 1)],
                ..Default::default()
            },
        )
        .unwrap();
        let sets = unsafe { pool.allocate_descriptor_sets(&[]) }.unwrap();
        assert_eq!(sets.len(), 0);
    }
}
