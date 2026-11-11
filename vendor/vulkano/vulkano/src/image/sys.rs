//! Low-level implementation of images.
//!
//! This module contains low-level wrappers around the Vulkan image types. All
//! other image types of this library, and all custom image types
//! that you create must wrap around the types in this module.
//!
//! See also [the parent module-level documentation] for more information about images.
//!
//! [the parent module-level documentation]: super

use super::{
    Image, ImageAspect, ImageAspects, ImageCreateFlags, ImageLayout, ImageMemory,
    ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageUsage, SampleCount,
    SparseImageMemoryRequirements, SubresourceLayout,
};
#[cfg(doc)]
use crate::format::DrmFormatModifierProperties;
use crate::{
    cache::OnceCache,
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, FormatFeatures},
    image::{
        max_mip_levels, ImageDrmFormatModifierInfo, ImageFormatInfo, ImageFormatProperties,
        ImageType,
    },
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    memory::{
        allocator::AllocationType, is_aligned, DedicatedTo, ExternalMemoryHandleTypes,
        MemoryPropertyFlags, MemoryRequirements, ResourceMemory,
    },
    sync::{OwnedSharing, Sharing},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use smallvec::{smallvec, SmallVec};
use std::{marker::PhantomData, mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// A raw image, with no memory backing it.
///
/// This is the basic image type, a direct translation of a `VkImage` object, but it is mostly
/// useless in this form. After creating a raw image, you must call `bind_memory` to make a
/// complete image object.
///
/// See also [the parent module-level documentation] for more information about images.
///
/// [the parent module-level documentation]: super
#[derive(Debug)]
pub struct RawImage {
    handle: vk::Image,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: ImageCreateFlags,
    image_type: ImageType,
    format: Format,
    format_features: FormatFeatures,
    view_formats: Vec<Format>,
    extent: [u32; 3],
    array_layers: u32,
    mip_levels: u32,
    samples: SampleCount,
    tiling: ImageTiling,
    usage: ImageUsage,
    stencil_usage: Option<ImageUsage>,
    sharing: OwnedSharing,
    initial_layout: ImageLayout,
    drm_format_modifier: Option<(u64, u32)>,
    external_memory_handle_types: ExternalMemoryHandleTypes,

    memory_requirements: SmallVec<[MemoryRequirements; 4]>,
    sparse_memory_requirements: Vec<SparseImageMemoryRequirements>,
    needs_destruction: bool, // `vkDestroyImage` is called only if true.
    subresource_layout: OnceCache<(ImageAspect, u32, u32), SubresourceLayout>,
}

impl RawImage {
    /// Creates a new `RawImage`, panicking on a validation error.
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
        create_info: &ImageCreateInfo<'_>,
    ) -> Result<RawImage, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `RawImage`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &ImageCreateInfo<'_>,
    ) -> Result<RawImage, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { RawImage::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &ImageCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        // TODO: sparse_address_space_size and extended_sparse_address_space_size limits
        // VUID-vkCreateImage-flags-00939
        // VUID-vkCreateImage-flags-09385
        // VUID-vkCreateImage-flags-09386

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &ImageCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        let create_info_fields1_vk = create_info.to_vk_fields1();
        let mut create_info_extensions_vk = create_info.to_vk_extensions(&create_info_fields1_vk);
        let create_info_vk = create_info.to_vk(&mut create_info_extensions_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_image)(
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

        unsafe { Self::from_handle(device, handle, create_info) }
    }

    /// Creates a new `RawImage` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    /// - If the image has memory bound to it, `bind_memory` must not be called on the returned
    ///   `RawImage`.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Image,
        create_info: &ImageCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        unsafe { Self::from_handle_with_destruction(device, handle, create_info, true) }
    }

    /// Creates a new `RawImage` from a raw object handle. Unlike `from_handle`, the created
    /// `RawImage` will not destroy the inner image when dropped.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    /// - If the image has memory bound to it, `bind_memory` must not be called on the returned
    ///   `RawImage`.
    /// - Caller must ensure the handle will not be destroyed for the lifetime of returned
    ///   `RawImage`.
    #[inline]
    pub unsafe fn from_handle_borrowed(
        device: &Arc<Device>,
        handle: vk::Image,
        create_info: &ImageCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        unsafe { Self::from_handle_with_destruction(device, handle, create_info, false) }
    }

    pub(super) unsafe fn from_handle_with_destruction(
        device: &Arc<Device>,
        handle: vk::Image,
        create_info: &ImageCreateInfo<'_>,
        needs_destruction: bool,
    ) -> Result<Self, VulkanError> {
        let &ImageCreateInfo {
            flags,
            image_type,
            format,
            view_formats,
            extent,
            array_layers,
            mip_levels,
            samples,
            tiling,
            usage,
            stencil_usage,
            sharing,
            initial_layout,
            drm_format_modifiers: _,
            drm_format_modifier_plane_layouts: _,
            external_memory_handle_types,
            _ne: _,
        } = create_info;

        let format_properties =
            unsafe { device.physical_device().format_properties_unchecked(format) };

        let drm_format_modifier = if tiling == ImageTiling::DrmFormatModifier {
            let drm_format_modifier =
                unsafe { Self::get_drm_format_modifier_properties(device, handle) }?;
            let drm_format_modifier_plane_count = format_properties
                .drm_format_modifier_properties
                .iter()
                .find(|properties| properties.drm_format_modifier == drm_format_modifier)
                .expect("couldn't get the DRM format modifier plane count for the image")
                .drm_format_modifier_plane_count;
            Some((drm_format_modifier, drm_format_modifier_plane_count))
        } else {
            None
        };

        let format_features = {
            let drm_format_modifiers: SmallVec<[_; 1]> =
                drm_format_modifier.map_or_else(Default::default, |(m, _)| smallvec![m]);
            format_properties.format_features(tiling, &drm_format_modifiers)
        };

        let memory_requirements = if needs_destruction {
            if flags.intersects(ImageCreateFlags::DISJOINT) {
                // VUID-VkImageMemoryRequirementsInfo2-image-01589
                // VUID-VkImageMemoryRequirementsInfo2-image-02279
                let plane_count = drm_format_modifier.map_or_else(
                    || format.planes().len(),
                    |(_, plane_count)| plane_count as usize,
                );

                (0..plane_count)
                    .map(|plane| unsafe {
                        Self::get_memory_requirements(device, handle, Some((plane, tiling)))
                    })
                    .collect()
            } else {
                // VUID-VkImageMemoryRequirementsInfo2-image-01590
                smallvec![unsafe { Self::get_memory_requirements(device, handle, None) }]
            }
        } else {
            smallvec![]
        };

        let sparse_memory_requirements = if flags
            .contains(ImageCreateFlags::SPARSE_BINDING | ImageCreateFlags::SPARSE_RESIDENCY)
        {
            unsafe { Self::get_sparse_memory_requirements(device, handle) }
        } else {
            Vec::new()
        };

        Ok(RawImage {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            flags,
            image_type,
            format,
            format_features,
            view_formats: view_formats.to_owned(),
            extent,
            array_layers,
            mip_levels,
            initial_layout,
            samples,
            tiling,
            usage,
            stencil_usage,
            sharing: sharing.to_owned(),
            drm_format_modifier,
            external_memory_handle_types,

            memory_requirements,
            sparse_memory_requirements,
            needs_destruction,
            subresource_layout: OnceCache::new(),
        })
    }

    unsafe fn get_memory_requirements(
        device: &Device,
        handle: vk::Image,
        plane_tiling: Option<(usize, ImageTiling)>,
    ) -> MemoryRequirements {
        let mut info_vk = vk::ImageMemoryRequirementsInfo2::default().image(handle);
        let mut plane_info_vk = None;

        if let Some((plane, tiling)) = plane_tiling {
            debug_assert!(
                device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_get_memory_requirements2
            );

            let plane_aspect = match tiling {
                // VUID-VkImagePlaneMemoryRequirementsInfo-planeAspect-02281
                ImageTiling::Optimal | ImageTiling::Linear => {
                    debug_assert!(
                        device.api_version() >= Version::V1_1
                            || device.enabled_extensions().khr_sampler_ycbcr_conversion
                    );
                    match plane {
                        0 => vk::ImageAspectFlags::PLANE_0,
                        1 => vk::ImageAspectFlags::PLANE_1,
                        2 => vk::ImageAspectFlags::PLANE_2,
                        _ => unreachable!(),
                    }
                }
                // VUID-VkImagePlaneMemoryRequirementsInfo-planeAspect-02282
                ImageTiling::DrmFormatModifier => {
                    debug_assert!(device.enabled_extensions().ext_image_drm_format_modifier);
                    match plane {
                        0 => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                        1 => vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                        2 => vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
                        3 => vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
                        _ => unreachable!(),
                    }
                }
            };

            let next = plane_info_vk
                .insert(vk::ImagePlaneMemoryRequirementsInfo::default().plane_aspect(plane_aspect));
            info_vk = info_vk.push(next);
        }

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
                    (fns.v1_1.get_image_memory_requirements2)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_get_memory_requirements2
                        .get_image_memory_requirements2_khr)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    )
                };
            }
        } else {
            unsafe {
                (fns.v1_0.get_image_memory_requirements)(
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

    unsafe fn get_sparse_memory_requirements(
        device: &Device,
        handle: vk::Image,
    ) -> Vec<SparseImageMemoryRequirements> {
        let fns = device.fns();

        if device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_get_memory_requirements2
        {
            let info2_vk = vk::ImageSparseMemoryRequirementsInfo2::default().image(handle);

            let mut count = 0;

            if device.api_version() >= Version::V1_1 {
                unsafe {
                    (fns.v1_1.get_image_sparse_memory_requirements2)(
                        device.handle(),
                        &info2_vk,
                        &mut count,
                        ptr::null_mut(),
                    )
                };
            } else {
                unsafe {
                    (fns.khr_get_memory_requirements2
                        .get_image_sparse_memory_requirements2_khr)(
                        device.handle(),
                        &info2_vk,
                        &mut count,
                        ptr::null_mut(),
                    )
                };
            }

            let mut requirements2_vk =
                vec![SparseImageMemoryRequirements::to_mut_vk2(); count as usize];

            if device.api_version() >= Version::V1_1 {
                unsafe {
                    (fns.v1_1.get_image_sparse_memory_requirements2)(
                        device.handle(),
                        &info2_vk,
                        &mut count,
                        requirements2_vk.as_mut_ptr(),
                    )
                };
            } else {
                unsafe {
                    (fns.khr_get_memory_requirements2
                        .get_image_sparse_memory_requirements2_khr)(
                        device.handle(),
                        &info2_vk,
                        &mut count,
                        requirements2_vk.as_mut_ptr(),
                    )
                };
            }

            unsafe { requirements2_vk.set_len(count as usize) };
            requirements2_vk
                .iter()
                .map(SparseImageMemoryRequirements::from_vk2)
                .collect()
        } else {
            let mut count = 0;

            unsafe {
                (fns.v1_0.get_image_sparse_memory_requirements)(
                    device.handle(),
                    handle,
                    &mut count,
                    ptr::null_mut(),
                )
            };

            let mut requirements_vk =
                vec![SparseImageMemoryRequirements::to_mut_vk(); count as usize];

            unsafe {
                (fns.v1_0.get_image_sparse_memory_requirements)(
                    device.handle(),
                    handle,
                    &mut count,
                    requirements_vk.as_mut_ptr(),
                )
            };

            unsafe { requirements_vk.set_len(count as usize) };
            requirements_vk
                .iter()
                .map(SparseImageMemoryRequirements::from_vk)
                .collect()
        }
    }

    unsafe fn get_drm_format_modifier_properties(
        device: &Device,
        handle: vk::Image,
    ) -> Result<u64, VulkanError> {
        let mut properties_vk = vk::ImageDrmFormatModifierPropertiesEXT::default();

        let fns = device.fns();
        unsafe {
            (fns.ext_image_drm_format_modifier
                .get_image_drm_format_modifier_properties_ext)(
                device.handle(),
                handle,
                &mut properties_vk,
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(properties_vk.drm_format_modifier)
    }

    /// Binds device memory to this image, panicking on a validation error.
    ///
    /// - If `self.flags()` does not contain `ImageCreateFlags::DISJOINT`, then `allocations` must
    ///   contain exactly one element.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and `self.tiling()` is
    ///   `ImageTiling::Linear` or `ImageTiling::Optimal`, then `allocations` must contain exactly
    ///   `self.format().unwrap().planes().len()` elements.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and `self.tiling()` is
    ///   `ImageTiling::DrmFormatModifier`, then `allocations` must contain exactly
    ///   `self.drm_format_modifier().unwrap().1` elements.
    ///
    /// This is a shortcut for `try_bind_memory().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_bind_memory`] returns a [`ValidationError`].
    ///
    /// [`try_bind_memory`]: Self::try_bind_memory
    #[track_caller]
    pub fn bind_memory(
        self,
        allocations: impl IntoIterator<Item = ResourceMemory>,
    ) -> Result<Image, (VulkanError, RawImage, Vec<ResourceMemory>)> {
        let allocations = allocations.into_iter().collect();

        self.bind_memory_inner(allocations)
    }

    #[track_caller]
    fn bind_memory_inner(
        self,
        allocations: SmallVec<[ResourceMemory; 4]>,
    ) -> Result<Image, (VulkanError, RawImage, Vec<ResourceMemory>)> {
        match self.try_bind_memory_inner(allocations) {
            Ok(res) => Ok(res),
            Err((err, raw_image, allocations)) => Err((err.unwrap(), raw_image, allocations)),
        }
    }

    /// Binds device memory to this image.
    ///
    /// - If `self.flags()` does not contain `ImageCreateFlags::DISJOINT`, then `allocations` must
    ///   contain exactly one element.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and `self.tiling()` is
    ///   `ImageTiling::Linear` or `ImageTiling::Optimal`, then `allocations` must contain exactly
    ///   `self.format().unwrap().planes().len()` elements.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and `self.tiling()` is
    ///   `ImageTiling::DrmFormatModifier`, then `allocations` must contain exactly
    ///   `self.drm_format_modifier().unwrap().1` elements.
    pub fn try_bind_memory(
        self,
        allocations: impl IntoIterator<Item = ResourceMemory>,
    ) -> Result<Image, (Validated<VulkanError>, RawImage, Vec<ResourceMemory>)> {
        let allocations = allocations.into_iter().collect();

        self.try_bind_memory_inner(allocations)
    }

    fn try_bind_memory_inner(
        self,
        allocations: SmallVec<[ResourceMemory; 4]>,
    ) -> Result<Image, (Validated<VulkanError>, RawImage, Vec<ResourceMemory>)> {
        if let Err(err) = self.validate_bind_memory(&allocations) {
            return Err((err.into(), self, allocations.into_vec()));
        }

        unsafe { self.bind_memory_unchecked_inner(allocations) }
            .map_err(|(err, image, allocations)| (err.into(), image, allocations))
    }

    fn validate_bind_memory(
        &self,
        allocations: &[ResourceMemory],
    ) -> Result<(), Box<ValidationError>> {
        if self.flags().intersects(ImageCreateFlags::SPARSE_BINDING) {
            return Err(Box::new(ValidationError {
                context: "self.flags()".into(),
                problem: "contains `ImageCreateFlags::SPARSE_BINDING`".into(),
                vuids: &["VUID-VkBindImageMemoryInfo-image-01045"],
                ..Default::default()
            }));
        }

        let physical_device = self.device().physical_device();

        if self.flags.intersects(ImageCreateFlags::DISJOINT) {
            match self.tiling {
                ImageTiling::Optimal | ImageTiling::Linear => {
                    if allocations.len() != self.format.planes().len() {
                        return Err(Box::new(ValidationError {
                            problem: "`self.flags()` contains `ImageCreateFlags::DISJOINT`, and \
                                `self.tiling()` is `ImageTiling::Optimal` or \
                                `ImageTiling::Linear`, but the length of `allocations` does not \
                                equal the number of planes in the format of the image"
                                .into(),
                            ..Default::default()
                        }));
                    }
                }
                ImageTiling::DrmFormatModifier => {
                    if allocations.len() != self.drm_format_modifier.unwrap().1 as usize {
                        return Err(Box::new(ValidationError {
                            problem: "`self.flags()` contains `ImageCreateFlags::DISJOINT`, and \
                                `self.tiling()` is `ImageTiling::DrmFormatModifier`, but the \
                                length of `allocations` does not equal the number of memory planes \
                                of the DRM format modifier of the image"
                                .into(),
                            ..Default::default()
                        }));
                    }
                }
            }
        } else {
            if allocations.len() != 1 {
                return Err(Box::new(ValidationError {
                    problem: "`self.flags()` does not contain `ImageCreateFlags::DISJOINT`, but \
                        the length of `allocations` is not 1"
                        .into(),
                    ..Default::default()
                }));
            }
        }

        for (index, (allocation, memory_requirements)) in allocations
            .iter()
            .zip(self.memory_requirements.iter())
            .enumerate()
        {
            match allocation.allocation_type() {
                AllocationType::Unknown => {
                    // This allocation type is suitable for all image tilings by definition.
                }
                AllocationType::Linear => {
                    if self.tiling() != ImageTiling::Linear {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`allocations[{}].allocation_type()` is `AllocationType::Linear`, \
                                but `self.tiling()` is not `ImageTiling::Linear`",
                                index
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }
                }
                AllocationType::NonLinear => {
                    if self.tiling() != ImageTiling::Optimal {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`allocations[{}].allocation_type()` is \
                                `AllocationType::NonLinear`, but `self.tiling()` is not \
                                `ImageTiling::Optimal`",
                                index
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }
                }
            }

            let memory = allocation.device_memory();
            let memory_offset = allocation.offset();
            let memory_type = &physical_device.memory_properties().memory_types
                [memory.memory_type_index() as usize];

            // VUID-VkBindImageMemoryInfo-commonparent
            assert_eq!(self.device(), memory.device());

            // VUID-VkBindImageMemoryInfo-image-07460
            // Ensured by taking ownership of `RawImage`.

            // VUID-VkBindImageMemoryInfo-memoryOffset-01046
            // Assume that `allocation` was created correctly.

            if memory_requirements.memory_type_bits & (1 << memory.memory_type_index()) == 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`allocation[{}].device_memory().memory_type_index()` is not a bit set in \
                        `self.memory_requirements().memory_type_bits`",
                        index
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBindImageMemoryInfo-pNext-01615",
                        "VUID-VkBindImageMemoryInfo-pNext-01619",
                    ],
                    ..Default::default()
                }));
            }

            if !is_aligned(memory_offset, memory_requirements.layout.alignment()) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`allocations[{}].offset()` is not aligned according to \
                        `self.memory_requirements().layout.alignment()`",
                        index
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBindImageMemoryInfo-pNext-01616",
                        "VUID-VkBindImageMemoryInfo-pNext-01620",
                    ],
                    ..Default::default()
                }));
            }

            if allocation.size() < memory_requirements.layout.size() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`allocations[{}].size()` is less than \
                        `self.memory_requirements().layout.size()`",
                        index
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBindImageMemoryInfo-pNext-01617",
                        "VUID-VkBindImageMemoryInfo-pNext-01621",
                    ],
                    ..Default::default()
                }));
            }

            if let Some(dedicated_to) = memory.dedicated_to() {
                match dedicated_to {
                    DedicatedTo::Image(id) if id == self.id => {}
                    _ => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`allocations[{}].device_memory()` is a dedicated allocation, but \
                                it is not dedicated to this image",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkBindImageMemoryInfo-memory-02628"],
                            ..Default::default()
                        }));
                    }
                }
                debug_assert!(memory_offset == 0); // This should be ensured by the allocator
            } else {
                if memory_requirements.requires_dedicated_allocation {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`self.memory_requirements().requires_dedicated_allocation` is \
                            `true`, but `allocations[{}].device_memory()` is not a \
                            dedicated allocation",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-image-01445"],
                        ..Default::default()
                    }));
                }
            }

            if self.flags.intersects(ImageCreateFlags::PROTECTED) {
                if !memory_type
                    .property_flags
                    .intersects(MemoryPropertyFlags::PROTECTED)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "the image was created with `ImageCreateFlags::PROTECTED`, \
                            but the `property_flags` of the memory type of \
                            `allocations[{}].device_memory()` does not contain \
                            `MemoryPropertyFlags::PROTECTED`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-None-01901"],
                        ..Default::default()
                    }));
                }
            } else {
                if memory_type
                    .property_flags
                    .intersects(MemoryPropertyFlags::PROTECTED)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "the image was not created with `ImageCreateFlags::PROTECTED`, \
                            but the `property_flags` of the memory type of \
                            `allocations[{}].device_memory()` contains \
                            `MemoryPropertyFlags::PROTECTED`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-None-01902"],
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
                        problem: format!(
                            "`allocations[{}].device_memory().export_handle_types()` is not empty, \
                            but it does not share at least one memory type with \
                            `self.external_memory_handle_types()`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-memory-02728"],
                        ..Default::default()
                    }));
                }

                for handle_type in memory.export_handle_types() {
                    let image_format_properties = unsafe {
                        physical_device.image_format_properties_unchecked(&ImageFormatInfo {
                            flags: self.flags,
                            format: self.format,
                            view_formats: &self.view_formats,
                            image_type: self.image_type,
                            tiling: self.tiling,
                            usage: self.usage,
                            stencil_usage: self.stencil_usage,
                            drm_format_modifier_info: self
                                .drm_format_modifier()
                                .map(|(drm_format_modifier, _)| ImageDrmFormatModifierInfo {
                                    drm_format_modifier,
                                    sharing: self.sharing.as_ref(),
                                    _ne: crate::NE,
                                })
                                .as_ref(),
                            external_memory_handle_type: Some(handle_type),
                            image_view_type: None,
                            _ne: crate::NE,
                        })
                    }
                    .map_err(|_| {
                        Box::new(ValidationError {
                            problem: "`PhysicalDevice::image_format_properties` returned an error"
                                .into(),
                            ..Default::default()
                        })
                    })?
                    .unwrap();

                    if image_format_properties
                        .external_memory_properties
                        .dedicated_only
                        && !memory.is_dedicated()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`allocations[{}].device_memory().export_handle_types()` has the \
                                `{:?}` flag set, which requires a dedicated allocation as returned \
                                by `PhysicalDevice::image_format_properties`, but \
                                `allocations[{}].device_memory()` is not a dedicated allocation",
                                index, handle_type, index,
                            )
                            .into(),
                            vuids: &["VUID-VkMemoryAllocateInfo-pNext-00639"],
                            ..Default::default()
                        }));
                    }

                    if !image_format_properties
                        .external_memory_properties
                        .exportable
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`allocations[{}].device_memory().export_handle_types()` has the \
                                `{:?}` flag set, but the flag is not supported for exporting, as \
                                returned by `PhysicalDevice::image_format_properties`",
                                index, handle_type,
                            )
                            .into(),
                            vuids: &["VUID-VkExportMemoryAllocateInfo-handleTypes-00656"],
                            ..Default::default()
                        }));
                    }

                    if !image_format_properties
                        .external_memory_properties
                        .compatible_handle_types
                        .contains(memory.export_handle_types())
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`allocation.device_memory().export_handle_types()` has the `{:?}` \
                                flag set, but the flag is not compatible with the other flags set, \
                                as returned by `PhysicalDevice::image_format_properties`",
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
                if !ExternalMemoryHandleTypes::from(handle_type)
                    .intersects(self.external_memory_handle_types)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`allocations[{}].device_memory()` is imported, but \
                            `self.external_memory_handle_types()` does not contain the imported \
                            handle type",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-memory-02989"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    /// # Safety
    ///
    /// - If `self.flags()` does not contain `ImageCreateFlags::DISJOINT`, then `allocations` must
    ///   contain exactly one element.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and `self.tiling()` is
    ///   `ImageTiling::Linear` or `ImageTiling::Optimal`, then `allocations` must contain exactly
    ///   `self.format().unwrap().planes().len()` elements.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and `self.tiling()` is
    ///   `ImageTiling::DrmFormatModifier`, then `allocations` must contain exactly
    ///   `self.drm_format_modifier().unwrap().1` elements.
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_memory_unchecked(
        self,
        allocations: impl IntoIterator<Item = ResourceMemory>,
    ) -> Result<Image, (VulkanError, RawImage, Vec<ResourceMemory>)> {
        let allocations = allocations.into_iter().collect();

        unsafe { self.bind_memory_unchecked_inner(allocations) }
    }

    unsafe fn bind_memory_unchecked_inner(
        self,
        allocations: SmallVec<[ResourceMemory; 4]>,
    ) -> Result<Image, (VulkanError, RawImage, Vec<ResourceMemory>)> {
        const PLANE_ASPECTS_VK_NORMAL: &[vk::ImageAspectFlags] = &[
            vk::ImageAspectFlags::PLANE_0,
            vk::ImageAspectFlags::PLANE_1,
            vk::ImageAspectFlags::PLANE_2,
        ];
        const PLANE_ASPECTS_VK_DRM_FORMAT_MODIFIER: &[vk::ImageAspectFlags] = &[
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
        ];
        let needs_plane = (self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_bind_memory2)
            && self.flags.intersects(ImageCreateFlags::DISJOINT);

        let plane_aspects_vk = if needs_plane {
            Some(match self.tiling {
                // VUID-VkBindImagePlaneMemoryInfo-planeAspect-02283
                ImageTiling::Optimal | ImageTiling::Linear => {
                    let plane_count = self.format.planes().len();
                    &PLANE_ASPECTS_VK_NORMAL[..plane_count]
                }
                // VUID-VkBindImagePlaneMemoryInfo-planeAspect-02284
                ImageTiling::DrmFormatModifier => {
                    let plane_count = self.drm_format_modifier.unwrap().1 as usize;
                    &PLANE_ASPECTS_VK_DRM_FORMAT_MODIFIER[..plane_count]
                }
            })
        } else {
            debug_assert_eq!(allocations.len(), 1);
            None
        };

        let mut plane_infos_vk: SmallVec<[_; 4]> = (0..allocations.len())
            .map(|plane_num| {
                plane_aspects_vk.map(|plane_aspects_vk| {
                    let plane_aspect_vk = plane_aspects_vk[plane_num];
                    vk::BindImagePlaneMemoryInfo::default().plane_aspect(plane_aspect_vk)
                })
            })
            .collect();

        let infos_vk: SmallVec<[_; 4]> = allocations
            .iter()
            .zip(&mut plane_infos_vk)
            .map(|(allocation, plane_info_vk)| {
                let mut info_vk = allocation.to_vk_bind_image_memory_info(self.handle);

                if let Some(next) = plane_info_vk {
                    info_vk = info_vk.push(next);
                }

                info_vk
            })
            .collect();

        let fns = self.device.fns();

        let result = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_bind_memory2
        {
            if self.device.api_version() >= Version::V1_1 {
                unsafe {
                    (fns.v1_1.bind_image_memory2)(
                        self.device.handle(),
                        infos_vk.len() as u32,
                        infos_vk.as_ptr(),
                    )
                }
            } else {
                unsafe {
                    (fns.khr_bind_memory2.bind_image_memory2_khr)(
                        self.device.handle(),
                        infos_vk.len() as u32,
                        infos_vk.as_ptr(),
                    )
                }
            }
        } else {
            let info_vk = &infos_vk[0];

            unsafe {
                (fns.v1_0.bind_image_memory)(
                    self.device.handle(),
                    info_vk.image,
                    info_vk.memory,
                    info_vk.memory_offset,
                )
            }
        }
        .result();

        if let Err(err) = result {
            return Err((VulkanError::from(err), self, allocations.into_vec()));
        }

        let layout = self.default_layout();

        Ok(unsafe { Image::from_raw(self, ImageMemory::Normal(allocations), layout) })
    }

    /// Converts a raw image into a full image without binding any memory.
    ///
    /// # Safety
    ///
    /// If `self.flags()` does not contain [`ImageCreateFlags::SPARSE_BINDING`]:
    ///
    /// - The image must already have a suitable memory allocation bound to it.
    ///
    /// If `self.flags()` does contain [`ImageCreateFlags::SPARSE_BINDING`]:
    ///
    /// - If `self.flags()` does not contain [`ImageCreateFlags::SPARSE_RESIDENCY`], then the image
    ///   must be fully bound with memory before any memory is accessed by the device.
    /// - If `self.flags()` contains [`ImageCreateFlags::SPARSE_RESIDENCY`], then if the sparse
    ///   memory requirements include [`ImageAspects::METADATA`], then the metadata mip tail must
    ///   be fully bound with memory before any memory is accessed by the device.
    /// - If `self.flags()` contains [`ImageCreateFlags::SPARSE_RESIDENCY`], then you must ensure
    ///   that any reads from the image are prepared to handle unexpected or inconsistent values,
    ///   as determined by the [`residency_non_resident_strict`] device property.
    ///
    /// [`residency_non_resident_strict`]: crate::device::DeviceProperties::residency_non_resident_strict
    pub unsafe fn assume_bound(self) -> Image {
        let memory = if self.flags().intersects(ImageCreateFlags::SPARSE_BINDING) {
            ImageMemory::Sparse
        } else {
            ImageMemory::External
        };
        let layout = self.default_layout();

        unsafe { Image::from_raw(self, memory, layout) }
    }

    fn default_layout(&self) -> ImageLayout {
        let usage = self
            .usage
            .difference(ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST);

        if usage.intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
            && usage
                .difference(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                .is_empty()
        {
            ImageLayout::ShaderReadOnlyOptimal
        } else if usage.intersects(ImageUsage::COLOR_ATTACHMENT)
            && usage.difference(ImageUsage::COLOR_ATTACHMENT).is_empty()
        {
            ImageLayout::ColorAttachmentOptimal
        } else if usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            && usage
                .difference(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                .is_empty()
        {
            ImageLayout::DepthStencilAttachmentOptimal
        } else {
            ImageLayout::General
        }
    }

    /// Returns the memory requirements for this image.
    ///
    /// - If the image is a swapchain image, this returns a slice with a length of 0.
    /// - If `self.flags()` does not contain `ImageCreateFlags::DISJOINT`, this returns a slice
    ///   with a length of 1.
    /// - If `self.flags()` does contain `ImageCreateFlags::DISJOINT`, this returns a slice with a
    ///   length equal to `self.format().unwrap().planes().len()`.
    #[inline]
    pub fn memory_requirements(&self) -> &[MemoryRequirements] {
        &self.memory_requirements
    }

    /// Returns the sparse memory requirements for this image.
    ///
    /// If `self.flags()` does not contain both `ImageCreateFlags::SPARSE_BINDING` and
    /// `ImageCreateFlags::SPARSE_RESIDENCY`, this returns an empty slice.
    #[inline]
    pub fn sparse_memory_requirements(&self) -> &[SparseImageMemoryRequirements] {
        &self.sparse_memory_requirements
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.flags
    }

    /// Returns the image type of the image.
    #[inline]
    pub fn image_type(&self) -> ImageType {
        self.image_type
    }

    /// Returns the image's format.
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns the formats that an image view created from this image can have.
    #[inline]
    pub fn view_formats(&self) -> &[Format] {
        &self.view_formats
    }

    /// Returns the extent of the image.
    #[inline]
    pub fn extent(&self) -> [u32; 3] {
        self.extent
    }

    /// Returns the number of array layers in the image.
    #[inline]
    pub fn array_layers(&self) -> u32 {
        self.array_layers
    }

    /// Returns the number of mip levels in the image.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Returns the initial layout of the image.
    #[inline]
    pub fn initial_layout(&self) -> ImageLayout {
        self.initial_layout
    }

    /// Returns the number of samples for the image.
    #[inline]
    pub fn samples(&self) -> SampleCount {
        self.samples
    }

    /// Returns the tiling of the image.
    #[inline]
    pub fn tiling(&self) -> ImageTiling {
        self.tiling
    }

    /// Returns the usage the image was created with.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    /// Returns the stencil usage the image was created with.
    #[inline]
    pub fn stencil_usage(&self) -> Option<ImageUsage> {
        self.stencil_usage
    }

    /// Returns the sharing the image was created with.
    #[inline]
    pub fn sharing(&self) -> Sharing<'_> {
        self.sharing.as_ref()
    }

    /// If `self.tiling()` is `ImageTiling::DrmFormatModifier`, returns the DRM format modifier
    /// of the image, and the number of memory planes.
    /// This was either provided in [`ImageCreateInfo::drm_format_modifiers`], or if
    /// multiple modifiers were provided, selected from the list by the Vulkan implementation.
    #[inline]
    pub fn drm_format_modifier(&self) -> Option<(u64, u32)> {
        self.drm_format_modifier
    }

    /// Returns the external memory handle types that are supported with this image.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.external_memory_handle_types
    }

    /// Returns an `ImageSubresourceLayers` covering the first mip level of the image. All aspects
    /// of the image are selected, or `plane0` if the image is multi-planar.
    #[inline]
    pub fn subresource_layers(&self) -> ImageSubresourceLayers {
        ImageSubresourceLayers {
            aspects: {
                let aspects = self.format.aspects();

                if aspects.intersects(ImageAspects::PLANE_0) {
                    ImageAspects::PLANE_0
                } else {
                    aspects
                }
            },
            ..Default::default()
        }
    }

    /// Returns an `ImageSubresourceRange` covering the whole image. If the image is multi-planar,
    /// only the `COLOR` aspect is selected.
    #[inline]
    pub fn subresource_range(&self) -> ImageSubresourceRange {
        ImageSubresourceRange {
            aspects: self.format.aspects()
                - (ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2),
            ..Default::default()
        }
    }

    /// Queries the memory layout of a single subresource of the image, panicking on a validation
    /// error.
    ///
    /// Only images with linear tiling are supported, if they do not have a format with both a
    /// depth and a stencil format. Images with optimal tiling have an opaque image layout that is
    /// not suitable for direct memory accesses, and likewise for combined depth/stencil formats.
    /// Multi-planar formats are supported, but you must specify one of the planes as the `aspect`,
    /// not [`ImageAspect::COLOR`].
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// This is a shortcut for `try_subresource_layout().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_subresource_layout`] returns a [`ValidationError`].
    ///
    /// [`try_subresource_layout`]: Self::try_subresource_layout
    #[track_caller]
    pub fn subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> SubresourceLayout {
        self.try_subresource_layout(aspect, mip_level, array_layer)
            .unwrap()
    }

    /// Queries the memory layout of a single subresource of the image.
    ///
    /// Only images with linear tiling are supported, if they do not have a format with both a
    /// depth and a stencil format. Images with optimal tiling have an opaque image layout that is
    /// not suitable for direct memory accesses, and likewise for combined depth/stencil formats.
    /// Multi-planar formats are supported, but you must specify one of the planes as the `aspect`,
    /// not [`ImageAspect::COLOR`].
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    pub fn try_subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<SubresourceLayout, Box<ValidationError>> {
        self.validate_subresource_layout(aspect, mip_level, array_layer)?;

        Ok(unsafe { self.subresource_layout_unchecked(aspect, mip_level, array_layer) })
    }

    fn validate_subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<(), Box<ValidationError>> {
        aspect.validate_device(&self.device).map_err(|err| {
            err.add_context("aspect")
                .set_vuids(&["VUID-VkImageSubresource-aspectMask-parameter"])
        })?;

        // VUID-VkImageSubresource-aspectMask-requiredbitmask
        // VUID-vkGetImageSubresourceLayout-aspectMask-00997
        // Ensured by use of enum `ImageAspect`.

        if !matches!(
            self.tiling,
            ImageTiling::Linear | ImageTiling::DrmFormatModifier
        ) {
            return Err(Box::new(ValidationError {
                context: "self.tiling()".into(),
                problem: "is not `ImageTiling::Linear` or `ImageTiling::DrmFormatModifier`".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-image-02270"],
                ..Default::default()
            }));
        }

        if mip_level >= self.mip_levels {
            return Err(Box::new(ValidationError {
                context: "mip_level".into(),
                problem: "is greater than the number of mip levels in the image".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-mipLevel-01716"],
                ..Default::default()
            }));
        }

        if array_layer >= self.array_layers {
            return Err(Box::new(ValidationError {
                context: "array_layer".into(),
                problem: "is greater than the number of array layers in the image".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-arrayLayer-01717"],
                ..Default::default()
            }));
        }

        let format = self.format;
        let format_aspects = format.aspects();

        if let Some((_, drm_format_modifier_plane_count)) = self.drm_format_modifier {
            match drm_format_modifier_plane_count {
                1 => {
                    if !matches!(aspect, ImageAspect::MemoryPlane0) {
                        return Err(Box::new(ValidationError {
                            problem: "the image has a DRM format modifier with 1 memory plane, \
                                but `aspect` is not `ImageAspect::MemoryPlane0`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        }));
                    }
                }
                2 => {
                    if !matches!(
                        aspect,
                        ImageAspect::MemoryPlane0 | ImageAspect::MemoryPlane1
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "the image has a DRM format modifier with 2 memory planes, \
                                but `aspect` is not `ImageAspect::MemoryPlane0` or \
                                `ImageAspect::MemoryPlane1`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        }));
                    }
                }
                3 => {
                    if !matches!(
                        aspect,
                        ImageAspect::MemoryPlane0
                            | ImageAspect::MemoryPlane1
                            | ImageAspect::MemoryPlane2
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "the image has a DRM format modifier with 3 memory planes, \
                                but `aspect` is not `ImageAspect::MemoryPlane0`, \
                                `ImageAspect::MemoryPlane1` or `ImageAspect::MemoryPlane2`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        }));
                    }
                }
                4 => {
                    if !matches!(
                        aspect,
                        ImageAspect::MemoryPlane0
                            | ImageAspect::MemoryPlane1
                            | ImageAspect::MemoryPlane2
                            | ImageAspect::MemoryPlane3
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "the image has a DRM format modifier with 4 memory planes, \
                                but `aspect` is not `ImageAspect::MemoryPlane0`, \
                                `ImageAspect::MemoryPlane1`, `ImageAspect::MemoryPlane2` or \
                                `ImageAspect::MemoryPlane3`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        }));
                    }
                }
                _ => unreachable!("image has more than 4 memory planes??"),
            }
        } else if format_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            // Follows from the combination of these three VUIDs. See:
            // https://github.com/KhronosGroup/Vulkan-Docs/issues/1942
            return Err(Box::new(ValidationError {
                context: "self.format()".into(),
                problem: "has both a depth and a stencil aspect".into(),
                vuids: &[
                    "VUID-vkGetImageSubresourceLayout-aspectMask-00997",
                    "VUID-vkGetImageSubresourceLayout-format-04462",
                    "VUID-vkGetImageSubresourceLayout-format-04463",
                ],
                ..Default::default()
            }));
        } else if format_aspects.intersects(ImageAspects::DEPTH) {
            if aspect != ImageAspect::Depth {
                return Err(Box::new(ValidationError {
                    problem: "`self.format()` is a depth format, but \
                        `aspect` is not `ImageAspect::Depth`"
                        .into(),
                    vuids: &["VUID-vkGetImageSubresourceLayout-format-04462"],
                    ..Default::default()
                }));
            }
        } else if format_aspects.intersects(ImageAspects::STENCIL) {
            if aspect != ImageAspect::Stencil {
                return Err(Box::new(ValidationError {
                    problem: "`self.format()` is a stencil format, but \
                        `aspect` is not `ImageAspect::Stencil`"
                        .into(),
                    vuids: &["VUID-vkGetImageSubresourceLayout-format-04463"],
                    ..Default::default()
                }));
            }
        } else if format_aspects.intersects(ImageAspects::COLOR) {
            if format.planes().is_empty() {
                if aspect != ImageAspect::Color {
                    return Err(Box::new(ValidationError {
                        problem: "`self.format()` is a color format with a single plane, but \
                        `aspect` is not `ImageAspect::Color`"
                            .into(),
                        vuids: &["VUID-vkGetImageSubresourceLayout-format-08886"],
                        ..Default::default()
                    }));
                }
            } else if format.planes().len() == 2 {
                if !matches!(aspect, ImageAspect::Plane0 | ImageAspect::Plane1) {
                    return Err(Box::new(ValidationError {
                        problem: "`self.format()` is a color format with two planes, but \
                            `aspect` is not `ImageAspect::Plane0` or `ImageAspect::Plane1`"
                            .into(),
                        vuids: &["VUID-vkGetImageSubresourceLayout-tiling-08717"],
                        ..Default::default()
                    }));
                }
            } else if format.planes().len() == 3 {
                if !matches!(
                    aspect,
                    ImageAspect::Plane0 | ImageAspect::Plane1 | ImageAspect::Plane2
                ) {
                    return Err(Box::new(ValidationError {
                        problem: "`self.format()` is a color format with three planes, but \
                            `aspect` is not `ImageAspect::Plane0`, `ImageAspect::Plane1` or \
                            `ImageAspect::Plane2`"
                            .into(),
                        vuids: &["VUID-vkGetImageSubresourceLayout-tiling-08717"],
                        ..Default::default()
                    }));
                }
            }
        }

        // TODO:  VUID-vkGetImageSubresourceLayout-tiling-02271
        //if self.tiling == ImageTiling::DrmFormatModifier {
        // Only one-plane image importing is possible for now.
        //}

        // VUID-vkGetImageSubresourceLayout-format-08886
        // VUID-vkGetImageSubresourceLayout-format-04462
        // VUID-vkGetImageSubresourceLayout-format-04463
        // VUID-vkGetImageSubresourceLayout-format-04464
        // VUID-vkGetImageSubresourceLayout-format-01581
        // VUID-vkGetImageSubresourceLayout-format-01582
        if self.tiling != ImageTiling::DrmFormatModifier && !format_aspects.contains(aspect.into())
        {
            return Err(Box::new(ValidationError {
                context: "array_layer".into(),
                problem: "is greater than the number of array layers in the image".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-arrayLayer-01717"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn subresource_layout_unchecked(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> SubresourceLayout {
        self.subresource_layout
            .get_or_insert(&(aspect, mip_level, array_layer), || {
                let subresource_vk = vk::ImageSubresource {
                    aspect_mask: aspect.into(),
                    mip_level,
                    array_layer,
                };

                let vk::SubresourceLayout {
                    offset,
                    size,
                    row_pitch,
                    array_pitch,
                    depth_pitch,
                } = {
                    let fns = self.device.fns();
                    let mut output = MaybeUninit::uninit();
                    unsafe {
                        (fns.v1_0.get_image_subresource_layout)(
                            self.device.handle(),
                            self.handle,
                            &subresource_vk,
                            output.as_mut_ptr(),
                        )
                    };
                    unsafe { output.assume_init() }
                };

                (
                    (aspect, mip_level, array_layer),
                    SubresourceLayout {
                        offset,
                        size,
                        row_pitch,
                        array_pitch: (self.array_layers > 1).then_some(array_pitch),
                        depth_pitch: matches!(self.image_type, ImageType::Dim3d)
                            .then_some(depth_pitch),
                    },
                )
            })
    }
}

impl Drop for RawImage {
    #[inline]
    fn drop(&mut self) {
        if !self.needs_destruction {
            return;
        }

        let fns = self.device.fns();
        unsafe { (fns.v1_0.destroy_image)(self.device.handle(), self.handle, ptr::null()) };
    }
}

unsafe impl VulkanObject for RawImage {
    type Handle = vk::Image;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RawImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(RawImage);

/// Parameters to create a new `Image`.
#[derive(Clone, Debug)]
pub struct ImageCreateInfo<'a> {
    /// Additional properties of the image.
    ///
    /// The default value is empty.
    pub flags: ImageCreateFlags,

    /// The basic image dimensionality to create the image with.
    ///
    /// The default value is `ImageType::Dim2d`.
    pub image_type: ImageType,

    /// The format used to store the image data.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

    /// The formats that an image view can have when it is created from this image.
    ///
    /// If the list is not empty, and `flags` does not contain
    /// [`ImageCreateFlags::MUTABLE_FORMAT`], then the list must contain at most one element,
    /// otherwise any number of elements are allowed. The view formats must be compatible with
    /// `format`. If `flags` also contains [`ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`],
    /// then the view formats can also be uncompressed formats that are merely size-compatible
    /// with `format`.
    ///
    /// If the list is empty, then depending on `flags`, a view must have the same format as
    /// `format`, can have any compatible format, or additionally any uncompressed size-compatible
    /// format. However, this is less efficient than specifying the possible view formats
    /// in advance.
    ///
    /// If this is not empty, then the device API version must be at least 1.2, or the
    /// [`khr_image_format_list`] extension must be enabled on the device.
    ///
    /// The default value is empty.
    ///
    /// [`khr_image_format_list`]: crate::device::DeviceExtensions::khr_image_format_list
    pub view_formats: &'a [Format],

    /// The width, height and depth of the image.
    ///
    /// If `image_type` is `ImageType::Dim2d`, then the depth must be 1.
    /// If `image_type` is `ImageType::Dim1d`, then the height and depth must be 1.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    /// The number of array layers to create the image with.
    ///
    /// On [portability
    /// subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `samples` is not [`SampleCount::Sample1`] and `array_layers` is not 1,
    /// the [`multisample_array_image`](crate::device::DeviceFeatures::multisample_array_image)
    /// feature must be enabled on the device.
    ///
    /// The default value is `1`.
    pub array_layers: u32,

    /// The number of mip levels to create the image with.
    ///
    /// The default value is `1`.
    pub mip_levels: u32,

    /// The number of samples per texel that the image should use.
    ///
    /// On [portability
    /// subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `samples` is not [`SampleCount::Sample1`] and `array_layers` is not 1,
    /// the [`multisample_array_image`](crate::device::DeviceFeatures::multisample_array_image)
    /// feature must be enabled on the device.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub samples: SampleCount,

    /// The memory arrangement of the texel blocks.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    /// How the image is going to be used.
    ///
    /// The default value is empty, which must be overridden.
    pub usage: ImageUsage,

    /// How the stencil aspect of the image is going to be used, if different from the regular
    /// `usage`.
    ///
    /// If this is `Some`, then the device API version must be at least 1.2, or the
    /// [`ext_separate_stencil_usage`](crate::device::DeviceExtensions::ext_separate_stencil_usage)
    /// extension must be enabled on the device. `format` must a stencil aspect.
    ///
    /// The default value is `None`.
    pub stencil_usage: Option<ImageUsage>,

    /// Whether the image can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<'a>,

    /// The image layout that the image will have when it is created.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub initial_layout: ImageLayout,

    /// A list of possible Linux DRM format modifiers that the image may be created with. If
    /// `tiling` is [`ImageTiling::DrmFormatModifier`], then at least one DRM format modifier must
    /// be provided. Otherwise, this must be empty.
    ///
    /// If more than one DRM format modifier is provided, then the Vulkan driver will choose the
    /// modifier in an implementation-defined manner. You can query the modifier that has been
    /// chosen, after creating the image, by calling [`Image::drm_format_modifier`].
    ///
    /// If exactly one DRM format modifier is provided, the image will always be created with that
    /// modifier. You can then optionally specify the subresource layout of each memory plane with
    /// the `drm_format_modifier_plane_layouts` field.
    ///
    /// The default value is empty.
    pub drm_format_modifiers: &'a [u64],

    /// If `drm_format_modifiers` contains exactly one element, optionally specifies an explicit
    /// subresource layout for each memory plane of the image.
    ///
    /// If not empty, the number of provided subresource layouts must equal the number of memory
    /// planes for `drm_format_modifiers[0]`, as reported by
    /// [`DrmFormatModifierProperties::drm_format_modifier_plane_count`]. The following additional
    /// requirements apply to each element:
    /// - [`SubresourceLayout::size`] must always be 0.
    /// - If `array_layers` is 1, then [`SubresourceLayout::array_pitch`] must be `None`.
    /// - If `image_type` is not [`ImageType::Dim3d`] or `extent[2]` is 1, then
    ///   [`SubresourceLayout::depth_pitch`] must be `None`.
    ///
    /// If `drm_format_modifiers` does not contain exactly one element, then this must be empty.
    ///
    /// The default value is empty.
    pub drm_format_modifier_plane_layouts: &'a [SubresourceLayout],

    /// The external memory handle types that are going to be used with the image.
    ///
    /// If this is not empty, then the device API version must be at least 1.1, or the
    /// [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
    /// extension must be enabled on the device. `initial_layout` must be set to
    /// [`ImageLayout::Undefined`].
    ///
    /// The default value is empty.
    pub external_memory_handle_types: ExternalMemoryHandleTypes,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ImageCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ImageCreateInfo<'a> {
    /// Returns a default `ImageCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim2d,
            format: Format::UNDEFINED,
            view_formats: &[],
            extent: [0; 3],
            array_layers: 1,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            stencil_usage: None,
            sharing: Sharing::Exclusive,
            initial_layout: ImageLayout::Undefined,
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            drm_format_modifiers: &[],
            drm_format_modifier_plane_layouts: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            image_type,
            format,
            view_formats,
            extent,
            array_layers,
            mip_levels,
            samples,
            tiling,
            usage,
            stencil_usage,
            sharing,
            initial_layout,
            drm_format_modifiers,
            drm_format_modifier_plane_layouts,
            external_memory_handle_types,
            _ne: _,
        } = self;

        let physical_device = device.physical_device();
        let device_properties = physical_device.properties();

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImageCreateInfo-flags-parameter"])
        })?;

        format.validate_device(device).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-VkImageCreateInfo-format-parameter"])
        })?;

        samples.validate_device(device).map_err(|err| {
            err.add_context("samples")
                .set_vuids(&["VUID-VkImageCreateInfo-samples-parameter"])
        })?;

        tiling.validate_device(device).map_err(|err| {
            err.add_context("tiling")
                .set_vuids(&["VUID-VkImageCreateInfo-tiling-parameter"])
        })?;

        usage.validate_device(device).map_err(|err| {
            err.add_context("usage")
                .set_vuids(&["VUID-VkImageCreateInfo-usage-parameter"])
        })?;

        if usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageCreateInfo-usage-requiredbitmask"],
                ..Default::default()
            }));
        }

        if format == Format::UNDEFINED {
            return Err(Box::new(ValidationError {
                context: "format".into(),
                problem: "is `Format::UNDEFINED`".into(),
                vuids: &["VUID-VkImageCreateInfo-pNext-01975"],
                ..Default::default()
            }));
        }

        let format_properties = unsafe { physical_device.format_properties_unchecked(format) };
        let image_create_drm_format_modifiers = &drm_format_modifiers;
        let image_create_maybe_linear = match tiling {
            ImageTiling::Linear => true,
            ImageTiling::Optimal => false,
            ImageTiling::DrmFormatModifier => {
                const DRM_FORMAT_MOD_LINEAR: u64 = 0;
                image_create_drm_format_modifiers.contains(&DRM_FORMAT_MOD_LINEAR)
            }
        };
        let image_create_format_features =
            format_properties.format_features(tiling, drm_format_modifiers);

        initial_layout.validate_device(device).map_err(|err| {
            err.add_context("initial_layout")
                .set_vuids(&["VUID-VkImageCreateInfo-initialLayout-parameter"])
        })?;

        if !matches!(
            initial_layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized
        ) {
            return Err(Box::new(ValidationError {
                context: "initial_layout".into(),
                problem: "is not `ImageLayout::Undefined` or `ImageLayout::Preinitialized`".into(),
                vuids: &["VUID-VkImageCreateInfo-initialLayout-00993"],
                ..Default::default()
            }));
        }

        if flags.intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
            && !flags.intersects(ImageCreateFlags::MUTABLE_FORMAT)
        {
            return Err(Box::new(ValidationError {
                context: "flags".into(),
                problem: "contains `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, but does not \
                    contain `ImageCreateFlags::MUTABLE_FORMAT`"
                    .into(),
                vuids: &["VUID-VkImageCreateInfo-flags-01573"],
                ..Default::default()
            }));
        }

        if flags.intersects(ImageCreateFlags::PROTECTED) {
            if !device.enabled_features().protected_memory {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `ImageCreateFlags::PROTECTED`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "protected_memory",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-flags-01890"],
                }));
            }

            // If VK_IMAGE_CREATE_SPARSE_ALIASED_BIT support is added, then it should be added to
            // this validation.
            if flags
                .intersects(ImageCreateFlags::SPARSE_BINDING | ImageCreateFlags::SPARSE_RESIDENCY)
            {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `BufferCreateFlags::PROTECTED`, but also \
                        contains one of the sparse memory flags"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-None-01891"],
                    ..Default::default()
                }));
            }
        }

        if !view_formats.is_empty() {
            if !(device.api_version() >= Version::V1_2
                || device.enabled_extensions().khr_image_format_list)
            {
                return Err(Box::new(ValidationError {
                    context: "view_formats".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_image_format_list")]),
                    ]),
                    ..Default::default()
                }));
            }

            if !flags.intersects(ImageCreateFlags::MUTABLE_FORMAT) && view_formats.len() != 1 {
                return Err(Box::new(ValidationError {
                    problem: "`flags` does not contain `ImageCreateFlags::MUTABLE_FORMAT`, but \
                        `view_formats` contains more than one element"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-04738"],
                    ..Default::default()
                }));
            }

            for (index, &view_format) in view_formats.iter().enumerate() {
                view_format.validate_device(device).map_err(|err| {
                    err.add_context(format!("view_formats[{}]", index))
                        .set_vuids(&["VUID-VkImageFormatListCreateInfo-pViewFormats-parameter"])
                })?;

                if flags.intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
                    && view_format.compression().is_none()
                {
                    if !(view_format.compatibility() == format.compatibility()
                        || view_format.block_size() == format.block_size())
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`flags` contains \
                                `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, and \
                                `view_formats[{}]` is an uncompressed format, but \
                                it is not compatible with `format`, and \
                                does not have an equal block size",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkImageCreateInfo-pNext-06722"],
                            ..Default::default()
                        }));
                    }
                } else {
                    if view_format.compatibility() != format.compatibility() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`flags` does not contain \
                                `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, or \
                                `view_format[{}]` is a compressed format, but \
                                it is not compatible with `create_info.format`",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkImageCreateInfo-pNext-06722"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        match image_type {
            ImageType::Dim1d => {
                if extent[1] != 1 {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim1d`, but `extent[1]` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00956"],
                        ..Default::default()
                    }));
                }

                if extent[2] != 1 {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim1d`, but `extent[2]` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00956"],
                        ..Default::default()
                    }));
                }

                if flags.intersects(ImageCreateFlags::SPARSE_RESIDENCY) {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim1d`, but `flags` contains \
                            `ImageCreateFlags::SPARSE_RESIDENCY`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00970"],
                        ..Default::default()
                    }));
                }
            }
            ImageType::Dim2d => {
                if extent[2] != 1 {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim2d`, but `extent[2]` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00957"],
                        ..Default::default()
                    }));
                }

                if flags.intersects(ImageCreateFlags::SPARSE_RESIDENCY)
                    && !device.enabled_features().sparse_residency_image2_d
                {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim2d`, and `flags` contains \
                            `ImageCreateFlags::SPARSE_RESIDENCY`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("sparse_residency_image2_d"),
                        ])]),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00971"],
                        ..Default::default()
                    }));
                }
            }
            ImageType::Dim3d => {
                if array_layers != 1 {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim3d`, but `array_layers` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00961"],
                        ..Default::default()
                    }));
                }

                if flags.intersects(ImageCreateFlags::SPARSE_RESIDENCY)
                    && !device.enabled_features().sparse_residency_image3_d
                {
                    return Err(Box::new(ValidationError {
                        problem: "`image_type` is `ImageType::Dim3d`, and `flags` contains \
                            `ImageCreateFlags::SPARSE_RESIDENCY`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("sparse_residency_image3_d"),
                        ])]),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00972"],
                        ..Default::default()
                    }));
                }
            }
        }

        if extent[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-extent-00944"],
                ..Default::default()
            }));
        }

        if extent[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-extent-00945"],
                ..Default::default()
            }));
        }

        if extent[2] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[2]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-extent-00946"],
                ..Default::default()
            }));
        }

        if array_layers == 0 {
            return Err(Box::new(ValidationError {
                context: "array_layers".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-arrayLayers-00948"],
                ..Default::default()
            }));
        }

        if mip_levels == 0 {
            return Err(Box::new(ValidationError {
                context: "mip_levels".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-mipLevels-00947"],
                ..Default::default()
            }));
        }

        let max_mip_levels = max_mip_levels(extent);
        debug_assert!(max_mip_levels >= 1);

        if mip_levels > max_mip_levels {
            return Err(Box::new(ValidationError {
                problem: "`mip_levels` is greater than the maximum allowed number of mip levels \
                    for `dimensions`"
                    .into(),
                vuids: &["VUID-VkImageCreateInfo-mipLevels-00958"],
                ..Default::default()
            }));
        }

        if samples != SampleCount::Sample1 {
            if image_type != ImageType::Dim2d {
                return Err(Box::new(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        `image_type` is not `ImageType::Dim2d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                }));
            }

            if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) {
                return Err(Box::new(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        `flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                }));
            }

            if mip_levels != 1 {
                return Err(Box::new(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        `mip_levels` is not 1"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                }));
            }

            if image_create_maybe_linear {
                return Err(Box::new(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        the image may have linear tiling"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                }));
            }

            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().multisample_array_image
                && array_layers != 1
            {
                return Err(Box::new(ValidationError {
                    problem: "this device is a portability subset device, \
                        `samples` is not `SampleCount::Sample1`, and \
                        `array_layers` is greater than 1"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "multisample_array_image",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-multisampleArrayImage-04460"],
                    ..Default::default()
                }));
            }
        }

        // Check limits for YCbCr formats
        if let Some(chroma_sampling) = format.ycbcr_chroma_sampling() {
            if mip_levels != 1 {
                return Err(Box::new(ValidationError {
                    problem: "`format` is a YCbCr format, but \
                        `mip_levels` is not 1"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-06410"],
                    ..Default::default()
                }));
            }

            if samples != SampleCount::Sample1 {
                return Err(Box::new(ValidationError {
                    problem: "`format` is a YCbCr format, but \
                        `samples` is not `SampleCount::Sample1`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-06411"],
                    ..Default::default()
                }));
            }

            if image_type != ImageType::Dim2d {
                return Err(Box::new(ValidationError {
                    problem: "`format` is a YCbCr format, but \
                        `image_type` is not `ImageType::Dim2d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-06412"],
                    ..Default::default()
                }));
            }

            if array_layers > 1 && !device.enabled_features().ycbcr_image_arrays {
                return Err(Box::new(ValidationError {
                    problem: "`format` is is a YCbCr format, and \
                        `array_layers` is greater than 1"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ycbcr_image_arrays",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-format-06413"],
                    ..Default::default()
                }));
            }

            match chroma_sampling {
                ChromaSampling::Mode444 => (),
                ChromaSampling::Mode422 => {
                    if extent[0] % 2 != 0 {
                        return Err(Box::new(ValidationError {
                            problem: "`format` is a YCbCr format with horizontal \
                                chroma subsampling, but \
                                `extent[0]` is not \
                                a multiple of 2"
                                .into(),
                            vuids: &["VUID-VkImageCreateInfo-format-04712"],
                            ..Default::default()
                        }));
                    }
                }
                ChromaSampling::Mode420 => {
                    if !(extent[0] % 2 == 0 && extent[1] % 2 == 0) {
                        return Err(Box::new(ValidationError {
                            problem: "`format` is a YCbCr format with horizontal and vertical \
                                chroma subsampling, but \
                                `extent[0]` and `extent[1]` are not both \
                                a multiple of 2"
                                .into(),
                            vuids: &[
                                "VUID-VkImageCreateInfo-format-04712",
                                "VUID-VkImageCreateInfo-format-04713",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        /* Check usage requirements */

        if usage.intersects(ImageUsage::STORAGE)
            && samples != SampleCount::Sample1
            && !device.enabled_features().shader_storage_image_multisample
        {
            return Err(Box::new(ValidationError {
                problem: "`usage` contains `ImageUsage::STORAGE`, but \
                    `samples` is not `SampleCount::Sample1`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "shader_storage_image_multisample",
                )])]),
                vuids: &["VUID-VkImageCreateInfo-usage-00968"],
                ..Default::default()
            }));
        }

        if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT) {
            if !usage.intersects(
                ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
            ) {
                return Err(Box::new(ValidationError {
                    context: "usage".into(),
                    problem: "contains `ImageUsage::TRANSIENT_ATTACHMENT`, but does not also \
                        contain one of `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-usage-00966"],
                    ..Default::default()
                }));
            }

            if !(usage
                - (ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT))
                .is_empty()
            {
                return Err(Box::new(ValidationError {
                    context: "usage".into(),
                    problem: "contains `ImageUsage::TRANSIENT_ATTACHMENT`, but also contains \
                        usages other than `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-usage-00963"],
                    ..Default::default()
                }));
            }
        }

        if usage.intersects(
            ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                | ImageUsage::INPUT_ATTACHMENT
                | ImageUsage::TRANSIENT_ATTACHMENT,
        ) {
            if extent[0] > device_properties.max_framebuffer_width {
                return Err(Box::new(ValidationError {
                    problem: "`usage` contains \
                        `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, \
                        `ImageUsage::INPUT_ATTACHMENT`, or \
                        `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        `extent[0]` exceeds the `max_framebuffer_width` limit"
                        .into(),
                    vuids: &[
                        "VUID-VkImageCreateInfo-usage-00964",
                        "VUID-VkImageCreateInfo-Format-02536",
                    ],
                    ..Default::default()
                }));
            }

            if extent[1] > device_properties.max_framebuffer_height {
                return Err(Box::new(ValidationError {
                    problem: "`usage` contains \
                        `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, \
                        `ImageUsage::INPUT_ATTACHMENT`, or \
                        `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        `extent[1]` exceeds the `max_framebuffer_height` limit"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-usage-00965"],
                    ..Default::default()
                }));
            }
        }

        if let Some(stencil_usage) = stencil_usage {
            if !(device.api_version() >= Version::V1_2
                || device.enabled_extensions().ext_separate_stencil_usage)
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("ext_separate_stencil_usage")]),
                    ]),
                    ..Default::default()
                }));
            }

            stencil_usage.validate_device(device).map_err(|err| {
                err.add_context("stencil_usage")
                    .set_vuids(&["VUID-VkImageStencilUsageCreateInfo-stencilUsage-parameter"])
            })?;

            if stencil_usage.is_empty() {
                return Err(Box::new(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "is empty".into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-usage-requiredbitmask"],
                    ..Default::default()
                }));
            }

            if stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                && !(stencil_usage
                    - (ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT))
                    .is_empty()
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "contains `ImageUsage::TRANSIENT_ATTACHMENT`, but also contains \
                        usages other than `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or \
                        `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-stencilUsage-02539"],
                    ..Default::default()
                }));
            }

            if format
                .aspects()
                .intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
            {
                if usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                    && !stencil_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`format` is a depth/stencil format, and \
                            `usage` contains `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, but \
                            `stencil_usage` does not also contain \
                            `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-format-02795"],
                        ..Default::default()
                    }));
                }

                if !usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                    && stencil_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`format` is a depth/stencil format, and \
                            `usage` does not contain `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, but \
                            `stencil_usage` does contain `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-format-02796"],
                        ..Default::default()
                    }));
                }

                if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                    && !stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`format` is a depth/stencil format, and \
                            `usage` contains `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                            `stencil_usage` does not also contain \
                            `ImageUsage::TRANSIENT_ATTACHMENT`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-format-02797"],
                        ..Default::default()
                    }));
                }

                if !usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                    && stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`format` is a depth/stencil format, and \
                            `usage` does not contain `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                            `stencil_usage` does contain \
                            `ImageUsage::TRANSIENT_ATTACHMENT`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-format-02798"],
                        ..Default::default()
                    }));
                }

                if stencil_usage.intersects(ImageUsage::INPUT_ATTACHMENT) {
                    if extent[0] > device_properties.max_framebuffer_width {
                        return Err(Box::new(ValidationError {
                            problem: "`stencil_usage` contains \
                                `ImageUsage::INPUT_ATTACHMENT`, but \
                                `extent[0]` exceeds the `max_framebuffer_width` limit"
                                .into(),
                            vuids: &["VUID-VkImageCreateInfo-Format-02536"],
                            ..Default::default()
                        }));
                    }

                    if extent[1] > device_properties.max_framebuffer_height {
                        return Err(Box::new(ValidationError {
                            problem: "`stencil_usage` contains \
                                `ImageUsage::INPUT_ATTACHMENT`, but \
                                `extent[1]` exceeds the `max_framebuffer_height` limit"
                                .into(),
                            vuids: &["VUID-VkImageCreateInfo-format-02537"],
                            ..Default::default()
                        }));
                    }
                }

                if stencil_usage.intersects(ImageUsage::STORAGE)
                    && samples != SampleCount::Sample1
                    && !device.enabled_features().shader_storage_image_multisample
                {
                    return Err(Box::new(ValidationError {
                        problem: "`stencil_usage` contains `ImageUsage::STORAGE`, but \
                            `samples` is not `SampleCount::Sample1`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("shader_storage_image_multisample"),
                        ])]),
                        vuids: &["VUID-VkImageCreateInfo-format-02538"],
                        ..Default::default()
                    }));
                }
            }
        }

        /* Check flags requirements */

        if flags.intersects(ImageCreateFlags::SPARSE_BINDING) {
            if !device.enabled_features().sparse_binding {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `ImageCreateFlags::SPARSE_BINDING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "sparse_binding",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-flags-00969"],
                }));
            }

            if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::SPARSE_BINDING`, but \
                        `usage` contains `ImageUsage::TRANSIENT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-None-01925"],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(ImageCreateFlags::SPARSE_RESIDENCY) {
            if !flags.intersects(ImageCreateFlags::SPARSE_BINDING) {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `ImageCreateFlags::SPARSE_RESIDENCY`, but does not also \
                        contain `ImageCreateFlags::SPARSE_BINDING`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-00987"],
                    ..Default::default()
                }));
            }

            if tiling == ImageTiling::Linear {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `ImageCreateFlags::SPARSE_RESIDENCY`, but `tiling` is \
                        `ImageTiling::Linear`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-tiling-04121"],
                    ..Default::default()
                }));
            }

            match samples {
                SampleCount::Sample2 => {
                    if !device.enabled_features().sparse_residency2_samples {
                        return Err(Box::new(ValidationError {
                            problem: "`flags` contains `ImageCreateFlags::SPARSE_RESIDENCY`, and \
                                `samples` is `SampleCount::Sample2`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("sparse_residency2_samples"),
                            ])]),
                            vuids: &["VUID-VkImageCreateInfo-imageType-00973"],
                            ..Default::default()
                        }));
                    }
                }
                SampleCount::Sample4 => {
                    if !device.enabled_features().sparse_residency4_samples {
                        return Err(Box::new(ValidationError {
                            problem: "`flags` contains `ImageCreateFlags::SPARSE_RESIDENCY`, and \
                                `samples` is `SampleCount::Sample4`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("sparse_residency4_samples"),
                            ])]),
                            vuids: &["VUID-VkImageCreateInfo-imageType-00974"],
                            ..Default::default()
                        }));
                    }
                }
                SampleCount::Sample8 => {
                    if !device.enabled_features().sparse_residency8_samples {
                        return Err(Box::new(ValidationError {
                            problem: "`flags` contains `ImageCreateFlags::SPARSE_RESIDENCY`, and \
                                `samples` is `SampleCount::Sample8`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("sparse_residency8_samples"),
                            ])]),
                            vuids: &["VUID-VkImageCreateInfo-imageType-00975"],
                            ..Default::default()
                        }));
                    }
                }
                SampleCount::Sample16 => {
                    if !device.enabled_features().sparse_residency16_samples {
                        return Err(Box::new(ValidationError {
                            problem: "`flags` contains `ImageCreateFlags::SPARSE_RESIDENCY`, and \
                                `samples` is `SampleCount::Sample16`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("sparse_residency16_samples"),
                            ])]),
                            vuids: &["VUID-VkImageCreateInfo-imageType-00976"],
                            ..Default::default()
                        }));
                    }
                }
                SampleCount::Sample1 | SampleCount::Sample32 | SampleCount::Sample64 => (),
            }
        }

        if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) {
            if image_type != ImageType::Dim2d {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`, but \
                        `image_type` is not `ImageType::Dim2d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-00949"],
                    ..Default::default()
                }));
            }

            if extent[0] != extent[1] {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`, but \
                        `extent[0]` does not equal `extent[1]`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-imageType-00954"],
                    ..Default::default()
                }));
            }

            if array_layers < 6 {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`, but \
                        `array_layers` is less than 6"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-imageType-00954"],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE) {
            if flags.intersects(ImageCreateFlags::SPARSE_BINDING) {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE`, but \
                        also contains `ImageCreateFlags::SPARSE_BINDING`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-09403"],
                    ..Default::default()
                }));
            }

            if image_type != ImageType::Dim3d {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE`, but \
                        `image_type` is not `ImageType::Dim3d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-00950"],
                    ..Default::default()
                }));
            }

            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().image_view2_d_on3_d_image
            {
                return Err(Box::new(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `flags` contains `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "image_view2_d_on3_d_image",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-imageView2DOn3DImage-04459"],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(ImageCreateFlags::DIM2D_VIEW_COMPATIBLE) {
            if image_type != ImageType::Dim3d {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DIM2D_VIEW_COMPATIBLE`, but \
                        `image_type` is not `ImageType::Dim3d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-07755"],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
            && format.compression().is_none()
        {
            return Err(Box::new(ValidationError {
                problem: "`flags` contains `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, \
                    but `format` is not a compressed format"
                    .into(),
                vuids: &["VUID-VkImageCreateInfo-flags-01572"],
                ..Default::default()
            }));
        }

        if flags.intersects(ImageCreateFlags::DISJOINT) {
            if format.planes().len() < 2 {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DISJOINT`, but `format` \
                        is not a multi-planar format"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-01577"],
                    ..Default::default()
                }));
            }

            if !image_create_format_features.intersects(FormatFeatures::DISJOINT) {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DISJOINT`, but the \
                        format features of `format` do not include \
                        `FormatFeatures::DISJOINT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-imageCreateFormatFeatures-02260"],
                    ..Default::default()
                }));
            }
        }

        /* Check sharing mode and queue families */

        match sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                if queue_family_indices.len() < 2 {
                    return Err(Box::new(ValidationError {
                        context: "sharing".into(),
                        problem: "is `Sharing::Concurrent`, but contains less than 2 elements"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-sharingMode-00942"],
                        ..Default::default()
                    }));
                }

                let queue_family_count = physical_device.queue_family_properties().len() as u32;

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
                            vuids: &["VUID-VkImageCreateInfo-sharingMode-01420"],
                            ..Default::default()
                        }));
                    }

                    if queue_family_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: format!("sharing[{}]", index).into(),
                            problem: "is not less than the number of queue families in the device"
                                .into(),
                            vuids: &["VUID-VkImageCreateInfo-sharingMode-01420"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if !(drm_format_modifier_plane_layouts.is_empty() || drm_format_modifiers.len() == 1) {
            return Err(Box::new(ValidationError {
                problem: "`drm_format_modifier_plane_layouts` is not empty, but \
                    `drm_format_modifiers` does not contain exactly one element"
                    .into(),
                ..Default::default()
            }));
        }

        match (tiling, !drm_format_modifiers.is_empty()) {
            (ImageTiling::DrmFormatModifier, true) => {
                if flags.intersects(ImageCreateFlags::MUTABLE_FORMAT) && view_formats.is_empty() {
                    return Err(Box::new(ValidationError {
                        problem: "`tiling` is `ImageTiling::DrmFormatModifier`, and \
                            `flags` contains `ImageCreateFlags::MUTABLE_FORMAT`, but \
                            `view_formats` is empty"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-tiling-02353"],
                        ..Default::default()
                    }));
                }

                if !drm_format_modifier_plane_layouts.is_empty() {
                    let drm_format_modifier = drm_format_modifiers[0];
                    let drm_format_modifier_properties = format_properties
                        .drm_format_modifier_properties
                        .iter()
                        .find(|properties| properties.drm_format_modifier == drm_format_modifier)
                        .ok_or_else(|| Box::new(ValidationError {
                            problem: "`drm_format_modifier_plane_layouts` is not empty, but \
                                `drm_format_modifiers[0]` is not one of the modifiers in \
                                `FormatProperties::drm_format_properties`, as returned by \
                                `PhysicalDevice::format_properties` for `format`".into(),
                            vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-drmFormatModifierPlaneCount-02265"],
                            ..Default::default()
                        }))?;

                    if drm_format_modifier_plane_layouts.len()
                        != drm_format_modifier_properties.drm_format_modifier_plane_count as usize
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`drm_format_modifier_plane_layouts` is not empty, but the \
                                number of provided subresource layouts does not equal the number \
                                of memory planes for `drm_format_modifiers[0]`, specified in \
                                `DrmFormatModifierProperties::drm_format_modifier_plane_count`, \
                                as returned by `PhysicalDevice::format_properties` for `format`"
                                .into(),
                            vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-drmFormatModifierPlaneCount-02265"],
                            ..Default::default()
                        }));
                    }

                    for (index, subresource_layout) in
                        drm_format_modifier_plane_layouts.iter().enumerate()
                    {
                        let &SubresourceLayout {
                            offset: _,
                            size,
                            row_pitch: _,
                            array_pitch,
                            depth_pitch,
                        } = subresource_layout;

                        if size != 0 {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "drm_format_modifier_plane_layouts[{}].size",
                                    index
                                )
                                .into(),
                                problem: "is not zero".into(),
                                vuids: &[
                                    "VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-size-02267",
                                ],
                                ..Default::default()
                            }));
                        }

                        if array_layers == 1 && array_pitch.is_some() {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`array_layers` is 1, but \
                                    `drm_format_modifier_plane_layouts[{}].array_pitch` is `Some`",
                                    index
                                )
                                .into(),
                                vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-arrayPitch-02268"],
                                ..Default::default()
                            }));
                        }

                        if extent[2] == 1 && depth_pitch.is_some() {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`extent[2]` is 1, but \
                                    `drm_format_modifier_plane_layouts[{}].depth_pitch` is `Some`",
                                    index
                                )
                                .into(),
                                vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-depthPitch-02269"],
                                ..Default::default()
                            }));
                        }
                    }
                }
            }
            (ImageTiling::DrmFormatModifier, false) => {
                return Err(Box::new(ValidationError {
                    problem: "`tiling` is `ImageTiling::DrmFormatModifier`, but \
                        `drm_format_modifiers` is empty"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-tiling-02261"],
                    ..Default::default()
                }));
            }
            (_, true) => {
                if tiling != ImageTiling::DrmFormatModifier {
                    return Err(Box::new(ValidationError {
                        problem: "`drm_format_modifiers` is not empty, but \
                            `tiling` is not `ImageTiling::DrmFormatModifier`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-pNext-02262"],
                        ..Default::default()
                    }));
                }
            }
            (_, false) => (),
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
                        .set_vuids(&["VUID-VkExternalMemoryImageCreateInfo-handleTypes-parameter"])
                })?;

            if initial_layout != ImageLayout::Undefined {
                return Err(Box::new(ValidationError {
                    problem: "`external_memory_handle_types` is not empty, but \
                        `initial_layout` is not `ImageLayout::Undefined`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-pNext-01443"],
                    ..Default::default()
                }));
            }
        }

        /*
            Some device limits can be exceeded, but only for particular image configurations, which
            must be queried with `image_format_properties`. See:
            https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap48.html#capabilities-image
            First, we check if this is the case, then query the device if so.
        */

        fn iter_or_none<T>(iter: impl IntoIterator<Item = T>) -> impl Iterator<Item = Option<T>> {
            let mut iter = iter.into_iter();
            [iter.next()].into_iter().chain(iter.map(Some))
        }

        for drm_format_modifier in iter_or_none(drm_format_modifiers.iter().copied()) {
            for external_memory_handle_type in iter_or_none(external_memory_handle_types) {
                let image_format_properties = unsafe {
                    physical_device.image_format_properties_unchecked(&ImageFormatInfo {
                        flags,
                        format,
                        image_type,
                        tiling,
                        usage,
                        stencil_usage,
                        external_memory_handle_type,
                        drm_format_modifier_info: drm_format_modifier
                            .map(|drm_format_modifier| ImageDrmFormatModifierInfo {
                                drm_format_modifier,
                                sharing,
                                ..Default::default()
                            })
                            .as_ref(),
                        ..Default::default()
                    })
                }
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::image_format_properties` \
                                    returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?;

                let image_format_properties = image_format_properties.ok_or_else(|| Box::new(ValidationError {
                    problem: "the combination of parameters of this image is not \
                        supported by the physical device, as returned by \
                        `PhysicalDevice::image_format_properties`"
                        .into(),
                    vuids: &[
                        "VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-drmFormatModifier-02264",
                        "VUID-VkImageDrmFormatModifierListCreateInfoEXT-pDrmFormatModifiers-02263",
                    ],
                    ..Default::default()
                }))?;

                let ImageFormatProperties {
                    max_extent,
                    max_mip_levels,
                    max_array_layers,
                    sample_counts,
                    max_resource_size: _,
                    external_memory_properties,
                    filter_cubic: _,
                    filter_cubic_minmax: _,
                } = image_format_properties;

                if extent[0] > max_extent[0] {
                    return Err(Box::new(ValidationError {
                        problem: "`extent[0]` exceeds `max_extent[0]` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-extent-02252"],
                        ..Default::default()
                    }));
                }

                if extent[1] > max_extent[1] {
                    return Err(Box::new(ValidationError {
                        problem: "`extent[1]` exceeds `max_extent[1]` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-extent-02253"],
                        ..Default::default()
                    }));
                }

                if extent[2] > max_extent[2] {
                    return Err(Box::new(ValidationError {
                        problem: "`extent[2]` exceeds `max_extent[2]` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-extent-02254"],
                        ..Default::default()
                    }));
                }

                if mip_levels > max_mip_levels {
                    return Err(Box::new(ValidationError {
                        problem: "`mip_levels` exceeds `max_mip_levels` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-mipLevels-02255"],
                        ..Default::default()
                    }));
                }

                if array_layers > max_array_layers {
                    return Err(Box::new(ValidationError {
                        problem: "`array_layers` exceeds `max_array_layers` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-arrayLayers-02256"],
                        ..Default::default()
                    }));
                }

                if !sample_counts.contains_enum(samples) {
                    return Err(Box::new(ValidationError {
                        problem: "`samples` is not present in the `sample_counts` value for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-samples-02258"],
                        ..Default::default()
                    }));
                }

                if !external_memory_properties
                    .compatible_handle_types
                    .contains(external_memory_handle_types)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`external_memory_handle_types` is not a subset of the \
                            `external_memory_properties.compatible_handle_types` value for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-pNext-00990"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &'a self,
        extensions_vk: &'a mut ImageCreateInfoExtensionsVk<'_>,
    ) -> vk::ImageCreateInfo<'a> {
        let &Self {
            flags,
            image_type,
            format,
            view_formats: _,
            extent,
            array_layers,
            mip_levels,
            samples,
            tiling,
            usage,
            stencil_usage: _,
            sharing,
            initial_layout,
            drm_format_modifiers: _,
            drm_format_modifier_plane_layouts: _,
            external_memory_handle_types: _,
            _ne: _,
        } = self;

        let (sharing_mode, queue_family_indices_vk) = match sharing {
            Sharing::Exclusive => (vk::SharingMode::EXCLUSIVE, [].as_slice()),
            Sharing::Concurrent(queue_family_indices) => {
                (vk::SharingMode::CONCURRENT, queue_family_indices)
            }
        };

        let mut val_vk = vk::ImageCreateInfo::default()
            .flags(flags.into())
            .image_type(image_type.into())
            .format(format.into())
            .extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            })
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(samples.into())
            .tiling(tiling.into())
            .usage(usage.into())
            .sharing_mode(sharing_mode)
            .queue_family_indices(queue_family_indices_vk)
            .initial_layout(initial_layout.into());

        let ImageCreateInfoExtensionsVk {
            drm_format_modifier_explicit_vk,
            drm_format_modifier_list_vk,
            external_memory_vk,
            format_list_vk,
            stencil_usage_vk,
        } = extensions_vk;

        if let Some(next) = drm_format_modifier_explicit_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = drm_format_modifier_list_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = external_memory_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = format_list_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = stencil_usage_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(
        &'a self,
        fields1_vk: &'a ImageCreateInfoFields1Vk,
    ) -> ImageCreateInfoExtensionsVk<'a> {
        let ImageCreateInfoFields1Vk {
            plane_layouts_vk,
            view_formats_vk,
        } = fields1_vk;

        let drm_format_modifier_explicit_vk = (!plane_layouts_vk.is_empty()).then(|| {
            vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(self.drm_format_modifiers[0])
                .plane_layouts(plane_layouts_vk)
        });

        let drm_format_modifier_list_vk =
            (plane_layouts_vk.is_empty() && !self.drm_format_modifiers.is_empty()).then(|| {
                vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                    .drm_format_modifiers(self.drm_format_modifiers)
            });

        let external_memory_vk = (!self.external_memory_handle_types.is_empty()).then(|| {
            vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(self.external_memory_handle_types.into())
        });

        let format_list_vk = (!view_formats_vk.is_empty())
            .then(|| vk::ImageFormatListCreateInfo::default().view_formats(view_formats_vk));

        let stencil_usage_vk = self.stencil_usage.map(|stencil_usage| {
            vk::ImageStencilUsageCreateInfo::default().stencil_usage(stencil_usage.into())
        });

        ImageCreateInfoExtensionsVk {
            drm_format_modifier_explicit_vk,
            drm_format_modifier_list_vk,
            external_memory_vk,
            format_list_vk,
            stencil_usage_vk,
        }
    }

    pub(crate) fn to_vk_fields1(&self) -> ImageCreateInfoFields1Vk {
        let plane_layouts_vk = self
            .drm_format_modifier_plane_layouts
            .iter()
            .map(SubresourceLayout::to_vk)
            .collect();

        let view_formats_vk = self
            .view_formats
            .iter()
            .copied()
            .map(vk::Format::from)
            .collect();

        ImageCreateInfoFields1Vk {
            plane_layouts_vk,
            view_formats_vk,
        }
    }
}

pub(crate) struct ImageCreateInfoExtensionsVk<'a> {
    pub(crate) drm_format_modifier_explicit_vk:
        Option<vk::ImageDrmFormatModifierExplicitCreateInfoEXT<'a>>,
    pub(crate) drm_format_modifier_list_vk: Option<vk::ImageDrmFormatModifierListCreateInfoEXT<'a>>,
    pub(crate) external_memory_vk: Option<vk::ExternalMemoryImageCreateInfo<'static>>,
    pub(crate) format_list_vk: Option<vk::ImageFormatListCreateInfo<'a>>,
    pub(crate) stencil_usage_vk: Option<vk::ImageStencilUsageCreateInfo<'static>>,
}

pub(crate) struct ImageCreateInfoFields1Vk {
    plane_layouts_vk: SmallVec<[vk::SubresourceLayout; 4]>,
    view_formats_vk: Vec<vk::Format>,
}

#[cfg(test)]
mod tests {
    use super::{ImageCreateInfo, ImageUsage, RawImage};
    use crate::{
        format::Format,
        image::{
            ImageAspect, ImageAspects, ImageCreateFlags, ImageSubresourceRange, ImageType,
            SampleCount, SubresourceRangeIterator,
        },
        memory::{DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory},
        DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError,
    };
    use smallvec::SmallVec;
    use std::iter;

    #[test]
    fn create_sampled() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = RawImage::new(
            &device,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = RawImage::new(
            &device,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_protected() {
        let (device, _) = gfx_dev_and_queue!(protected_memory);

        let image = RawImage::new(
            &device,
            &ImageCreateInfo {
                flags: ImageCreateFlags::PROTECTED,
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(image.flags(), ImageCreateFlags::PROTECTED);

        let requirements = image.memory_requirements()[0];

        let memory_type_index = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, ty)| {
                (requirements.memory_type_bits & (1 << id)) != 0
                    && ty.property_flags.contains(MemoryPropertyFlags::PROTECTED)
            })
            .expect("Failed to find suitable memory type") as u32;

        let memory = DeviceMemory::allocate(
            &device,
            &MemoryAllocateInfo {
                allocation_size: requirements.layout.size(),
                memory_type_index,
                ..Default::default()
            },
        )
        .expect("Failed to allocate memory");

        let resource_memory = ResourceMemory::new_dedicated(memory);

        let result = image.bind_memory(iter::once(resource_memory));
        assert!(result.is_ok());
    }

    #[test]
    fn create_protected_and_sparse_is_disallowed() {
        let (device, _) = gfx_dev_and_queue!(protected_memory, sparse_binding);

        match RawImage::try_new(
            &device,
            &ImageCreateInfo {
                flags: (ImageCreateFlags::PROTECTED | ImageCreateFlags::SPARSE_BINDING),
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                usage: ImageUsage::SAMPLED,
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
                panic!("RawImage::try_new succeeded when it should have failed!");
            }
        }
    }

    #[test]
    fn create_unprotected_image_bind_protected_memory() {
        let (device, _) = gfx_dev_and_queue!(protected_memory);

        let image = RawImage::try_new(
            &device,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        )
        .unwrap();

        let requirements = image.memory_requirements()[0];

        let memory_type_index = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, ty)| {
                (requirements.memory_type_bits & (1 << id)) == 0
                    && ty.property_flags.contains(MemoryPropertyFlags::PROTECTED)
            })
            .expect("Failed to find suitable memory type") as u32;

        let memory = DeviceMemory::allocate(
            &device,
            &MemoryAllocateInfo {
                allocation_size: requirements.layout.size(),
                memory_type_index,
                ..Default::default()
            },
        )
        .expect("Failed to allocate memory");

        let resource_memory = ResourceMemory::new_dedicated(memory);

        let res = image.try_bind_memory(iter::once(resource_memory));
        match res {
            Err((Validated::ValidationError(_), _, _)) => {}
            Ok(_) => {
                panic!("bind_memory succeeded when it should have failed!");
            }
            Err((Validated::Error(err), _, _)) => {
                panic!(
                    "Expected a ValidationError, but got a runtime error: {:?}",
                    err
                );
            }
        }
    }

    #[test]
    fn create_protected_image_bind_unprotected_memory() {
        let (device, _) = gfx_dev_and_queue!(protected_memory);

        let image = RawImage::try_new(
            &device,
            &ImageCreateInfo {
                flags: ImageCreateFlags::PROTECTED,
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(image.flags(), ImageCreateFlags::PROTECTED);

        let requirements = image.memory_requirements()[0];

        let memory_type_index = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, ty)| {
                (requirements.memory_type_bits & (1 << id)) == 0
                    && !ty.property_flags.contains(MemoryPropertyFlags::PROTECTED)
            })
            .expect("Failed to find suitable memory type") as u32;

        let memory = DeviceMemory::allocate(
            &device,
            &MemoryAllocateInfo {
                allocation_size: requirements.layout.size(),
                memory_type_index,
                ..Default::default()
            },
        )
        .expect("Failed to allocate memory");

        let resource_memory = ResourceMemory::new_dedicated(memory);

        let res = image.try_bind_memory(iter::once(resource_memory));
        match res {
            Err((Validated::ValidationError(_), _, _)) => {}
            Ok(_) => {
                panic!("try_bind_memory succeeded when it should have failed!");
            }
            Err((Validated::Error(err), _, _)) => {
                panic!(
                    "Expected a ValidationError, but got a runtime error: {:?}",
                    err
                );
            }
        }
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        assert!(matches!(
            RawImage::try_new(
                &device,
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [32, 32, 1],
                    mip_levels: 0,
                    usage: ImageUsage::SAMPLED,
                    ..Default::default()
                },
            ),
            Err(Validated::ValidationError(_))
        ),);
    }

    #[test]
    fn mipmaps_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::try_new(
            &device,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                mip_levels: u32::MAX,
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        );

        match res {
            Err(Validated::ValidationError(_)) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn shader_storage_image_multisample() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::try_new(
            &device,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 32, 1],
                samples: SampleCount::Sample2,
                usage: ImageUsage::STORAGE,
                ..Default::default()
            },
        );

        match res {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([Requires::DeviceFeature(
                            "shader_storage_image_multisample"
                        )])],),
                        ..
                    }
                ) => {}
            Err(Validated::ValidationError(_)) => (), // unlikely but possible
            _ => panic!(),
        };
    }

    #[test]
    fn compressed_not_color_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::try_new(
            &device,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::ASTC_5x4_UNORM_BLOCK,
                extent: [32, 32, 1],
                usage: ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        );

        match res {
            Err(Validated::ValidationError(_)) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn transient_forbidden_with_some_usages() {
        let (device, _) = gfx_dev_and_queue!();

        assert!(matches!(
            RawImage::try_new(
                &device,
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [32, 32, 1],
                    usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
            ),
            Err(Validated::ValidationError(_))
        ))
    }

    #[test]
    fn cubecompatible_dims_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::try_new(
            &device,
            &ImageCreateInfo {
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [32, 64, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        );

        match res {
            Err(Validated::ValidationError(_)) => (),
            _ => panic!(),
        };
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn subresource_range_iterator() {
        // A fictitious set of aspects that no real image would actually ever have.
        let image_aspect_list: SmallVec<[ImageAspect; 4]> = (ImageAspects::COLOR
            | ImageAspects::DEPTH
            | ImageAspects::STENCIL
            | ImageAspects::PLANE_0)
            .into_iter()
            .collect();
        let image_mip_levels = 6;
        let image_array_layers = 8;

        let mip = image_array_layers as DeviceSize;
        let asp = mip * image_mip_levels as DeviceSize;

        // Whole image
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR
                    | ImageAspects::DEPTH
                    | ImageAspects::STENCIL
                    | ImageAspects::PLANE_0,
                level_count: Some(6),
                layer_count: Some(8),
                ..Default::default()
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(iter.next(), Some(0 * asp..4 * asp));
        assert_eq!(iter.next(), None);

        // Only some aspects
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR | ImageAspects::DEPTH | ImageAspects::PLANE_0,
                level_count: Some(6),
                layer_count: Some(8),
                ..Default::default()
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(iter.next(), Some(0 * asp..2 * asp));
        assert_eq!(iter.next(), Some(3 * asp..4 * asp));
        assert_eq!(iter.next(), None);

        // Two aspects, and only some of the mip levels
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                base_mip_level: 2,
                level_count: Some(2),
                layer_count: Some(8),
                ..Default::default()
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );
        assert_eq!(iter.next(), Some(1 * asp + 2 * mip..1 * asp + 4 * mip));
        assert_eq!(iter.next(), Some(2 * asp + 2 * mip..2 * asp + 4 * mip));
        assert_eq!(iter.next(), None);

        // One aspect, one mip level, only some of the array layers
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                level_count: Some(1),
                base_array_layer: 2,
                layer_count: Some(2),
                ..Default::default()
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(
            iter.next(),
            Some(0 * asp + 0 * mip + 2..0 * asp + 0 * mip + 4)
        );
        assert_eq!(iter.next(), None);

        // Two aspects, two mip levels, only some of the array layers
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                base_mip_level: 2,
                level_count: Some(2),
                base_array_layer: 6,
                layer_count: Some(2),
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );
        assert_eq!(
            iter.next(),
            Some(1 * asp + 2 * mip + 6..1 * asp + 2 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(1 * asp + 3 * mip + 6..1 * asp + 3 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(2 * asp + 2 * mip + 6..2 * asp + 2 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(2 * asp + 3 * mip + 6..2 * asp + 3 * mip + 8)
        );
        assert_eq!(iter.next(), None);
    }
}
