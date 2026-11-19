//! An opaque data structure that is used to accelerate spatial queries on geometry data.
//!
//! Acceleration structures contain geometry data, arranged in such a way that the device can
//! easily search through the data and check for intersections between the geometry and rays
//! (lines). The geometry data can consist of either triangles, or axis-aligned bounding boxes
//! (AABBs).
//!
//! Acceleration structures come in two forms: top-level and bottom-level. A bottom-level
//! acceleration structure holds the actual geometry data, while a top-level structure contains
//! instances of (references to) one or more bottom-level structures. A top-level structure is
//! intended to contain the whole rendered scene (or the relevant parts of it), while a
//! bottom-level structure may contain individual objects within the scene. This two-level
//! arrangement allows you to easily rearrange the scene, adding and removing parts of it as
//! needed.
//!
//! # Building an acceleration structure
//!
//! When an acceleration structure object is created, it is in an uninitialized state and contains
//! garbage data. To be able to use it for anything, you must first *build* the structure on the
//! device, using the [`build_acceleration_structure`] or [`build_acceleration_structure_indirect`]
//! command buffer commands. You use [`BuildAccelerationStructureMode::Build`] to build a structure
//! from scratch.
//!
//! When specifying geometry data for an acceleration structure build, certain triangle, AABB or
//! instance values mark that item as *inactive*. An inactive item is completely ignored within
//! the acceleration structure, and acts as if it's not there. The following special values
//! make an item inactive:
//! - For triangles, if the vertex format is a floating-point type, then the triangle is inactive
//!   if any of its vertices have NaN as their first coordinate. For integer vertex formats, it is
//!   not possible to mark triangles as inactive.
//! - For AABBs, if [`AabbPositions::min[0]`](AabbPositions::min) is NaN.
//! - For instances, if [`AccelerationStructureInstance::acceleration_structure_reference`] is 0.
//!
//! # Updating an acceleration structure
//!
//! Once an acceleration structure is built, if it was built previously with the
//! [`BuildAccelerationStructureFlags::ALLOW_UPDATE`] flag, then it is possible to update the
//! structure with new data. You use [`BuildAccelerationStructureMode::Update`] for this, which
//! specifies the source acceleration structure to use as a starting point for the update.
//! This can be the same as the destination structure (the update will happen in-place), or a
//! different one.
//!
//! An update operation is limited in which parts of the data it may change. You may change most
//! buffers and their contents, as well as strides and offsets:
//! - [`AccelerationStructureBuildGeometryInfo::scratch_data`]
//! - [`AccelerationStructureGeometryTrianglesData::vertex_data`]
//! - [`AccelerationStructureGeometryTrianglesData::vertex_stride`]
//! - [`AccelerationStructureGeometryTrianglesData::transform_data`] (but the variant of `Option`
//!   must not change)
//! - [`AccelerationStructureGeometryAabbsData::data`]
//! - [`AccelerationStructureGeometryAabbsData::stride`]
//! - [`AccelerationStructureGeometryInstancesData::data`]
//! - [`AccelerationStructureBuildRangeInfo::primitive_offset`]
//! - [`AccelerationStructureBuildRangeInfo::first_vertex`] if no index buffer is used
//! - [`AccelerationStructureBuildRangeInfo::transform_offset`]
//!
//! No other values may be changed, including in particular the variant or number of elements in
//! [`AccelerationStructureBuildGeometryInfo::geometries`], and the value of
//! [`AccelerationStructureBuildRangeInfo::primitive_count`].
//! The enum variants and data in [`AccelerationStructureGeometryTrianglesData::index_data`] must
//! not be changed, but it is allowed to specify a new index buffer, as long as it contains the
//! exact same indices as the old one.
//!
//! An update operation may not change the inactive status of an item: active items must remain
//! active in the update and inactive items must remain inactive.
//!
//! # Accessing an acceleration structure in a shader
//!
//! Acceleration structures can be bound to and accessed in any shader type. They are accessed
//! as descriptors, like buffers and images, and are declared in GLSL like so:
//!
//! ```glsl
//! layout (set = N, binding = N) uniform accelerationStructureEXT nameOfTheVariable;
//! ```
//!
//! You must enable either the `GL_EXT_ray_query` or the `GL_EXT_ray_tracing` GLSL extensions in
//! the shader to use this.
//!
//! On the Vulkano side, you can then create a descriptor set layout using
//! [`DescriptorType::AccelerationStructure`] as a descriptor type, and write the
//! acceleration structure to a descriptor set using
//! [`WriteDescriptorSet::acceleration_structure`].
//!
//! [`build_acceleration_structure`]: crate::command_buffer::AutoCommandBufferBuilder::build_acceleration_structure
//! [`build_acceleration_structure_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::build_acceleration_structure_indirect
//! [`DescriptorType::AccelerationStructure`]: crate::descriptor_set::layout::DescriptorType::AccelerationStructure
//! [`WriteDescriptorSet::acceleration_structure`]: crate::descriptor_set::WriteDescriptorSet::acceleration_structure

use crate::{
    buffer::{Buffer, BufferCreateFlags, BufferUsage, IndexType},
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    DeviceAddress, DeviceSize, Packed24_8, Requires, RequiresAllOf, RequiresOneOf, Validated,
    ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, hash::Hash, mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// An opaque data structure that is used to accelerate spatial queries on geometry data.
#[derive(Debug)]
pub struct AccelerationStructure {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    handle: vk::AccelerationStructureKHR,
    id: NonZero<u64>,

    create_flags: AccelerationStructureCreateFlags,
    buffer: Arc<Buffer>,
    offset: DeviceSize,
    size: DeviceSize,
    ty: AccelerationStructureType,
}

impl AccelerationStructure {
    /// Creates a new `AccelerationStructure`, panicking on a validation error.
    ///
    /// The [`acceleration_structure`] feature must be enabled on the device.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - `create_info.buffer` (and any subbuffer it overlaps with) must not be accessed while it
    ///   is bound to the acceleration structure.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`acceleration_structure`]: crate::device::DeviceFeatures::acceleration_structure
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub unsafe fn new(
        device: &Arc<Device>,
        create_info: &AccelerationStructureCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        match unsafe { Self::try_new(device, create_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `AccelerationStructure`.
    ///
    /// The [`acceleration_structure`] feature must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - `create_info.buffer` (and any subbuffer it overlaps with) must not be accessed while it
    ///   is bound to the acceleration structure.
    ///
    /// [`acceleration_structure`]: crate::device::DeviceFeatures::acceleration_structure
    #[inline]
    pub unsafe fn try_new(
        device: &Arc<Device>,
        create_info: &AccelerationStructureCreateInfo<'_>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &AccelerationStructureCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_extensions().khr_acceleration_structure {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_acceleration_structure",
                )])]),
                ..Default::default()
            }));
        }

        if !device.enabled_features().acceleration_structure {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "acceleration_structure",
                )])]),
                vuids: &["VUID-vkCreateAccelerationStructureKHR-accelerationStructure-03611"],
                ..Default::default()
            }));
        }

        // VUID-vkCreateAccelerationStructureKHR-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &AccelerationStructureCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.khr_acceleration_structure
                    .create_acceleration_structure_khr)(
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

    /// Creates a new `AccelerationStructure` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::AccelerationStructureKHR,
        create_info: &AccelerationStructureCreateInfo<'_>,
    ) -> Arc<Self> {
        let &AccelerationStructureCreateInfo {
            create_flags,
            buffer,
            offset,
            size,
            ty,
            _ne: _,
        } = create_info;

        Arc::new(Self {
            device: InstanceOwnedDebugWrapper(device.clone()),
            handle,
            id: Self::next_id(),

            create_flags,
            buffer: buffer.clone(),
            offset,
            size,
            ty,
        })
    }

    /// Returns the flags the acceleration structure was created with.
    #[inline]
    pub fn create_flags(&self) -> AccelerationStructureCreateFlags {
        self.create_flags
    }

    /// Returns the buffer that the acceleration structure is stored in.
    #[inline]
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Returns the offset from the start of [`buffer`] where the acceleration structure is stored.
    ///
    /// [`buffer`]: Self::buffer
    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    /// Returns the size of the acceleration structure.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the type of the acceleration structure.
    #[inline]
    pub fn ty(&self) -> AccelerationStructureType {
        self.ty
    }

    /// Returns the device address of the acceleration structure.
    ///
    /// The device address of the acceleration structure may be different from the device address
    /// of the underlying buffer.
    pub fn device_address(&self) -> NonZero<DeviceAddress> {
        let info_vk = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(self.handle);
        let fns = self.device.fns();
        let ptr = unsafe {
            (fns.khr_acceleration_structure
                .get_acceleration_structure_device_address_khr)(
                self.device.handle(), &info_vk
            )
        };

        NonZero::new(ptr).unwrap()
    }
}

impl Drop for AccelerationStructure {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe {
            (fns.khr_acceleration_structure
                .destroy_acceleration_structure_khr)(
                self.device.handle(), self.handle, ptr::null()
            )
        }
    }
}

unsafe impl VulkanObject for AccelerationStructure {
    type Handle = vk::AccelerationStructureKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for AccelerationStructure {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(AccelerationStructure);

vulkan_enum! {
    #[non_exhaustive]

    /// The type of an acceleration structure.
    AccelerationStructureType = AccelerationStructureTypeKHR(i32);

    /// Refers to bottom-level acceleration structures. This type can be bound to a descriptor.
    TopLevel = TOP_LEVEL,

    /// Contains AABBs or geometry to be intersected.
    BottomLevel = BOTTOM_LEVEL,

    /// The type is determined at build time.
    ///
    /// Use of this type is discouraged, it is preferred to specify the type at create time.
    Generic = GENERIC,
}

/// Parameters to create a new `AccelerationStructure`.
#[derive(Clone, Debug)]
pub struct AccelerationStructureCreateInfo<'a> {
    /// Specifies how to create the acceleration structure.
    ///
    /// The default value is empty.
    pub create_flags: AccelerationStructureCreateFlags,

    /// The subbuffer to store the acceleration structure on.
    ///
    /// The subbuffer must have an `offset` that is a multiple of 256, and its `usage` must include
    /// [`BufferUsage::ACCELERATION_STRUCTURE_STORAGE`]. It must not be accessed while it is bound
    /// to the acceleration structure.
    ///
    /// There is no default value.
    pub buffer: &'a Arc<Buffer>,

    /// The offset (in bytes) from the start of `buffer` where the acceleration structure will be
    /// stored.
    ///
    /// This must be a multiple of 256.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The size (in bytes) required for the acceleration structure.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// The type of acceleration structure to create.
    ///
    /// The default value is [`AccelerationStructureType::Generic`].
    pub ty: AccelerationStructureType,

    /* TODO: enable
    // TODO: document
    pub device_address: DeviceAddress, */
    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> AccelerationStructureCreateInfo<'a> {
    /// Returns a default `AccelerationStructureCreateInfo` with the provided `buffer`.
    #[inline]
    pub const fn new(buffer: &'a Arc<Buffer>) -> Self {
        Self {
            create_flags: AccelerationStructureCreateFlags::empty(),
            buffer,
            offset: 0,
            size: 0,
            ty: AccelerationStructureType::Generic,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            create_flags,
            buffer,
            offset,
            size,
            ty,
            _ne: _,
        } = self;

        create_flags.validate_device(device).map_err(|err| {
            err.add_context("create_flags")
                .set_vuids(&["VUID-VkAccelerationStructureCreateInfoKHR-createFlags-parameter"])
        })?;

        ty.validate_device(device).map_err(|err| {
            err.add_context("ty")
                .set_vuids(&["VUID-VkAccelerationStructureCreateInfoKHR-type-parameter"])
        })?;

        if !buffer
            .usage()
            .intersects(BufferUsage::ACCELERATION_STRUCTURE_STORAGE)
        {
            return Err(Box::new(ValidationError {
                context: "buffer".into(),
                problem: "the buffer was not created with the `ACCELERATION_STRUCTURE_STORAGE` \
                    usage"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-buffer-03614"],
                ..Default::default()
            }));
        }

        if buffer
            .flags()
            .intersects(BufferCreateFlags::SPARSE_RESIDENCY)
        {
            return Err(Box::new(ValidationError {
                context: "buffer.flags()".into(),
                problem: "contains `BufferCreateFlags::SPARSE_RESIDENCY`".into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-buffer-03615"],
                ..Default::default()
            }));
        }

        if offset
            .checked_add(size)
            .is_none_or(|end| end > buffer.size())
        {
            return Err(Box::new(ValidationError {
                problem: "`offset + size` is greater than `buffer.size()`".into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-offset-03616"],
                ..Default::default()
            }));
        }

        if !offset.is_multiple_of(256) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 256".into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-offset-03734"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureCreateInfoKHR<'static> {
        let &Self {
            create_flags,
            buffer,
            offset,
            size,
            ty,
            _ne: _,
        } = self;

        vk::AccelerationStructureCreateInfoKHR::default()
            .create_flags(create_flags.into())
            .buffer(buffer.handle())
            .offset(offset)
            .size(size)
            .ty(ty.into())
            .device_address(0) // TODO: allow user to specify
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that control how an acceleration structure is created.
    AccelerationStructureCreateFlags = AccelerationStructureCreateFlagsKHR(u32);

    /* TODO: enable
    // TODO: document
    DEVICE_ADDRESS_CAPTURE_REPLAY = DEVICE_ADDRESS_CAPTURE_REPLAY_KHR, */

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER_CAPTURE_REPLAY = DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MOTION = MOTION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing_motion_blur)]),
    ]),*/
}

/// Geometries and other parameters for an acceleration structure build operation.
#[derive(Clone, Debug)]
pub struct AccelerationStructureBuildGeometryInfo<'a> {
    /// The type of acceleration structure to build.
    ///
    /// The default value is `AccelerationStructureType::Generic`, which must be overridden.
    pub ty: AccelerationStructureType,

    /// Additional properties of how the acceleration structure should be built.
    ///
    /// The default value is empty.
    pub flags: BuildAccelerationStructureFlags,

    /// The mode that the build command should operate in.
    ///
    /// This is ignored when calling [`Device::acceleration_structure_build_sizes`].
    ///
    /// The default value is [`BuildAccelerationStructureMode::Build`].
    pub mode: BuildAccelerationStructureMode,

    /// The acceleration structure to use as the source of an acceleration structure update.
    ///
    /// This can be `None` when calling [`Device::acceleration_structure_build_sizes`] or when
    /// `mode` is `BuildAccelerationStructureMode::Build`, but must be `Some` otherwise.
    ///
    /// This can be the same as `dst_acceleration_structure`, in which case the update happens
    /// in-place.
    ///
    /// The default value is `None`.
    pub src_acceleration_structure: Option<&'a Arc<AccelerationStructure>>,

    /// The acceleration structure to build or update.
    ///
    /// This can be `None` when calling [`Device::acceleration_structure_build_sizes`], but must be
    /// `Some` otherwise.
    ///
    /// This can be the same as `src_acceleration_structure`, in which case the update happens
    /// in-place.
    ///
    /// The default value is `None`.
    pub dst_acceleration_structure: Option<&'a Arc<AccelerationStructure>>,

    /// The geometries that will be built into `dst_acceleration_structure`.
    ///
    /// `ty` restricts which geometry types can be used:
    /// - [`TopLevel`] only allows a single [`Instances`] geometry.
    /// - [`BottomLevel`] only allows one of [`Triangles`] or [`Aabbs`] geometries.
    ///
    /// The default value is empty.
    ///
    /// [`TopLevel`]: AccelerationStructureType::TopLevel
    /// [`BottomLevel`]: AccelerationStructureType::BottomLevel
    /// [`Triangles`]: AccelerationStructureGeometryData::Triangles
    /// [`Aabbs`]: AccelerationStructureGeometryData::Aabbs
    /// [`Instances`]: AccelerationStructureGeometryData::Instances
    pub geometries: &'a [AccelerationStructureGeometry<'a>],

    /// Scratch memory to be used for the build.
    ///
    /// This can be `0` (null) when calling [`Device::acceleration_structure_build_sizes`], but
    /// must be a valid buffer device address otherwise.
    ///
    /// The default value is `0` (null).
    pub scratch_data: DeviceAddress,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AccelerationStructureBuildGeometryInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> AccelerationStructureBuildGeometryInfo<'a> {
    /// Returns a default `AccelerationStructureBuildGeometryInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            ty: AccelerationStructureType::Generic,
            flags: BuildAccelerationStructureFlags::empty(),
            mode: BuildAccelerationStructureMode::Build,
            src_acceleration_structure: None,
            dst_acceleration_structure: None,
            geometries: &[],
            scratch_data: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ty,
            flags,
            mode,
            src_acceleration_structure,
            dst_acceleration_structure,
            geometries,
            scratch_data: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkAccelerationStructureBuildGeometryInfoKHR-flags-parameter"])
        })?;

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode") // vuids?
        })?;

        if let Some(src_acceleration_structure) = src_acceleration_structure {
            // VUID-VkAccelerationStructureBuildGeometryInfoKHR-commonparent
            assert_eq!(device, src_acceleration_structure.device().as_ref());
        }

        if let Some(dst_acceleration_structure) = dst_acceleration_structure {
            // VUID-VkAccelerationStructureBuildGeometryInfoKHR-commonparent
            assert_eq!(device, dst_acceleration_structure.device().as_ref());
        }

        match ty {
            AccelerationStructureType::TopLevel => {
                for (index, geometry) in geometries.iter().enumerate() {
                    if !matches!(
                        geometry.geometry,
                        AccelerationStructureGeometryData::Instances(_),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`ty` is `AccelerationStructureType::TopLevel`, but \
                                `geometries` doesn't contain only \
                                `AccelerationStructureGeometryData::Instances` geometries"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03789"],
                            ..Default::default()
                        }));
                    }

                    geometry
                        .validate(device)
                        .map_err(|err| err.add_context(format!("geometries[{}]", index)))?;
                }

                if geometries.len() != 1 {
                    return Err(Box::new(ValidationError {
                        problem: "`ty` is `AccelerationStructureType::TopLevel`, but \
                            `geometries.len()` is not 1"
                            .into(),
                        vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03790"],
                        ..Default::default()
                    }));
                }
            }
            AccelerationStructureType::BottomLevel => {
                let mut prev_geometry_type = geometries
                    .first()
                    .map(|geometry| geometry.geometry.geometry_type())
                    .unwrap_or_default();

                for (index, geometry) in geometries.iter().enumerate() {
                    if matches!(
                        geometry.geometry,
                        AccelerationStructureGeometryData::Instances(_),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`ty` is `AccelerationStructureType::BottomLevel`, but \
                                `geometries` contains an \
                                `AccelerationStructureGeometryData::Instances` geometry"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03791"],
                            ..Default::default()
                        }));
                    }

                    let geometry_type = geometry.geometry.geometry_type();

                    if geometry_type != prev_geometry_type {
                        return Err(Box::new(ValidationError {
                            problem: "`ty` is `AccelerationStructureType::BottomLevel`, but \
                                `geometries` don't all have the same geometry type"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03792"],
                            ..Default::default()
                        }));
                    }

                    geometry
                        .validate(device)
                        .map_err(|err| err.add_context(format!("geometries[{}]", index)))?;

                    prev_geometry_type = geometry_type;
                }

                let max_geometry_count = device
                    .physical_device()
                    .properties()
                    .max_geometry_count
                    .unwrap();

                if geometries.len() as u64 > max_geometry_count {
                    return Err(Box::new(ValidationError {
                        problem: "`geometries.len()` exceeds the `max_geometry_count` limit".into(),
                        vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03793"],
                        ..Default::default()
                    }));
                }
            }
            AccelerationStructureType::Generic => {
                return Err(Box::new(ValidationError {
                    context: "ty".into(),
                    problem: "is `AccelerationStructureType::Generic`".into(),
                    vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03654"],
                    ..Default::default()
                }));
            }
        }

        if flags.contains(
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE
                | BuildAccelerationStructureFlags::PREFER_FAST_BUILD,
        ) {
            return Err(Box::new(ValidationError {
                context: "flags".into(),
                problem: "contains both `BuildAccelerationStructureFlags::PREFER_FAST_TRACE` and \
                    `BuildAccelerationStructureFlags::PREFER_FAST_BUILD`"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-flags-03796"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a AccelerationStructureBuildGeometryInfoFields1Vk,
    ) -> vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
        let &Self {
            ty,
            flags,
            mode,
            src_acceleration_structure,
            dst_acceleration_structure,
            geometries: _,
            scratch_data,
            _ne: _,
        } = self;
        let AccelerationStructureBuildGeometryInfoFields1Vk { geometries_vk } = fields1_vk;

        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty.into())
            .flags(flags.into())
            .mode(mode.into())
            .src_acceleration_structure(
                src_acceleration_structure.map_or_else(Default::default, VulkanObject::handle),
            )
            .dst_acceleration_structure(
                dst_acceleration_structure.map_or_else(Default::default, VulkanObject::handle),
            )
            .geometries(geometries_vk)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_data,
            })
    }

    pub(crate) fn to_vk_fields1(&self) -> AccelerationStructureBuildGeometryInfoFields1Vk {
        let &Self { geometries, .. } = self;

        let geometries_vk = geometries
            .iter()
            .map(AccelerationStructureGeometry::to_vk)
            .collect();

        AccelerationStructureBuildGeometryInfoFields1Vk { geometries_vk }
    }
}

pub(crate) struct AccelerationStructureBuildGeometryInfoFields1Vk {
    pub(crate) geometries_vk: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags to control how an acceleration structure should be built.
    BuildAccelerationStructureFlags = BuildAccelerationStructureFlagsKHR(u32);

    /// The built acceleration structure can be updated later by building it again with
    /// [`BuildAccelerationStructureMode::Update`].
    ///
    /// The building process may take more time and memory than normal.
    ALLOW_UPDATE = ALLOW_UPDATE,

    /// The built acceleration structure can be used later as the source in a copy operation with
    /// [`CopyAccelerationStructureMode::Compact`].
    ///
    /// The building process may take more time and memory than normal.
    ALLOW_COMPACTION = ALLOW_COMPACTION,

    /// Prioritize for best trace performance, with possibly longer build times.
    PREFER_FAST_TRACE = PREFER_FAST_TRACE,

    /// Prioritize for shorter build time, with possibly suboptimal trace performance.
    PREFER_FAST_BUILD = PREFER_FAST_BUILD,

    /// Prioritize low acceleration structure and scratch memory size, with possibly longer build
    /// times or suboptimal trace performance.
    LOW_MEMORY = LOW_MEMORY,

    /* TODO: enable
    // TODO: document
    MOTION = MOTION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing_motion_blur)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_OPACITY_MICROMAP_UPDATE = ALLOW_OPACITY_MICROMAP_UPDATE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_DISABLE_OPACITY_MICROMAPS = ALLOW_DISABLE_OPACITY_MICROMAPS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_OPACITY_MICROMAP_DATA_UPDATE = ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_DISPLACEMENT_MICROMAP_UPDATE = ALLOW_DISPLACEMENT_MICROMAP_UPDATE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_displacement_micromap)]),
    ]), */
}

vulkan_enum! {
    #[non_exhaustive]

    /// What mode an acceleration structure build command should operate in.
    BuildAccelerationStructureMode = BuildAccelerationStructureModeKHR(i32);

    /// Build a new acceleration structure from scratch.
    Build = BUILD,

    /// Update a previously built source acceleration structure with new data, storing the updated
    /// structure in the destination.
    ///
    /// The destination acceleration structure must have been built with the
    /// [`BuildAccelerationStructureFlags::ALLOW_UPDATE`] flag.
    Update = UPDATE,
}

#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometry<'a> {
    /// The type of geometry data.
    ///
    /// There is no default value.
    pub geometry: AccelerationStructureGeometryData<'a>,

    /// Additional properties of how the geometry should be built.
    ///
    /// The default value is empty.
    pub flags: GeometryFlags,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> AccelerationStructureGeometry<'a> {
    /// Returns a default `AccelerationStructureGeometry` with the provided `geometry`.
    #[inline]
    pub const fn new(geometry: AccelerationStructureGeometryData<'a>) -> Self {
        Self {
            geometry,
            flags: GeometryFlags::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref geometry,
            flags,
            _ne: _,
        } = self;

        geometry.validate(device)?;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkAccelerationStructureGeometryKHR-flags-parameter"])
        })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureGeometryKHR<'static> {
        let &Self {
            ref geometry,
            flags,
            _ne: _,
        } = self;

        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(geometry.geometry_type().into())
            .geometry(geometry.to_vk())
            .flags(flags.into())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags to control how an acceleration structure geometry should be built.
    GeometryFlags = GeometryFlagsKHR(u32);

    /// The geometry does not invoke the any-hit shaders, even if it is present in a hit group.
    OPAQUE = OPAQUE,

    /// The any-hit shader will never be called more than once for each primitive in the geometry.
    NO_DUPLICATE_ANY_HIT_INVOCATION = NO_DUPLICATE_ANY_HIT_INVOCATION,
}

/// The type of geometry data in an acceleration structure.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AccelerationStructureGeometryData<'a> {
    /// The geometry consists of triangle data.
    Triangles(AccelerationStructureGeometryTrianglesData<'a>),

    /// The geometry consists of axis-aligned bounding box data.
    Aabbs(AccelerationStructureGeometryAabbsData<'a>),

    /// The geometry consists of instance data.
    Instances(AccelerationStructureGeometryInstancesData<'a>),
}

impl AccelerationStructureGeometryData<'_> {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        match self {
            AccelerationStructureGeometryData::Triangles(geometry) => geometry.validate(device),
            AccelerationStructureGeometryData::Aabbs(geometry) => geometry.validate(device),
            AccelerationStructureGeometryData::Instances(geometry) => geometry.validate(device),
        }
    }

    pub(crate) fn geometry_type(&self) -> GeometryType {
        match self {
            AccelerationStructureGeometryData::Triangles(_) => GeometryType::Triangles,
            AccelerationStructureGeometryData::Aabbs(_) => GeometryType::Aabbs,
            AccelerationStructureGeometryData::Instances(_) => GeometryType::Instances,
        }
    }

    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureGeometryDataKHR<'static> {
        match self {
            AccelerationStructureGeometryData::Triangles(geometry) => {
                vk::AccelerationStructureGeometryDataKHR {
                    triangles: geometry.to_vk(),
                }
            }
            AccelerationStructureGeometryData::Aabbs(geometry) => {
                vk::AccelerationStructureGeometryDataKHR {
                    aabbs: geometry.to_vk(),
                }
            }
            AccelerationStructureGeometryData::Instances(geometry) => {
                vk::AccelerationStructureGeometryDataKHR {
                    instances: geometry.to_vk(),
                }
            }
        }
    }
}

impl<'a> From<AccelerationStructureGeometryTrianglesData<'a>>
    for AccelerationStructureGeometryData<'a>
{
    #[inline]
    fn from(value: AccelerationStructureGeometryTrianglesData<'a>) -> Self {
        Self::Triangles(value)
    }
}

impl<'a> From<AccelerationStructureGeometryAabbsData<'a>>
    for AccelerationStructureGeometryData<'a>
{
    #[inline]
    fn from(value: AccelerationStructureGeometryAabbsData<'a>) -> Self {
        Self::Aabbs(value)
    }
}

impl<'a> From<AccelerationStructureGeometryInstancesData<'a>>
    for AccelerationStructureGeometryData<'a>
{
    #[inline]
    fn from(value: AccelerationStructureGeometryInstancesData<'a>) -> Self {
        Self::Instances(value)
    }
}

// TODO: `vulkan_enum!` can't generate non-public types.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum GeometryType {
    #[default]
    Triangles = vk::GeometryTypeKHR::TRIANGLES.as_raw(),
    Aabbs = vk::GeometryTypeKHR::AABBS.as_raw(),
    Instances = vk::GeometryTypeKHR::INSTANCES.as_raw(),
}

impl From<GeometryType> for vk::GeometryTypeKHR {
    #[inline]
    fn from(val: GeometryType) -> Self {
        vk::GeometryTypeKHR::from_raw(val as i32)
    }
}

/// A bottom-level geometry consisting of triangles.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryTrianglesData<'a> {
    /// The format of each vertex in `vertex_data`.
    ///
    /// This works in the same way as formats for vertex buffers.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub vertex_format: Format,

    /// The address of the vertex data itself, consisting of an array of `vertex_format` values.
    ///
    /// This can be `0` (null) when calling [`Device::acceleration_structure_build_sizes`], but
    /// must be a valid buffer device address otherwise.
    ///
    /// The default value is `0` (null).
    pub vertex_data: DeviceAddress,

    /// The number of bytes between the start of successive elements in `vertex_data`.
    ///
    /// This must be a multiple of the smallest component size (in bytes) of `vertex_format`.
    ///
    /// The default value is 0, which must be overridden.
    pub vertex_stride: u32,

    /// The highest vertex index that may be read from `vertex_data`.
    ///
    /// The default value is 0, which must be overridden.
    pub max_vertex: u32,

    /// If indices are to be used, the type of each index in `index_data`.
    ///
    /// The default value is `None`.
    pub index_type: Option<IndexType>,

    /// If indices are to be used, the address where the index data will be read from.
    ///
    /// The indices will be used to index into the elements of `vertex_data`.
    ///
    /// The default value is `0` (null).
    pub index_data: DeviceAddress,

    /// Optionally, the address of a [`TransformMatrix`] used to transform the vertices in
    /// `vertex_data` to the space in which the acceleration structure is defined.
    ///
    /// The default value is `0` (null).
    pub transform_data: DeviceAddress,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AccelerationStructureGeometryTrianglesData<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AccelerationStructureGeometryTrianglesData<'_> {
    /// Returns a default `AccelerationStructureGeometryTrianglesData`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            vertex_format: Format::UNDEFINED,
            vertex_data: 0,
            vertex_stride: 0,
            max_vertex: 0,
            index_type: None,
            index_data: 0,
            transform_data: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            vertex_format,
            vertex_data: _,
            vertex_stride,
            max_vertex: _,
            index_type,
            index_data,
            transform_data: _,
            _ne: _,
        } = self;

        vertex_format.validate_device(device).map_err(|err| {
            err.add_context("vertex_format").set_vuids(&[
                "VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexFormat-parameter",
            ])
        })?;

        let format_properties = unsafe {
            device
                .physical_device()
                .format_properties_unchecked(vertex_format)
        };

        if !format_properties
            .buffer_features
            .intersects(FormatFeatures::ACCELERATION_STRUCTURE_VERTEX_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "vertex_format".into(),
                problem: "format features do not contain \
                    `FormatFeature::ACCELERATION_STRUCTURE_VERTEX_BUFFER`"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexFormat-03797"],
                ..Default::default()
            }));
        }

        let smallest_component_bits = vertex_format
            .components()
            .into_iter()
            .filter(|&c| c != 0)
            .min()
            .unwrap() as u32;
        let smallest_component_bytes = ((smallest_component_bits + 7) & !7) / 8;

        if !vertex_stride.is_multiple_of(smallest_component_bytes) {
            return Err(Box::new(ValidationError {
                problem: "`vertex_stride` is not a multiple of the byte size of the \
                    smallest component of `vertex_format`"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexStride-03735"],
                ..Default::default()
            }));
        }

        if let Some(index_type) = index_type {
            if !matches!(index_type, IndexType::U16 | IndexType::U32) {
                return Err(Box::new(ValidationError {
                    context: "index_type".into(),
                    problem: "is not `IndexType::U16` or `IndexType::U32`".into(),
                    vuids: &[
                        "VUID-VkAccelerationStructureGeometryTrianglesDataKHR-indexType-03798",
                    ],
                    ..Default::default()
                }));
            }
        } else {
            if index_data != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`index_type` is `None`, but `index_data` is not 0".into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureGeometryTrianglesDataKHR<'static> {
        let &Self {
            vertex_format,
            vertex_data,
            vertex_stride,
            max_vertex,
            index_type,
            index_data,
            transform_data,
            _ne,
        } = self;

        vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vertex_format.into())
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_data,
            })
            .vertex_stride(vertex_stride as DeviceSize)
            .max_vertex(max_vertex)
            .index_type(index_type.map_or(vk::IndexType::NONE_KHR, Into::into))
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_data,
            })
            .transform_data(vk::DeviceOrHostAddressConstKHR {
                device_address: transform_data,
            })
    }
}

/// A 3x4 row-major affine transformation matrix.
///
/// The first three columns must be a 3x3 invertible matrix.
pub type TransformMatrix = [[f32; 4]; 3];

/// A bottom-level geometry consisting of axis-aligned bounding boxes.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryAabbsData<'a> {
    /// The AABB data itself, consisting of an array of [`AabbPositions`] structs.
    ///
    /// This can be `0` (null) when calling [`Device::acceleration_structure_build_sizes`], but
    /// must be a valid buffer device address otherwise.
    ///
    /// The default value is `0` (null).
    pub data: DeviceAddress,

    /// The number of bytes between the start of successive elements in `data`.
    ///
    /// This must be a multiple of 8.
    ///
    /// The default value is 0, which must be overridden.
    pub stride: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AccelerationStructureGeometryAabbsData<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AccelerationStructureGeometryAabbsData<'_> {
    /// Returns a default `AccelerationStructureGeometryAabbsData`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: 0,
            stride: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, _device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            data: _,
            stride,
            _ne: _,
        } = self;

        if !stride.is_multiple_of(8) {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is not a multiple of 8".into(),
                vuids: &["VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureGeometryAabbsDataKHR<'static> {
        let &Self {
            data,
            stride,
            _ne: _,
        } = self;

        vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: data,
            })
            .stride(stride as DeviceSize)
    }
}

/// Specifies two opposing corners of an axis-aligned bounding box.
///
/// Each value in `min` must be less than or equal to the corresponding value in `max`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct AabbPositions {
    /// The minimum of the corner coordinates of the bounding box.
    ///
    /// The default value is `[0.0; 3]`.
    pub min: [f32; 3],

    /// The maximum of the corner coordinates of the bounding box.
    ///
    /// The default value is `[0.0; 3]`.
    pub max: [f32; 3],
}

unsafe impl Pod for AabbPositions {}
unsafe impl Zeroable for AabbPositions {}

/// A top-level geometry consisting of instances of bottom-level acceleration structures.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryInstancesData<'a> {
    pub array_of_pointers: bool,

    /// The instance data itself.
    ///
    /// The default value is `0` (null), which must be overridden.
    pub data: DeviceAddress,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AccelerationStructureGeometryInstancesData<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AccelerationStructureGeometryInstancesData<'_> {
    /// Returns a default `AccelerationStructureGeometryInstancesData`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            array_of_pointers: false,
            data: 0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, _device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            array_of_pointers: _,
            data: _,
            _ne: _,
        } = self;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureGeometryInstancesDataKHR<'static> {
        let &Self {
            array_of_pointers,
            data,
            _ne: _,
        } = self;

        vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(array_of_pointers)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: data,
            })
    }
}

/// Specifies a bottom-level acceleration structure instance when building a top-level structure.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct AccelerationStructureInstance {
    /// A 3x4 transformation matrix to be applied to the bottom-level acceleration structure.
    ///
    /// The first three columns must be a 3x3 invertible matrix.
    ///
    /// The default value is a 3x3 identity matrix, with the fourth column filled with zeroes.
    pub transform: TransformMatrix,

    /// Low 24 bits: A custom index value to be accessible via the `InstanceCustomIndexKHR`
    /// built-in variable in ray shaders. The default value is 0.
    ///
    /// High 8 bits: A visibility mask for the geometry. The instance will not be hit if the cull
    /// mask ANDed with this mask is zero. The default value is 0xFF.
    pub instance_custom_index_and_mask: Packed24_8,

    /// Low 24 bits: An offset used in calculating the binding table index of the hit shader.
    /// The default value is 0.
    ///
    /// High 8 bits: [`GeometryInstanceFlags`] to apply to the instance. The `From` trait can be
    /// used to convert the flags into a `u8` value. The default value is empty.
    pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,

    /// The device address of the bottom-level acceleration structure in this instance.
    ///
    /// The default value is 0 (null).
    pub acceleration_structure_reference: DeviceAddress,
}

unsafe impl Pod for AccelerationStructureInstance {}
unsafe impl Zeroable for AccelerationStructureInstance {}

impl Default for AccelerationStructureInstance {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AccelerationStructureInstance {
    /// Returns a default `AccelerationStructureInstance`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
            acceleration_structure_reference: 0,
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags for an instance in a top-level acceleration structure.
    GeometryInstanceFlags = GeometryInstanceFlagsKHR(u32);

    /// Disable face culling for the instance.
    TRIANGLE_FACING_CULL_DISABLE = TRIANGLE_FACING_CULL_DISABLE,

    /// Flip the facing (front vs back) of triangles.
    TRIANGLE_FLIP_FACING = TRIANGLE_FLIP_FACING,

    /// Geometries in this instance will act as if [`GeometryFlags::OPAQUE`] were specified.
    FORCE_OPAQUE = FORCE_OPAQUE,

    /// Geometries in this instance will act as if [`GeometryFlags::OPAQUE`] were not specified.
    FORCE_NO_OPAQUE = FORCE_NO_OPAQUE,

    /* TODO: enable
    // TODO: document
    FORCE_OPACITY_MICROMAP_2_STATE = FORCE_OPACITY_MICROMAP_2_STATE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DISABLE_OPACITY_MICROMAPS = DISABLE_OPACITY_MICROMAPS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */
}

impl From<GeometryInstanceFlags> for u8 {
    #[inline]
    fn from(value: GeometryInstanceFlags) -> Self {
        value.0 as u8
    }
}

/// Counts and offsets for an acceleration structure build operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct AccelerationStructureBuildRangeInfo {
    /// The number of primitives.
    ///
    /// The default value is 0.
    pub primitive_count: u32,

    /// The offset (in bytes) into the buffer holding geometry data,
    /// to where the first primitive is stored.
    ///
    /// The default value is 0.
    pub primitive_offset: u32,

    /// The index of the first vertex to build from.
    ///
    /// This is used only for triangle geometries.
    ///
    /// The default value is 0.
    pub first_vertex: u32,

    /// The offset (in bytes) into the buffer holding transform matrices,
    /// to where the matrix is stored.
    ///
    /// This is used only for triangle geometries.
    ///
    /// The default value is 0.
    pub transform_offset: u32,
}

unsafe impl Pod for AccelerationStructureBuildRangeInfo {}
unsafe impl Zeroable for AccelerationStructureBuildRangeInfo {}

impl AccelerationStructureBuildRangeInfo {
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> vk::AccelerationStructureBuildRangeInfoKHR {
        let &Self {
            primitive_count,
            primitive_offset,
            first_vertex,
            transform_offset,
        } = self;

        vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count,
            primitive_offset,
            first_vertex,
            transform_offset,
        }
    }
}

/// Parameters for copying an acceleration structure.
#[derive(Clone, Debug)]
pub struct CopyAccelerationStructureInfo<'a> {
    /// The acceleration structure to copy from.
    ///
    /// There is no default value.
    pub src: &'a Arc<AccelerationStructure>,

    /// The acceleration structure to copy into.
    ///
    /// There is no default value.
    pub dst: &'a Arc<AccelerationStructure>,

    /// Additional operations to perform during the copy.
    ///
    /// The default value is [`CopyAccelerationStructureMode::Clone`].
    pub mode: CopyAccelerationStructureMode,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyAccelerationStructureInfo<'a> {
    /// Returns a default `CopyAccelerationStructureInfo` with the provided `src` and `dst`.
    #[inline]
    pub const fn new(
        src: &'a Arc<AccelerationStructure>,
        dst: &'a Arc<AccelerationStructure>,
    ) -> Self {
        Self {
            src,
            dst,
            mode: CopyAccelerationStructureMode::Clone,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src,
            dst,
            mode,
            _ne: _,
        } = self;

        // VUID-VkCopyAccelerationStructureInfoKHR-commonparent
        assert_eq!(device, src.device().as_ref());
        assert_eq!(device, dst.device().as_ref());

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode")
                .set_vuids(&["VUID-VkCopyAccelerationStructureInfoKHR-mode-parameter"])
        })?;

        if !matches!(
            mode,
            CopyAccelerationStructureMode::Compact | CopyAccelerationStructureMode::Clone
        ) {
            return Err(Box::new(ValidationError {
                context: "mode".into(),
                problem: "is not `CopyAccelerationStructureMode::Compact` or \
                    `CopyAccelerationStructureMode::Clone`"
                    .into(),
                vuids: &["VUID-VkCopyAccelerationStructureInfoKHR-mode-03410"],
                ..Default::default()
            }));
        }

        if src.buffer() == dst.buffer() {
            return Err(Box::new(ValidationError {
                problem: "`src` and `dst` share the same buffer".into(),
                vuids: &["VUID-VkCopyAccelerationStructureInfoKHR-dst-07791"],
                ..Default::default()
            }));
        }

        // VUID-VkCopyAccelerationStructureInfoKHR-src-04963
        // TODO: unsafe

        // VUID-VkCopyAccelerationStructureInfoKHR-src-03411
        // TODO: unsafe

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::CopyAccelerationStructureInfoKHR<'static> {
        let &Self {
            src,
            dst,
            mode,
            _ne: _,
        } = self;

        vk::CopyAccelerationStructureInfoKHR::default()
            .src(src.handle())
            .dst(dst.handle())
            .mode(mode.into())
    }
}

/// Parameters for copying from an acceleration structure into memory.
#[derive(Clone, Debug)]
pub struct CopyAccelerationStructureToMemoryInfo<'a> {
    /// The acceleration structure to copy from.
    ///
    /// There is no default value.
    pub src: &'a Arc<AccelerationStructure>,

    /// The address to copy the acceleration structure to.
    ///
    /// This must be a valid buffer device address.
    ///
    /// The default value is `0` (null), which must be overridden.
    pub dst: DeviceAddress,

    /// Additional operations to perform during the copy.
    ///
    /// The default value is [`CopyAccelerationStructureMode::Serialize`].
    pub mode: CopyAccelerationStructureMode,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyAccelerationStructureToMemoryInfo<'a> {
    /// Returns a default `CopyAccelerationStructureToMemoryInfo` with the provided `src`.
    #[inline]
    pub const fn new(src: &'a Arc<AccelerationStructure>) -> Self {
        Self {
            src,
            dst: 0,
            mode: CopyAccelerationStructureMode::Serialize,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src,
            dst: _,
            mode,
            _ne: _,
        } = self;

        assert_eq!(device, src.device().as_ref());

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode")
                .set_vuids(&["VUID-VkCopyAccelerationStructureToMemoryInfoKHR-mode-parameter"])
        })?;

        if !matches!(mode, CopyAccelerationStructureMode::Serialize) {
            return Err(Box::new(ValidationError {
                context: "mode".into(),
                problem: "is not `CopyAccelerationStructureMode::Serialize`".into(),
                vuids: &["VUID-VkCopyAccelerationStructureToMemoryInfoKHR-mode-03412"],
                ..Default::default()
            }));
        }

        // VUID-VkCopyAccelerationStructureToMemoryInfoKHR-src-04959
        // TODO: unsafe

        // VUID-VkCopyAccelerationStructureToMemoryInfoKHR-dst-03561
        // TODO: unsafe

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::CopyAccelerationStructureToMemoryInfoKHR<'static> {
        let &Self {
            src,
            dst,
            mode,
            _ne: _,
        } = self;

        vk::CopyAccelerationStructureToMemoryInfoKHR::default()
            .src(src.handle())
            .dst(vk::DeviceOrHostAddressKHR {
                device_address: dst,
            })
            .mode(mode.into())
    }
}

/// Parameters for copying from memory into an acceleration structure.
#[derive(Clone, Debug)]
pub struct CopyMemoryToAccelerationStructureInfo<'a> {
    /// The address to copy the acceleration structure from.
    ///
    /// This must be a valid buffer device address.
    ///
    /// The default value is `0` (null), which must be overridden.
    pub src: DeviceAddress,

    /// The acceleration structure to copy into.
    ///
    /// There is no default value.
    pub dst: &'a Arc<AccelerationStructure>,

    /// Additional operations to perform during the copy.
    ///
    /// The default value is [`CopyAccelerationStructureMode::Deserialize`].
    pub mode: CopyAccelerationStructureMode,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyMemoryToAccelerationStructureInfo<'a> {
    /// Returns a default `CopyMemoryToAccelerationStructureInfo` with the specified `src`.
    #[inline]
    pub const fn new(dst: &'a Arc<AccelerationStructure>) -> Self {
        Self {
            src: 0,
            dst,
            mode: CopyAccelerationStructureMode::Deserialize,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src: _,
            dst,
            mode,
            _ne: _,
        } = self;

        assert_eq!(device, dst.device().as_ref());

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode")
                .set_vuids(&["VUID-VkCopyMemoryToAccelerationStructureInfoKHR-mode-parameter"])
        })?;

        if !matches!(mode, CopyAccelerationStructureMode::Deserialize) {
            return Err(Box::new(ValidationError {
                context: "mode".into(),
                problem: "is not `CopyAccelerationStructureMode::Deserialize`".into(),
                vuids: &["VUID-VkCopyMemoryToAccelerationStructureInfoKHR-mode-03413"],
                ..Default::default()
            }));
        }

        // VUID-VkCopyMemoryToAccelerationStructureInfoKHR-src-04960
        // TODO: unsafe

        // VUID-VkCopyMemoryToAccelerationStructureInfoKHR-pInfo-03414
        // TODO: unsafe

        // VUID-VkCopyMemoryToAccelerationStructureInfoKHR-dst-03746
        // TODO: unsafe

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::CopyMemoryToAccelerationStructureInfoKHR<'static> {
        let &Self {
            src,
            dst,
            mode,
            _ne: _,
        } = self;

        vk::CopyMemoryToAccelerationStructureInfoKHR::default()
            .src(vk::DeviceOrHostAddressConstKHR {
                device_address: src,
            })
            .dst(dst.handle())
            .mode(mode.into())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// What mode an acceleration structure copy command should operate in.
    CopyAccelerationStructureMode = CopyAccelerationStructureModeKHR(i32);

    /// Copy the source into the destination.
    /// This is a shallow copy: if the source holds references to other acceleration structures,
    /// only the references are copied, not the other acceleration structures.
    ///
    /// Both source and destination must have been created with the same
    /// [`AccelerationStructureCreateInfo`].
    Clone = CLONE,

    /// Create a more compact version of the source in the destination.
    /// This is a shallow copy: if the source holds references to other acceleration structures,
    /// only the references are copied, not the other acceleration structures.
    ///
    /// The source acceleration structure must have been built with the
    /// [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`] flag.
    Compact = COMPACT,

    /// Serialize the acceleration structure into data in a semi-opaque format,
    /// that can be deserialized by a compatible Vulkan implementation.
    Serialize = SERIALIZE,

    /// Deserialize data back into an acceleration structure.
    Deserialize = DESERIALIZE,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Where the building of an acceleration structure will take place.
    AccelerationStructureBuildType = AccelerationStructureBuildTypeKHR(i32);

    /// Building will take place on the host.
    Host = HOST,

    /// Building will take place on the device.
    Device = DEVICE,

    /// Building will take place on either the host or the device.
    HostOrDevice = HOST_OR_DEVICE,
}

/// The minimum sizes needed for various resources during an acceleration structure build
/// operation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AccelerationStructureBuildSizesInfo {
    /// The minimum required size of the acceleration structure for a build or update operation.
    pub acceleration_structure_size: DeviceSize,

    /// The minimum required size of the scratch data buffer for an update operation.
    pub update_scratch_size: DeviceSize,

    /// The minimum required size of the scratch data buffer for a build operation.
    pub build_scratch_size: DeviceSize,
}

impl AccelerationStructureBuildSizesInfo {
    pub(crate) fn to_mut_vk() -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        vk::AccelerationStructureBuildSizesInfoKHR::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::AccelerationStructureBuildSizesInfoKHR<'_>) -> Self {
        let &vk::AccelerationStructureBuildSizesInfoKHR {
            acceleration_structure_size,
            update_scratch_size,
            build_scratch_size,
            ..
        } = val_vk;

        AccelerationStructureBuildSizesInfo {
            acceleration_structure_size,
            update_scratch_size,
            build_scratch_size,
        }
    }
}
