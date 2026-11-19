use super::{DeviceMemory, MemoryPropertyFlags};
use crate::{
    buffer::{Buffer, BufferCreateFlags},
    device::{Device, DeviceOwned},
    image::{
        mip_level_extent, Image, ImageAspects, ImageCreateFlags, SparseImageFormatProperties,
        SparseImageMemoryRequirements,
    },
    memory::{is_aligned, MemoryRequirements},
    sync::semaphore::Semaphore,
    DeviceSize, ValidationError, VulkanObject as _,
};
use ash::vk;
use smallvec::SmallVec;

/// Parameters to execute sparse bind operations on a queue.
#[derive(Clone, Debug)]
pub struct BindSparseInfo<'a> {
    /// The semaphores to wait for before beginning the execution of this batch of
    /// sparse bind operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: &'a [&'a Semaphore],

    /// The bind operations to perform for buffers.
    ///
    /// The default value is empty.
    pub buffer_binds: &'a [SparseBufferMemoryBindInfo<'a>],

    /// The bind operations to perform for images with an opaque memory layout.
    ///
    /// This should be used for mip tail regions, the metadata aspect, and for the normal regions
    /// of images that do not have the `SPARSE_RESIDENCY` flag set.
    ///
    /// The default value is empty.
    pub image_opaque_binds: &'a [SparseImageOpaqueMemoryBindInfo<'a>],

    /// The bind operations to perform for images with a known memory layout.
    ///
    /// This type of sparse bind can only be used for images that have the `SPARSE_RESIDENCY`
    /// flag set.
    /// Only the normal texel regions can be bound this way, not the mip tail regions or metadata
    /// aspect.
    ///
    /// The default value is empty.
    pub image_binds: &'a [SparseImageMemoryBindInfo<'a>],

    /// The semaphores to signal after the execution of this batch of sparse bind operations
    /// has completed.
    ///
    /// The default value is empty.
    pub signal_semaphores: &'a [&'a Semaphore],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BindSparseInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BindSparseInfo<'_> {
    /// Returns a default `BindSparseInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            wait_semaphores: &[],
            buffer_binds: &[],
            image_opaque_binds: &[],
            image_binds: &[],
            signal_semaphores: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            wait_semaphores,
            buffer_binds,
            image_opaque_binds,
            image_binds,
            signal_semaphores,
            _ne: _,
        } = self;

        for semaphore in wait_semaphores {
            assert_eq!(device, semaphore.device().as_ref());
        }

        for (index, buffer_bind_info) in buffer_binds.iter().enumerate() {
            buffer_bind_info
                .validate(device)
                .map_err(|err| err.add_context(format!("buffer_binds[{}]", index)))?;
        }

        for (index, image_opaque_bind_info) in image_opaque_binds.iter().enumerate() {
            image_opaque_bind_info
                .validate(device)
                .map_err(|err| err.add_context(format!("image_opaque_binds[{}]", index)))?;
        }

        for (index, image_bind_info) in image_binds.iter().enumerate() {
            image_bind_info
                .validate(device)
                .map_err(|err| err.add_context(format!("image_binds[{}]", index)))?;
        }

        for semaphore in signal_semaphores {
            assert_eq!(device, semaphore.device().as_ref());
        }

        Ok(())
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a BindSparseInfoFields1Vk<'_>,
    ) -> vk::BindSparseInfo<'a> {
        let BindSparseInfoFields1Vk {
            wait_semaphores_vk,
            buffer_bind_infos_vk,
            image_opaque_bind_infos_vk,
            image_bind_infos_vk,
            signal_semaphores_vk,
        } = fields1_vk;

        vk::BindSparseInfo::default()
            .wait_semaphores(wait_semaphores_vk)
            .buffer_binds(buffer_bind_infos_vk)
            .image_opaque_binds(image_opaque_bind_infos_vk)
            .image_binds(image_bind_infos_vk)
            .signal_semaphores(signal_semaphores_vk)
    }

    pub(crate) fn to_vk_fields1<'a>(
        &'a self,
        fields2_vk: &'a BindSparseInfoFields2Vk,
    ) -> BindSparseInfoFields1Vk<'a> {
        let &Self {
            wait_semaphores,
            buffer_binds,
            image_opaque_binds,
            image_binds,
            signal_semaphores,
            _ne: _,
        } = self;
        let BindSparseInfoFields2Vk {
            buffer_bind_infos_fields1_vk,
            image_opaque_bind_infos_fields1_vk,
            image_bind_infos_fields1_vk,
        } = fields2_vk;

        let wait_semaphores_vk = wait_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect();

        let buffer_bind_infos_vk = buffer_binds
            .iter()
            .zip(buffer_bind_infos_fields1_vk)
            .map(|(buffer_bind_info, fields1_vk)| buffer_bind_info.to_vk(fields1_vk))
            .collect();

        let image_opaque_bind_infos_vk = image_opaque_binds
            .iter()
            .zip(image_opaque_bind_infos_fields1_vk)
            .map(|(image_opaque_bind_info, fields1_vk)| image_opaque_bind_info.to_vk(fields1_vk))
            .collect();

        let image_bind_infos_vk = image_binds
            .iter()
            .zip(image_bind_infos_fields1_vk)
            .map(|(image_bind_info, fields1_vk)| image_bind_info.to_vk(fields1_vk))
            .collect();

        let signal_semaphores_vk = signal_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect();

        BindSparseInfoFields1Vk {
            wait_semaphores_vk,
            buffer_bind_infos_vk,
            image_opaque_bind_infos_vk,
            image_bind_infos_vk,
            signal_semaphores_vk,
        }
    }

    pub(crate) fn to_vk_fields2(&self) -> BindSparseInfoFields2Vk {
        let &Self {
            wait_semaphores: _,
            buffer_binds,
            image_opaque_binds,
            image_binds,
            signal_semaphores: _,
            _ne: _,
        } = self;

        let buffer_bind_infos_fields1_vk = buffer_binds
            .iter()
            .map(SparseBufferMemoryBindInfo::to_vk_fields1)
            .collect();

        let image_opaque_bind_infos_fields1_vk = image_opaque_binds
            .iter()
            .map(SparseImageOpaqueMemoryBindInfo::to_vk_fields1)
            .collect();

        let image_bind_infos_fields1_vk = image_binds
            .iter()
            .map(SparseImageMemoryBindInfo::to_vk_fields1)
            .collect();

        BindSparseInfoFields2Vk {
            buffer_bind_infos_fields1_vk,
            image_opaque_bind_infos_fields1_vk,
            image_bind_infos_fields1_vk,
        }
    }
}

pub(crate) struct BindSparseInfoFields1Vk<'a> {
    pub(crate) wait_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
    pub(crate) buffer_bind_infos_vk: SmallVec<[vk::SparseBufferMemoryBindInfo<'a>; 4]>,
    pub(crate) image_opaque_bind_infos_vk: SmallVec<[vk::SparseImageOpaqueMemoryBindInfo<'a>; 4]>,
    pub(crate) image_bind_infos_vk: SmallVec<[vk::SparseImageMemoryBindInfo<'a>; 4]>,
    pub(crate) signal_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
}

pub(crate) struct BindSparseInfoFields2Vk {
    pub(crate) buffer_bind_infos_fields1_vk: SmallVec<[SparseBufferMemoryBindInfoFields1Vk; 4]>,
    pub(crate) image_opaque_bind_infos_fields1_vk:
        SmallVec<[SparseImageOpaqueMemoryBindInfoFields1Vk; 4]>,
    pub(crate) image_bind_infos_fields1_vk: SmallVec<[SparseImageMemoryBindInfoFields1Vk; 4]>,
}

/// Parameters for sparse bind operations on a buffer.
#[derive(Clone, Debug)]
pub struct SparseBufferMemoryBindInfo<'a> {
    /// The buffer to perform the binding operations on.
    ///
    /// There is no default value.
    pub buffer: &'a Buffer,

    /// The bind operations to perform.
    ///
    /// The default value is empty.
    pub binds: &'a [SparseBufferMemoryBind<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> SparseBufferMemoryBindInfo<'a> {
    /// Returns a default `SparseBufferMemoryBindInfo` with the provided `buffer`.
    #[inline]
    pub const fn new(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            binds: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            buffer,
            binds,
            _ne: _,
        } = self;

        assert_eq!(device, buffer.device().as_ref());
        assert!(!binds.is_empty());

        if !buffer.flags().intersects(BufferCreateFlags::SPARSE_BINDING) {
            return Err(Box::new(ValidationError {
                context: "buffer.flags()".into(),
                problem: "does not contain `BufferCreateFlags::SPARSE_BINDING`".into(),
                // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2253
                ..Default::default()
            }));
        }

        let &MemoryRequirements {
            layout,
            memory_type_bits,
            prefers_dedicated_allocation: _,
            requires_dedicated_allocation: _,
        } = buffer.memory_requirements();
        let external_memory_handle_types = buffer.external_memory_handle_types();

        for (index, bind) in binds.iter().enumerate() {
            bind.validate(device)
                .map_err(|err| err.add_context(format!("binds[{}]", index)))?;

            let &SparseBufferMemoryBind {
                offset,
                size,
                memory,
                _ne: _,
            } = bind;

            if offset >= layout.size() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].offset` is not less than \
                        `buffer.memory_requirements().layout.size()`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseMemoryBind-resourceOffset-01099"],
                    ..Default::default()
                }));
            }

            if size > layout.size() - offset {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset + binds[{0}].size` is greater than \
                        `buffer.memory_requirements().layout.size()`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseMemoryBind-size-01100"],
                    ..Default::default()
                }));
            }

            if !is_aligned(offset, layout.alignment()) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].offset` is not aligned according to \
                        `buffer.memory_requirements().layout.alignment()`",
                        index
                    )
                    .into(),
                    // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2255,
                    ..Default::default()
                }));
            }

            if !(size == layout.size() - offset || is_aligned(size, layout.alignment())) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset + binds[{0}].size` is not equal to \
                        `buffer.memory_requirements().layout.size()`, but is not aligned \
                        according to `buffer.memory_requirements().layout.alignment()`",
                        index
                    )
                    .into(),
                    // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2255,
                    ..Default::default()
                }));
            }

            if let Some((memory, memory_offset)) = memory {
                if !is_aligned(memory_offset, layout.alignment()) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].memory.1` is not aligned according to \
                            `buffer.memory_requirements().layout.alignment()`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseMemoryBind-memory-01096"],
                        ..Default::default()
                    }));
                }

                if memory_type_bits & (1 << memory.memory_type_index()) == 0 {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].memory.0.memory_type_index()` is not a bit set in \
                            `buffer.memory_requirements().memory_type_bits`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseMemoryBind-memory-01096"],
                        ..Default::default()
                    }));
                }

                if !memory.export_handle_types().is_empty() {
                    if !external_memory_handle_types.intersects(memory.export_handle_types()) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{}].memory.0.export_handle_types()` is not empty, but \
                                it does not share at least one memory type with \
                                `buffer.external_memory_handle_types()`",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkSparseMemoryBind-memory-02730"],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(handle_type) = memory.imported_handle_type() {
                    if !external_memory_handle_types.intersects(handle_type.into()) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{}].memory.0` is imported, but \
                                `buffer.external_memory_handle_types()` \
                                does not contain the imported handle type",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkSparseMemoryBind-memory-02731"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a SparseBufferMemoryBindInfoFields1Vk,
    ) -> vk::SparseBufferMemoryBindInfo<'a> {
        let &Self {
            buffer,
            binds: _,
            _ne: _,
        } = self;
        let SparseBufferMemoryBindInfoFields1Vk { binds_vk } = fields1_vk;

        vk::SparseBufferMemoryBindInfo::default()
            .buffer(buffer.handle())
            .binds(binds_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> SparseBufferMemoryBindInfoFields1Vk {
        let &Self {
            buffer: _,
            binds,
            _ne: _,
        } = self;

        let binds_vk = binds.iter().map(SparseBufferMemoryBind::to_vk).collect();

        SparseBufferMemoryBindInfoFields1Vk { binds_vk }
    }
}

pub(crate) struct SparseBufferMemoryBindInfoFields1Vk {
    pub(crate) binds_vk: SmallVec<[vk::SparseMemoryBind; 4]>,
}

/// Parameters for a single sparse bind operation on a buffer.
#[derive(Clone, Debug)]
pub struct SparseBufferMemoryBind<'a> {
    /// The offset in bytes from the start of the buffer's memory, where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The size in bytes of the memory to be (un)bound.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// If `Some`, specifies the memory and an offset into that memory that is to be bound.
    /// The provided memory must match the buffer's memory requirements.
    ///
    /// If `None`, specifies that existing memory at the specified location is to be unbound.
    ///
    /// The default value is `None`.
    pub memory: Option<(&'a DeviceMemory, DeviceSize)>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SparseBufferMemoryBind<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SparseBufferMemoryBind<'_> {
    /// Returns a default `SparseBufferMemoryBind`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            offset: 0,
            size: 0,
            memory: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            offset: _,
            size,
            memory,
            _ne: _,
        } = self;

        if size == 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSparseMemoryBind-size-01098"],
                ..Default::default()
            }));
        }

        if let Some((memory, memory_offset)) = memory {
            let memory_type = &device.physical_device().memory_properties().memory_types
                [memory.memory_type_index() as usize];

            if memory_type
                .property_flags
                .intersects(MemoryPropertyFlags::LAZILY_ALLOCATED)
            {
                return Err(Box::new(ValidationError {
                    problem: "`memory.0.memory_type_index()` refers to a memory type whose \
                        `property_flags` contains `MemoryPropertyFlags::LAZILY_ALLOCATED`"
                        .into(),
                    vuids: &["VUID-VkSparseMemoryBind-memory-01097"],
                    ..Default::default()
                }));
            }

            if memory_offset >= memory.allocation_size() {
                return Err(Box::new(ValidationError {
                    problem: "`memory.1` is not less than `memory.0.allocation_size()`".into(),
                    vuids: &["VUID-VkSparseMemoryBind-memoryOffset-01101"],
                    ..Default::default()
                }));
            }

            if size > memory.allocation_size() - memory_offset {
                return Err(Box::new(ValidationError {
                    problem: "`size` is greater than `memory.0.allocation_size()` minus \
                        `memory.1`"
                        .into(),
                    vuids: &["VUID-VkSparseMemoryBind-size-01102"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::SparseMemoryBind {
        let &Self {
            offset,
            size,
            memory,
            _ne: _,
        } = self;

        let (memory, memory_offset) = memory
            .as_ref()
            .map_or_else(Default::default, |(memory, memory_offset)| {
                (memory.handle(), *memory_offset)
            });

        vk::SparseMemoryBind {
            resource_offset: offset,
            size,
            memory,
            memory_offset,
            flags: vk::SparseMemoryBindFlags::empty(),
        }
    }
}

/// Parameters for sparse bind operations on parts of an image with an opaque memory layout.
///
/// This type of sparse bind should be used for mip tail regions, the metadata aspect, and for the
/// normal regions of images that do not have the `sparse_residency` flag set.
#[derive(Clone, Debug)]
pub struct SparseImageOpaqueMemoryBindInfo<'a> {
    /// The image to perform the binding operations on.
    ///
    /// There is no default value.
    pub image: &'a Image,

    /// The bind operations to perform.
    ///
    /// The default value is empty.
    pub binds: &'a [SparseImageOpaqueMemoryBind<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> SparseImageOpaqueMemoryBindInfo<'a> {
    /// Returns a default `SparseImageOpaqueMemoryBindInfo` with the provided `image`.
    #[inline]
    pub const fn new(image: &'a Image) -> Self {
        Self {
            image,
            binds: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            image,
            binds,
            _ne: _,
        } = self;

        assert_eq!(device, image.device().as_ref());
        assert!(!binds.is_empty());

        if !image.flags().intersects(ImageCreateFlags::SPARSE_BINDING) {
            return Err(Box::new(ValidationError {
                context: "image.flags()".into(),
                problem: "does not contain `ImageCreateFlags::SPARSE_BINDING`".into(),
                // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2253
                ..Default::default()
            }));
        }

        let &MemoryRequirements {
            layout,
            memory_type_bits,
            prefers_dedicated_allocation: _,
            requires_dedicated_allocation: _,
        } = &image.memory_requirements()[0]; // TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2259
        let metadata_memory_requirements = image.sparse_memory_requirements().iter().find(|reqs| {
            reqs.format_properties
                .aspects
                .intersects(ImageAspects::METADATA)
        });
        let external_memory_handle_types = image.external_memory_handle_types();

        for (index, bind) in binds.iter().enumerate() {
            bind.validate(device)
                .map_err(|err| err.add_context(format!("binds[{}]", index)))?;

            let &SparseImageOpaqueMemoryBind {
                offset,
                size,
                memory,
                metadata,
                _ne: _,
            } = bind;

            if metadata {
                // VkSparseMemoryBind spec:
                // If flags contains VK_SPARSE_MEMORY_BIND_METADATA_BIT,
                // the binding range must be within the mip tail region of the metadata aspect.
                // This metadata region is defined by:
                // metadataRegion = [base, base + imageMipTailSize)
                // base = imageMipTailOffset + imageMipTailStride × n

                let &SparseImageMemoryRequirements {
                    format_properties: _,
                    image_mip_tail_first_lod: _,
                    image_mip_tail_size,
                    image_mip_tail_offset,
                    image_mip_tail_stride,
                } = metadata_memory_requirements.ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].metadata` is `true`, but there is no \
                            `SparseImageMemoryRequirements` element in `image.memory()` where \
                            `format_properties.aspects` contains `ImageAspects::METADATA`",
                            index
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    })
                })?;

                let offset_from_mip_tail = offset.checked_sub(image_mip_tail_offset);

                if let Some(image_mip_tail_stride) = image_mip_tail_stride {
                    let (array_layer, offset_from_array_layer) = offset_from_mip_tail
                        .map(|offset_from_mip_tail| {
                            (
                                offset_from_mip_tail / image_mip_tail_stride,
                                offset_from_mip_tail % image_mip_tail_stride,
                            )
                        })
                        .filter(|&(array_layer, offset_from_array_layer)| {
                            array_layer < image.array_layers() as DeviceSize
                                && offset_from_array_layer < image_mip_tail_size
                        })
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                problem: format!(
                                    "`binds[{0}].metadata` is `true`, but `binds[{0}].offset` \
                                    does not fall within the metadata mip tail binding range \
                                    for any array layer of `image`",
                                    index
                                )
                                .into(),
                                vuids: &["VUID-VkSparseImageOpaqueMemoryBindInfo-pBinds-01103"],
                                ..Default::default()
                            })
                        })?;

                    if size > image_mip_tail_size - offset_from_array_layer {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{0}].metadata` is `true`, and `binds[{0}].offset` \
                                falls within the metadata mip tail binding range for \
                                array layer {1} of `image`, but \
                                `binds[{0}].offset + binds[{0}].size` is greater than the \
                                end of that binding range",
                                index, array_layer
                            )
                            .into(),
                            vuids: &["VUID-VkSparseImageOpaqueMemoryBindInfo-pBinds-01103"],
                            ..Default::default()
                        }));
                    }
                } else {
                    let offset_from_mip_tail = offset_from_mip_tail
                        .filter(|&offset_from_mip_tail| offset_from_mip_tail < image_mip_tail_size)
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                problem: format!(
                                    "`binds[{0}].metadata` is `true`, but `binds[{0}].offset` \
                                    does not fall within the metadata mip tail binding range \
                                    of `image`",
                                    index
                                )
                                .into(),
                                vuids: &["VUID-VkSparseImageOpaqueMemoryBindInfo-pBinds-01103"],
                                ..Default::default()
                            })
                        })?;

                    if size > image_mip_tail_size - offset_from_mip_tail {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{0}].metadata` is `true`, but \
                                `binds[{0}].offset + binds[{0}].size` is greater than \
                                the end of the metadata mip tail binding range of `image`",
                                index,
                            )
                            .into(),
                            vuids: &["VUID-VkSparseImageOpaqueMemoryBindInfo-pBinds-01103"],
                            ..Default::default()
                        }));
                    }
                }
            } else {
                // VkSparseMemoryBind spec:
                // If flags does not contain VK_SPARSE_MEMORY_BIND_METADATA_BIT,
                // the binding range must be within the range [0,VkMemoryRequirements::size).

                if offset >= layout.size() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].offset` is not less than \
                            `image.memory_requirements()[0].layout.size()`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseMemoryBind-resourceOffset-01099"],
                        ..Default::default()
                    }));
                }

                if size > layout.size() - offset {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{0}].offset + binds[{0}].size` is greater than \
                            `image.memory_requirements()[0].layout.size()`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseMemoryBind-size-01100"],
                        ..Default::default()
                    }));
                }
            }

            if !is_aligned(offset, layout.alignment()) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].offset` is not aligned according to \
                        `image.memory_requirements()[0].layout.alignment()`",
                        index
                    )
                    .into(),
                    // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2255,
                    ..Default::default()
                }));
            }

            if !(size == layout.size() - offset || is_aligned(size, layout.alignment())) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset + binds[{}].size` is not equal to \
                        `image.memory_requirements()[0].layout.size()`, but is not aligned \
                        according to `image.memory_requirements()[0].layout.alignment()`",
                        index
                    )
                    .into(),
                    // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2255,
                    ..Default::default()
                }));
            }

            if let Some((memory, memory_offset)) = memory {
                if !is_aligned(memory_offset, layout.alignment()) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].memory.1` is not aligned according to \
                            `image.memory_requirements()[0].layout.alignment()`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseMemoryBind-memory-01096"],
                        ..Default::default()
                    }));
                }

                if memory_type_bits & (1 << memory.memory_type_index()) == 0 {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].memory.0.memory_type_index()` is not a bit set in \
                            `image.memory_requirements()[0].memory_type_bits`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseMemoryBind-memory-01096"],
                        ..Default::default()
                    }));
                }

                if !memory.export_handle_types().is_empty() {
                    if !external_memory_handle_types.intersects(memory.export_handle_types()) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{}].memory.0.export_handle_types()` is not empty, but \
                                it does not share at least one memory type with \
                                `image.external_memory_handle_types()`",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkSparseMemoryBind-memory-02730"],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(handle_type) = memory.imported_handle_type() {
                    if !external_memory_handle_types.intersects(handle_type.into()) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{}].memory.0` is imported, but \
                                `image.external_memory_handle_types()` \
                                does not contain the imported handle type",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkSparseMemoryBind-memory-02731"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a SparseImageOpaqueMemoryBindInfoFields1Vk,
    ) -> vk::SparseImageOpaqueMemoryBindInfo<'a> {
        let &Self {
            image,
            binds: _,
            _ne: _,
        } = self;
        let SparseImageOpaqueMemoryBindInfoFields1Vk { binds_vk } = fields1_vk;

        vk::SparseImageOpaqueMemoryBindInfo::default()
            .image(image.handle())
            .binds(binds_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> SparseImageOpaqueMemoryBindInfoFields1Vk {
        let &Self {
            image: _,
            binds,
            _ne: _,
        } = self;

        let binds_vk = binds
            .iter()
            .map(SparseImageOpaqueMemoryBind::to_vk)
            .collect();

        SparseImageOpaqueMemoryBindInfoFields1Vk { binds_vk }
    }
}

pub(crate) struct SparseImageOpaqueMemoryBindInfoFields1Vk {
    pub(crate) binds_vk: SmallVec<[vk::SparseMemoryBind; 4]>,
}

/// Parameters for a single sparse bind operation on parts of an image with an opaque memory
/// layout.
///
/// This type of sparse bind should be used for mip tail regions, the metadata aspect, and for the
/// normal regions of images that do not have the `SPARSE_RESIDENCY` flag set.
#[derive(Clone, Debug)]
pub struct SparseImageOpaqueMemoryBind<'a> {
    /// The offset in bytes from the start of the image's memory, where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The size in bytes of the memory to be (un)bound.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// If `Some`, specifies the memory and an offset into that memory that is to be bound.
    /// The provided memory must match the image's memory requirements.
    ///
    /// If `None`, specifies that existing memory at the specified location is to be unbound.
    ///
    /// The default value is `None`.
    pub memory: Option<(&'a DeviceMemory, DeviceSize)>,

    /// Sets whether the binding should apply to the metadata aspect of the image, or to the
    /// normal texel data.
    ///
    /// The default value is `false`.
    pub metadata: bool,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SparseImageOpaqueMemoryBind<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SparseImageOpaqueMemoryBind<'_> {
    /// Returns a default `SparseImageOpaqueMemoryBind`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            offset: 0,
            size: 0,
            memory: None,
            metadata: false,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            offset: _,
            size,
            memory,
            metadata: _,
            _ne: _,
        } = self;

        if size == 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSparseMemoryBind-size-01098"],
                ..Default::default()
            }));
        }

        if let Some((memory, memory_offset)) = memory {
            let memory_type = &device.physical_device().memory_properties().memory_types
                [memory.memory_type_index() as usize];

            if memory_type
                .property_flags
                .intersects(MemoryPropertyFlags::LAZILY_ALLOCATED)
            {
                return Err(Box::new(ValidationError {
                    problem: "`memory.0.memory_type_index()` refers to a memory type whose \
                        `property_flags` contains `MemoryPropertyFlags::LAZILY_ALLOCATED`"
                        .into(),
                    vuids: &["VUID-VkSparseMemoryBind-memory-01097"],
                    ..Default::default()
                }));
            }

            if memory_offset >= memory.allocation_size() {
                return Err(Box::new(ValidationError {
                    problem: "`memory.1` is not less than `memory.0.allocation_size()`".into(),
                    vuids: &["VUID-VkSparseMemoryBind-memoryOffset-01101"],
                    ..Default::default()
                }));
            }

            if size > memory.allocation_size() - memory_offset {
                return Err(Box::new(ValidationError {
                    problem: "`size` is greater than `memory.0.allocation_size()` minus \
                        `memory.1`"
                        .into(),
                    vuids: &["VUID-VkSparseMemoryBind-size-01102"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::SparseMemoryBind {
        let &Self {
            offset,
            size,
            memory,
            metadata,
            _ne: _,
        } = self;

        let (memory, memory_offset) = memory
            .as_ref()
            .map_or_else(Default::default, |(memory, memory_offset)| {
                (memory.handle(), *memory_offset)
            });

        vk::SparseMemoryBind {
            resource_offset: offset,
            size,
            memory,
            memory_offset,
            flags: if metadata {
                vk::SparseMemoryBindFlags::METADATA
            } else {
                vk::SparseMemoryBindFlags::empty()
            },
        }
    }
}

/// Parameters for sparse bind operations on parts of an image with a known memory layout.
///
/// This type of sparse bind can only be used for images that have the `sparse_residency` flag set.
/// Only the normal texel regions can be bound this way, not the mip tail regions or metadata
/// aspect.
#[derive(Clone, Debug)]
pub struct SparseImageMemoryBindInfo<'a> {
    /// The image to perform the binding operations on.
    ///
    /// There is no default value.
    pub image: &'a Image,

    /// The bind operations to perform.
    ///
    /// The default value is empty.
    pub binds: &'a [SparseImageMemoryBind<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> SparseImageMemoryBindInfo<'a> {
    /// Returns a default `SparseImageMemoryBindInfo` with the provided `image`.
    #[inline]
    pub const fn new(image: &'a Image) -> Self {
        Self {
            image,
            binds: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            image,
            binds,
            _ne: _,
        } = self;

        assert_eq!(device, image.device().as_ref());
        assert!(!binds.is_empty());

        if !image.flags().intersects(ImageCreateFlags::SPARSE_BINDING) {
            return Err(Box::new(ValidationError {
                context: "image.flags()".into(),
                problem: "does not contain `ImageCreateFlags::SPARSE_BINDING`".into(),
                // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2253
                ..Default::default()
            }));
        }

        if !image.flags().intersects(ImageCreateFlags::SPARSE_RESIDENCY) {
            return Err(Box::new(ValidationError {
                context: "image.flags()".into(),
                problem: "does not contain `ImageCreateFlags::SPARSE_RESIDENCY`".into(),
                vuids: &["VUID-VkSparseImageMemoryBindInfo-image-02901"],
                ..Default::default()
            }));
        }

        let image_format_subsampled_extent = image
            .format()
            .ycbcr_chroma_sampling()
            .map_or(image.extent(), |s| s.subsampled_extent(image.extent()));
        let external_memory_handle_types = image.external_memory_handle_types();

        for (index, bind) in binds.iter().enumerate() {
            bind.validate(device)
                .map_err(|err| err.add_context(format!("binds[{}]", index)))?;

            let &SparseImageMemoryBind {
                aspects,
                mip_level,
                array_layer,
                offset,
                extent,
                memory,
                _ne: _,
            } = bind;

            if mip_level >= image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].mip_level` is not less than `image.mip_levels()`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-subresource-01106"],
                    ..Default::default()
                }));
            }

            if array_layer >= image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].array_layer` is not less than `image.array_layers()`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-subresource-01106"],
                    ..Default::default()
                }));
            }

            if !image.format().aspects().contains(aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].aspects` is not a subset of the aspects of the \
                        format of `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-subresource-01106"],
                    ..Default::default()
                }));
            }

            let &SparseImageMemoryRequirements {
                format_properties:
                    SparseImageFormatProperties {
                        aspects,
                        image_granularity,
                        flags: _,
                    },
                image_mip_tail_first_lod,
                image_mip_tail_size: _,
                image_mip_tail_offset: _,
                image_mip_tail_stride: _,
            } = image
                .sparse_memory_requirements()
                .iter()
                .find(|reqs| reqs.format_properties.aspects == aspects)
                .ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "there is no `SparseImageMemoryRequirements` element in \
                            `image.memory()` where `format_properties.aspects` equals \
                            binds[{}].aspects",
                            index
                        )
                        .into(),
                        // vuids: https://github.com/KhronosGroup/Vulkan-Docs/issues/2254
                        ..Default::default()
                    })
                })?;

            if mip_level >= image_mip_tail_first_lod {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].mip_level` is not less than \
                        `SparseImageMemoryRequirements::image_mip_tail_first_lod` for `image`",
                        index
                    )
                    .into(),
                    // vuids: TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/2258
                    ..Default::default()
                }));
            }

            if offset[0] % image_granularity[0] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].offset[0]` is not a multiple of \
                        `SparseImageMemoryRequirements::format_properties.image_granularity[0]` \
                        for `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-offset-01107"],
                    ..Default::default()
                }));
            }

            if offset[1] % image_granularity[1] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].offset[1]` is not a multiple of `
                        `SparseImageMemoryRequirements::format_properties.image_granularity[1]` \
                        for `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-offset-01109"],
                    ..Default::default()
                }));
            }

            if offset[2] % image_granularity[2] != 0 {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{}].offset[2]` is not a multiple of `
                        `SparseImageMemoryRequirements::format_properties.image_granularity[2]` \
                        for `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-offset-01111"],
                    ..Default::default()
                }));
            }

            let mut subresource_extent = mip_level_extent(image.extent(), mip_level).unwrap();

            // Only subsample if there are no other aspects.
            if aspects.intersects(ImageAspects::PLANE_1 | ImageAspects::PLANE_2)
                && (aspects - (ImageAspects::PLANE_1 | ImageAspects::PLANE_2)).is_empty()
            {
                subresource_extent = image_format_subsampled_extent;
            }

            if offset[0]
                .checked_add(extent[0])
                .is_none_or(|sum| sum > subresource_extent[0])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset[0]` + `binds[{0}].extent[0]` is greater than the \
                        width of the selected subresource of `image`",
                        index
                    )
                    .into(),
                    // VUID? See https://github.com/KhronosGroup/Vulkan-Docs/issues/2490
                    ..Default::default()
                }));
            }

            if offset[1]
                .checked_add(extent[1])
                .is_none_or(|sum| sum > subresource_extent[1])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset[1]` + `binds[{0}].extent[1]` is greater than the \
                        height of the selected subresource of `image`",
                        index
                    )
                    .into(),
                    // VUID? See https://github.com/KhronosGroup/Vulkan-Docs/issues/2490
                    ..Default::default()
                }));
            }

            if offset[2]
                .checked_add(extent[2])
                .is_none_or(|sum| sum > subresource_extent[2])
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset[2]` + `binds[{0}].extent[2]` is greater than the \
                        depth of the selected subresource of `image`",
                        index
                    )
                    .into(),
                    // VUID? See https://github.com/KhronosGroup/Vulkan-Docs/issues/2490
                    ..Default::default()
                }));
            }

            if !(offset[0] + extent[0] == subresource_extent[0]
                || extent[0] % image_granularity[0] == 0)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset[0]` + `binds[{0}].extent[0]` is not equal to the \
                        width of the selected subresource of `image`, but it is not a multiple of \
                        `SparseImageMemoryRequirements::format_properties.image_granularity[0]` \
                        for `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-extent-01108"],
                    ..Default::default()
                }));
            }

            if !(offset[1] + extent[1] == subresource_extent[1]
                || extent[1] % image_granularity[1] == 0)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset[1]` + `binds[{0}].extent[1]` is not equal to the \
                        height of the selected subresource of `image`, but it is not a multiple of \
                        `SparseImageMemoryRequirements::format_properties.image_granularity[1]` \
                        for `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-extent-01110"],
                    ..Default::default()
                }));
            }

            if !(offset[2] + extent[2] == subresource_extent[2]
                || extent[2] % image_granularity[2] == 0)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`binds[{0}].offset[2]` + `binds[{0}].extent[2]` is not equal to the \
                        depth of the selected subresource of `image`, but it is not a multiple of \
                        `SparseImageMemoryRequirements::format_properties.image_granularity[2]` \
                        for `image`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkSparseImageMemoryBind-extent-01112"],
                    ..Default::default()
                }));
            }

            if let Some((memory, memory_offset)) = memory {
                let &MemoryRequirements {
                    layout,
                    memory_type_bits,
                    prefers_dedicated_allocation: _,
                    requires_dedicated_allocation: _,
                } = &image.memory_requirements()[0]; // TODO: what to do about disjoint images?

                if !is_aligned(memory_offset, layout.alignment()) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].memory.1` is not aligned according to \
                            `image.memory_requirements()[0].layout.alignment()`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseImageMemoryBind-memory-01105"],
                        ..Default::default()
                    }));
                }

                if memory_type_bits & (1 << memory.memory_type_index()) == 0 {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`binds[{}].memory.0.memory_type_index()` is not a bit set in \
                            `image.memory_requirements()[0].memory_type_bits`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSparseImageMemoryBind-memory-01105"],
                        ..Default::default()
                    }));
                }

                if !memory.export_handle_types().is_empty() {
                    if !external_memory_handle_types.intersects(memory.export_handle_types()) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{}].memory.0.export_handle_types()` is not empty, but \
                                it does not share at least one memory type with \
                                `image.external_memory_handle_types()`",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkSparseImageMemoryBind-memory-02732"],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(handle_type) = memory.imported_handle_type() {
                    if !external_memory_handle_types.intersects(handle_type.into()) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`binds[{}].memory.0` is imported, but \
                                `image.external_memory_handle_types()` \
                                does not contain the imported handle type",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkSparseImageMemoryBind-memory-02733"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a SparseImageMemoryBindInfoFields1Vk,
    ) -> vk::SparseImageMemoryBindInfo<'a> {
        let &Self {
            image,
            binds: _,
            _ne: _,
        } = self;
        let SparseImageMemoryBindInfoFields1Vk { binds_vk } = fields1_vk;

        vk::SparseImageMemoryBindInfo::default()
            .image(image.handle())
            .binds(binds_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> SparseImageMemoryBindInfoFields1Vk {
        let &Self {
            image: _,
            binds,
            _ne: _,
        } = self;

        let binds_vk = binds.iter().map(SparseImageMemoryBind::to_vk).collect();

        SparseImageMemoryBindInfoFields1Vk { binds_vk }
    }
}

pub(crate) struct SparseImageMemoryBindInfoFields1Vk {
    pub(crate) binds_vk: SmallVec<[vk::SparseImageMemoryBind; 4]>,
}

/// Parameters for a single sparse bind operation on parts of an image with a known memory layout.
///
/// This type of sparse bind can only be used for images that have the `SPARSE_RESIDENCY` flag set.
/// Only the normal texel regions can be bound this way, not the mip tail regions or metadata
/// aspect.
#[derive(Clone, Debug)]
pub struct SparseImageMemoryBind<'a> {
    /// The aspects of the image where memory is to be (un)bound.
    ///
    /// The default value is `ImageAspects::empty()`, which must be overridden.
    pub aspects: ImageAspects,

    /// The mip level of the image where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub mip_level: u32,

    /// The array layer of the image where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub array_layer: u32,

    /// The offset in texels (or for compressed images, texel blocks) from the origin of the image,
    /// where memory is to be (un)bound.
    ///
    /// This must be a multiple of the
    /// [`SparseImageFormatProperties::image_granularity`](crate::image::SparseImageFormatProperties::image_granularity)
    /// value of the image.
    ///
    /// The default value is `[0; 3]`.
    pub offset: [u32; 3],

    /// The extent in texels (or for compressed images, texel blocks) of the image where
    /// memory is to be (un)bound.
    ///
    /// This must be a multiple of the
    /// [`SparseImageFormatProperties::image_granularity`](crate::image::SparseImageFormatProperties::image_granularity)
    /// value of the image, or `offset + extent` for that dimension must equal the image's total
    /// extent.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    /// If `Some`, specifies the memory and an offset into that memory that is to be bound.
    /// The provided memory must match the image's memory requirements.
    ///
    /// If `None`, specifies that existing memory at the specified location is to be unbound.
    ///
    /// The default value is `None`.
    pub memory: Option<(&'a DeviceMemory, DeviceSize)>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SparseImageMemoryBind<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SparseImageMemoryBind<'_> {
    /// Returns a default `SparseImageMemoryBind`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            aspects: ImageAspects::empty(),
            mip_level: 0,
            array_layer: 0,
            offset: [0; 3],
            extent: [0; 3],
            memory: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            aspects,
            mip_level: _,
            array_layer: _,
            offset: _,
            extent,
            memory,
            _ne: _,
        } = self;

        aspects.validate_device(device).map_err(|err| {
            err.add_context("aspects")
                .set_vuids(&["VUID-VkImageSubresource-aspectMask-parameter"])
        })?;

        if aspects.is_empty() {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageSubresource-aspectMask-requiredbitmask"],
                ..Default::default()
            }));
        }

        if let Some((memory, memory_offset)) = memory {
            if memory_offset >= memory.allocation_size() {
                return Err(Box::new(ValidationError {
                    problem: "`memory.1` is not less than `memory.0.allocation_size()`".into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        }

        if extent[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSparseImageMemoryBind-extent-09388"],
                ..Default::default()
            }));
        }

        if extent[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSparseImageMemoryBind-extent-09389"],
                ..Default::default()
            }));
        }

        if extent[2] == 0 {
            return Err(Box::new(ValidationError {
                context: "extent[2]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSparseImageMemoryBind-extent-09390"],
                ..Default::default()
            }));
        }

        // VUID-VkSparseImageMemoryBind-memory-01104
        // If the sparseResidencyAliased feature is not enabled, and if any other resources are
        // bound to ranges of memory, the range of memory being bound must not overlap with
        // those bound ranges

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::SparseImageMemoryBind {
        let &Self {
            aspects,
            mip_level,
            array_layer,
            offset,
            extent,
            memory,
            _ne: _,
        } = self;

        let (memory, memory_offset) = memory
            .as_ref()
            .map_or_else(Default::default, |(memory, memory_offset)| {
                (memory.handle(), *memory_offset)
            });

        vk::SparseImageMemoryBind {
            subresource: vk::ImageSubresource {
                aspect_mask: aspects.into(),
                mip_level,
                array_layer,
            },
            offset: vk::Offset3D {
                x: offset[0] as i32,
                y: offset[1] as i32,
                z: offset[2] as i32,
            },
            extent: vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            },
            memory,
            memory_offset,
            flags: vk::SparseMemoryBindFlags::empty(),
        }
    }
}
