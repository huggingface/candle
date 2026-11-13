use crate::{
    command_buffer::{RecordingCommandBuffer, Result},
    Id,
};
use ash::vk;
use smallvec::SmallVec;
use vulkano::{
    buffer::Buffer,
    device::DeviceOwned,
    image::{Image, ImageLayout, ImageSubresourceRange},
    sync::{AccessFlags, DependencyFlags, PipelineStages},
    DeviceSize, Version, VulkanObject,
};

/// # Commands to synchronize resource accesses
impl RecordingCommandBuffer<'_> {
    #[track_caller]
    pub unsafe fn pipeline_barrier(&mut self, dependency_info: &DependencyInfo<'_>) -> &mut Self {
        unsafe { self.try_pipeline_barrier(dependency_info) }.unwrap()
    }

    pub unsafe fn try_pipeline_barrier(
        &mut self,
        dependency_info: &DependencyInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.pipeline_barrier_unchecked(dependency_info) })
    }

    pub unsafe fn pipeline_barrier_unchecked(
        &mut self,
        dependency_info: &DependencyInfo<'_>,
    ) -> &mut Self {
        if dependency_info.is_empty() {
            return self;
        }

        let &DependencyInfo {
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = dependency_info;

        if self.device().enabled_features().synchronization2 {
            let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                .iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        _ne: _,
                    } = barrier;

                    vk::MemoryBarrier2::default()
                        .src_stage_mask(src_stages.into())
                        .src_access_mask(src_access.into())
                        .dst_stage_mask(dst_stages.into())
                        .dst_access_mask(dst_access.into())
                })
                .collect();

            let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                .iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        buffer,
                        offset,
                        size,
                        _ne: _,
                    } = barrier;

                    let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(src_stages.into())
                        .src_access_mask(src_access.into())
                        .dst_stage_mask(dst_stages.into())
                        .dst_access_mask(dst_access.into())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(buffer.handle())
                        .offset(offset)
                        .size(size.unwrap_or(vk::WHOLE_SIZE))
                })
                .collect();

            let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                .iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        old_layout,
                        new_layout,
                        image,
                        subresource_range,
                        _ne: _,
                    } = barrier;

                    let image = unsafe { self.accesses.image_unchecked(image) };

                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src_stages.into())
                        .src_access_mask(src_access.into())
                        .dst_stage_mask(dst_stages.into())
                        .dst_access_mask(dst_access.into())
                        .old_layout(old_layout.into())
                        .new_layout(new_layout.into())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.handle())
                        .subresource_range(subresource_range.clone().to_vk())
                })
                .collect();

            let dependency_info_vk = vk::DependencyInfo::default()
                .dependency_flags(dependency_flags.into())
                .memory_barriers(&memory_barriers_vk)
                .buffer_memory_barriers(&buffer_memory_barriers_vk)
                .image_memory_barriers(&image_memory_barriers_vk);

            let fns = self.device().fns();
            let cmd_pipeline_barrier2 = if self.device().api_version() >= Version::V1_3 {
                fns.v1_3.cmd_pipeline_barrier2
            } else {
                fns.khr_synchronization2.cmd_pipeline_barrier2_khr
            };

            unsafe { cmd_pipeline_barrier2(self.handle(), &dependency_info_vk) };
        } else {
            let mut src_stage_mask = vk::PipelineStageFlags::empty();
            let mut dst_stage_mask = vk::PipelineStageFlags::empty();

            let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                .iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        _ne: _,
                    } = barrier;

                    src_stage_mask |= src_stages.into();
                    dst_stage_mask |= dst_stages.into();

                    vk::MemoryBarrier::default()
                        .src_access_mask(src_access.into())
                        .dst_access_mask(dst_access.into())
                })
                .collect();

            let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                .iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        buffer,
                        offset,
                        size,
                        _ne: _,
                    } = barrier;

                    src_stage_mask |= src_stages.into();
                    dst_stage_mask |= dst_stages.into();

                    let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(src_access.into())
                        .dst_access_mask(dst_access.into())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(buffer.handle())
                        .offset(offset)
                        .size(size.unwrap_or(vk::WHOLE_SIZE))
                })
                .collect();

            let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                .iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        old_layout,
                        new_layout,
                        image,
                        subresource_range,
                        _ne: _,
                    } = barrier;

                    src_stage_mask |= src_stages.into();
                    dst_stage_mask |= dst_stages.into();

                    let image = unsafe { self.accesses.image_unchecked(image) };

                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(src_access.into())
                        .dst_access_mask(dst_access.into())
                        .old_layout(old_layout.into())
                        .new_layout(new_layout.into())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.handle())
                        .subresource_range(subresource_range.clone().to_vk())
                })
                .collect();

            if src_stage_mask.is_empty() {
                // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
                // VK_PIPELINE_STAGE_2_NONE in the first scope."
                src_stage_mask |= vk::PipelineStageFlags::TOP_OF_PIPE;
            }

            if dst_stage_mask.is_empty() {
                // "VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is [...] equivalent to
                // VK_PIPELINE_STAGE_2_NONE in the second scope."
                dst_stage_mask |= vk::PipelineStageFlags::BOTTOM_OF_PIPE;
            }

            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.cmd_pipeline_barrier)(
                    self.handle(),
                    src_stage_mask,
                    dst_stage_mask,
                    dependency_flags.into(),
                    memory_barriers_vk.len() as u32,
                    memory_barriers_vk.as_ptr(),
                    buffer_memory_barriers_vk.len() as u32,
                    buffer_memory_barriers_vk.as_ptr(),
                    image_memory_barriers_vk.len() as u32,
                    image_memory_barriers_vk.as_ptr(),
                )
            };
        }

        self
    }
}

/// Dependency info for barriers in a pipeline barrier command.
///
/// A pipeline barrier creates a dependency between commands submitted before the barrier (the
/// source scope) and commands submitted after it (the destination scope). Each `DependencyInfo`
/// consists of multiple individual barriers that concern either a single resource or operate
/// globally.
///
/// Each barrier has a set of source/destination pipeline stages and source/destination memory
/// access types. The pipeline stages create an *execution dependency*: the `src_stages` of
/// commands submitted before the barrier must be completely finished before any of the
/// `dst_stages` of commands after the barrier are allowed to start. The memory access types create
/// a *memory dependency*: in addition to the execution dependency, any `src_access`
/// performed before the barrier must be made available and visible before any `dst_access`
/// are made after the barrier.
#[derive(Clone, Debug)]
pub struct DependencyInfo<'a> {
    /// Flags to modify how the execution and memory dependencies are formed.
    ///
    /// The default value is empty.
    pub dependency_flags: DependencyFlags,

    /// Memory barriers for global operations and accesses, not limited to a single resource.
    ///
    /// The default value is empty.
    pub memory_barriers: &'a [MemoryBarrier<'a>],

    /// Memory barriers for individual buffers.
    ///
    /// The default value is empty.
    pub buffer_memory_barriers: &'a [BufferMemoryBarrier<'a>],

    /// Memory barriers for individual images.
    ///
    /// The default value is empty.
    pub image_memory_barriers: &'a [ImageMemoryBarrier<'a>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for DependencyInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyInfo<'_> {
    /// Returns a default `DependencyInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            dependency_flags: DependencyFlags::empty(),
            memory_barriers: &[],
            buffer_memory_barriers: &[],
            image_memory_barriers: &[],
            _ne: crate::NE,
        }
    }

    /// Returns `true` if `self` doesn't contain any barriers.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.memory_barriers.is_empty()
            && self.buffer_memory_barriers.is_empty()
            && self.image_memory_barriers.is_empty()
    }
}

/// A memory barrier that is applied globally.
#[derive(Clone, Debug)]
pub struct MemoryBarrier<'a> {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub dst_access: AccessFlags,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for MemoryBarrier<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBarrier<'_> {
    /// Returns a default `MemoryBarrier`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            _ne: crate::NE,
        }
    }
}

/// A memory barrier that is applied to a single buffer.
#[derive(Clone, Debug)]
pub struct BufferMemoryBarrier<'a> {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub dst_access: AccessFlags,

    /// The buffer to apply the barrier to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub buffer: Id<Buffer>,

    /// The byte offset from `buffer` to apply the barrier to.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The byte size to apply the barrier to.
    ///
    /// If set to `None`, the barrier applies until the end of the buffer.
    ///
    /// The default value is `None`.
    pub size: Option<DeviceSize>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BufferMemoryBarrier<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BufferMemoryBarrier<'_> {
    /// Returns a default `BufferMemoryBarrier`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            buffer: Id::INVALID,
            offset: 0,
            size: None,
            _ne: crate::NE,
        }
    }
}

/// A memory barrier that is applied to a single image.
#[derive(Clone, Debug)]
pub struct ImageMemoryBarrier<'a> {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub dst_access: AccessFlags,

    /// The layout that the specified `subresource_range` of `image` is expected to be in when the
    /// source scope completes.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub old_layout: ImageLayout,

    /// The layout that the specified `subresource_range` of `image` will be transitioned to before
    /// the destination scope begins.
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub new_layout: ImageLayout,

    /// The image to apply the barrier to.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub image: Id<Image>,

    /// The subresource range of `image` to apply the barrier to.
    ///
    /// The default value is [`ImageSubresourceRange::default()`].
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ImageMemoryBarrier<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ImageMemoryBarrier<'_> {
    /// Returns a default `ImageMemoryBarrier`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::Undefined,
            image: Id::INVALID,
            subresource_range: ImageSubresourceRange::new(),
            _ne: crate::NE,
        }
    }
}
