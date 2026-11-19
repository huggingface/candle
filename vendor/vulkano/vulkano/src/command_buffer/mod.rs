//! Recording commands to execute on the device.
//!
//! With Vulkan, to get the device to perform work, even relatively simple tasks, you must create a
//! command buffer. A command buffer is a list of commands that will executed by the device.
//! You must first record commands to a recording command buffer, then end the recording to turn it
//! into an actual command buffer, and then it can be used. Depending on how a command buffer is
//! created, it can be used only once, or reused many times.
//!
//! # Command pools and allocators
//!
//! Command buffers are allocated from *command pools*. A command pool holds memory that is used to
//! record the sequence of commands in a command buffer. Command pools are not thread-safe, and
//! therefore commands cannot be recorded to a single command buffer from multiple threads at a
//! time.
//!
//! Raw command pools are unsafe to use, so Vulkano uses [command buffer allocators] to manage
//! command buffers and pools, to ensure their memory is used efficiently, and to protect against
//! invalid usage. Vulkano provides the [`StandardCommandBufferAllocator`] for this purpose, but
//! you can also create your own by implementing the [`CommandBufferAllocator`] trait.
//!
//! # Primary and secondary command buffers
//!
//! There are two levels of command buffers:
//!
//! - A primary command buffer can be executed on a queue, and is the main command buffer level. It
//!   cannot be executed within another command buffer.
//! - A secondary command buffer can only be executed within a primary command buffer, not directly
//!   on a queue.
//!
//! Using secondary command buffers, there is slightly more overhead than using primary command
//! buffers alone, but there are also advantages. A single command buffer cannot be recorded
//! from multiple threads at a time, so if you want to divide the recording work among several
//! threads, each thread must record its own command buffer. While it is possible for these to be
//! all primary command buffers, there are limitations: a render pass or query cannot span multiple
//! primary command buffers, while secondary command buffers can [inherit] this state from their
//! parent primary command buffer. Therefore, to have a single render pass or query that is shared
//! across all the command buffers, you must record secondary command buffers.
//!
//! # Recording a command buffer
//!
//! To record a new command buffer, the most direct way is to create a new
//! [`AutoCommandBufferBuilder`]. You can then call methods on this object to record new commands
//! to the command buffer. When you are done recording, you call [`build`] to finalize the command
//! buffer and turn it into either a [`PrimaryAutoCommandBuffer`] or a
//! [`SecondaryAutoCommandBuffer`].
//!
//! # Submitting a primary command buffer
//!
//! Once a primary command buffer is recorded and built, you can submit the
//! [`PrimaryAutoCommandBuffer`] to a queue. Submitting a command buffer returns an object that
//! implements the [`GpuFuture`] trait and that represents the moment when the execution will end
//! on the GPU.
//!
//! ```
//! use vulkano::command_buffer::{
//!     AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
//!     SubpassContents,
//! };
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! # let queue: std::sync::Arc<vulkano::device::Queue> = return;
//! # let vertex_buffer: vulkano::buffer::Subbuffer<[u32]> = return;
//! # let render_pass_begin_info: vulkano::command_buffer::RenderPassBeginInfo = return;
//! # let graphics_pipeline: std::sync::Arc<vulkano::pipeline::graphics::GraphicsPipeline> = return;
//! # let command_buffer_allocator: std::sync::Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator> = return;
//! #
//! let mut cb = AutoCommandBufferBuilder::primary(
//!     command_buffer_allocator.clone(),
//!     queue.queue_family_index(),
//!     CommandBufferUsage::OneTimeSubmit,
//! )
//! .unwrap();
//!
//! cb.begin_render_pass(render_pass_begin_info, Default::default())
//!     .unwrap()
//!     .bind_pipeline_graphics(graphics_pipeline.clone())
//!     .unwrap()
//!     .bind_vertex_buffers(0, vertex_buffer.clone())
//!     .unwrap();
//! unsafe { cb.draw(vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();
//!
//! cb.end_render_pass(Default::default()).unwrap();
//!
//! let cb = cb.build().unwrap();
//!
//! let future = cb.execute(queue.clone());
//! ```
//!
//! [`StandardCommandBufferAllocator`]: allocator::StandardCommandBufferAllocator
//! [`CommandBufferAllocator`]: allocator::CommandBufferAllocator
//! [inherit]: CommandBufferInheritanceInfo
//! [`build`]: AutoCommandBufferBuilder::build
//! [`GpuFuture`]: crate::sync::GpuFuture

#[allow(unused_imports)] // everything is exported for future-proofing
pub use self::auto::commands::{
    acceleration_structure::*, clear::*, copy::*, debug::*, dynamic_state::*, pipeline::*,
    query::*, render_pass::*, secondary::*,
};
pub use self::{
    auto::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer},
    sys::{CommandBuffer, CommandBufferBeginInfo, RecordingCommandBuffer},
    traits::{
        CommandBufferExecError, CommandBufferExecFuture, PrimaryCommandBufferAbstract,
        SecondaryCommandBufferAbstract,
    },
};
use crate::{
    buffer::{Buffer, Subbuffer},
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, SampleCount},
    macros::vulkan_enum,
    query::{QueryControlFlags, QueryPipelineStatisticFlags},
    range_map::RangeMap,
    render_pass::{Framebuffer, Subpass},
    sync::{
        semaphore::{Semaphore, SemaphoreType},
        PipelineStageAccessFlags, PipelineStages,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
#[cfg(doc)]
use crate::{
    device::{DeviceFeatures, DeviceProperties},
    pipeline::graphics::vertex_input::VertexInputRate,
};
use ash::vk;
use ash::vk::TaggedStructure;

use bytemuck::{Pod, Zeroable};
use foldhash::HashMap;
use smallvec::SmallVec;
use std::{ops::Range, sync::Arc};

pub mod allocator;
pub mod auto;
mod commands;
pub mod pool;
mod sys;
mod traits;

pub mod raw {
    #[allow(unused_imports)] // everything is exported for future-proofing
    pub use crate::command_buffer::commands::{
        acceleration_structure::*, clear::*, copy::*, debug::*, dynamic_state::*, pipeline::*,
        query::*, render_pass::*, secondary::*, sync::*,
    };
}

/// Used as buffer contents to provide input for the
/// [`AutoCommandBufferBuilder::dispatch_indirect`] command.
///
/// # Safety
///
/// - The `x`, `y` and `z` values must not be greater than the respective elements of the
///   [`max_compute_work_group_count`](DeviceProperties::max_compute_work_group_count) device
///   limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

unsafe impl Pod for DispatchIndirectCommand {}
unsafe impl Zeroable for DispatchIndirectCommand {}

/// Used as buffer contents to provide input for the
/// [`AutoCommandBufferBuilder::draw_indirect`] command.
///
/// # Safety
///
/// - Every vertex number within the specified range must fall within the range of the bound
///   vertex-rate vertex buffers.
/// - Every instance number within the specified range must fall within the range of the bound
///   instance-rate vertex buffers.
/// - If the [`draw_indirect_first_instance`](DeviceFeatures::draw_indirect_first_instance) feature
///   is not enabled, then `first_instance` must be `0`.
/// - If an [instance divisor](VertexInputRate::Instance) other than 1 is used, and the
///   [`supports_non_zero_first_instance`](DeviceProperties::supports_non_zero_first_instance)
///   device property is `false`, then `first_instance` must be `0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

unsafe impl Pod for DrawIndirectCommand {}
unsafe impl Zeroable for DrawIndirectCommand {}

/// Used as buffer contents to provide input for the
/// [`AutoCommandBufferBuilder::draw_mesh_tasks_indirect`] command.
///
/// # Safety
///
/// - If the graphics pipeline **does not** include a task shader, then the `group_count_x`,
///   `group_count_y` and `group_count_z` values must not be greater than the respective elements
///   of the [`max_mesh_work_group_count`](DeviceProperties::max_mesh_work_group_count) device
///   limit, and the product of these three values must not be greater than the
///   [`max_mesh_work_group_total_count`](DeviceProperties::max_mesh_work_group_total_count) device
///   limit.
/// - If the graphics pipeline **does** include a task shader, then the `group_count_x`,
///   `group_count_y` and `group_count_z` values must not be greater than the respective elements
///   of the [`max_task_work_group_count`](DeviceProperties::max_task_work_group_count) device
///   limit, and the product of these three values must not be greater than the
///   [`max_task_work_group_total_count`](DeviceProperties::max_task_work_group_total_count) device
///   limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawMeshTasksIndirectCommand {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

unsafe impl Pod for DrawMeshTasksIndirectCommand {}
unsafe impl Zeroable for DrawMeshTasksIndirectCommand {}

/// Used as buffer contents to provide input for the
/// [`AutoCommandBufferBuilder::draw_indexed_indirect`] command.
///
/// # Safety
///
/// - Every index within the specified range must fall within the range of the bound index buffer.
/// - Every vertex number that is retrieved from the index buffer must fall within the range of the
///   bound vertex-rate vertex buffers.
/// - Every vertex number that is retrieved from the index buffer, if it is not the special
///   primitive restart value, must be no greater than the
///   [`max_draw_indexed_index_value`](DeviceProperties::max_draw_indexed_index_value) device
///   limit.
/// - Every instance number within the specified range must fall within the range of the bound
///   instance-rate vertex buffers.
/// - If the [`draw_indirect_first_instance`](DeviceFeatures::draw_indirect_first_instance) feature
///   is not enabled, then `first_instance` must be `0`.
/// - If an [instance divisor](VertexInputRate::Instance) other than 1 is used, and the
///   [`supports_non_zero_first_instance`](DeviceProperties::supports_non_zero_first_instance)
///   device property is `false`, then `first_instance` must be `0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: u32,
    pub first_instance: u32,
}

unsafe impl Pod for DrawIndexedIndirectCommand {}
unsafe impl Zeroable for DrawIndexedIndirectCommand {}

/// Used as buffer contents to provide input for the
/// [`AutoCommandBufferBuilder::trace_rays_indirect`] command.
///
/// # Safety
///
/// - The `width`, `height` and `depth` values must not exceed the product of the respective
///   [`max_compute_work_group_count`](DeviceProperties::max_compute_work_group_count) elements and
///   [`max_compute_work_group_size`](DeviceProperties::max_compute_work_group_size) elements.
/// - `width` * `height` * `depth` must not exceed
///   [`max_ray_dispatch_invocation_count`](RayTracingPipelineProperties::max_ray_dispatch_invocation_count).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceRaysIndirectCommand {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

unsafe impl Pod for TraceRaysIndirectCommand {}
unsafe impl Zeroable for TraceRaysIndirectCommand {}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what a subpass in a command buffer will contain.
    SubpassContents = SubpassContents(i32);

    /// The subpass will only directly contain commands.
    Inline = INLINE,

    /// The subpass will only contain secondary command buffers invocations.
    SecondaryCommandBuffers = SECONDARY_COMMAND_BUFFERS,
}

impl From<SubpassContents> for vk::RenderingFlags {
    #[inline]
    fn from(val: SubpassContents) -> Self {
        match val {
            SubpassContents::Inline => Self::empty(),
            SubpassContents::SecondaryCommandBuffers => Self::CONTENTS_SECONDARY_COMMAND_BUFFERS,
        }
    }
}

vulkan_enum! {
    /// Determines the kind of command buffer to create.
    CommandBufferLevel = CommandBufferLevel(i32);

    /// Primary command buffers can be executed on a queue, and can call secondary command buffers.
    /// Render passes must begin and end within the same primary command buffer.
    Primary = PRIMARY,

    /// Secondary command buffers cannot be executed on a queue, but can be executed by a primary
    /// command buffer. If created for a render pass, they must fit within a single render subpass.
    Secondary = SECONDARY,
}

/// The context that a secondary command buffer can inherit from the primary command
/// buffer it's executed in.
#[derive(Clone, Debug)]
pub struct CommandBufferInheritanceInfo {
    /// If `Some`, the secondary command buffer is required to be executed within a render pass
    /// instance, and can only call draw operations.
    /// If `None`, it must be executed outside a render pass instance, and can execute dispatch and
    /// transfer operations, but not drawing operations.
    ///
    /// The default value is `None`.
    pub render_pass: Option<CommandBufferInheritanceRenderPassType>,

    /// If `Some`, the secondary command buffer is allowed to be executed within a primary that has
    /// an occlusion query active. The inner `QueryControlFlags` specifies which flags the
    /// active occlusion is allowed to have enabled.
    /// If `None`, the primary command buffer cannot have an occlusion query active when this
    /// secondary command buffer is executed.
    ///
    /// The `inherited_queries` feature must be enabled if this is `Some`.
    ///
    /// The default value is `None`.
    pub occlusion_query: Option<QueryControlFlags>,

    /// Which `PipelineStatistics` queries are allowed to be active on the primary command buffer
    /// when this secondary command buffer is executed.
    ///
    /// If this value is not empty, the [`pipeline_statistics_query`] feature must be enabled on
    /// the device.
    ///
    /// The default value is [`QueryPipelineStatisticFlags::empty()`].
    ///
    /// [`pipeline_statistics_query`]: crate::device::DeviceFeatures::pipeline_statistics_query
    pub pipeline_statistics: QueryPipelineStatisticFlags,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for CommandBufferInheritanceInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CommandBufferInheritanceInfo {
    /// Returns a default `CommandBufferInheritanceInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            render_pass: None,
            occlusion_query: None,
            pipeline_statistics: QueryPipelineStatisticFlags::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref render_pass,
            occlusion_query,
            pipeline_statistics,
            _ne: _,
        } = self;

        if let Some(render_pass) = render_pass {
            // VUID-VkCommandBufferBeginInfo-flags-06000
            // VUID-VkCommandBufferBeginInfo-flags-06002
            // Ensured by the definition of the `CommandBufferInheritanceRenderPassType` enum.

            match render_pass {
                CommandBufferInheritanceRenderPassType::BeginRenderPass(render_pass_info) => {
                    render_pass_info
                        .validate(device)
                        .map_err(|err| err.add_context("render_pass"))?;
                }
                CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                    rendering_info
                        .validate(device)
                        .map_err(|err| err.add_context("render_pass"))?;
                }
            }
        }

        if let Some(control_flags) = occlusion_query {
            control_flags.validate_device(device).map_err(|err| {
                err.add_context("occlusion_query")
                    .set_vuids(&["VUID-VkCommandBufferInheritanceInfo-queryFlags-00057"])
            })?;

            if !device.enabled_features().inherited_queries {
                return Err(Box::new(ValidationError {
                    context: "occlusion_query".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "inherited_queries",
                    )])]),
                    vuids: &["VUID-VkCommandBufferInheritanceInfo-occlusionQueryEnable-00056"],
                }));
            }

            if control_flags.intersects(QueryControlFlags::PRECISE)
                && !device.enabled_features().occlusion_query_precise
            {
                return Err(Box::new(ValidationError {
                    context: "occlusion_query".into(),
                    problem: "contains `QueryControlFlags::PRECISE`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "occlusion_query_precise",
                    )])]),
                    vuids: &["VUID-vkBeginCommandBuffer-commandBuffer-00052"],
                }));
            }
        }

        pipeline_statistics.validate_device(device).map_err(|err| {
            err.add_context("pipeline_statistics")
                .set_vuids(&["VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-02789"])
        })?;

        if pipeline_statistics.count() > 0 && !device.enabled_features().pipeline_statistics_query {
            return Err(Box::new(ValidationError {
                context: "pipeline_statistics".into(),
                problem: "is not empty".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "pipeline_statistics_query",
                )])]),
                vuids: &["VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-00058"],
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk<'a>(
        &self,
        extensions_vk: &'a mut CommandBufferInheritanceInfoExtensionsVk<'_>,
    ) -> vk::CommandBufferInheritanceInfo<'a> {
        let &Self {
            ref render_pass,
            occlusion_query,
            pipeline_statistics,
            _ne: _,
        } = self;

        let (render_pass_vk, subpass_vk, framebuffer_vk) = render_pass
            .as_ref()
            .and_then(|render_pass| match render_pass {
                CommandBufferInheritanceRenderPassType::BeginRenderPass(render_pass_info) => {
                    let CommandBufferInheritanceRenderPassInfo {
                        subpass,
                        framebuffer,
                    } = render_pass_info;

                    Some((
                        subpass.render_pass().handle(),
                        subpass.index(),
                        framebuffer
                            .as_ref()
                            .map(|fb| fb.handle())
                            .unwrap_or_default(),
                    ))
                }
                CommandBufferInheritanceRenderPassType::BeginRendering(_) => None,
            })
            .unwrap_or_default();

        let (occlusion_query_enable, query_flags_vk) = occlusion_query
            .map(|flags| (true, flags.into()))
            .unwrap_or_default();

        let mut val_vk = vk::CommandBufferInheritanceInfo::default()
            .render_pass(render_pass_vk)
            .subpass(subpass_vk)
            .framebuffer(framebuffer_vk)
            .occlusion_query_enable(occlusion_query_enable)
            .query_flags(query_flags_vk)
            .pipeline_statistics(pipeline_statistics.into());

        let CommandBufferInheritanceInfoExtensionsVk {
            rendering_info_vk: rendering_vk,
        } = extensions_vk;

        if let Some(next) = rendering_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions<'a>(
        &self,
        fields1_vk: &'a CommandBufferInheritanceInfoFields1Vk,
    ) -> CommandBufferInheritanceInfoExtensionsVk<'a> {
        let CommandBufferInheritanceInfoFields1Vk {
            rendering_info_fields1_vk,
        } = fields1_vk;

        let rendering_info_vk = self
            .render_pass
            .as_ref()
            .zip(rendering_info_fields1_vk.as_ref())
            .and_then(
                |(render_pass, rendering_info_fields1_vk)| match render_pass {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(_) => None,
                    CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                        Some(rendering_info.to_vk(rendering_info_fields1_vk))
                    }
                },
            );

        CommandBufferInheritanceInfoExtensionsVk { rendering_info_vk }
    }

    pub(crate) fn to_vk_fields1(&self) -> CommandBufferInheritanceInfoFields1Vk {
        let rendering_info_fields1_vk =
            self.render_pass
                .as_ref()
                .and_then(|render_pass| match render_pass {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(_) => None,
                    CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                        Some(rendering_info.to_vk_fields1())
                    }
                });

        CommandBufferInheritanceInfoFields1Vk {
            rendering_info_fields1_vk,
        }
    }
}

pub(crate) struct CommandBufferInheritanceInfoExtensionsVk<'a> {
    pub(crate) rendering_info_vk: Option<vk::CommandBufferInheritanceRenderingInfo<'a>>,
}

pub(crate) struct CommandBufferInheritanceInfoFields1Vk {
    pub(crate) rendering_info_fields1_vk: Option<CommandBufferInheritanceRenderingInfoFields1Vk>,
}

/// Selects the type of render pass for command buffer inheritance.
#[derive(Clone, Debug)]
pub enum CommandBufferInheritanceRenderPassType {
    /// The secondary command buffer will be executed within a render pass begun with
    /// `begin_render_pass`, using a `RenderPass` object and `Framebuffer`.
    BeginRenderPass(CommandBufferInheritanceRenderPassInfo),

    /// The secondary command buffer will be executed within a render pass begun with
    /// `begin_rendering`, using dynamic rendering.
    BeginRendering(CommandBufferInheritanceRenderingInfo),
}

impl From<Subpass> for CommandBufferInheritanceRenderPassType {
    #[inline]
    fn from(val: Subpass) -> Self {
        Self::BeginRenderPass(val.into())
    }
}

impl From<CommandBufferInheritanceRenderPassInfo> for CommandBufferInheritanceRenderPassType {
    #[inline]
    fn from(val: CommandBufferInheritanceRenderPassInfo) -> Self {
        Self::BeginRenderPass(val)
    }
}

impl From<CommandBufferInheritanceRenderingInfo> for CommandBufferInheritanceRenderPassType {
    #[inline]
    fn from(val: CommandBufferInheritanceRenderingInfo) -> Self {
        Self::BeginRendering(val)
    }
}

/// The render pass context that a secondary command buffer is created for.
#[derive(Clone, Debug)]
pub struct CommandBufferInheritanceRenderPassInfo {
    /// The render subpass that this secondary command buffer must be executed within.
    ///
    /// There is no default value.
    pub subpass: Subpass,

    /// The framebuffer object that will be used when calling the command buffer.
    /// This parameter is optional and is an optimization hint for the implementation.
    ///
    /// The default value is `None`.
    pub framebuffer: Option<Arc<Framebuffer>>,
}

impl CommandBufferInheritanceRenderPassInfo {
    /// Returns a default `CommandBufferInheritanceRenderPassInfo` with the provided `subpass`.
    #[inline]
    pub const fn new(subpass: Subpass) -> Self {
        Self {
            subpass,
            framebuffer: None,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn subpass(subpass: Subpass) -> Self {
        Self::new(subpass)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            subpass,
            framebuffer,
        } = self;

        // VUID-VkCommandBufferInheritanceInfo-commonparent
        assert_eq!(device, subpass.render_pass().device().as_ref());

        // VUID-VkCommandBufferBeginInfo-flags-06001
        // Ensured by how the `Subpass` type is constructed.

        if let Some(framebuffer) = framebuffer {
            // VUID-VkCommandBufferInheritanceInfo-commonparent
            assert_eq!(device, framebuffer.device().as_ref());

            if !framebuffer
                .render_pass()
                .is_compatible_with(subpass.render_pass())
            {
                return Err(Box::new(ValidationError {
                    problem: "`framebuffer` is not compatible with `subpass.render_pass()`".into(),
                    vuids: &["VUID-VkCommandBufferBeginInfo-flags-00055"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}

impl From<Subpass> for CommandBufferInheritanceRenderPassInfo {
    #[inline]
    fn from(subpass: Subpass) -> Self {
        Self {
            subpass,
            framebuffer: None,
        }
    }
}

/// The dynamic rendering context that a secondary command buffer is created for.
#[derive(Clone, Debug)]
pub struct CommandBufferInheritanceRenderingInfo {
    /// If not `0`, indicates that multiview rendering will be enabled, and specifies the view
    /// indices that are rendered to. The value is a bitmask, so that that for example `0b11` will
    /// draw to the first two views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, then the [`multiview`] feature must be enabled on the device.
    ///
    /// The default value is `0`.
    ///
    /// [`multiview`]: crate::device::DeviceFeatures::multiview
    pub view_mask: u32,

    /// The formats of the color attachments that will be used during rendering.
    ///
    /// If an element is `None`, it indicates that the attachment will not be used.
    ///
    /// The default value is empty.
    pub color_attachment_formats: Vec<Option<Format>>,

    /// The format of the depth attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no depth attachment will be used.
    ///
    /// The default value is `None`.
    pub depth_attachment_format: Option<Format>,

    /// The format of the stencil attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no stencil attachment will be used.
    ///
    /// The default value is `None`.
    pub stencil_attachment_format: Option<Format>,

    /// The number of samples that the color, depth and stencil attachments will have.
    ///
    /// The default value is [`SampleCount::Sample1`]
    pub rasterization_samples: SampleCount,
}

impl Default for CommandBufferInheritanceRenderingInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CommandBufferInheritanceRenderingInfo {
    /// Returns a default `CommandBufferInheritanceRenderingInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            view_mask: 0,
            color_attachment_formats: Vec::new(),
            depth_attachment_format: None,
            stencil_attachment_format: None,
            rasterization_samples: SampleCount::Sample1,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            view_mask,
            ref color_attachment_formats,
            depth_attachment_format,
            stencil_attachment_format,
            rasterization_samples,
        } = self;

        let properties = device.physical_device().properties();

        if view_mask != 0 && !device.enabled_features().multiview {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "multiview",
                )])]),
                vuids: &["VUID-VkCommandBufferInheritanceRenderingInfo-multiview-06008"],
            }));
        }

        let view_count = u32::BITS - view_mask.leading_zeros();

        if view_count > properties.max_multiview_view_count.unwrap_or(0) {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "the number of views exceeds the \
                    `max_multiview_view_count` limit"
                    .into(),
                vuids: &["VUID-VkCommandBufferInheritanceRenderingInfo-viewMask-06009"],
                ..Default::default()
            }));
        }

        for (index, format) in color_attachment_formats
            .iter()
            .enumerate()
            .flat_map(|(i, f)| f.map(|f| (i, f)))
        {
            format.validate_device(device).map_err(|err| {
                err.add_context(format!("color_attachment_formats[{}]", index)).set_vuids(
                    &["VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-parameter"],
                )
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachment_formats[{}]", index).into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            let format_properties =
                unsafe { device.physical_device().format_properties_unchecked(format) };
            let potential_format_features = format_properties.potential_format_features();

            if !potential_format_features.intersects(FormatFeatures::COLOR_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachment_formats[{}]", index).into(),
                    problem: "the potential format features do not contain \
                        `FormatFeatures::COLOR_ATTACHMENT`".into(),
                    vuids: &["VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-06006"],
                    ..Default::default()
                }));
            }
        }

        if let Some(format) = depth_attachment_format {
            format.validate_device(device).map_err(|err| {
                err.add_context("depth_attachment_format").set_vuids(&[
                    "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-parameter",
                ])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::DEPTH) {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "does not have a depth aspect".into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06540",
                    ],
                    ..Default::default()
                }));
            }

            let format_properties =
                unsafe { device.physical_device().format_properties_unchecked(format) };
            let potential_format_features = format_properties.potential_format_features();

            if !potential_format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "the potential format features do not contain \
                        `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06007",
                    ],
                    ..Default::default()
                }));
            }
        }

        if let Some(format) = stencil_attachment_format {
            format.validate_device(device).map_err(|err| {
                err.add_context("stencil_attachment_format").set_vuids(&["VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-parameter"])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::STENCIL) {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "does not have a stencil aspect".into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06541",
                    ],
                    ..Default::default()
                }));
            }

            let format_properties =
                unsafe { device.physical_device().format_properties_unchecked(format) };
            let potential_format_features = format_properties.potential_format_features();

            if !potential_format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "the potential format features do not contain \
                        `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06199",
                    ],
                    ..Default::default()
                }));
            }
        }

        if let (Some(depth_format), Some(stencil_format)) =
            (depth_attachment_format, stencil_attachment_format)
        {
            if depth_format != stencil_format {
                return Err(Box::new(ValidationError {
                    problem: "`depth_attachment_format` and `stencil_attachment_format` are both \
                        `Some`, but are not equal"
                        .into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06200",
                    ],
                    ..Default::default()
                }));
            }
        }

        rasterization_samples
            .validate_device(device)
            .map_err(|err| {
                err.add_context("rasterization_samples").set_vuids(&[
                    "VUID-VkCommandBufferInheritanceRenderingInfo-rasterizationSamples-parameter",
                ])
            })?;

        Ok(())
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a CommandBufferInheritanceRenderingInfoFields1Vk,
    ) -> vk::CommandBufferInheritanceRenderingInfo<'a> {
        let &Self {
            view_mask,
            color_attachment_formats: _,
            depth_attachment_format,
            stencil_attachment_format,
            rasterization_samples,
        } = self;
        let CommandBufferInheritanceRenderingInfoFields1Vk {
            color_attachment_formats_vk,
        } = fields1_vk;

        vk::CommandBufferInheritanceRenderingInfo::default()
            .flags(vk::RenderingFlags::empty())
            .view_mask(view_mask)
            .color_attachment_formats(color_attachment_formats_vk)
            .depth_attachment_format(
                depth_attachment_format.map_or(vk::Format::UNDEFINED, Into::into),
            )
            .stencil_attachment_format(
                stencil_attachment_format.map_or(vk::Format::UNDEFINED, Into::into),
            )
            .rasterization_samples(rasterization_samples.into())
    }

    pub(crate) fn to_vk_fields1(&self) -> CommandBufferInheritanceRenderingInfoFields1Vk {
        let Self {
            color_attachment_formats,
            ..
        } = self;

        let color_attachment_formats_vk = color_attachment_formats
            .iter()
            .map(|format| format.map_or(vk::Format::UNDEFINED, Into::into))
            .collect();

        CommandBufferInheritanceRenderingInfoFields1Vk {
            color_attachment_formats_vk,
        }
    }
}

pub(crate) struct CommandBufferInheritanceRenderingInfoFields1Vk {
    pub(crate) color_attachment_formats_vk: SmallVec<[vk::Format; 4]>,
}

/// Usage flags to pass when creating a command buffer.
///
/// The safest option is `SimultaneousUse`, but it may be slower than the other two.
// NOTE: The ordering is important: the variants are listed from least to most permissive!
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum CommandBufferUsage {
    /// The command buffer can only be submitted once before being destroyed. Any further submit is
    /// forbidden. This makes it possible for the implementation to perform additional
    /// optimizations.
    OneTimeSubmit = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT.as_raw(),

    /// The command buffer can be used multiple times, but must not execute or record more than
    /// once simultaneously. In other words, it is as if executing the command buffer borrows
    /// it mutably.
    MultipleSubmit = 0,

    /// The command buffer can be executed multiple times in parallel on different queues.
    /// If it's a secondary command buffer, it can be recorded to multiple primary command buffers
    /// at once.
    SimultaneousUse = vk::CommandBufferUsageFlags::SIMULTANEOUS_USE.as_raw(),
}

impl From<CommandBufferUsage> for vk::CommandBufferUsageFlags {
    #[inline]
    fn from(val: CommandBufferUsage) -> Self {
        Self::from_raw(val as u32)
    }
}

/// Parameters to submit command buffers to a queue.
#[derive(Clone, Debug)]
pub struct SubmitInfo<'a> {
    /// The semaphores to wait for before beginning the execution of this batch of
    /// command buffer operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: &'a [SemaphoreSubmitInfo<'a>],

    /// The command buffers to execute.
    ///
    /// The default value is empty.
    pub command_buffers: &'a [CommandBufferSubmitInfo<'a>],

    /// The semaphores to signal after the execution of this batch of command buffer operations
    /// has completed.
    ///
    /// The default value is empty.
    pub signal_semaphores: &'a [SemaphoreSubmitInfo<'a>],

    /// Indicates whether the submission is protected.
    ///
    /// The default value is false, indicating an unprotected submit.
    pub protected_submit: bool,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SubmitInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SubmitInfo<'_> {
    /// Returns a default `SubmitInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            wait_semaphores: &[],
            command_buffers: &[],
            signal_semaphores: &[],
            protected_submit: false,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            wait_semaphores,
            command_buffers,
            signal_semaphores,
            protected_submit,
            _ne: _,
        } = self;

        for (index, semaphore_submit_info) in wait_semaphores.iter().enumerate() {
            semaphore_submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("wait_semaphores[{}]", index)))?;
        }

        for (index, command_buffer_submit_info) in command_buffers.iter().enumerate() {
            command_buffer_submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("command_buffers[{}]", index)))?;

            if !protected_submit && command_buffer_submit_info.command_buffer.is_protected() {
                return Err(Box::new(ValidationError {
                    context: format!("command_buffer[{}]", index).into(),
                    problem: "is a protected command buffer, but `protected_submit` is `false`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "protected_memory",
                    )])]),
                    vuids: &["VUID-VkSubmitInfo-pNext-04120"],
                }));
            } else if protected_submit && !command_buffer_submit_info.command_buffer.is_protected()
            {
                return Err(Box::new(ValidationError {
                    context: format!("command_buffer[{}]", index).into(),
                    problem: "is an unprotected command buffer, but `protected_submit` is `true`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "protected_memory",
                    )])]),
                    vuids: &["VUID-VkSubmitInfo-pNext-04148"],
                }));
            }
        }

        for (index, semaphore_submit_info) in signal_semaphores.iter().enumerate() {
            semaphore_submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("signal_semaphores[{}]", index)))?;

            let &SemaphoreSubmitInfo {
                semaphore: _,
                value: _,
                stages,
                _ne: _,
            } = semaphore_submit_info;

            if stages != PipelineStages::ALL_COMMANDS && !device.enabled_features().synchronization2
            {
                return Err(Box::new(ValidationError {
                    context: format!("signal_semaphores[{}].stages", index).into(),
                    problem: "is not `PipelineStages::ALL_COMMANDS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    vuids: &["VUID-vkQueueSubmit2-synchronization2-03866"],
                }));
            }
        }

        // unsafe
        // VUID-VkSubmitInfo2-semaphore-03882
        // VUID-VkSubmitInfo2-semaphore-03883
        // VUID-VkSubmitInfo2-semaphore-03884

        Ok(())
    }

    pub(crate) fn to_vk2<'a>(&self, fields1_vk: &'a SubmitInfo2Fields1Vk) -> vk::SubmitInfo2<'a> {
        let SubmitInfo2Fields1Vk {
            wait_semaphore_infos_vk,
            command_buffer_infos_vk,
            signal_semaphore_infos_vk,
        } = fields1_vk;

        vk::SubmitInfo2::default()
            .flags(vk::SubmitFlags::empty()) // TODO:
            .wait_semaphore_infos(wait_semaphore_infos_vk)
            .command_buffer_infos(command_buffer_infos_vk)
            .signal_semaphore_infos(signal_semaphore_infos_vk)
    }

    pub(crate) fn to_vk2_fields1(&self) -> SubmitInfo2Fields1Vk {
        let &Self {
            wait_semaphores,
            command_buffers,
            signal_semaphores,
            protected_submit: _,
            _ne: _,
        } = self;

        SubmitInfo2Fields1Vk {
            wait_semaphore_infos_vk: wait_semaphores
                .iter()
                .map(SemaphoreSubmitInfo::to_vk2)
                .collect(),
            command_buffer_infos_vk: command_buffers
                .iter()
                .map(CommandBufferSubmitInfo::to_vk2)
                .collect(),
            signal_semaphore_infos_vk: signal_semaphores
                .iter()
                .map(SemaphoreSubmitInfo::to_vk2)
                .collect(),
        }
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a SubmitInfoFields1Vk,
        extensions_vk: &'a mut SubmitInfoExtensionsVk<'_>,
    ) -> vk::SubmitInfo<'a> {
        let SubmitInfoFields1Vk {
            wait_semaphores_vk,
            wait_dst_stage_mask_vk,
            wait_semaphore_values_vk: _,
            command_buffers_vk,
            signal_semaphores_vk,
            signal_semaphore_values_vk: _,
        } = fields1_vk;

        let mut val_vk = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores_vk)
            .wait_dst_stage_mask(wait_dst_stage_mask_vk)
            .command_buffers(command_buffers_vk)
            .signal_semaphores(signal_semaphores_vk);

        let SubmitInfoExtensionsVk {
            timeline_semaphore_vk,
            protected_vk,
        } = extensions_vk;

        if let Some(next) = timeline_semaphore_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = protected_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions<'a>(
        &self,
        fields1_vk: &'a SubmitInfoFields1Vk,
    ) -> SubmitInfoExtensionsVk<'a> {
        let &Self {
            wait_semaphores,
            command_buffers: _,
            signal_semaphores,
            protected_submit,
            _ne: _,
        } = self;
        let SubmitInfoFields1Vk {
            wait_semaphores_vk: _,
            wait_dst_stage_mask_vk: _,
            command_buffers_vk: _,
            signal_semaphores_vk: _,
            wait_semaphore_values_vk,
            signal_semaphore_values_vk,
        } = fields1_vk;

        let timeline_semaphore_vk = (wait_semaphores.iter())
            .chain(signal_semaphores.iter())
            .any(|semaphore_submit_info| {
                semaphore_submit_info.semaphore.semaphore_type() == SemaphoreType::Timeline
            })
            .then(|| {
                vk::TimelineSemaphoreSubmitInfo::default()
                    .wait_semaphore_values(wait_semaphore_values_vk)
                    .signal_semaphore_values(signal_semaphore_values_vk)
            });

        let protected_vk =
            protected_submit.then(|| vk::ProtectedSubmitInfo::default().protected_submit(true));

        SubmitInfoExtensionsVk {
            timeline_semaphore_vk,
            protected_vk,
        }
    }

    pub(crate) fn to_vk_fields1(&self) -> SubmitInfoFields1Vk {
        let &Self {
            wait_semaphores,
            command_buffers,
            signal_semaphores,
            protected_submit: _,
            _ne: _,
        } = self;

        let mut wait_semaphores_vk = SmallVec::with_capacity(wait_semaphores.len());
        let mut wait_dst_stage_mask_vk = SmallVec::with_capacity(wait_semaphores.len());
        let mut wait_semaphore_values_vk = SmallVec::with_capacity(wait_semaphores.len());

        for semaphore_submit_info in wait_semaphores {
            let &SemaphoreSubmitInfo {
                semaphore,
                value,
                stages,
                _ne: _,
            } = semaphore_submit_info;

            wait_semaphores_vk.push(semaphore.handle());
            wait_dst_stage_mask_vk.push(stages.into());
            wait_semaphore_values_vk.push(value);
        }

        let command_buffers_vk = command_buffers
            .iter()
            .map(CommandBufferSubmitInfo::to_vk)
            .collect();

        let mut signal_semaphores_vk = SmallVec::with_capacity(signal_semaphores.len());
        let mut signal_semaphore_values_vk = SmallVec::with_capacity(signal_semaphores.len());

        for semaphore_submit_info in signal_semaphores {
            let &SemaphoreSubmitInfo {
                semaphore,
                value,
                stages: _,
                _ne: _,
            } = semaphore_submit_info;

            signal_semaphores_vk.push(semaphore.handle());
            signal_semaphore_values_vk.push(value);
        }

        SubmitInfoFields1Vk {
            wait_semaphores_vk,
            wait_dst_stage_mask_vk,
            wait_semaphore_values_vk,
            command_buffers_vk,
            signal_semaphores_vk,
            signal_semaphore_values_vk,
        }
    }
}

pub(crate) struct SubmitInfo2Fields1Vk {
    pub(crate) wait_semaphore_infos_vk: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 4]>,
    pub(crate) command_buffer_infos_vk: SmallVec<[vk::CommandBufferSubmitInfo<'static>; 4]>,
    pub(crate) signal_semaphore_infos_vk: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 4]>,
}

pub(crate) struct SubmitInfoExtensionsVk<'a> {
    pub(crate) timeline_semaphore_vk: Option<vk::TimelineSemaphoreSubmitInfo<'a>>,
    pub(crate) protected_vk: Option<vk::ProtectedSubmitInfo<'a>>,
}

pub(crate) struct SubmitInfoFields1Vk {
    pub(crate) wait_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
    pub(crate) wait_dst_stage_mask_vk: SmallVec<[vk::PipelineStageFlags; 4]>,
    pub(crate) wait_semaphore_values_vk: SmallVec<[u64; 4]>,
    pub(crate) command_buffers_vk: SmallVec<[vk::CommandBuffer; 4]>,
    pub(crate) signal_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
    pub(crate) signal_semaphore_values_vk: SmallVec<[u64; 4]>,
}

/// Parameters for a command buffer in a queue submit operation.
#[derive(Clone, Debug)]
pub struct CommandBufferSubmitInfo<'a> {
    /// The command buffer to execute.
    ///
    /// There is no default value.
    pub command_buffer: &'a CommandBuffer,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CommandBufferSubmitInfo<'a> {
    /// Returns a default `CommandBufferSubmitInfo` with the provided `command_buffer`.
    #[inline]
    pub const fn new(command_buffer: &'a CommandBuffer) -> Self {
        Self {
            command_buffer,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            command_buffer,
            _ne: _,
        } = self;

        // VUID?
        assert_eq!(device, command_buffer.device().as_ref());

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::CommandBufferSubmitInfo<'static> {
        let &Self {
            command_buffer,
            _ne: _,
        } = self;

        vk::CommandBufferSubmitInfo::default()
            .command_buffer(command_buffer.handle())
            .device_mask(0) // TODO:
    }

    pub(crate) fn to_vk(&self) -> vk::CommandBuffer {
        let &Self {
            command_buffer,
            _ne: _,
        } = self;

        command_buffer.handle()
    }
}

/// Parameters for a semaphore signal or wait operation in a queue submit operation.
#[derive(Clone, Debug)]
pub struct SemaphoreSubmitInfo<'a> {
    /// The semaphore to signal or wait for.
    ///
    /// There is no default value.
    pub semaphore: &'a Semaphore,

    /// If `semaphore.semaphore_type()` is [`SemaphoreType::Timeline`], specifies the value that
    /// will be used for the semaphore operation:
    /// - If it's a signal operation, then the semaphore's value will be set to this value when it
    ///   is signaled.
    /// - If it's a wait operation, then the semaphore will wait until its value is greater than or
    ///   equal to this value.
    ///
    /// If `semaphore.semaphore_type()` is [`SemaphoreType::Binary`], then this must be `0`.
    ///
    /// The default value is `0`.
    pub value: u64,

    /// For a semaphore wait operation, specifies the pipeline stages in the second synchronization
    /// scope: stages of queue operations following the wait operation that can start executing
    /// after the semaphore is signaled.
    ///
    /// For a semaphore signal operation, specifies the pipeline stages in the first
    /// synchronization scope: stages of queue operations preceding the signal operation that
    /// must complete before the semaphore is signaled.
    /// If this value does not equal [`ALL_COMMANDS`], then the [`synchronization2`] feature must
    /// be enabled on the device.
    ///
    /// The default value is [`ALL_COMMANDS`].
    ///
    /// [`ALL_COMMANDS`]: PipelineStages::ALL_COMMANDS
    /// [`synchronization2`]: crate::device::DeviceFeatures::synchronization2
    pub stages: PipelineStages,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> SemaphoreSubmitInfo<'a> {
    /// Returns a default `SemaphoreSubmitInfo` with the provided `semaphore`.
    #[inline]
    pub const fn new(semaphore: &'a Semaphore) -> Self {
        Self {
            semaphore,
            value: 0,
            stages: PipelineStages::ALL_COMMANDS,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            semaphore,
            value,
            stages,
            _ne: _,
        } = self;

        // VUID?
        assert_eq!(device, semaphore.device().as_ref());

        match semaphore.semaphore_type() {
            SemaphoreType::Binary => {
                if value != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`semaphore.semaphore_type()` is `SemaphoreType::Binary`, but \
                            `value` is not `0`"
                            .into(),
                        ..Default::default()
                    }));
                }
            }
            SemaphoreType::Timeline => {}
        }

        stages.validate_device(device).map_err(|err| {
            err.add_context("stages")
                .set_vuids(&["VUID-VkSemaphoreSubmitInfo-stageMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 && stages.contains_flags2() {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "synchronization2",
                )])]),
                ..Default::default()
            }));
        }

        if !device.enabled_features().geometry_shader
            && stages.intersects(PipelineStages::GEOMETRY_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "geometry_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03929"],
            }));
        }

        if !device.enabled_features().tessellation_shader
            && stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            )
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                    `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "tessellation_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03930"],
            }));
        }

        if !device.enabled_features().conditional_rendering
            && stages.intersects(PipelineStages::CONDITIONAL_RENDERING)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "conditional_rendering",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03931"],
            }));
        }

        if !device.enabled_features().fragment_density_map
            && stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "fragment_density_map",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03932"],
            }));
        }

        if !device.enabled_features().transform_feedback
            && stages.intersects(PipelineStages::TRANSFORM_FEEDBACK)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "transform_feedback",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03933"],
            }));
        }

        if !device.enabled_features().mesh_shader && stages.intersects(PipelineStages::MESH_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::MESH_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "mesh_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03934"],
            }));
        }

        if !device.enabled_features().task_shader && stages.intersects(PipelineStages::TASK_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::TASK_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "task_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03935"],
            }));
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
            && stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::DeviceFeature("attachment_fragment_shading_rate")]),
                    RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                ]),
                vuids: &["VUID-VkMemoryBarrier2-shadingRateImage-07316"],
            }));
        }

        if !device.enabled_features().subpass_shading
            && stages.intersects(PipelineStages::SUBPASS_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "subpass_shading",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-04957"],
            }));
        }

        if !device.enabled_features().invocation_mask
            && stages.intersects(PipelineStages::INVOCATION_MASK)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "invocation_mask",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-04995"],
            }));
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
            && stages.intersects(PipelineStages::RAY_TRACING_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "ray_tracing_pipeline",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-07946"],
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::SemaphoreSubmitInfo<'static> {
        let &Self {
            semaphore,
            value,
            stages,
            _ne: _,
        } = self;

        vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore.handle())
            .value(value)
            .stage_mask(stages.into())
            .device_index(0) // TODO:
    }
}

#[derive(Clone, Debug, Default)]
#[doc(hidden)]
pub struct OldSubmitInfo {
    pub(crate) wait_semaphores: Vec<Arc<Semaphore>>,
    pub(crate) command_buffers: Vec<Arc<dyn PrimaryCommandBufferAbstract>>,
    pub(crate) signal_semaphores: Vec<Arc<Semaphore>>,
    pub(crate) protected_submit: bool,
}

#[derive(Debug, Default)]
pub struct CommandBufferState {
    has_been_submitted: bool,
    pending_submits: u32,
}

impl CommandBufferState {
    pub(crate) fn has_been_submitted(&self) -> bool {
        self.has_been_submitted
    }

    pub(crate) fn is_submit_pending(&self) -> bool {
        self.pending_submits != 0
    }

    pub(crate) unsafe fn add_queue_submit(&mut self) {
        self.has_been_submitted = true;
        self.pending_submits += 1;
    }

    pub(crate) unsafe fn set_submit_finished(&mut self) {
        self.pending_submits -= 1;
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CommandBufferResourcesUsage {
    pub(crate) buffers: Vec<CommandBufferBufferUsage>,
    pub(crate) images: Vec<CommandBufferImageUsage>,
    pub(crate) buffer_indices: HashMap<Arc<Buffer>, usize>,
    pub(crate) image_indices: HashMap<Arc<Image>, usize>,
}

#[derive(Debug)]
pub(crate) struct CommandBufferBufferUsage {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) ranges: RangeMap<DeviceSize, CommandBufferBufferRangeUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommandBufferBufferRangeUsage {
    pub(crate) first_use: Option<ResourceUseRef>,
    pub(crate) mutable: bool,
}

#[derive(Debug)]
pub(crate) struct CommandBufferImageUsage {
    pub(crate) image: Arc<Image>,
    pub(crate) ranges: RangeMap<DeviceSize, CommandBufferImageRangeUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommandBufferImageRangeUsage {
    pub(crate) first_use: Option<ResourceUseRef>,
    pub(crate) mutable: bool,
    pub(crate) expected_layout: ImageLayout,
    pub(crate) final_layout: ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceUseRef {
    pub command_index: usize,
    pub command_name: &'static str,
    pub resource_in_command: ResourceInCommand,
    pub secondary_use_ref: Option<SecondaryResourceUseRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryResourceUseRef {
    pub command_index: usize,
    pub command_name: &'static str,
    pub resource_in_command: ResourceInCommand,
}

impl From<ResourceUseRef> for SecondaryResourceUseRef {
    #[inline]
    fn from(val: ResourceUseRef) -> Self {
        let ResourceUseRef {
            command_index,
            command_name,
            resource_in_command,
            secondary_use_ref,
        } = val;

        debug_assert!(secondary_use_ref.is_none());

        SecondaryResourceUseRef {
            command_index,
            command_name,
            resource_in_command,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResourceInCommand {
    AccelerationStructure { index: u32 },
    ColorAttachment { index: u32 },
    ColorResolveAttachment { index: u32 },
    DepthStencilAttachment,
    DepthStencilResolveAttachment,
    DescriptorSet { set: u32, binding: u32, index: u32 },
    Destination,
    FramebufferAttachment { index: u32 },
    GeometryAabbsData { index: u32 },
    GeometryInstancesData,
    GeometryTrianglesTransformData { index: u32 },
    GeometryTrianglesIndexData { index: u32 },
    GeometryTrianglesVertexData { index: u32 },
    ImageMemoryBarrier { index: u32 },
    IndexBuffer,
    IndirectBuffer,
    ScratchData,
    SecondaryCommandBuffer { index: u32 },
    Source,
    VertexBuffer { binding: u32 },
    ShaderBindingTableBuffer,
}

#[doc(hidden)]
#[derive(Debug, Default)]
pub struct SecondaryCommandBufferResourcesUsage {
    pub(crate) buffers: Vec<SecondaryCommandBufferBufferUsage>,
    pub(crate) images: Vec<SecondaryCommandBufferImageUsage>,
}

#[derive(Debug)]
pub(crate) struct SecondaryCommandBufferBufferUsage {
    pub(crate) use_ref: ResourceUseRef,
    pub(crate) buffer: Subbuffer<[u8]>,
    pub(crate) range: Range<DeviceSize>,
    pub(crate) memory_access: PipelineStageAccessFlags,
}

#[derive(Debug)]
pub(crate) struct SecondaryCommandBufferImageUsage {
    pub(crate) use_ref: ResourceUseRef,
    pub(crate) image: Arc<Image>,
    pub(crate) subresource_range: ImageSubresourceRange,
    pub(crate) memory_access: PipelineStageAccessFlags,
    pub(crate) start_layout: ImageLayout,
    pub(crate) end_layout: ImageLayout,
}
