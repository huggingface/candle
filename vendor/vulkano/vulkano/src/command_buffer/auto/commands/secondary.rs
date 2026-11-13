use crate::{
    command_buffer::{
        auto::{RenderPassStateType, Resource, ResourceUseRef2},
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, CommandBufferInheritanceRenderPassType, ResourceInCommand,
        SecondaryCommandBufferAbstract, SecondaryCommandBufferBufferUsage,
        SecondaryCommandBufferImageUsage, SecondaryCommandBufferResourcesUsage, SubpassContents,
    },
    device::DeviceOwned,
    query::QueryType,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{cmp::min, iter, sync::Arc};

/// # Commands to execute a secondary command buffer inside a primary command buffer.
///
/// These commands can be called on any queue that can execute the commands recorded in the
/// secondary command buffer.
impl<L> AutoCommandBufferBuilder<L> {
    /// Executes a secondary command buffer.
    ///
    /// If the `flags` that `command_buffer` was created with are more restrictive than those of
    /// `self`, then `self` will be restricted to match. E.g. executing a secondary command buffer
    /// with `Flags::OneTimeSubmit` will set `self`'s flags to `Flags::OneTimeSubmit` also.
    pub fn execute_commands(
        &mut self,
        command_buffer: Arc<dyn SecondaryCommandBufferAbstract>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let command_buffer = DropUnlockCommandBuffer::new(command_buffer)?;
        self.validate_execute_commands(iter::once(&**command_buffer))?;

        Ok(unsafe { self.execute_commands_locked(smallvec![command_buffer]) })
    }

    /// Executes multiple secondary command buffers in a vector.
    ///
    /// This requires that the secondary command buffers do not have resource conflicts; an error
    /// will be returned if there are any. Use `execute_commands` if you want to ensure that
    /// resource conflicts are automatically resolved.
    // TODO ^ would be nice if this just worked without errors
    pub fn execute_commands_from_vec(
        &mut self,
        command_buffers: Vec<Arc<dyn SecondaryCommandBufferAbstract>>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let command_buffers: SmallVec<[_; 4]> = command_buffers
            .into_iter()
            .map(DropUnlockCommandBuffer::new)
            .collect::<Result<_, _>>()?;

        self.validate_execute_commands(command_buffers.iter().map(|cb| &***cb))?;

        Ok(unsafe { self.execute_commands_locked(command_buffers) })
    }

    fn validate_execute_commands<'a>(
        &self,
        command_buffers: impl Iterator<Item = &'a dyn SecondaryCommandBufferAbstract> + Clone,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_execute_commands(command_buffers.clone().map(|cb| cb.as_raw()))?;

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            if render_pass_state.contents != SubpassContents::SecondaryCommandBuffers {
                return Err(Box::new(ValidationError {
                    problem: "a render pass instance is active, but its current subpass contents \
                        is `SubpassContents::SecondaryCommandBuffers`"
                        .into(),
                    vuids: &[
                        "VUID-vkCmdExecuteCommands-contents-06018",
                        "VUID-vkCmdExecuteCommands-flags-06024",
                    ],
                    ..Default::default()
                }));
            }
        }

        if !self.builder_state.queries.is_empty()
            && !self.device().enabled_features().inherited_queries
        {
            return Err(Box::new(ValidationError {
                problem: "a query is active".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "inherited_queries",
                )])]),
                vuids: &["VUID-vkCmdExecuteCommands-commandBuffer-00101"],
                ..Default::default()
            }));
        }

        for (command_buffer_index, command_buffer) in command_buffers.enumerate() {
            let inheritance_info = command_buffer.inheritance_info();

            if let Some(render_pass_state) = &self.builder_state.render_pass {
                let inheritance_render_pass =
                    inheritance_info.render_pass.as_ref().ok_or_else(|| {
                        Box::new(ValidationError {
                            problem: format!(
                                "a render pass instance is active, but \
                                `command_buffers[{}].inheritance_info().render_pass` is `None`",
                                command_buffer_index,
                            )
                            .into(),
                            vuids: &["VUID-vkCmdExecuteCommands-pCommandBuffers-00096"],
                            ..Default::default()
                        })
                    })?;

                match (&render_pass_state.render_pass, inheritance_render_pass) {
                    (
                        RenderPassStateType::BeginRenderPass(state),
                        CommandBufferInheritanceRenderPassType::BeginRenderPass(inheritance_info),
                    ) => {
                        if !inheritance_info
                            .subpass
                            .render_pass()
                            .is_compatible_with(state.subpass.render_pass())
                        {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "command_buffers[{}].inheritance_info().render_pass\
                                    .subpass.render_pass()",
                                    command_buffer_index
                                )
                                .into(),
                                problem: "is not compatible with the current render pass instance"
                                    .into(),
                                vuids: &["VUID-vkCmdExecuteCommands-pBeginInfo-06020"],
                                ..Default::default()
                            }));
                        }

                        if inheritance_info.subpass.index() != state.subpass.index() {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "command_buffers[{}].inheritance_info().render_pass\
                                    .subpass.index()",
                                    command_buffer_index
                                )
                                .into(),
                                problem: "is not equal to the index of the current \
                                    subpass instance"
                                    .into(),
                                vuids: &["VUID-vkCmdExecuteCommands-pCommandBuffers-06019"],
                                ..Default::default()
                            }));
                        }

                        if let Some(framebuffer) = &inheritance_info.framebuffer {
                            if framebuffer != state.framebuffer.as_ref().unwrap() {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .framebuffer",
                                        command_buffer_index
                                    )
                                    .into(),
                                    problem: "is `Some`, but is not equal to the framebuffer of \
                                        the current render pass instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-pCommandBuffers-00099"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                    (
                        RenderPassStateType::BeginRendering(_),
                        CommandBufferInheritanceRenderPassType::BeginRendering(inheritance_info),
                    ) => {
                        let attachments = render_pass_state.attachments.as_ref().unwrap();

                        if inheritance_info.color_attachment_formats.len()
                            != attachments.color_attachments.len()
                        {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "command_buffers[{}].inheritance_info().render_pass\
                                    .color_attachment_formats.len()",
                                    command_buffer_index
                                )
                                .into(),
                                problem: "is not equal to the number of color attachments in the \
                                    current subpass instance"
                                    .into(),
                                vuids: &["VUID-vkCmdExecuteCommands-colorAttachmentCount-06027"],
                                ..Default::default()
                            }));
                        }

                        for (color_attachment_index, image_view, inherited_format) in attachments
                            .color_attachments
                            .iter()
                            .zip(inheritance_info.color_attachment_formats.iter().copied())
                            .enumerate()
                            .filter_map(|(i, (a, f))| {
                                a.as_ref().map(|a| (i as u32, &a.image_view, f))
                            })
                        {
                            let required_format = image_view.format();

                            if Some(required_format) != inherited_format {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .color_attachment_formats[{}]",
                                        command_buffer_index, color_attachment_index
                                    )
                                    .into(),
                                    problem: "is not equal to the format of the \
                                        corresponding color attachment in the current subpass \
                                        instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-imageView-06028"],
                                    ..Default::default()
                                }));
                            }

                            if image_view.image().samples()
                                != inheritance_info.rasterization_samples
                            {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .rasterization_samples",
                                        command_buffer_index,
                                    )
                                    .into(),
                                    problem: "is not equal to the number of samples of the \
                                        attachments in the current subpass instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-pNext-06035"],
                                    ..Default::default()
                                }));
                            }
                        }

                        if let Some((image_view, format)) = attachments
                            .depth_attachment
                            .as_ref()
                            .map(|a| (&a.image_view, inheritance_info.depth_attachment_format))
                        {
                            if Some(image_view.format()) != format {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .depth_attachment_format",
                                        command_buffer_index
                                    )
                                    .into(),
                                    problem: "is not equal to the format of the \
                                        depth attachment in the current subpass instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-pDepthAttachment-06029"],
                                    ..Default::default()
                                }));
                            }

                            if image_view.image().samples()
                                != inheritance_info.rasterization_samples
                            {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .rasterization_samples",
                                        command_buffer_index,
                                    )
                                    .into(),
                                    problem: "is not equal to the number of samples of the \
                                        attachments in the current subpass instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-pNext-06036"],
                                    ..Default::default()
                                }));
                            }
                        }

                        if let Some((image_view, format)) = attachments
                            .stencil_attachment
                            .as_ref()
                            .map(|a| (&a.image_view, inheritance_info.stencil_attachment_format))
                        {
                            if Some(image_view.format()) != format {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .stencil_attachment_format",
                                        command_buffer_index
                                    )
                                    .into(),
                                    problem: "is not equal to the format of the \
                                        stencil attachment in the current subpass instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-pStencilAttachment-06030"],
                                    ..Default::default()
                                }));
                            }

                            if image_view.image().samples()
                                != inheritance_info.rasterization_samples
                            {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().render_pass\
                                        .rasterization_samples",
                                        command_buffer_index,
                                    )
                                    .into(),
                                    problem: "is not equal to the number of samples of the \
                                        attachments in the current subpass instance"
                                        .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-pNext-06037"],
                                    ..Default::default()
                                }));
                            }
                        }

                        if inheritance_info.view_mask
                            != render_pass_state.rendering_info.as_ref().view_mask
                        {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "command_buffers[{}].inheritance_info().render_pass\
                                    .view_mask",
                                    command_buffer_index,
                                )
                                .into(),
                                problem: "is not equal to the `view_mask` of the current subpass \
                                    instance"
                                    .into(),
                                vuids: &["VUID-vkCmdExecuteCommands-viewMask-06031"],
                                ..Default::default()
                            }));
                        }
                    }
                    (
                        RenderPassStateType::BeginRenderPass(_),
                        CommandBufferInheritanceRenderPassType::BeginRendering(_),
                    ) => {
                        return Err(Box::new(ValidationError {
                            context: format!(
                                "command_buffers[{}].inheritance_info().render_pass",
                                command_buffer_index
                            )
                            .into(),
                            problem: "is `CommandBufferInheritanceRenderPassType::\
                                BeginRendering`, but the current render pass instance was begun \
                                with `begin_render_pass`"
                                .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    }
                    (
                        RenderPassStateType::BeginRendering(_),
                        CommandBufferInheritanceRenderPassType::BeginRenderPass(_),
                    ) => {
                        return Err(Box::new(ValidationError {
                            context: format!(
                                "command_buffers[{}].inheritance_info().render_pass",
                                command_buffer_index
                            )
                            .into(),
                            problem: "is `CommandBufferInheritanceRenderPassType::\
                                BeginRenderPass`, but the current render pass instance was begun \
                                with `begin_rendering`"
                                .into(),
                            vuids: &["VUID-vkCmdExecuteCommands-pBeginInfo-06025"],
                            ..Default::default()
                        }));
                    }
                }

                // TODO:
                // VUID-vkCmdExecuteCommands-commandBuffer-06533
                // VUID-vkCmdExecuteCommands-commandBuffer-06534
                // VUID-vkCmdExecuteCommands-pCommandBuffers-06535
                // VUID-vkCmdExecuteCommands-pCommandBuffers-06536
            } else {
                if inheritance_info.render_pass.is_some() {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "command_buffers[{}].inheritance_info().render_pass",
                            command_buffer_index
                        )
                        .into(),
                        problem: "is `Some`, but a render pass instance is not active".into(),
                        vuids: &["VUID-vkCmdExecuteCommands-pCommandBuffers-00100"],
                        ..Default::default()
                    }));
                }
            }

            for state in self.builder_state.queries.values() {
                match state.query_pool.query_type() {
                    QueryType::Occlusion => {
                        let inherited_flags =
                            inheritance_info.occlusion_query.ok_or_else(|| {
                                Box::new(ValidationError {
                                    context: format!(
                                        "command_buffers[{}].inheritance_info().occlusion_query",
                                        command_buffer_index
                                    )
                                    .into(),
                                    problem:
                                        "is `None`, but an occlusion query is currently active"
                                            .into(),
                                    vuids: &["VUID-vkCmdExecuteCommands-commandBuffer-00102"],
                                    ..Default::default()
                                })
                            })?;

                        if !inherited_flags.contains(state.flags) {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "command_buffers[{}].inheritance_info().occlusion_query",
                                    command_buffer_index
                                )
                                .into(),
                                problem: "is not a superset of the flags of the active \
                                    occlusion query"
                                    .into(),
                                vuids: &["VUID-vkCmdExecuteCommands-commandBuffer-00103"],
                                ..Default::default()
                            }));
                        }
                    }
                    QueryType::PipelineStatistics => {
                        let inherited_flags = inheritance_info.pipeline_statistics;

                        if !inherited_flags.contains(state.query_pool.pipeline_statistics()) {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "command_buffers[{}].inheritance_info().pipeline_statistics",
                                    command_buffer_index
                                )
                                .into(),
                                problem: "is not a superset of the flags of the active \
                                    `PipelineStatistics` query"
                                    .into(),
                                vuids: &["VUID-vkCmdExecuteCommands-commandBuffer-00104"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                }
            }
        }

        // TODO:
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00091
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00092
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00093
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00105

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn execute_commands_unchecked(
        &mut self,
        command_buffers: SmallVec<[Arc<dyn SecondaryCommandBufferAbstract>; 4]>,
    ) -> &mut Self {
        unsafe {
            self.execute_commands_locked(
                command_buffers
                    .into_iter()
                    .map(DropUnlockCommandBuffer::new)
                    .collect::<Result<_, _>>()
                    .unwrap(),
            )
        }
    }

    unsafe fn execute_commands_locked(
        &mut self,
        command_buffers: SmallVec<[DropUnlockCommandBuffer; 4]>,
    ) -> &mut Self {
        // Secondary command buffer could leave the primary in any state.
        self.builder_state.reset_non_render_pass_states();

        self.add_command(
            "execute_commands",
            command_buffers
                .iter()
                .enumerate()
                .flat_map(|(index, command_buffer)| {
                    let index = index as u32;
                    let SecondaryCommandBufferResourcesUsage { buffers, images } =
                        command_buffer.resources_usage();

                    (buffers.iter().map(move |usage| {
                        let &SecondaryCommandBufferBufferUsage {
                            use_ref,
                            ref buffer,
                            ref range,
                            memory_access,
                        } = usage;

                        (
                            ResourceUseRef2 {
                                resource_in_command: ResourceInCommand::SecondaryCommandBuffer {
                                    index,
                                },
                                secondary_use_ref: Some(use_ref.into()),
                            },
                            Resource::Buffer {
                                buffer: buffer.clone(),
                                range: range.clone(),
                                memory_access,
                            },
                        )
                    }))
                    .chain(images.iter().map(move |usage| {
                        let &SecondaryCommandBufferImageUsage {
                            use_ref,
                            ref image,
                            subresource_range,
                            memory_access,
                            start_layout,
                            end_layout,
                        } = usage;

                        (
                            ResourceUseRef2 {
                                resource_in_command: ResourceInCommand::SecondaryCommandBuffer {
                                    index,
                                },
                                secondary_use_ref: Some(use_ref.into()),
                            },
                            Resource::Image {
                                image: image.clone(),
                                subresource_range,
                                memory_access,
                                start_layout,
                                end_layout,
                            },
                        )
                    }))
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.execute_commands_locked(&command_buffers) };
            },
        );

        self
    }
}

impl RecordingCommandBuffer {
    unsafe fn execute_commands_locked(
        &mut self,
        command_buffers: &[DropUnlockCommandBuffer],
    ) -> &mut Self {
        if command_buffers.is_empty() {
            return self;
        }

        let command_buffers_vk: SmallVec<[_; 4]> =
            command_buffers.iter().map(|cb| cb.handle()).collect();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_execute_commands)(
                self.handle(),
                command_buffers_vk.len() as u32,
                command_buffers_vk.as_ptr(),
            )
        };

        // If the secondary is non-concurrent or one-time use, that restricts the primary as
        // well.
        self.usage = command_buffers
            .iter()
            .map(|cb| cb.usage())
            .fold(self.usage, min);

        self
    }
}

pub(crate) struct DropUnlockCommandBuffer(Arc<dyn SecondaryCommandBufferAbstract>);

impl DropUnlockCommandBuffer {
    pub(crate) fn new(
        command_buffer: Arc<dyn SecondaryCommandBufferAbstract>,
    ) -> Result<Self, Box<ValidationError>> {
        command_buffer.lock_record()?;
        Ok(Self(command_buffer))
    }
}

impl std::ops::Deref for DropUnlockCommandBuffer {
    type Target = Arc<dyn SecondaryCommandBufferAbstract>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for DropUnlockCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.unlock();
        }
    }
}
