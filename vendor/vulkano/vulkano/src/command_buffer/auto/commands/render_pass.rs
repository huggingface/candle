use crate::{
    command_buffer::{
        auto::{
            BeginRenderPassState, BeginRenderingState, RenderPassState, RenderPassStateAttachments,
            RenderPassStateType, Resource,
        },
        raw,
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, ResourceInCommand, SubpassContents,
    },
    device::Device,
    format::{ClearColorValue, ClearValue},
    image::{view::ImageView, ImageAspects, ImageLayout},
    pipeline::graphics::subpass::OwnedPipelineRenderingCreateInfo,
    render_pass::{AttachmentLoadOp, AttachmentStoreOp, Framebuffer, RenderPass, ResolveMode},
    sync::PipelineStageAccessFlags,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;
use smallvec::SmallVec;
use std::{cmp::min, sync::Arc};

/// # Commands for render passes.
///
/// These commands require a graphics queue.
impl<L> AutoCommandBufferBuilder<L> {
    /// Begins a render pass using a render pass object and framebuffer.
    ///
    /// You must call this or `begin_rendering` before you can record draw commands.
    ///
    /// `contents` specifies what kinds of commands will be recorded in the render pass, either
    /// draw commands or executions of secondary command buffers.
    pub fn begin_render_pass(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_render_pass(&render_pass_begin_info, &subpass_begin_info)?;

        Ok(unsafe { self.begin_render_pass_unchecked(render_pass_begin_info, subpass_begin_info) })
    }

    fn validate_begin_render_pass(
        &self,
        render_pass_begin_info: &RenderPassBeginInfo,
        subpass_begin_info: &SubpassBeginInfo,
    ) -> Result<(), Box<ValidationError>> {
        let render_pass_begin_info_raw = raw::RenderPassBeginInfo {
            render_pass: &render_pass_begin_info.render_pass,
            framebuffer: &render_pass_begin_info.framebuffer,
            render_area_offset: render_pass_begin_info.render_area_offset,
            render_area_extent: render_pass_begin_info.render_area_extent,
            clear_values: &render_pass_begin_info.clear_values,
            _ne: crate::NE,
        };
        let subpass_begin_info_raw = raw::SubpassBeginInfo {
            contents: subpass_begin_info.contents,
            _ne: crate::NE,
        };
        self.inner
            .validate_begin_render_pass(&render_pass_begin_info_raw, &subpass_begin_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is already active".into(),
                vuids: &["VUID-vkCmdBeginRenderPass2-renderpass"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBeginRenderPass2-initialLayout-03100
        // TODO:

        // VUID-vkCmdBeginRenderPass2-srcStageMask-06453
        // TODO:

        // VUID-vkCmdBeginRenderPass2-dstStageMask-06454
        // TODO:

        // VUID-vkCmdBeginRenderPass2-framebuffer-02533
        // For any attachment in framebuffer that is used by renderPass and is bound to memory
        // locations that are also bound to another attachment used by renderPass, and if at least
        // one of those uses causes either attachment to be written to, both attachments
        // must have had the VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT set

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_render_pass_unchecked(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> &mut Self {
        let &RenderPassBeginInfo {
            ref render_pass,
            ref framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values: _,
            _ne: _,
        } = &render_pass_begin_info;

        let subpass = render_pass.first_subpass();
        self.builder_state.render_pass = Some(RenderPassState {
            contents: subpass_begin_info.contents,
            render_area_offset,
            render_area_extent,

            rendering_info: OwnedPipelineRenderingCreateInfo::from_subpass(&subpass),
            attachments: Some(RenderPassStateAttachments::from_subpass(
                &subpass,
                framebuffer,
            )),

            render_pass: BeginRenderPassState {
                subpass,
                framebuffer: Some(framebuffer.clone()),
            }
            .into(),
        });

        self.add_render_pass_begin(
            "begin_render_pass",
            render_pass
                .attachments()
                .iter()
                .enumerate()
                .map(|(index, desc)| {
                    let image_view = &framebuffer.attachments()[index];
                    let index = index as u32;

                    (
                        ResourceInCommand::FramebufferAttachment { index }.into(),
                        Resource::Image {
                            image: image_view.image().clone(),
                            subresource_range: *image_view.subresource_range(),
                            // TODO: suboptimal
                            memory_access: PipelineStageAccessFlags::FragmentShader_InputAttachmentRead
                                | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentRead
                                | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                            start_layout: desc.initial_layout,
                            end_layout: desc.final_layout,
                        },
                    )
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let render_pass_begin_info_raw = raw::RenderPassBeginInfo {
                    render_pass: &render_pass_begin_info.render_pass,
                    framebuffer: &render_pass_begin_info.framebuffer,
                    render_area_offset: render_pass_begin_info.render_area_offset,
                    render_area_extent: render_pass_begin_info.render_area_extent,
                    clear_values: &render_pass_begin_info.clear_values,
                    _ne: crate::NE,
                };
                let subpass_begin_info_raw = raw::SubpassBeginInfo {
                    contents: subpass_begin_info.contents,
                    _ne: crate::NE,
                };
                unsafe {
                    out.begin_render_pass_unchecked(
                        &render_pass_begin_info_raw,
                        &subpass_begin_info_raw,
                    )
                };
            },
        );

        self
    }

    /// Advances to the next subpass of the render pass previously begun with `begin_render_pass`.
    pub fn next_subpass(
        &mut self,
        subpass_end_info: SubpassEndInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_next_subpass(&subpass_end_info, &subpass_begin_info)?;

        Ok(unsafe { self.next_subpass_unchecked(subpass_end_info, subpass_begin_info) })
    }

    fn validate_next_subpass(
        &self,
        _subpass_end_info: &SubpassEndInfo,
        subpass_begin_info: &SubpassBeginInfo,
    ) -> Result<(), Box<ValidationError>> {
        let subpass_end_info_raw = raw::SubpassEndInfo { _ne: crate::NE };
        let subpass_begin_info_raw = raw::SubpassBeginInfo {
            contents: subpass_begin_info.contents,
            _ne: crate::NE,
        };
        self.inner
            .validate_next_subpass(&subpass_end_info_raw, &subpass_begin_info_raw)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdNextSubpass2-renderpass"],
                ..Default::default()
            })
        })?;

        let begin_render_pass_state = match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => state,
            RenderPassStateType::BeginRendering(_) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was not begun with \
                        `begin_render_pass`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        };

        if begin_render_pass_state.subpass.is_last_subpass() {
            return Err(Box::new(ValidationError {
                problem: "the current subpass is the last subpass of the render pass".into(),
                vuids: &["VUID-vkCmdNextSubpass2-None-03102"],
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(Box::new(ValidationError {
                problem: "a query that was begun in the current subpass is still active".into(),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn next_subpass_unchecked(
        &mut self,
        _subpass_end_info: SubpassEndInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> &mut Self {
        let render_pass_state = self.builder_state.render_pass.as_mut().unwrap();
        let begin_render_pass_state = match &mut render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(x) => x,
            _ => unreachable!(),
        };

        begin_render_pass_state.subpass.next_subpass();
        render_pass_state.contents = subpass_begin_info.contents;
        render_pass_state.rendering_info =
            OwnedPipelineRenderingCreateInfo::from_subpass(&begin_render_pass_state.subpass);
        render_pass_state.attachments = Some(RenderPassStateAttachments::from_subpass(
            &begin_render_pass_state.subpass,
            begin_render_pass_state.framebuffer.as_ref().unwrap(),
        ));

        if render_pass_state.rendering_info.as_ref().view_mask != 0 {
            // When multiview is enabled, at the beginning of each subpass, all
            // non-render pass state is undefined.
            self.builder_state.reset_non_render_pass_states();
        }

        self.add_command(
            "next_subpass",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                let subpass_end_info_raw = raw::SubpassEndInfo { _ne: crate::NE };
                let subpass_begin_info_raw = raw::SubpassBeginInfo {
                    contents: subpass_begin_info.contents,
                    _ne: crate::NE,
                };
                unsafe {
                    out.next_subpass_unchecked(&subpass_end_info_raw, &subpass_begin_info_raw)
                };
            },
        );

        self
    }

    /// Ends the render pass previously begun with `begin_render_pass`.
    ///
    /// This must be called after you went through all the subpasses.
    pub fn end_render_pass(
        &mut self,
        subpass_end_info: SubpassEndInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_render_pass(&subpass_end_info)?;

        Ok(unsafe { self.end_render_pass_unchecked(subpass_end_info) })
    }

    fn validate_end_render_pass(
        &self,
        _subpass_end_info: &SubpassEndInfo,
    ) -> Result<(), Box<ValidationError>> {
        let subpass_end_info_raw = raw::SubpassEndInfo { _ne: crate::NE };
        self.inner.validate_end_render_pass(&subpass_end_info_raw)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-renderpass"],
                ..Default::default()
            })
        })?;

        let begin_render_pass_state = match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => state,
            RenderPassStateType::BeginRendering(_) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was not begun with \
                        `begin_render_pass`"
                        .into(),
                    vuids: &["VUID-vkCmdEndRenderPass2-None-06171"],
                    ..Default::default()
                }));
            }
        };

        if !begin_render_pass_state.subpass.is_last_subpass() {
            return Err(Box::new(ValidationError {
                problem: "the current subpass is not the last subpass of the render pass".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-None-03103"],
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(Box::new(ValidationError {
                problem: "a query that was begun in the current subpass is still active".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-None-07005"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_render_pass_unchecked(
        &mut self,
        _subpass_end_info: SubpassEndInfo,
    ) -> &mut Self {
        self.builder_state.render_pass = None;

        self.add_render_pass_end(
            "end_render_pass",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                let subpass_end_info_raw = raw::SubpassEndInfo { _ne: crate::NE };
                unsafe { out.end_render_pass_unchecked(&subpass_end_info_raw) };
            },
        );

        self
    }

    /// Begins a render pass without a render pass object or framebuffer.
    ///
    /// Requires the [`dynamic_rendering`] device feature.
    ///
    /// You must call this or `begin_render_pass` before you can record draw commands.
    ///
    /// [`dynamic_rendering`]: crate::device::DeviceFeatures::dynamic_rendering
    pub fn begin_rendering(
        &mut self,
        mut rendering_info: RenderingInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        rendering_info.set_auto_extent_layers();
        self.validate_begin_rendering(&rendering_info)?;

        Ok(unsafe { self.begin_rendering_unchecked(rendering_info) })
    }

    fn validate_begin_rendering(
        &self,
        rendering_info: &RenderingInfo,
    ) -> Result<(), Box<ValidationError>> {
        let color_attachments_raw = rendering_info
            .color_attachments
            .iter()
            .map(convert_rendering_attachment_info)
            .collect::<Vec<_>>();
        let depth_attachment = convert_rendering_attachment_info(&rendering_info.depth_attachment);
        let stencil_attachment =
            convert_rendering_attachment_info(&rendering_info.stencil_attachment);
        let rendering_info_raw = raw::RenderingInfo {
            render_area_offset: rendering_info.render_area_offset,
            render_area_extent: rendering_info.render_area_extent,
            layer_count: rendering_info.layer_count,
            view_mask: rendering_info.view_mask,
            color_attachments: &color_attachments_raw,
            depth_attachment: Some(&depth_attachment),
            stencil_attachment: Some(&stencil_attachment),
            contents: rendering_info.contents,
            _ne: crate::NE,
        };
        self.inner.validate_begin_rendering(&rendering_info_raw)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is already active".into(),
                vuids: &["VUID-vkCmdBeginRendering-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_rendering_unchecked(
        &mut self,
        mut rendering_info: RenderingInfo,
    ) -> &mut Self {
        rendering_info.set_auto_extent_layers();

        let &RenderingInfo {
            render_area_offset,
            render_area_extent,
            layer_count: _,
            view_mask: _,
            ref color_attachments,
            ref depth_attachment,
            ref stencil_attachment,
            contents,
            _ne,
        } = &rendering_info;

        self.builder_state.render_pass = Some(RenderPassState {
            contents,
            render_area_offset,
            render_area_extent,

            rendering_info: OwnedPipelineRenderingCreateInfo::from_rendering_info(&rendering_info),
            attachments: Some(RenderPassStateAttachments::from_rendering_info(
                &rendering_info,
            )),

            render_pass: BeginRenderingState {
                pipeline_used: false,
            }
            .into(),
        });

        self.add_render_pass_begin(
            "begin_rendering",
            (color_attachments
                .iter()
                .enumerate()
                .filter_map(|(index, attachment_info)| {
                    attachment_info
                        .as_ref()
                        .map(|attachment_info| (index as u32, attachment_info))
                })
                .flat_map(|(index, attachment_info)| {
                    let &RenderingAttachmentInfo {
                        ref image_view,
                        image_layout,
                        ref resolve_info,
                        load_op: _,
                        store_op: _,
                        clear_value: _,
                        _ne: _,
                    } = attachment_info;

                    [
                        Some((
                            ResourceInCommand::ColorAttachment { index }.into(),
                            Resource::Image {
                                image: image_view.image().clone(),
                                subresource_range: *image_view.subresource_range(),
                                // TODO: suboptimal
                                memory_access: PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentRead
                                    | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite,
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )),
                        resolve_info.as_ref().map(|resolve_info| {
                            let &RenderingAttachmentResolveInfo {
                                mode: _,
                                ref image_view,
                                image_layout,
                            } = resolve_info;

                            (
                                ResourceInCommand::ColorResolveAttachment { index }.into(),
                                Resource::Image {
                                    image: image_view.image().clone(),
                                    subresource_range: *image_view.subresource_range(),
                                    // TODO: suboptimal
                                    memory_access: PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentRead
                                        | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite,
                                    start_layout: image_layout,
                                    end_layout: image_layout,
                                },
                            )
                        }),
                    ]
                    .into_iter()
                    .flatten()
                }))
            .chain(depth_attachment.iter().flat_map(|attachment_info| {
                let &RenderingAttachmentInfo {
                    ref image_view,
                    image_layout,
                    ref resolve_info,
                    load_op: _,
                    store_op: _,
                    clear_value: _,
                    _ne: _,
                } = attachment_info;

                [
                    Some((
                        ResourceInCommand::DepthStencilAttachment.into(),
                        Resource::Image {
                            image: image_view.image().clone(),
                            subresource_range: *image_view.subresource_range(),
                            // TODO: suboptimal
                            memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )),
                    resolve_info.as_ref().map(|resolve_info| {
                        let &RenderingAttachmentResolveInfo {
                            mode: _,
                            ref image_view,
                            image_layout,
                        } = resolve_info;

                        (
                            ResourceInCommand::DepthStencilResolveAttachment.into(),
                            Resource::Image {
                                image: image_view.image().clone(),
                                subresource_range: *image_view.subresource_range(),
                                // TODO: suboptimal
                                memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            }))
            .chain(stencil_attachment.iter().flat_map(|attachment_info| {
                let &RenderingAttachmentInfo {
                    ref image_view,
                    image_layout,
                    ref resolve_info,
                    load_op: _,
                    store_op: _,
                    clear_value: _,
                    _ne: _,
                } = attachment_info;

                [
                    Some((
                        ResourceInCommand::DepthStencilAttachment.into(),
                        Resource::Image {
                            image: image_view.image().clone(),
                            subresource_range: *image_view.subresource_range(),
                            // TODO: suboptimal
                            memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )),
                    resolve_info.as_ref().map(|resolve_info| {
                        let &RenderingAttachmentResolveInfo {
                            mode: _,
                            ref image_view,
                            image_layout,
                        } = resolve_info;

                        (
                            ResourceInCommand::DepthStencilResolveAttachment.into(),
                            Resource::Image {
                                image: image_view.image().clone(),
                                subresource_range: *image_view.subresource_range(),
                                // TODO: suboptimal
                                memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            }))
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let color_attachments_raw = rendering_info
                    .color_attachments
                    .iter()
                    .map(convert_rendering_attachment_info)
                    .collect::<Vec<_>>();
                let depth_attachment =
                    convert_rendering_attachment_info(&rendering_info.depth_attachment);
                let stencil_attachment =
                    convert_rendering_attachment_info(&rendering_info.stencil_attachment);
                let rendering_info_raw = raw::RenderingInfo {
                    render_area_offset: rendering_info.render_area_offset,
                    render_area_extent: rendering_info.render_area_extent,
                    layer_count: rendering_info.layer_count,
                    view_mask: rendering_info.view_mask,
                    color_attachments: &color_attachments_raw,
                    depth_attachment: Some(&depth_attachment),
                    stencil_attachment: Some(&stencil_attachment),
                    contents: rendering_info.contents,
                    _ne: crate::NE,
                };
                unsafe { out.begin_rendering_unchecked(&rendering_info_raw) };
            },
        );

        self
    }

    /// Ends the render pass previously begun with `begin_rendering`.
    pub fn end_rendering(&mut self) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_rendering()?;

        Ok(unsafe { self.end_rendering_unchecked() })
    }

    fn validate_end_rendering(&self) -> Result<(), Box<ValidationError>> {
        self.inner.validate_end_rendering()?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &[
                    "VUID-vkCmdEndRendering-renderpass",
                    "VUID-vkCmdEndRendering-commandBuffer-06162",
                ],
                ..Default::default()
            })
        })?;

        match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(_) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was not begun with \
                        `begin_rendering`"
                        .into(),
                    vuids: &["VUID-vkCmdEndRendering-None-06161"],
                    ..Default::default()
                }))
            }
            RenderPassStateType::BeginRendering(_) => (),
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(Box::new(ValidationError {
                problem: "a query that was begun in the current render pass instance \
                    is still active"
                    .into(),
                vuids: &["VUID-vkCmdEndRendering-None-06999"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_rendering_unchecked(&mut self) -> &mut Self {
        self.builder_state.render_pass = None;

        self.add_render_pass_end(
            "end_rendering",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.end_rendering_unchecked() };
            },
        );

        self
    }

    /// Clears specific regions of specific attachments of the framebuffer.
    ///
    /// `attachments` specify the types of attachments and their clear values.
    /// `rects` specify the regions to clear.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). And the command must be inside
    /// render pass.
    ///
    /// If the render pass instance this is recorded in uses multiview,
    /// then `ClearRect.base_array_layer` must be zero and `ClearRect.layer_count` must be one.
    ///
    /// The rectangle area must be inside the render area ranges.
    pub fn clear_attachments(
        &mut self,
        attachments: SmallVec<[ClearAttachment; 4]>,
        rects: SmallVec<[ClearRect; 4]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_attachments(&attachments, &rects)?;

        Ok(unsafe { self.clear_attachments_unchecked(attachments, rects) })
    }

    fn validate_clear_attachments(
        &self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_clear_attachments(attachments, rects)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdClearAttachments-renderpass"],
                ..Default::default()
            })
        })?;

        if render_pass_state.contents != SubpassContents::Inline {
            return Err(Box::new(ValidationError {
                problem: "the contents of the current subpass instance is not \
                    `SubpassContents::Inline`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        let mut layer_count = u32::MAX;

        for (clear_index, &clear_attachment) in attachments.iter().enumerate() {
            match clear_attachment {
                ClearAttachment::Color {
                    color_attachment,
                    clear_value,
                } => {
                    let attachment_format = *render_pass_state
                        .rendering_info
                        .as_ref()
                        .color_attachment_formats
                        .get(color_attachment as usize)
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                context: format!("attachments[{}].color_attachment", clear_index)
                                    .into(),
                                problem: "is not less than the number of color attachments in the \
                                current subpass instance"
                                    .into(),
                                vuids: &["VUID-vkCmdClearAttachments-aspectMask-07271"],
                                ..Default::default()
                            })
                        })?;

                    if let Some(attachment_format) = attachment_format {
                        let required_numeric_type = attachment_format
                            .numeric_format_color()
                            .unwrap()
                            .numeric_type();

                        if clear_value.numeric_type() != required_numeric_type {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`attachments[{0}].clear_value` is `ClearColorValue::{1:?}`, \
                                    but the color attachment specified by \
                                    `attachments[{0}].color_attachment` requires a clear value \
                                    of type `ClearColorValue::{2:?}`",
                                    clear_index,
                                    clear_value.numeric_type(),
                                    required_numeric_type,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdClearAttachments-aspectMask-02501"],
                                ..Default::default()
                            }));
                        }
                    }

                    let image_view = render_pass_state
                        .attachments
                        .as_ref()
                        .and_then(|attachments| attachments.depth_attachment.as_ref())
                        .map(|attachment_info| &attachment_info.image_view);

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        layer_count = min(layer_count, image_view.subresource_range_layer_count());
                    }
                }
                ClearAttachment::Depth(_)
                | ClearAttachment::Stencil(_)
                | ClearAttachment::DepthStencil(_) => {
                    if matches!(
                        clear_attachment,
                        ClearAttachment::Depth(_) | ClearAttachment::DepthStencil(_)
                    ) && render_pass_state
                        .rendering_info
                        .as_ref()
                        .depth_attachment_format
                        .is_none()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`attachments[{0}]` is `ClearAttachment::Depth` or \
                                `ClearAttachment::DepthStencil`, but \
                                the current subpass instance does not have a depth attachment",
                                clear_index,
                            )
                            .into(),
                            vuids: &["VUID-vkCmdClearAttachments-aspectMask-02502"],
                            ..Default::default()
                        }));
                    }

                    if matches!(
                        clear_attachment,
                        ClearAttachment::Stencil(_) | ClearAttachment::DepthStencil(_)
                    ) && render_pass_state
                        .rendering_info
                        .as_ref()
                        .stencil_attachment_format
                        .is_none()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`attachments[{0}]` is `ClearAttachment::Stencil` or \
                                `ClearAttachment::DepthStencil`, but \
                                the current subpass instance does not have a stencil attachment",
                                clear_index,
                            )
                            .into(),
                            vuids: &["VUID-vkCmdClearAttachments-aspectMask-02503"],
                            ..Default::default()
                        }));
                    }

                    let image_view = render_pass_state
                        .attachments
                        .as_ref()
                        .and_then(|attachments| attachments.depth_attachment.as_ref())
                        .map(|attachment_info| &attachment_info.image_view);

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        layer_count = min(layer_count, image_view.subresource_range_layer_count());
                    }
                }
            }
        }

        for (rect_index, rect) in rects.iter().enumerate() {
            for i in 0..2 {
                // TODO: This check will always pass in secondary command buffers because of how
                // it's set in `with_level`.
                // It needs to be checked during `execute_commands` instead.

                if rect.offset[i] < render_pass_state.render_area_offset[i] {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`rects[{0}].offset[{i}]` is less than \
                            `render_area_offset[{i}]` of the current render pass instance",
                            rect_index
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearAttachments-pRects-00016"],
                        ..Default::default()
                    }));
                }

                if rect.offset[i] + rect.extent[i]
                    > render_pass_state.render_area_offset[i]
                        + render_pass_state.render_area_extent[i]
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`rects[{0}].offset[{i}] + rects[{0}].extent[{i}]` is \
                            greater than `render_area_offset[{i}] + render_area_extent[{i}]` \
                            of the current render pass instance",
                            rect_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearAttachments-pRects-00016"],
                        ..Default::default()
                    }));
                }
            }

            if !rect
                .base_array_layer
                .checked_add(rect.layer_count)
                .is_some_and(|end| end <= layer_count)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`rects[{0}].base_array_layer + rects[{0}].layer_count` is greater than \
                        the number of array layers in the current render pass instance",
                        rect_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearAttachments-pRects-06937"],
                    ..Default::default()
                }));
            }

            if render_pass_state.rendering_info.as_ref().view_mask != 0
                && !(rect.base_array_layer == 0 && rect.layer_count == 1)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the current render pass instance has a non-zero `view_mask`, but \
                        `(rects[{0}].base_array_layer, rects[{0}].layer_count)` is not `(0, 1)`",
                        rect_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearAttachments-baseArrayLayer-00018"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_attachments_unchecked(
        &mut self,
        attachments: SmallVec<[ClearAttachment; 4]>,
        rects: SmallVec<[ClearRect; 4]>,
    ) -> &mut Self {
        self.add_command(
            "clear_attachments",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.clear_attachments_unchecked(&attachments, &rects) };
            },
        );

        self
    }
}

fn convert_rendering_attachment_info(
    attachment_info: &Option<RenderingAttachmentInfo>,
) -> Option<raw::RenderingAttachmentInfo<'_>> {
    attachment_info
        .as_ref()
        .map(|attachment_info| raw::RenderingAttachmentInfo {
            image_view: &attachment_info.image_view,
            image_layout: attachment_info.image_layout,
            resolve_info: attachment_info.resolve_info.as_ref().map(|resolve_info| {
                raw::RenderingAttachmentResolveInfo {
                    mode: resolve_info.mode,
                    image_view: &resolve_info.image_view,
                    image_layout: resolve_info.image_layout,
                }
            }),
            load_op: attachment_info.load_op,
            store_op: attachment_info.store_op,
            clear_value: attachment_info.clear_value,
            _ne: crate::NE,
        })
}

/// Parameters to begin a new render pass.
#[derive(Clone, Debug)]
pub struct RenderPassBeginInfo {
    /// The render pass to begin.
    ///
    /// If this is not the render pass that `framebuffer` was created with, it must be compatible
    /// with that render pass.
    ///
    /// The default value is the render pass of `framebuffer`.
    pub render_pass: Arc<RenderPass>,

    /// The framebuffer to use for rendering.
    ///
    /// There is no default value.
    pub framebuffer: Arc<Framebuffer>,

    /// The offset from the top left corner of the framebuffer that will be rendered to.
    ///
    /// The default value is `[0, 0]`.
    pub render_area_offset: [u32; 2],

    /// The size of the area that will be rendered to.
    ///
    /// `render_area_offset + render_area_extent` must not be greater than
    /// [`framebuffer.extent()`].
    ///
    /// The default value is [`framebuffer.extent()`].
    pub render_area_extent: [u32; 2],

    /// Provides, for each attachment in `render_pass` that has a load operation of
    /// [`AttachmentLoadOp::Clear`], the clear values that should be used for the attachments in
    /// the framebuffer.
    /// There must be exactly [`framebuffer.attachments().len()`] elements provided,
    /// and each one must match the attachment format.
    ///
    /// To skip over an attachment whose load operation is something else, provide `None`.
    ///
    /// The default value is empty, which must be overridden if the framebuffer has attachments.
    pub clear_values: Vec<Option<ClearValue>>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl RenderPassBeginInfo {
    #[inline]
    pub fn framebuffer(framebuffer: Arc<Framebuffer>) -> Self {
        let render_area_extent = framebuffer.extent();

        Self {
            render_pass: framebuffer.render_pass().clone(),
            framebuffer,
            render_area_offset: [0, 0],
            render_area_extent,
            clear_values: Vec::new(),
            _ne: crate::NE,
        }
    }
}

/// Parameters to begin a new subpass within a render pass.
#[derive(Clone, Debug)]
pub struct SubpassBeginInfo {
    /// What kinds of commands will be recorded in the subpass.
    ///
    /// The default value is [`SubpassContents::Inline`].
    pub contents: SubpassContents,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for SubpassBeginInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SubpassBeginInfo {
    /// Returns a default `SubpassBeginInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            contents: SubpassContents::Inline,
            _ne: crate::NE,
        }
    }
}

/// Parameters to end the current subpass within a render pass.
#[derive(Clone, Debug)]
pub struct SubpassEndInfo {
    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for SubpassEndInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SubpassEndInfo {
    /// Returns a default `SubpassEndInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self { _ne: crate::NE }
    }
}

/// Parameters to begin rendering.
#[derive(Clone, Debug)]
pub struct RenderingInfo {
    /// The offset from the top left corner of the attachments that will be rendered to.
    ///
    /// This value must be smaller than the smallest width and height of the attachment images.
    ///
    /// The default value is `[0, 0]`.
    pub render_area_offset: [u32; 2],

    /// The size of the area that will be rendered to.
    ///
    /// This value plus `render_area_offset` must be no larger than the smallest width and height
    /// of the attachment images.
    /// If one of the elements is set to 0, the extent will be calculated automatically from the
    /// extents of the attachment images to be the largest allowed. At least one attachment image
    /// must be specified in that case.
    ///
    /// The default value is `[0, 0]`.
    pub render_area_extent: [u32; 2],

    /// The number of layers of the attachments that will be rendered to.
    ///
    /// This must be no larger than the smallest number of array layers of the attachment images.
    /// If set to 0, the number of layers will be calculated automatically from the
    /// layer ranges of the attachment images to be the largest allowed. At least one attachment
    /// image must be specified in that case.
    ///
    /// If the render pass uses multiview (`view_mask` is not 0), then this value must be 0 or 1.
    ///
    /// The default value is `0`.
    pub layer_count: u32,

    /// If not `0`, enables multiview rendering, and specifies the view indices that are rendered
    /// to. The value is a bitmask, so that that for example `0b11` will draw to the first two
    /// views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, the [`multiview`](crate::device::DeviceFeatures::multiview)
    /// feature must be enabled on the device.
    ///
    /// The default value is `0`.
    pub view_mask: u32,

    /// The color attachments to use for rendering.
    ///
    /// The number of color attachments must be less than the
    /// [`max_color_attachments`](crate::device::DeviceProperties::max_color_attachments) limit of
    /// the physical device. All color attachments must have the same `samples` value.
    ///
    /// The default value is empty.
    pub color_attachments: Vec<Option<RenderingAttachmentInfo>>,

    /// The depth attachment to use for rendering.
    ///
    /// If set to `Some`, the image view must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub depth_attachment: Option<RenderingAttachmentInfo>,

    /// The stencil attachment to use for rendering.
    ///
    /// If set to `Some`, the image view must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub stencil_attachment: Option<RenderingAttachmentInfo>,

    /// What kinds of commands will be recorded in the render pass: either inline draw commands, or
    /// executions of secondary command buffers.
    ///
    /// If recorded in a secondary command buffer, this must be [`SubpassContents::Inline`].
    ///
    /// The default value is [`SubpassContents::Inline`].
    pub contents: SubpassContents,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for RenderingInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl RenderingInfo {
    /// Returns a default `RenderingInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            render_area_offset: [0, 0],
            render_area_extent: [0, 0],
            layer_count: 0,
            view_mask: 0,
            color_attachments: Vec::new(),
            depth_attachment: None,
            stencil_attachment: None,
            contents: SubpassContents::Inline,
            _ne: crate::NE,
        }
    }

    pub(crate) fn set_auto_extent_layers(&mut self) {
        let &mut RenderingInfo {
            render_area_offset,
            ref mut render_area_extent,
            ref mut layer_count,
            view_mask,
            ref color_attachments,
            ref depth_attachment,
            ref stencil_attachment,
            contents: _,
            _ne: _,
        } = self;

        let is_auto_extent = render_area_extent[0] == 0 || render_area_extent[1] == 0;
        let is_auto_layers = *layer_count == 0;

        if (is_auto_extent || is_auto_layers)
            && !(color_attachments.is_empty()
                && depth_attachment.is_none()
                && stencil_attachment.is_none())
        {
            let mut auto_extent = [u32::MAX, u32::MAX];
            let mut auto_layers = if view_mask != 0 { 1 } else { u32::MAX };

            if is_auto_extent {
                *render_area_extent = [u32::MAX, u32::MAX];
            }

            if is_auto_layers {
                if view_mask != 0 {
                    *layer_count = 1;
                } else {
                    *layer_count = u32::MAX;
                }
            }

            for image_view in color_attachments
                .iter()
                .flatten()
                .chain(depth_attachment.iter())
                .chain(stencil_attachment.iter())
                .flat_map(|attachment_info| {
                    Some(&attachment_info.image_view).into_iter().chain(
                        attachment_info
                            .resolve_info
                            .as_ref()
                            .map(|resolve_info| &resolve_info.image_view),
                    )
                })
            {
                let image_view_extent = image_view.image().extent();
                let image_view_array_layers = image_view.subresource_range_layer_count();

                auto_extent[0] = auto_extent[0].min(image_view_extent[0]);
                auto_extent[1] = auto_extent[1].min(image_view_extent[1]);
                auto_layers = auto_layers.min(image_view_array_layers);
            }

            if is_auto_extent {
                // Subtract the offset from the calculated max extent.
                // If there is an underflow, then the offset is too large, and validation should
                // catch that later.
                for i in 0..2 {
                    render_area_extent[i] = auto_extent[i]
                        .checked_sub(render_area_offset[i])
                        .unwrap_or(1);
                }
            }

            if is_auto_layers {
                *layer_count = auto_layers;
            }
        }
    }
}

/// Parameters to specify properties of an attachment.
#[derive(Clone, Debug)]
pub struct RenderingAttachmentInfo {
    /// The image view to use as the attachment.
    ///
    /// There is no default value.
    pub image_view: Arc<ImageView>,

    /// The image layout that `image_view` should be in during rendering.
    ///
    /// The default value is [`ImageLayout::ColorAttachmentOptimal`] if `image_view` has a color
    /// format, [`ImageLayout::DepthStencilAttachmentOptimal`] if `image_view` has a depth/stencil
    /// format.
    pub image_layout: ImageLayout,

    /// The resolve operation that should be performed at the end of rendering.
    ///
    /// The default value is `None`.
    pub resolve_info: Option<RenderingAttachmentResolveInfo>,

    /// What the implementation should do with the attachment at the start of rendering.
    ///
    /// The default value is [`AttachmentLoadOp::DontCare`].
    pub load_op: AttachmentLoadOp,

    /// What the implementation should do with the attachment at the end of rendering.
    ///
    /// The default value is [`AttachmentStoreOp::DontCare`].
    pub store_op: AttachmentStoreOp,

    /// If `load_op` is [`AttachmentLoadOp::Clear`],
    /// specifies the clear value that should be used for the attachment.
    ///
    /// If `load_op` is something else, provide `None`.
    ///
    /// The default value is `None`.
    pub clear_value: Option<ClearValue>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl RenderingAttachmentInfo {
    /// Returns a default `RenderingAttachmentInfo` with the provided `image_view`.
    #[inline]
    pub fn new(image_view: Arc<ImageView>) -> Self {
        let aspects = image_view.format().aspects();
        let image_layout = if aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            ImageLayout::DepthStencilAttachmentOptimal
        } else {
            ImageLayout::ColorAttachmentOptimal
        };

        Self {
            image_view,
            image_layout,
            resolve_info: None,
            load_op: AttachmentLoadOp::DontCare,
            store_op: AttachmentStoreOp::DontCare,
            clear_value: None,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image_view(image_view: Arc<ImageView>) -> Self {
        Self::new(image_view)
    }
}

/// Parameters to specify the resolve behavior of an attachment.
#[derive(Clone, Debug)]
pub struct RenderingAttachmentResolveInfo {
    /// How the resolve operation should be performed.
    ///
    /// The default value is [`ResolveMode::Average`].
    pub mode: ResolveMode,

    /// The image view that the result of the resolve operation should be written to.
    ///
    /// There is no default value.
    pub image_view: Arc<ImageView>,

    /// The image layout that `image_view` should be in during the resolve operation.
    ///
    /// The default value is [`ImageLayout::ColorAttachmentOptimal`] if `image_view` has a color
    /// format, [`ImageLayout::DepthStencilAttachmentOptimal`] if `image_view` has a depth/stencil
    /// format.
    pub image_layout: ImageLayout,
}

impl RenderingAttachmentResolveInfo {
    /// Returns a default `RenderingAttachmentResolveInfo` with the provided `image_view`.
    #[inline]
    pub fn new(image_view: Arc<ImageView>) -> Self {
        let aspects = image_view.format().aspects();
        let image_layout = if aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            ImageLayout::DepthStencilAttachmentOptimal
        } else {
            ImageLayout::ColorAttachmentOptimal
        };

        Self {
            mode: ResolveMode::Average,
            image_view,
            image_layout,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image_view(image_view: Arc<ImageView>) -> Self {
        Self::new(image_view)
    }
}

/// Clear attachment type, used in [`clear_attachments`] command.
///
/// [`clear_attachments`]: AutoCommandBufferBuilder::clear_attachments
#[derive(Clone, Copy, Debug)]
pub enum ClearAttachment {
    /// Clear the color attachment at the specified index, with the specified clear value.
    Color {
        color_attachment: u32,
        clear_value: ClearColorValue,
    },

    /// Clear the depth attachment with the specified depth value.
    Depth(f32),

    /// Clear the stencil attachment with the specified stencil value.
    Stencil(u32),

    /// Clear the depth and stencil attachments with the specified depth and stencil values.
    DepthStencil((f32, u32)),
}

impl ClearAttachment {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        if let ClearAttachment::Depth(depth) | ClearAttachment::DepthStencil((depth, _)) = self {
            if !(0.0..=1.0).contains(depth)
                && !device.enabled_extensions().ext_depth_range_unrestricted
            {
                return Err(Box::new(ValidationError {
                    problem: "is `ClearAttachment::Depth` or `ClearAttachment::DepthStencil`, and \
                        the depth value is not between 0.0 and 1.0 inclusive"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-VkClearDepthStencilValue-depth-00022"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::ClearAttachment {
        match *self {
            ClearAttachment::Color {
                color_attachment,
                clear_value,
            } => vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment,
                clear_value: vk::ClearValue {
                    color: clear_value.to_vk(),
                },
            },
            ClearAttachment::Depth(depth) => vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
                },
            },
            ClearAttachment::Stencil(stencil) => vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::STENCIL,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil,
                    },
                },
            },
            ClearAttachment::DepthStencil((depth, stencil)) => vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                },
            },
        }
    }
}

/// Specifies the clear region for the [`clear_attachments`] command.
///
/// [`clear_attachments`]: AutoCommandBufferBuilder::clear_attachments
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClearRect {
    /// The rectangle offset.
    pub offset: [u32; 2],

    /// The width and height of the rectangle.
    pub extent: [u32; 2],

    /// The first array layer to be cleared.
    pub base_array_layer: u32,

    /// The number of array layers to be cleared.
    pub layer_count: u32,
}

impl ClearRect {
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::ClearRect {
        let &Self {
            offset,
            extent,
            base_array_layer,
            layer_count,
        } = self;

        vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D {
                    x: offset[0] as i32,
                    y: offset[1] as i32,
                },
                extent: vk::Extent2D {
                    width: extent[0],
                    height: extent[1],
                },
            },
            base_array_layer,
            layer_count,
        }
    }
}
