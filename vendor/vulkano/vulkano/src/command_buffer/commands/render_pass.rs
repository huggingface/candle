pub use crate::command_buffer::{ClearAttachment, ClearRect};
use crate::{
    command_buffer::{sys::RecordingCommandBuffer, CommandBufferLevel, SubpassContents},
    device::{Device, DeviceOwned, QueueFlags},
    format::{ClearValue, NumericType},
    image::{view::ImageView, ImageAspects, ImageLayout, ImageUsage, SampleCount},
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Framebuffer, RenderPass,
        ResolveMode, SubpassDescription,
    },
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin_info: &RenderPassBeginInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_begin_render_pass(render_pass_begin_info, subpass_begin_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_begin_render_pass(
        &mut self,
        render_pass_begin_info: &RenderPassBeginInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_render_pass(render_pass_begin_info, subpass_begin_info)?;

        Ok(unsafe { self.begin_render_pass_unchecked(render_pass_begin_info, subpass_begin_info) })
    }

    pub(crate) fn validate_begin_render_pass(
        &self,
        render_pass_begin_info: &RenderPassBeginInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if self.level() != CommandBufferLevel::Primary {
            return Err(Box::new(ValidationError {
                problem: "this command buffer is not a primary command buffer".into(),
                vuids: &["VUID-vkCmdBeginRenderPass2-bufferlevel"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBeginRenderPass2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        render_pass_begin_info
            .validate(self.device())
            .map_err(|err| err.add_context("render_pass_begin_info"))?;

        subpass_begin_info
            .validate(self.device())
            .map_err(|err| err.add_context("subpass_begin_info"))?;

        let &RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area_offset: _,
            render_area_extent: _,
            clear_values: _,
            _ne: _,
        } = render_pass_begin_info;

        for (attachment_index, (attachment_desc, image_view)) in render_pass
            .attachments()
            .iter()
            .zip(framebuffer.attachments())
            .enumerate()
        {
            let attachment_index = attachment_index as u32;
            let &AttachmentDescription {
                initial_layout,
                final_layout,
                stencil_initial_layout,
                stencil_final_layout,
                ..
            } = attachment_desc;

            for layout in [
                Some(initial_layout),
                Some(final_layout),
                stencil_initial_layout,
                stencil_final_layout,
            ]
            .into_iter()
            .flatten()
            {
                match layout {
                    ImageLayout::ColorAttachmentOptimal => {
                        if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::ColorAttachmentOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::COLOR_ATTACHMENT`",
                                    attachment_index,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03094"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal => {
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the \
                                    `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                                    `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                                    `ImageLayout::DepthStencilAttachmentOptimal`, \
                                    `ImageLayout::DepthStencilReadOnlyOptimal`, \
                                    `ImageLayout::DepthAttachmentOptimal`, \
                                    `ImageLayout::DepthReadOnlyOptimal`, \
                                    `ImageLayout::StencilAttachmentOptimal` or \
                                    `ImageLayout::StencilReadOnlyOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::DEPTH_STENCIL_ATTACHMENT`",
                                    attachment_index,
                                )
                                .into(),
                                vuids: &[
                                    "VUID-vkCmdBeginRenderPass2-initialLayout-03096",
                                    "VUID-vkCmdBeginRenderPass2-initialLayout-02844",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::ShaderReadOnlyOptimal => {
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::ShaderReadOnlyOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::SAMPLED` or `ImageUsage::INPUT_ATTACHMENT`",
                                    attachment_index,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03097"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::TransferSrcOptimal => {
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_SRC) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::TransferSrcOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::TRANSFER_SRC`",
                                    attachment_index,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03098"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::TransferDstOptimal => {
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_DST) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::TransferDstOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::TRANSFER_DST`",
                                    attachment_index,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03099"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::Undefined
                    | ImageLayout::General
                    | ImageLayout::Preinitialized
                    | ImageLayout::PresentSrc => (),
                }
            }
        }

        for subpass_desc in render_pass.subpasses() {
            let &SubpassDescription {
                flags: _,
                view_mask: _,
                input_attachments,
                color_attachments,
                color_resolve_attachments,
                depth_stencil_attachment,
                depth_stencil_resolve_attachment,
                depth_resolve_mode: _,
                stencil_resolve_mode: _,
                preserve_attachments: _,
                _ne: _,
            } = subpass_desc;

            for atch_ref in input_attachments
                .iter()
                .chain(color_attachments.iter())
                .chain(color_resolve_attachments.iter())
                .chain(depth_stencil_attachment)
                .chain(depth_stencil_resolve_attachment)
                .flatten()
            {
                let image_view = &framebuffer.attachments()[atch_ref.attachment as usize];

                match atch_ref.layout {
                    ImageLayout::ColorAttachmentOptimal => {
                        if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::ColorAttachmentOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::COLOR_ATTACHMENT`",
                                    atch_ref.attachment,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03094"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal => {
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the \
                                    `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                                    `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                                    `ImageLayout::DepthStencilAttachmentOptimal`, \
                                    `ImageLayout::DepthStencilReadOnlyOptimal`, \
                                    `ImageLayout::DepthAttachmentOptimal`, \
                                    `ImageLayout::DepthReadOnlyOptimal`, \
                                    `ImageLayout::StencilAttachmentOptimal` or \
                                    `ImageLayout::StencilReadOnlyOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::DEPTH_STENCIL_ATTACHMENT`",
                                    atch_ref.attachment,
                                )
                                .into(),
                                vuids: &[
                                    "VUID-vkCmdBeginRenderPass2-initialLayout-03096",
                                    "VUID-vkCmdBeginRenderPass2-initialLayout-02844",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::ShaderReadOnlyOptimal => {
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::ShaderReadOnlyOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::SAMPLED` or `ImageUsage::INPUT_ATTACHMENT`",
                                    atch_ref.attachment,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03097"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::TransferSrcOptimal => {
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_SRC) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::TransferSrcOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::TRANSFER_SRC`",
                                    atch_ref.attachment,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03098"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::TransferDstOptimal => {
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_DST) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`framebuffer.attachments()[{0}]` is used in `render_pass` \
                                    with the `ImageLayout::TransferDstOptimal` layout, but \
                                    `framebuffer.attachments()[{0}].usage()` does not contain \
                                    `ImageUsage::TRANSFER_DST`",
                                    atch_ref.attachment,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdBeginRenderPass2-initialLayout-03099"],
                                ..Default::default()
                            }));
                        }
                    }
                    ImageLayout::Undefined
                    | ImageLayout::General
                    | ImageLayout::Preinitialized
                    | ImageLayout::PresentSrc => (),
                }
            }
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
        render_pass_begin_info: &RenderPassBeginInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> &mut Self {
        let render_pass_begin_info_fields1_vk = render_pass_begin_info.to_vk_fields1();
        let render_pass_begin_info_vk =
            render_pass_begin_info.to_vk(&render_pass_begin_info_fields1_vk);

        let subpass_begin_info_vk = subpass_begin_info.to_vk();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_2
            || self.device().enabled_extensions().khr_create_renderpass2
        {
            if self.device().api_version() >= Version::V1_2 {
                unsafe {
                    (fns.v1_2.cmd_begin_render_pass2)(
                        self.handle(),
                        &render_pass_begin_info_vk,
                        &subpass_begin_info_vk,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_create_renderpass2.cmd_begin_render_pass2_khr)(
                        self.handle(),
                        &render_pass_begin_info_vk,
                        &subpass_begin_info_vk,
                    )
                };
            }
        } else {
            debug_assert!(subpass_begin_info_vk.p_next.is_null());

            unsafe {
                (fns.v1_0.cmd_begin_render_pass)(
                    self.handle(),
                    &render_pass_begin_info_vk,
                    subpass_begin_info_vk.contents,
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn next_subpass(
        &mut self,
        subpass_end_info: &SubpassEndInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_next_subpass(subpass_end_info, subpass_begin_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_next_subpass(
        &mut self,
        subpass_end_info: &SubpassEndInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_next_subpass(subpass_end_info, subpass_begin_info)?;

        Ok(unsafe { self.next_subpass_unchecked(subpass_end_info, subpass_begin_info) })
    }

    pub(crate) fn validate_next_subpass(
        &self,
        subpass_end_info: &SubpassEndInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if self.level() != CommandBufferLevel::Primary {
            return Err(Box::new(ValidationError {
                problem: "this command buffer is not a primary command buffer".into(),
                vuids: &["VUID-vkCmdNextSubpass2-bufferlevel"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdNextSubpass2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        subpass_end_info
            .validate(self.device())
            .map_err(|err| err.add_context("subpass_end_info"))?;

        subpass_begin_info
            .validate(self.device())
            .map_err(|err| err.add_context("subpass_begin_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn next_subpass_unchecked(
        &mut self,
        subpass_end_info: &SubpassEndInfo<'_>,
        subpass_begin_info: &SubpassBeginInfo<'_>,
    ) -> &mut Self {
        let fns = self.device().fns();

        let subpass_end_info_vk = subpass_end_info.to_vk();
        let subpass_begin_info_vk = subpass_begin_info.to_vk();

        if self.device().api_version() >= Version::V1_2
            || self.device().enabled_extensions().khr_create_renderpass2
        {
            if self.device().api_version() >= Version::V1_2 {
                unsafe {
                    (fns.v1_2.cmd_next_subpass2)(
                        self.handle(),
                        &subpass_begin_info_vk,
                        &subpass_end_info_vk,
                    )
                };
            } else {
                unsafe {
                    (fns.khr_create_renderpass2.cmd_next_subpass2_khr)(
                        self.handle(),
                        &subpass_begin_info_vk,
                        &subpass_end_info_vk,
                    )
                };
            }
        } else {
            debug_assert!(subpass_begin_info_vk.p_next.is_null());
            debug_assert!(subpass_end_info_vk.p_next.is_null());

            unsafe { (fns.v1_0.cmd_next_subpass)(self.handle(), subpass_begin_info_vk.contents) };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn end_render_pass(&mut self, subpass_end_info: &SubpassEndInfo<'_>) -> &mut Self {
        unsafe { self.try_end_render_pass(subpass_end_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_end_render_pass(
        &mut self,
        subpass_end_info: &SubpassEndInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_render_pass(subpass_end_info)?;

        Ok(unsafe { self.end_render_pass_unchecked(subpass_end_info) })
    }

    pub(crate) fn validate_end_render_pass(
        &self,
        subpass_end_info: &SubpassEndInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if self.level() != CommandBufferLevel::Primary {
            return Err(Box::new(ValidationError {
                problem: "this command buffer is not a primary command buffer".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-bufferlevel"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdEndRenderPass2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        subpass_end_info
            .validate(self.device())
            .map_err(|err| err.add_context("subpass_end_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_render_pass_unchecked(
        &mut self,
        subpass_end_info: &SubpassEndInfo<'_>,
    ) -> &mut Self {
        let subpass_end_info_vk = subpass_end_info.to_vk();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_2
            || self.device().enabled_extensions().khr_create_renderpass2
        {
            if self.device().api_version() >= Version::V1_2 {
                unsafe { (fns.v1_2.cmd_end_render_pass2)(self.handle(), &subpass_end_info_vk) };
            } else {
                unsafe {
                    (fns.khr_create_renderpass2.cmd_end_render_pass2_khr)(
                        self.handle(),
                        &subpass_end_info_vk,
                    )
                };
            }
        } else {
            debug_assert!(subpass_end_info_vk.p_next.is_null());

            unsafe { (fns.v1_0.cmd_end_render_pass)(self.handle()) };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn begin_rendering(&mut self, rendering_info: &RenderingInfo<'_>) -> &mut Self {
        unsafe { self.try_begin_rendering(rendering_info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_begin_rendering(
        &mut self,
        rendering_info: &RenderingInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let mut rendering_info = rendering_info.clone();
        rendering_info.set_auto_extent_layers();
        self.validate_begin_rendering(&rendering_info)?;

        Ok(unsafe { self.begin_rendering_unchecked(&rendering_info) })
    }

    pub(crate) fn validate_begin_rendering(
        &self,
        rendering_info: &RenderingInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        let device = self.device();

        if !device.enabled_features().dynamic_rendering {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "dynamic_rendering",
                )])]),
                vuids: &["VUID-vkCmdBeginRendering-dynamicRendering-06446"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBeginRendering-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        rendering_info
            .validate(self.device())
            .map_err(|err| err.add_context("rendering_info"))?;

        let &RenderingInfo {
            render_area_offset: _,
            render_area_extent: _,
            layer_count: _,
            view_mask: _,
            color_attachments: _,
            depth_attachment: _,
            stencil_attachment: _,
            contents,
            _ne: _,
        } = rendering_info;

        if self.level() == CommandBufferLevel::Secondary
            && contents == SubpassContents::SecondaryCommandBuffers
        {
            return Err(Box::new(ValidationError {
                problem: "this command buffer is a secondary command buffer, but \
                    `rendering_info.contents` is `SubpassContents::SecondaryCommandBuffers`"
                    .into(),
                vuids: &["VUID-vkCmdBeginRendering-commandBuffer-06068"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_rendering_unchecked(
        &mut self,
        rendering_info: &RenderingInfo<'_>,
    ) -> &mut Self {
        let mut rendering_info = rendering_info.clone();
        rendering_info.set_auto_extent_layers();

        let rendering_info_fields1_vk = rendering_info.to_vk_fields1();
        let rendering_info_vk = rendering_info.to_vk(&rendering_info_fields1_vk);

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            unsafe { (fns.v1_3.cmd_begin_rendering)(self.handle(), &rendering_info_vk) };
        } else {
            unsafe {
                (fns.khr_dynamic_rendering.cmd_begin_rendering_khr)(
                    self.handle(),
                    &rendering_info_vk,
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn end_rendering(&mut self) -> &mut Self {
        unsafe { self.try_end_rendering() }.unwrap()
    }

    #[inline]
    pub unsafe fn try_end_rendering(&mut self) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_rendering()?;

        Ok(unsafe { self.end_rendering_unchecked() })
    }

    pub(crate) fn validate_end_rendering(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdEndRendering-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_rendering_unchecked(&mut self) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            unsafe { (fns.v1_3.cmd_end_rendering)(self.handle()) };
        } else {
            unsafe { (fns.khr_dynamic_rendering.cmd_end_rendering_khr)(self.handle()) };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn clear_attachments(
        &mut self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> &mut Self {
        unsafe { self.try_clear_attachments(attachments, rects) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_clear_attachments(
        &mut self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_attachments(attachments, rects)?;

        Ok(unsafe { self.clear_attachments_unchecked(attachments, rects) })
    }

    pub(crate) fn validate_clear_attachments(
        &self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdClearAttachments-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        for (clear_index, clear_attachment) in attachments.iter().enumerate() {
            clear_attachment
                .validate(self.device())
                .map_err(|err| err.add_context(format!("attachments[{}]", clear_index)))?;
        }

        for (rect_index, rect) in rects.iter().enumerate() {
            let &ClearRect {
                offset: _,
                extent,
                base_array_layer: _,
                layer_count,
            } = rect;

            if extent[0] == 0 {
                return Err(Box::new(ValidationError {
                    context: format!("rects[{}].extent[0]", rect_index).into(),
                    problem: "is 0".into(),
                    vuids: &["VUID-vkCmdClearAttachments-rect-02682"],
                    ..Default::default()
                }));
            }

            if extent[1] == 0 {
                return Err(Box::new(ValidationError {
                    context: format!("rects[{}].extent[1]", rect_index).into(),
                    problem: "is 0".into(),
                    vuids: &["VUID-vkCmdClearAttachments-rect-02683"],
                    ..Default::default()
                }));
            }

            if layer_count == 0 {
                return Err(Box::new(ValidationError {
                    context: format!("rects[{}].layer_count", rect_index).into(),
                    problem: "is zero".into(),
                    vuids: &["VUID-vkCmdClearAttachments-layerCount-01934"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_attachments_unchecked(
        &mut self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> &mut Self {
        if attachments.is_empty() || rects.is_empty() {
            return self;
        }

        let attachments_vk: SmallVec<[_; 4]> =
            attachments.iter().map(ClearAttachment::to_vk).collect();
        let rects_vk: SmallVec<[_; 4]> = rects.iter().map(ClearRect::to_vk).collect();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_clear_attachments)(
                self.handle(),
                attachments_vk.len() as u32,
                attachments_vk.as_ptr(),
                rects_vk.len() as u32,
                rects_vk.as_ptr(),
            )
        };

        self
    }
}

/// Parameters to begin a new render pass.
#[derive(Clone, Debug)]
pub struct RenderPassBeginInfo<'a> {
    /// The render pass to begin.
    ///
    /// If this is not the render pass that `framebuffer` was created with, it must be compatible
    /// with that render pass.
    ///
    /// The default value is the render pass of `framebuffer`.
    pub render_pass: &'a RenderPass,

    /// The framebuffer to use for rendering.
    ///
    /// There is no default value.
    pub framebuffer: &'a Framebuffer,

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
    pub clear_values: &'a [Option<ClearValue>],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> RenderPassBeginInfo<'a> {
    /// Returns a default `RenderPassBeginInfo` with the provided `framebuffer`.
    // TODO: make const
    #[inline]
    pub fn new(framebuffer: &'a Framebuffer) -> Self {
        let render_area_extent = framebuffer.extent();

        Self {
            render_pass: framebuffer.render_pass(),
            framebuffer,
            render_area_offset: [0, 0],
            render_area_extent,
            clear_values: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            render_pass,
            framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values,
            _ne,
        } = self;

        // VUID-VkRenderPassBeginInfo-commonparent
        // VUID-vkCmdBeginRenderPass2-framebuffer-02779
        assert_eq!(device, framebuffer.device().as_ref());

        if !render_pass.is_compatible_with(framebuffer.render_pass()) {
            return Err(Box::new(ValidationError {
                problem: "`render_pass` is not compatible with `framebuffer.render_pass()`".into(),
                vuids: &["VUID-VkRenderPassBeginInfo-renderPass-00904"],
                ..Default::default()
            }));
        }

        if render_area_extent[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "render_area_extent[0]".into(),
                problem: "is 0".into(),
                vuids: &["VUID-VkRenderPassBeginInfo-None-08996"],
                ..Default::default()
            }));
        }

        if render_area_extent[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "render_area_extent[1]".into(),
                problem: "is 0".into(),
                vuids: &["VUID-VkRenderPassBeginInfo-None-08997"],
                ..Default::default()
            }));
        }

        if render_area_offset[0] + render_area_extent[0] > framebuffer.extent()[0] {
            return Err(Box::new(ValidationError {
                problem: "`render_area_offset[0] + render_area_extent[0]` is greater than \
                    `framebuffer.extent()[0]`"
                    .into(),
                vuids: &["VUID-VkRenderPassBeginInfo-pNext-02852"],
                ..Default::default()
            }));
        }

        if render_area_offset[1] + render_area_extent[1] > framebuffer.extent()[1] {
            return Err(Box::new(ValidationError {
                problem: "`render_area_offset[1] + render_area_extent[1]` is greater than \
                    `framebuffer.extent()[1]`"
                    .into(),
                vuids: &["VUID-VkRenderPassBeginInfo-pNext-02853"],
                ..Default::default()
            }));
        }

        if clear_values.len() != render_pass.attachments().len() {
            return Err(Box::new(ValidationError {
                problem: "`clear_values.len()` is not equal to `render_pass.attachments().len()`"
                    .into(),
                vuids: &["VUID-VkRenderPassBeginInfo-clearValueCount-00902"],
                ..Default::default()
            }));
        }

        // VUID-VkRenderPassBeginInfo-clearValueCount-04962
        for (attachment_index, (attachment_desc, clear_value)) in render_pass
            .attachments()
            .iter()
            .zip(clear_values)
            .enumerate()
        {
            match (clear_value, attachment_desc.required_clear_value()) {
                (None, None) => continue,
                (None, Some(_)) => {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`render_pass.attachments()[{0}]` requires a clear value, but \
                            `clear_values[{0}]` is `None`",
                            attachment_index
                        )
                        .into(),
                        ..Default::default()
                    }));
                }
                (Some(_), None) => {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`render_pass.attachments()[{0}]` does not require a clear value, but \
                            `clear_values[{0}]` is `Some`",
                            attachment_index
                        )
                        .into(),
                        ..Default::default()
                    }));
                }
                (Some(clear_value), Some(required_clear_value)) => {
                    if required_clear_value != clear_value.clear_value_type() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`clear_values[{0}]` is `ClearValue::{1:?}`, but \
                                `render_pass.attachments()[{0}]` requires a clear value of type \
                                `ClearValue::{2:?}`",
                                attachment_index,
                                clear_value.clear_value_type(),
                                required_clear_value,
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }

                    clear_value.validate(device).map_err(|err| {
                        err.add_context(format!("clear_values[{}]", attachment_index))
                    })?;
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a RenderPassBeginInfoFields1Vk,
    ) -> vk::RenderPassBeginInfo<'a> {
        let &Self {
            render_pass,
            framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values: _,
            _ne,
        } = self;
        let RenderPassBeginInfoFields1Vk { clear_values_vk } = fields1_vk;

        vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(framebuffer.handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D {
                    x: render_area_offset[0] as i32,
                    y: render_area_offset[1] as i32,
                },
                extent: vk::Extent2D {
                    width: render_area_extent[0],
                    height: render_area_extent[1],
                },
            })
            .clear_values(clear_values_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> RenderPassBeginInfoFields1Vk {
        let clear_values_vk = self
            .clear_values
            .iter()
            .map(|clear_value| {
                clear_value
                    .as_ref()
                    .map_or_else(Default::default, ClearValue::to_vk)
            })
            .collect();

        RenderPassBeginInfoFields1Vk { clear_values_vk }
    }
}

pub(crate) struct RenderPassBeginInfoFields1Vk {
    pub(crate) clear_values_vk: SmallVec<[vk::ClearValue; 4]>,
}

/// Parameters to begin a new subpass within a render pass.
#[derive(Clone, Debug)]
pub struct SubpassBeginInfo<'a> {
    /// What kinds of commands will be recorded in the subpass.
    ///
    /// The default value is [`SubpassContents::Inline`].
    pub contents: SubpassContents,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SubpassBeginInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SubpassBeginInfo<'_> {
    /// Returns a default `SubpassBeginInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            contents: SubpassContents::Inline,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self { contents, _ne: _ } = self;

        contents.validate_device(device).map_err(|err| {
            err.add_context("contents")
                .set_vuids(&["VUID-VkSubpassBeginInfo-contents-parameter"])
        })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::SubpassBeginInfo<'static> {
        let &Self { contents, _ne: _ } = self;

        vk::SubpassBeginInfo::default().contents(contents.into())
    }
}

/// Parameters to end the current subpass within a render pass.
#[derive(Clone, Debug)]
pub struct SubpassEndInfo<'a> {
    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SubpassEndInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SubpassEndInfo<'_> {
    /// Returns a default `SubpassEndInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self { _ne: crate::NE }
    }

    pub(crate) fn validate(&self, _device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self { _ne: _ } = self;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::SubpassEndInfo<'static> {
        let &Self { _ne: _ } = self;

        vk::SubpassEndInfo::default()
    }
}

/// Parameters to begin rendering.
#[derive(Clone, Debug)]
pub struct RenderingInfo<'a> {
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
    pub color_attachments: &'a [Option<RenderingAttachmentInfo<'a>>],

    /// The depth attachment to use for rendering.
    ///
    /// If set to `Some`, the image view must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub depth_attachment: Option<&'a Option<RenderingAttachmentInfo<'a>>>,

    /// The stencil attachment to use for rendering.
    ///
    /// If set to `Some`, the image view must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub stencil_attachment: Option<&'a Option<RenderingAttachmentInfo<'a>>>,

    /// What kinds of commands will be recorded in the render pass: either inline draw commands, or
    /// executions of secondary command buffers.
    ///
    /// If recorded in a secondary command buffer, this must be [`SubpassContents::Inline`].
    ///
    /// The default value is [`SubpassContents::Inline`].
    pub contents: SubpassContents,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for RenderingInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl RenderingInfo<'_> {
    /// Returns a default `RenderingInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            render_area_offset: [0, 0],
            render_area_extent: [0, 0],
            layer_count: 0,
            view_mask: 0,
            color_attachments: &[],
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
            color_attachments,
            depth_attachment,
            stencil_attachment,
            contents: _,
            _ne: _,
        } = self;

        let is_auto_extent = render_area_extent[0] == 0 || render_area_extent[1] == 0;
        let is_auto_layers = *layer_count == 0;

        if (is_auto_extent || is_auto_layers)
            && !(color_attachments.is_empty()
                && depth_attachment.is_none_or(Option::is_none)
                && stencil_attachment.is_none_or(Option::is_none))
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
                .chain(depth_attachment.and_then(Option::as_ref))
                .chain(stencil_attachment.and_then(Option::as_ref))
                .flat_map(|attachment_info| {
                    Some(attachment_info.image_view).into_iter().chain(
                        attachment_info
                            .resolve_info
                            .as_ref()
                            .map(|resolve_info| resolve_info.image_view),
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            render_area_offset,
            render_area_extent,
            layer_count,
            view_mask,
            color_attachments,
            depth_attachment,
            stencil_attachment,
            contents,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        contents.validate_device(device).map_err(|err| {
            err.add_context("contents")
                .set_vuids(&["VUID-VkRenderingInfo-flags-parameter"])
        })?;

        if render_area_extent[0] == 0 {
            return Err(Box::new(ValidationError {
                context: "render_area_extent[0]".into(),
                problem: "is 0".into(),
                vuids: &["VUID-VkRenderingInfo-None-08994"],
                ..Default::default()
            }));
        }

        if render_area_extent[1] == 0 {
            return Err(Box::new(ValidationError {
                context: "render_area_extent[1]".into(),
                problem: "is 0".into(),
                vuids: &["VUID-VkRenderingInfo-None-08995"],
                ..Default::default()
            }));
        }

        if render_area_offset[0] + render_area_extent[0] > properties.max_framebuffer_width {
            return Err(Box::new(ValidationError {
                problem: "`render_area_offset[0] + render_area_extent[0]` is greater than the \
                    `max_framebuffer_width` limit"
                    .into(),
                vuids: &["VUID-VkRenderingInfo-pNext-07815"],
                ..Default::default()
            }));
        }

        if render_area_offset[1] + render_area_extent[1] > properties.max_framebuffer_height {
            return Err(Box::new(ValidationError {
                problem: "`render_area_offset[1] + render_area_extent[1]` is greater than the \
                    `max_framebuffer_height` limit"
                    .into(),
                vuids: &["VUID-VkRenderingInfo-pNext-07816"],
                ..Default::default()
            }));
        }

        // No VUID, but for sanity it makes sense to treat this the same as in framebuffers.
        if view_mask != 0 && layer_count != 1 {
            return Err(Box::new(ValidationError {
                problem: "`view_mask` is not 0, but `layer_count` is not 1".into(),
                // vuids?
                ..Default::default()
            }));
        }

        if view_mask != 0 && !device.enabled_features().multiview {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "is not 0".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "multiview",
                )])]),
                vuids: &["VUID-VkRenderingInfo-multiview-06127"],
            }));
        }

        let highest_view_index = u32::BITS - view_mask.leading_zeros();

        if highest_view_index > properties.max_multiview_view_count.unwrap_or(0) {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "the highest enabled view index is not less than the \
                    `max_multiview_view_count` limit"
                    .into(),
                vuids: &["VUID-VkRenderingInfo-viewMask-06128"],
                ..Default::default()
            }));
        }

        let mut samples = None;

        if color_attachments.len() > properties.max_color_attachments as usize {
            return Err(Box::new(ValidationError {
                context: "color_attachments".into(),
                problem: "the number of elements is greater than the `max_color_attachments` limit"
                    .into(),
                vuids: &["VUID-VkRenderingInfo-colorAttachmentCount-06106"],
                ..Default::default()
            }));
        }

        for (attachment_index, attachment_info) in
            color_attachments
                .iter()
                .enumerate()
                .filter_map(|(index, attachment_info)| {
                    attachment_info
                        .as_ref()
                        .map(|attachment_info| (index, attachment_info))
                })
        {
            attachment_info.validate(device).map_err(|err| {
                err.add_context(format!("color_attachments[{}]", attachment_index))
            })?;

            let &RenderingAttachmentInfo {
                image_view,
                image_layout,
                ref resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachments[{}].image_view.usage()", attachment_index)
                        .into(),
                    problem: "does not contain `ImageUsage::COLOR_ATTACHMENT".into(),
                    vuids: &["VUID-VkRenderingInfo-colorAttachmentCount-06087"],
                    ..Default::default()
                }));
            }

            if render_area_offset[0] + render_area_extent[0] > image_view.image().extent()[0] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`render_area_offset[0] + render_area_extent[0]` is greater than \
                        `color_attachments[{}].image_view.image().extent()[0]`",
                        attachment_index,
                    )
                    .into(),
                    vuids: &["VUID-VkRenderingInfo-pNext-06079"],
                    ..Default::default()
                }));
            }

            if render_area_offset[1] + render_area_extent[1] > image_view.image().extent()[1] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`render_area_offset[1] + render_area_extent[1]` is greater than \
                        `color_attachments[{}].image_view.image().extent()[1]`",
                        attachment_index,
                    )
                    .into(),
                    vuids: &["VUID-VkRenderingInfo-pNext-06080"],
                    ..Default::default()
                }));
            }

            match samples {
                Some(samples) => {
                    if samples != image_view.image().samples() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`color_attachments[{0}].image_view.image().samples()` \
                                is not equal to the number of samples of the other attachments",
                                attachment_index
                            )
                            .into(),
                            vuids: &["VUID-VkRenderingInfo-imageView-06070"],
                            ..Default::default()
                        }));
                    }
                }
                None => samples = Some(image_view.image().samples()),
            }

            if matches!(
                image_layout,
                ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachments[{0}].image_layout", attachment_index)
                        .into(),
                    problem: "is `ImageLayout::DepthStencilAttachmentOptimal`, \
                        `ImageLayout::DepthStencilReadOnlyOptimal`, \
                        `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                        `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                        `ImageLayout::DepthAttachmentOptimal`, \
                        `ImageLayout::DepthReadOnlyOptimal`, \
                        `ImageLayout::StencilAttachmentOptimal` or \
                        `ImageLayout::StencilReadOnlyOptimal`"
                        .into(),
                    vuids: &[
                        "VUID-VkRenderingInfo-colorAttachmentCount-06090",
                        "VUID-VkRenderingInfo-colorAttachmentCount-06096",
                        "VUID-VkRenderingInfo-colorAttachmentCount-06100",
                    ],
                    ..Default::default()
                }));
            }

            if let Some(resolve_info) = resolve_info {
                let &RenderingAttachmentResolveInfo {
                    mode: _,
                    image_view: _,
                    image_layout: resolve_image_layout,
                } = resolve_info;

                if matches!(
                    resolve_image_layout,
                    ImageLayout::DepthStencilAttachmentOptimal
                        | ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                        | ImageLayout::DepthAttachmentOptimal
                        | ImageLayout::DepthReadOnlyOptimal
                        | ImageLayout::StencilAttachmentOptimal
                        | ImageLayout::StencilReadOnlyOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "color_attachments[{0}].resolve_info.image_layout",
                            attachment_index
                        )
                        .into(),
                        problem: "is `ImageLayout::DepthStencilAttachmentOptimal`, \
                            `ImageLayout::DepthStencilReadOnlyOptimal`, \
                            `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                            `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                            `ImageLayout::DepthAttachmentOptimal`, \
                            `ImageLayout::DepthReadOnlyOptimal`, \
                            `ImageLayout::StencilAttachmentOptimal` or \
                            `ImageLayout::StencilReadOnlyOptimal`"
                            .into(),
                        vuids: &[
                            "VUID-VkRenderingInfo-colorAttachmentCount-06091",
                            "VUID-VkRenderingInfo-colorAttachmentCount-06097",
                            "VUID-VkRenderingInfo-colorAttachmentCount-06101",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if let Some(Some(attachment_info)) = depth_attachment {
            attachment_info
                .validate(device)
                .map_err(|err| err.add_context("depth_attachment"))?;

            let &RenderingAttachmentInfo {
                image_view,
                image_layout,
                ref resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            if !image_view
                .format()
                .aspects()
                .intersects(ImageAspects::DEPTH)
            {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment.image_view.format()".into(),
                    problem: "does not have a depth aspect".into(),
                    vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06547"],
                    ..Default::default()
                }));
            }

            if !image_view
                .usage()
                .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment.image_view.usage()".into(),
                    problem: "does not contain `ImageUsage::DEPTH_STENCIL_ATTACHMENT`".into(),
                    vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06088"],
                    ..Default::default()
                }));
            }

            if render_area_offset[0] + render_area_extent[0] > image_view.image().extent()[0] {
                return Err(Box::new(ValidationError {
                    problem: "`render_area_offset[0] + render_area_extent[0]` is greater than \
                        `depth_attachment.image_view.image().extent()[0]`"
                        .into(),
                    vuids: &["VUID-VkRenderingInfo-pNext-06079"],
                    ..Default::default()
                }));
            }

            if render_area_offset[1] + render_area_extent[1] > image_view.image().extent()[1] {
                return Err(Box::new(ValidationError {
                    problem: "`render_area_offset[1] + render_area_extent[1]` is greater than \
                        `depth_attachment.image_view.image().extent()[1]`"
                        .into(),
                    vuids: &["VUID-VkRenderingInfo-pNext-06080"],
                    ..Default::default()
                }));
            }

            match samples {
                Some(samples) => {
                    if samples != image_view.image().samples() {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment.image_view.image().samples()` \
                                is not equal to the number of samples of the other attachments"
                                .into(),
                            vuids: &["VUID-VkRenderingInfo-imageView-06070"],
                            ..Default::default()
                        }));
                    }
                }
                None => samples = Some(image_view.image().samples()),
            }

            if matches!(image_layout, ImageLayout::ColorAttachmentOptimal) {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment.image_layout".into(),
                    problem: "is `ImageLayout::ColorAttachmentOptimal`".into(),
                    vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06092"],
                    ..Default::default()
                }));
            }

            if let Some(resolve_info) = resolve_info {
                let &RenderingAttachmentResolveInfo {
                    mode,
                    image_view: _,
                    image_layout: resolve_image_layout,
                } = resolve_info;

                if !properties
                    .supported_depth_resolve_modes
                    .is_some_and(|modes| modes.contains_enum(mode))
                {
                    return Err(Box::new(ValidationError {
                        problem: "`depth_attachment.resolve_info.mode` is not one of the modes in \
                            the `supported_depth_resolve_modes` device property"
                            .into(),
                        vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06102"],
                        ..Default::default()
                    }));
                }

                if matches!(
                    resolve_image_layout,
                    ImageLayout::ColorAttachmentOptimal
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        context: "depth_attachment.resolve_info.image_layout".into(),
                        problem: "is `ImageLayout::ColorAttachmentOptimal` or \
                            `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`"
                            .into(),
                        vuids: &[
                            "VUID-VkRenderingInfo-pDepthAttachment-06093",
                            "VUID-VkRenderingInfo-pDepthAttachment-06098",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if let Some(Some(attachment_info)) = stencil_attachment {
            attachment_info
                .validate(device)
                .map_err(|err| err.add_context("stencil_attachment"))?;

            let &RenderingAttachmentInfo {
                image_view,
                image_layout,
                ref resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            if !image_view
                .format()
                .aspects()
                .intersects(ImageAspects::STENCIL)
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment.image_view.format()".into(),
                    problem: "does not have a stencil aspect".into(),
                    vuids: &["VUID-VkRenderingInfo-pStencilAttachment-06548"],
                    ..Default::default()
                }));
            }

            if !image_view
                .usage()
                .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment.image_view.usage()".into(),
                    problem: "does not contain `ImageUsage::DEPTH_STENCIL_ATTACHMENT`".into(),
                    vuids: &["VUID-VkRenderingInfo-pStencilAttachment-06089"],
                    ..Default::default()
                }));
            }

            if render_area_offset[0] + render_area_extent[0] > image_view.image().extent()[0] {
                return Err(Box::new(ValidationError {
                    problem: "`render_area_offset[0] + render_area_extent[0]` is greater than \
                        `stencil_attachment.image_view.image().extent()[0]`"
                        .into(),
                    vuids: &["VUID-VkRenderingInfo-pNext-06079"],
                    ..Default::default()
                }));
            }

            if render_area_offset[1] + render_area_extent[1] > image_view.image().extent()[1] {
                return Err(Box::new(ValidationError {
                    problem: "`render_area_offset[1] + render_area_extent[1]` is greater than \
                        `stencil_attachment.image_view.image().extent()[1]`"
                        .into(),
                    vuids: &["VUID-VkRenderingInfo-pNext-06080"],
                    ..Default::default()
                }));
            }

            if let Some(samples) = samples {
                if samples != image_view.image().samples() {
                    return Err(Box::new(ValidationError {
                        problem: "`stencil_attachment.image_view.image().samples()` \
                            is not equal to the number of samples of the other attachments"
                            .into(),
                        vuids: &["VUID-VkRenderingInfo-imageView-06070"],
                        ..Default::default()
                    }));
                }
            }

            if matches!(image_layout, ImageLayout::ColorAttachmentOptimal) {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment.image_layout".into(),
                    problem: "is `ImageLayout::ColorAttachmentOptimal`".into(),
                    vuids: &["VUID-VkRenderingInfo-pStencilAttachment-06094"],
                    ..Default::default()
                }));
            }

            if let Some(resolve_info) = resolve_info {
                let &RenderingAttachmentResolveInfo {
                    mode,
                    image_view: _,
                    image_layout: resolve_image_layout,
                } = resolve_info;

                if !properties
                    .supported_stencil_resolve_modes
                    .is_some_and(|modes| modes.contains_enum(mode))
                {
                    return Err(Box::new(ValidationError {
                        problem:
                            "`stencil_attachment.resolve_info.mode` is not one of the modes in \
                            the `supported_stencil_resolve_modes` device property"
                                .into(),
                        vuids: &["VUID-VkRenderingInfo-pStencilAttachment-06103"],
                        ..Default::default()
                    }));
                }

                if matches!(
                    resolve_image_layout,
                    ImageLayout::ColorAttachmentOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        context: "stencil_attachment.resolve_info.image_layout".into(),
                        problem: "is `ImageLayout::ColorAttachmentOptimal` or \
                            `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`"
                            .into(),
                        vuids: &[
                            "VUID-VkRenderingInfo-pStencilAttachment-06095",
                            "VUID-VkRenderingInfo-pStencilAttachment-06099",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if let (Some(Some(depth_attachment_info)), Some(Some(stencil_attachment_info))) =
            (depth_attachment, stencil_attachment)
        {
            if depth_attachment_info.image_view != stencil_attachment_info.image_view {
                return Err(Box::new(ValidationError {
                    problem: "`depth_attachment` and `stencil_attachment` are both `Some`, but \
                        `depth_attachment.image_view` does not equal \
                        `stencil_attachment.image_view`"
                        .into(),
                    vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06085"],
                    ..Default::default()
                }));
            }

            if depth_attachment_info.image_layout != stencil_attachment_info.image_layout
                && !device.enabled_features().separate_depth_stencil_layouts
            {
                return Err(Box::new(ValidationError {
                    problem: "`depth_attachment` and `stencil_attachment` are both `Some`, and \
                        `depth_attachment.image_layout` does not equal \
                        `stencil_attachment.attachment_ref.layout`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "separate_depth_stencil_layouts",
                    )])]),
                    ..Default::default()
                }));
            }

            match (
                &depth_attachment_info.resolve_info,
                &stencil_attachment_info.resolve_info,
            ) {
                (None, None) => (),
                (None, Some(_)) | (Some(_), None) => {
                    if !properties.independent_resolve_none.unwrap_or(false) {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment` and `stencil_attachment` are both \
                                `Some`, and the `independent_resolve_none` device property is \
                                `false`, but one of `depth_attachment.resolve_info` and \
                                `stencil_attachment.resolve_info` is `Some` while the other is \
                                `None`"
                                .into(),
                            vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06104"],
                            ..Default::default()
                        }));
                    }
                }
                (Some(depth_resolve_info), Some(stencil_resolve_info)) => {
                    if depth_resolve_info.image_layout != stencil_resolve_info.image_layout
                        && !device.enabled_features().separate_depth_stencil_layouts
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment` and `stencil_attachment` are both \
                                `Some`, and `depth_attachment.resolve_info` and \
                                `stencil_attachment.resolve_info` are also both `Some`, and \
                                `depth_attachment.resolve_info.image_layout` does not equal \
                                `stencil_attachment.resolve_info.image_layout`"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("separate_depth_stencil_layouts"),
                            ])]),
                            ..Default::default()
                        }));
                    }

                    if !properties.independent_resolve.unwrap_or(false)
                        && depth_resolve_info.mode != stencil_resolve_info.mode
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment` and `stencil_attachment` are both \
                                `Some`, and `depth_attachment.resolve_info` and \
                                `stencil_attachment.resolve_info` are also both `Some`, and \
                                the `independent_resolve` device property is `false`, but \
                                `depth_attachment.resolve_info.mode` does not equal \
                                `stencil_attachment.resolve_info.mode`"
                                .into(),
                            vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06105"],
                            ..Default::default()
                        }));
                    }

                    if depth_resolve_info.image_view != stencil_resolve_info.image_view {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment` and `stencil_attachment` are both \
                                `Some`, and `depth_attachment.resolve_info` and \
                                `stencil_attachment.resolve_info` are also both `Some`, but \
                                `depth_attachment.resolve_info.image_view` does not equal \
                                `stencil_attachment.resolve_info.image_view`"
                                .into(),
                            vuids: &["VUID-VkRenderingInfo-pDepthAttachment-06086"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a RenderingInfoFields1Vk,
    ) -> vk::RenderingInfo<'a> {
        let &Self {
            render_area_offset,
            render_area_extent,
            layer_count,
            view_mask,
            color_attachments: _,
            depth_attachment: _,
            stencil_attachment: _,
            contents,
            _ne: _,
        } = self;
        let RenderingInfoFields1Vk {
            color_attachments_vk,
            depth_attachment_vk,
            stencil_attachment_vk,
        } = fields1_vk;

        vk::RenderingInfo::default()
            .flags(contents.into())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D {
                    x: render_area_offset[0] as i32,
                    y: render_area_offset[1] as i32,
                },
                extent: vk::Extent2D {
                    width: render_area_extent[0],
                    height: render_area_extent[1],
                },
            })
            .layer_count(layer_count)
            .view_mask(view_mask)
            .color_attachments(color_attachments_vk)
            .depth_attachment(depth_attachment_vk)
            .stencil_attachment(stencil_attachment_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> RenderingInfoFields1Vk {
        let &Self {
            color_attachments,
            depth_attachment,
            stencil_attachment,
            ..
        } = self;

        let color_attachments_vk = color_attachments
            .iter()
            .map(|attachment_info| {
                attachment_info.as_ref().map_or(
                    vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null()),
                    RenderingAttachmentInfo::to_vk,
                )
            })
            .collect();

        let depth_attachment_vk = depth_attachment.and_then(Option::as_ref).map_or(
            vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null()),
            RenderingAttachmentInfo::to_vk,
        );

        let stencil_attachment_vk = stencil_attachment.and_then(Option::as_ref).map_or(
            vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null()),
            RenderingAttachmentInfo::to_vk,
        );

        RenderingInfoFields1Vk {
            color_attachments_vk,
            depth_attachment_vk,
            stencil_attachment_vk,
        }
    }
}

pub(crate) struct RenderingInfoFields1Vk {
    pub(crate) color_attachments_vk: SmallVec<[vk::RenderingAttachmentInfo<'static>; 2]>,
    pub(crate) depth_attachment_vk: vk::RenderingAttachmentInfo<'static>,
    pub(crate) stencil_attachment_vk: vk::RenderingAttachmentInfo<'static>,
}

/// Parameters to specify properties of an attachment.
#[derive(Clone, Debug)]
pub struct RenderingAttachmentInfo<'a> {
    /// The image view to use as the attachment.
    ///
    /// There is no default value.
    pub image_view: &'a ImageView,

    /// The image layout that `image_view` should be in during rendering.
    ///
    /// The default value is [`ImageLayout::ColorAttachmentOptimal`] if `image_view` has a color
    /// format, [`ImageLayout::DepthStencilAttachmentOptimal`] if `image_view` has a depth/stencil
    /// format.
    pub image_layout: ImageLayout,

    /// The resolve operation that should be performed at the end of rendering.
    ///
    /// The default value is `None`.
    pub resolve_info: Option<RenderingAttachmentResolveInfo<'a>>,

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

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> RenderingAttachmentInfo<'a> {
    /// Returns a default `RenderingAttachmentInfo` with the provided `image_view`.
    // TODO: make const
    #[inline]
    pub fn new(image_view: &'a ImageView) -> Self {
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            image_view,
            image_layout,
            ref resolve_info,
            load_op,
            store_op,
            clear_value,
            _ne,
        } = self;

        image_layout.validate_device(device).map_err(|err| {
            err.add_context("image_layout")
                .set_vuids(&["VUID-VkRenderingAttachmentInfo-imageLayout-parameter"])
        })?;

        load_op.validate_device(device).map_err(|err| {
            err.add_context("load_op")
                .set_vuids(&["VUID-VkRenderingAttachmentInfo-loadOp-parameter"])
        })?;

        store_op.validate_device(device).map_err(|err| {
            err.add_context("store_op")
                .set_vuids(&["VUID-VkRenderingAttachmentInfo-storeOp-parameter"])
        })?;

        if matches!(
            image_layout,
            ImageLayout::Undefined
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::TransferSrcOptimal
                | ImageLayout::TransferDstOptimal
                | ImageLayout::Preinitialized
                | ImageLayout::PresentSrc
        ) {
            return Err(Box::new(ValidationError {
                context: "image_layout".into(),
                problem: "is `ImageLayout::Undefined`, \
                    `ImageLayout::ShaderReadOnlyOptimal`, \
                    `ImageLayout::TransferSrcOptimal`, \
                    `ImageLayout::TransferDstOptimal`, \
                    `ImageLayout::Preinitialized` or \
                    `ImageLayout::PresentSrc`"
                    .into(),
                vuids: &[
                    "VUID-VkRenderingAttachmentInfo-imageView-06135",
                    "VUID-VkRenderingAttachmentInfo-imageView-06145",
                ],
                ..Default::default()
            }));
        }

        if let Some(resolve_info) = resolve_info {
            resolve_info
                .validate(device)
                .map_err(|err| err.add_context("resolve_info"))?;

            let &RenderingAttachmentResolveInfo {
                mode: _,
                image_view: resolve_image_view,
                image_layout: _,
            } = resolve_info;

            if image_view.image().samples() == SampleCount::Sample1 {
                return Err(Box::new(ValidationError {
                    problem: "`resolve_info` is `Some`, but \
                        `image_view.image().samples()` is `SampleCount::Sample1`"
                        .into(),
                    vuids: &["VUID-VkRenderingAttachmentInfo-imageView-06132"],
                    ..Default::default()
                }));
            }

            if image_view.format() != resolve_image_view.format() {
                return Err(Box::new(ValidationError {
                    problem: "`resolve_info.image_view.format()` does not equal \
                        `image_view.format()`"
                        .into(),
                    vuids: &["VUID-VkRenderingAttachmentInfo-imageView-06134"],
                    ..Default::default()
                }));
            }
        }

        match (clear_value, load_op == AttachmentLoadOp::Clear) {
            (None, false) => (),
            (None, true) => {
                return Err(Box::new(ValidationError {
                    problem: "`load_op` is `AttachmentLoadOp::Clear`, but \
                        `clear_value` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }
            (Some(_), false) => {
                return Err(Box::new(ValidationError {
                    problem: "`load_op` is not `AttachmentLoadOp::Clear`, but \
                        `clear_value` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (Some(clear_value), true) => {
                clear_value
                    .validate(device)
                    .map_err(|err| err.add_context("clear_value"))?;
            }
        };

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::RenderingAttachmentInfo<'static> {
        let &Self {
            image_view,
            image_layout,
            ref resolve_info,
            load_op,
            store_op,
            clear_value,
            _ne: _,
        } = self;

        let (resolve_mode, resolve_image_view, resolve_image_layout) =
            resolve_info.as_ref().map_or(
                (
                    vk::ResolveModeFlags::NONE,
                    Default::default(),
                    Default::default(),
                ),
                RenderingAttachmentResolveInfo::to_vk,
            );

        vk::RenderingAttachmentInfo::default()
            .image_view(image_view.handle())
            .image_layout(image_layout.into())
            .resolve_mode(resolve_mode)
            .resolve_image_view(resolve_image_view)
            .resolve_image_layout(resolve_image_layout)
            .load_op(load_op.into())
            .store_op(store_op.into())
            .clear_value(
                clear_value
                    .as_ref()
                    .map_or_else(Default::default, ClearValue::to_vk),
            )
    }
}

/// Parameters to specify the resolve behavior of an attachment.
#[derive(Clone, Debug)]
pub struct RenderingAttachmentResolveInfo<'a> {
    /// How the resolve operation should be performed.
    ///
    /// The default value is [`ResolveMode::Average`].
    pub mode: ResolveMode,

    /// The image view that the result of the resolve operation should be written to.
    ///
    /// There is no default value.
    pub image_view: &'a ImageView,

    /// The image layout that `image_view` should be in during the resolve operation.
    ///
    /// The default value is [`ImageLayout::ColorAttachmentOptimal`] if `image_view` has a color
    /// format, [`ImageLayout::DepthStencilAttachmentOptimal`] if `image_view` has a depth/stencil
    /// format.
    pub image_layout: ImageLayout,
}

impl<'a> RenderingAttachmentResolveInfo<'a> {
    /// Returns a default `RenderingAttachmentResolveInfo` with the provided `image_view`.
    // TODO: make const
    #[inline]
    pub fn new(image_view: &'a ImageView) -> Self {
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            mode,
            image_view,
            image_layout,
        } = self;

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode")
                .set_vuids(&["VUID-VkRenderingAttachmentInfo-resolveMode-parameter"])
        })?;

        image_layout.validate_device(device).map_err(|err| {
            err.add_context("image_layout")
                .set_vuids(&["VUID-VkRenderingAttachmentInfo-resolveImageLayout-parameter"])
        })?;

        if let Some(numeric_format) = image_view.format().numeric_format_color() {
            match numeric_format.numeric_type() {
                NumericType::Float => {
                    if mode != ResolveMode::Average {
                        return Err(Box::new(ValidationError {
                            problem: "`image_view.format()` is a floating-point color format, but \
                                `mode` is not `ResolveMode::Average`"
                                .into(),
                            vuids: &["VUID-VkRenderingAttachmentInfo-imageView-06129"],
                            ..Default::default()
                        }));
                    }
                }
                NumericType::Int | NumericType::Uint => {
                    if mode != ResolveMode::SampleZero {
                        return Err(Box::new(ValidationError {
                            problem: "`image_view.format()` is an integer color format, but \
                                `mode` is not `ResolveMode::SampleZero`"
                                .into(),
                            vuids: &["VUID-VkRenderingAttachmentInfo-imageView-06130"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if image_view.image().samples() != SampleCount::Sample1 {
            return Err(Box::new(ValidationError {
                context: "image_view.image().samples()".into(),
                problem: "is not `SampleCount::Sample1`".into(),
                vuids: &["VUID-VkRenderingAttachmentInfo-imageView-06133"],
                ..Default::default()
            }));
        }

        if matches!(
            image_layout,
            ImageLayout::Undefined
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::TransferSrcOptimal
                | ImageLayout::TransferDstOptimal
                | ImageLayout::Preinitialized
                | ImageLayout::PresentSrc
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::DepthReadOnlyOptimal
                | ImageLayout::StencilReadOnlyOptimal
        ) {
            return Err(Box::new(ValidationError {
                context: "image_layout".into(),
                problem: "is `ImageLayout::Undefined`, \
                    `ImageLayout::ShaderReadOnlyOptimal`, \
                    `ImageLayout::TransferSrcOptimal`, \
                    `ImageLayout::TransferDstOptimal`, \
                    `ImageLayout::Preinitialized`, \
                    `ImageLayout::PresentSrc`, \
                    `ImageLayout::DepthStencilReadOnlyOptimal`, \
                    `ImageLayout::DepthReadOnlyOptimal` or \
                    `ImageLayout::StencilReadOnlyOptimal`"
                    .into(),
                vuids: &[
                    "VUID-VkRenderingAttachmentInfo-imageView-06136",
                    "VUID-VkRenderingAttachmentInfo-imageView-06137",
                    "VUID-VkRenderingAttachmentInfo-imageView-06146",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> (vk::ResolveModeFlags, vk::ImageView, vk::ImageLayout) {
        let &Self {
            mode,
            image_view,
            image_layout,
        } = self;

        (mode.into(), image_view.handle(), image_layout.into())
    }
}
