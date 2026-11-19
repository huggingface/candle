//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_create_renderpass2.html>
//!
//! Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-dynamicrendering>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::khr::create_renderpass2::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateRenderPass2.html>
    #[inline]
    pub unsafe fn create_render_pass2(
        &self,
        create_info: &vk::RenderPassCreateInfo2<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::RenderPass> {
        let mut renderpass = mem::MaybeUninit::uninit();
        (self.fp.create_render_pass2_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            renderpass.as_mut_ptr(),
        )
        .assume_init_on_success(renderpass)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginRenderPass2.html>
    #[inline]
    pub unsafe fn cmd_begin_render_pass2(
        &self,
        command_buffer: vk::CommandBuffer,
        render_pass_begin_info: &vk::RenderPassBeginInfo<'_>,
        subpass_begin_info: &vk::SubpassBeginInfo<'_>,
    ) {
        (self.fp.cmd_begin_render_pass2_khr)(
            command_buffer,
            render_pass_begin_info,
            subpass_begin_info,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdNextSubpass2.html>
    #[inline]
    pub unsafe fn cmd_next_subpass2(
        &self,
        command_buffer: vk::CommandBuffer,
        subpass_begin_info: &vk::SubpassBeginInfo<'_>,
        subpass_end_info: &vk::SubpassEndInfo<'_>,
    ) {
        (self.fp.cmd_next_subpass2_khr)(command_buffer, subpass_begin_info, subpass_end_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEndRenderPass2.html>
    #[inline]
    pub unsafe fn cmd_end_render_pass2(
        &self,
        command_buffer: vk::CommandBuffer,
        subpass_end_info: &vk::SubpassEndInfo<'_>,
    ) {
        (self.fp.cmd_end_render_pass2_khr)(command_buffer, subpass_end_info)
    }
}
