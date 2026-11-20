//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_dynamic_rendering.html>

use crate::vk;

impl crate::khr::dynamic_rendering::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginRenderingKHR.html>
    #[inline]
    pub unsafe fn cmd_begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        rendering_info: &vk::RenderingInfoKHR<'_>,
    ) {
        (self.fp.cmd_begin_rendering_khr)(command_buffer, rendering_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEndRenderingKHR.html>
    #[inline]
    pub unsafe fn cmd_end_rendering(&self, command_buffer: vk::CommandBuffer) {
        (self.fp.cmd_end_rendering_khr)(command_buffer)
    }
}
