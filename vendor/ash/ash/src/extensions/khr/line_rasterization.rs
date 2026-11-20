//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_line_rasterization.html>

use crate::vk;

impl crate::khr::line_rasterization::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetLineStippleKHR.html>
    #[inline]
    pub unsafe fn cmd_set_line_stipple(
        &self,
        command_buffer: vk::CommandBuffer,
        line_stipple_factor: u32,
        line_stipple_pattern: u16,
    ) {
        (self.fp.cmd_set_line_stipple_khr)(
            command_buffer,
            line_stipple_factor,
            line_stipple_pattern,
        )
    }
}
