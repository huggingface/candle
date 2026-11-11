//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_depth_clamp_control.html>

use crate::vk;

impl crate::ext::depth_clamp_control::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetDepthClampRangeEXT.html>
    #[inline]
    #[doc(alias = "vkCmdSetDepthClampRangeEXT")]
    pub unsafe fn cmd_set_depth_clamp_range(
        &self,
        command_buffer: vk::CommandBuffer,
        depth_clamp_mode: vk::DepthClampModeEXT,
        depth_clamp_range: &vk::DepthClampRangeEXT,
    ) {
        (self.fp.cmd_set_depth_clamp_range_ext)(command_buffer, depth_clamp_mode, depth_clamp_range)
    }
}
