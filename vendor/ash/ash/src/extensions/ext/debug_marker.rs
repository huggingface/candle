//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_debug_marker.html>

use crate::vk;
use crate::VkResult;

impl crate::ext::debug_marker::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDebugMarkerSetObjectNameEXT.html>
    #[inline]
    pub unsafe fn debug_marker_set_object_name(
        &self,
        name_info: &vk::DebugMarkerObjectNameInfoEXT<'_>,
    ) -> VkResult<()> {
        (self.fp.debug_marker_set_object_name_ext)(self.handle, name_info).result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDebugMarkerBeginEXT.html>
    #[inline]
    pub unsafe fn cmd_debug_marker_begin(
        &self,
        command_buffer: vk::CommandBuffer,
        marker_info: &vk::DebugMarkerMarkerInfoEXT<'_>,
    ) {
        (self.fp.cmd_debug_marker_begin_ext)(command_buffer, marker_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDebugMarkerEndEXT.html>
    #[inline]
    pub unsafe fn cmd_debug_marker_end(&self, command_buffer: vk::CommandBuffer) {
        (self.fp.cmd_debug_marker_end_ext)(command_buffer)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDebugMarkerInsertEXT.html>
    #[inline]
    pub unsafe fn cmd_debug_marker_insert(
        &self,
        command_buffer: vk::CommandBuffer,
        marker_info: &vk::DebugMarkerMarkerInfoEXT<'_>,
    ) {
        (self.fp.cmd_debug_marker_insert_ext)(command_buffer, marker_info)
    }
}
