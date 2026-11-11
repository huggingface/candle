//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_dynamic_rendering_local_read.html>

use crate::vk;

impl crate::khr::dynamic_rendering_local_read::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetRenderingAttachmentLocationsKHR.html>
    #[inline]
    #[doc(alias = "vkCmdSetRenderingAttachmentLocations")]
    pub unsafe fn cmd_set_rendering_attachment_locations(
        &self,
        command_buffer: vk::CommandBuffer,
        location_info: &vk::RenderingAttachmentLocationInfoKHR<'_>,
    ) {
        (self.fp.cmd_set_rendering_attachment_locations_khr)(command_buffer, location_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetRenderingInputAttachmentIndicesKHR.html>
    #[inline]
    #[doc(alias = "vkCmdSetRenderingInputAttachmentIndices")]
    pub unsafe fn cmd_set_rendering_input_attachment_indices(
        &self,
        command_buffer: vk::CommandBuffer,
        input_attachment_index_info: &vk::RenderingInputAttachmentIndexInfoKHR<'_>,
    ) {
        (self.fp.cmd_set_rendering_input_attachment_indices_khr)(
            command_buffer,
            input_attachment_index_info,
        )
    }
}
