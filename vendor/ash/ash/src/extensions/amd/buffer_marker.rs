//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMD_buffer_marker.html>

use crate::vk;

impl crate::amd::buffer_marker::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteBufferMarkerAMD.html>
    #[inline]
    pub unsafe fn cmd_write_buffer_marker(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_stage: vk::PipelineStageFlags,
        dst_buffer: vk::Buffer,
        dst_offset: vk::DeviceSize,
        marker: u32,
    ) {
        (self.fp.cmd_write_buffer_marker_amd)(
            command_buffer,
            pipeline_stage,
            dst_buffer,
            dst_offset,
            marker,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteBufferMarker2AMD.html>
    ///
    // Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>
    #[inline]
    pub unsafe fn cmd_write_buffer_marker2(
        &self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags2,
        dst_buffer: vk::Buffer,
        dst_offset: vk::DeviceSize,
        marker: u32,
    ) {
        #[allow(deprecated)]
        (self.fp.cmd_write_buffer_marker2_amd)(
            command_buffer,
            stage,
            dst_buffer,
            dst_offset,
            marker,
        )
    }
}
