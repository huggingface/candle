//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_synchronization2.html>

use crate::vk;
use crate::VkResult;

impl crate::khr::synchronization2::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier2KHR.html>
    #[inline]
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfoKHR<'_>,
    ) {
        (self.fp.cmd_pipeline_barrier2_khr)(command_buffer, dependency_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdResetEvent2KHR.html>
    #[inline]
    pub unsafe fn cmd_reset_event2(
        &self,
        command_buffer: vk::CommandBuffer,
        event: vk::Event,
        stage_mask: vk::PipelineStageFlags2KHR,
    ) {
        (self.fp.cmd_reset_event2_khr)(command_buffer, event, stage_mask)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetEvent2KHR.html>
    #[inline]
    pub unsafe fn cmd_set_event2(
        &self,
        command_buffer: vk::CommandBuffer,
        event: vk::Event,
        dependency_info: &vk::DependencyInfoKHR<'_>,
    ) {
        (self.fp.cmd_set_event2_khr)(command_buffer, event, dependency_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWaitEvents2KHR.html>
    #[inline]
    pub unsafe fn cmd_wait_events2(
        &self,
        command_buffer: vk::CommandBuffer,
        events: &[vk::Event],
        dependency_infos: &[vk::DependencyInfoKHR<'_>],
    ) {
        assert_eq!(events.len(), dependency_infos.len());
        (self.fp.cmd_wait_events2_khr)(
            command_buffer,
            events.len() as u32,
            events.as_ptr(),
            dependency_infos.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp2KHR.html>
    #[inline]
    pub unsafe fn cmd_write_timestamp2(
        &self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags2KHR,
        query_pool: vk::QueryPool,
        query: u32,
    ) {
        (self.fp.cmd_write_timestamp2_khr)(command_buffer, stage, query_pool, query)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit2KHR.html>
    #[inline]
    pub unsafe fn queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2KHR<'_>],
        fence: vk::Fence,
    ) -> VkResult<()> {
        (self.fp.queue_submit2_khr)(queue, submits.len() as u32, submits.as_ptr(), fence).result()
    }
}
