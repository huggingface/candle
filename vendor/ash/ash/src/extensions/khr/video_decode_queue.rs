//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_video_decode_queue.html>

use crate::vk;

impl crate::khr::video_decode_queue::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDecodeVideoKHR.html>
    #[inline]
    #[doc(alias = "vkCmdDecodeVideoKHR")]
    pub unsafe fn cmd_decode_video(
        &self,
        command_buffer: vk::CommandBuffer,
        decode_info: &vk::VideoDecodeInfoKHR<'_>,
    ) {
        (self.fp.cmd_decode_video_khr)(command_buffer, decode_info)
    }
}
