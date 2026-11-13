//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_present_wait.html>

use crate::vk;
use crate::VkResult;

impl crate::khr::present_wait::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForPresentKHR.html>
    #[inline]
    pub unsafe fn wait_for_present(
        &self,
        swapchain: vk::SwapchainKHR,
        present_id: u64,
        timeout: u64,
    ) -> VkResult<()> {
        (self.fp.wait_for_present_khr)(self.handle, swapchain, present_id, timeout).result()
    }
}
