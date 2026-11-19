//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_present_wait2.html>

use crate::vk;
use crate::VkResult;

impl crate::khr::present_wait2::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForPresent2KHR.html>
    #[inline]
    pub unsafe fn wait_for_present2(
        &self,
        swapchain: vk::SwapchainKHR,
        present_wait2_info: &vk::PresentWait2InfoKHR<'_>,
    ) -> VkResult<()> {
        (self.fp.wait_for_present2_khr)(self.handle, swapchain, present_wait2_info).result()
    }
}
