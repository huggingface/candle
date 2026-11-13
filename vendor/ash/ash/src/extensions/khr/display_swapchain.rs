//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_display_swapchain.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use alloc::vec::Vec;

impl crate::khr::display_swapchain::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateSharedSwapchainsKHR.html>
    #[inline]
    pub unsafe fn create_shared_swapchains(
        &self,
        create_infos: &[vk::SwapchainCreateInfoKHR<'_>],
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<Vec<vk::SwapchainKHR>> {
        let mut swapchains = Vec::with_capacity(create_infos.len());
        (self.fp.create_shared_swapchains_khr)(
            self.handle,
            create_infos.len() as u32,
            create_infos.as_ptr(),
            allocation_callbacks.to_raw_ptr(),
            swapchains.as_mut_ptr(),
        )
        .set_vec_len_on_success(swapchains, create_infos.len())
    }
}
