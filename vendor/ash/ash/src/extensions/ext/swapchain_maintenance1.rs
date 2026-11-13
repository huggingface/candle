//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_swapchain_maintenance1.html>

use crate::vk;
use crate::VkResult;

impl crate::ext::swapchain_maintenance1::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkReleaseSwapchainImagesEXT.html>
    #[inline]
    pub unsafe fn release_swapchain_images(
        &self,
        release_info: &vk::ReleaseSwapchainImagesInfoEXT<'_>,
    ) -> VkResult<()> {
        (self.fp.release_swapchain_images_ext)(self.handle, release_info).result()
    }
}
