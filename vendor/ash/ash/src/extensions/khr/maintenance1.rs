//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_maintenance1.html>

use crate::vk;

impl crate::khr::maintenance1::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkTrimCommandPoolKHR.html>
    #[inline]
    pub unsafe fn trim_command_pool(
        &self,
        command_pool: vk::CommandPool,
        flags: vk::CommandPoolTrimFlagsKHR,
    ) {
        (self.fp.trim_command_pool_khr)(self.handle, command_pool, flags)
    }
}
