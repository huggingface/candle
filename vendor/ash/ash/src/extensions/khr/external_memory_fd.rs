//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_external_memory_fd.html>

use crate::vk;
use crate::VkResult;
use core::mem;

impl crate::khr::external_memory_fd::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryFdKHR.html>
    #[inline]
    pub unsafe fn get_memory_fd(&self, get_fd_info: &vk::MemoryGetFdInfoKHR<'_>) -> VkResult<i32> {
        let mut fd = mem::MaybeUninit::uninit();
        (self.fp.get_memory_fd_khr)(self.handle, get_fd_info, fd.as_mut_ptr())
            .assume_init_on_success(fd)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryFdPropertiesKHR.html>
    #[inline]
    pub unsafe fn get_memory_fd_properties(
        &self,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        fd: i32,
        memory_fd_properties: &mut vk::MemoryFdPropertiesKHR<'_>,
    ) -> VkResult<()> {
        (self.fp.get_memory_fd_properties_khr)(self.handle, handle_type, fd, memory_fd_properties)
            .result()
    }
}
