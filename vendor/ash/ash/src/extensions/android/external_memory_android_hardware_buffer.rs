//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_ANDROID_external_memory_android_hardware_buffer.html>

use crate::vk;
use crate::VkResult;
use core::mem;

impl crate::android::external_memory_android_hardware_buffer::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetAndroidHardwareBufferPropertiesANDROID.html>
    #[inline]
    pub unsafe fn get_android_hardware_buffer_properties(
        &self,
        buffer: *const vk::AHardwareBuffer,
        properties: &mut vk::AndroidHardwareBufferPropertiesANDROID<'_>,
    ) -> VkResult<()> {
        (self.fp.get_android_hardware_buffer_properties_android)(self.handle, buffer, properties)
            .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryAndroidHardwareBufferANDROID.html>
    #[inline]
    pub unsafe fn get_memory_android_hardware_buffer(
        &self,
        info: &vk::MemoryGetAndroidHardwareBufferInfoANDROID<'_>,
    ) -> VkResult<*mut vk::AHardwareBuffer> {
        let mut buffer = mem::MaybeUninit::uninit();
        (self.fp.get_memory_android_hardware_buffer_android)(self.handle, info, buffer.as_mut_ptr())
            .assume_init_on_success(buffer)
    }
}
